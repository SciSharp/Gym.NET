using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.MuJoCo;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.MuJoCo;
using NumSharp;

namespace Gym.Showreel;

/// <summary>
///     A scripted policy: picks the next action from the env and its last observation. Each episode gets a fresh instance,
///     because some policies keep state (BipedalWalker's gait phase, the toy-text plans).
/// </summary>
internal interface IPolicy {
    /// <summary>The next action, in the form <see cref="IEnv.Step(object?)"/> accepts.</summary>
    /// <param name="env">The env as <c>make</c> built it (wrappers included); policies read the unwrapped env through it.</param>
    /// <param name="observation">The last observation, as <see cref="IEnv"/> returns it.</param>
    /// <returns>The action: a 0-d int64 <see cref="NDArray"/> for discrete spaces, a float32 array for boxes.</returns>
    object Act(IEnv env, object? observation);
}

/// <summary>A policy from a delegate, for the stateless controllers.</summary>
/// <param name="Rule">The rule: env and observation in, action out.</param>
internal sealed record DelegatePolicy(Func<IEnv, object?, object> Rule) : IPolicy {
    /// <inheritdoc />
    public object Act(IEnv env, object? observation) => Rule(env, observation);
}

/// <summary>
///     One episode the showreel plays: a registered id, a seed, the policy that plays it, how fast it plays, and the stamp
///     its ending earns.
/// </summary>
/// <param name="Id">The registered id, e.g. <c>LunarLander-v3</c>.</param>
/// <param name="Family">Gymnasium's env family: <c>classic_control</c>, <c>box2d</c>, <c>toy_text</c> or <c>mujoco</c>.</param>
/// <param name="Seed">The reset seed; the policies were checked to play well from it.</param>
/// <param name="StepsPerSecond">Playback speed in env steps per video second (the env's own render fps by default).</param>
/// <param name="Policy">Makes a fresh policy for each playback.</param>
/// <param name="Stamp">The text stamped on the panel when the episode ends, from (terminated, truncated, return); null for none.</param>
/// <param name="Blurb">One line for the docs site's gallery: what the agent is doing.</param>
/// <param name="Hands">Episodes played back to back (Blackjack deals several hands, Reacher reaches several targets); 1 for the others.</param>
/// <param name="HandLabel">What one of those episodes is called in the status line (<c>HAND</c>, <c>TARGET</c>).</param>
/// <param name="ScoresHands">Whether the status counts won and lost episodes (Blackjack) rather than showing the return.</param>
/// <param name="Kwargs">
///     Constructor arguments added to <c>render_mode="rgb_array"</c> when the env is made, by Python name, or null. Only
///     presentation is set this way (InvertedDoublePendulum's camera); never anything that changes the dynamics.
/// </param>
/// <param name="RenderEveryStep">
///     Whether the env's <c>render()</c> changes what its next frame shows, so the player must render after every step, as
///     human mode does, and not only the frames it displays: LunarLander ages and removes its exhaust particles there,
///     BipedalWalker sweeps its lidar one ray a frame. Rendering only the shown frames left LunarLander's particles
///     unaged, a cloud of them after a mid-episode join. Costs a render per step, cheap for these 2-D envs; the MuJoCo
///     envs, whose frames are costly, have no such state.
/// </param>
internal sealed record EpisodeRecipe(
    string Id,
    string Family,
    int Seed,
    double StepsPerSecond,
    Func<IPolicy> Policy,
    Func<bool, bool, double, string?> Stamp,
    string Blurb,
    int Hands = 1,
    string HandLabel = "HAND",
    bool ScoresHands = false,
    IReadOnlyDictionary<string, object?>? Kwargs = null,
    bool RenderEveryStep = false) {
    /// <summary>Ticks a finished episode is held on screen before the next one starts (multi-episode recipes only).</summary>
    public int HandPauseTicks => 2;

    /// <summary>The keyword arguments the showreel makes the env with: <c>render_mode="rgb_array"</c> plus <see cref="Kwargs"/>.</summary>
    /// <returns>A new dictionary for <c>Gymnasium.Make</c>.</returns>
    public Dictionary<string, object?> MakeArguments() {
        var arguments = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
        if (Kwargs is not null)
            foreach (var (name, value) in Kwargs)
                arguments[name] = value;
        return arguments;
    }
}

/// <summary>
///     The 24 episodes of the showreel, one per visually distinct registered id (the maintainer's choice: variants that
///     only change the action space or the version, such as LunarLanderContinuous-v3 or Ant-v4, look the same and are
///     left out), and the policies that play them.
/// </summary>
/// <remarks>
///     <para>
///     Every policy and seed was checked to play a good episode: CartPole balances for all 500 steps while swaying,
///     MountainCar climbs in 122 steps, Pendulum swings up and holds, Acrobot reaches the line in 65 steps, LunarLander
///     lands for 248.6, BipedalWalker finishes the course for 314.5, the hardcore walker trips over a stump (on purpose,
///     for the GAME OVER stamp), CarRacing scores 880.7, and the toy-text agents follow value iteration over the env's own
///     transition table. LunarLander and BipedalWalker use Gymnasium's own heuristics as Gym.NET ships them.
///     </para>
///     <para>
///     Eight of the eleven MuJoCo envs play linear policies committed in <c>tools/showreel/policies</c>: HalfCheetah's is
///     the autopilot trained for <c>examples/HalfCheetah</c> on the same physics, the others were
///     trained by <see cref="Ars"/> (the <c>train</c> command), and each file records its training and evaluation returns.
///     Three are scripted instead, each after its trained policy looked wrong on screen: InvertedPendulum
///     (<see cref="InvertedPendulumBalance"/>, which keeps the cart centred), Reacher (<see cref="ReacherReach"/>, whose
///     inverse kinematics beat the linear policy's plateau) and Pusher (<see cref="PusherScript"/>, which moves where the
///     linear policy barely did).
///     </para>
/// </remarks>
internal static class Episodes {
    /// <summary>
    ///     <c>camera_name="track"</c>: the locomotion models' own camera that follows the body's centre of mass.
    /// </summary>
    /// <remarks>
    ///     Declared before <see cref="All"/> on purpose: static initializers run in textual order, and the recipes below
    ///     read this field while <see cref="All"/> is built.
    /// </remarks>
    private static readonly IReadOnlyDictionary<string, object?> TrackCamera = new Dictionary<string, object?> { ["camera_name"] = "track" };

    /// <summary>
    ///     The recipes in the video's order, which the gallery follows: classic control, MuJoCo's control tasks, Box2D,
    ///     MuJoCo's locomotion, toy text.
    /// </summary>
    public static IReadOnlyList<EpisodeRecipe> All { get; } = new[] {
        new EpisodeRecipe("CartPole-v1", "classic_control", 3, 50, () => new DelegatePolicy(CartPoleSway),
            (term, trunc, ret) => trunc ? "500 STEPS!" : "FELL", "A linear controller balances the pole while gliding the cart from side to side."),
        new EpisodeRecipe("MountainCar-v0", "classic_control", 4, 30, () => new DelegatePolicy(MountainCarPump),
            (term, _, _) => term ? "SUMMIT!" : null, "Too weak to drive straight up: it pushes with its velocity to swing higher each time."),
        // Pendulum only truncates (200 steps); a return above -300 means it spent the episode up, a failed swing scores
        // around -1,000.
        new EpisodeRecipe("Pendulum-v1", "classic_control", 5, 30, () => new DelegatePolicy(PendulumSwingUp),
            (_, _, ret) => ret > -300 ? "UPRIGHT!" : null, "Energy pumping swings it up, then a PD controller catches it at the top."),
        new EpisodeRecipe("Acrobot-v1", "classic_control", 1, 15, () => new DelegatePolicy(AcrobotPump),
            (term, _, _) => term ? "REACHED!" : null, "Torque on the elbow, timed with the swing, lifts the tip over the line."),
        new EpisodeRecipe("InvertedPendulum-v5", "mujoco", 0, 25, () => new DelegatePolicy(InvertedPendulumBalance),
            (term, trunc, _) => trunc ? "BALANCED!" : term ? "FELL" : null, "MuJoCo's cart-pole: linear state feedback holds the pole up and the cart centred."),
        // Gymnasium sets no camera for this env, and MuJoCo's free camera shows it a few pixels tall in a quarter panel.
        new EpisodeRecipe("InvertedDoublePendulum-v5", "mujoco", 0, 20, Trained("InvertedDoublePendulum-v5"),
            (term, trunc, _) => trunc ? "BALANCED!" : term ? "FELL" : null, "Two poles stacked on a cart, both held upright by a linear policy.",
            Kwargs: new Dictionary<string, object?> {
                ["default_camera_config"] = new Dictionary<string, object?> {
                    ["trackbodyid"] = 0L, ["distance"] = 2.6, ["lookat"] = new[] { 0.0, 0.0, 0.55 }, ["elevation"] = -5.0,
                },
            }),
        // One Reacher episode is 50 steps (a second at its own 50 fps): four targets in a row at half speed make a clip.
        new EpisodeRecipe("Reacher-v5", "mujoco", 0, 25, () => new DelegatePolicy(ReacherReach),
            (_, _, ret) => ret > -6 ? "REACHED!" : null, "Inverse kinematics finds the joint angles that put the fingertip on each new target.",
            Hands: 4, HandLabel: "TARGET"),
        // Seed 73: the best of seeds 0-199 for the scripted push (the puck ends 0.145 from the goal; return -39.0).
        new EpisodeRecipe("Pusher-v5", "mujoco", 73, 20, () => new PusherScript(),
            (_, _, ret) => ret > -35 ? "PUSHED!" : null, "A seven-joint arm, steered by its fingertip's Jacobian, sweeps to the puck and nudges it goalward."),
        new EpisodeRecipe("LunarLander-v3", "box2d", 7, 50, () => new DelegatePolicy((env, o) => LunarLanderEnv.Heuristic(env, (NDArray)o!)),
            (term, _, ret) => term && ret > 100 ? "LANDED!" : term ? "CRASH" : null, "Gymnasium's own heuristic lands between the flags.",
            RenderEveryStep: true),
        new EpisodeRecipe("BipedalWalker-v3", "box2d", 0, 50, () => new BipedalPolicy(),
            (term, _, ret) => term && ret > 0 ? "FINISH!" : term ? "GAME OVER" : null, "Gymnasium's gait state machine walks the whole course.",
            RenderEveryStep: true),
        new EpisodeRecipe("BipedalWalkerHardcore-v3", "box2d", 10, 50, () => new BipedalPolicy(),
            (term, _, _) => term ? "GAME OVER" : null, "The same gait meets stumps, pits and stairs. It does not go well.",
            RenderEveryStep: true),
        // CarRacing terminates when the lap closes; seed 11 visits 98% of the tiles but meets the 1,000-step limit first,
        // so it truncates and earns no stamp (the score in its own frame tells the story).
        new EpisodeRecipe("CarRacing-v3", "box2d", 11, 50, () => new DelegatePolicy(CarRacingPursuit),
            (term, _, _) => term ? "LAP!" : null, "Pure pursuit on the track's centre line, braking into the bends."),
        // The locomotion envs render through their model's own "track" camera (mode trackcom), named explicitly: with no
        // camera named, Gymnasium 1.4 forces MuJoCo's free camera, which never follows a body, so a runner leaves the
        // frame within a second (Gym.NET draws it the same way).
        new EpisodeRecipe("HalfCheetah-v5", "mujoco", 0, 20, Trained("HalfCheetah-v5"),
            (_, _, _) => null, "The autopilot trained in C# on this engine; in Gymnasium it scores the same, to the last bit.", Kwargs: TrackCamera),
        new EpisodeRecipe("Hopper-v5", "mujoco", 0, 125, Trained("Hopper-v5"),
            (term, _, _) => term ? "FELL" : null, "One leg, hopping forward without falling over.", Kwargs: TrackCamera),
        new EpisodeRecipe("Walker2d-v5", "mujoco", 0, 125, Trained("Walker2d-v5"),
            (term, _, _) => term ? "FELL" : null, "Two legs, walking forward.", Kwargs: TrackCamera),
        new EpisodeRecipe("Ant-v5", "mujoco", 0, 20, Trained("Ant-v5"),
            (term, _, _) => term ? "FLIPPED" : null, "Four legs, eight motors, walking forward.", Kwargs: TrackCamera),
        // A linear policy keeps Humanoid up for about 120 steps on average; seed 73 is the longest of seeds 0-199 (203
        // steps), and three-quarter speed stretches that past the 3.9 s the panel is on screen.
        new EpisodeRecipe("Humanoid-v5", "mujoco", 73, 50, Trained("Humanoid-v5"),
            (term, _, _) => term ? "FELL" : null, "Seventeen motors keep a humanoid on its feet for a while (shown at three-quarter speed).",
            Kwargs: TrackCamera),
        new EpisodeRecipe("HumanoidStandup-v5", "mujoco", 0, 67, Trained("HumanoidStandup-v5"),
            (_, _, _) => null, "Lying on the floor, working its way up.", Kwargs: TrackCamera),
        new EpisodeRecipe("Swimmer-v5", "mujoco", 0, 50, Trained("Swimmer-v5"),
            (_, _, _) => null, "Three links in a viscous fluid swim forward with a travelling wave (shown at twice its speed).", Kwargs: TrackCamera),
        new EpisodeRecipe("Blackjack-v1", "toy_text", 40, 1.6, () => new DelegatePolicy(BlackjackBasic),
            (_, _, ret) => ret > 0 ? "WIN!" : ret < 0 ? "BUST" : "PUSH", "Basic strategy against the dealer, hand after hand.", Hands: 5, ScoresHands: true),
        new EpisodeRecipe("FrozenLake-v1", "toy_text", 1, 8, () => new ToyTextPlan(),
            (term, _, ret) => ret > 0 ? "GOAL!" : term ? "SPLASH" : null, "The ice is slippery; value iteration still finds the safest way across."),
        new EpisodeRecipe("FrozenLake8x8-v1", "toy_text", 2, 12, () => new ToyTextPlan(),
            (term, _, ret) => ret > 0 ? "GOAL!" : term ? "SPLASH" : null, "The 8x8 lake, sliding past every hole."),
        new EpisodeRecipe("CliffWalking-v1", "toy_text", 0, 4, () => new ToyTextPlan(),
            (term, _, _) => term ? "SAFE!" : null, "The shortest path, one step from the cliff's edge."),
        new EpisodeRecipe("Taxi-v4", "toy_text", 12, 4, () => new ToyTextPlan(),
            (term, _, ret) => term && ret > 0 ? "DELIVERED!" : null, "Pick the passenger up, drop them at the hotel."),
    };

    /// <summary>Finds a recipe by id.</summary>
    /// <param name="id">The registered id.</param>
    /// <returns>The recipe.</returns>
    /// <exception cref="KeyNotFoundException">No recipe plays that id.</exception>
    public static EpisodeRecipe Get(string id) =>
        All.FirstOrDefault(r => r.Id == id) ?? throw new KeyNotFoundException($"no showreel recipe for '{id}'");

    /// <summary>
    ///     A factory of agents playing the committed policy of an id (<c>policies/&lt;id&gt;.json</c> next to the program),
    ///     the file read once, on first use, and shared by every player.
    /// </summary>
    /// <param name="id">The registered id.</param>
    /// <returns>The factory.</returns>
    /// <remarks>The file is read lazily, so a missing policy fails when its env is first played, naming the file.</remarks>
    private static Func<IPolicy> Trained(string id) {
        var policy = new Lazy<LinearPolicy>(() => LinearPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policies", id + ".json")));
        return () => new LinearPolicyAgent(policy.Value);
    }

    /// <summary>Reads an observation's values as float64, whatever its dtype.</summary>
    /// <param name="observation">An <see cref="NDArray"/> observation.</param>
    /// <returns>The values in C order.</returns>
    /// <exception cref="InvalidCastException">The observation is not an <see cref="NDArray"/>.</exception>
    private static double[] Values(object? observation) {
        var a = (NDArray)observation!;
        using NDArray d = a.astype(NPTypeCode.Double);
        return d.ToArray<double>();
    }

    /// <summary>A discrete action as Gym.NET's discrete spaces sample them: a 0-d int64 array.</summary>
    /// <param name="a">The action index.</param>
    /// <returns>The action.</returns>
    private static NDArray Discrete(long a) => NDArray.Scalar(a);

    /// <summary>
    ///     CartPole: a linear state-feedback controller around a set point that sways sinusoidally, so the cart glides from
    ///     side to side with the pole balanced (a still cart makes a dull clip).
    /// </summary>
    /// <param name="env">The env, whose elapsed steps set the sway's phase.</param>
    /// <param name="o">The observation <c>[x, x_dot, theta, theta_dot]</c>.</param>
    /// <returns>Push right (1) when the feedback is positive, else left (0).</returns>
    private static object CartPoleSway(IEnv env, object? o) {
        double[] s = Values(o);
        // The step counter lives in TimeLimit; reading it keeps the policy stateless and the sway in phase with the env.
        long t = Convert.ToInt64(env.GetWrapperAttr("_elapsed_steps"), CultureInfo.InvariantCulture);
        double target = 1.2 * Math.Sin(t * 2 * Math.PI / 250.0);
        double u = 0.6 * (s[0] - target) + 1.2 * s[1] + 12.0 * s[2] + 2.0 * s[3];
        return Discrete(u > 0 ? 1 : 0);
    }

    /// <summary>
    ///     InvertedPendulum: linear state feedback <c>u = 5.330 x + 8.469 theta + 1.109 x_dot + 0.634 theta_dot</c>, found by
    ///     random search to hold the pole for all 1,000 steps on seeds 0-9 with the cart at mean |x| = 0.003. The ARS policy
    ///     balanced too, but let the cart drift to the end of its rail, out of the camera's centre.
    /// </summary>
    /// <param name="env">Unused.</param>
    /// <param name="o">The observation <c>[x, theta, x_dot, theta_dot]</c>.</param>
    /// <returns>A float32 force in [-3, 3].</returns>
    private static object InvertedPendulumBalance(IEnv env, object? o) {
        double[] s = Values(o);
        double u = 5.330 * s[0] + 8.469 * s[1] + 1.109 * s[2] + 0.634 * s[3];
        return np.array(new[] { (float)Math.Clamp(u, -3.0, 3.0) });
    }

    /// <summary>
    ///     Reacher: analytic inverse kinematics of the two-link arm (links 0.1 and 0.11) gives the joint angles that put the
    ///     fingertip on the target, choosing the elbow side nearer the current pose, and a joint-space PD controller
    ///     (gain 0.5, damping 0.1) drives the joints there. Measured on seeds 0-19: mean return -4.14 (-7.2 to -0.7),
    ///     where the ARS linear policy plateaued near -9.
    /// </summary>
    /// <param name="env">Unused.</param>
    /// <param name="o">
    ///     The observation: cos and sin of both joints, the target's x and y, both joint velocities, and the fingertip's
    ///     offset from the target.
    /// </param>
    /// <returns>A float32 pair of joint torques in [-1, 1].</returns>
    private static object ReacherReach(IEnv env, object? o) {
        double[] s = Values(o);
        double q1 = Math.Atan2(s[2], s[0]), q2 = Math.Atan2(s[3], s[1]);
        double x = s[4], y = s[5];
        const double l1 = 0.1, l2 = 0.11, kp = 0.5, kd = 0.1;
        // The target is always within reach in Reacher (radius at most 0.2); the clamp only guards rounding at the edge.
        double c2 = Math.Clamp((x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1, 1);
        double target1 = 0, target2 = 0, nearest = double.MaxValue;
        foreach (double elbow in new[] { 1.0, -1.0 }) {
            double t2 = elbow * Math.Acos(c2);
            double t1 = Math.Atan2(y, x) - Math.Atan2(l2 * Math.Sin(t2), l1 + l2 * Math.Cos(t2));
            double distance = Math.Abs(Math.IEEERemainder(t1 - q1, 2 * Math.PI)) + Math.Abs(Math.IEEERemainder(t2 - q2, 2 * Math.PI));
            if (distance < nearest) {
                nearest = distance;
                target1 = t1;
                target2 = t2;
            }
        }
        double u1 = kp * Math.IEEERemainder(target1 - q1, 2 * Math.PI) - kd * s[6];
        double u2 = kp * Math.IEEERemainder(target2 - q2, 2 * Math.PI) - kd * s[7];
        return np.array(new[] { (float)Math.Clamp(u1, -1, 1), (float)Math.Clamp(u2, -1, 1) });
    }

    /// <summary>MountainCar: accelerate in the direction of motion, the energy-pumping rule.</summary>
    /// <param name="env">Unused.</param>
    /// <param name="o">The observation <c>[position, velocity]</c>.</param>
    /// <returns>Accelerate right (2) when moving right or at rest, else left (0).</returns>
    private static object MountainCarPump(IEnv env, object? o) => Discrete(Values(o)[1] >= 0 ? 2 : 0);

    /// <summary>
    ///     Pendulum: pump energy with full torque along the swing until the pendulum has the energy of the upright rest,
    ///     then a PD controller near the top. Energy is <c>w^2/2 + 15 cos(theta)</c> for Gymnasium's g = 10, m = l = 1.
    /// </summary>
    /// <param name="env">Unused.</param>
    /// <param name="o">The observation <c>[cos theta, sin theta, theta_dot]</c>.</param>
    /// <returns>A float32 torque in [-2, 2].</returns>
    private static object PendulumSwingUp(IEnv env, object? o) {
        double[] s = Values(o);
        double th = Math.Atan2(s[1], s[0]), w = s[2];
        double energy = 0.5 * w * w + 15.0 * Math.Cos(th);
        double u = Math.Cos(th) > 0.85 ? -(12.0 * th + 2.5 * w) : energy < 15.0 ? 2.0 * (w >= 0 ? 1 : -1) : -0.5 * Math.Sign(w);
        return np.array(new[] { (float)Math.Clamp(u, -2.0, 2.0) });
    }

    /// <summary>
    ///     Acrobot: torque in the direction the second link turns, which feeds the swing energy (measured over seeds 0-9: the
    ///     tip reaches the line in 83 steps on average; seed 1 in 65).
    /// </summary>
    /// <param name="env">Unused.</param>
    /// <param name="o">The observation; index 5 is the second joint's angular velocity.</param>
    /// <returns>+1 torque (2) or -1 torque (0).</returns>
    private static object AcrobotPump(IEnv env, object? o) => Discrete(Values(o)[5] > 0 ? 2 : 0);

    /// <summary>
    ///     CarRacing: pure pursuit. Aim at the centre-line point five track steps past the nearest one, steer by the angle
    ///     to it, throttle up to a speed that drops with the angle, brake hard into sharp bends.
    /// </summary>
    /// <param name="env">The env; its unwrapped <see cref="CarRacingEnv"/> exposes the car and the track.</param>
    /// <param name="o">Unused (the 96x96 pixels are not needed when the track is known).</param>
    /// <returns>A float32 <c>[steer, gas, brake]</c>.</returns>
    private static object CarRacingPursuit(IEnv env, object? o) {
        var race = (CarRacingEnv)env.Unwrapped;
        var hull = race.Car!.Hull!;
        var p = hull.Position;
        double angle = hull.Angle;
        var track = race.Track;
        int nearest = 0;
        double best = double.MaxValue;
        for (int i = 0; i < track.Count; i++) {
            double dx0 = track[i].X - p.x, dy0 = track[i].Y - p.y, d = dx0 * dx0 + dy0 * dy0;
            if (d < best) {
                best = d;
                nearest = i;
            }
        }
        var target = track[(nearest + 5) % track.Count];
        double dx = target.X - p.x, dy = target.Y - p.y;
        // The hull's forward axis is its local +y: (-sin a, cos a) in world coordinates.
        double fx = -Math.Sin(angle), fy = Math.Cos(angle);
        double error = Math.Atan2(fx * dy - fy * dx, fx * dx + fy * dy);
        var v = hull.LinearVelocity;
        double speed = Math.Sqrt(v.x * v.x + v.y * v.y);
        float steer = (float)Math.Clamp(-error * 2.5, -1, 1);
        float gas = speed < 45 - 30 * Math.Min(1, Math.Abs(error)) ? 0.6f : 0f;
        float brake = Math.Abs(error) > 0.6 && speed > 30 ? 0.4f : 0f;
        return np.array(new[] { steer, gas, brake });
    }

    /// <summary>
    ///     Blackjack: basic strategy for a dealer who stands on 17 (Gymnasium's rules): hit soft totals below 18 (and soft
    ///     18 against 9, 10 or an ace), hit hard totals below 12, hit 12-16 against a 7 or better or an ace, hit 12 against
    ///     2 or 3.
    /// </summary>
    /// <param name="env">Unused.</param>
    /// <param name="o">The observation tuple <c>(player sum, dealer card, usable ace)</c> of 0-d arrays.</param>
    /// <returns>Hit (1) or stick (0).</returns>
    private static object BlackjackBasic(IEnv env, object? o) {
        var t = (object?[])o!;
        long sum = Convert.ToInt64(((NDArray)t[0]!).GetValue(), CultureInfo.InvariantCulture);
        long dealer = Convert.ToInt64(((NDArray)t[1]!).GetValue(), CultureInfo.InvariantCulture);
        bool soft = Convert.ToInt64(((NDArray)t[2]!).GetValue(), CultureInfo.InvariantCulture) != 0;
        bool hit = soft
            ? sum < 18 || (sum == 18 && (dealer >= 9 || dealer == 1))
            : sum < 12 || (sum < 17 && (dealer >= 7 || dealer == 1)) || (sum == 12 && dealer <= 3);
        return Discrete(hit ? 1 : 0);
    }

    /// <summary>
    ///     Pusher: an operational-space controller on the fingertip (<c>tips_arm</c>), through three phases: hover behind
    ///     the puck (on the line from the goal through it), descend, then push toward a point just past the puck on the
    ///     goal's side; losing the puck sideways sends it back to hovering. Torques are <c>J^T (kp (target - tip) - kd
    ///     tip_velocity) - 0.3 qdot</c>, clipped to the motors' ±2.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The 3x7 Jacobian is taken by forward differences of <c>MjKinematics</c> on a scratch <see cref="MjData"/>, so the
    ///     env's own data is only read. Every position comes from that scratch copy, never from the env's data: after
    ///     <c>mj_step</c>, <c>xpos</c> still describes the step's start while <c>qpos</c> has moved on, and differencing the
    ///     two made a nonsense Jacobian (found while building this).
    ///     </para>
    ///     <para>
    ///     It does not solve Pusher: the point behind the puck often lies inside the arm's close-in reach, which the joint
    ///     limits forbid. Measured over seeds 0-199 its best episode (seed 73) leaves the puck 0.145 from the goal, return
    ///     -39.0, against -46.7 for the ARS linear policy, which barely moved the arm. It is kept because the sweep to the
    ///     puck shows the env at work.
    ///     </para>
    /// </remarks>
    private sealed class PusherScript : IPolicy, IDisposable {
        /// <summary>The position gain on the fingertip error.</summary>
        private const double Kp = 20;

        /// <summary>The damping gain on the fingertip velocity.</summary>
        private const double Kd = 2;

        /// <summary>How far behind the puck (away from the goal) the push starts.</summary>
        private const double Behind = 0.07;

        /// <summary>How high above the puck the fingertip hovers before descending.</summary>
        private const double Hover = 0.06;

        /// <summary>The phase: 0 hover behind the puck, 1 descend, 2 push.</summary>
        private int _phase;

        /// <summary>A scratch copy of the simulation for the kinematics, created on the first step.</summary>
        private MjData? _scratch;

        /// <inheritdoc />
        /// <exception cref="InvalidCastException">The env is not a MuJoCo env.</exception>
        public object Act(IEnv env, object? observation) {
            var mujoco = (MujocoEnv)env.Unwrapped;
            MjModel model = mujoco.Model;
            _scratch ??= new MjData(model);
            double[] q;
            using (NDArray qpos = mujoco.Data.Qpos)
                q = qpos.ToArray<double>();
            double[] qd;
            using (NDArray qvel = mujoco.Data.Qvel)
                qd = qvel.ToArray<double>();
            Kinematics(model, q);
            double[] tip = Position("tips_arm"), puck = Position("object"), goal = Position("goal");
            var jacobian = new double[3, 7];
            for (int i = 0; i < 7; i++) {
                double[] nudged = (double[])q.Clone();
                nudged[i] += 1e-5;
                Kinematics(model, nudged);
                double[] moved = Position("tips_arm");
                for (int k = 0; k < 3; k++)
                    jacobian[k, i] = (moved[k] - tip[k]) / 1e-5;
            }
            double ux = puck[0] - goal[0], uy = puck[1] - goal[1], length = Math.Sqrt(ux * ux + uy * uy) + 1e-9;
            ux /= length;
            uy /= length;
            double bx = puck[0] + Behind * ux, by = puck[1] + Behind * uy, z = puck[2];
            double[] target;
            if (_phase == 0) {
                target = new[] { bx, by, z + Hover };
                if (Distance(tip, target) < 0.03)
                    _phase = 1;
            } else if (_phase == 1) {
                target = new[] { bx, by, z };
                if (Distance(tip, target) < 0.02)
                    _phase = 2;
            } else {
                target = new[] { puck[0] - 0.05 * ux, puck[1] - 0.05 * uy, z };
                double tx = tip[0] - puck[0], ty = tip[1] - puck[1];
                double along = tx * ux + ty * uy, side = Math.Abs(-tx * uy + ty * ux);
                if (side > 0.05 || along < 0)
                    _phase = 0;
            }
            var velocity = new double[3];
            for (int k = 0; k < 3; k++)
                for (int i = 0; i < 7; i++)
                    velocity[k] += jacobian[k, i] * qd[i];
            var action = new float[7];
            for (int i = 0; i < 7; i++) {
                double torque = -0.3 * qd[i];
                for (int k = 0; k < 3; k++)
                    torque += jacobian[k, i] * (Kp * (target[k] - tip[k]) - Kd * velocity[k]);
                action[i] = (float)Math.Clamp(torque, -2, 2);
            }
            return np.array(action);
        }

        /// <summary>Sets the scratch copy's positions and recomputes its kinematics.</summary>
        /// <param name="model">The model.</param>
        /// <param name="q">The positions.</param>
        private void Kinematics(MjModel model, double[] q) {
            using (NDArray state = np.array(q))
                Mujoco.MjSetState(model, _scratch!, state, MjtState.Qpos);
            Mujoco.MjKinematics(model, _scratch!);
        }

        /// <summary>A body frame's world position in the scratch copy.</summary>
        /// <param name="body">The body's name.</param>
        /// <returns>x, y, z.</returns>
        private double[] Position(string body) {
            using NDArray x = _scratch!.Body(body).Xpos;
            return x.ToArray<double>();
        }

        /// <summary>The Euclidean distance between two points.</summary>
        /// <param name="a">One point.</param>
        /// <param name="b">The other.</param>
        /// <returns>The distance.</returns>
        private static double Distance(double[] a, double[] b) =>
            Math.Sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));

        /// <summary>Releases the scratch copy's arrays (the player disposes its policy with its env).</summary>
        public void Dispose() {
            _scratch?.Dispose();
            _scratch = null;
        }
    }

    /// <summary>BipedalWalker: Gymnasium's gait state machine, one instance per episode because it keeps the phase.</summary>
    private sealed class BipedalPolicy : IPolicy {
        /// <summary>The gait state machine.</summary>
        private readonly BipedalWalkerHeuristics _gait = new();

        /// <inheritdoc />
        public object Act(IEnv env, object? observation) => _gait.StepHeuristic((NDArray)observation!);
    }

    /// <summary>
    ///     The toy-text agents: value iteration (discount 0.99) over the unwrapped env's own transition table <c>P</c>, then
    ///     the greedy action in the current state. Computed once per episode, at the first step.
    /// </summary>
    private sealed class ToyTextPlan : IPolicy {
        /// <summary>The greedy action per state, or null before the first step.</summary>
        private long[]? _plan;

        /// <inheritdoc />
        /// <exception cref="InvalidOperationException">The env has no transition table.</exception>
        public object Act(IEnv env, object? observation) {
            _plan ??= Solve(env.Unwrapped switch {
                FrozenLakeEnv f => f.P,
                CliffWalkingEnv c => c.P,
                TaxiEnv x => x.P,
                _ => throw new InvalidOperationException($"{env.Unwrapped.GetType().Name} has no transition table"),
            });
            long state = Convert.ToInt64(((NDArray)observation!).GetValue(), CultureInfo.InvariantCulture);
            return Discrete(_plan[state]);
        }

        /// <summary>Value iteration to convergence (1e-10), then the greedy policy.</summary>
        /// <param name="p">The transition table: state, then action, then the possible transitions.</param>
        /// <returns>The greedy action per state; ties go to the lowest action, as NumPy's argmax would.</returns>
        private static long[] Solve(IReadOnlyDictionary<long, IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>>> p) {
            const double gamma = 0.99;
            int states = p.Count, actions = p[0].Count;
            var v = new double[states];
            double Q(int s, int a) {
                double q = 0;
                foreach (var t in p[s][a])
                    q += t.Probability * (t.Reward + (t.Terminated ? 0 : gamma * v[t.NextState]));
                return q;
            }
            for (int sweep = 0; sweep < 10_000; sweep++) {
                double delta = 0;
                for (int s = 0; s < states; s++) {
                    double best = double.NegativeInfinity;
                    for (int a = 0; a < actions; a++)
                        best = Math.Max(best, Q(s, a));
                    delta = Math.Max(delta, Math.Abs(best - v[s]));
                    v[s] = best;
                }
                if (delta < 1e-10)
                    break;
            }
            var plan = new long[states];
            for (int s = 0; s < states; s++) {
                double best = double.NegativeInfinity;
                for (int a = 0; a < actions; a++) {
                    double q = Q(s, a);
                    if (q > best + 1e-12) {
                        best = q;
                        plan[s] = a;
                    }
                }
            }
            return plan;
        }
    }
}
