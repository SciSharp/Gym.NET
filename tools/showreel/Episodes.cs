using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
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
/// <param name="Family">Gymnasium's env family: <c>classic_control</c>, <c>box2d</c> or <c>toy_text</c>.</param>
/// <param name="Seed">The reset seed; the policies were checked to play well from it.</param>
/// <param name="StepsPerSecond">Playback speed in env steps per video second (the env's own render fps by default).</param>
/// <param name="Policy">Makes a fresh policy for each playback.</param>
/// <param name="Stamp">The text stamped on the panel when the episode ends, from (terminated, truncated, return); null for none.</param>
/// <param name="Blurb">One line for the docs site's gallery: what the agent is doing.</param>
/// <param name="Hands">Episodes played back to back (Blackjack deals several hands); 1 for the others.</param>
internal sealed record EpisodeRecipe(
    string Id,
    string Family,
    int Seed,
    double StepsPerSecond,
    Func<IPolicy> Policy,
    Func<bool, bool, double, string?> Stamp,
    string Blurb,
    int Hands = 1) {
    /// <summary>Ticks a finished hand is held on screen before the next one is dealt (Blackjack only).</summary>
    public int HandPauseTicks => 2;
}

/// <summary>
///     The thirteen episodes of the showreel, one per visually distinct registered id (the maintainer's choice: variants
///     that only change the action space, such as LunarLanderContinuous-v3, look the same and are left out), and the
///     policies that play them.
/// </summary>
/// <remarks>
///     Every policy and seed was checked to play a good episode: CartPole balances for all 500 steps while swaying,
///     MountainCar climbs in 122 steps, Pendulum swings up and holds, Acrobot reaches the line in 65 steps, LunarLander
///     lands for 248.6, BipedalWalker finishes the course for 314.5, the hardcore walker trips over a stump (on purpose,
///     for the GAME OVER stamp), CarRacing scores 880.7, and the toy-text agents follow value iteration over the env's own
///     transition table. LunarLander and BipedalWalker use Gymnasium's own heuristics as Gym.NET ships them.
/// </remarks>
internal static class Episodes {
    /// <summary>The recipes in the order the gallery lists them: classic control, Box2D, toy text.</summary>
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
        new EpisodeRecipe("LunarLander-v3", "box2d", 7, 50, () => new DelegatePolicy((env, o) => LunarLanderEnv.Heuristic(env, (NDArray)o!)),
            (term, _, ret) => term && ret > 100 ? "LANDED!" : term ? "CRASH" : null, "Gymnasium's own heuristic lands between the flags."),
        new EpisodeRecipe("BipedalWalker-v3", "box2d", 0, 50, () => new BipedalPolicy(),
            (term, _, ret) => term && ret > 0 ? "FINISH!" : term ? "GAME OVER" : null, "Gymnasium's gait state machine walks the whole course."),
        new EpisodeRecipe("BipedalWalkerHardcore-v3", "box2d", 10, 50, () => new BipedalPolicy(),
            (term, _, _) => term ? "GAME OVER" : null, "The same gait meets stumps, pits and stairs. It does not go well."),
        // CarRacing terminates when the lap closes; seed 11 visits 98% of the tiles but meets the 1,000-step limit first,
        // so it truncates and earns no stamp (the score in its own frame tells the story).
        new EpisodeRecipe("CarRacing-v3", "box2d", 11, 50, () => new DelegatePolicy(CarRacingPursuit),
            (term, _, _) => term ? "LAP!" : null, "Pure pursuit on the track's centre line, braking into the bends."),
        new EpisodeRecipe("Blackjack-v1", "toy_text", 40, 1.6, () => new DelegatePolicy(BlackjackBasic),
            (_, _, ret) => ret > 0 ? "WIN!" : ret < 0 ? "BUST" : "PUSH", "Basic strategy against the dealer, hand after hand.", Hands: 5),
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
