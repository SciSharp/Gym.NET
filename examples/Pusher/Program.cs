#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.MuJoCo;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using NumSharp;

namespace Gym.Examples.Pusher {
    /// <summary>
    ///     Sweeps Pusher-v5's arm to the puck and nudges it toward the goal with operational-space control of the
    ///     fingertip. <c>dotnet run --project examples/Pusher -c Release</c> plays one seeded episode headless and
    ///     prints its steps and return; <c>-- --render</c> plays it in a window.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The agent steers the fingertip (the body <c>tips_arm</c>) rather than the seven joints: hover behind the
    ///     puck on the line from the goal through it, descend, then push toward a point just past the puck; losing the
    ///     puck sideways sends it back to hovering. The torques are <c>J^T (kp (target - tip) - kd tip_velocity) - 0.3
    ///     q_dot</c>, clipped to the motors' ±2, with the fingertip's 3x7 Jacobian taken by finite differences of the
    ///     kinematics.
    ///     </para>
    ///     <para>
    ///     The Jacobian uses a scratch copy of the simulation (<c>MjData</c>) and reads every position from it: after a
    ///     step, MuJoCo's body positions (<c>xpos</c>) still describe the step's start while the joint positions
    ///     (<c>qpos</c>) have moved on, so differencing one against the other gives a wrong Jacobian.
    ///     </para>
    ///     <para>
    ///     It does not solve Pusher: the point behind the puck often lies inside the reach the arm's joint limits
    ///     forbid. Over seeds 0-199 its best episode, seed 73, leaves the puck 0.145 from the goal (return -39.0),
    ///     where a linear policy trained by Augmented Random Search barely moved the arm (-46.7).
    ///     </para>
    ///     <para>
    ///     The default seed, 73, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class Program {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Pusher-v5";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 73;

        /// <summary>Plays the episodes the arguments ask for with the agent.</summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> is refused, with the reason
        ///     this env cannot be played from the keyboard. Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played, 2 when an option's value is not a whole number, or for <c>--play</c>.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> sets Avalonia's viewer before the env first draws.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                Console.Error.WriteLine($"{Id}: its 7 motors would need 14 keys held in combinations, too many to play; --render shows the agent instead.");
                return 2;
            }
            // Human mode draws inside every reset and step into the app-wide viewer (DR-20); without one it raises
            // DependencyNotInstalled at the first frame, as Gymnasium does without pygame.
            if (render)
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
            double sum = 0;
            for (long episode = 0; episode < episodes; episode++) {
                (int steps, double total, string ending) = PlayEpisode(seed + episode, render);
                sum += total;
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{Id}, seed {seed + episode}: {steps} step{(steps == 1 ? "" : "s")}, return {total:F1}, {ending}"));
            }
            if (episodes > 1)
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Mean return over {episodes} episodes: {sum / episodes:F1}"));
            return 0;
        }

        /// <summary>Plays one episode with the agent and reports how it went.</summary>
        /// <param name="seed">The reset seed, which draws the starting state.</param>
        /// <param name="render">
        ///     Whether to show the episode in a window, in human mode, paced at the env's frame rate.
        /// </param>
        /// <returns>
        ///     The steps taken, the return (float64, as every Gym.NET reward), and how the episode ended.
        /// </returns>
        /// <remarks>
        ///     The arrays a reset or a step returns belong to the caller (D19): each result is released once the agent
        ///     has acted on its observation, and the last one when the episode ends.
        /// </remarks>
        private static (int Steps, double Return, string Ending) PlayEpisode(long seed, bool render) {
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, render ? new { render_mode = "human" } : null);
            // The agent keeps its phase and a scratch simulation for the episode, then releases the scratch arrays.
            using var agent = new PusherAgent();
            ResetResult<NDArray> reset = env.Reset(seed: seed);
            // The result whose observation the agent reads next; a boxed copy shares its arrays, so disposing it frees them.
            IDisposable held = reset;
            NDArray observation = reset.Observation;
            int steps = 0;
            double total = 0;
            try {
                while (true) {
                    StepResult<NDArray> step;
                    using (NDArray action = agent.Act((MujocoEnv)env.Unwrapped))
                        step = env.Step(action);
                    // The agent is done with the previous observation: release it with its info, and hold the new result.
                    held.Dispose();
                    held = step;
                    observation = step.Observation;
                    steps++;
                    total += step.Reward;
                    if (step.Terminated || step.Truncated)
                        return (steps, total, step.Terminated ? "terminated" : "truncated by the time limit");
                }
            } finally {
                held.Dispose();
            }
        }

        /// <summary>
        ///     The agent: operational-space control of the fingertip through three phases, hover behind the puck,
        ///     descend, push (see the program's remarks). One instance per episode: it keeps the phase and a scratch
        ///     simulation.
        /// </summary>
        private sealed class PusherAgent : IDisposable {
            /// <summary>The position gain on the fingertip's error.</summary>
            private const double Kp = 20;

            /// <summary>The damping gain on the fingertip's velocity.</summary>
            private const double Kd = 2;

            /// <summary>How far behind the puck, away from the goal, the push starts.</summary>
            private const double Behind = 0.07;

            /// <summary>How high above the puck the fingertip hovers before it descends.</summary>
            private const double Hover = 0.06;

            /// <summary>The phase: 0 hover behind the puck, 1 descend, 2 push.</summary>
            private int _phase;

            /// <summary>A scratch copy of the simulation for the kinematics, created at the first step.</summary>
            private MjData? _scratch;

            /// <summary>The torques for the arm's current state.</summary>
            /// <param name="pusher">The unwrapped env, whose model and data are read, never written.</param>
            /// <returns>A new float32 array of seven torques in [-2, 2]; the caller disposes it.</returns>
            public NDArray Act(MujocoEnv pusher) {
                MjModel model = pusher.Model;
                _scratch ??= new MjData(model);
                double[] q, qd;
                using (NDArray qpos = pusher.Data.Qpos)
                    q = qpos.ToArray<double>();
                using (NDArray qvel = pusher.Data.Qvel)
                    qd = qvel.ToArray<double>();
                Kinematics(model, q);
                double[] tip = Position("tips_arm"), puck = Position("object"), goal = Position("goal");
                // The fingertip's 3x7 Jacobian by forward differences: nudge one joint at a time and see where the tip goes.
                var jacobian = new double[3, 7];
                for (int i = 0; i < 7; i++) {
                    double[] nudged = (double[])q.Clone();
                    nudged[i] += 1e-5;
                    Kinematics(model, nudged);
                    double[] moved = Position("tips_arm");
                    for (int k = 0; k < 3; k++)
                        jacobian[k, i] = (moved[k] - tip[k]) / 1e-5;
                }
                // The push direction: from the goal through the puck, so pushing along it moves the puck goalward.
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
                    // Losing the puck sideways, or overtaking it, starts the approach over.
                    if (side > 0.05 || along < 0)
                        _phase = 0;
                }
                var velocity = new double[3];
                for (int k = 0; k < 3; k++)
                    for (int i = 0; i < 7; i++)
                        velocity[k] += jacobian[k, i] * qd[i];
                var action = new float[7];
                for (int i = 0; i < 7; i++) {
                    // J^T (kp error - kd velocity), plus joint damping that keeps the redundant joints from flailing.
                    double torque = -0.3 * qd[i];
                    for (int k = 0; k < 3; k++)
                        torque += jacobian[k, i] * (Kp * (target[k] - tip[k]) - Kd * velocity[k]);
                    action[i] = (float)Math.Clamp(torque, -2, 2);
                }

                return np.array(action);
            }

            /// <summary>Sets the scratch copy's joint positions and recomputes its body positions.</summary>
            /// <param name="model">The model.</param>
            /// <param name="q">The joint positions.</param>
            private void Kinematics(MjModel model, double[] q) {
                using (NDArray state = np.array(q))
                    Mujoco.MjSetState(model, _scratch!, state, MjtState.Qpos);
                Mujoco.MjKinematics(model, _scratch!);
            }

            /// <summary>A body frame's world position in the scratch copy.</summary>
            /// <param name="body">The body's name in Pusher's model.</param>
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

            /// <summary>Releases the scratch copy's arrays.</summary>
            public void Dispose() {
                _scratch?.Dispose();
                _scratch = null;
            }
        }

        /// <summary>Reads a whole-number option's value, or the fallback when the option is absent.</summary>
        /// <param name="args">The command line.</param>
        /// <param name="name">The option, <c>--seed</c> or <c>--episodes</c>.</param>
        /// <param name="fallback">The value when the option is absent.</param>
        /// <param name="value">The value read, or <paramref name="fallback"/>.</param>
        /// <returns>
        ///     False, after printing why, when the option has no value or one that is not a non-negative whole number.
        /// </returns>
        private static bool TryNumber(string[] args, string name, long fallback, out long value) {
            value = fallback;
            int at = Array.IndexOf(args, name);
            if (at < 0)
                return true;
            // Gymnasium's seeds are non-negative, and so is a count of episodes: NumberStyles.None takes digits only.
            if (at + 1 < args.Length && long.TryParse(args[at + 1], NumberStyles.None, CultureInfo.InvariantCulture, out value))
                return true;
            Console.Error.WriteLine($"{name} needs a non-negative whole number");
            return false;
        }
    }
}
