using System;
using System.Globalization;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Python.Runtime;

namespace Gym.Tests.Parity.Pair {
    /// <summary>Where the actions of a <see cref="GymPair.Run"/> come from (plan §5).</summary>
    public interface IActionSource {
        /// <summary>How reports name the source, for example <c>policy cartpole_balance</c>.</summary>
        string Description { get; }

        /// <summary>Called once after the trajectory's reset, before the first <see cref="Next"/>.</summary>
        /// <param name="pair">The pair being driven.</param>
        void Start(GymPair pair);

        /// <summary>The next action, carrying identical bits to both sides.</summary>
        /// <param name="pair">The pair being driven.</param>
        /// <returns>The action; the caller disposes it after the step.</returns>
        /// <exception cref="ParityException">Thrown by a source that compares something itself (a sampled source's two samples).</exception>
        ParityAction Next(GymPair pair);
    }

    /// <summary>The action sources of plan §5: open-loop, a Python policy, and seeded sampling.</summary>
    public static class ActionSources {
        /// <summary>
        ///     Open-loop discrete actions from a function of the step number (1 for the first step), independent of either
        ///     side's observations, so the trajectory is fixed in advance.
        /// </summary>
        /// <param name="actionAt">The action of each step.</param>
        /// <param name="description">How reports name the sequence, for example <c>0, 0, 1, 1, ...</c>.</param>
        /// <returns>The source.</returns>
        public static IActionSource OpenLoop(Func<int, long> actionAt, string description) => new OpenLoopSource(actionAt, description);

        /// <summary>The same discrete action at every step, for example to drive CartPole into termination.</summary>
        /// <param name="action">The action.</param>
        /// <returns>The source.</returns>
        public static IActionSource Constant(long action) => new OpenLoopSource(_ => action, $"always {action.ToString(CultureInfo.InvariantCulture)}");

        /// <summary>
        ///     A named policy evaluated on Gymnasium's own last observation, the way the port plan's A.7 recorded its
        ///     trajectories. Both sides receive the policy's action, so the comparison is of dynamics alone even when the .NET
        ///     side's observations differ.
        /// </summary>
        /// <param name="policy">
        ///     A policy <see cref="PyGymEnv.Policy"/> knows: <c>cartpole_balance</c> (a proportional-derivative controller
        ///     computed with Python's float64 arithmetic) or <c>lunar_lander_heuristic</c> (Gymnasium's own
        ///     <c>lunar_lander.heuristic</c>).
        /// </param>
        /// <returns>The source.</returns>
        public static IActionSource Policy(string policy) => new PolicySource(policy);

        /// <summary>
        ///     Each side seeds its own action space with <paramref name="seed"/> after the reset and samples it every step; the
        ///     two samples must match before the step is taken, which checks the action spaces' sampling as a by-product.
        /// </summary>
        /// <param name="seed">The action spaces' seed.</param>
        /// <returns>The source.</returns>
        public static IActionSource Sampled(long seed) => new SampledSource(seed);

        /// <summary>See <see cref="OpenLoop"/>.</summary>
        /// <param name="ActionAt">The action of each step.</param>
        /// <param name="Description">How reports name the sequence.</param>
        private sealed record OpenLoopSource(Func<int, long> ActionAt, string Description) : IActionSource {
            /// <inheritdoc />
            public void Start(GymPair pair) { }

            /// <inheritdoc />
            public ParityAction Next(GymPair pair) => ParityAction.Discrete(ActionAt(pair.StepsTaken + 1));
        }

        /// <summary>See <see cref="Policy"/>.</summary>
        /// <param name="Name">The policy's name.</param>
        private sealed record PolicySource(string Name) : IActionSource {
            /// <inheritdoc />
            public string Description => $"policy {Name}";

            /// <inheritdoc />
            public void Start(GymPair pair) { }

            /// <inheritdoc />
            public ParityAction Next(GymPair pair) => pair.Python.Policy(Name);
        }

        /// <summary>See <see cref="Sampled"/>.</summary>
        /// <param name="Seed">The action spaces' seed.</param>
        private sealed record SampledSource(long Seed) : IActionSource {
            /// <inheritdoc />
            public string Description => $"action_space.sample() seeded {Seed.ToString(CultureInfo.InvariantCulture)}";

            /// <inheritdoc />
            public void Start(GymPair pair) {
                pair.Python.SeedActionSpace(Seed);
                pair.DotNet.SeedActionSpace(Seed);
            }

            /// <inheritdoc />
            public ParityAction Next(GymPair pair) {
                ParityAction action = PythonRuntime.Run(() => {
                    PyObject python = pair.Python.SampleActionRaw();
                    object dotnet;
                    try {
                        dotnet = pair.DotNet.SampleAction();
                    } catch {
                        // The Python sample has no owner yet: release it here, under the GIL, before the failure propagates.
                        python.Dispose();
                        throw;
                    }

                    // Takes ownership of both samples, so they are released even when the comparison below throws.
                    return ParityAction.Sampled(python, dotnet);
                });

                var differences = ByteContract.Compare("action", action.Facts, ValueFacts.Of(action.DotNet));
                if (differences.Count > 0) {
                    action.Dispose();
                    pair.Fail($"action_space.sample() for step #{pair.StepsTaken + 1}", differences);
                }

                return action;
            }
        }
    }
}
