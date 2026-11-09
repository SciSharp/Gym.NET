using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Wrappers {
    /// <summary>
    ///     The action wrappers that keep state across steps (Gymnasium's <c>StickyAction</c> and <c>RepeatAction</c>) against
    ///     Gymnasium's, live: sampled episodes through each over discrete, continuous and slippery envs, the sticky run's
    ///     state afterwards, and every refusal with its message.
    /// </summary>
    /// <remarks>
    ///     <c>StickyAction</c> draws from the env's own generator, between the env's draws (FrozenLake's slippery moves), so
    ///     the pair's generator comparison after every step checks the order of the draws, not only their values. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_action.py">gymnasium/wrappers/stateful_action.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class StatefulActionLiveParityTests : ParityTestBase {
        /// <summary>
        ///     <c>StickyAction(make(id), p, duration)</c> through 200 sampled steps across episodes: fixed and drawn run
        ///     lengths, Discrete and Box actions, and FrozenLake's slippery draws interleaved with the wrapper's.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="probability">The probability that a run starts.</param>
        /// <param name="low">The run length, or its lower bound.</param>
        /// <param name="high">The upper bound; equal to <paramref name="low"/> for an int duration.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 0.25, 1L, 1L)]
        [DataRow("CartPole-v1", 0.5, 1L, 3L)]
        [DataRow("Pendulum-v1", 0.3, 2L, 2L)]
        [DataRow("MountainCarContinuous-v0", 0.4, 2L, 4L)]
        [DataRow("FrozenLake-v1", 0.25, 1L, 2L)]
        public void StickyAction_Episodes_MatchGymnasium(string id, double probability, long low, long high) {
            bool isRange = low != high;
            using var pair = new GymPair(PyGymEnv.Wrap($"StickyAction(make('{id}'), {probability}, {(isRange ? $"({low}, {high})" : low.ToString())})", () => {
                using PyObject made = Python.gymnasium.make(id);
                if (!isRange)
                    return Python.gymnasium.wrappers.StickyAction(made, probability, low);
                using PyObject range = Python.evalStmts($"result = ({low}, {high})\n", "result");
                return Python.gymnasium.wrappers.StickyAction(made, probability, range);
            }), new GymNetEnv(new StickyAction<NDArray, NDArray>(WrapperParity.Make(id), probability, isRange ? (object)(low, high) : low)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "the run's state", "result = (env.is_sticky_actions, int(env.num_repeats), env.repeats_taken, env.repeat_action_duration_range)",
                env => {
                    var sticky = (StickyAction<NDArray, NDArray>)env;
                    return new object?[] { sticky.IsStickyActions, sticky.NumRepeats, sticky.RepeatsTaken, sticky.RepeatActionDurationRange };
                });
            WrapperParity.AssertAttribute(pair, "last_action", "result = env.last_action", env => ((StickyAction<NDArray, NDArray>)env).LastAction);
        }

        /// <summary>
        ///     <c>StickyAction</c>'s refusals, in Gymnasium's order: probabilities outside [0, 1) and NaN, then durations that
        ///     are a list, too long, reversed, or below 1 (an int below 1 is reported as the pair it becomes).
        /// </summary>
        [TestMethod]
        public void StickyAction_Refusals_MatchGymnasium() {
            foreach (double probability in new[] { -1.0, 1.0, 1.5, double.NaN }) {
                AssertRefusal($"StickyAction(make('CartPole-v1'), {probability})", () => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1");
                    return Python.gymnasium.wrappers.StickyAction(made, probability, null);
                }, () => new StickyAction<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), probability));
            }

            var durations = new (string Python, object DotNet)[] {
                ("-4", -4L),
                ("0", 0L),
                ("(0, 0)", (0L, 0L)),
                ("(4, 2)", (4L, 2L)),
                ("(1, 2, 3)", (1L, 2L, 3L)),
                ("[1, 2]", new List<object?> { 1L, 2L }),
                ("1.5", 1.5),
            };
            foreach ((string python, object dotnet) in durations) {
                AssertRefusal($"StickyAction(make('CartPole-v1'), 0.5, {python})", () => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1");
                    using PyObject duration = Python.evalStmts($"result = {python}\n", "result");
                    return Python.gymnasium.wrappers.StickyAction(made, 0.5, duration);
                }, () => new StickyAction<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 0.5, dotnet));
            }
        }

        /// <summary>
        ///     <c>RepeatAction(make(id), num_repeats)</c> through 150 sampled steps: the summed rewards, the last step's
        ///     observation and info, and the early stop at an episode's end.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="numRepeats">The repeats.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 4L)]
        [DataRow("CartPole-v1", 1L)]
        [DataRow("Pendulum-v1", 3L)]
        [DataRow("FrozenLake-v1", 2L)]
        [DataRow("Taxi-v4", 5L)]
        public void RepeatAction_Episodes_MatchGymnasium(string id, long numRepeats) {
            using var pair = new GymPair(PyGymEnv.Wrap($"RepeatAction(make('{id}'), {numRepeats})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.RepeatAction(made, numRepeats);
            }), new GymNetEnv(new RepeatAction<NDArray, NDArray>(WrapperParity.Make(id), numRepeats)));
            WrapperParity.RunEpisodes(pair, 150, TestContext);
        }

        /// <summary><c>RepeatAction</c>'s refusals of counts below 1.</summary>
        [TestMethod]
        public void RepeatAction_Refusals_MatchGymnasium() {
            foreach (long numRepeats in new[] { 0L, -1L, -100L }) {
                AssertRefusal($"RepeatAction(make('CartPole-v1'), {numRepeats})", () => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1");
                    return Python.gymnasium.wrappers.RepeatAction(made, numRepeats);
                }, () => new RepeatAction<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), numRepeats));
            }
        }

        /// <summary>Compares a wrapper's refusal on both sides: the same exception type and message.</summary>
        /// <param name="label">The comparison's name in reports.</param>
        /// <param name="python">Builds Gymnasium's wrapper (under the GIL).</param>
        /// <param name="dotnet">Builds Gym.NET's wrapper.</param>
        /// <exception cref="ParityException">Thrown when the outcomes differ.</exception>
        private static void AssertRefusal(string label, Func<PyObject> python, Func<object> dotnet) =>
            ByteContract.AssertSame(label, Outcome.CapturePython(python), Outcome.Capture(dotnet));
    }
}
