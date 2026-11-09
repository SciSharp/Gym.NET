using System;
using System.Linq;
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
    ///     The observation wrappers that keep observations across steps (Gymnasium's <c>DelayObservation</c>,
    ///     <c>FrameStackObservation</c> and <c>MaxAndSkipObservation</c>) against Gymnasium's, live: sampled episodes through
    ///     each over array, integer, tuple and dict observations, what each keeps afterwards, and every refusal.
    /// </summary>
    /// <remarks>
    ///     What these cells pin beyond the arithmetic: the zeros of <c>create_zero_array</c> before a delay has passed, the
    ///     deque's padding after each reset (the reset's observation, the space's zero, or a given element), the stacks of
    ///     <c>concatenate</c> for every space kind, and the frame skip's stale buffer after an episode that ends early. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py">gymnasium/wrappers/stateful_observation.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class StatefulObservationLiveParityTests : ParityTestBase {
        /// <summary>
        ///     <c>DelayObservation(make(id), delay)</c> through 150 sampled steps across episodes, and the queue afterwards: no
        ///     delay, CartPole's float32 zeros, FrozenLake's <c>np.int64(0)</c> beside its Python ints, Blackjack's tuple of
        ///     zeros.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="delay">The delay.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 0L)]
        [DataRow("CartPole-v1", 3L)]
        [DataRow("FrozenLake-v1", 2L)]
        [DataRow("Blackjack-v1", 1L)]
        public void DelayObservation_Episodes_MatchGymnasium(string id, long delay) {
            using var pair = new GymPair(PyGymEnv.Wrap($"DelayObservation(make('{id}'), {delay})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.DelayObservation(made, delay);
            }), new GymNetEnv(id == "Blackjack-v1"
                ? new DelayObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make(id), delay)
                : new DelayObservation<NDArray, NDArray>(WrapperParity.Make(id), delay)));
            WrapperParity.RunEpisodes(pair, 150, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_queue", "result = tuple(env.observation_queue)",
                env => ((System.Collections.IEnumerable)Gym.Utils.PyAttributes.GetAttr(env, "observation_queue")!).Cast<object?>().ToArray());
        }

        /// <summary><c>DelayObservation</c>'s refusal of a negative delay.</summary>
        [TestMethod]
        public void DelayObservation_Refusals_MatchGymnasium() {
            AssertRefusal("DelayObservation(make('CartPole-v1'), -1)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.DelayObservation(made, -1);
            }, () => new DelayObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), -1));
        }

        /// <summary>
        ///     <c>FrameStackObservation(make(id), stack_size, padding_type)</c> through 150 sampled steps, the stacked space,
        ///     and the deque and padding afterwards: CartPole padded by the reset and by zeros, FrozenLake's ints stacked into
        ///     an int64 array, Blackjack's tuples into a tuple of arrays, and a stack of one.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="stackSize">The stack size.</param>
        /// <param name="padding">The padding mode.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 4L, "reset")]
        [DataRow("CartPole-v1", 3L, "zero")]
        [DataRow("CartPole-v1", 1L, "reset")]
        [DataRow("FrozenLake-v1", 3L, "reset")]
        [DataRow("Blackjack-v1", 2L, "zero")]
        [DataRow("Blackjack-v1", 3L, "reset")]
        public void FrameStackObservation_Episodes_MatchGymnasium(string id, long stackSize, string padding) {
            using var pair = new GymPair(PyGymEnv.Wrap($"FrameStackObservation(make('{id}'), {stackSize}, padding_type='{padding}')", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.FrameStackObservation(made, stackSize, padding);
            }), new GymNetEnv(id == "Blackjack-v1"
                ? new FrameStackObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make(id), stackSize, padding)
                : new FrameStackObservation<NDArray, NDArray>(WrapperParity.Make(id), stackSize, padding)));
            WrapperParity.RunEpisodes(pair, 150, TestContext);
            AssertFrameStackState(pair);
        }

        /// <summary>
        ///     <c>FrameStackObservation</c> padded with a given element, <c>np.array([1, -1, 0, 2], dtype=np.float32)</c>, over
        ///     CartPole: the padding stays the given one after every reset.
        /// </summary>
        [TestMethod]
        public void FrameStackObservation_CustomPadding_MatchesGymnasium() {
            using NDArray padding = np.array(new float[] { 1, -1, 0, 2 });
            using var pair = new GymPair(PyGymEnv.Wrap("FrameStackObservation(make('CartPole-v1'), 3, padding_type=np.array([1, -1, 0, 2], dtype=np.float32))", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject given = Python.evalStmts("import numpy as np\nresult = np.array([1, -1, 0, 2], dtype=np.float32)\n", "result");
                return Python.gymnasium.wrappers.FrameStackObservation(made, 3, given);
            }), new GymNetEnv(new FrameStackObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 3, padding)));
            WrapperParity.RunEpisodes(pair, 120, TestContext);
            AssertFrameStackState(pair);
        }

        /// <summary>
        ///     <c>FrameStackObservation</c> over a <c>Dict</c> observation (<c>TimeAwareObservation(flatten=False)</c>'s
        ///     <c>{"obs", "time"}</c>): each key stacked into its own array.
        /// </summary>
        /// <param name="padding">The padding mode.</param>
        [TestMethod]
        [DataRow("reset")]
        [DataRow("zero")]
        public void FrameStackObservation_Dict_MatchesGymnasium(string padding) {
            using var pair = new GymPair(PyGymEnv.Wrap($"FrameStackObservation(TimeAwareObservation(make('CartPole-v1'), flatten=False), 3, padding_type='{padding}')", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject timed = Python.gymnasium.wrappers.TimeAwareObservation(made, flatten: false);
                return Python.gymnasium.wrappers.FrameStackObservation(timed, 3, padding);
            }), new GymNetEnv(new FrameStackObservation<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), flatten: false), 3, padding)));
            WrapperParity.RunEpisodes(pair, 120, TestContext);
            AssertFrameStackState(pair);
        }

        /// <summary>
        ///     <c>FrameStackObservation</c>'s refusals: a stack of 0, a misspelt mode, and an array outside the space (its
        ///     <c>repr</c> and the space's <c>str</c> in the message).
        /// </summary>
        [TestMethod]
        public void FrameStackObservation_Refusals_MatchGymnasium() {
            AssertRefusal("FrameStackObservation(make('CartPole-v1'), 0)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.FrameStackObservation(made, 0);
            }, () => new FrameStackObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 0));
            AssertRefusal("FrameStackObservation(make('CartPole-v1'), 3, padding_type='unknown')", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.FrameStackObservation(made, 3, "unknown");
            }, () => new FrameStackObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 3, "unknown"));
            using NDArray outside = np.array(new long[] { 1, 2, 3, 4, 5 });
            AssertRefusal("FrameStackObservation(make('CartPole-v1'), 3, padding_type=np.array([1, 2, 3, 4, 5]))", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject given = Python.evalStmts("import numpy as np\nresult = np.array([1, 2, 3, 4, 5])\n", "result");
                return Python.gymnasium.wrappers.FrameStackObservation(made, 3, given);
            }, () => new FrameStackObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 3, outside));
        }

        /// <summary>
        ///     <c>MaxAndSkipObservation(make(id), skip)</c> through 150 sampled steps: CartPole's episodes ending inside a skip
        ///     (the stale buffer), FrozenLake's ints in an int64 buffer (a NumPy scalar maximum), Pendulum's and MountainCar's
        ///     float32 states, and rewards summed as Python floats.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="skip">The skip.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 4L)]
        [DataRow("CartPole-v1", 2L)]
        [DataRow("FrozenLake-v1", 3L)]
        [DataRow("Pendulum-v1", 5L)]
        [DataRow("MountainCar-v0", 3L)]
        public void MaxAndSkipObservation_Episodes_MatchGymnasium(string id, long skip) {
            using var pair = new GymPair(PyGymEnv.Wrap($"MaxAndSkipObservation(make('{id}'), {skip})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.MaxAndSkipObservation(made, skip);
            }), new GymNetEnv(new MaxAndSkipObservation<NDArray, NDArray>(WrapperParity.Make(id), skip)));
            WrapperParity.RunEpisodes(pair, 150, TestContext);
            WrapperParity.AssertAttribute(pair, "_skip", "result = env._skip", env => Gym.Utils.PyAttributes.GetAttr(env, "_skip"));
        }

        /// <summary><c>MaxAndSkipObservation(AddRenderObservation(make("CartPole-v1", render_mode="rgb_array")), 2)</c>: the maximum of two uint8 frames.</summary>
        [TestMethod]
        public void MaxAndSkipObservation_Frames_MatchGymnasium() {
            using var pair = new GymPair(PyGymEnv.Wrap("MaxAndSkipObservation(AddRenderObservation(make('CartPole-v1', render_mode='rgb_array')), 2)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                using PyObject frames = Python.gymnasium.wrappers.AddRenderObservation(made);
                return Python.gymnasium.wrappers.MaxAndSkipObservation(frames, 2);
            }), new GymNetEnv(new MaxAndSkipObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array")), 2)));
            WrapperParity.RunEpisodes(pair, 30, TestContext);
        }

        /// <summary><c>MaxAndSkipObservation</c>'s refusals: skips of 0 and 1, and a <c>Dict</c> space, which has no shape.</summary>
        [TestMethod]
        public void MaxAndSkipObservation_Refusals_MatchGymnasium() {
            foreach (long skip in new[] { 0L, 1L }) {
                AssertRefusal($"MaxAndSkipObservation(make('CartPole-v1'), {skip})", () => {
                    using PyObject made = Python.gymnasium.make("CartPole-v1");
                    return Python.gymnasium.wrappers.MaxAndSkipObservation(made, skip);
                }, () => new MaxAndSkipObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), skip));
            }

            AssertRefusal("MaxAndSkipObservation(TimeAwareObservation(make('CartPole-v1'), flatten=False))", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject timed = Python.gymnasium.wrappers.TimeAwareObservation(made, flatten: false);
                return Python.gymnasium.wrappers.MaxAndSkipObservation(timed);
            }, () => new MaxAndSkipObservation<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), flatten: false)));
        }

        /// <summary>Compares a frame stack's space, mode, padding and deque after its run.</summary>
        /// <param name="pair">The pair, after its run.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private static void AssertFrameStackState(GymPair pair) {
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
            WrapperParity.AssertAttribute(pair, "padding_type", "result = env.padding_type", env => Gym.Utils.PyAttributes.GetAttr(env, "padding_type"));
            WrapperParity.AssertAttribute(pair, "padding_value", "result = env.padding_value", env => Gym.Utils.PyAttributes.GetAttr(env, "padding_value"));
            WrapperParity.AssertAttribute(pair, "obs_queue", "result = tuple(env.obs_queue)",
                env => ((System.Collections.IEnumerable)Gym.Utils.PyAttributes.GetAttr(env, "obs_queue")!).Cast<object?>().ToArray());
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
