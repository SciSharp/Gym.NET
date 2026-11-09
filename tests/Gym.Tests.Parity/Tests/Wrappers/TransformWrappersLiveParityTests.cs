using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Tests.Parity.Tests.Spaces;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Wrappers {
    /// <summary>
    ///     Gymnasium's stateless observation, action and reward wrappers (<c>transform_observation.py</c>,
    ///     <c>transform_action.py</c>, <c>transform_reward.py</c>) and the helpers <c>create_zero_array</c> and
    ///     <c>rescale_box</c>, against Gymnasium's, live: whole episodes through each wrapper over the built-in envs whose
    ///     spaces it takes, the attributes it computes, and every refusal with its message.
    /// </summary>
    /// <remarks>
    ///     The arithmetic is NumPy's, so the cells pick envs that exercise its dtype rules: float32 boxes against weak
    ///     Python bounds and float64 ones (<c>rescale_box</c>'s float64 difference), uint8 images through float64 weights
    ///     (grayscale), float32 <c>linspace</c> edges (the discretizers), and signed zeros through <c>np.clip</c>
    ///     (<c>ClipReward</c> over Sutton and Barto's <c>-0.0</c> rewards). Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py">gymnasium/wrappers/transform_observation.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_action.py">transform_action.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_reward.py">transform_reward.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class TransformWrappersLiveParityTests : ParityTestBase {
        /// <summary><c>TransformObservation(make(id), lambda obs: obs * 2 + 1, space)</c>, with and without a space of its own.</summary>
        /// <param name="withSpace">Whether the wrapper gets <c>Box(-20, 20, (4,), float32)</c> instead of keeping the env's.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void TransformObservation_Episodes_MatchGymnasium(bool withSpace) {
            using var pair = new GymPair(PyGymEnv.Wrap($"TransformObservation(make('CartPole-v1'), lambda obs: obs * 2 + 1, {(withSpace ? "Box(-20, 20, (4,))" : "None")})", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject func = WrapperParity.Lambda("lambda obs: obs * 2 + 1");
                using PyObject space = withSpace ? Python.evalStmts("from gymnasium.spaces import Box\nresult = Box(-20, 20, (4,))\n", "result") : Oracle.PyConvert.ToPython(null);
                return Python.gymnasium.wrappers.TransformObservation(made, func, space);
            }), new GymNetEnv(new TransformObservation<NDArray, NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), obs => {
                using NDArray doubled = obs * 2;
                return doubled + 1;
            }, withSpace ? new Box(-20, 20, new Shape(4)) : null)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary>
        ///     <c>FilterObservation</c> over a <c>Dict</c> observation (<c>TimeAwareObservation(flatten=False)</c>'s
        ///     <c>{"obs", "time"}</c>): the kept keys in the filter's order, the space in its sorted order.
        /// </summary>
        /// <param name="keys">The keys to keep, comma-separated.</param>
        [TestMethod]
        [DataRow("time")]
        [DataRow("obs")]
        [DataRow("time,obs")]
        public void FilterObservation_Dict_MatchesGymnasium(string keys) {
            string[] kept = keys.Split(',');
            using var pair = new GymPair(PyGymEnv.Wrap($"FilterObservation(TimeAwareObservation(make('CartPole-v1'), flatten=False), {keys})", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject timed = Python.gymnasium.wrappers.TimeAwareObservation(made, flatten: false);
                return Python.gymnasium.wrappers.FilterObservation(timed, kept.Cast<object?>().ToList());
            }), new GymNetEnv(new FilterObservation<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), flatten: false), kept.Cast<object?>().ToList())));
            WrapperParity.RunEpisodes(pair, 120, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary><c>FilterObservation</c> over Blackjack's <c>Tuple</c> observation, by positive and negative indices.</summary>
        /// <param name="indices">The indices to keep, comma-separated.</param>
        [TestMethod]
        [DataRow("0,2")]
        [DataRow("-1")]
        [DataRow("2,0")]
        public void FilterObservation_Tuple_MatchesGymnasium(string indices) {
            object?[] kept = indices.Split(',').Select(i => (object?)long.Parse(i)).ToArray();
            using var pair = new GymPair(PyGymEnv.Wrap($"FilterObservation(make('Blackjack-v1'), ({indices},))", () => {
                using PyObject made = Python.gymnasium.make("Blackjack-v1");
                using PyObject keys = Python.evalStmts($"result = ({indices},)\n", "result");
                return Python.gymnasium.wrappers.FilterObservation(made, keys);
            }), new GymNetEnv(new FilterObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make("Blackjack-v1"), new TupleKeys(kept))));
            WrapperParity.RunEpisodes(pair, 120, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary>
        ///     <c>FilterObservation</c>'s refusals: a string of characters that are no keys, a bool and an int (no sequences),
        ///     an empty filter, string keys of a tuple, duplicate and out-of-range indices, and a <c>Box</c> env.
        /// </summary>
        [TestMethod]
        public void FilterObservation_Refusals_MatchGymnasium() {
            AssertRefusal("FilterObservation(TimeAware, 'key')", () => {
                using PyObject env = TimeAwarePython();
                using var key = new PyString("key");
                return Python.gymnasium.wrappers.FilterObservation(env, key);
            }, () => new FilterObservation<object?, NDArray>(TimeAwareDotNet(), "key"));
            AssertRefusal("FilterObservation(TimeAware, False)", () => {
                using PyObject env = TimeAwarePython();
                using PyObject flag = false.ToPython();
                return Python.gymnasium.wrappers.FilterObservation(env, flag);
            }, () => new FilterObservation<object?, NDArray>(TimeAwareDotNet(), false));
            AssertRefusal("FilterObservation(TimeAware, 1)", () => {
                using PyObject env = TimeAwarePython();
                using var one = new PyInt(1);
                return Python.gymnasium.wrappers.FilterObservation(env, one);
            }, () => new FilterObservation<object?, NDArray>(TimeAwareDotNet(), 1L));
            AssertRefusal("FilterObservation(TimeAware, ())", () => {
                using PyObject env = TimeAwarePython();
                using var empty = new PyTuple();
                return Python.gymnasium.wrappers.FilterObservation(env, empty);
            }, () => new FilterObservation<object?, NDArray>(TimeAwareDotNet(), new TupleKeys(System.Array.Empty<object?>())));
            AssertRefusal("FilterObservation(TimeAware, ['obs', 'missing'])", () => {
                using PyObject env = TimeAwarePython();
                return Python.gymnasium.wrappers.FilterObservation(env, new List<object?> { "obs", "missing" });
            }, () => new FilterObservation<object?, NDArray>(TimeAwareDotNet(), new List<object?> { "obs", "missing" }));
            AssertRefusal("FilterObservation(Blackjack, ['a'])", () => {
                using PyObject env = Python.gymnasium.make("Blackjack-v1");
                return Python.gymnasium.wrappers.FilterObservation(env, new List<object?> { "a" });
            }, () => new FilterObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make("Blackjack-v1"), new List<object?> { "a" }));
            AssertRefusal("FilterObservation(Blackjack, [0, 0])", () => {
                using PyObject env = Python.gymnasium.make("Blackjack-v1");
                return Python.gymnasium.wrappers.FilterObservation(env, new List<object?> { 0L, 0L });
            }, () => new FilterObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make("Blackjack-v1"), new List<object?> { 0L, 0L }));
            AssertRefusal("FilterObservation(Blackjack, [1, 3, 5])", () => {
                using PyObject env = Python.gymnasium.make("Blackjack-v1");
                return Python.gymnasium.wrappers.FilterObservation(env, new List<object?> { 1L, 3L, 5L });
            }, () => new FilterObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make("Blackjack-v1"), new List<object?> { 1L, 3L, 5L }));
            AssertRefusal("FilterObservation(CartPole, ['obs'])", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.FilterObservation(env, new List<object?> { "obs" });
            }, () => new FilterObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), new List<object?> { "obs" }));
        }

        /// <summary>
        ///     <c>GrayscaleObservation(AddRenderObservation(make("CartPole-v1", render_mode="rgb_array")))</c>: every frame's
        ///     grayscale, uint8 truncation of the float64 weighted sum, with and without the channel axis.
        /// </summary>
        /// <param name="keepDim">Whether the channel axis is kept.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void GrayscaleObservation_Episodes_MatchGymnasium(bool keepDim) {
            using var pair = new GymPair(PyGymEnv.Wrap($"GrayscaleObservation(AddRenderObservation(make('CartPole-v1', render_mode='rgb_array')), keep_dim={keepDim})", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                using PyObject frames = Python.gymnasium.wrappers.AddRenderObservation(made);
                return Python.gymnasium.wrappers.GrayscaleObservation(frames, keepDim);
            }), new GymNetEnv(new GrayscaleObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array")), keepDim)));
            WrapperParity.RunEpisodes(pair, 40, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary><c>GrayscaleObservation</c>'s refusals: a <c>Discrete</c> space, a 1-D box, and a float image box.</summary>
        [TestMethod]
        public void GrayscaleObservation_Refusals_MatchGymnasium() {
            AssertRefusal("GrayscaleObservation(FrozenLake)", () => {
                using PyObject env = Python.gymnasium.make("FrozenLake-v1");
                return Python.gymnasium.wrappers.GrayscaleObservation(env);
            }, () => new GrayscaleObservation<NDArray, NDArray>(WrapperParity.Make("FrozenLake-v1")));
            AssertRefusal("GrayscaleObservation(CartPole)", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.GrayscaleObservation(env);
            }, () => new GrayscaleObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1")));
            AssertRefusal("GrayscaleObservation(DtypeObservation(frames, float32))", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                using PyObject frames = Python.gymnasium.wrappers.AddRenderObservation(made);
                using PyObject dtype = Python.numpy.dtype("float32");
                using PyObject floats = Python.gymnasium.wrappers.DtypeObservation(frames, dtype);
                return Python.gymnasium.wrappers.GrayscaleObservation(floats);
            }, () => new GrayscaleObservation<NDArray, NDArray>(new DtypeObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array")), "float32")));
        }

        /// <summary><c>ReshapeObservation</c> of CartPole's <c>(4,)</c> into <c>(2, 2)</c>, and of a rendered frame into <c>(600, 400, 3)</c>.</summary>
        /// <param name="frames">Whether the observations are rendered frames instead of CartPole's state.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void ReshapeObservation_Episodes_MatchGymnasium(bool frames) {
            long[] shape = frames ? new[] { 600L, 400L, 3L } : new[] { 2L, 2L };
            using var pair = new GymPair(PyGymEnv.Wrap($"ReshapeObservation(..., {string.Join(", ", shape)})", () => {
                if (!frames) {
                    using PyObject made = Python.gymnasium.make("CartPole-v1");
                    return Python.gymnasium.wrappers.ReshapeObservation(made, shape);
                }

                using PyObject rendered = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                using PyObject framesEnv = Python.gymnasium.wrappers.AddRenderObservation(rendered);
                return Python.gymnasium.wrappers.ReshapeObservation(framesEnv, shape);
            }), new GymNetEnv(frames
                ? new ReshapeObservation<object?, NDArray>(new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array")), shape)
                : new ReshapeObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), shape)));
            WrapperParity.RunEpisodes(pair, frames ? 20 : 150, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary><c>ReshapeObservation</c>'s refusals: a product that differs, and a <c>Discrete</c> space.</summary>
        [TestMethod]
        public void ReshapeObservation_Refusals_MatchGymnasium() {
            AssertRefusal("ReshapeObservation(CartPole, (3, 2))", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.ReshapeObservation(env, new[] { 3L, 2L });
            }, () => new ReshapeObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), new[] { 3L, 2L }));
            AssertRefusal("ReshapeObservation(CartPole, (-1, 2))", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.ReshapeObservation(env, new[] { -1L, 2L });
            }, () => new ReshapeObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), new[] { -1L, 2L }));
            AssertRefusal("ReshapeObservation(FrozenLake, (1,))", () => {
                using PyObject env = Python.gymnasium.make("FrozenLake-v1");
                return Python.gymnasium.wrappers.ReshapeObservation(env, new[] { 1L });
            }, () => new ReshapeObservation<NDArray, NDArray>(WrapperParity.Make("FrozenLake-v1"), new[] { 1L }));
        }

        /// <summary>
        ///     <c>RescaleObservation</c> over Pendulum (float32 array bounds), MountainCar (Python int bounds, so int64 arrays
        ///     promote the arithmetic to float64) and CartPole (infinite components kept infinite).
        /// </summary>
        /// <param name="id">The env.</param>
        [TestMethod]
        [DataRow("Pendulum-v1")]
        [DataRow("MountainCar-v0")]
        [DataRow("CartPole-v1")]
        public void RescaleObservation_Episodes_MatchGymnasium(string id) {
            (object min, object max) = RescaleBounds(id);
            using var pair = new GymPair(PyGymEnv.Wrap($"RescaleObservation(make('{id}'), ...)", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.RescaleObservation(made, min, max);
            }), new GymNetEnv(new RescaleObservation<NDArray, NDArray>(WrapperParity.Make(id), min, max)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary>
        ///     <c>RescaleObservation</c>'s and <c>rescale_box</c>'s refusals: a bound of another shape, a bool, an infinite
        ///     component given a finite bound, reversed bounds, and a <c>Discrete</c> space.
        /// </summary>
        [TestMethod]
        public void RescaleObservation_Refusals_MatchGymnasium() {
            AssertRefusal("RescaleObservation(Pendulum, zeros(2), 1)", () => {
                using PyObject env = Python.gymnasium.make("Pendulum-v1");
                return Python.gymnasium.wrappers.RescaleObservation(env, np.zeros(new Shape(2), np.float32), 1L);
            }, () => new RescaleObservation<NDArray, NDArray>(WrapperParity.Make("Pendulum-v1"), np.zeros(new Shape(2), np.float32), 1L));
            AssertRefusal("RescaleObservation(Pendulum, True, 1)", () => {
                using PyObject env = Python.gymnasium.make("Pendulum-v1");
                return Python.gymnasium.wrappers.RescaleObservation(env, true, 1L);
            }, () => new RescaleObservation<NDArray, NDArray>(WrapperParity.Make("Pendulum-v1"), true, 1L));
            AssertRefusal("RescaleObservation(CartPole, -1, 1)", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.RescaleObservation(env, -1L, 1L);
            }, () => new RescaleObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), -1L, 1L));
            AssertRefusal("RescaleObservation(Pendulum, 1, -1)", () => {
                using PyObject env = Python.gymnasium.make("Pendulum-v1");
                return Python.gymnasium.wrappers.RescaleObservation(env, 1L, -1L);
            }, () => new RescaleObservation<NDArray, NDArray>(WrapperParity.Make("Pendulum-v1"), 1L, -1L));
            AssertRefusal("RescaleObservation(FrozenLake, 0, 1)", () => {
                using PyObject env = Python.gymnasium.make("FrozenLake-v1");
                return Python.gymnasium.wrappers.RescaleObservation(env, 0L, 1L);
            }, () => new RescaleObservation<NDArray, NDArray>(WrapperParity.Make("FrozenLake-v1"), 0L, 1L));
        }

        /// <summary>
        ///     <c>DtypeObservation</c>: CartPole's float32 state as float64 and as int16 (truncated), FrozenLake's
        ///     <c>Discrete</c> cell as float32 and int32 (a 0-d array, NumPy's scalar), and the spaces that follow.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="dtype">The dtype name.</param>
        [TestMethod]
        [DataRow("CartPole-v1", "float64")]
        [DataRow("CartPole-v1", "int16")]
        [DataRow("FrozenLake-v1", "float32")]
        [DataRow("FrozenLake-v1", "int32")]
        [DataRow("Taxi-v4", "int64")]
        public void DtypeObservation_Episodes_MatchGymnasium(string id, string dtype) {
            using var pair = new GymPair(PyGymEnv.Wrap($"DtypeObservation(make('{id}'), np.dtype('{dtype}'))", () => {
                using PyObject made = Python.gymnasium.make(id);
                using PyObject type = Python.numpy.dtype(dtype);
                return Python.gymnasium.wrappers.DtypeObservation(made, type);
            }), new GymNetEnv(new DtypeObservation<NDArray, NDArray>(WrapperParity.Make(id), dtype)));
            WrapperParity.RunEpisodes(pair, 150, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary><c>DtypeObservation</c>'s refusal of Blackjack's <c>Tuple</c> space.</summary>
        [TestMethod]
        public void DtypeObservation_Refusals_MatchGymnasium() {
            AssertRefusal("DtypeObservation(Blackjack, float32)", () => {
                using PyObject env = Python.gymnasium.make("Blackjack-v1");
                using PyObject type = Python.numpy.dtype("float32");
                return Python.gymnasium.wrappers.DtypeObservation(env, type);
            }, () => new DtypeObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make("Blackjack-v1"), "float32"));
        }

        /// <summary>
        ///     <c>AddRenderObservation</c> over CartPole: the frame alone, the frame beside the state
        ///     (<c>{"pixels", "state"}</c>), and the frame added to a <c>Dict</c> observation.
        /// </summary>
        /// <param name="renderOnly">Whether the frame replaces the observation.</param>
        /// <param name="dictEnv">Whether the env's observation is <c>TimeAwareObservation(flatten=False)</c>'s dict.</param>
        [TestMethod]
        [DataRow(true, false)]
        [DataRow(false, false)]
        [DataRow(false, true)]
        public void AddRenderObservation_Episodes_MatchGymnasium(bool renderOnly, bool dictEnv) {
            using var pair = new GymPair(PyGymEnv.Wrap($"AddRenderObservation(..., render_only={renderOnly}), dict env {dictEnv}", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                if (!dictEnv)
                    return Python.gymnasium.wrappers.AddRenderObservation(made, renderOnly);
                using PyObject timed = Python.gymnasium.wrappers.TimeAwareObservation(made, flatten: false);
                return Python.gymnasium.wrappers.AddRenderObservation(timed, renderOnly);
            }), new GymNetEnv(dictEnv
                ? new AddRenderObservation<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), flatten: false), renderOnly)
                : new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), renderOnly)));
            WrapperParity.RunEpisodes(pair, 20, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
        }

        /// <summary>
        ///     <c>AddRenderObservation</c>'s refusals: no render mode, an <c>ansi</c> render mode (text, no array), and a dict
        ///     env that already has the render key.
        /// </summary>
        [TestMethod]
        public void AddRenderObservation_Refusals_MatchGymnasium() {
            AssertRefusal("AddRenderObservation(make('CartPole-v1'))", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.AddRenderObservation(env);
            }, () => new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1")));
            AssertRefusal("AddRenderObservation(make('FrozenLake-v1', render_mode='ansi'))", () => {
                using PyObject env = Python.gymnasium.make("FrozenLake-v1", render_mode: "ansi");
                return Python.gymnasium.wrappers.AddRenderObservation(env);
            }, () => new AddRenderObservation<NDArray, NDArray>(WrapperParity.Make("FrozenLake-v1", "ansi")));
            AssertRefusal("AddRenderObservation(TimeAware dict, render_only=False, render_key='time')", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1", render_mode: "rgb_array");
                using PyObject timed = Python.gymnasium.wrappers.TimeAwareObservation(made, flatten: false);
                return Python.gymnasium.wrappers.AddRenderObservation(timed, false, "time");
            }, () => new AddRenderObservation<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1", "rgb_array"), flatten: false), false, "time"));
        }

        /// <summary>
        ///     <c>DiscretizeObservation</c> over MountainCar (10 bins each), Pendulum (3, 4 and 5 bins, as <c>Discrete</c> and
        ///     <c>MultiDiscrete</c>) and Acrobot (3 bins over 6 elements): the observations, the float32 bin edges and
        ///     <c>revert_observation</c> of the last observation.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="bins">The bins, comma-separated (one value is used for every element).</param>
        /// <param name="multidiscrete">Whether observations are MultiDiscrete.</param>
        [TestMethod]
        [DataRow("MountainCar-v0", "10", false)]
        [DataRow("Pendulum-v1", "3,4,5", false)]
        [DataRow("Pendulum-v1", "3,4,5", true)]
        [DataRow("Acrobot-v1", "3", false)]
        public void DiscretizeObservation_Episodes_MatchGymnasium(string id, string bins, bool multidiscrete) {
            object binsArgument = BinsArgument(bins);
            using var pair = new GymPair(PyGymEnv.Wrap($"DiscretizeObservation(make('{id}'), {bins}, multidiscrete={multidiscrete})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.DiscretizeObservation(made, binsArgument, multidiscrete);
            }), new GymNetEnv(new DiscretizeObservation<NDArray, NDArray>(WrapperParity.Make(id), binsArgument, multidiscrete)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "observation_space", "result = repr(env.observation_space)", env => env.ObservationSpace.ToString());
            WrapperParity.AssertAttribute(pair, "bin_edges", "result = tuple(env.bin_edges)", env => ((DiscretizeObservation<NDArray, NDArray>)env).BinEdges.Cast<object?>().ToArray());
            // A fixed discretized observation on both sides (each side's space would sample its own stream): a third of the
            // Discrete range, or the middle bin of each MultiDiscrete element.
            WrapperParity.AssertAttribute(pair, "revert_observation(fixed observation)",
                "import numpy as np\no = env.observation_space.nvec // 2 if env.multidiscrete else int(env.observation_space.n) // 3\nresult = env.revert_observation(o)\n",
                env => {
                    var wrapper = (DiscretizeObservation<NDArray, NDArray>)env;
                    object fixedObservation = multidiscrete
                        ? np.array(wrapper.Bins.Select(b => b / 2).ToArray())
                        : (object)(wrapper.Bins.Aggregate(1L, (product, b) => product * b) / 3);
                    try {
                        (NDArray low, NDArray high) = wrapper.RevertObservation(fixedObservation);
                        return new object?[] { low, high };
                    } finally {
                        (fixedObservation as IDisposable)?.Dispose();
                    }
                });
        }

        /// <summary><c>DiscretizeObservation</c>'s refusals: CartPole's infinite bounds, mismatched bins, and a <c>Discrete</c> space.</summary>
        [TestMethod]
        public void DiscretizeObservation_Refusals_MatchGymnasium() {
            AssertRefusal("DiscretizeObservation(CartPole, 3)", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.DiscretizeObservation(env, 3L);
            }, () => new DiscretizeObservation<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 3L));
            AssertRefusal("DiscretizeObservation(MountainCar, (3, 4, 5))", () => {
                using PyObject env = Python.gymnasium.make("MountainCar-v0");
                return Python.gymnasium.wrappers.DiscretizeObservation(env, (3L, 4L, 5L));
            }, () => new DiscretizeObservation<NDArray, NDArray>(WrapperParity.Make("MountainCar-v0"), (3L, 4L, 5L)));
            AssertRefusal("DiscretizeObservation(FrozenLake, 3)", () => {
                using PyObject env = Python.gymnasium.make("FrozenLake-v1");
                return Python.gymnasium.wrappers.DiscretizeObservation(env, 3L);
            }, () => new DiscretizeObservation<NDArray, NDArray>(WrapperParity.Make("FrozenLake-v1"), 3L));
        }

        /// <summary><c>TransformAction(make("MountainCarContinuous-v0"), lambda a: a * 0.5, None)</c>: float32 actions halved (a weak Python float).</summary>
        [TestMethod]
        public void TransformAction_Episodes_MatchGymnasium() {
            using var pair = new GymPair(PyGymEnv.Wrap("TransformAction(make('MountainCarContinuous-v0'), lambda a: a * 0.5, None)", () => {
                using PyObject made = Python.gymnasium.make("MountainCarContinuous-v0");
                using PyObject func = WrapperParity.Lambda("lambda a: a * 0.5");
                using PyObject none = Oracle.PyConvert.ToPython(null);
                return Python.gymnasium.wrappers.TransformAction(made, func, none);
            }), new GymNetEnv(new TransformAction<NDArray, NDArray, NDArray>(WrapperParity.Make("MountainCarContinuous-v0"), action => action * 0.5, null)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
        }

        /// <summary><c>ClipAction</c> over Pendulum and MountainCarContinuous: unbounded samples clipped to the env's bounds.</summary>
        /// <param name="id">The env.</param>
        [TestMethod]
        [DataRow("Pendulum-v1")]
        [DataRow("MountainCarContinuous-v0")]
        public void ClipAction_Episodes_MatchGymnasium(string id) {
            using var pair = new GymPair(PyGymEnv.Wrap($"ClipAction(make('{id}'))", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.ClipAction(made);
            }), new GymNetEnv(new ClipAction<NDArray, NDArray>(WrapperParity.Make(id))));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "action_space", "result = repr(env.action_space)", env => env.ActionSpace.ToString());
        }

        /// <summary>
        ///     <c>RescaleAction</c> over Pendulum (Python float bounds, float64 arrays) and MountainCarContinuous (float32
        ///     array bounds): sampled actions mapped back onto the env's bounds.
        /// </summary>
        /// <param name="id">The env.</param>
        [TestMethod]
        [DataRow("Pendulum-v1")]
        [DataRow("MountainCarContinuous-v0")]
        public void RescaleAction_Episodes_MatchGymnasium(string id) {
            // Arms cast to object: an NDArray arm would otherwise make the double an NDArray (NumSharp's implicit conversion).
            object min = id == "Pendulum-v1" ? (object)(-1.0) : np.array(new[] { -10f });
            object max = id == "Pendulum-v1" ? (object)1.0 : np.array(new[] { 30f });
            using var pair = new GymPair(PyGymEnv.Wrap($"RescaleAction(make('{id}'), ...)", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.RescaleAction(made, min, max);
            }), new GymNetEnv(new RescaleAction<NDArray, NDArray>(WrapperParity.Make(id), min, max)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "action_space", "result = repr(env.action_space)", env => env.ActionSpace.ToString());
        }

        /// <summary><c>RescaleAction</c>'s refusals: equal bounds (no inverse), and a <c>Discrete</c> action space.</summary>
        [TestMethod]
        public void RescaleAction_Refusals_MatchGymnasium() {
            AssertRefusal("RescaleAction(Pendulum, np.float32(0), np.float32(0))", () => {
                using PyObject env = Python.gymnasium.make("Pendulum-v1");
                using PyObject zero = Python.numpy.float32(0f);
                using PyObject alsoZero = Python.numpy.float32(0f);
                return Python.gymnasium.wrappers.RescaleAction(env, zero, alsoZero);
            }, () => new RescaleAction<NDArray, NDArray>(WrapperParity.Make("Pendulum-v1"), 0f, 0f));
            AssertRefusal("RescaleAction(CartPole, -1, 1)", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.RescaleAction(env, -1L, 1L);
            }, () => new RescaleAction<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), -1L, 1L));
        }

        /// <summary>
        ///     <c>DiscretizeAction</c> over Pendulum (5 bins, <c>Discrete(5)</c>) and MountainCarContinuous (7 bins,
        ///     <c>MultiDiscrete</c>): the bin centers and <c>revert_action</c> of a center.
        /// </summary>
        /// <param name="id">The env.</param>
        /// <param name="bins">The bins.</param>
        /// <param name="multidiscrete">Whether actions are MultiDiscrete.</param>
        [TestMethod]
        [DataRow("Pendulum-v1", 5L, false)]
        [DataRow("MountainCarContinuous-v0", 7L, true)]
        public void DiscretizeAction_Episodes_MatchGymnasium(string id, long bins, bool multidiscrete) {
            using var pair = new GymPair(PyGymEnv.Wrap($"DiscretizeAction(make('{id}'), {bins}, multidiscrete={multidiscrete})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.DiscretizeAction(made, bins, multidiscrete);
            }), new GymNetEnv(new DiscretizeAction<NDArray, NDArray>(WrapperParity.Make(id), bins, multidiscrete)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
            WrapperParity.AssertAttribute(pair, "action_space", "result = repr(env.action_space)", env => env.ActionSpace.ToString());
            WrapperParity.AssertAttribute(pair, "bin_centers", "result = tuple(env.bin_centers)", env => ((DiscretizeAction<NDArray, NDArray>)env).BinCenters.Cast<object?>().ToArray());
            WrapperParity.AssertAttribute(pair, "revert_action(action(1))",
                "import numpy as np\na = env.action(np.array([1]) if env.multidiscrete else 1)\nresult = (a, env.revert_action(a))\n",
                env => {
                    var wrapper = (DiscretizeAction<NDArray, NDArray>)env;
                    using NDArray discrete = multidiscrete ? np.array(new[] { 1L }) : NDArray.Scalar(1L);
                    NDArray continuous = wrapper.Action(discrete);
                    return new object?[] { continuous, wrapper.RevertAction(continuous) };
                });
        }

        /// <summary><c>TransformReward</c> over CartPole (<c>2 * r + 1</c>) and Pendulum (<c>r / 16</c>).</summary>
        /// <param name="id">The env.</param>
        [TestMethod]
        [DataRow("CartPole-v1")]
        [DataRow("Pendulum-v1")]
        public void TransformReward_Episodes_MatchGymnasium(string id) {
            bool cartPole = id == "CartPole-v1";
            using var pair = new GymPair(PyGymEnv.Wrap($"TransformReward(make('{id}'), ...)", () => {
                using PyObject made = Python.gymnasium.make(id);
                using PyObject func = WrapperParity.Lambda(cartPole ? "lambda r: 2 * r + 1" : "lambda r: r / 16");
                return Python.gymnasium.wrappers.TransformReward(made, func);
            }), new GymNetEnv(new TransformReward<NDArray, NDArray>(WrapperParity.Make(id), cartPole ? r => 2 * r + 1 : r => r / 16)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
        }

        /// <summary>
        ///     <c>ClipReward</c> with both bounds and with one, over CartPole, Pendulum and Sutton and Barto's CartPole, whose
        ///     <c>-0.0</c> rewards show <c>np.clip</c>'s signed-zero rules (a two-sided clip keeps <c>-0.0</c>, a lower bound
        ///     alone gives <c>+0.0</c>).
        /// </summary>
        /// <param name="id">The env, <c>CartPole-SB</c> for Sutton and Barto's rewards.</param>
        /// <param name="min">The lower bound, or NaN for none.</param>
        /// <param name="max">The upper bound, or NaN for none.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 0.0, 0.5)]
        [DataRow("Pendulum-v1", -1.0, double.NaN)]
        [DataRow("Pendulum-v1", double.NaN, -2.0)]
        [DataRow("CartPole-SB", 0.0, 1.0)]
        [DataRow("CartPole-SB", 0.0, double.NaN)]
        [DataRow("CartPole-SB", double.NaN, 0.0)]
        public void ClipReward_Episodes_MatchGymnasium(string id, double min, double max) {
            bool suttonBarto = id == "CartPole-SB";
            string envId = suttonBarto ? "CartPole-v1" : id;
            object? low = double.IsNaN(min) ? null : min;
            object? high = double.IsNaN(max) ? null : max;
            using var pair = new GymPair(PyGymEnv.Wrap($"ClipReward(make('{envId}'{(suttonBarto ? ", sutton_barto_reward=True" : "")}), {low}, {high})", () => {
                using PyObject made = Python.gymnasium.make(envId, kwargs: suttonBarto ? new Dictionary<string, object?> { ["sutton_barto_reward"] = true } : null);
                return Python.gymnasium.wrappers.ClipReward(made, low, high);
            }), new GymNetEnv(new ClipReward<NDArray, NDArray>(suttonBarto
                ? (Env<NDArray, NDArray>)Gymnasium.Make(envId, new { sutton_barto_reward = true })
                : WrapperParity.Make(envId), low, high)));
            WrapperParity.RunEpisodes(pair, 200, TestContext);
        }

        /// <summary><c>ClipReward</c>'s refusals: no bounds, reversed numbers, and reversed arrays.</summary>
        [TestMethod]
        public void ClipReward_Refusals_MatchGymnasium() {
            AssertRefusal("ClipReward(CartPole, None, None)", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.ClipReward(env, null, null);
            }, () => new ClipReward<NDArray, NDArray>(WrapperParity.Make("CartPole-v1")));
            AssertRefusal("ClipReward(CartPole, 1, -1)", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.ClipReward(env, 1L, -1L);
            }, () => new ClipReward<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), 1L, -1L));
            AssertRefusal("ClipReward(CartPole, [1, 1], [0, 0])", () => {
                using PyObject env = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.ClipReward(env, np.array(new[] { 1L, 1L }), np.array(new[] { 0L, 0L }));
            }, () => new ClipReward<NDArray, NDArray>(WrapperParity.Make("CartPole-v1"), np.array(new[] { 1L, 1L }), np.array(new[] { 0L, 0L })));
        }

        /// <summary>
        ///     <c>create_zero_array</c> of every space of the catalog and of Gymnasium's test catalog: the bound nearest zero
        ///     for boxes that exclude zero, <c>start</c> for discrete spaces, and the composites built of them.
        /// </summary>
        /// <param name="name">The space, by its Python spelling.</param>
        [TestMethod]
        [DynamicData(nameof(AllSpaces))]
        public void CreateZeroArray_MatchesGymnasium(string name) {
            SpaceSpec spec = AllSpaceSpecs[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            ByteContract.AssertSame($"create_zero_array({name})",
                Outcome.CapturePython(() => Python.gymnasium.wrappers.utils.create_zero_array(python.Object)),
                Outcome.Capture(() => WrapperFunctions.CreateZeroArray(dotnet)));
        }

        /// <summary>
        ///     <c>rescale_box</c> over boxes of float32, float64 and integer dtypes with array and scalar bounds: the new box's
        ///     <c>repr</c>, and the forward and backward maps of the box's own bounds and of five seeded samples.
        /// </summary>
        /// <param name="boxCase">The box and bounds, by name.</param>
        [TestMethod]
        [DataRow("float32-arrays")]
        [DataRow("float32-ints")]
        [DataRow("float32-floats")]
        [DataRow("float64-mixed-inf")]
        [DataRow("int64-floats")]
        public void RescaleBox_MatchesGymnasium(string boxCase) {
            (Func<Box> dotnetBox, string pythonBox, object min, object max) = RescaleBoxCase(boxCase);
            using Box box = dotnetBox();
            (Box rescaled, Func<NDArray, NDArray> forward, Func<NDArray, NDArray> backward) = WrapperFunctions.RescaleBox(box, min, max);
            using (rescaled) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject pyBox = Python.evalStmts("import numpy as np\nfrom gymnasium.spaces import Box\nresult = " + pythonBox + "\n", "result");
                    using PyObject pyMin = Oracle.PyConvert.ToPython(min);
                    using PyObject pyMax = Oracle.PyConvert.ToPython(max);
                    return Python.gymnasium.wrappers.utils.rescale_box(pyBox, pyMin, pyMax);
                });
                ByteContract.AssertSame($"{boxCase}: repr(new box)",
                    Outcome.CapturePython(() => Python.evalStmts("result = repr(r[0])\n", "result", ("r", python.Object))),
                    Outcome.Capture(() => rescaled.ToString()));

                // The box's own finite bounds and seeded samples through both maps.
                using NDArray samples = Samples(box);
                for (int i = 0; i < (int)samples.shape[0]; i++) {
                    using NDArray point = samples[i].copy();
                    ByteContract.AssertSame($"{boxCase}: forward(sample {i})",
                        Outcome.CapturePython(() => {
                            using PyObject pyPoint = point.ToNumpyCopy();
                            return Python.evalStmts("result = r[1](x)\n", "result", ("r", python.Object), ("x", pyPoint));
                        }),
                        Outcome.Capture(() => forward(point)));
                    ByteContract.AssertSame($"{boxCase}: backward(sample {i})",
                        Outcome.CapturePython(() => {
                            using PyObject pyPoint = point.ToNumpyCopy();
                            return Python.evalStmts("result = r[2](x)\n", "result", ("r", python.Object), ("x", pyPoint));
                        }),
                        Outcome.Capture(() => backward(point)));
                }
            }
        }

        /// <summary>Every space of both catalogs, by name.</summary>
        public static IEnumerable<object[]> AllSpaces => AllSpaceSpecs.Keys.Select(name => new object[] { name });

        /// <summary>The suite's space catalog and Gymnasium's own test catalog, by Python spelling (the first spelling wins).</summary>
        private static readonly Dictionary<string, SpaceSpec> AllSpaceSpecs = SpaceSpecLiveParityTests.Catalog.Values
            .Concat(GymnasiumTestingSpaces.Fundamental).Concat(GymnasiumTestingSpaces.Composite)
            .GroupBy(spec => spec.PythonText).ToDictionary(group => group.Key, group => group.First());

        /// <summary>A Python tuple of filter keys for Gym.NET (an <see cref="System.Runtime.CompilerServices.ITuple"/> of the items).</summary>
        private sealed class TupleKeys : System.Runtime.CompilerServices.ITuple {
            /// <summary>The items.</summary>
            private readonly object?[] _items;

            /// <summary>Wraps the items.</summary>
            /// <param name="items">The items.</param>
            public TupleKeys(object?[] items) => _items = items;

            /// <inheritdoc />
            public object? this[int index] => _items[index];

            /// <inheritdoc />
            public int Length => _items.Length;

            /// <summary>Python's tuple repr of the items, as messages print the keys.</summary>
            /// <returns><c>(0, 2)</c>.</returns>
            public override string ToString() => Gym.Utils.PyBuiltins.Repr(_items);
        }

        /// <summary>Compares a refusal: Gymnasium's construction and Gym.NET's raise the same exception type and message.</summary>
        /// <param name="label">The comparison's name.</param>
        /// <param name="python">Builds Gymnasium's wrapper (under the GIL).</param>
        /// <param name="dotnet">Builds Gym.NET's wrapper.</param>
        /// <exception cref="ParityException">Thrown when the outcomes differ.</exception>
        private static void AssertRefusal(string label, Func<PyObject> python, Func<object> dotnet) =>
            ByteContract.AssertSame(label, Outcome.CapturePython(python), Outcome.Capture(dotnet));

        /// <summary>Gymnasium's <c>TimeAwareObservation(make("CartPole-v1"), flatten=False)</c>, a <c>Dict</c> env. Must run under the GIL.</summary>
        /// <returns>A new owned reference.</returns>
        private static PyObject TimeAwarePython() {
            using PyObject made = Python.gymnasium.make("CartPole-v1");
            return Python.gymnasium.wrappers.TimeAwareObservation(made, flatten: false);
        }

        /// <summary>Gym.NET's <c>TimeAwareObservation(make("CartPole-v1"), flatten=False)</c>.</summary>
        /// <returns>The env.</returns>
        private static TimeAwareObservation<NDArray, NDArray> TimeAwareDotNet() => new(WrapperParity.Make("CartPole-v1"), flatten: false);

        /// <summary>The bounds of each <c>RescaleObservation</c> cell.</summary>
        /// <param name="id">The env.</param>
        /// <returns>The lower and upper bounds, as both sides take them.</returns>
        private static (object Min, object Max) RescaleBounds(string id) => id switch {
            // Every element cast to object: a long arm beside NDArray arms would otherwise become an NDArray.
            "Pendulum-v1" => ((object)np.array(new[] { -2f, -1f, -10f }), (object)np.array(new[] { 1f, 0f, 1f })),
            "MountainCar-v0" => ((object)(-1L), (object)1L),
            _ => ((object)np.array(new[] { -1f, float.NegativeInfinity, -1f, float.NegativeInfinity }), (object)np.array(new[] { 1f, float.PositiveInfinity, 1f, float.PositiveInfinity })),
        };

        /// <summary>The bins argument of a cell: one number, or a Python tuple of them.</summary>
        /// <param name="bins">The bins, comma-separated.</param>
        /// <returns>A <see cref="long"/> or a value tuple (a Python tuple to <see cref="Oracle.PyConvert.ToPython"/>, a sequence to Gym.NET).</returns>
        private static object BinsArgument(string bins) {
            long[] values = bins.Split(',').Select(long.Parse).ToArray();
            return values.Length == 1 ? values[0] : new TupleKeys(values.Select(v => (object?)v).ToArray());
        }

        /// <summary>The boxes and bounds of the <c>rescale_box</c> cells: Gym.NET's box, the same box in Python, and the bounds.</summary>
        /// <param name="name">The case.</param>
        /// <returns>The case.</returns>
        private static (Func<Box> DotNet, string Python, object Min, object Max) RescaleBoxCase(string name) => name switch {
            // The bounds cast to object, so a number arm stays a number beside the array arms.
            "float32-arrays" => (() => new Box(np.array(new[] { 0f, 1f, -5f }), np.array(new[] { 1f, 3f, 5f })),
                "Box(np.array([0, 1, -5], dtype=np.float32), np.array([1, 3, 5], dtype=np.float32))",
                (object)np.array(new[] { -5f, 0f, -1f }), (object)np.array(new[] { 5f, 1f, 1f })),
            "float32-ints" => (() => new Box(-1.2, 0.6, new Shape(2), np.float32), "Box(-1.2, 0.6, (2,), np.float32)", (object)(-1L), (object)1L),
            "float32-floats" => (() => new Box(-2.0, 2.0, new Shape(3), np.float32), "Box(-2.0, 2.0, (3,), np.float32)", (object)(-0.3), (object)0.7),
            "float64-mixed-inf" => (() => new Box(np.array(new[] { 0.0, double.NegativeInfinity, 1.0 }), np.array(new[] { 1.0, 2.0, double.PositiveInfinity }), dtype: np.float64),
                "Box(np.array([0.0, -np.inf, 1.0]), np.array([1.0, 2.0, np.inf]), dtype=np.float64)",
                (object)np.array(new[] { -1.0, double.NegativeInfinity, 0.0 }), (object)np.array(new[] { 1.0, 5.0, double.PositiveInfinity })),
            _ => (() => new Box(0, 10, new Shape(3), np.int64), "Box(0, 10, (3,), np.int64)", (object)0.0, (object)1.0),
        };

        /// <summary>The box's finite bounds stacked with five samples of seed 0, as float rows of the box's dtype.</summary>
        /// <param name="box">The box.</param>
        /// <returns>A new <c>(7, n)</c> array.</returns>
        private static NDArray Samples(Box box) {
            box.Seed(0);
            var rows = new NDArray[7];
            try {
                using NDArray lowFinite = np.isfinite(box.Low);
                using NDArray lowZeros = np.zeros_like(box.Low);
                using NDArray low = np.where(lowFinite, box.Low, lowZeros);
                using NDArray highFinite = np.isfinite(box.High);
                using NDArray highZeros = np.zeros_like(box.High);
                using NDArray high = np.where(highFinite, box.High, highZeros);
                rows[0] = low.copy();
                rows[1] = high.copy();
                for (int i = 0; i < 5; i++)
                    rows[2 + i] = box.Sample();
                return np.stack(rows);
            } finally {
                foreach (NDArray? row in rows)
                    row?.Dispose();
            }
        }
    }
}
