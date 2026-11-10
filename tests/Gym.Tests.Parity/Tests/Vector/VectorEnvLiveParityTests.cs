using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Vector {
    /// <summary>
    ///     The vector API of P6.1 against Gymnasium's, live: <c>make_vec</c> in sync and vector-entry-point modes,
    ///     <c>SyncVectorEnv</c> in every autoreset mode over envs of every observation kind (CartPole's arrays, the toy_text
    ///     ints with their int and float infos, Taxi's action masks, Blackjack's tuples), seeds and <c>reset_mask</c>, the
    ///     observation modes, <c>get_attr</c>/<c>set_attr</c>/<c>call</c>, spaces, metadata, specs and errors, and
    ///     <c>CartPoleVectorEnv</c> (P4.4) step for step, its generator included.
    /// </summary>
    /// <remarks>
    ///     A lockstep here resets both vector envs with the same seed, seeds both batched action spaces alike, compares every
    ///     sampled batch of actions, then steps each side with its own sample and compares the whole result: observations,
    ///     rewards, flags and the merged infos with their masks. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/sync_vector_env.py">gymnasium/vector/sync_vector_env.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/classic_control/cartpole.py#L356">cartpole.py's CartPoleVectorEnv</see>.
    /// </remarks>
    [TestClass]
    public sealed class VectorEnvLiveParityTests : ParityTestBase {
        /// <summary>Steps a vector env and returns the step with its rewards as float64 values (CartPoleVectorEnv's are float32, Gym.NET's float64, DR-21).</summary>
        private const string StepFloat64RewardsStatements =
            "import numpy as np\n" +
            "o, r, te, tr, i = env.step(actions)\n" +
            "result = (o, np.asarray(r, dtype=np.float64), te, tr, i)\n";

        /// <summary>A <c>SyncVectorEnv</c> of FrozenLake 4x4 and 8x8 with <c>observation_mode</c> given, the spaces differing.</summary>
        private const string DifferentSpacesStatements =
            "import gymnasium as gym\n" +
            "env = gym.vector.SyncVectorEnv([lambda: gym.make('FrozenLake-v1'), lambda: gym.make('FrozenLake8x8-v1')], observation_mode=mode)\n";

        /// <summary>
        ///     <c>make_vec(id, n, "sync", vector_kwargs={"autoreset_mode": mode})</c> over CartPole-v1 in all three modes: 300
        ///     steps of sampled actions across several episode ends (<c>DISABLED</c> resetting the ended sub-envs through
        ///     <c>reset_mask</c>, as its users must).
        /// </summary>
        /// <param name="mode">The autoreset mode's value.</param>
        [TestMethod]
        [DataRow("NextStep")]
        [DataRow("SameStep")]
        [DataRow("Disabled")]
        public void SyncVectorEnv_CartPole_EveryAutoresetMode_MatchesGymnasium(string mode) {
            var vectorKwargs = new Dictionary<string, object?> { ["autoreset_mode"] = mode };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "sync", vectorKwargs));
            using IVectorEnv dotnet = Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = mode });
            Lockstep($"CartPole-v1 x3 {mode}", python, dotnet, seed: 42, steps: 300, resetEnded: mode == "Disabled");
        }

        /// <summary>
        ///     Toy_text sub-envs whose observations are ints and whose infos mix an int <c>prob</c> (resets) with float ones
        ///     (steps), batched into one dtype by the first value seen, as NumPy casts: FrozenLake in <c>NEXT_STEP</c> and
        ///     <c>SAME_STEP</c> (its <c>final_obs</c> object array and <c>final_info</c> dict), Taxi's int8 action masks, and
        ///     Blackjack's tuple observations batched into a tuple of arrays.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="mode">The autoreset mode's value.</param>
        [TestMethod]
        [DataRow("FrozenLake-v1", "NextStep")]
        [DataRow("FrozenLake-v1", "SameStep")]
        [DataRow("Taxi-v4", "NextStep")]
        [DataRow("Blackjack-v1", "NextStep")]
        [DataRow("Blackjack-v1", "SameStep")]
        public void SyncVectorEnv_ToyText_InfosAndTuples_MatchGymnasium(string id, string mode) {
            var vectorKwargs = new Dictionary<string, object?> { ["autoreset_mode"] = mode };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec(id, 4, "sync", vectorKwargs));
            using IVectorEnv dotnet = Gymnasium.MakeVec(id, numEnvs: 4, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = mode });
            Lockstep($"{id} x4 {mode}", python, dotnet, seed: 7, steps: 120);
        }

        /// <summary>
        ///     Seeds as Gymnasium takes them: an integer seeds sub-env <c>i</c> with <c>seed + i</c>, a list seeds each (<c>None</c>
        ///     keeping a generator), and a list of the wrong length raises its <c>ValueError</c>.
        /// </summary>
        [TestMethod]
        public void SyncVectorEnv_Seeds_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "sync"));
            using IVectorEnv dotnet = Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            ByteContract.AssertSame("reset(seed=5)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 5)), Outcome.Capture(() => ResetTuple(dotnet.Reset(5))));
            ByteContract.AssertSame("reset(seed=[1, None, 3])", Outcome.CapturePython(() => Python.gymnasium.vector.VectorEnv.reset(python.Object, new long?[] { 1, null, 3 })),
                Outcome.Capture(() => ResetTuple(dotnet.Reset(new BigInteger?[] { 1, null, 3 }))));
            ByteContract.AssertSame("reset(seed=[1, 2])", Outcome.CapturePython(() => Python.gymnasium.vector.VectorEnv.reset(python.Object, new long?[] { 1, 2 })),
                Outcome.Capture(() => ResetTuple(dotnet.Reset(new BigInteger?[] { 1, 2 }))));
            ByteContract.AssertSame("reset() after the seeds", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object)), Outcome.Capture(() => ResetTuple(dotnet.Reset())));
        }

        /// <summary>
        ///     <c>reset(options={"reset_mask": mask})</c>: only the masked sub-envs reset (the others keep their observations),
        ///     and the mask's refusals: not an array, another shape, another dtype, nothing masked.
        /// </summary>
        [TestMethod]
        public void SyncVectorEnv_ResetMask_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "sync"));
            using IVectorEnv dotnet = Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            ByteContract.AssertSame("reset(seed=3)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)), Outcome.Capture(() => ResetTuple(dotnet.Reset(3))));

            using NDArray mask = np.array(new[] { true, false, true });
            using NDArray wrongShape = np.array(new[,] { { true }, { false }, { true } });
            using NDArray wrongDtype = np.array(new[] { 1L, 0L, 1L });
            using NDArray nothing = np.array(new[] { false, false, false });
            var cases = new (string Text, object? Mask)[] {
                ("[True, False, True]", mask), ("a list", new List<object?> { true, false, true }), ("shape (3, 1)", wrongShape), ("int64", wrongDtype), ("all False", nothing),
            };
            foreach ((string text, object? value) in cases) {
                var options = new Dictionary<string, object?> { ["reset_mask"] = value };
                ByteContract.AssertSame($"reset(seed=9, options={{'reset_mask': {text}}})", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 9, options)),
                    Outcome.Capture(() => ResetTuple(dotnet.Reset(9, new Info { ["reset_mask"] = value }))));
            }
        }

        /// <summary>
        ///     <c>observation_mode="different"</c> over FrozenLake 4x4 and 8x8 (<c>MultiDiscrete([16 64])</c>) through a lockstep,
        ///     and <c>"same"</c> over them refused with Gymnasium's <c>RuntimeError</c>.
        /// </summary>
        [TestMethod]
        public void SyncVectorEnv_ObservationModes_MatchGymnasium() {
            Func<IEnv>[] fns = { () => Gymnasium.Make("FrozenLake-v1"), () => Gymnasium.Make("FrozenLake8x8-v1") };
            Outcome same = Outcome.CapturePython(() => {
                using PyObject mode = PyConvert.ToPython("same");
                return Python.evalStmts(DifferentSpacesStatements, "env", ("mode", mode));
            });
            ByteContract.AssertSame("SyncVectorEnv(FrozenLake 4x4 and 8x8, observation_mode='same')", same, Outcome.Capture(() => new SyncVectorEnv(fns, observationMode: "same")));

            using PyRef python = PyRef.Of(() => {
                using PyObject mode = PyConvert.ToPython("different");
                return Python.evalStmts(DifferentSpacesStatements, "env", ("mode", mode));
            });
            using var dotnet = new SyncVectorEnv(fns, observationMode: "different");
            ByteContract.AssertSame("repr(observation_space)", Outcome.CapturePython(() => {
                using PyObject space = Python.gymnasium.Env.observation_space(python.Object);
                return Python.builtins.repr(space);
            }), Outcome.Returned(ValueFacts.FromText(((IVectorEnv)dotnet).ObservationSpace.ToString()!)));
            Lockstep("FrozenLake 4x4 + 8x8 'different'", python, dotnet, seed: 3, steps: 80);

            Outcome invalid = Outcome.CapturePython(() => {
                using PyObject mode = PyConvert.ToPython("bogus");
                return Python.evalStmts(DifferentSpacesStatements, "env", ("mode", mode));
            });
            ByteContract.AssertSame("observation_mode='bogus'", invalid, Outcome.Capture(() => new SyncVectorEnv(fns, observationMode: "bogus")));
        }

        /// <summary>
        ///     <c>get_attr</c>, <c>set_attr</c> (a list per sub-env, one value for all, a list of the wrong length) and <c>call</c>
        ///     through the sub-envs' wrappers, and the steps after a <c>set_attr</c> changed the physics.
        /// </summary>
        [TestMethod]
        public void SyncVectorEnv_Attributes_MatchGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, "sync"));
            using IVectorEnv dotnet = Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "sync");
            var sync = (SyncVectorEnv)dotnet;
            ByteContract.AssertSame("get_attr('gravity')", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.get_attr(python.Object, "gravity")),
                Outcome.Returned(ValueFacts.Of(sync.GetAttr("gravity").ToArray())));

            using (PyRef values = PyRef.Of(() => PyConvert.ToPython(new List<object?> { 9.8, 5.0, 1.62 })))
                Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.set_attr(python.Object, "gravity", values.Object)).Expect("set_attr('gravity', [9.8, 5.0, 1.62])");
            sync.SetAttr("gravity", new List<object?> { 9.8, 5.0, 1.62 });
            ByteContract.AssertSame("get_attr('gravity') after set_attr", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.get_attr(python.Object, "gravity")),
                Outcome.Returned(ValueFacts.Of(sync.GetAttr("gravity").ToArray())));
            Lockstep("CartPole-v1 x3 after set_attr('gravity')", python, dotnet, seed: 1, steps: 60);

            using (PyRef one = PyRef.Of(() => PyConvert.ToPython(0.03)))
                Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.set_attr(python.Object, "tau", one.Object)).Expect("set_attr('tau', 0.03)");
            sync.SetAttr("tau", 0.03);
            ByteContract.AssertSame("get_attr('tau')", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.get_attr(python.Object, "tau")),
                Outcome.Returned(ValueFacts.Of(sync.GetAttr("tau").ToArray())));

            Outcome wrongLength = Outcome.CapturePython(() => {
                using PyObject values = PyConvert.ToPython(new List<object?> { 1.0, 2.0 });
                return Python.gymnasium.vector.SyncVectorEnv.set_attr(python.Object, "tau", values);
            });
            ByteContract.AssertSame("set_attr('tau', [1.0, 2.0])", wrongLength, Outcome.Capture(() => {
                sync.SetAttr("tau", new List<object?> { 1.0, 2.0 });
                return null;
            }));
            ByteContract.AssertSame("call('np_random_seed')", Outcome.CapturePython(() => Python.gymnasium.vector.SyncVectorEnv.call(python.Object, "np_random_seed")),
                Outcome.Returned(ValueFacts.Of(sync.Call("np_random_seed").ToArray())));
        }

        /// <summary>
        ///     What a vector env shows of itself: <c>str(envs)</c> with and without a spec, both batched spaces and both single
        ///     spaces, <c>num_envs</c>, the metadata with its <c>autoreset_mode</c> member, and the spec <c>make_vec</c> records.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="mode">The vectorization mode's value.</param>
        [TestMethod]
        [DataRow("CartPole-v1", "sync")]
        [DataRow("CartPole-v1", "vector_entry_point")]
        [DataRow("FrozenLake-v1", "sync")]
        [DataRow("Blackjack-v1", "sync")]
        public void VectorEnv_Description_MatchesGymnasium(string id, string mode) {
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec(id, 2, mode));
            using IVectorEnv dotnet = Gymnasium.MakeVec(id, numEnvs: 2, vectorizationMode: mode);
            ByteContract.AssertSame($"str(make_vec('{id}', 2, '{mode}'))", Outcome.CapturePython(() => Python.builtins.str(python.Object)), Outcome.Returned(ValueFacts.FromText(dotnet.ToString()!)));
            var spaces = new (string Name, Func<PyObject, PyObject> Python, Func<IVectorEnv, Gym.Spaces.Space> DotNet)[] {
                ("observation_space", Python.gymnasium.Env.observation_space, env => env.ObservationSpace),
                ("action_space", Python.gymnasium.Env.action_space, env => env.ActionSpace),
                ("single_observation_space", Python.gymnasium.vector.VectorEnv.single_observation_space, env => env.SingleObservationSpace),
                ("single_action_space", Python.gymnasium.vector.VectorEnv.single_action_space, env => env.SingleActionSpace),
            };
            foreach ((string name, Func<PyObject, PyObject> pythonSpace, Func<IVectorEnv, Gym.Spaces.Space> dotnetSpace) in spaces) {
                ByteContract.AssertSame($"repr({name})", Outcome.CapturePython(() => {
                    using PyObject space = pythonSpace(python.Object);
                    return Python.builtins.repr(space);
                }), Outcome.Returned(ValueFacts.FromText(dotnetSpace(dotnet).ToString()!)));
            }

            ByteContract.AssertSame("num_envs", Outcome.CapturePython(() => Python.gymnasium.vector.VectorEnv.num_envs(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.NumEnvs)));
            ByteContract.AssertSame("metadata", Outcome.CapturePython(() => Python.gymnasium.Env.metadata(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.Metadata)));
            ByteContract.AssertSame("spec.kwargs", Outcome.CapturePython(() => {
                using PyObject spec = Python.gymnasium.Env.spec(python.Object);
                return Python.gymnasium.envs.registration.EnvSpec.kwargs(spec);
            }), Outcome.Returned(ValueFacts.Of(dotnet.Spec!.Kwargs)));
        }

        /// <summary>
        ///     <c>make_vec</c>'s errors: an unknown vectorization mode, vector kwargs with the vector entry point, a vector entry
        ///     point asked of an env without one, and an unknown autoreset mode.
        /// </summary>
        [TestMethod]
        public void MakeVec_Errors_MatchGymnasium() {
            ByteContract.AssertSame("make_vec('CartPole-v1', vectorization_mode='bogus')", Outcome.CapturePython(() => Python.gymnasium.make_vec("CartPole-v1", 2, "bogus")),
                Outcome.Capture(() => Gymnasium.MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "bogus")));
            var copyFalse = new Dictionary<string, object?> { ["copy"] = false };
            ByteContract.AssertSame("make_vec('CartPole-v1', 'vector_entry_point', vector_kwargs={'copy': False})",
                Outcome.CapturePython(() => Python.gymnasium.make_vec("CartPole-v1", 2, "vector_entry_point", copyFalse)),
                Outcome.Capture(() => Gymnasium.MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "vector_entry_point", vectorKwargs: new Info { ["copy"] = false })));
            ByteContract.AssertSame("make_vec('Pendulum-v1', 'vector_entry_point')", Outcome.CapturePython(() => Python.gymnasium.make_vec("Pendulum-v1", 2, "vector_entry_point")),
                Outcome.Capture(() => Gymnasium.MakeVec("Pendulum-v1", numEnvs: 2, vectorizationMode: "vector_entry_point")));
            var bogusMode = new Dictionary<string, object?> { ["autoreset_mode"] = "Bogus" };
            ByteContract.AssertSame("make_vec('CartPole-v1', 'sync', vector_kwargs={'autoreset_mode': 'Bogus'})", Outcome.CapturePython(() => Python.gymnasium.make_vec("CartPole-v1", 2, "sync", bogusMode)),
                Outcome.Capture(() => Gymnasium.MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = "Bogus" })));
        }

        /// <summary><c>render()</c> of a sync vector env: every sub-env's <c>rgb_array</c> frame, pixel for pixel.</summary>
        [TestMethod]
        public void SyncVectorEnv_Render_MatchesGymnasium() {
            var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 2, "sync", null, kwargs));
            using IVectorEnv dotnet = Gymnasium.MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "sync", kwargs: new Info { ["render_mode"] = "rgb_array" });
            ByteContract.AssertSame("reset(seed=4)", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 4)), Outcome.Capture(() => ResetTuple(dotnet.Reset(4))));
            ByteContract.AssertSame("render()", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => Frames(dotnet.Render())));
        }

        /// <summary>
        ///     <c>CartPoleVectorEnv</c> through <c>make_vec</c>'s default (the vector entry point): 600 sampled steps of 4 carts,
        ///     the state after every step, through terminations, re-draws and (with a short limit) truncations; Sutton and
        ///     Barto's rewards; CartPole-v0's limit.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="maxEpisodeSteps">The <c>max_episode_steps</c> kwarg, or 0 for the spec's.</param>
        /// <param name="suttonBarto">Whether <c>sutton_barto_reward</c> is set.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 0, false)]
        [DataRow("CartPole-v1", 25, false)]
        [DataRow("CartPole-v1", 0, true)]
        [DataRow("CartPole-v0", 0, false)]
        public void CartPoleVectorEnv_Episodes_MatchGymnasium(string id, int maxEpisodeSteps, bool suttonBarto) {
            var kwargs = new Dictionary<string, object?>();
            var options = new Info();
            if (maxEpisodeSteps > 0) {
                kwargs["max_episode_steps"] = (long)maxEpisodeSteps;
                options["max_episode_steps"] = (long)maxEpisodeSteps;
            }

            if (suttonBarto) {
                kwargs["sutton_barto_reward"] = true;
                options["sutton_barto_reward"] = true;
            }

            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec(id, 4, null, null, kwargs));
            using IVectorEnv dotnet = Gymnasium.MakeVec(id, numEnvs: 4, kwargs: options);
            Assert.IsTrue(dotnet is CartPoleVectorEnv, "make_vec's default for CartPole is its vector entry point.");
            Lockstep($"{id} x4 vector entry point", python, dotnet, seed: 13, steps: 600, float64Rewards: true, stateAttribute: "state");
        }

        /// <summary>
        ///     <c>CartPoleVectorEnv</c>'s reset bounds, its refusals (an action outside the space, a step before reset) and its
        ///     frames, a list of one frame per cart.
        /// </summary>
        [TestMethod]
        public void CartPoleVectorEnv_OptionsErrorsAndFrames_MatchGymnasium() {
            var kwargs = new Dictionary<string, object?> { ["render_mode"] = "rgb_array" };
            using PyRef python = PyRef.Of(() => Python.gymnasium.make_vec("CartPole-v1", 3, null, null, kwargs));
            using IVectorEnv dotnet = Gymnasium.MakeVec("CartPole-v1", numEnvs: 3, kwargs: new Info { ["render_mode"] = "rgb_array" });

            using (NDArray early = np.array(new[] { 0L, 1L, 0L })) {
                ByteContract.AssertSame("step before reset", Outcome.CapturePython(() => {
                    using PyObject actions = early.ToNumpyCopy();
                    return Python.gymnasium.Env.step(python.Object, actions);
                }), Outcome.Capture(() => StepTuple(dotnet.Step(early))));
            }

            var bounds = new Dictionary<string, object?> { ["low"] = -0.2, ["high"] = 0.3 };
            ByteContract.AssertSame("reset(seed=6, options={'low': -0.2, 'high': 0.3})", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 6, bounds)),
                Outcome.Capture(() => ResetTuple(dotnet.Reset(6, new Info { ["low"] = -0.2, ["high"] = 0.3 }))));
            ByteContract.AssertSame("render() after reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => Frames(dotnet.Render())));

            using (NDArray invalid = np.array(new[] { 0L, 2L, 1L })) {
                ByteContract.AssertSame("step(array([0, 2, 1]))", Outcome.CapturePython(() => {
                    using PyObject actions = invalid.ToNumpyCopy();
                    return Python.gymnasium.Env.step(python.Object, actions);
                }), Outcome.Capture(() => StepTuple(dotnet.Step(invalid))));
            }

            Lockstep("CartPole-v1 x3 rgb_array", python, dotnet, seed: 8, steps: 40, float64Rewards: true, framesEvery: 10);
        }

        /// <summary>
        ///     Runs both vector envs in lockstep: one seeded reset, then <paramref name="steps"/> steps of the batched action
        ///     spaces' seeded samples, compared before each step, with every result compared whole.
        /// </summary>
        /// <param name="label">What runs, for the reports.</param>
        /// <param name="python">Gymnasium's vector env.</param>
        /// <param name="dotnet">Gym.NET's vector env.</param>
        /// <param name="seed">The reset seed and the action spaces' seed.</param>
        /// <param name="steps">How many steps.</param>
        /// <param name="float64Rewards">Whether Gymnasium's rewards are widened to float64 before the comparison (CartPoleVectorEnv's float32 ones).</param>
        /// <param name="resetEnded">Whether ended sub-envs are reset through <c>reset_mask</c> after each step (<c>DISABLED</c> mode).</param>
        /// <param name="stateAttribute">An attribute of the vector env compared after every step, or <see langword="null"/>.</param>
        /// <param name="framesEvery">Renders both every this many steps; 0 never.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        internal static void Lockstep(string label, PyRef python, IVectorEnv dotnet, long seed, int steps, bool float64Rewards = false, bool resetEnded = false,
            string? stateAttribute = null, int framesEvery = 0) {
            ByteContract.AssertSame($"{label}: reset(seed={seed})", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, seed)), Outcome.Capture(() => ResetTuple(dotnet.Reset(seed))));
            using PyRef pythonSpace = PyRef.Of(() => Python.gymnasium.Env.action_space(python.Object));
            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(pythonSpace.Object, seed)).Expect($"{label}: action_space.seed({seed})");
            dotnet.ActionSpace.Seed(seed);
            for (int i = 1; i <= steps; i++) {
                using ParityAction action = PythonRuntime.Run(() => ParityAction.Sampled(Python.gymnasium.spaces.space.Space.sample(pythonSpace.Object), dotnet.ActionSpace.Sample()!));
                ByteContract.AssertSame($"{label}: action_space.sample() #{i}", Outcome.Returned(action.Facts), Outcome.Returned(ValueFacts.Of(action.DotNet)));
                Outcome pythonStep = float64Rewards
                    ? Outcome.CapturePython(() => Python.evalStmts(StepFloat64RewardsStatements, "result", ("env", python.Object), ("actions", action.PythonValue())))
                    : Outcome.CapturePython(() => Python.gymnasium.Env.step(python.Object, action.PythonValue()));
                // The terminations and truncations decide which sub-envs a DISABLED run resets; read before the result is released.
                bool[] ended = Array.Empty<bool>();
                ByteContract.AssertSame($"{label}: step #{i}", pythonStep, Outcome.Capture(() => {
                    VectorStepResult<object?> result = dotnet.Step(action.DotNet);
                    if (resetEnded) {
                        using NDArray endedArray = np.logical_or(result.Terminations, result.Truncations);
                        ended = endedArray.ToArray<bool>();
                    }

                    return StepTuple(result);
                }));
                if (stateAttribute is not null)
                    ByteContract.AssertSame($"{label}: {stateAttribute} after step #{i}", Outcome.CapturePython(() => Python.builtins.getattr(python.Object, stateAttribute)),
                        Outcome.Capture(() => PyAttributes.GetAttr(dotnet, stateAttribute)));
                if (framesEvery > 0 && i % framesEvery == 0)
                    ByteContract.AssertSame($"{label}: render() after step #{i}", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)), Outcome.Capture(() => Frames(dotnet.Render())));
                if (resetEnded && ended.Any(flag => flag)) {
                    using NDArray mask = np.array(ended);
                    var options = new Dictionary<string, object?> { ["reset_mask"] = mask };
                    ByteContract.AssertSame($"{label}: reset(options={{'reset_mask': ...}}) after step #{i}", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, null, options)),
                        Outcome.Capture(() => ResetTuple(dotnet.Reset(seed: null, options: new Info { ["reset_mask"] = mask }))));
                }
            }
        }

        /// <summary>A Gym.NET reset as the tuple Gymnasium returns.</summary>
        /// <param name="result">The reset.</param>
        /// <returns><c>(observations, infos)</c>.</returns>
        internal static object?[] ResetTuple(VectorResetResult<object?> result) => new[] { result.Observations, result.Infos };

        /// <summary>A Gym.NET step as the tuple Gymnasium returns.</summary>
        /// <param name="result">The step.</param>
        /// <returns><c>(observations, rewards, terminations, truncations, infos)</c>.</returns>
        internal static object?[] StepTuple(VectorStepResult<object?> result) => new object?[] { result.Observations, result.Rewards, result.Terminations, result.Truncations, result.Infos };

        /// <summary>Gym.NET's renders as the frames Gymnasium returns (a tuple or list of arrays), or <see langword="null"/>.</summary>
        /// <param name="renders">The renders.</param>
        /// <returns>The frames, or <see langword="null"/> for none.</returns>
        private static object?[]? Frames(IReadOnlyList<RenderResult>? renders) =>
            renders?.Select(render => render is RenderResult.Rgb rgb ? (object?)rgb.Frame : null).ToArray();
    }
}
