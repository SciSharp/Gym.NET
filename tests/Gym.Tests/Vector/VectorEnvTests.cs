#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Gym.Tests.Helpers;

namespace Gym.Tests.Vector {
    /// <summary>
    ///     Headless contract tests of the vector API of P6.1 that need no Python: the Python-style enums, <c>MakeVec</c>'s
    ///     typed and untyped forms and modes, <c>SyncVectorEnv</c>'s ownership (copies, its own buffer, disposal), the info
    ///     merging rules, the vector wrappers and <c>CartPoleVectorEnv</c>'s refusals. Byte-for-byte agreement with Gymnasium
    ///     is the parity suite's job (<c>tests/Gym.Tests.Parity/Tests/Vector</c>).
    /// </summary>
    [TestClass]
    public class VectorEnvTests {
        /// <summary>
        ///     <see cref="AutoresetMode"/> and <see cref="VectorizeMode"/> are C# enums that behave as Python's enums where
        ///     Python sees them (DR-84): parsed from their values only (never a member name), named and valued as Gymnasium's,
        ///     printed with Python's <c>repr</c> and <c>str</c>, read from a member, its string or nothing (the default), and
        ///     equal only to themselves under Python's <c>==</c>.
        /// </summary>
        [TestMethod]
        public void Enums_ParsePrintAndCompare_AsPythons() {
            Assert.AreEqual(AutoresetMode.SameStep, AutoresetModes.Of("SameStep"));
            Assert.AreEqual("'SAME_STEP' is not a valid AutoresetMode", Assert.ThrowsExactly<ValueError>(() => AutoresetModes.Of("SAME_STEP")).Message);
            Assert.AreEqual("'SameStep ' is not a valid AutoresetMode", Assert.ThrowsExactly<ValueError>(() => AutoresetModes.Of("SameStep ")).Message);
            Assert.AreEqual("<AutoresetMode.NEXT_STEP: 'NextStep'>", AutoresetModes.Repr(AutoresetMode.NextStep));
            Assert.AreEqual("<AutoresetMode.NEXT_STEP: 'NextStep'>", PyBuiltins.Repr(AutoresetMode.NextStep), "A boxed member prints as Python's repr.");
            Assert.AreEqual("AutoresetMode.NEXT_STEP", AutoresetModes.Str(AutoresetMode.NextStep));
            Assert.AreEqual("AutoresetMode.SAME_STEP", PyBuiltins.Str(AutoresetMode.SameStep), "An f-string shows the member's str.");
            Assert.AreEqual("DISABLED", AutoresetModes.Name(AutoresetMode.Disabled));
            Assert.AreEqual("Disabled", AutoresetModes.Value(AutoresetMode.Disabled));
            CollectionAssert.AreEqual(new[] { AutoresetMode.NextStep, AutoresetMode.SameStep, AutoresetMode.Disabled }, AutoresetModes.All.ToArray(), "Gymnasium's order.");

            Assert.AreEqual(AutoresetMode.NextStep, AutoresetModes.From(null), "No mode is NEXT_STEP, Gymnasium's default.");
            Assert.AreEqual(AutoresetMode.Disabled, AutoresetModes.From("Disabled"), "A string reads as Python's AutoresetMode(value).");
            Assert.AreEqual(AutoresetMode.SameStep, AutoresetModes.From(AutoresetMode.SameStep));
            Assert.AreEqual("7 is not a valid AutoresetMode", Assert.ThrowsExactly<ValueError>(() => AutoresetModes.From((AutoresetMode)7)).Message, "A cast integer is no member.");
            Assert.AreEqual("1 is not a valid AutoresetMode", Assert.ThrowsExactly<ValueError>(() => AutoresetModes.From(1L)).Message);

            // Python's ==: a member equals itself only, never its value nor its index (AutoresetMode is a plain Enum).
            Assert.IsTrue(PyEquality.AreEqual(AutoresetMode.SameStep, AutoresetMode.SameStep));
            Assert.IsFalse(PyEquality.AreEqual(AutoresetMode.SameStep, "SameStep"));
            Assert.IsFalse(PyEquality.AreEqual(AutoresetMode.NextStep, 0L));
            Assert.AreEqual("Gym.Vector.AutoresetMode", PyBuiltins.TypeName(AutoresetMode.NextStep));

            Assert.AreEqual(VectorizeMode.VectorEntryPoint, VectorizeModes.Of("vector_entry_point"));
            Assert.AreEqual("<VectorizeMode.SYNC: 'sync'>", VectorizeModes.Repr(VectorizeMode.Sync));
            Assert.AreEqual("VectorizeMode.ASYNC", VectorizeModes.Str(VectorizeMode.Async));
            Assert.AreEqual("sync", VectorizeModes.Value(VectorizeMode.Sync));
            Assert.AreEqual("'SYNC' is not a valid VectorizeMode", Assert.ThrowsExactly<ValueError>(() => VectorizeModes.Of("SYNC")).Message);
            CollectionAssert.AreEqual(new[] { VectorizeMode.Async, VectorizeMode.Sync, VectorizeMode.VectorEntryPoint }, VectorizeModes.All.ToArray(), "Gymnasium's order.");
        }

        /// <summary>
        ///     <see cref="RenderMode"/> stands for Gymnasium's render-mode strings (DR-84, DR-93): each member's string, parsed back
        ///     by value only, <see langword="null"/> from <c>TryParse</c> for a mode no member names (Gymnasium accepts those),
        ///     printed and compared as the Python <c>str</c> it is, and turned into its string in <c>make</c>'s kwargs.
        /// </summary>
        [TestMethod]
        public void RenderMode_StandsForGymnasiumsStrings() {
            Assert.AreEqual(RenderModes.RgbArray, RenderModes.Name(RenderMode.RgbArray));
            Assert.AreEqual("state_pixels_list", RenderModes.Name(RenderMode.StatePixelsList));
            Assert.AreEqual(11, RenderModes.All.Count);
            foreach (RenderMode mode in RenderModes.All)
                Assert.AreEqual(mode, RenderModes.Parse(RenderModes.Name(mode)), $"{mode} round-trips through its string.");

            Assert.AreEqual(RenderMode.Human, RenderModes.Parse("human"));
            Assert.AreEqual("'RgbArray' is not a valid RenderMode", Assert.ThrowsExactly<ValueError>(() => RenderModes.Parse("RgbArray")).Message, "C# names are not values.");
            Assert.IsNull(RenderModes.TryParse("no-render-mode"), "A mode Gymnasium accepts but no member names.");
            Assert.IsNull(RenderModes.TryParse(null));
            Assert.IsTrue(RenderModes.TryParse("ansi", out RenderMode ansi));
            Assert.AreEqual(RenderMode.Ansi, ansi);
            Assert.IsFalse(RenderModes.TryParse("Testing mode", out _));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => RenderModes.Name((RenderMode)99));

            Assert.AreEqual("'rgb_array'", PyBuiltins.Repr(RenderMode.RgbArray), "A member is the Python str it stands for.");
            Assert.AreEqual("rgb_array", PyBuiltins.Str(RenderMode.RgbArray));
            Assert.AreEqual("str", PyBuiltins.TypeName(RenderMode.RgbArray));
            Assert.IsTrue(PyEquality.AreEqual(RenderMode.Human, "human"));
            Assert.IsFalse(PyEquality.AreEqual(RenderMode.Human, 0L), "Not an integer to Python.");

            Info kwargs = PyKwargs.From(new { render_mode = RenderMode.RgbArray, sutton_barto_reward = true });
            Assert.AreEqual("rgb_array", kwargs["render_mode"], "make's kwargs hold Gymnasium's string.");
            Assert.AreEqual(true, kwargs["sutton_barto_reward"]);
            Assert.AreEqual("human", PyKwargs.From(new Info { ["render_mode"] = RenderMode.Human })["render_mode"]);
        }

        /// <summary>
        ///     <c>MakeVec</c> picks the vector entry point by default and sync when asked; the typed form builds
        ///     <c>SyncVectorEnv&lt;TObs, TAct&gt;</c> or refuses other types; the async mode builds <c>AsyncVectorEnv</c> (P6.2).
        /// </summary>
        [TestMethod]
        public void MakeVec_BuildsTheModeAskedFor() {
            using (IVectorEnv byDefault = MakeVec("CartPole-v1", numEnvs: 3))
                Assert.IsInstanceOfType(byDefault, typeof(CartPoleVectorEnv));
            using (IVectorEnv sync = MakeVec("CartPole-v1", numEnvs: 3, vectorizationMode: "sync"))
                Assert.IsInstanceOfType(sync, typeof(SyncVectorEnv));
            using (VectorEnv<NDArray, NDArray> typed = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: VectorizeMode.Sync)) {
                Assert.IsInstanceOfType(typed, typeof(SyncVectorEnv<NDArray, NDArray>));
                Assert.AreEqual("SyncVectorEnv(CartPole-v1, num_envs=2)", typed.ToString());
                PyAssert.AreEqual(new Info { ["num_envs"] = 2L, ["vectorization_mode"] = "sync" }, typed.Spec!.Kwargs);
            }

            Assert.ThrowsExactly<TypeError>(() => MakeVec<object?[], NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "sync"));
            using (IVectorEnv asynchronous = MakeVec("CartPole-v1", numEnvs: 2, vectorizationMode: "async")) {
                Assert.IsInstanceOfType(asynchronous, typeof(AsyncVectorEnv));
                Assert.AreEqual("AsyncVectorEnv(CartPole-v1, num_envs=2)", asynchronous.ToString());
            }

            using (VectorEnv<NDArray, NDArray> typedAsync = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "async"))
                Assert.IsInstanceOfType(typedAsync, typeof(AsyncVectorEnv<NDArray, NDArray>));
            Assert.AreEqual("Invalid vectorization mode: 'bogus', valid modes: ['async', 'sync', 'vector_entry_point']",
                Assert.ThrowsExactly<ValueError>(() => MakeVec("CartPole-v1", vectorizationMode: "bogus")).Message);
        }

        /// <summary>
        ///     With <c>copy</c> (the default) every observation batch is a new array the caller owns; without it the env hands out
        ///     its own buffer, the same array every call, overwritten in place.
        /// </summary>
        [TestMethod]
        public void SyncVectorEnv_Copy_DecidesWhoOwnsTheBatch() {
            Func<IEnv>[] fns = Enumerable.Repeat<Func<IEnv>>(() => Make("CartPole-v1"), 2).ToArray();
            using (var copying = new SyncVectorEnv<NDArray, NDArray>(fns)) {
                using NDArray first = copying.Reset(seed: 1).Observations;
                using NDArray second = copying.Reset(seed: 1).Observations;
                Assert.AreNotSame(first, second);
                Assert.IsTrue(np.array_equal(first, second), "The same seed gives the same batch.");
            }

            using var sharing = new SyncVectorEnv<NDArray, NDArray>(fns, copy: false);
            NDArray buffer = sharing.Reset(seed: 1).Observations;
            using NDArray actions = np.array(new[] { 0L, 1L });
            NDArray stepped = sharing.Step(actions).Observations;
            Assert.AreSame(buffer, stepped, "Without copy, every call returns the env's own buffer.");
        }

        /// <summary>Disposing a sync vector env closes and disposes the sub-envs it built; a second close does nothing.</summary>
        [TestMethod]
        public void SyncVectorEnv_Dispose_ReleasesItsSubEnvs() {
            var envs = new List<IEnv>();
            var vector = new SyncVectorEnv(Enumerable.Repeat<Func<IEnv>>(() => {
                IEnv env = Make("CartPole-v1");
                envs.Add(env);
                return env;
            }, 2));
            (vector.Reset(seed: 0).Observations as IDisposable)?.Dispose();
            vector.Dispose();
            Assert.IsTrue(vector.Closed);
            Assert.AreEqual(2, envs.Count);
            vector.Close();
        }

        /// <summary>
        ///     The info merging rules (Gymnasium's <c>_add_info</c>): a batch typed by the first value seen (an int truncating a
        ///     later float), arrays batched along a new axis, strings in an object batch, nested dicts merged recursively, and a
        ///     mask per key.
        /// </summary>
        [TestMethod]
        public void AddInfo_MergesAsGymnasium() {
            using var vector = new InfoProbe(3);
            Info merged = vector.Merge(new Info(), new Info { ["prob"] = 1L, ["name"] = "a", ["extra"] = new Info { ["x"] = 1.5 } }, 0);
            using NDArray mask = np.array(new sbyte[] { 1, 0, 1 });
            merged = vector.Merge(merged, new Info { ["prob"] = 0.75, ["mask"] = mask }, 2);

            var prob = (NDArray)merged["prob"]!;
            Assert.AreEqual(NPTypeCode.Int64, prob.typecode, "The first value, an int, made an int64 batch.");
            CollectionAssert.AreEqual(new long[] { 1, 0, 0 }, prob.ToArray<long>(), "0.75 truncated into it, as NumPy casts.");
            CollectionAssert.AreEqual(new[] { true, false, true }, ((NDArray)merged["_prob"]!).ToArray<bool>());
            CollectionAssert.AreEqual(new object?[] { "a", null, null }, (object?[])merged["name"]!);
            var extra = (Info)merged["extra"]!;
            CollectionAssert.AreEqual(new[] { 1.5, 0.0, 0.0 }, ((NDArray)extra["x"]!).ToArray<double>());
            var masks = (NDArray)merged["mask"]!;
            CollectionAssert.AreEqual(new long[] { 3, 3 }, masks.shape);
            CollectionAssert.AreEqual(new[] { "prob", "_prob", "name", "_name", "extra", "_extra", "mask", "_mask" }, merged.Keys.ToArray(), "Each key, then its mask.");
        }

        /// <summary>
        ///     The vector wrappers: a reward wrapper transforms the rewards and forwards the rest; an observation wrapper refuses a
        ///     <c>SAME_STEP</c> env with Gymnasium's message; a wrapper prints itself around its env and forwards the spec.
        /// </summary>
        [TestMethod]
        public void VectorWrappers_ForwardAndTransform() {
            using var wrapped = new DoubledRewards(MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "sync"));
            Assert.AreEqual("<DoubledRewards, SyncVectorEnv(CartPole-v1, num_envs=2)>", wrapped.ToString());
            Assert.AreEqual("CartPole-v1", wrapped.Spec!.Id);
            Assert.AreEqual(2, wrapped.NumEnvs);
            wrapped.Reset(seed: 2).Observations.Dispose();
            using NDArray actions = np.array(new[] { 1L, 0L });
            (NDArray observations, NDArray rewards, NDArray terminations, NDArray truncations, Info _) = wrapped.Step(actions);
            CollectionAssert.AreEqual(new[] { 2.0, 2.0 }, rewards.ToArray<double>());
            observations.Dispose();
            rewards.Dispose();
            terminations.Dispose();
            truncations.Dispose();

            using VectorEnv<NDArray, NDArray> sameStep = MakeVec<NDArray, NDArray>("CartPole-v1", numEnvs: 2, vectorizationMode: "sync", vectorKwargs: new Info { ["autoreset_mode"] = "SameStep" });
            Assert.AreEqual("Expected autoreset_mode to be NEXT_STEP or DISABLED, got AutoresetMode.SAME_STEP",
                Assert.ThrowsExactly<ValueError>(() => new Halved(sameStep)).Message);
        }

        /// <summary><c>CartPoleVectorEnv</c> refuses a step before reset and actions outside its space, with Gymnasium's texts.</summary>
        [TestMethod]
        public void CartPoleVectorEnv_Refusals() {
            using var env = new CartPoleVectorEnv(numEnvs: 2);
            using NDArray actions = np.array(new[] { 0L, 1L });
            Assert.AreEqual("Call reset before using step method.", Assert.ThrowsExactly<AssertionError>(() => env.Step(actions)).Message);
            env.Reset(seed: 0).Observations.Dispose();
            using NDArray outside = np.array(new[] { 0L, 2L });
            StringAssert.EndsWith(Assert.ThrowsExactly<AssertionError>(() => env.Step(outside)).Message, " (<class 'numpy.ndarray'>) invalid");
            Assert.AreEqual(new BigInteger(0), env.NpRandomSeed);
        }

        /// <summary>A vector env that exposes the protected info merge, over two sub-envs of nothing.</summary>
        private sealed class InfoProbe : VectorEnv<NDArray, NDArray> {
            /// <summary>Creates the probe.</summary>
            /// <param name="numEnvs">The batches' length.</param>
            public InfoProbe(int numEnvs) => NumEnvs = numEnvs;

            /// <summary>The merge under test.</summary>
            /// <param name="vectorInfos">The infos so far.</param>
            /// <param name="envInfo">The sub-env's info.</param>
            /// <param name="envNum">The sub-env.</param>
            /// <returns>The merged infos.</returns>
            public Info Merge(Info vectorInfos, Info envInfo, int envNum) => AddInfo(vectorInfos, envInfo, envNum);

            /// <summary>Not used.</summary>
            /// <param name="seed">Unused.</param>
            /// <param name="options">Unused.</param>
            /// <returns>Never.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override VectorResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotSupportedException();

            /// <summary>Not used.</summary>
            /// <param name="actions">Unused.</param>
            /// <returns>Never.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override VectorStepResult<NDArray> Step(NDArray actions) => throw new NotSupportedException();
        }

        /// <summary>A reward wrapper that doubles every reward.</summary>
        private sealed class DoubledRewards : VectorRewardWrapper<NDArray, NDArray> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The vector env.</param>
            public DoubledRewards(VectorEnv<NDArray, NDArray> env) : base(env) { }

            /// <summary>Doubles the rewards.</summary>
            /// <param name="rewards">The env's rewards; released here.</param>
            /// <returns>New rewards.</returns>
            public override NDArray Rewards(NDArray rewards) {
                using (rewards)
                    return rewards * 2.0;
            }
        }

        /// <summary>An observation wrapper that halves the observations.</summary>
        private sealed class Halved : VectorObservationWrapper<NDArray, NDArray, NDArray> {
            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The vector env.</param>
            public Halved(VectorEnv<NDArray, NDArray> env) : base(env) { }

            /// <summary>Halves the observations.</summary>
            /// <param name="observations">The env's observations; released here.</param>
            /// <returns>New observations.</returns>
            public override NDArray Observations(NDArray observations) {
                using (observations)
                    return observations / 2.0f;
            }
        }
    }
}
