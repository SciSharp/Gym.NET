using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Reference;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Harness {
    /// <summary>
    ///     Plan §11: a parity harness that cannot fail is worthless, and one that cannot pass is useless. Each test here plants
    ///     a known defect (or none) and checks that the harness reports exactly it: the right call, the right field, nothing
    ///     else. They run against live Gymnasium but never against Gym.NET's product code, so they stay valid whatever the
    ///     port's state.
    /// </summary>
    [TestClass]
    public sealed class HarnessSelfTests : ParityTestBase {
        /// <summary>
        ///     Positive control: the faithful float64 transcription runs a whole CartPole-v1 episode (500 steps of the balancing
        ///     policy) in lockstep with Gymnasium with nothing reported: observations, rewards, flags, info, the float64 state
        ///     and the PCG64 state all bit-identical at every call.
        /// </summary>
        [TestMethod]
        public void Pair_ReportsNothing_ForAFaithfulTranscription() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.Faithful(), stateAttribute: "state");
            PairRun run = pair.Run(0, ActionSources.Policy("cartpole_balance"), maxSteps: 500);
            TestContext.WriteLine(run.ToString());
            Assert.AreEqual(500, run.Steps, "The balancing policy keeps the pole up for the whole episode, so all 500 steps are compared.");
            Assert.IsFalse(run.Ended);
        }

        /// <summary>
        ///     Negative control, plan §11 row 1: <c>tau = 0.02f</c> (today's Gym.NET defect B-11, isolated) is reported at step 1,
        ///     in the float64 state, and nowhere earlier: the reset, which does not use <c>tau</c>, still matches.
        /// </summary>
        [TestMethod]
        public void Pair_ReportsAFloat32Tau_AtStep1_InTheState() {
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.WithFloat32Tau(), stateAttribute: "state");
            var failure = Assert.ThrowsExactly<ParityException>(() => pair.Run(0, ActionSources.Policy("cartpole_balance"), maxSteps: 500));
            TestContext.WriteLine(failure.Message);

            Divergence divergence = failure.Divergence;
            Assert.AreEqual(1, divergence.Step, "tau first acts in step 1.");
            Assert.IsTrue(divergence.Has("state"), "The float64 state must differ.");
            Assert.IsFalse(divergence.Has("rng"), "tau does not touch the generator.");
            Assert.IsFalse(divergence.Has("reward") || divergence.Has("terminated") || divergence.Has("truncated") || divergence.Has("info"));
        }

        /// <summary>
        ///     State injection without a reset compares dynamics alone: the faithful transcription stays identical for 50 steps
        ///     from an injected state, and the generators, which no seeded reset has related, are left out with a note instead
        ///     of being reported as different. (Reading Gymnasium's <c>np_random</c> there would even create an entropy-seeded
        ///     one.) With <c>tau = 0.02f</c> the same comparison reports the state at the first step, and nothing about the
        ///     generators.
        /// </summary>
        [TestMethod]
        public void Pair_ComparesAnInjectedState_WithoutComparingUnseededGenerators() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using (var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.Faithful(), stateAttribute: "state")) {
                pair.InjectState("state", 0.01, -0.02, 0.03, 0.04);
                for (int i = 0; i < 50; i++) {
                    using var action = ParityAction.Discrete(i % 2);
                    pair.Step(action);
                }

                Assert.IsTrue(pair.Notes.Any(n => n.StartsWith("rng not compared: no seeded reset", System.StringComparison.Ordinal)), "The skipped generator check must be noted.");
            }

            using (var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.WithFloat32Tau(), stateAttribute: "state")) {
                pair.InjectState("state", 0.01, -0.02, 0.03, 0.04);
                using var action = ParityAction.Discrete(1);
                var failure = Assert.ThrowsExactly<ParityException>(() => pair.Step(action));
                TestContext.WriteLine(failure.Message);
                Assert.IsNull(failure.Divergence.Step, "Without a reset there is no trajectory position.");
                Assert.IsTrue(failure.Divergence.Has("state"));
                Assert.IsFalse(failure.Divergence.Has("rng"));
            }
        }

        /// <summary>
        ///     Negative control, plan §11 row 2: one extra draw is reported by the generator-state check at the reset itself,
        ///     although the observation (drawn before it) is still identical: the check sees a wrong draw before any value does.
        /// </summary>
        [TestMethod]
        public void Pair_ReportsAnExtraDraw_AtTheReset_ThroughTheGeneratorStateAlone() {
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.WithExtraDraw(), stateAttribute: "state");
            var failure = Assert.ThrowsExactly<ParityException>(() => pair.Reset(0));
            TestContext.WriteLine(failure.Message);

            Divergence divergence = failure.Divergence;
            Assert.AreEqual(0, divergence.Step);
            CollectionAssert.AreEqual(new[] { "rng" }, divergence.Fields.Select(f => f.Field).Distinct().ToArray(), "Only the generator state may differ.");
        }

        /// <summary>
        ///     Negative control, plan §11 row 3: one channel of one pixel off by one, in a real Gymnasium frame, is reported as
        ///     exactly one differing pixel with delta 1, in the lowest histogram bucket, at that pixel.
        /// </summary>
        [TestMethod]
        public void FrameDiff_ReportsOnePixelOffByOne() {
            Frame python;
            using (PyGymEnv env = PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array")) {
                env.Reset(0);
                python = env.Render();
            }

            byte[] altered = (byte[])python.Rgb.Clone();
            int offset = (123 * python.Width + 321) * 3 + 1;
            altered[offset] = (byte)(altered[offset] == 255 ? 254 : altered[offset] + 1);
            var dotnet = new Frame(python.Height, python.Width, altered);

            FrameComparison comparison = FrameDiff.Compare(python, dotnet);
            TestContext.WriteLine(comparison.Describe());
            Assert.AreEqual(1L, comparison.DifferingPixels);
            Assert.AreEqual(1, comparison.MaxChannelDelta);
            Assert.AreEqual<(int, int, int, int)?>((321, 123, 321, 123), comparison.Bounds, "The bounding box is the one altered pixel: x 321, y 123.");
            Assert.AreEqual(1L, comparison.Histogram[0], "Delta 1 belongs to the first bucket.");
            Assert.IsTrue(FrameDiff.Compare(python, new Frame(python.Height, python.Width, (byte[])python.Rgb.Clone())).Identical, "An exact copy must compare identical.");
        }

        /// <summary>
        ///     Negative control, plan §11 row 4: an info with the same keys and values in a different order is a divergence (the
        ///     key order of Gymnasium's infos is observable), and the report says so; the same order matches.
        /// </summary>
        [TestMethod]
        public void ByteContract_ReportsInfoKeysInADifferentOrder() {
            // {'lives': 3, 'score': 1.5}, built in that insertion order; the Python ints and float come from PyConvert.
            var info = new Dictionary<string, object?> { ["lives"] = 3L, ["score"] = 1.5 };
            ValueFacts python = Outcome.CapturePython(() => PyConvert.ToKwargs(info)).Expect("{'lives': 3, 'score': 1.5}");

            var swapped = new OrderedDictionary { { "score", 1.5 }, { "lives", 3L } };
            var failures = ByteContract.Compare("info", python, ValueFacts.Of(swapped));
            TestContext.WriteLine(string.Join("\n", failures));
            Assert.AreEqual(1, failures.Count);
            StringAssert.Contains(failures[0].Detail, "same keys in a different order");

            var ordered = new OrderedDictionary { { "lives", 3L }, { "score", 1.5 } };
            Assert.AreEqual(0, ByteContract.Compare("info", python, ValueFacts.Of(ordered)).Count, "The same keys in the same order, with equal values, match.");
        }

        /// <summary>
        ///     Negative control, plan §11 row 5: a reward computed in float32 is reported with its float64 bits and its ulp
        ///     distance. 0.1f widened is 0x3FB99999A0000000 against Python's 0x3FB999999999999A: 107374182 ulps apart.
        /// </summary>
        [TestMethod]
        public void ByteContract_ReportsAFloat32Reward_WithItsUlpDistance() {
            ValueFacts python = PythonFloat(0.1);
            var failures = ByteContract.Compare("reward", python, ValueFacts.Of(0.1f), FieldPolicy.Float64Value);
            TestContext.WriteLine(string.Join("\n", failures));
            Assert.AreEqual(1, failures.Count);
            StringAssert.Contains(failures[0].Detail, "107374182 ulp apart in float64");
            Assert.AreEqual(0, ByteContract.Compare("reward", python, ValueFacts.Of(0.1), FieldPolicy.Float64Value).Count, "The same double matches.");
            Assert.AreEqual(0, ByteContract.Compare("reward", PythonFloat(1.0), ValueFacts.Of(1.0f), FieldPolicy.Float64Value).Count, "A float32 value that widens exactly matches under the float64-value policy.");
        }

        /// <summary>
        ///     Negative control, plan §11 row 6: an export Python still holds keeps the leak gate from settling, and releasing it
        ///     lets the gate settle. (The test's own gate then passes, because the export is released before it ends.)
        /// </summary>
        [TestMethod]
        public void LeakGate_ReportsAnExportPythonStillHolds() {
            LeakGate.Baseline baseline = LeakGate.Capture();
            // A Python list holding the exported view stands for any Python-side owner (an env storing an action): the C#
            // handle is gone, but Python still pins the NumSharp buffer.
            PyRef holder;
            using (NDArray values = np.arange(8.0)) {
                holder = PyRef.Of(() => {
                    using PyObject exported = values.ToNumpy();
                    var list = new PyList();
                    list.Append(exported);
                    return list;
                });
            }

            try {
                Assert.IsFalse(LeakGate.WaitForReturn(baseline, 500, out string held), "A live export must keep the gate from settling.");
                TestContext.WriteLine(held);
            } finally {
                // Releasing the list drops the last Python reference to the view; its weakref finalizer ends the export.
                holder.Dispose();
            }

            Assert.IsTrue(LeakGate.WaitForReturn(baseline, 10_000, out string released), $"Releasing the export must settle the gate: {released}.");
        }

        /// <summary>
        ///     The dtype rule: a float32 array against a float64 one is a divergence even when every value agrees, and the report
        ///     says that casting would make them identical, which tells a missing cast (B-13's kind) from wrong values.
        /// </summary>
        [TestMethod]
        public void ByteContract_ReportsADtypeWidth_EvenWhenTheValuesAgree() {
            // np.array([1.5, -2.25, 0.1], dtype=np.float32): NumPy rounds 0.1 to float32 exactly as the (float) cast does.
            using NDArray narrow = np.array(new[] { 1.5f, -2.25f, 0.1f });
            ValueFacts python = Outcome.CapturePython(() => narrow.ToNumpyCopy()).Expect("np.array([1.5, -2.25, 0.1], dtype=np.float32)");
            using NDArray wider = np.array(new[] { 1.5, -2.25, (double)0.1f });
            var failures = ByteContract.Compare("observation", python, ValueFacts.Of(wider));
            TestContext.WriteLine(string.Join("\n", failures));
            Assert.AreEqual(1, failures.Count);
            Assert.AreEqual("observation.dtype", failures[0].Path);
            StringAssert.Contains(failures[0].Detail, "casting dotnet to <f4 would make them identical");
        }

        /// <summary>
        ///     The scalar rule: NumPy's <c>np.int64(3)</c> equals NumSharp's 0-d int64 array (NumSharp has no scalar types),
        ///     but not a 0-d int32 array, and not a 1-element 1-D array.
        /// </summary>
        [TestMethod]
        public void ByteContract_TreatsANumpyScalarAsA0dArray_OfTheSameDtypeOnly() {
            ValueFacts python = Outcome.CapturePython(() => Python.numpy.int64(3)).Expect("np.int64(3)");
            using NDArray same = NDArray.Scalar<long>(3L);
            using NDArray narrower = NDArray.Scalar<int>(3);
            using NDArray vector = np.array(new[] { 3L });
            Assert.AreEqual(0, ByteContract.Compare("value", python, ValueFacts.Of(same)).Count);
            Assert.AreEqual(1, ByteContract.Compare("value", python, ValueFacts.Of(narrower)).Count, "int64 against int32 is a dtype difference.");
            Assert.AreEqual(1, ByteContract.Compare("value", python, ValueFacts.Of(vector)).Count, "A scalar against a (1,) array is a difference.");
        }

        /// <summary>A Python <c>float</c>, described as the Python side describes one (<c>&lt;f8</c> bits).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The description.</returns>
        private static ValueFacts PythonFloat(double value) {
            return Outcome.CapturePython(() => PyConvert.ToPython(value)).Expect($"float({value.ToString("R", System.Globalization.CultureInfo.InvariantCulture)})");
        }

        /// <summary>
        ///     Error parity: a Python <c>ValueError</c> matches a <c>NumSharp.ValueError</c> with the same message (port plan
        ///     §3.4), and is reported against the wrong .NET type, against a different message, and against a return.
        /// </summary>
        [TestMethod]
        public void ByteContract_ComparesRaisedExceptions_ByMappedTypeAndMessage() {
            Outcome python = Outcome.CapturePython(() => {
                using var text = new PyString("x");
                return Python.builtins.@int(text);
            });
            Assert.IsNotNull(python.Raised);
            string message = python.Raised!.Message;

            Assert.AreEqual(0, ByteContract.Compare("call", python, Outcome.Capture(() => throw new ValueError(message))).Count);
            Assert.AreEqual("call.type", ByteContract.Compare("call", python, Outcome.Capture(() => throw new System.ArgumentException(message))).Single().Path);
            Assert.AreEqual("call.message", ByteContract.Compare("call", python, Outcome.Capture(() => throw new ValueError("another message"))).Single().Path);
            Assert.AreEqual("raised", ByteContract.Compare("call", python, Outcome.Capture(() => 0L)).Single().Field);
        }
    }
}
