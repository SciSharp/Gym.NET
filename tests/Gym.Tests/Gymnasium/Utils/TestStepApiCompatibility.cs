#nullable enable
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>
    ///     Gymnasium's step-API compatibility tests, run against <see cref="StepApiCompatibility"/>: the single-env cases
    ///     through the typed conversions, the vector cases (<c>is_vector_env=True</c>) and the dispatcher through the tuple form.
    /// </summary>
    /// <remarks>
    ///     The Python tuples' <c>0, 0</c> observation and reward are <c>0L</c> and <c>0.0</c>; <c>data_equivalence</c> of the
    ///     round trip is the field-wise comparison of the result types (<see cref="PyEquality"/> for the infos). As in the
    ///     Python test, one info object flows through both conversions, which edit it in place. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_step_api_compatibility.py">tests/utils/test_step_api_compatibility.py</see>.
    /// </remarks>
    [TestClass]
    public class TestStepApiCompatibility {
        /// <summary>The single-env rows of <c>test_to_done_step_api</c>'s parametrization.</summary>
        /// <returns>done, the info (as entries), expected terminated, expected truncated.</returns>
        public static IEnumerable<object[]> DoneReturns() {
            yield return new object[] { false, new Dictionary<string, object?> { ["Test-info"] = true }, false, false };
            yield return new object[] { false, new Dictionary<string, object?> { ["TimeLimit.truncated"] = false }, false, false };
            yield return new object[] { true, new Dictionary<string, object?>(), true, false };
            yield return new object[] { true, new Dictionary<string, object?> { ["TimeLimit.truncated"] = true }, false, true };
            yield return new object[] { true, new Dictionary<string, object?> { ["Test-info"] = true }, true, false };
        }

        /// <summary>
        ///     Converting a done-API transition splits <c>done</c> by <c>"TimeLimit.truncated"</c>, removes that key, and
        ///     converting back restores the original.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_step_api_compatibility.py#L57">tests/utils/test_step_api_compatibility.py (test_to_done_step_api)</see>.</remarks>
        /// <param name="done">The done flag.</param>
        /// <param name="infoEntries">The info.</param>
        /// <param name="expectedTerminated">The expected <c>terminated</c>.</param>
        /// <param name="expectedTruncated">The expected <c>truncated</c>.</param>
        [TestMethod]
        [DynamicData(nameof(DoneReturns))]
        public void TestToDoneStepApi(bool done, Dictionary<string, object?> infoEntries, bool expectedTerminated, bool expectedTruncated) {
            // One info object throughout, as in Python: the conversions edit it in place, so the round trip compares the
            // original transition, edited, with the converted one.
            var doneReturns = new DoneStepResult<object?>(0L, 0.0, done, new Info(infoEntries));

            (_, _, bool terminated, bool truncated, Info info) = StepApiCompatibility.ConvertToTerminatedTruncatedStepApi(doneReturns);
            Assert.AreEqual(expectedTerminated, terminated);
            Assert.AreEqual(expectedTruncated, truncated);
            Assert.IsFalse(info.ContainsKey("TimeLimit.truncated"));

            DoneStepResult<object?> roundtrippedReturns = StepApiCompatibility.ConvertToDoneStepApi(new StepResult<object?>(0L, 0.0, terminated, truncated, info));
            Assert.AreEqual(doneReturns, roundtrippedReturns);
        }

        /// <summary>The single-env rows of <c>test_to_terminated_truncated_step_api</c>'s parametrization.</summary>
        /// <returns>terminated, truncated, the info (as entries), expected done, expected truncated.</returns>
        public static IEnumerable<object[]> TerminatedTruncatedReturns() {
            yield return new object[] { false, false, new Dictionary<string, object?> { ["Test-info"] = true }, false, false };
            yield return new object[] { true, false, new Dictionary<string, object?>(), true, false };
            yield return new object[] { false, true, new Dictionary<string, object?>(), true, true };
        }

        /// <summary>
        ///     Converting a terminated/truncated transition merges the flags into <c>done</c> and records
        ///     <c>"TimeLimit.truncated"</c> only when the episode ended; converting back restores the original.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_step_api_compatibility.py#L115">tests/utils/test_step_api_compatibility.py (test_to_terminated_truncated_step_api)</see>.</remarks>
        /// <param name="terminated">The terminated flag.</param>
        /// <param name="truncated">The truncated flag.</param>
        /// <param name="infoEntries">The info.</param>
        /// <param name="expectedDone">The expected <c>done</c>.</param>
        /// <param name="expectedTruncated">The expected <c>"TimeLimit.truncated"</c> when done.</param>
        [TestMethod]
        [DynamicData(nameof(TerminatedTruncatedReturns))]
        public void TestToTerminatedTruncatedStepApi(bool terminated, bool truncated, Dictionary<string, object?> infoEntries, bool expectedDone, bool expectedTruncated) {
            // One info object throughout, as in Python (see TestToDoneStepApi).
            var terminatedTruncatedReturns = new StepResult<object?>(0L, 0.0, terminated, truncated, new Info(infoEntries));

            (_, _, bool done, Info info) = StepApiCompatibility.ConvertToDoneStepApi(terminatedTruncatedReturns);
            Assert.AreEqual(expectedDone, done);
            if (expectedDone)
                Assert.AreEqual(expectedTruncated, info["TimeLimit.truncated"]);
            else
                Assert.IsFalse(info.ContainsKey("TimeLimit.truncated"));

            StepResult<object?> roundtrippedReturns = StepApiCompatibility.ConvertToTerminatedTruncatedStepApi(new DoneStepResult<object?>(0L, 0.0, done, info));
            Assert.AreEqual(terminatedTruncatedReturns, roundtrippedReturns);
        }

        /// <summary>
        ///     <c>terminated=True</c> and <c>truncated=True</c> cannot be encoded in the done API; it becomes <c>done=True</c>
        ///     with <c>{"TimeLimit.truncated": False}</c>.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_step_api_compatibility.py#L148">tests/utils/test_step_api_compatibility.py (test_edge_case)</see>.</remarks>
        [TestMethod]
        public void TestEdgeCase() {
            (_, _, bool done, Info info) = StepApiCompatibility.ConvertToDoneStepApi(new StepResult<object?>(0L, 0.0, true, true, new Info()));
            Assert.IsTrue(done);
            Assert.AreEqual(new Info { ["TimeLimit.truncated"] = false }, info);

            // Test with vector dict info
            object?[] vectorDict = StepApiCompatibility.ConvertToDoneStepApi(new object?[] { 0L, 0L, np.array(new[] { true }), np.array(new[] { true }), new Info() }, isVectorEnv: true);
            Assert.IsTrue(np.all((NDArray)vectorDict[2]!));
            Assert.IsTrue(EnvChecker.DataEquivalence(vectorDict[3], new Info { ["TimeLimit.truncated"] = np.array(new[] { false }) }));

            // Test with vector list info
            object?[] vectorList = StepApiCompatibility.ConvertToDoneStepApi(new object?[] {
                0L, 0L, np.array(new[] { true }), np.array(new[] { true }), new List<Info> { new() { ["Test-Info"] = true } },
            }, isVectorEnv: true);
            Assert.IsTrue(np.all((NDArray)vectorList[2]!));
            Assert.IsTrue(EnvChecker.DataEquivalence(vectorList[3], new List<Info> { new() { ["Test-Info"] = true, ["TimeLimit.truncated"] = false } }));
        }

        /// <summary>
        ///     The vector rows of <c>test_to_done_step_api</c> through the tuple form: a list of infos, a dict with an array,
        ///     and an empty dict; the key is popped, and converting back restores the returns.
        /// </summary>
        /// <param name="variant">0: a list of infos; 1: a dict of arrays; 2: an empty dict.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_step_api_compatibility.py#L57">tests/utils/test_step_api_compatibility.py (test_to_done_step_api)</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        public void TestToDoneStepApi_Vector(int variant) {
            (object?[] doneReturns, bool[] expectedTerminated, bool[] expectedTruncated) = variant switch {
                0 => (new object?[] { 0L, 0L, np.array(new[] { false, true, true }), new List<Info> { new(), new(), new() { ["TimeLimit.truncated"] = true } } },
                    new[] { false, true, false }, new[] { false, false, true }),
                1 => (new object?[] { 0L, 0L, np.array(new[] { false, true, true }), new Info { ["TimeLimit.truncated"] = np.array(new[] { false, false, true }) } },
                    new[] { false, true, false }, new[] { false, false, true }),
                _ => (new object?[] { 0L, 0L, np.array(new[] { false, true }), new Info() }, new[] { false, true }, new[] { false, false }),
            };

            object?[] converted = StepApiCompatibility.ConvertToTerminatedTruncatedStepApi(doneReturns, isVectorEnv: true);
            CollectionAssert.AreEqual(expectedTerminated, ((NDArray)converted[2]!).ToArray<bool>());
            CollectionAssert.AreEqual(expectedTruncated, ((NDArray)converted[3]!).ToArray<bool>());

            if (converted[4] is List<Info> list)
                Assert.IsTrue(list.All(subInfo => !subInfo.ContainsKey("TimeLimit.truncated")));
            else
                Assert.IsFalse(((Info)converted[4]!).ContainsKey("TimeLimit.truncated"));

            object?[] roundtrippedReturns = StepApiCompatibility.ConvertToDoneStepApi(new object?[] { 0L, 0L, converted[2], converted[3], converted[4] }, isVectorEnv: true);
            Assert.IsTrue(EnvChecker.DataEquivalence(doneReturns, roundtrippedReturns));
        }

        /// <summary>
        ///     The vector rows of <c>test_to_terminated_truncated_step_api</c>: dict infos with and without a truncation, and a
        ///     list of infos; the key is recorded for the ended sub-envs, and converting back restores the returns.
        /// </summary>
        /// <param name="variant">0: a dict; 1: a dict with no truncation; 2: a list of infos.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_step_api_compatibility.py#L115">tests/utils/test_step_api_compatibility.py (test_to_terminated_truncated_step_api)</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        public void TestToTerminatedTruncatedStepApi_Vector(int variant) {
            (object?[] returns, bool[] expectedDone, bool[] expectedTruncated) = variant switch {
                0 => (new object?[] { 0L, 0L, np.array(new[] { false, true, false }), np.array(new[] { false, false, true }), new Info() },
                    new[] { false, true, true }, new[] { false, false, true }),
                1 => (new object?[] { 0L, 0L, np.array(new[] { false, true }), np.array(new[] { false, false }), new Info() }, new[] { false, true }, new[] { false, false }),
                _ => (new object?[] { 0L, 0L, np.array(new[] { false, true, false }), np.array(new[] { false, false, true }), new List<Info> { new() { ["Test-Info"] = true }, new(), new() } },
                    new[] { false, true, true }, new[] { false, false, true }),
            };

            object?[] converted = StepApiCompatibility.ConvertToDoneStepApi(returns, isVectorEnv: true);
            CollectionAssert.AreEqual(expectedDone, ((NDArray)converted[2]!).ToArray<bool>());

            if (converted[3] is List<Info> list) {
                for (int i = 0; i < list.Count; i++) {
                    if (expectedDone[i])
                        Assert.AreEqual(expectedTruncated[i], System.Convert.ToBoolean(list[i]["TimeLimit.truncated"]), $"sub-env {i}");
                    else
                        Assert.IsFalse(list[i].ContainsKey("TimeLimit.truncated"), $"sub-env {i}");
                }
            } else {
                var info = (Info)converted[3]!;
                if (expectedDone.Any(done => done))
                    CollectionAssert.AreEqual(expectedTruncated, ((NDArray)info["TimeLimit.truncated"]!).ToArray<bool>());
                else
                    Assert.IsFalse(info.ContainsKey("TimeLimit.truncated"));
            }

            object?[] roundtrippedReturns = StepApiCompatibility.ConvertToTerminatedTruncatedStepApi(new object?[] { 0L, 0L, converted[2], converted[3] }, isVectorEnv: true);
            Assert.IsTrue(EnvChecker.DataEquivalence(returns, roundtrippedReturns));
        }

        /// <summary>
        ///     <c>step_api_compatibility</c> dispatches on the tuple's length: either form converts to the asked one, and a tuple
        ///     already in it comes back as it is.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/step_api_compatibility.py#L139">gymnasium/utils/step_api_compatibility.py (step_api_compatibility)</see>.</remarks>
        [TestMethod]
        public void TestStepApiCompatibilityDispatch() {
            var fiveReturns = new object?[] { 0L, 0.0, false, true, new Info() };
            Assert.AreSame(fiveReturns, StepApiCompatibility.Dispatch(fiveReturns));
            object?[] fourReturns = StepApiCompatibility.Dispatch(fiveReturns, outputTruncationBool: false);
            Assert.AreEqual(4, fourReturns.Length);
            Assert.AreEqual(true, fourReturns[2]);
            Assert.AreEqual(true, ((Info)fourReturns[3]!)["TimeLimit.truncated"]);
            Assert.AreSame(fourReturns, StepApiCompatibility.Dispatch(fourReturns, outputTruncationBool: false));
            object?[] backToFive = StepApiCompatibility.Dispatch(fourReturns);
            CollectionAssert.AreEqual(new object?[] { 0L, 0.0, false, true }, backToFive.Take(4).ToArray());
            Assert.ThrowsExactly<AssertionError>(() => StepApiCompatibility.Dispatch(new object?[] { 0L, 0.0, false }));
            Assert.AreEqual("Unexpected value of infos, as is_vector_envs=False, expects `info` to be a list or dict, actual type: <class 'str'>",
                Assert.ThrowsExactly<TypeError>(() => StepApiCompatibility.Dispatch(new object?[] { 0L, 0.0, np.array(new[] { true }), "info" }, isVectorEnv: true)).Message);
        }
    }
}
