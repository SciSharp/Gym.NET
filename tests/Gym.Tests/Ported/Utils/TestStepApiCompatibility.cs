#nullable enable
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Ported.Utils {
    /// <summary>
    ///     Port of <c>tests/utils/test_step_api_compatibility.py</c> @ 9e04324, single-env cases (<c>is_vector_env=False</c>);
    ///     the vector cases come with the vector API (P6).
    /// </summary>
    /// <remarks>
    ///     The Python tuples' <c>0, 0</c> observation and reward are <c>0L</c> and <c>0.0</c>; <c>data_equivalence</c> of the
    ///     round trip is the field-wise comparison of the result types (<see cref="PyEquality"/> for the infos). As in the
    ///     Python test, one info object flows through both conversions, which edit it in place.
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
        ///     Port of <c>test_to_done_step_api</c> (single env): converting a done-API transition splits <c>done</c> by
        ///     <c>"TimeLimit.truncated"</c>, removes that key, and converting back restores the original.
        /// </summary>
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
        ///     Port of <c>test_to_terminated_truncated_step_api</c> (single env): converting a terminated/truncated transition
        ///     merges the flags into <c>done</c> and records <c>"TimeLimit.truncated"</c> only when the episode ended; converting
        ///     back restores the original.
        /// </summary>
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
        ///     Port of <c>test_edge_case</c> (single env): <c>terminated=True</c> and <c>truncated=True</c> cannot be encoded
        ///     in the done API; it becomes <c>done=True</c> with <c>{"TimeLimit.truncated": False}</c>.
        /// </summary>
        [TestMethod]
        public void TestEdgeCase() {
            (_, _, bool done, Info info) = StepApiCompatibility.ConvertToDoneStepApi(new StepResult<object?>(0L, 0.0, true, true, new Info()));
            Assert.IsTrue(done);
            Assert.AreEqual(new Info { ["TimeLimit.truncated"] = false }, info);
        }
    }
}
