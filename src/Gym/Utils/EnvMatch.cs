#nullable enable
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.utils.env_match</c>: drives two envs with the same seed and actions and asserts they behave
    ///     the same, for checking a port, a refactor or a wrapper against the original (a legacy env against its new
    ///     version during a migration, for one).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_match.py">gymnasium/utils/env_match.py</see>.</remarks>
    public static class EnvMatch {
        /// <summary>The comparisons <c>info_comparison</c> accepts.</summary>
        private static readonly string[] InfoComparisons = { "equivalence", "superset", "skip", "keys-equivalence", "keys-superset" };

        /// <summary>
        ///     Asserts that two envs are identical for <paramref name="numSteps"/> steps (Gymnasium's
        ///     <c>check_environments_match</c>): the same spaces, then a seeded reset of both and steps with actions sampled
        ///     from <paramref name="envA"/>'s seeded action space, comparing observations, rewards, signals, infos and renders,
        ///     and resetting both (seeded again) whenever either episode ends.
        /// </summary>
        /// <param name="envA">The first env; its action space supplies the actions.</param>
        /// <param name="envB">The second env.</param>
        /// <param name="numSteps">The number of steps; 0 checks the reset only.</param>
        /// <param name="seed">The seed of the resets and of <paramref name="envA"/>'s action space.</param>
        /// <param name="skipObs">Skip the observation space and the observations.</param>
        /// <param name="skipRew">Skip the rewards.</param>
        /// <param name="skipTerminal">Skip the <c>terminated</c> signals.</param>
        /// <param name="skipTruncated">Skip the <c>truncated</c> signals.</param>
        /// <param name="skipRender">Skip the renders; they are skipped anyway when either env renders nothing or <c>human</c>.</param>
        /// <param name="infoComparison">
        ///     <c>equivalence</c> (the infos are equivalent), <c>superset</c> (every entry of A's info is in B's, equivalent),
        ///     <c>keys-equivalence</c> (the same keys), <c>keys-superset</c> (B's keys include A's), or <c>skip</c>.
        /// </param>
        /// <exception cref="AssertionError">
        ///     Thrown at the first difference, with Gymnasium's message naming the step and both values; with an empty message
        ///     for an unknown <paramref name="infoComparison"/> or different spaces, as Gymnasium's bare asserts.
        /// </exception>
        /// <exception cref="KeyError">Thrown in <c>superset</c> mode when B's info lacks one of A's keys, as Python's <c>info_b[key]</c> raises.</exception>
        /// <remarks>
        ///     Both envs are stepped as far as the check gets and left there. The observations, infos and frames are compared
        ///     and dropped; the frames are released at once.
        /// </remarks>
        public static void CheckEnvironmentsMatch(IEnv envA, IEnv envB, int numSteps, long seed = 0, bool skipObs = false, bool skipRew = false,
            bool skipTerminal = false, bool skipTruncated = false, bool skipRender = false, string infoComparison = "equivalence") {
            skipRender = skipRender || envA.Unwrapped.RenderMode is null or "human" || envB.Unwrapped.RenderMode is null or "human";

            Assert(InfoComparisons.Contains(infoComparison), "");
            Assert(envA.ActionSpace.Equals(envB.ActionSpace), "");
            Assert(skipObs || envA.ObservationSpace.Equals(envB.ObservationSpace), "");

            envA.ActionSpace.Seed(seed);
            (object? obsA, Info infoA) = envA.Reset(seed);
            (object? obsB, Info infoB) = envB.Reset(seed);

            Assert(skipObs || EnvChecker.DataEquivalence(obsA, obsB),
                $"resetting observation is not equivalent, observation_a = {PyBuiltins.Str(obsA)}, observation_b = {PyBuiltins.Str(obsB)}");
            CompareInfos(infoA, infoB, infoComparison, "resetting", "");
            if (!skipRender)
                Assert(RendersEqual(envA, envB), "resetting render is not equivalent");

            for (int step = 0; step < numSteps; step++) {
                string at = $" in step = {step.ToString(CultureInfo.InvariantCulture)}";
                object? action = envA.ActionSpace.Sample();
                StepResult<object?> a = envA.Step(action);
                StepResult<object?> b = envB.Step(action);
                Assert(skipObs || EnvChecker.DataEquivalence(a.Observation, b.Observation),
                    $"stepping observation is not equivalent{at}, observation_a = {PyBuiltins.Str(a.Observation)}, observation_b = {PyBuiltins.Str(b.Observation)}");
                Assert(skipRew || EnvChecker.DataEquivalence(a.Reward, b.Reward), $"stepping reward is not equivalent{at}, reward_a = {PyBuiltins.Str(a.Reward)}, reward_b = {PyBuiltins.Str(b.Reward)}");
                Assert(skipTerminal || a.Terminated == b.Terminated,
                    $"stepping terminal is not equivalent{at}, terminal_a = {PyBuiltins.Str(a.Terminated)}, terminal_b = {PyBuiltins.Str(b.Terminated)}");
                Assert(skipTruncated || a.Truncated == b.Truncated,
                    $"stepping truncated is not equivalent{at}, truncated_a = {PyBuiltins.Str(a.Truncated)}, truncated_b = {PyBuiltins.Str(b.Truncated)}");
                CompareInfos(a.Info, b.Info, infoComparison, "stepping", at);
                if (!skipRender)
                    Assert(RendersEqual(envA, envB), $"stepping render is not equivalent{at}");

                if (a.Terminated || a.Truncated || b.Terminated || b.Truncated) {
                    (obsA, infoA) = envA.Reset(seed);
                    (obsB, infoB) = envB.Reset(seed);
                    Assert(skipObs || EnvChecker.DataEquivalence(obsA, obsB),
                        $"resetting observation is not equivalent{at}, observation_a = {PyBuiltins.Str(obsA)}, observation_b = {PyBuiltins.Str(obsB)}");
                    CompareInfos(infoA, infoB, infoComparison, "resetting", at);
                    if (!skipRender)
                        Assert(RendersEqual(envA, envB), $"resetting render is not equivalent{at}");
                }
            }
        }

        /// <summary>The info comparison of one reset or step, in the mode asked for, with Gymnasium's messages.</summary>
        /// <param name="infoA">A's info.</param>
        /// <param name="infoB">B's info.</param>
        /// <param name="mode">The <c>info_comparison</c> mode.</param>
        /// <param name="what"><c>resetting</c> or <c>stepping</c>.</param>
        /// <param name="at">The <c> in step = N</c> part of the message, empty for the first reset.</param>
        /// <exception cref="AssertionError">Thrown when the infos do not compare.</exception>
        /// <exception cref="KeyError">Thrown in <c>superset</c> mode when B lacks a key of A.</exception>
        private static void CompareInfos(Info infoA, Info infoB, string mode, string what, string at) {
            switch (mode) {
                case "equivalence":
                    Assert(EnvChecker.DataEquivalence(infoA, infoB), $"{what} info is not equivalent{at}, info_a = {PyBuiltins.Str(infoA)}, info_b = {PyBuiltins.Str(infoB)}");
                    break;
                case "superset":
                    foreach (string key in infoA.Keys) {
                        Assert(EnvChecker.DataEquivalence(infoA[key], infoB[key]),
                            $"{what} info is not a superset{at}, key {key} present in info_a with value = {PyBuiltins.Str(infoA[key])}, in info_b with value = {PyBuiltins.Str(infoB[key])}");
                    }

                    break;
                case "keys-equivalence":
                    Assert(infoA.Keys.Count == infoB.Keys.Count && infoA.Keys.All(infoB.ContainsKey),
                        $"{what} info keys are not equivalent{at}, info_a's keys are {DictKeys(infoA)}, info_b's keys are {DictKeys(infoB)}");
                    break;
                case "keys-superset":
                    // Gymnasium prints info_b.keys() - info_a.keys(), B's extra keys, although A's missing ones failed the check.
                    Assert(infoA.Keys.All(infoB.ContainsKey), $"{what} info keys are not a superset{at}, keys not present in info_b are: {SetRepr(infoB.Keys.Where(key => !infoA.ContainsKey(key)))}");
                    break;
            }
        }

        /// <summary>Renders both envs and compares the results (Python's <c>np.all(a.render() == b.render())</c>), releasing both frames.</summary>
        /// <param name="envA">The first env.</param>
        /// <param name="envB">The second env.</param>
        /// <returns><see langword="true"/> when the renders are equal.</returns>
        private static bool RendersEqual(IEnv envA, IEnv envB) {
            using RenderResult a = envA.Render();
            using RenderResult b = envB.Render();
            return a.Equals(b);
        }

        /// <summary>Python's <c>str(d.keys())</c>: <c>dict_keys(['a', 'b'])</c>.</summary>
        /// <param name="info">The dict.</param>
        /// <returns>The text.</returns>
        private static string DictKeys(Info info) => $"dict_keys({PyBuiltins.Repr(info.Keys.ToList())})";

        /// <summary>Python's <c>str(set)</c> of strings: <c>{'a', 'b'}</c>, or <c>set()</c> when empty; in the given order, where Python's follows hashing.</summary>
        /// <param name="items">The strings.</param>
        /// <returns>The text.</returns>
        private static string SetRepr(IEnumerable<string> items) {
            List<string> list = items.ToList();
            return list.Count == 0 ? "set()" : "{" + string.Join(", ", list.Select(PyBuiltins.Repr)) + "}";
        }

        /// <summary>Python's <c>assert condition, message</c>.</summary>
        /// <param name="condition">The asserted condition.</param>
        /// <param name="message">The message, empty for a bare <c>assert</c>.</param>
        /// <exception cref="AssertionError">Thrown when <paramref name="condition"/> is <see langword="false"/>.</exception>
        private static void Assert(bool condition, string message) {
            if (!condition)
                throw new AssertionError(message);
        }
    }
}
