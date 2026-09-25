using System;
using System.Collections.Generic;
using Gym.Tests.Parity.Oracle;

namespace Gym.Tests.Parity.Pair {
    /// <summary>
    ///     What one <c>reset()</c> or <c>step()</c> returned, described: Gymnasium's <c>(observation, info)</c> or
    ///     <c>(observation, reward, terminated, truncated, info)</c>.
    /// </summary>
    /// <param name="Observation">The observation.</param>
    /// <param name="Reward">The reward; <see langword="null"/> for a reset.</param>
    /// <param name="Terminated">The terminated flag; <see langword="null"/> for a reset.</param>
    /// <param name="Truncated">The truncated flag; <see langword="null"/> for a reset.</param>
    /// <param name="Info">The info dictionary.</param>
    public sealed record Transition(ValueFacts Observation, ValueFacts? Reward, ValueFacts? Terminated, ValueFacts? Truncated, ValueFacts Info) {
        /// <summary>The empty info dictionary, Gymnasium's <c>{}</c>.</summary>
        public static readonly ValueFacts EmptyInfo = ValueFacts.FromDict(Array.Empty<KeyValuePair<ValueFacts, ValueFacts>>());

        /// <summary>Whether the episode ended with this step (terminated or truncated), so a trajectory stops here.</summary>
        public bool Ended => IsTrue(Terminated) || IsTrue(Truncated);

        /// <summary>Reads a described <c>(observation, info)</c> pair.</summary>
        /// <param name="value">The described tuple.</param>
        /// <returns>The transition.</returns>
        /// <exception cref="ArgumentException">Thrown when the value is not a 2-tuple.</exception>
        public static Transition FromReset(ValueFacts value) {
            if (value.Kind != FactKind.Seq || value.Items.Count != 2)
                throw new ArgumentException($"reset() must return (observation, info); got {value.Render()}.", nameof(value));
            return new Transition(value.Items[0], null, null, null, value.Items[1]);
        }

        /// <summary>Reads a described 5-tuple.</summary>
        /// <param name="value">The described tuple.</param>
        /// <returns>The transition.</returns>
        /// <exception cref="ArgumentException">Thrown when the value is not a 5-tuple.</exception>
        public static Transition FromStep(ValueFacts value) {
            if (value.Kind != FactKind.Seq || value.Items.Count != 5)
                throw new ArgumentException($"step() must return (observation, reward, terminated, truncated, info); got {value.Render()}.", nameof(value));
            return new Transition(value.Items[0], value.Items[1], value.Items[2], value.Items[3], value.Items[4]);
        }

        /// <summary>Whether a described flag is a true boolean (<c>True</c> or <c>np.True_</c>, or a 0-d array holding it).</summary>
        /// <param name="flag">The flag, or <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when it is set.</returns>
        private static bool IsTrue(ValueFacts? flag) => flag?.AsScalar() is { Kind: FactKind.Bool, Boolean: true };
    }

    /// <summary>The result of one call on one side: a <see cref="Pair.Transition"/>, or the exception the call raised.</summary>
    public sealed class CallResult {
        /// <summary>Creates a result; exactly one argument is non-null.</summary>
        /// <param name="transition">What the call returned.</param>
        /// <param name="raised">What it raised.</param>
        private CallResult(Transition? transition, RaisedError? raised) {
            Transition = transition;
            Raised = raised;
        }

        /// <summary>What the call returned, or <see langword="null"/> when it raised.</summary>
        public Transition? Transition { get; }

        /// <summary>What the call raised, or <see langword="null"/> when it returned.</summary>
        public RaisedError? Raised { get; }

        /// <summary>A call that returned.</summary>
        /// <param name="transition">What it returned.</param>
        /// <returns>The result.</returns>
        public static CallResult Of(Transition transition) => new(transition, null);

        /// <summary>A call that raised.</summary>
        /// <param name="error">What it raised.</param>
        /// <returns>The result.</returns>
        public static CallResult Threw(RaisedError error) => new(null, error);

        /// <summary>Reads the outcome of a Python <c>reset</c> or <c>step</c> call.</summary>
        /// <param name="outcome">The outcome, whose value is the described tuple.</param>
        /// <param name="reset">Whether the call was <c>reset</c> (a 2-tuple) rather than <c>step</c> (a 5-tuple).</param>
        /// <returns>The result.</returns>
        public static CallResult FromOutcome(Outcome outcome, bool reset) {
            if (outcome.Raised != null)
                return Threw(outcome.Raised);
            return Of(reset ? Transition.FromReset(outcome.Value!) : Transition.FromStep(outcome.Value!));
        }

        /// <summary>This result as an <see cref="Outcome"/> of the whole tuple, for comparing raised exceptions.</summary>
        /// <returns>The outcome.</returns>
        public Outcome ToOutcome() => Raised != null ? Outcome.Threw(Raised) : Outcome.Returned(Transition!.Observation);
    }
}
