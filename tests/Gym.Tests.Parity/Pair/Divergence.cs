using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Gym.Tests.Parity.Pair {
    /// <summary>One way in which the two sides' results of a call differ.</summary>
    /// <param name="Field">
    ///     The top-level field, in comparison order: <c>raised</c>, <c>observation</c>, <c>reward</c>, <c>terminated</c>,
    ///     <c>truncated</c>, <c>info</c>, <c>rng</c>, <c>action</c> or <c>frame</c>, or the name a direct comparison was given.
    ///     Self-tests assert on it.
    /// </param>
    /// <param name="Path">Where inside the field: <c>observation</c>, <c>info['lives']</c>, <c>rng['state']['state']</c>.</param>
    /// <param name="Detail">What differs, with both values: bits, decimal, ulp distance, element index, counts.</param>
    public sealed record FieldDivergence(string Field, string Path, string Detail);

    /// <summary>
    ///     The first call at which Gym.NET stopped matching Gymnasium, with everything needed to act on it: which call, every
    ///     field that differs (in a fixed order, so reports are stable), the actions that led there, and both sides' state
    ///     before and after.
    /// </summary>
    /// <remarks>
    ///     Only the first diverging call is reported. Later calls start from different states, so their differences are
    ///     consequences; the fields of that one call, though, are all listed, because together they tell a missing cast
    ///     (observation dtype only) from a wrong draw (rng and observation values).
    /// </remarks>
    public sealed class Divergence {
        /// <summary>What was compared, for example <c>gymnasium.make('CartPole-v1').unwrapped vs Gym.NET CartPoleEnv</c>.</summary>
        public required string Case { get; init; }

        /// <summary>The call whose results differ, for example <c>step #1: step(1)</c> or <c>reset(seed=0)</c>.</summary>
        public required string Call { get; init; }

        /// <summary>The trajectory position: 0 for the reset, n for the n-th step; <see langword="null"/> outside a trajectory.</summary>
        public int? Step { get; init; }

        /// <summary>Every field that differs, in comparison order; never empty.</summary>
        public required IReadOnlyList<FieldDivergence> Fields { get; init; }

        /// <summary>The actions of the trajectory up to and including the diverging call, as the Python side received them.</summary>
        public IReadOnlyList<string> Actions { get; init; } = Array.Empty<string>();

        /// <summary>Notes that qualify the comparison, for example a field one side cannot expose.</summary>
        public IReadOnlyList<string> Notes { get; init; } = Array.Empty<string>();

        /// <summary>Gymnasium's state before the call (for example <c>env.unwrapped.state</c>), when it can be read.</summary>
        public string? PythonStateBefore { get; init; }

        /// <summary>Gym.NET's state before the call, when it can be read.</summary>
        public string? DotNetStateBefore { get; init; }

        /// <summary>Gymnasium's state after the call, when it can be read.</summary>
        public string? PythonStateAfter { get; init; }

        /// <summary>Gym.NET's state after the call, when it can be read.</summary>
        public string? DotNetStateAfter { get; init; }

        /// <summary>Files written for this divergence (frame PNGs), also attached to the test result.</summary>
        public IReadOnlyList<string> Artifacts { get; init; } = Array.Empty<string>();

        /// <summary>The first differing field, the headline of the report.</summary>
        public FieldDivergence Primary => Fields[0];

        /// <summary>Whether any differing field is <paramref name="field"/>.</summary>
        /// <param name="field">A top-level field name.</param>
        /// <returns><see langword="true"/> when that field differs.</returns>
        public bool Has(string field) => Fields.Any(f => f.Field == field);

        /// <summary>Renders the full report, the message of <see cref="ParityException"/>.</summary>
        /// <returns>A multi-line report.</returns>
        public string Format() {
            var text = new StringBuilder();
            text.AppendLine($"Gymnasium parity divergence at {Call}");
            text.AppendLine($"  case:    {Case}");
            if (Step is { } step)
                text.AppendLine($"  step:    {step} (0 = reset)");
            if (Actions.Count > 0)
                text.AppendLine($"  actions: {FormatActions(Actions)}");
            text.AppendLine("  differs (python vs dotnet):");
            foreach (FieldDivergence field in Fields)
                text.AppendLine($"    {field.Path}: {field.Detail}");
            foreach (string note in Notes)
                text.AppendLine($"  note:    {note}");
            AppendState(text, "python before", PythonStateBefore);
            AppendState(text, "dotnet before", DotNetStateBefore);
            AppendState(text, "python after ", PythonStateAfter);
            AppendState(text, "dotnet after ", DotNetStateAfter);
            foreach (string artifact in Artifacts)
                text.AppendLine($"  artifact: {artifact}");
            return text.ToString();
        }

        /// <inheritdoc />
        public override string ToString() => Format();

        /// <summary>Lists the actions, eliding the middle of a long trajectory: the tail is where the divergence is.</summary>
        /// <param name="actions">The actions in order.</param>
        /// <returns>At most the first 5 and the last 20, with the count of the elided ones.</returns>
        private static string FormatActions(IReadOnlyList<string> actions) {
            if (actions.Count <= 25)
                return "[" + string.Join(", ", actions) + "]";
            return "[" + string.Join(", ", actions.Take(5)) + $", ... {actions.Count - 25} more ..., " + string.Join(", ", actions.Skip(actions.Count - 20)) + "]";
        }

        /// <summary>Appends one state line when the state is known.</summary>
        /// <param name="text">The report.</param>
        /// <param name="label">The line's label.</param>
        /// <param name="state">The state, or <see langword="null"/> to skip the line.</param>
        private static void AppendState(StringBuilder text, string label, string? state) {
            if (state != null)
                text.AppendLine($"  {label}: {state}");
        }
    }

    /// <summary>
    ///     Thrown by the harness at the first divergence from Gymnasium. Its message is the full <see cref="Divergence"/>
    ///     report, so an unhandled one fails the test with everything needed to reproduce and diagnose it; self-tests catch it
    ///     to assert on <see cref="Divergence"/>.
    /// </summary>
    public sealed class ParityException : Exception {
        /// <summary>Wraps a divergence.</summary>
        /// <param name="divergence">What differs.</param>
        public ParityException(Divergence divergence) : base(divergence.Format()) {
            Divergence = divergence;
        }

        /// <summary>The structured report.</summary>
        public Divergence Divergence { get; }
    }
}
