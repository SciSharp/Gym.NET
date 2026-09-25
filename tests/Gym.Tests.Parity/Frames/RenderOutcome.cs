using System;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     What one <c>render()</c> call did: returned a frame, returned <c>None</c> (no state yet), or raised (a state pygame
    ///     cannot draw, such as a NaN or a coordinate beyond Sint16). Extreme-state cells compare these whole.
    /// </summary>
    /// <param name="Frame">The frame, when one was returned.</param>
    /// <param name="Raised">The exception, when the call raised.</param>
    public sealed record RenderOutcome(Frame? Frame, RaisedError? Raised) {
        /// <summary>Whether the call returned <c>None</c> (<see langword="null"/> on the .NET side).</summary>
        public bool ReturnedNone => Frame is null && Raised is null;

        /// <summary>
        ///     Requires <paramref name="dotnet"/> to have done what <paramref name="python"/> did: the same pixels, both
        ///     <c>None</c>, or the same exception (type mapped per <see cref="ExceptionMap"/>, message verbatim).
        /// </summary>
        /// <param name="what">The render call, for the report (<c>render() at state [300, 0, 0, 0]</c>).</param>
        /// <param name="python">Gymnasium's outcome.</param>
        /// <param name="dotnet">Gym.NET's outcome.</param>
        /// <exception cref="ParityException">Thrown when they differ, with frame artifacts when both returned frames.</exception>
        public static void AssertSame(string what, RenderOutcome python, RenderOutcome dotnet) {
            if (python.Frame != null && dotnet.Frame != null) {
                FrameDiff.AssertSame(what, python.Frame, dotnet.Frame, "render outcome");
                return;
            }

            ByteContract.AssertSame(what, python.AsOutcome(), dotnet.AsOutcome(), field: "render");
        }

        /// <summary>
        ///     The outcome in <see cref="Outcome"/>'s terms, for comparing a raise or a <c>None</c>: a frame is described by
        ///     its size only (pixel comparisons go through <see cref="FrameDiff"/>, which reports them properly).
        /// </summary>
        /// <returns>The outcome.</returns>
        private Outcome AsOutcome() {
            if (Raised != null)
                return Outcome.Threw(Raised);
            return Outcome.Returned(Frame is null ? ValueFacts.NoneValue() : ValueFacts.FromText($"a {Frame.Height}x{Frame.Width} frame", "frame"));
        }

        /// <summary>Renders the outcome for a report.</summary>
        /// <returns><c>None</c>, <c>raised Type: message</c>, or the frame's size.</returns>
        public override string ToString() => Raised != null ? $"raised {Raised}" : Frame is null ? "None" : $"a {Frame.Height}x{Frame.Width} frame";

        /// <summary>Runs a render call and captures what it did.</summary>
        /// <param name="render">The call; <see langword="null"/> for <c>None</c>.</param>
        /// <returns>The outcome.</returns>
        public static RenderOutcome Capture(Func<Frame?> render) {
            try {
                return new RenderOutcome(render(), null);
            } catch (Exception exception) when (exception is not Microsoft.VisualStudio.TestTools.UnitTesting.UnitTestAssertException) {
                return new RenderOutcome(null, RaisedError.FromDotNet(exception));
            }
        }
    }
}
