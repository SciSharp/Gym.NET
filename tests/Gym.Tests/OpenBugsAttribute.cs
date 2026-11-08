using System;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests {
    /// <summary>
    ///     Marks a unit test that asserts the reference behavior (Gymnasium's, NumPy's or Python's) and fails today, because
    ///     Gym.NET or NumSharp does not provide it yet. Shorthand for <c>[TestCategory("OpenBugs")]</c>, NumSharp's category
    ///     for the same purpose, with the ids of the open items attached; the twin of <c>Gym.Tests.Parity.OpenBugsAttribute</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Decision DR-01 (docs/plans/DESIGN_REVIEW.md): every NumSharp gap is documented as an N-item and proven by such a
    ///     test, so the gap is visible in every open-bugs run and the test turns green by itself the day NumSharp ships the
    ///     fix. Gated runs exclude these tests with <c>--filter "TestCategory!=OpenBugs"</c>; a plain <c>dotnet test</c> runs
    ///     them too, and they fail, which is NumSharp's convention.
    ///     </para>
    ///     <para>
    ///     Lifecycle: the fix lands, the test passes, then the attribute is removed so the test becomes a gate. Forgetting
    ///     the last step leaves the fix unguarded, because gated runs still skip the test. A deliberate, documented deviation
    ///     from the reference is not an open bug.
    ///     </para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false, Inherited = true)]
    public sealed class OpenBugsAttribute : TestCategoryBaseAttribute {
        /// <summary>The category name, shared with NumSharp's suites and the parity suite so one filter works everywhere.</summary>
        public const string Category = "OpenBugs";

        /// <summary>Marks the test as a known gap.</summary>
        /// <param name="items">
        ///     What has to land for the test to pass: N-items of docs/plans/COMPLETION_PLAN.md §7 (<c>"N-1"</c>), ids of the
        ///     port plan's defect register (<c>"B-11"</c>) or its phases (<c>"P2"</c>).
        /// </param>
        public OpenBugsAttribute(params string[] items) {
            Items = items;
        }

        /// <summary>The ids passed to the constructor, in order.</summary>
        public IReadOnlyList<string> Items { get; }

        /// <inheritdoc />
        public override IList<string> TestCategories => new[] { Category };
    }
}
