using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity {
    /// <summary>
    ///     Marks a parity test that asserts Gymnasium's answer and fails today, because Gym.NET does not match that behavior
    ///     yet. Shorthand for <c>[TestCategory("OpenBugs")]</c>, NumSharp's category for the same purpose, with the ids of
    ///     the open defects attached.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The test's expectation is Gymnasium's, never today's Gym.NET output: when Gym.NET's fix lands, the same test passes
    ///     unchanged. Excluded from gated runs with <c>--filter "TestCategory!=OpenBugs"</c>; run the open gaps alone with
    ///     <c>--filter "TestCategory=OpenBugs"</c>, which prints each one's first divergence.
    ///     </para>
    ///     <para>
    ///     Lifecycle, as in NumSharp: a change fixes the defect, the test starts passing, then the attribute is removed so the
    ///     cell becomes a gate. Forgetting that last step leaves the fix unguarded, because gated runs still skip the test.
    ///     This is for real gaps only. A deliberate, documented deviation from Gymnasium is not an open bug.
    ///     </para>
    /// </remarks>
    [AttributeUsage(AttributeTargets.Class | AttributeTargets.Method, AllowMultiple = false, Inherited = true)]
    public sealed class OpenBugsAttribute : TestCategoryBaseAttribute {
        /// <summary>The category name, shared with NumSharp's suites so one filter works everywhere.</summary>
        public const string Category = "OpenBugs";

        /// <summary>Marks the test as a known gap.</summary>
        /// <param name="defects">
        ///     What has to land for the test to pass: ids from the defect register of docs/plans/GYMNASIUM_PORT_PLAN.md
        ///     (Appendix B, <c>"B-11"</c>) or its workstreams and phases (<c>"F6"</c>, <c>"P1"</c>).
        /// </param>
        public OpenBugsAttribute(params string[] defects) {
            Defects = defects;
        }

        /// <summary>The ids passed to the constructor, in order; printed when the test fails.</summary>
        public IReadOnlyList<string> Defects { get; }

        /// <inheritdoc />
        public override IList<string> TestCategories => new[] { Category };

        /// <summary>Finds the attribute on a test method, or on its class when the method has none.</summary>
        /// <param name="testClass">The test class.</param>
        /// <param name="methodName">The test method's name (<c>TestContext.TestName</c>).</param>
        /// <returns>The attribute, or <see langword="null"/> when the test is not an open bug.</returns>
        public static OpenBugsAttribute? Find(Type testClass, string? methodName) {
            MethodInfo? method = methodName is null
                ? null
                : testClass.GetMethods(BindingFlags.Public | BindingFlags.Instance).FirstOrDefault(m => m.Name == methodName);
            return method?.GetCustomAttribute<OpenBugsAttribute>() ?? testClass.GetCustomAttribute<OpenBugsAttribute>();
        }
    }
}
