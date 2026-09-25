using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     The reference stack <c>tools/parity/manifest.json</c> pins: the Gymnasium commit and the versions of Python and of
    ///     the packages whose output the suite compares. A live oracle on another stack proves nothing, so every run is
    ///     checked against it (docs/plans/GYM_PARITY_INTEROP_TESTING.md §3).
    /// </summary>
    public sealed class StackManifest {
        /// <summary>Creates a manifest from already-parsed values; <see cref="Load"/> reads the file.</summary>
        /// <param name="stack">The pinned values by key.</param>
        /// <param name="referencePlatform">The runtime identifier whose results are authoritative.</param>
        private StackManifest(IReadOnlyDictionary<string, string> stack, string referencePlatform) {
            Stack = stack;
            ReferencePlatform = referencePlatform;
        }

        /// <summary>
        ///     The pinned values, keyed as <see cref="GymPythonSession"/> reads them from the live interpreter
        ///     (<c>numpy.version</c>, ...) plus <c>gymnasium.commit</c>, which it reads from the submodule's checkout. A value
        ///     ending in <c>.*</c> matches by prefix.
        /// </summary>
        public IReadOnlyDictionary<string, string> Stack { get; }

        /// <summary>
        ///     The runtime identifier (<c>win-x64</c>) of the platform whose results are authoritative (port plan §1.3 and
        ///     §14.9). Cells pinned to it report Inconclusive elsewhere instead of failing.
        /// </summary>
        public string ReferencePlatform { get; }

        /// <summary>Reads and validates the manifest.</summary>
        /// <param name="path">The manifest file, normally <see cref="RepositoryLayout.ManifestPath"/>.</param>
        /// <returns>The parsed manifest.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
        /// <exception cref="JsonException">Thrown when the file is not valid JSON.</exception>
        /// <exception cref="InvalidDataException">Thrown when <c>stack</c> or <c>referencePlatform</c> is missing or malformed.</exception>
        public static StackManifest Load(string path) {
            using var document = JsonDocument.Parse(File.ReadAllText(path));
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("stack", out JsonElement stack) || stack.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException($"{path} has no 'stack' object.");
            if (!root.TryGetProperty("referencePlatform", out JsonElement platform) || platform.ValueKind != JsonValueKind.String)
                throw new InvalidDataException($"{path} has no 'referencePlatform' string.");

            var values = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (JsonProperty property in stack.EnumerateObject()) {
                if (property.Value.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException($"{path}: stack.{property.Name} must be a string.");
                values[property.Name] = property.Value.GetString()!;
            }

            return new StackManifest(values, platform.GetString()!);
        }

        /// <summary>Whether <paramref name="actual"/> satisfies the pinned <paramref name="expected"/> value.</summary>
        /// <param name="expected">The pin: an exact value, or a prefix followed by <c>.*</c> (<c>3.12.*</c>).</param>
        /// <param name="actual">The live value.</param>
        /// <returns><see langword="true"/> when the values are equal, or when a <c>.*</c> pin's prefix (dot included) starts the live value.</returns>
        public static bool Matches(string expected, string actual) {
            if (expected.EndsWith(".*", StringComparison.Ordinal))
                return actual.StartsWith(expected.Substring(0, expected.Length - 1), StringComparison.Ordinal);
            return string.Equals(expected, actual, StringComparison.Ordinal);
        }
    }

    /// <summary>One checked property of the live stack.</summary>
    /// <param name="Key">What was checked, for example <c>numpy.version</c>.</param>
    /// <param name="Expected">The required value, as the report prints it.</param>
    /// <param name="Actual">The live value, or why it could not be read.</param>
    /// <param name="Matches">Whether the live value satisfies the requirement.</param>
    public sealed record StackItem(string Key, string Expected, string Actual, bool Matches);

    /// <summary>
    ///     The comparison of the embedded interpreter with the manifest, plus the structural checks no manifest entry could
    ///     express (Gymnasium is imported from the submodule, the submodule is unmodified, hash randomization is off).
    /// </summary>
    /// <param name="Items">Every check, in report order.</param>
    public sealed record StackReport(IReadOnlyList<StackItem> Items) {
        /// <summary>The checks that failed. Any entry makes the run fail, or report Inconclusive with <c>GYMNET_PARITY_ALLOW_STACK_DRIFT=1</c>.</summary>
        public IReadOnlyList<StackItem> Drift => Items.Where(i => !i.Matches).ToList();

        /// <summary>Renders every check as an aligned table, drifted rows marked, for failure messages and test output.</summary>
        /// <returns>One line per check.</returns>
        public string Format() {
            int width = Items.Count == 0 ? 0 : Items.Max(i => i.Key.Length);
            var text = new StringBuilder();
            foreach (StackItem item in Items) {
                text.Append(item.Matches ? "  ok     " : "  DRIFT  ")
                    .Append(item.Key.PadRight(width))
                    .Append("  ")
                    .Append(item.Actual);
                if (!item.Matches)
                    text.Append("  (expected ").Append(item.Expected).Append(')');
                text.AppendLine();
            }

            return text.ToString();
        }
    }
}
