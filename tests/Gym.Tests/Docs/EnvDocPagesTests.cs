using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Reflection;
using System.Text.Json;
using System.Text.RegularExpressions;
using Gym.Envs;
using Gym.Envs.Registration;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Docs {
    /// <summary>
    ///     Holds the docs site's environment pages (<c>docs/environments/&lt;family&gt;/&lt;page&gt;.md</c>, DR-104) to the
    ///     code, so an API or registration change that leaves a page wrong fails the unit gate (DR-105). The pages are
    ///     written by hand; these tests read the facts they state and compare each with its source:
    ///     the infobox's facts table with the registry and the env (ids, step limits, thresholds, both spaces, render
    ///     modes, fps and frame size, the make call, the class); the Arguments table with the env class's constructor
    ///     (names and defaults); the media with the example's recording (the files exist, the caption's seed, steps and
    ///     return are the episode's); and every code include with the compiled samples it shows.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The parsing follows the page template of round 35 (DR-111…DR-116): the infobox is an HTML table of
    ///     <c>&lt;tr&gt;&lt;td&gt;Label&lt;/td&gt;&lt;td&gt;…&lt;/td&gt;&lt;/tr&gt;</c> rows, the Arguments table puts Gymnasium's
    ///     keyword over the C# parameter and the type over the default (each in backticks, split by <c>&lt;br&gt;</c>), and
    ///     media are referenced as <c>media/…</c> beside the page. A page written another way fails here with the row it
    ///     could not read, which is the point: the test is the template's contract.
    ///     </para>
    ///     <para>
    ///     Footgun: the pages live outside the test project, found by walking up from the test's output folder to the
    ///     folder holding <c>Gym.NET.sln</c>; a test run from a copied output folder without the repository fails to find
    ///     them, loudly.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class EnvDocPagesTests {
        /// <summary>The repository's root: the nearest folder above the test's output that holds <c>Gym.NET.sln</c>.</summary>
        private static readonly Lazy<string> Root = new(() => {
            for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent) {
                if (File.Exists(Path.Combine(dir.FullName, "Gym.NET.sln")))
                    return dir.FullName;
            }
            throw new InvalidOperationException($"no Gym.NET.sln above {AppContext.BaseDirectory}: the env pages are read from the repository");
        });

        /// <summary>Every env page: the Markdown files under <c>docs/environments</c> but the section's index.</summary>
        /// <returns>The pages' full paths, in a stable order.</returns>
        private static IReadOnlyList<string> Pages() =>
            Directory.EnumerateFiles(Path.Combine(Root.Value, "docs", "environments"), "*.md", SearchOption.AllDirectories)
                .Where(p => !string.Equals(Path.GetFileName(p), "index.md", StringComparison.OrdinalIgnoreCase))
                .OrderBy(p => p, StringComparer.Ordinal)
                .ToList();

        /// <summary>The infobox's facts table of a page: each row's label and its value cell's HTML.</summary>
        /// <param name="page">The page's Markdown.</param>
        /// <returns>The rows by label.</returns>
        /// <exception cref="AssertFailedException">The page has no infobox.</exception>
        private static Dictionary<string, string> Facts(string page) {
            Match box = Regex.Match(page, "<aside class=\"gn-infobox\".*?</aside>", RegexOptions.Singleline);
            Assert.IsTrue(box.Success, "no <aside class=\"gn-infobox\"> on the page");
            return Regex.Matches(box.Value, "<tr><td>([^<]+)</td><td>(.*?)</td></tr>", RegexOptions.Singleline)
                .ToDictionary(m => m.Groups[1].Value.Trim(), m => m.Groups[2].Value);
        }

        /// <summary>
        ///     The HTML-decoded text of every <c>&lt;code&gt;</c> span in a fragment, in order, without the <c>&lt;wbr&gt;</c>
        ///     break hints a page puts inside a long one.
        /// </summary>
        /// <param name="html">The fragment.</param>
        /// <returns>The code texts.</returns>
        /// <remarks>
        ///     A long make call (<c>Gymnasium.Make("MountainCarContinuous-v0")</c>) has no break opportunity before its id's
        ///     hyphen, so in the infobox's half-width column it pushed the table past the box; a <c>&lt;wbr&gt;</c> after
        ///     the parenthesis lets it wrap there, and the text compared is the call without it.
        /// </remarks>
        private static List<string> Codes(string html) =>
            Regex.Matches(html, "<code>(.*?)</code>")
                .Select(m => WebUtility.HtmlDecode(Regex.Replace(m.Groups[1].Value, "<wbr\\s*/?>", ""))).ToList();

        /// <summary>
        ///     The facts table against the registry and the env: every id and its step limit, the thresholds, both spaces
        ///     as Gymnasium prints them, the render modes, frame size and fps, the make call and the class.
        /// </summary>
        [TestMethod]
        public void FactsTable_MatchesTheRegistryAndTheEnv() {
            var pages = Pages();
            Assert.IsTrue(pages.Count > 0, "no env pages found");
            foreach (string path in pages) {
                string page = File.ReadAllText(path);
                string name = Path.GetFileName(path);
                Dictionary<string, string> facts = Facts(page);

                // Ids: "<code>CartPole-v1</code> 500 steps", one per id the page covers.
                var ids = Regex.Matches(facts["Ids"], "<code>([A-Za-z0-9]+-v[0-9]+)</code>\\s*([0-9,]+) steps")
                    .Select(m => (Id: m.Groups[1].Value, Steps: long.Parse(m.Groups[2].Value, NumberStyles.AllowThousands, CultureInfo.InvariantCulture)))
                    .ToList();
                Assert.IsTrue(ids.Count > 0, $"{name}: the Ids row names no id");
                foreach (var (id, steps) in ids)
                    Assert.AreEqual(steps, Gym.Gymnasium.Spec(id).MaxEpisodeSteps, $"{name}: {id}'s step limit");
                string first = ids[0].Id;
                EnvSpec spec = Gym.Gymnasium.Spec(first);

                // Thresholds: "solved at 475 (v1), 195 (v0)" names each version's; "solved at 4,800" the first id's. A
                // threshold may be negative (MountainCar's −110, Acrobot's −100), written with the typographic minus.
                var byVersion = Regex.Matches(facts["Reward"], "([−-]?[0-9][0-9,.]*) \\((v[0-9]+)\\)")
                    .ToDictionary(m => m.Groups[2].Value, m => Number(m.Groups[1].Value));
                if (byVersion.Count > 0) {
                    foreach (var (id, _) in ids) {
                        string version = id[(id.LastIndexOf("-v", StringComparison.Ordinal) + 1)..];
                        if (byVersion.TryGetValue(version, out double threshold))
                            Assert.AreEqual(threshold, Gym.Gymnasium.Spec(id).RewardThreshold, $"{name}: {id}'s threshold");
                    }
                } else {
                    Match solved = Regex.Match(facts["Reward"], "solved at ([−-]?[0-9][0-9,.]*)");
                    if (solved.Success)
                        Assert.AreEqual(Number(solved.Groups[1].Value), spec.RewardThreshold, $"{name}: {first}'s threshold");
                    else
                        // A registered threshold the page leaves out is a fact missing from the box, not a pass.
                        Assert.IsNull(spec.RewardThreshold, $"{name}: {first} is registered with a threshold of {spec.RewardThreshold}, which the Reward row does not state");
                }

                using IEnv env = Gym.Gymnasium.Make(first);
                // Compared with runs of spaces collapsed: a Box prints its bounds with NumPy's column alignment
                // ("[-4.8               -inf ...]"), which HTML collapses on screen, so the page states the collapsed text.
                Assert.AreEqual(Collapse(env.ActionSpace.ToString()!), Collapse(Codes(facts["Action space"]).Single()), $"{name}: the action space");
                Assert.AreEqual(Collapse(env.ObservationSpace.ToString()!), Collapse(Codes(facts["Observation space"]).Single()), $"{name}: the observation space");

                // Render modes: "<code>human</code>, <code>rgb_array</code> · 600×400 · 50 fps".
                string modesRow = facts["Render modes"];
                var modes = Codes(modesRow);
                var registered = ((IEnumerable<object?>)env.Metadata["render_modes"]!).Select(m => m?.ToString()).ToList();
                CollectionAssert.AreEqual(registered, modes, $"{name}: the render modes");
                Match fps = Regex.Match(modesRow, "([0-9]+) fps");
                Assert.IsTrue(fps.Success, $"{name}: the Render modes row states no fps");
                Assert.AreEqual(long.Parse(fps.Groups[1].Value, CultureInfo.InvariantCulture), Convert.ToInt64(env.Metadata["render_fps"], CultureInfo.InvariantCulture), $"{name}: the fps");
                Match size = Regex.Match(modesRow, "([0-9]+)×([0-9]+)");
                if (size.Success && modes.Contains("rgb_array")) {
                    using IEnv drawn = Gym.Gymnasium.Make(first, new { render_mode = "rgb_array" });
                    drawn.Reset(seed: 0);
                    using RenderResult frame = drawn.Render();
                    NDArray pixels = (NDArray)frame;
                    Assert.AreEqual($"{size.Groups[1].Value}x{size.Groups[2].Value}", $"{pixels.shape[1]}x{pixels.shape[0]}", $"{name}: the frame size");
                }

                // Make: the call names the page's first id.
                Assert.AreEqual($"Gymnasium.Make(\"{first}\")", Codes(facts["Make"]).Single(), $"{name}: the make call");

                // Class: the first xref is the env class the first id is registered with; every xref names a type.
                var classes = Regex.Matches(facts["Class"], "xref:([A-Za-z0-9_.]+)").Select(m => m.Groups[1].Value).ToList();
                Assert.IsTrue(classes.Count > 0, $"{name}: the Class row names no type");
                Assert.AreEqual(((Type)spec.EntryPoint!).FullName, classes[0], $"{name}: the class");
                // A page without a vector entry point says so in words ("none of its own"): no xref to check then.
                facts.TryGetValue("Vector env", out string? vector);
                var vectorUids = vector is null ? [] : Regex.Matches(vector, "xref:([A-Za-z0-9_.]+)").Select(m => m.Groups[1].Value).ToList();
                foreach (string uid in classes.Concat(vectorUids))
                    Assert.IsNotNull(FindType(uid), $"{name}: no type {uid}");
                if (spec.VectorEntryPoint is Type vectorType && vector is not null)
                    StringAssert.Contains(vector, $"xref:{vectorType.FullName}", $"{name}: the vector env");
            }
        }

        /// <summary>
        ///     The Arguments table against the env class's constructor: each C# parameter exists, its Gymnasium keyword is
        ///     its snake_case name (how <c>Make</c>'s kwargs bind), and its stated default is the parameter's.
        /// </summary>
        [TestMethod]
        public void ArgumentsTable_MatchesTheConstructor() {
            foreach (string path in Pages()) {
                string page = File.ReadAllText(path);
                string name = Path.GetFileName(path);
                string first = Regex.Match(Facts(page)["Ids"], "<code>([A-Za-z0-9]+-v[0-9]+)</code>").Groups[1].Value;
                var envType = (Type)Gym.Gymnasium.Spec(first).EntryPoint!;
                var parameters = envType.GetConstructors().SelectMany(c => c.GetParameters()).GroupBy(p => p.Name!).ToDictionary(g => g.Key, g => g.First());

                string section = Regex.Match(page, "\n## Arguments\\s*\n(.*?)\n## ", RegexOptions.Singleline).Groups[1].Value;
                var rows = Regex.Matches(section, "^\\| (.+?) \\| (.+?) \\| (.+?) \\|\\s*$", RegexOptions.Multiline)
                    .Where(m => m.Groups[1].Value.Contains("<br>", StringComparison.Ordinal)).ToList();
                Assert.IsTrue(rows.Count > 0, $"{name}: no Arguments rows");
                foreach (Match row in rows) {
                    string[] names = row.Groups[1].Value.Split("<br>");
                    string python = names[0].Trim().Trim('`'), csharp = names[1].Trim().Trim('`');
                    Assert.IsTrue(parameters.TryGetValue(csharp, out ParameterInfo? parameter), $"{name}: {envType.Name} has no parameter {csharp}");
                    if (python != "(Gym.NET)")
                        Assert.AreEqual(SnakeCase(csharp), python, $"{name}: {csharp}'s Gymnasium keyword");
                    string stated = row.Groups[2].Value.Split("<br>").Last().Trim().Trim('`');
                    Assert.IsTrue(DefaultMatches(parameter!.DefaultValue, stated), $"{name}: {csharp}'s default is {parameter.DefaultValue ?? "null"}, the page says {stated}");
                }
            }
        }

        /// <summary>
        ///     The media against the recording: every file a page plays or shows exists beside it, and the infobox caption's
        ///     seed, steps and return are those of the episode JSON <c>--record</c> wrote (DR-112).
        /// </summary>
        [TestMethod]
        public void Media_ExistAndTheCaptionMatchesTheRecording() {
            foreach (string path in Pages()) {
                string page = File.ReadAllText(path);
                string name = Path.GetFileName(path);
                string folder = Path.GetDirectoryName(path)!;
                var media = Regex.Matches(page, "(?:src|poster|data-trace)=\"(media/[^\"]+)\"").Select(m => m.Groups[1].Value).Distinct().ToList();
                Assert.IsTrue(media.Count > 0, $"{name}: the page plays no recording");
                foreach (string file in media)
                    Assert.IsTrue(File.Exists(Path.Combine(folder, file)), $"{name}: {file} is missing");

                string trace = media.Single(m => m.EndsWith(".episode.json", StringComparison.Ordinal));
                using JsonDocument episode = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, trace)));
                Match caption = Regex.Match(page, "class=\"gn-infobox-caption\">(.*?)</p>", RegexOptions.Singleline);
                Assert.IsTrue(caption.Success, $"{name}: no infobox caption");
                Match numbers = Regex.Match(caption.Groups[1].Value, "seed ([0-9]+), ([0-9,]+) steps, return ([0-9,.\\-−]+)");
                Assert.IsTrue(numbers.Success, $"{name}: the caption states no seed, steps and return");
                Assert.AreEqual(episode.RootElement.GetProperty("seed").GetInt64(), long.Parse(numbers.Groups[1].Value, CultureInfo.InvariantCulture), $"{name}: the caption's seed");
                Assert.AreEqual(episode.RootElement.GetProperty("steps").GetInt64(), long.Parse(numbers.Groups[2].Value, NumberStyles.AllowThousands, CultureInfo.InvariantCulture), $"{name}: the caption's steps");
                double recorded = episode.RootElement.GetProperty("return").GetDouble();
                double stated = double.Parse(numbers.Groups[3].Value.Replace('−', '-').TrimEnd('.'), NumberStyles.Number, CultureInfo.InvariantCulture);
                // The caption rounds the return to the digits it shows.
                int digits = numbers.Groups[3].Value.TrimEnd('.').Contains('.') ? numbers.Groups[3].Value.TrimEnd('.').Split('.')[1].Length : 0;
                Assert.AreEqual(Math.Round(recorded, digits), stated, $"{name}: the caption's return");
            }
        }

        /// <summary>
        ///     Every code include against the compiled samples: the file exists; a <c>#Region</c> include names a region the
        ///     file has; a <c>?range=</c> include (the quick start: the file's usings, then a region's lines) takes only
        ///     <c>using</c> lines and blanks, then exactly the lines of one region, so moving a sample's lines fails here
        ///     before the page shows the wrong ones.
        /// </summary>
        [TestMethod]
        public void CodeIncludes_PointAtTheCompiledSamples() {
            foreach (string path in Pages()) {
                string page = File.ReadAllText(path);
                string name = Path.GetFileName(path);
                var includes = Regex.Matches(page, "\\[!code-csharp\\[\\]\\(([^)#?]+)(#[A-Za-z0-9_]+)?(\\?[^)]+)?\\)\\]").ToList();
                Assert.IsTrue(includes.Count > 0, $"{name}: no code includes");
                foreach (Match include in includes) {
                    string file = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(path)!, include.Groups[1].Value));
                    Assert.IsTrue(File.Exists(file), $"{name}: {include.Groups[1].Value} is missing");
                    string[] lines = File.ReadAllLines(file);
                    var regions = Regions(lines);
                    if (include.Groups[2].Success) {
                        Assert.IsTrue(regions.ContainsKey(include.Groups[2].Value[1..]), $"{name}: {Path.GetFileName(file)} has no region {include.Groups[2].Value[1..]}");
                        continue;
                    }
                    Match range = Regex.Match(include.Groups[3].Value, "range=([0-9]+)-([0-9]+),([0-9]+)-([0-9]+)");
                    Assert.IsTrue(range.Success, $"{name}: an include without a region states no two-part range: {include.Value}");
                    int[] r = Enumerable.Range(1, 4).Select(g => int.Parse(range.Groups[g].Value, CultureInfo.InvariantCulture)).ToArray();
                    for (int i = r[0]; i <= r[1]; i++)
                        Assert.IsTrue(lines[i - 1].Length == 0 || lines[i - 1].StartsWith("using ", StringComparison.Ordinal), $"{name}: line {i} of {Path.GetFileName(file)} is not a using: {lines[i - 1]}");
                    var shown = lines[(r[2] - 1)..r[3]];
                    Assert.IsTrue(regions.Values.Any(body => body.SequenceEqual(shown)), $"{name}: lines {r[2]}-{r[3]} of {Path.GetFileName(file)} are not a whole region");
                }
            }
        }

        /// <summary>The <c>#region</c> blocks of a source file: each name and the lines between its markers.</summary>
        /// <param name="lines">The file's lines.</param>
        /// <returns>The regions by name.</returns>
        private static Dictionary<string, string[]> Regions(string[] lines) {
            var regions = new Dictionary<string, string[]>();
            for (int i = 0; i < lines.Length; i++) {
                Match open = Regex.Match(lines[i], "^\\s*#region ([A-Za-z0-9_]+)\\s*$");
                if (!open.Success)
                    continue;
                int end = Array.FindIndex(lines, i + 1, l => Regex.IsMatch(l, "^\\s*#endregion"));
                regions[open.Groups[1].Value] = lines[(i + 1)..end];
            }
            return regions;
        }

        /// <summary>
        ///     A number as a page writes it: thousands separated by commas, and a negative one with the typographic minus
        ///     (U+2212) the pages use, or an ASCII hyphen.
        /// </summary>
        /// <param name="text">The number's text, such as <c>4,800</c> or <c>−110</c>.</param>
        /// <returns>Its value.</returns>
        /// <exception cref="FormatException">The text is not a number, which fails the test that read it.</exception>
        private static double Number(string text) =>
            double.Parse(text.Replace('−', '-'), NumberStyles.Number, CultureInfo.InvariantCulture);

        /// <summary>A text with every run of whitespace made one space, as HTML shows it.</summary>
        /// <param name="text">The text.</param>
        /// <returns>The collapsed text.</returns>
        private static string Collapse(string text) => Regex.Replace(text, "\\s+", " ");

        /// <summary>A C# parameter name in Gymnasium's snake_case: <c>suttonBartoReward</c> becomes <c>sutton_barto_reward</c>.</summary>
        /// <param name="name">The camelCase name.</param>
        /// <returns>The snake_case name.</returns>
        private static string SnakeCase(string name) => Regex.Replace(name, "(?<=[a-z0-9])([A-Z])", "_$1").ToLowerInvariant();

        /// <summary>Whether a parameter's default value is the one a page states in its Arguments table.</summary>
        /// <param name="value">The parameter's default.</param>
        /// <param name="stated">The page's text: <c>null</c>, <c>true</c>, a number, or a quoted string.</param>
        /// <returns>Whether they agree.</returns>
        private static bool DefaultMatches(object? value, string stated) => value switch {
            null => stated == "null",
            bool b => stated == (b ? "true" : "false"),
            string s => stated == $"\"{s}\"",
            IConvertible number => double.TryParse(stated, NumberStyles.Float, CultureInfo.InvariantCulture, out double d)
                && Convert.ToDouble(number, CultureInfo.InvariantCulture) == d,
            _ => false,
        };

        /// <summary>Finds a type by its full name in the loaded Gym assemblies.</summary>
        /// <param name="fullName">The type's full name (a DocFX uid).</param>
        /// <returns>The type, or null.</returns>
        private static Type? FindType(string fullName) {
            // Touching the env assembly loads it, so its types are searchable even before any env is made.
            _ = typeof(Gym.Environments.Envs.Classic.CartPoleEnv).Assembly;
            return AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(fullName)).FirstOrDefault(t => t is not null);
        }
    }
}
