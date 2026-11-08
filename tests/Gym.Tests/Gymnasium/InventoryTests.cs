#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Gymnasium {
    /// <summary>
    ///     The API inventory (port plan §14.6, completion plan P0r.4): every public name Gymnasium exports, as listed by
    ///     <c>tools/parity/inventory.py</c>, is either present in Gym.NET under the port plan's §3.8 naming or listed in
    ///     <c>inventory.expected-missing.txt</c> with the phase that ports it. The list is a ratchet: a name that appears
    ///     fails the test until it is removed from the list, and a name that disappears fails it until it is listed, so the
    ///     list always says exactly what is left. 2.0 needs it empty but for what is out of scope.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Presence is by name only, not fidelity: the parity suites judge behavior. A Gymnasium class is present when a
    ///     public type of that name (generic arity aside), or one carrying <c>[PyName("…")]</c> with it, lives in the
    ///     module's namespace; a function or value when a public static member of a public type in that namespace has the
    ///     name with underscores dropped, compared case-insensitively (<c>np_random</c> is <c>NpRandom</c>, <c>flatdim</c>
    ///     may be <c>FlatDim</c>), or carries <c>[PyName("…")]</c> with it.
    ///     </para>
    ///     <para>
    ///     The inventory and the list are embedded resources (<c>Gym.Tests.csproj</c>), so the test reads no files at run
    ///     time; regenerate the inventory with <c>python tools/parity/inventory.py --write</c> after a re-pin.
    ///     </para>
    /// </remarks>
    [TestClass]
    public class InventoryTests {
        /// <summary>
        ///     Gymnasium modules mapped to the namespaces their names live in (port plan §3.8), longest prefix first. The
        ///     top-level <c>gymnasium</c> functions and <c>gymnasium.envs.registration</c> sit on the static class
        ///     <c>Gymnasium</c> in <c>Gym</c> (D16), its types in <c>Gym.Envs.Registration</c>.
        /// </summary>
        private static readonly (string Module, string[] Namespaces)[] NamespaceMap = {
            ("gymnasium.envs.registration", new[] { "Gym", "Gym.Envs.Registration" }),
            ("gymnasium.envs.classic_control", new[] { "Gym.Environments.Envs.Classic" }),
            ("gymnasium.envs.toy_text", new[] { "Gym.Environments.Envs.ToyText" }),
            ("gymnasium.envs.box2d", new[] { "Gym.Environments.Envs.Box2D" }),
            ("gymnasium.envs.mujoco", new[] { "Gym.Environments.Envs.MuJoCo" }),
            ("gymnasium.envs", new[] { "Gym.Environments.Envs" }),
            ("gymnasium.core", new[] { "Gym.Envs" }),
            ("gymnasium.spaces", new[] { "Gym.Spaces" }),
            ("gymnasium.error", new[] { "Gym.Exceptions" }),
            ("gymnasium.logger", new[] { "Gym" }),
            ("gymnasium.wrappers.vector", new[] { "Gym.Wrappers.Vector" }),
            ("gymnasium.wrappers", new[] { "Gym.Wrappers" }),
            ("gymnasium.vector", new[] { "Gym.Vector" }),
            // Gym.NET's Logger owns the colouring its warnings use (Logger.Colorize), as gymnasium.logger imports it.
            ("gymnasium.utils.colorize", new[] { "Gym.Utils", "Gym" }),
            ("gymnasium.utils", new[] { "Gym.Utils" }),
            ("gymnasium.experimental.functional", new[] { "Gym.Experimental.Functional" }),
            ("gymnasium", new[] { "Gym" }),
        };

        /// <summary>MSTest's context, for the inventory summary.</summary>
        public TestContext TestContext { get; set; } = null!;

        /// <summary>
        ///     The names Gym.NET lacks are exactly the listed ones: none missing without a phase, none listed that is
        ///     already there.
        /// </summary>
        [TestMethod]
        public void MissingNames_AreExactlyTheListedOnes() {
            IReadOnlyList<InventoryEntry> entries = LoadInventory().Entries;
            Dictionary<string, string> listed = LoadExpectedMissing();
            ILookup<string, Type> types = PublicTypesByNamespace();

            var missing = entries.Where(e => !IsPresent(e, types)).Select(e => e.Key).ToHashSet(StringComparer.Ordinal);
            var unlisted = missing.Where(key => !listed.ContainsKey(key)).OrderBy(k => k, StringComparer.Ordinal).ToList();
            var arrived = listed.Keys.Where(key => !missing.Contains(key)).OrderBy(k => k, StringComparer.Ordinal).ToList();

            TestContext.WriteLine($"{entries.Count} Gymnasium names: {entries.Count - missing.Count} present, {missing.Count} missing.");
            foreach (var phase in listed.Where(l => missing.Contains(l.Key)).GroupBy(l => l.Value).OrderBy(g => g.Key, StringComparer.Ordinal))
                TestContext.WriteLine($"  {phase.Count(),4}  {phase.Key}");

            Assert.AreEqual(0, unlisted.Count,
                "Gymnasium names Gym.NET lacks that inventory.expected-missing.txt does not list (port them, or list each with its phase):" +
                Environment.NewLine + string.Join(Environment.NewLine, unlisted));
            Assert.AreEqual(0, arrived.Count,
                "Names inventory.expected-missing.txt lists that Gym.NET now has, or Gymnasium no longer exports (remove them from the list):" +
                Environment.NewLine + string.Join(Environment.NewLine, arrived));
        }

        /// <summary>The inventory was scanned from the Gymnasium commit <c>tools/parity/manifest.json</c> pins, so it lists today's names.</summary>
        [TestMethod]
        public void Inventory_IsOfThePinnedCommit() {
            using JsonDocument manifest = JsonDocument.Parse(Resource("Gym.Tests.parity_manifest.json"));
            string pinned = manifest.RootElement.GetProperty("stack").GetProperty("gymnasium.commit").GetString()!;
            Assert.AreEqual(pinned, LoadInventory().Commit,
                "tools/parity/gymnasium_inventory.json is stale: run `python tools/parity/inventory.py --write` after the re-pin.");
        }

        /// <summary>Whether Gym.NET has a Gymnasium name under §3.8's naming.</summary>
        /// <param name="entry">The name.</param>
        /// <param name="types">Gym.NET's public types by namespace.</param>
        /// <returns><see langword="true"/> when present.</returns>
        private static bool IsPresent(InventoryEntry entry, ILookup<string, Type> types) {
            IEnumerable<Type> candidates = NamespacesOf(entry.Module).SelectMany(ns => types[ns]);
            if (entry.Kind == "class")
                return candidates.Any(t => TypeName(t) == entry.Name || PyName(t) == entry.Name);

            // Module functions and values are static members: PascalCase of the snake_case name, underscores dropped.
            string wanted = entry.Name.Replace("_", "", StringComparison.Ordinal);
            const BindingFlags statics = BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly;
            return candidates.Any(t => t.GetMembers(statics).Any(m =>
                m.MemberType is MemberTypes.Method or MemberTypes.Property or MemberTypes.Field &&
                (string.Equals(m.Name, wanted, StringComparison.OrdinalIgnoreCase) || PyName(m) == entry.Name)));
        }

        /// <summary>The namespaces a Gymnasium module's names live in: the longest mapped prefix's.</summary>
        /// <param name="module">The defining module, for example <c>gymnasium.spaces.box</c>.</param>
        /// <returns>The namespaces.</returns>
        /// <exception cref="InvalidOperationException">Thrown for a module outside the <c>gymnasium</c> package, which the map cannot place.</exception>
        private static string[] NamespacesOf(string module) {
            foreach ((string prefix, string[] namespaces) in NamespaceMap) {
                if (module == prefix || module.StartsWith(prefix + ".", StringComparison.Ordinal))
                    return namespaces;
            }

            throw new InvalidOperationException($"No namespace mapping for module '{module}'.");
        }

        /// <summary>Every public type of Gym.NET's product assemblies, by namespace.</summary>
        /// <returns>The lookup.</returns>
        private static ILookup<string, Type> PublicTypesByNamespace() {
            Assembly[] assemblies = { typeof(IEnv).Assembly, typeof(CartPoleEnv).Assembly, typeof(Surface).Assembly };
            return assemblies.SelectMany(a => a.GetExportedTypes()).ToLookup(t => t.Namespace ?? "", StringComparer.Ordinal);
        }

        /// <summary>A type's name without its generic arity (<c>Env`2</c> is <c>Env</c>).</summary>
        /// <param name="type">The type.</param>
        /// <returns>The name.</returns>
        private static string TypeName(Type type) {
            int tick = type.Name.IndexOf('`');
            return tick < 0 ? type.Name : type.Name.Substring(0, tick);
        }

        /// <summary>The Python name a member or type declares with <c>[PyName]</c>, read by attribute name so any target the attribute gains counts.</summary>
        /// <param name="member">The member or type.</param>
        /// <returns>The declared name, or <see langword="null"/>.</returns>
        private static string? PyName(MemberInfo member) {
            Attribute? attribute = member.GetCustomAttributes(inherit: false).OfType<Attribute>().FirstOrDefault(a => a.GetType().Name == "PyNameAttribute");
            return attribute?.GetType().GetProperty("Name")?.GetValue(attribute) as string;
        }

        /// <summary>Reads the embedded inventory.</summary>
        /// <returns>The commit it was scanned from and its entries.</returns>
        private static (string Commit, IReadOnlyList<InventoryEntry> Entries) LoadInventory() {
            using JsonDocument document = JsonDocument.Parse(Resource("Gym.Tests.gymnasium_inventory.json"));
            JsonElement root = document.RootElement;
            var entries = root.GetProperty("entries").EnumerateArray()
                .Select(e => new InventoryEntry(e.GetProperty("module").GetString()!, e.GetProperty("name").GetString()!, e.GetProperty("kind").GetString()!))
                .ToList();
            return (root.GetProperty("gymnasium_commit").GetString()!, entries);
        }

        /// <summary>Reads the embedded expected-missing list: one <c>module.name phase…</c> per line, <c>#</c> comments.</summary>
        /// <returns>Each listed name with its phase or reason.</returns>
        /// <exception cref="InvalidDataException">Thrown for a name listed twice, which would hide a removal.</exception>
        private static Dictionary<string, string> LoadExpectedMissing() {
            var listed = new Dictionary<string, string>(StringComparer.Ordinal);
            foreach (string raw in Resource("Gym.Tests.inventory.expected-missing.txt").Split('\n')) {
                string line = raw.Trim();
                if (line.Length == 0 || line.StartsWith('#'))
                    continue;
                int space = line.IndexOfAny(new[] { ' ', '\t' });
                string key = space < 0 ? line : line.Substring(0, space);
                string phase = space < 0 ? "(no phase)" : line.Substring(space).Trim();
                if (!listed.TryAdd(key, phase))
                    throw new InvalidDataException($"inventory.expected-missing.txt lists {key} twice.");
            }

            return listed;
        }

        /// <summary>An embedded resource's text.</summary>
        /// <param name="name">The resource's logical name.</param>
        /// <returns>Its UTF-8 text.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the project does not embed it.</exception>
        private static string Resource(string name) {
            using Stream stream = typeof(InventoryTests).Assembly.GetManifestResourceStream(name)
                                  ?? throw new InvalidOperationException($"The resource {name} is not embedded; see Gym.Tests.csproj.");
            using var reader = new StreamReader(stream);
            return reader.ReadToEnd();
        }

        /// <summary>One Gymnasium name of the inventory.</summary>
        /// <param name="Module">The defining module.</param>
        /// <param name="Name">The name.</param>
        /// <param name="Kind"><c>class</c>, <c>function</c>, <c>value</c>, or <c>unknown</c> for a lazily exported name.</param>
        private sealed record InventoryEntry(string Module, string Name, string Kind) {
            /// <summary>The name's key in the expected-missing list, <c>module.name</c>.</summary>
            public string Key => $"{Module}.{Name}";
        }
    }
}
