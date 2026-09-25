#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Text;
using System.Text.RegularExpressions;
using Gym.Exceptions;
using Gym.Utils;

namespace Gym.Envs.Registration {
    /// <summary>
    ///     Port of Gymnasium's <c>EnvCreator</c> protocol (<c>gymnasium/envs/registration.py:47</c> @ 9e04324): a function
    ///     that builds an env from keyword arguments, one of the two forms of <see cref="EnvSpec.EntryPoint"/> (the other is
    ///     a <c>"Namespace:Class"</c> string).
    /// </summary>
    /// <param name="kwargs">The keyword arguments, Python's <c>**kwargs</c>, by name.</param>
    /// <returns>The new env.</returns>
    public delegate IEnv EnvCreator(IReadOnlyDictionary<string, object?> kwargs);

    /// <summary>
    ///     Port of <c>WrapperSpec</c> (<c>gymnasium/envs/registration.py:59</c> @ 9e04324): the record of one wrapper applied on
    ///     top of an env, which <see cref="EnvSpec.AdditionalWrappers"/> accumulates as <see cref="Wrapper{TWObs, TWAct, TObs, TAct}.Spec"/>
    ///     walks up a stack. <c>make</c> (P3) re-creates the wrapper from it.
    /// </summary>
    /// <param name="Name">The wrapper's class name (<c>Wrapper.ClassName()</c>).</param>
    /// <param name="EntryPoint">Where the wrapper lives, <c>"Namespace:Class"</c> (Python's <c>"module:Class"</c>).</param>
    /// <param name="Kwargs">
    ///     The wrapper's constructor arguments, recorded by <see cref="IRecordConstructorArgs"/>; <see langword="null"/> for a
    ///     wrapper that does not record them, which <c>make</c> then cannot re-create.
    /// </param>
    /// <remarks>Equality is the dataclass's: every field, the kwargs as Python dicts compare.</remarks>
    public sealed record WrapperSpec(string Name, string EntryPoint, Info? Kwargs) {
        /// <summary>
        ///     Port of the <c>Wrapper.wrapper_spec(**kwargs)</c> classmethod (<c>gymnasium/core.py:421</c>): the spec of a
        ///     wrapper type, named by its class and entry point, with the given constructor arguments.
        /// </summary>
        /// <param name="wrapperType">The wrapper's type (open or closed generic).</param>
        /// <param name="kwargs">The constructor arguments to record.</param>
        /// <returns>The spec.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="wrapperType"/> is <see langword="null"/>.</exception>
        public static WrapperSpec For(Type wrapperType, Info? kwargs) {
            if (wrapperType is null)
                throw new ArgumentNullException(nameof(wrapperType));
            string name = PyAttributes.PyTypeName(wrapperType);
            return new WrapperSpec(name, $"{wrapperType.Namespace}:{name}", kwargs);
        }

        /// <summary><c>copy.deepcopy(wrapper_spec)</c>: the same spec with its own copy of the kwargs.</summary>
        /// <returns>The copy.</returns>
        public WrapperSpec DeepCopy() => new(Name, EntryPoint, Kwargs?.DeepCopy());

        /// <summary>The dataclass equality: the same name, entry point and kwargs (compared as Python dicts).</summary>
        /// <param name="other">The other spec; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when all fields are equal.</returns>
        public bool Equals(WrapperSpec? other) {
            if (other is null)
                return false;
            if (ReferenceEquals(this, other))
                return true;
            return string.Equals(Name, other.Name, StringComparison.Ordinal) && string.Equals(EntryPoint, other.EntryPoint, StringComparison.Ordinal)
                   && (Kwargs is null ? other.Kwargs is null : other.Kwargs is not null && Kwargs.Equals(other.Kwargs));
        }

        /// <summary>A hash consistent with <see cref="Equals(WrapperSpec)"/> (the kwargs contribute their count).</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(StringComparer.Ordinal.GetHashCode(Name), StringComparer.Ordinal.GetHashCode(EntryPoint), Kwargs?.Count ?? -1);

        /// <summary>The dataclass repr: <c>WrapperSpec(name='TimeLimit', entry_point='Gym.Wrappers:TimeLimit', kwargs={'max_episode_steps': 10})</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"WrapperSpec(name={PyBuiltins.Repr(Name)}, entry_point={PyBuiltins.Repr(EntryPoint)}, kwargs={PyBuiltins.Repr(Kwargs)})";
    }

    /// <summary>
    ///     Port of <c>EnvSpec</c> (<c>gymnasium/envs/registration.py:73</c> @ 9e04324): everything <c>make</c> needs to build an
    ///     env and the wrappers it adds (time limit, order enforcing, env checker), plus the wrappers applied since. An env
    ///     built by <c>make</c> carries it as <see cref="IEnv.Spec"/>; a wrapper's <c>Spec</c> is a copy extended with the
    ///     wrapper, so editing one never changes another.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like the dataclass, every field is settable, and <see cref="Namespace"/>, <see cref="Name"/> and
    ///     <see cref="Version"/> are parsed from <see cref="Id"/> once, at construction (<c>__post_init__</c>): assigning
    ///     <see cref="Id"/> later does not re-parse them, and neither does a <c>with</c> expression, as editing a
    ///     deep-copied spec in Python does not.
    ///     </para>
    ///     <para>
    ///     Equality is the dataclass's, over every field in declaration order, <see cref="Kwargs"/> compared as Python dicts.
    ///     <c>to_json</c>, <c>from_json</c> and <c>make</c> come with the registry (P3).
    ///     </para>
    /// </remarks>
    public sealed record EnvSpec {
        /// <summary>
        ///     Creates a spec; the namespace, name and version are parsed from <paramref name="id"/>, as
        ///     <c>__post_init__</c> does.
        /// </summary>
        /// <param name="id">The id <c>make</c> takes: <c>[namespace/](env-name)[-v(version)]</c>, for example <c>"CartPole-v1"</c>.</param>
        /// <param name="entryPoint">
        ///     The env's location, a <c>"Namespace:Class"</c> string, or an <see cref="EnvCreator"/>; <see langword="null"/> when
        ///     the spec only describes an env.
        /// </param>
        /// <param name="rewardThreshold">The return at which the task counts as solved, or <see langword="null"/>.</param>
        /// <param name="nondeterministic">Whether the same seed and actions can still give different observations.</param>
        /// <param name="maxEpisodeSteps">The step limit a <c>TimeLimit</c> enforces, or <see langword="null"/> for none.</param>
        /// <param name="orderEnforce">Whether <c>make</c> adds <c>OrderEnforcing</c> (reset before step and render).</param>
        /// <param name="disableEnvChecker">Whether <c>make</c> leaves out the passive env checker.</param>
        /// <param name="kwargs">The keyword arguments the env is constructed with; <see langword="null"/> is an empty dict (the dataclass's <c>default_factory</c>).</param>
        /// <param name="additionalWrappers">The wrappers applied on top, in order; <see langword="null"/> is none.</param>
        /// <param name="vectorEntryPoint">The location of a vectorized implementation, or <see langword="null"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="id"/> is <see langword="null"/>.</exception>
        /// <exception cref="Error">Thrown for a malformed id, with Gymnasium's message (see <see cref="EnvRegistry.ParseEnvId"/>).</exception>
        public EnvSpec(string id, object? entryPoint = null, double? rewardThreshold = null, bool nondeterministic = false, long? maxEpisodeSteps = null,
            bool orderEnforce = true, bool disableEnvChecker = false, Info? kwargs = null, IEnumerable<WrapperSpec>? additionalWrappers = null,
            object? vectorEntryPoint = null) {
            Id = id ?? throw new ArgumentNullException(nameof(id));
            EntryPoint = entryPoint;
            RewardThreshold = rewardThreshold;
            Nondeterministic = nondeterministic;
            MaxEpisodeSteps = maxEpisodeSteps;
            OrderEnforce = orderEnforce;
            DisableEnvChecker = disableEnvChecker;
            Kwargs = kwargs ?? new Info();
            AdditionalWrappers = additionalWrappers is null ? Array.Empty<WrapperSpec>() : new List<WrapperSpec>(additionalWrappers).ToArray();
            VectorEntryPoint = vectorEntryPoint;
            (Namespace, Name, Version) = EnvRegistry.ParseEnvId(id);
        }

        /// <summary>The id, <c>[namespace/](env-name)[-v(version)]</c>.</summary>
        public string Id { get; set; }

        /// <summary>A <c>"Namespace:Class"</c> string, an <see cref="EnvCreator"/>, or <see langword="null"/>.</summary>
        public object? EntryPoint { get; set; }

        /// <summary>The return at which the task counts as solved, or <see langword="null"/>.</summary>
        public double? RewardThreshold { get; set; }

        /// <summary>Whether the same seed and actions can still give different observations.</summary>
        public bool Nondeterministic { get; set; }

        /// <summary>The step limit a <c>TimeLimit</c> enforces, or <see langword="null"/> for none.</summary>
        public long? MaxEpisodeSteps { get; set; }

        /// <summary>Whether <c>make</c> adds <c>OrderEnforcing</c>; <see langword="true"/> by default.</summary>
        public bool OrderEnforce { get; set; }

        /// <summary>Whether <c>make</c> leaves out the passive env checker.</summary>
        public bool DisableEnvChecker { get; set; }

        /// <summary>The keyword arguments the env is constructed with, a Python dict.</summary>
        public Info Kwargs { get; set; }

        /// <summary>The namespace parsed from <see cref="Id"/> at construction, or <see langword="null"/>.</summary>
        public string? Namespace { get; set; }

        /// <summary>The name parsed from <see cref="Id"/> at construction.</summary>
        public string Name { get; set; }

        /// <summary>The version parsed from <see cref="Id"/> at construction, or <see langword="null"/> for an unversioned id.</summary>
        public long? Version { get; set; }

        /// <summary>The wrappers applied on top of the env, in order (Python's tuple: replace it, do not edit it).</summary>
        public IReadOnlyList<WrapperSpec> AdditionalWrappers { get; set; }

        /// <summary>The location of a vectorized implementation (a string or a creator), or <see langword="null"/>.</summary>
        public object? VectorEntryPoint { get; set; }

        /// <summary>
        ///     <c>copy.deepcopy(spec)</c>: a spec with the same fields whose kwargs and wrapper specs are copies, so a wrapper
        ///     can edit the copy (<c>TimeLimit</c> sets <see cref="MaxEpisodeSteps"/>) without touching the original.
        /// </summary>
        /// <returns>The copy; entry points are shared, as Python's deepcopy shares functions.</returns>
        public EnvSpec DeepCopy() {
            var wrappers = new WrapperSpec[AdditionalWrappers.Count];
            for (int i = 0; i < wrappers.Length; i++)
                wrappers[i] = AdditionalWrappers[i].DeepCopy();
            // The copy constructor keeps Namespace/Name/Version as they are, rather than re-parsing Id.
            return this with { Kwargs = Kwargs.DeepCopy(), AdditionalWrappers = wrappers };
        }

        /// <summary>
        ///     Port of <c>EnvSpec.pprint</c>: the spec as <c>key=value</c> lines, the id always and the other fields when they
        ///     differ from their defaults (or always with <paramref name="printAll"/>).
        /// </summary>
        /// <param name="disablePrint">Return the text instead of writing it to the console.</param>
        /// <param name="includeEntryPoints">Also print the entry points of the env and of each wrapper.</param>
        /// <param name="printAll">Print every field, including those at their default values.</param>
        /// <returns>The text when <paramref name="disablePrint"/> is set, otherwise <see langword="null"/> (it was printed).</returns>
        public string? Pprint(bool disablePrint = false, bool includeEntryPoints = false, bool printAll = false) {
            var output = new StringBuilder($"id={Id}");
            if (printAll || includeEntryPoints)
                output.Append($"\nentry_point={PyBuiltins.Str(EntryPoint)}");

            if (printAll || RewardThreshold is not null)
                output.Append($"\nreward_threshold={PyBuiltins.Str(RewardThreshold)}");
            if (printAll || Nondeterministic)
                output.Append($"\nnondeterministic={PyBuiltins.Str(Nondeterministic)}");

            if (printAll || MaxEpisodeSteps is not null)
                output.Append($"\nmax_episode_steps={PyBuiltins.Str(MaxEpisodeSteps)}");
            if (printAll || !OrderEnforce)
                output.Append($"\norder_enforce={PyBuiltins.Str(OrderEnforce)}");
            if (printAll || DisableEnvChecker)
                output.Append($"\ndisable_env_checker={PyBuiltins.Str(DisableEnvChecker)}");

            if (printAll || AdditionalWrappers.Count > 0) {
                var wrapperOutput = new List<string>();
                foreach (WrapperSpec wrapperSpec in AdditionalWrappers) {
                    if (includeEntryPoints)
                        wrapperOutput.Add($"\n\tname={wrapperSpec.Name}, entry_point={wrapperSpec.EntryPoint}, kwargs={PyBuiltins.Str(wrapperSpec.Kwargs)}");
                    else
                        wrapperOutput.Add($"\n\tname={wrapperSpec.Name}, kwargs={PyBuiltins.Str(wrapperSpec.Kwargs)}");
                }

                if (wrapperOutput.Count == 0)
                    output.Append("\nadditional_wrappers=[]");
                else
                    output.Append($"\nadditional_wrappers=[{string.Join(",", wrapperOutput)}\n]");
            }

            if (disablePrint)
                return output.ToString();
            Console.WriteLine(output.ToString());
            return null;
        }

        /// <summary>The dataclass equality: every field equal, in declaration order, <see cref="Kwargs"/> as Python dicts.</summary>
        /// <param name="other">The other spec; may be <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when all fields are equal.</returns>
        public bool Equals(EnvSpec? other) {
            if (other is null)
                return false;
            if (ReferenceEquals(this, other))
                return true;
            if (!string.Equals(Id, other.Id, StringComparison.Ordinal) || !PyEquality.AreEqual(EntryPoint, other.EntryPoint) || RewardThreshold != other.RewardThreshold
                || Nondeterministic != other.Nondeterministic || MaxEpisodeSteps != other.MaxEpisodeSteps || OrderEnforce != other.OrderEnforce
                || DisableEnvChecker != other.DisableEnvChecker || !Kwargs.Equals(other.Kwargs) || !string.Equals(Namespace, other.Namespace, StringComparison.Ordinal)
                || !string.Equals(Name, other.Name, StringComparison.Ordinal) || Version != other.Version || AdditionalWrappers.Count != other.AdditionalWrappers.Count
                || !PyEquality.AreEqual(VectorEntryPoint, other.VectorEntryPoint))
                return false;
            for (int i = 0; i < AdditionalWrappers.Count; i++) {
                if (!AdditionalWrappers[i].Equals(other.AdditionalWrappers[i]))
                    return false;
            }

            return true;
        }

        /// <summary>A hash consistent with <see cref="Equals(EnvSpec)"/>: the id and the counts of kwargs and wrappers.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => HashCode.Combine(StringComparer.Ordinal.GetHashCode(Id), Kwargs.Count, AdditionalWrappers.Count);

        /// <summary>
        ///     The dataclass repr, fields in declaration order: <c>EnvSpec(id='CartPole-v1', entry_point=..., ...,
        ///     additional_wrappers=(), vector_entry_point=None)</c>. Wrapper and warning messages embed it.
        /// </summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            var wrappers = new StringBuilder("(");
            for (int i = 0; i < AdditionalWrappers.Count; i++) {
                if (i > 0)
                    wrappers.Append(", ");
                wrappers.Append(AdditionalWrappers[i]);
            }

            // Python's one-element tuple keeps its trailing comma.
            if (AdditionalWrappers.Count == 1)
                wrappers.Append(',');
            wrappers.Append(')');

            return $"EnvSpec(id={PyBuiltins.Repr(Id)}, entry_point={PyBuiltins.Repr(EntryPoint)}, reward_threshold={PyBuiltins.Repr(RewardThreshold)}, "
                   + $"nondeterministic={PyBuiltins.Repr(Nondeterministic)}, max_episode_steps={PyBuiltins.Repr(MaxEpisodeSteps)}, "
                   + $"order_enforce={PyBuiltins.Repr(OrderEnforce)}, disable_env_checker={PyBuiltins.Repr(DisableEnvChecker)}, kwargs={PyBuiltins.Repr(Kwargs)}, "
                   + $"namespace={PyBuiltins.Repr(Namespace)}, name={PyBuiltins.Repr(Name)}, version={PyBuiltins.Repr(Version)}, additional_wrappers={wrappers}, "
                   + $"vector_entry_point={PyBuiltins.Repr(VectorEntryPoint)})";
        }
    }

    /// <summary>
    ///     The module-level functions of <c>gymnasium/envs/registration.py</c> @ 9e04324. P1 ports the id grammar that
    ///     <see cref="EnvSpec"/> needs; the registry (<c>register</c>, <c>make</c>, <c>spec</c>, ...) joins it in P3 and is
    ///     re-exported by the top-level <c>Gymnasium</c> class.
    /// </summary>
    public static class EnvRegistry {
        /// <summary>
        ///     Port of <c>ENV_ID_RE</c>: <c>[namespace/](env-name)[-v(version)]</c>, matched against the whole id (Python's
        ///     <c>fullmatch</c>, hence <c>\A</c>/<c>\z</c> rather than <c>$</c>, which would also match before a final newline).
        /// </summary>
        /// <remarks>
        ///     .NET's <c>\w</c> and Python's agree on ASCII and on letters and digits; for rare non-ASCII marks the two Unicode
        ///     definitions can differ.
        /// </remarks>
        private static readonly Regex EnvIdRegex = new(@"\A(?:(?<namespace>[\w:-]+)/)?(?:(?<name>[\w:.-]+?))(?:-v(?<version>\d+))?\z", RegexOptions.CultureInvariant);

        /// <summary>
        ///     Port of <c>parse_env_id</c>: splits an id into its namespace, name and version, the inverse of
        ///     <see cref="GetEnvId"/>. <c>"ALE/Pong-v5"</c> is <c>("ALE", "Pong", 5)</c>; <c>"CartPole"</c> is
        ///     <c>(null, "CartPole", null)</c>.
        /// </summary>
        /// <param name="envId">The id.</param>
        /// <returns>The namespace (or <see langword="null"/>), the name, and the version (or <see langword="null"/>).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envId"/> is <see langword="null"/>.</exception>
        /// <exception cref="Error">
        ///     Thrown when the id does not match, with Gymnasium's message: <c>Malformed environment ID: {env_id}. (Currently
        ///     all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))</c>.
        /// </exception>
        /// <exception cref="OverflowException">Thrown for a version beyond <see langword="long"/>, which Python's unbounded int would hold.</exception>
        public static (string? Namespace, string Name, long? Version) ParseEnvId(string envId) {
            if (envId is null)
                throw new ArgumentNullException(nameof(envId));
            Match match = EnvIdRegex.Match(envId);
            if (!match.Success)
                throw new Error($"Malformed environment ID: {envId}. (Currently all IDs must be of the form [namespace/](env-name)-v(version). (namespace is optional))");
            Group ns = match.Groups["namespace"], name = match.Groups["name"], version = match.Groups["version"];
            return (ns.Success ? ns.Value : null, name.Value, version.Success ? ParseVersion(version.Value) : null);
        }

        /// <summary>
        ///     Port of <c>get_env_id</c>: joins a namespace, name and version into an id, the inverse of
        ///     <see cref="ParseEnvId"/>: <c>("ALE", "Pong", 5)</c> is <c>"ALE/Pong-v5"</c>.
        /// </summary>
        /// <param name="ns">The namespace, or <see langword="null"/>.</param>
        /// <param name="name">The name.</param>
        /// <param name="version">The version, or <see langword="null"/>.</param>
        /// <returns>The id.</returns>
        public static string GetEnvId(string? ns, string name, long? version) {
            string fullName = name;
            if (ns is not null)
                fullName = $"{ns}/{name}";
            if (version is not null)
                fullName = $"{fullName}-v{version.Value.ToString(CultureInfo.InvariantCulture)}";
            return fullName;
        }

        /// <summary>
        ///     Python's <c>int(version)</c> of the matched digits, which may be any Unicode decimal digits (<c>\d</c> matches
        ///     them in both languages, and <c>int</c> reads them).
        /// </summary>
        /// <param name="digits">The matched digits.</param>
        /// <returns>The version.</returns>
        /// <exception cref="OverflowException">Thrown when the value exceeds <see langword="long"/>.</exception>
        private static long ParseVersion(string digits) {
            BigInteger value = BigInteger.Zero;
            foreach (char digit in digits)
                value = value * 10 + (int)char.GetNumericValue(digit);
            return checked((long)value);
        }
    }
}
