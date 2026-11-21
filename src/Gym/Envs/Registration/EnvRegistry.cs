#nullable enable
using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using Gym.Exceptions;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;

namespace Gym.Envs.Registration {
    public static partial class EnvRegistry {
        /// <summary>Entry-point strings already resolved to their class, so <c>make</c> scans the loaded assemblies once per name.</summary>
        private static readonly ConcurrentDictionary<string, Type> ResolvedEntryPoints = new(StringComparer.Ordinal);

        /// <summary>
        ///     <see cref="CurrentNamespace"/>'s storage: one value per thread and async flow (DR-57), so a scope one thread
        ///     opens never renames what another thread registers. Needs no lock: no other flow ever sees this flow's value.
        /// </summary>
        private static readonly AsyncLocal<string?> CurrentNamespaceSlot = new();

        /// <summary>
        ///     The global registry (Python's <c>gymnasium.registry</c>): every id <c>make</c> knows, in registration order. Env
        ///     assemblies fill it from their module initializers (DR-12); <see cref="Register"/> adds to it.
        /// </summary>
        public static EnvSpecRegistry Registry { get; } = new();

        /// <summary>
        ///     The namespace <see cref="Register"/> puts ids in, overriding the id's own (Python's module global
        ///     <c>current_namespace</c>); <see langword="null"/> keeps each id's namespace. Set it through
        ///     <see cref="Namespace"/>, which restores it.
        /// </summary>
        /// <remarks>
        ///     Unlike Python's module global, the value belongs to the current thread and async flow (an
        ///     <see cref="AsyncLocal{T}"/>, DR-57): two threads registering inside scopes of their own at the same moment each
        ///     get their own namespace, where one global would hand both the namespace opened last. A thread, task or async
        ///     continuation started inside a scope inherits its value as it was at the start. On one thread it behaves as
        ///     Gymnasium's global does. Footgun: a value set inside an <see langword="async"/> method does not flow back to
        ///     its caller once the method returns (how <see cref="AsyncLocal{T}"/> works), so open the scope in the code that
        ///     registers rather than in a helper it awaits.
        /// </remarks>
        public static string? CurrentNamespace {
            get => CurrentNamespaceSlot.Value;
            set => CurrentNamespaceSlot.Value = value;
        }

        /// <summary>
        ///     The highest registered version of <paramref name="name"/> in <paramref name="ns"/> (Python's
        ///     <c>find_highest_version</c>), which <c>make</c> uses for an id given without a version.
        /// </summary>
        /// <param name="ns">The namespace, or <see langword="null"/>.</param>
        /// <param name="name">The env name.</param>
        /// <returns>The version, or <see langword="null"/> when no versioned id of that name is registered.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L302">gymnasium/envs/registration.py</see>.</remarks>
        public static long? FindHighestVersion(string? ns, string name) {
            long? highest = null;
            foreach (EnvSpec envSpec in Registry.Values) {
                if (envSpec.Namespace == ns && envSpec.Name == name && envSpec.Version is long version && (highest is null || version > highest))
                    highest = version;
            }

            return highest;
        }

        /// <summary>
        ///     Registers an env under <paramref name="id"/> for <c>make</c> (Python's <c>register</c>). The id's namespace is
        ///     replaced by <see cref="CurrentNamespace"/> when one is set; re-registering an id replaces it with a warning.
        /// </summary>
        /// <param name="id">The id, <c>[namespace/](env-name)[-v(version)]</c>.</param>
        /// <param name="entryPoint">
        ///     What builds the env: its class (<c>typeof(CartPoleEnv)</c>), an <see cref="EnvCreator"/>, or a
        ///     <c>"Namespace:Class"</c> string resolved when the env is made.
        /// </param>
        /// <param name="rewardThreshold">The return at which the task counts as solved.</param>
        /// <param name="nondeterministic">Whether the same seed and actions can still give different observations.</param>
        /// <param name="maxEpisodeSteps">The step limit <c>make</c>'s <c>TimeLimit</c> enforces, or <see langword="null"/> for none.</param>
        /// <param name="orderEnforce">Whether <c>make</c> adds <c>OrderEnforcing</c>.</param>
        /// <param name="disableEnvChecker">Whether <c>make</c> leaves out <c>PassiveEnvChecker</c>.</param>
        /// <param name="additionalWrappers">Wrappers <c>make</c> applies on top, in order.</param>
        /// <param name="vectorEntryPoint">What builds a vectorized version (for <c>make_vec</c>).</param>
        /// <param name="kwargs">
        ///     The env's constructor arguments by Python name: a dictionary (an <see cref="Info"/> is kept as the spec's own,
        ///     as Python keeps the dict passed), an options record, or an anonymous object.
        /// </param>
        /// <exception cref="ValueError">Thrown when neither entry point is given: <c>Either `entry_point` or `vector_entry_point` (or both) must be provided</c>.</exception>
        /// <exception cref="Error">Thrown for a malformed id.</exception>
        /// <exception cref="RegistrationError">Thrown when a versioned and an unversioned id of the same name would both be registered, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="kwargs"/> is not a mapping.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L563">gymnasium/envs/registration.py</see>.</remarks>
        public static void Register(string id, object? entryPoint = null, double? rewardThreshold = null, bool nondeterministic = false, long? maxEpisodeSteps = null,
            bool orderEnforce = true, bool disableEnvChecker = false, IEnumerable<WrapperSpec>? additionalWrappers = null, object? vectorEntryPoint = null,
            object? kwargs = null) {
            if (entryPoint is null && vectorEntryPoint is null)
                throw new ValueError("Either `entry_point` or `vector_entry_point` (or both) must be provided");
            (string? ns, string name, long? version) = ParseEnvId(id);

            Info kwargsDict = kwargs as Info ?? PyKwargs.From(kwargs);
            string? nsId;
            string? currentNamespace = CurrentNamespace;
            if (currentNamespace is not null) {
                object? custom = kwargsDict.Get("namespace");
                if (custom is not null && !PyEquality.AreEqual(custom, currentNamespace))
                    Logger.Warn($"Custom namespace `{PyBuiltins.Str(custom)}` is being overridden by namespace `{currentNamespace}`. If you are developing a plugin you shouldn't specify a namespace in `register` calls. The namespace is specified through the entry point package metadata.");
                nsId = currentNamespace;
            } else {
                nsId = ns;
            }

            string fullEnvId = GetEnvId(nsId, name, version);
            var newSpec = new EnvSpec(fullEnvId, entryPoint, rewardThreshold, nondeterministic, maxEpisodeSteps, orderEnforce, disableEnvChecker, kwargsDict,
                additionalWrappers, vectorEntryPoint);
            CheckSpecRegister(newSpec);

            if (Registry.ContainsKey(newSpec.Id))
                Logger.Warn($"Overriding environment {newSpec.Id} already in registry.");
            Registry[newSpec.Id] = newSpec;
        }

        /// <summary>
        ///     Sets <see cref="CurrentNamespace"/> until the returned scope is disposed (Python's <c>namespace</c> context
        ///     manager): <c>using (EnvRegistry.Namespace("MyPlugin")) Register("Env-v0", ...);</c> registers
        ///     <c>MyPlugin/Env-v0</c>. Inside the scope every id lands in <paramref name="ns"/>, even one that names a
        ///     namespace of its own, as in Gymnasium.
        /// </summary>
        /// <param name="ns">The namespace.</param>
        /// <returns>The scope; disposing it restores the namespace that was current before, once.</returns>
        /// <remarks>
        ///     The scope applies to the thread and async flow that opened it, and to the threads and tasks started inside it
        ///     (DR-57); other threads keep their own namespace. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L554">gymnasium/envs/registration.py</see>.
        /// </remarks>
        public static IDisposable Namespace(string ns) {
            // No lock: the slot is this flow's own, so nothing can change it between the read and the write.
            string? old = CurrentNamespace;
            CurrentNamespace = ns;
            return new NamespaceScope(old);
        }

        /// <summary>
        ///     A no-op that marks a module as used (Python's <c>register_envs</c>), kept so code written against Gymnasium reads
        ///     the same. Registering an env assembly's ids is its <c>Register()</c>'s job, which a module initializer calls
        ///     (DR-12, DR-17).
        /// </summary>
        /// <param name="envModule">Anything; ignored.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L548">gymnasium/envs/registration.py</see>.</remarks>
        public static void RegisterEnvs(object? envModule) { }

        /// <summary>
        ///     The spec registered under <paramref name="envId"/> (Python's <c>spec</c>), without the version and namespace
        ///     fallbacks of <c>make</c>: the id must be registered exactly.
        /// </summary>
        /// <param name="envId">The id.</param>
        /// <returns>The registered spec itself (not a copy).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envId"/> is <see langword="null"/>.</exception>
        /// <exception cref="Error">Thrown for a malformed id, or one that is not registered (<c>No registered env with id: X</c>).</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown (with a suggestion when one is close).</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown in its namespace.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest, or only an unversioned id exists.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L993">gymnasium/envs/registration.py</see>.</remarks>
        public static EnvSpec Spec(string envId) {
            EnvSpec? envSpec = Registry.Get(envId ?? throw new ArgumentNullException(nameof(envId)));
            if (envSpec is null) {
                (string? ns, string name, long? version) = ParseEnvId(envId);
                CheckVersionExists(ns, name, version);
                throw new Error($"No registered env with id: {envId}");
            }

            return envSpec;
        }

        /// <summary>
        ///     Builds the env registered under <paramref name="id"/> with <c>make</c>'s wrappers (Python's <c>make</c>): the
        ///     env, then <c>PassiveEnvChecker</c> (unless disabled), <c>OrderEnforcing</c> (when the spec enforces order),
        ///     <c>TimeLimit</c> (when a step limit applies), the spec's additional wrappers, and <c>RenderCollection</c> for a
        ///     <c>*_list</c> render mode the env does not provide itself.
        /// </summary>
        /// <param name="id">
        ///     The id: <c>"CartPole-v1"</c>, an unversioned <c>"CartPole"</c> (the latest version, with a warning), or
        ///     <c>"Assembly:Id"</c>, which first loads the assembly and runs its registration, as Python imports the module.
        /// </param>
        /// <param name="kwargs">
        ///     The env's constructor arguments by Python name, over the spec's: a dictionary, an options record, or an anonymous
        ///     object (<c>new { render_mode = "rgb_array" }</c>).
        /// </param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c> at all.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env, through its untyped view; the caller owns it.</returns>
        /// <exception cref="TypeError">
        ///     Thrown when <paramref name="id"/> is <see langword="null"/>, when the env's constructor rejects the kwargs
        ///     (<c>CartPoleEnv.__init__() got an unexpected keyword argument 'x' was raised from the environment creator for
        ///     CartPole-v1 with kwargs ({'x': 1})</c>), or when the entry point does not build an env.
        /// </exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point, with Gymnasium's messages.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when an <c>"Assembly:Id"</c> or a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created, or a pre-applied wrapper disagrees with the spec.</exception>
        /// <remarks>
        ///     <para>
        ///     The env and every wrapper are built through <see cref="PyKwargs.Construct"/>, which binds keywords by Python name
        ///     and closes the generic wrappers over the env's types. When a step after the env's construction fails (a wrapper
        ///     refusing the env), the env built so far is disposed before the error propagates.
        ///     </para>
        ///     <para>
        ///     Where an env lists both <c>rgb_array</c> and <c>rgb_array_list</c> and <c>human</c> is asked for, Python picks
        ///     either by its set order; Gym.NET always renders <c>rgb_array</c> for the <c>HumanRendering</c> wrapper.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L640">gymnasium/envs/registration.py</see>.</para>
        /// </remarks>
        public static IEnv Make(string id, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) {
            if (id is null)
                throw new TypeError($"Expected id to be a string or EnvSpec, got {PyBuiltins.TypeRepr(null)}");
            return MakeCore(FindSpec(id), kwargs, maxEpisodeSteps, disableEnvChecker, null);
        }

        /// <summary>Builds the env a spec describes, with <c>make</c>'s wrappers (Python's <c>make(env_spec)</c>); see <see cref="Make(string, object, long?, bool?)"/>.</summary>
        /// <param name="spec">The spec, registered or not (one read back from JSON, for example).</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, or as <see cref="Make(string, object, long?, bool?)"/> does.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static IEnv Make(EnvSpec spec, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) {
            if (spec is null)
                throw new TypeError($"Expected id to be a string or EnvSpec, got {PyBuiltins.TypeRepr(null)}");
            return MakeCore(spec, kwargs, maxEpisodeSteps, disableEnvChecker, null);
        }

        /// <summary>
        ///     <see cref="Make(string, object, long?, bool?)"/> typed (DR-10): the env comes back as
        ///     <c>Env&lt;TObs, TAct&gt;</c>. An env class of other types is refused before it is built; a stack whose recorded
        ///     wrappers change the types is checked once built.
        /// </summary>
        /// <typeparam name="TObs">The observation type the env must have (<c>NDArray</c> for CartPole).</typeparam>
        /// <typeparam name="TAct">The action type the env must have (<c>NDArray</c> for a <c>Discrete</c> action, DR-08).</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the env is not an <c>Env&lt;TObs, TAct&gt;</c>, or as <see cref="Make(string, object, long?, bool?)"/> does.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no entry point.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the id's namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a module named by the id or the entry point does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(string id, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) {
            if (id is null)
                throw new TypeError($"Expected id to be a string or EnvSpec, got {PyBuiltins.TypeRepr(null)}");
            return (Env<TObs, TAct>)MakeCore(FindSpec(id), kwargs, maxEpisodeSteps, disableEnvChecker, (typeof(TObs), typeof(TAct)));
        }

        /// <summary><see cref="Make(EnvSpec, object, long?, bool?)"/> typed (DR-10); see <see cref="Make{TObs, TAct}(string, object, long?, bool?)"/>.</summary>
        /// <typeparam name="TObs">The observation type the env must have.</typeparam>
        /// <typeparam name="TAct">The action type the env must have.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <param name="maxEpisodeSteps">The step limit instead of the spec's; <c>-1</c> adds no <c>TimeLimit</c>.</param>
        /// <param name="disableEnvChecker">Whether to leave out <c>PassiveEnvChecker</c>; <see langword="null"/> follows the spec.</param>
        /// <returns>The env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/> or the env is not an <c>Env&lt;TObs, TAct&gt;</c>.</exception>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when a string entry point names a module that does not exist.</exception>
        /// <exception cref="AttributeError">Thrown when a string entry point's namespace has no such class.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's <c>ClassMetadata</c> is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        public static Env<TObs, TAct> Make<TObs, TAct>(EnvSpec spec, object? kwargs = null, long? maxEpisodeSteps = null, bool? disableEnvChecker = null) {
            if (spec is null)
                throw new TypeError($"Expected id to be a string or EnvSpec, got {PyBuiltins.TypeRepr(null)}");
            return (Env<TObs, TAct>)MakeCore(spec, kwargs, maxEpisodeSteps, disableEnvChecker, (typeof(TObs), typeof(TAct)));
        }

        /// <summary>
        ///     The class a <c>"Namespace:Class"</c> entry point names (Python's <c>load_env_creator</c>, which imports
        ///     <c>module</c> and reads its attribute): a class of that name in that namespace among the loaded assemblies,
        ///     after loading the assembly named by the namespace (or a prefix of it) when none declares it yet.
        /// </summary>
        /// <param name="name">The entry point, <c>"Gym.Wrappers:TimeLimit"</c>; a generic class is named without its arity.</param>
        /// <returns>The class (an open generic definition for a generic one).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when the string has no colon or more than one (Python's unpacking: <c>not enough values to unpack (expected 2, got 1)</c>).</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when no loaded or loadable assembly declares the namespace: <c>No module named 'X'</c>.</exception>
        /// <exception cref="AttributeError">Thrown when the namespace has no such class: <c>module 'X' has no attribute 'Y'</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L533">gymnasium/envs/registration.py</see>.</remarks>
        public static Type LoadEnvCreator(string name) {
            if (name is null)
                throw new ArgumentNullException(nameof(name));
            string[] parts = name.Split(':');
            if (parts.Length < 2)
                throw new ValueError($"not enough values to unpack (expected 2, got {parts.Length.ToString(CultureInfo.InvariantCulture)})");
            if (parts.Length > 2)
                throw new ValueError("too many values to unpack (expected 2)");
            (string module, string attribute) = (parts[0], parts[1]);

            if (ResolvedEntryPoints.TryGetValue(name, out Type? cached))
                return cached;
            Type? found = FindClass(module, attribute);
            if (found is null && !NamespaceExists(module)) {
                // Python imports the module here; the counterpart is loading the assembly named after the namespace.
                LoadAssemblyFor(module);
                found = FindClass(module, attribute);
            }

            if (found is null) {
                if (MissingModule(module) is { } missing)
                    throw new ModuleNotFoundError($"No module named '{missing}'");
                throw new AttributeError($"module '{module}' has no attribute '{attribute}'");
            }

            ResolvedEntryPoints[name] = found;
            return found;
        }

        /// <summary>
        ///     The <c>"Namespace:Class"</c> string that names a class as an entry point, the form <c>load_env_creator</c> reads
        ///     back: <c>"Gym.Environments.Envs.Classic:CartPoleEnv"</c>, a generic class without its arity
        ///     (<c>"Gym.Wrappers:TimeLimit"</c>). A spec's JSON writes a class entry point this way.
        /// </summary>
        /// <param name="type">The class.</param>
        /// <returns>The entry-point string.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="type"/> is <see langword="null"/>.</exception>
        public static string EntryPointName(Type type) {
            if (type is null)
                throw new ArgumentNullException(nameof(type));
            Type definition = type.IsGenericType ? type.GetGenericTypeDefinition() : type;
            return $"{definition.Namespace}:{PyAttributes.PyTypeName(definition)}";
        }

        /// <summary>
        ///     Prints (or returns) every registered id grouped by namespace (Python's <c>pprint_registry</c>): a
        ///     <c>===== namespace =====</c> header, then the ids sorted, <paramref name="numCols"/> to a line.
        /// </summary>
        /// <param name="printRegistry">The registry to print; <see langword="null"/> is <see cref="Registry"/>.</param>
        /// <param name="numCols">How many ids per line.</param>
        /// <param name="excludeNamespaces">Namespaces to leave out, or <see langword="null"/>.</param>
        /// <param name="disablePrint">Return the text instead of writing it to the console.</param>
        /// <returns>The text when <paramref name="disablePrint"/> is set, otherwise <see langword="null"/>.</returns>
        /// <exception cref="ZeroDivisionError">Thrown when <paramref name="numCols"/> is 0 (Python's <c>ZeroDivisionError</c>).</exception>
        /// <remarks>
        ///     An id without a namespace is grouped by its entry point, as in Gymnasium: a string
        ///     <c>"gymnasium.envs.classic_control.cartpole:CartPoleEnv"</c> by its third dotted part (<c>classic_control</c>),
        ///     else its second, else the env's name; a function by <c>None</c>. A class entry point, which Python groups under
        ///     <c>None</c> too, is grouped here by the last part of its namespace, the env family (<c>Classic</c> for
        ///     <c>Gym.Environments.Envs.Classic</c>), so Gym.NET's own ids print in families.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L1017">gymnasium/envs/registration.py</see>.
        /// </remarks>
        public static string? PprintRegistry(IReadOnlyDictionary<string, EnvSpec>? printRegistry = null, int numCols = 3, IEnumerable<string?>? excludeNamespaces = null,
            bool disablePrint = false) {
            printRegistry ??= Registry;
            // defaultdict(list): the namespaces in the order first seen.
            var namespaceEnvs = new List<(string? Namespace, List<string> Ids)>();
            int maxJustify = 0;

            foreach (EnvSpec envSpec in printRegistry.Values) {
                string? ns = envSpec.Namespace;
                if (ns is null && envSpec.EntryPoint is string entryPoint) {
                    // Obtain the namespace from the entry point's module path.
                    string envEntryPoint = Regex.Replace(entryPoint, @":\w+", "");
                    string[] splitEntryPoint = envEntryPoint.Split('.');
                    if (splitEntryPoint.Length >= 3)
                        ns = splitEntryPoint[2];
                    else if (splitEntryPoint.Length > 1)
                        ns = splitEntryPoint[1];
                    else
                        ns = envSpec.Name;
                } else if (ns is null && envSpec.EntryPoint is Type entryClass) {
                    string typeNamespace = entryClass.Namespace ?? "";
                    ns = typeNamespace.Length == 0 ? envSpec.Name : typeNamespace.Substring(typeNamespace.LastIndexOf('.') + 1);
                }

                int group = namespaceEnvs.FindIndex(entry => entry.Namespace == ns);
                if (group < 0) {
                    namespaceEnvs.Add((ns, new List<string>()));
                    group = namespaceEnvs.Count - 1;
                }

                namespaceEnvs[group].Ids.Add(envSpec.Id);
                maxJustify = Math.Max(maxJustify, CodePointLength(envSpec.Name));
            }

            List<string?>? excluded = excludeNamespaces?.ToList();
            var output = new List<string>();
            foreach ((string? ns, List<string> envIds) in namespaceEnvs) {
                if (excluded is not null && excluded.Contains(ns))
                    continue;

                var namespaceOutput = new StringBuilder($"===== {ns ?? "None"} =====\n");
                List<string> sorted = envIds.OrderBy(envId => envId, Comparer<string>.Create(PyBuiltins.CompareStrings)).ToList();
                for (int count = 1; count <= sorted.Count; count++) {
                    // ljust pads to max_justify code points; ids longer than the longest name are left as they are.
                    string envId = sorted[count - 1];
                    namespaceOutput.Append(envId).Append(' ', Math.Max(0, maxJustify - CodePointLength(envId))).Append(' ');
                    // Python's %: a num_cols of 0 raises ZeroDivisionError("integer modulo by zero"), NumSharp's class
                    if (PyMath.Mod(count, numCols) == 0) {
                        TrimEndSpaces(namespaceOutput);
                        if (count != sorted.Count)
                            namespaceOutput.Append('\n');
                    }
                }

                TrimEndSpaces(namespaceOutput);
                output.Add(namespaceOutput.ToString());
            }

            string text = string.Join("\n", output);
            if (disablePrint)
                return text;
            Console.WriteLine(text);
            return null;
        }

        /// <summary>
        ///     <c>make</c> proper: everything after the spec is found, with the typed form's check when
        ///     <paramref name="expected"/> is given.
        /// </summary>
        /// <param name="envSpec">The spec.</param>
        /// <param name="kwargs">The call's kwargs, any form <see cref="PyKwargs.From"/> takes.</param>
        /// <param name="maxEpisodeSteps">The call's step limit.</param>
        /// <param name="disableEnvChecker">The call's checker switch.</param>
        /// <param name="expected">The <c>Env&lt;TObs, TAct&gt;</c> type arguments the typed form requires, or <see langword="null"/>.</param>
        /// <returns>The wrapped env.</returns>
        /// <exception cref="Error">Thrown when the spec has no entry point.</exception>
        /// <exception cref="TypeError">Thrown when the env cannot be built from the kwargs, or is not of the expected types.</exception>
        /// <exception cref="InvalidMetadata">Thrown when the env class's metadata is not a dict.</exception>
        /// <exception cref="ValueError">Thrown when a recorded wrapper cannot be re-created.</exception>
        private static IEnv MakeCore(EnvSpec envSpec, object? kwargs, long? maxEpisodeSteps, bool? disableEnvChecker, (Type Obs, Type Act)? expected) {
            // Update the env spec kwargs with the make kwargs.
            Info envSpecKwargs = envSpec.Kwargs.DeepCopy();
            envSpecKwargs.Update(PyKwargs.From(kwargs));

            // Load the environment creator.
            object envCreator = envSpec.EntryPoint switch {
                null => throw new Error($"{envSpec.Id} registered but entry_point is not specified"),
                Type or EnvCreator => envSpec.EntryPoint,
                string entryPoint => LoadEnvCreator(entryPoint),
                // Python would call str.split on it: load_env_creator(entry_point) with a non-string.
                _ => throw new AttributeError($"'{PyBuiltins.TypeName(envSpec.EntryPoint)}' object has no attribute 'split'"),
            };

            // The typed form refuses an env class of other types before building it; recorded wrappers may change the
            // types, so a spec with any is checked once built instead.
            if (expected is { } types && envCreator is Type envClass && envSpec.AdditionalWrappers.Count == 0 && EnvTypeArguments(envClass) is { } declared
                && (declared.Obs != types.Obs || declared.Act != types.Act))
                throw TypeMismatch(envSpec.Id, envClass, types);

            // Determine if to use the rendering.
            object? renderModes = null;
            if (envCreator is Type creatorClass && ClassMetadataOf(creatorClass) is { } envCreatorMetadata) {
                CheckMetadata(envCreatorMetadata);
                renderModes = Dict.Get(envCreatorMetadata, "render_modes");
            }

            object? renderMode = envSpecKwargs.Get("render_mode");
            bool applyHumanRendering = false;
            bool applyRenderCollection = false;

            // If the mode is not valid, try applying the HumanRendering/RenderCollection wrappers.
            if (renderMode is not null && renderModes is not null && !PyIn(renderMode, renderModes)) {
                string[] displayableModes = new[] { "rgb_array", "rgb_array_list" }.Where(mode => PyIn(mode, renderModes)).ToArray();
                if (renderMode is "human" && displayableModes.Length > 0) {
                    Logger.Warn("You are trying to use 'human' rendering for an environment that doesn't natively support it. The HumanRendering wrapper is being applied to your environment.");
                    envSpecKwargs["render_mode"] = displayableModes[0];
                    applyHumanRendering = true;
                } else if ((renderMode as string ?? throw new AttributeError($"'{PyBuiltins.TypeName(renderMode)}' object has no attribute 'endswith'")).EndsWith("_list", StringComparison.Ordinal)
                           && PyIn(((string)renderMode).Substring(0, ((string)renderMode).Length - "_list".Length), renderModes)) {
                    envSpecKwargs["render_mode"] = ((string)renderMode).Substring(0, ((string)renderMode).Length - "_list".Length);
                    applyRenderCollection = true;
                } else {
                    Logger.Warn($"The environment is being initialised with render_mode={PyBuiltins.Repr(renderMode)} that is not in the possible render_modes ({PyBuiltins.Str(renderModes)}).");
                }
            }

            object created;
            try {
                created = envCreator is Type type ? PyKwargs.Construct(type, envSpecKwargs) : ((EnvCreator)envCreator)(envSpecKwargs);
            } catch (TypeError e) {
                if (e.Message.Contains("got an unexpected keyword argument 'render_mode'", StringComparison.Ordinal) && applyHumanRendering)
                    throw new Error($"You passed render_mode='human' although {envSpec.Id} doesn't implement human-rendering natively. Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old rendering API, which is not supported by the HumanRendering wrapper.", e);
                throw Rewrap(e, $"{e.Message} was raised from the environment creator for {envSpec.Id} with kwargs ({PyBuiltins.Str(envSpecKwargs)})");
            }

            if (created is not IEnv env) {
                (created as IDisposable)?.Dispose();
                throw new TypeError($"The environment must inherit from the gymnasium.Env class, actual class: {PyBuiltins.TypeRepr(created)}. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.");
            }

            try {
                // Set the minimal env spec for the environment.
                env.Unwrapped.Spec = new EnvSpec(envSpec.Id, envSpec.EntryPoint, envSpec.RewardThreshold, envSpec.Nondeterministic, maxEpisodeSteps: null, orderEnforce: false,
                    disableEnvChecker: true, kwargs: envSpecKwargs, additionalWrappers: null, vectorEntryPoint: envSpec.VectorEntryPoint);

                // Check if pre-wrapped wrappers.
                EnvSpec currentSpec = env.Spec ?? throw new AssertionError("");
                int numPriorWrappers = currentSpec.AdditionalWrappers.Count;
                // Only validate if the spec expects at least num_prior_wrappers (entry points can add their own wrappers).
                if (numPriorWrappers <= envSpec.AdditionalWrappers.Count && !envSpec.AdditionalWrappers.Take(numPriorWrappers).SequenceEqual(currentSpec.AdditionalWrappers)) {
                    // Python raises inside the loop's first iteration, naming the first pair whether or not it differs.
                    throw new ValueError($"The environment's wrapper spec {currentSpec.AdditionalWrappers[0]} is different from the saved `EnvSpec` additional wrapper {envSpec.AdditionalWrappers[0]}");
                }

                // Run the environment checker as the lowest level wrapper.
                if (disableEnvChecker == false || (disableEnvChecker is null && !envSpec.DisableEnvChecker))
                    env = Wrap(typeof(PassiveEnvChecker<,>), env, null);

                // Add the order enforcing wrapper.
                if (envSpec.OrderEnforce)
                    env = Wrap(typeof(OrderEnforcing<,>), env, null);

                // Add the time limit wrapper.
                if (maxEpisodeSteps != -1) {
                    if (maxEpisodeSteps is not null)
                        env = Wrap(typeof(TimeLimit<,>), env, new Info { ["max_episode_steps"] = maxEpisodeSteps });
                    else if (envSpec.MaxEpisodeSteps is not null)
                        env = Wrap(typeof(TimeLimit<,>), env, new Info { ["max_episode_steps"] = envSpec.MaxEpisodeSteps });
                }

                foreach (WrapperSpec wrapperSpec in envSpec.AdditionalWrappers.Skip(numPriorWrappers)) {
                    if (wrapperSpec.Kwargs is null)
                        throw new ValueError($"{wrapperSpec.Name} wrapper does not inherit from `gymnasium.utils.RecordConstructorArgs`, therefore, the wrapper cannot be recreated.");
                    env = Wrap(LoadEnvCreator(wrapperSpec.EntryPoint), env, wrapperSpec.Kwargs);
                }

                // Add the human rendering wrapper.
                if (applyHumanRendering)
                    env = Wrap(typeof(HumanRendering<,>), env, null);
                if (applyRenderCollection)
                    env = Wrap(typeof(RenderCollection<,>), env, null);

                if (expected is { } final && !(EnvTypeArguments(env.GetType()) is { } actual && actual.Obs == final.Obs && actual.Act == final.Act))
                    throw TypeMismatch(envSpec.Id, env.GetType(), final);
                return env;
            } catch {
                // Nobody else holds the env yet: release what was built before the error propagates.
                try {
                    env.Dispose();
                } catch {
                    // The original error is the one to report.
                }

                throw;
            }
        }

        /// <summary>
        ///     Wraps an env as Python's <c>Wrapper(env=env, **kwargs)</c> does: the wrapper class (generic definitions are
        ///     closed over the env's types) constructed with <c>env</c> and the kwargs by Python name.
        /// </summary>
        /// <param name="wrapperType">The wrapper class.</param>
        /// <param name="env">The env to wrap.</param>
        /// <param name="kwargs">The wrapper's other arguments, or <see langword="null"/>.</param>
        /// <returns>The wrapper.</returns>
        /// <exception cref="TypeError">Thrown when the kwargs also name <c>env</c>, when they do not fit the constructor, or when the class is not an env.</exception>
        private static IEnv Wrap(Type wrapperType, IEnv env, Info? kwargs) {
            var arguments = new Info { ["env"] = env };
            if (kwargs is not null) {
                if (kwargs.ContainsKey("env"))
                    throw new TypeError($"{PyAttributes.PyTypeName(wrapperType)}() got multiple values for keyword argument 'env'");
                arguments.Update(kwargs);
            }

            object wrapper = PyKwargs.Construct(wrapperType, arguments);
            return wrapper as IEnv ?? throw new TypeError($"{PyAttributes.PyTypeName(wrapperType)} is not an env, so it cannot wrap one");
        }

        /// <summary>
        ///     Python's <c>raise type(e)(message) from e</c>: an error of the same class with a new message and the original as
        ///     its cause, or a plain <see cref="TypeError"/> when the class has no <c>(message, inner)</c> constructor.
        /// </summary>
        /// <param name="original">The error caught.</param>
        /// <param name="message">The new message.</param>
        /// <returns>The error to throw.</returns>
        private static Exception Rewrap(TypeError original, string message) {
            try {
                if (Activator.CreateInstance(original.GetType(), message, original) is Exception same)
                    return same;
            } catch (MissingMethodException) {
                // No (message, inner) constructor: fall back to the base class below.
            }

            return new TypeError(message, original);
        }

        /// <summary>The typed form's error: the env's types are not the ones asked for.</summary>
        /// <param name="id">The env id.</param>
        /// <param name="envClass">The env's class.</param>
        /// <param name="expected">The types asked for.</param>
        /// <returns>The error to throw.</returns>
        private static TypeError TypeMismatch(string id, Type envClass, (Type Obs, Type Act) expected) {
            string actual = EnvTypeArguments(envClass) is { } declared ? $"Env<{declared.Obs.Name}, {declared.Act.Name}>" : "no Env<TObs, TAct>";
            return new TypeError($"{id} makes {PyAttributes.PyTypeName(envClass)}, an {actual}, not the Env<{expected.Obs.Name}, {expected.Act.Name}> asked for; use the untyped Make or ask for the env's own types");
        }

        /// <summary>The <c>TObs</c> and <c>TAct</c> of the <c>Env&lt;TObs, TAct&gt;</c> a class derives from, or <see langword="null"/>.</summary>
        /// <param name="type">The class.</param>
        /// <returns>The type arguments.</returns>
        private static (Type Obs, Type Act)? EnvTypeArguments(Type type) {
            for (Type? level = type; level != null; level = level.BaseType) {
                if (level.IsGenericType && level.GetGenericTypeDefinition() == typeof(Env<,>)) {
                    Type[] arguments = level.GetGenericArguments();
                    return (arguments[0], arguments[1]);
                }
            }

            return null;
        }

        /// <summary>
        ///     An env class's class-level metadata (Python's <c>env_creator.metadata</c>, read before the env exists): the
        ///     public static <c>ClassMetadata</c> property or field of the class or its nearest base declaring one.
        ///     <see langword="null"/> when none does, or when it is <see langword="null"/>, as a wrapper's is (Python's wrapper
        ///     classes expose metadata only through an instance property, which <c>make</c> skips).
        /// </summary>
        /// <param name="type">The class.</param>
        /// <returns>The metadata object, which <see cref="CheckMetadata"/> then validates, or <see langword="null"/>.</returns>
        private static object? ClassMetadataOf(Type type) {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly;
            for (Type? level = type; level != null; level = level.BaseType) {
                if (level.GetProperty("ClassMetadata", flags) is { } property && property.GetIndexParameters().Length == 0)
                    return property.GetValue(null);
                if (level.GetField("ClassMetadata", flags) is { } field)
                    return field.GetValue(null);
            }

            return null;
        }

        /// <summary>
        ///     Python's <c>_check_metadata</c>: refuses metadata that is not a dict, and warns when <c>render_modes</c> is
        ///     missing or not iterable.
        /// </summary>
        /// <param name="testingMetadata">The class metadata.</param>
        /// <exception cref="InvalidMetadata">Thrown when it is not a dict: <c>Expect the environment metadata to be dict, actual type: ...</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L470">gymnasium/envs/registration.py</see>.</remarks>
        private static void CheckMetadata(object testingMetadata) {
            IReadOnlyDictionary<string, object?> metadata = Dict.Of(testingMetadata)
                ?? throw new InvalidMetadata($"Expect the environment metadata to be dict, actual type: {PyBuiltins.TypeRepr(testingMetadata)}");
            object? renderModes = metadata.TryGetValue("render_modes", out object? modes) ? modes : null;
            if (renderModes is null)
                Logger.Warn($"The environment creator metadata doesn't include `render_modes`, contains: {PyBuiltins.Repr(metadata.Keys.ToList())}");
            else if (renderModes is not IEnumerable)
                Logger.Warn($"Expects the environment metadata render_modes to be a Iterable, actual type: {PyBuiltins.TypeRepr(renderModes)}");
        }

        /// <summary>
        ///     Python's <c>_find_spec</c>: imports the module of a <c>"module:id"</c> id, then looks the id up, falling back to
        ///     the latest version for an unversioned id (with a warning) and warning when a newer version exists.
        /// </summary>
        /// <param name="envId">The id as given to <c>make</c>.</param>
        /// <returns>The registered spec.</returns>
        /// <exception cref="ValueError">Thrown when the id has more than one colon.</exception>
        /// <exception cref="ModuleNotFoundError">Thrown when the module part names no assembly or namespace, with Gymnasium's hint appended.</exception>
        /// <exception cref="Error">Thrown for a malformed or unregistered id.</exception>
        /// <exception cref="NamespaceNotFound">Thrown when the namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when the version is older than the latest.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L488">gymnasium/envs/registration.py</see>.</remarks>
        private static EnvSpec FindSpec(string envId) {
            // The environment name can include an unloaded module in "module:env_name" style.
            string envName = envId;
            if (envId.Contains(':')) {
                string[] parts = envId.Split(':');
                if (parts.Length > 2)
                    throw new ValueError("too many values to unpack (expected 2)");
                string module = parts[0];
                envName = parts[1];
                try {
                    ImportModule(module);
                } catch (ModuleNotFoundError e) {
                    throw new ModuleNotFoundError($"{e.Message}. Environment registration via importing a module failed. Check whether '{module}' contains env registration and can be imported.", e);
                }
            }

            // Load the env spec from the registry.
            EnvSpec? envSpec = Registry.Get(envName);

            // Update the env spec if no version is provided, warn if it is out of date.
            (string? ns, string name, long? version) = ParseEnvId(envName);
            long? latestVersion = FindHighestVersion(ns, name);
            if (version is not null && latestVersion is not null && latestVersion > version)
                Logger.Deprecation($"The environment {envName} is out of date. You should consider upgrading to version `v{latestVersion.Value.ToString(CultureInfo.InvariantCulture)}`.");
            if (version is null && latestVersion is not null) {
                version = latestVersion;
                string newEnvId = GetEnvId(ns, name, version);
                envSpec = Registry.Get(newEnvId);
                Logger.Warn($"Using the latest versioned environment `{newEnvId}` instead of the unversioned environment `{envName}`.");
            }

            if (envSpec is null) {
                CheckVersionExists(ns, name, version);
                throw new Error($"No registered env with id: {envName}. Did you register it, or import the package that registers it? Use `gymnasium.pprint_registry()` to see all of the registered environments.");
            }

            return envSpec;
        }

        /// <summary>
        ///     Python's <c>importlib.import_module(module)</c> for an id's module part: loads the assembly of that name (or the
        ///     assemblies declaring that namespace) and runs its module initializer, which registers its ids (DR-12).
        /// </summary>
        /// <param name="module">An assembly name or a namespace.</param>
        /// <exception cref="ModuleNotFoundError">Thrown when no assembly or namespace has that name: <c>No module named 'X'</c>.</exception>
        private static void ImportModule(string module) {
            var assemblies = AppDomain.CurrentDomain.GetAssemblies()
                .Where(assembly => string.Equals(assembly.GetName().Name, module, StringComparison.Ordinal) || DeclaresNamespace(assembly, module))
                .ToList();
            if (assemblies.Count == 0 && TryLoadAssembly(module) is { } loaded)
                assemblies.Add(loaded);
            if (assemblies.Count == 0)
                throw new ModuleNotFoundError($"No module named '{MissingModule(module) ?? module}'");
            foreach (Assembly assembly in assemblies)
                RuntimeHelpers.RunModuleConstructor(assembly.ManifestModule.ModuleHandle);
        }

        /// <summary>
        ///     The class <paramref name="attribute"/> in namespace <paramref name="module"/> among the loaded assemblies, by its
        ///     Python name (without generic arity); a public class wins over a non-public one, a top-level class over a nested one.
        /// </summary>
        /// <param name="module">The namespace.</param>
        /// <param name="attribute">The class name.</param>
        /// <returns>The class, or <see langword="null"/>.</returns>
        private static Type? FindClass(string module, string attribute) {
            Type? best = null;
            foreach (Assembly assembly in AppDomain.CurrentDomain.GetAssemblies()) {
                foreach (Type type in LoadableTypes(assembly)) {
                    if (type.Namespace != module || PyAttributes.PyTypeName(type) != attribute)
                        continue;
                    if (best is null || Rank(type) > Rank(best))
                        best = type;
                }
            }

            return best;

            // Public before non-public, top level before nested.
            static int Rank(Type type) => (type.IsVisible ? 2 : 0) + (type.IsNested ? 0 : 1);
        }

        /// <summary>Whether any loaded assembly declares the namespace or one nested in it (a Python package with submodules).</summary>
        /// <param name="module">The namespace.</param>
        /// <returns><see langword="true"/> when it exists.</returns>
        private static bool NamespaceExists(string module) => AppDomain.CurrentDomain.GetAssemblies().Any(assembly => DeclaresNamespace(assembly, module));

        /// <summary>Whether an assembly declares a type in the namespace or in one nested in it.</summary>
        /// <param name="assembly">The assembly.</param>
        /// <param name="module">The namespace.</param>
        /// <returns><see langword="true"/> when it does.</returns>
        private static bool DeclaresNamespace(Assembly assembly, string module) =>
            LoadableTypes(assembly).Any(type => type.Namespace is { } ns && (ns == module || ns.StartsWith(module + ".", StringComparison.Ordinal)));

        /// <summary>
        ///     The first dotted prefix of <paramref name="module"/> that no loaded assembly declares, the name Python's
        ///     <c>ModuleNotFoundError</c> reports (<c>import a.b.c</c> with no <c>a.b</c> says <c>No module named 'a.b'</c>).
        /// </summary>
        /// <param name="module">The namespace.</param>
        /// <returns>The missing prefix, or <see langword="null"/> when the whole namespace exists.</returns>
        private static string? MissingModule(string module) {
            string[] parts = module.Split('.');
            for (int i = 1; i <= parts.Length; i++) {
                string prefix = string.Join(".", parts, 0, i);
                if (!NamespaceExists(prefix))
                    return prefix;
            }

            return null;
        }

        /// <summary>Loads the assembly named by a namespace or its longest prefix that names one: <c>Gym.Environments.Envs.Classic</c> tries <c>Gym.Environments.Envs.Classic</c>, then <c>Gym.Environments.Envs</c>, then <c>Gym.Environments</c>, then <c>Gym</c>.</summary>
        /// <param name="module">The namespace.</param>
        private static void LoadAssemblyFor(string module) {
            string[] parts = module.Split('.');
            for (int i = parts.Length; i >= 1; i--) {
                if (TryLoadAssembly(string.Join(".", parts, 0, i)) is not null)
                    return;
            }
        }

        /// <summary>Loads an assembly by simple name, or returns <see langword="null"/> when there is none to load.</summary>
        /// <param name="name">The assembly's simple name.</param>
        /// <returns>The assembly, or <see langword="null"/>.</returns>
        private static Assembly? TryLoadAssembly(string name) {
            try {
                return Assembly.Load(new AssemblyName(name));
            } catch (Exception e) when (e is FileNotFoundException or FileLoadException or BadImageFormatException or ArgumentException) {
                // No such assembly next to the app: the caller reports the missing module.
                return null;
            }
        }

        /// <summary>An assembly's types, those that load when some of its dependencies do not.</summary>
        /// <param name="assembly">The assembly.</param>
        /// <returns>The types.</returns>
        private static IEnumerable<Type> LoadableTypes(Assembly assembly) {
            try {
                return assembly.GetTypes();
            } catch (ReflectionTypeLoadException e) {
                return e.Types.Where(type => type is not null)!;
            }
        }

        /// <summary>
        ///     Python's <c>_check_namespace_exists</c>: a namespace some registered id uses passes; otherwise the error suggests
        ///     the closest one, or the package to install.
        /// </summary>
        /// <param name="ns">The namespace, or <see langword="null"/> (always exists).</param>
        /// <exception cref="NamespaceNotFound">Thrown when it is unknown: <c>Namespace X not found. Did you mean: `Y`?</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L322">gymnasium/envs/registration.py</see>.</remarks>
        private static void CheckNamespaceExists(string? ns) {
            if (ns is null)
                return;

            var namespaces = new HashSet<string>(Registry.Values.Where(envSpec => envSpec.Namespace is not null).Select(envSpec => envSpec.Namespace!), StringComparer.Ordinal);
            if (namespaces.Contains(ns))
                return;

            IReadOnlyList<string>? suggestion = namespaces.Count > 0 ? Difflib.GetCloseMatches(ns, namespaces, n: 1) : null;
            string suggestionMsg = suggestion is { Count: > 0 } ? $"Did you mean: `{suggestion[0]}`?" : $"Have you installed the proper package for {ns}?";
            throw new NamespaceNotFound($"Namespace {ns} not found. {suggestionMsg}");
        }

        /// <summary>Python's <c>_check_name_exists</c>: the namespace first, then the name within it, suggesting the closest name.</summary>
        /// <param name="ns">The namespace.</param>
        /// <param name="name">The env name.</param>
        /// <exception cref="NamespaceNotFound">Thrown when the namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown: <c>Environment `X` doesn't exist in namespace N. Did you mean: `Y`?</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L349">gymnasium/envs/registration.py</see>.</remarks>
        private static void CheckNameExists(string? ns, string name) {
            CheckNamespaceExists(ns);

            var names = new HashSet<string>(Registry.Values.Where(envSpec => envSpec.Namespace == ns).Select(envSpec => envSpec.Name), StringComparer.Ordinal);
            if (names.Contains(name))
                return;

            IReadOnlyList<string> suggestion = Difflib.GetCloseMatches(name, names, n: 1);
            // `if ns`: an empty namespace reads as none, as in Python.
            string namespaceMsg = !string.IsNullOrEmpty(ns) ? $" in namespace {ns}" : "";
            string suggestionMsg = suggestion.Count > 0 ? $" Did you mean: `{suggestion[0]}`?" : "";
            throw new NameNotFound($"Environment `{name}` doesn't exist{namespaceMsg}.{suggestionMsg}");
        }

        /// <summary>
        ///     Python's <c>_check_version_exists</c>: the complete test of an id, with the best hint available. Returns when the
        ///     id is registered, or when it has no version and its name exists.
        /// </summary>
        /// <param name="ns">The namespace.</param>
        /// <param name="name">The env name.</param>
        /// <param name="version">The version, or <see langword="null"/>.</param>
        /// <exception cref="NamespaceNotFound">Thrown when the namespace is unknown.</exception>
        /// <exception cref="NameNotFound">Thrown when the name is unknown.</exception>
        /// <exception cref="DeprecatedEnv">Thrown when only an unversioned id exists, or the version is older than the latest.</exception>
        /// <exception cref="VersionNotFound">Thrown when the version is newer than any registered.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L371">gymnasium/envs/registration.py</see>.</remarks>
        private static void CheckVersionExists(string? ns, string name, long? version) {
            if (Registry.ContainsKey(GetEnvId(ns, name, version)))
                return;

            CheckNameExists(ns, name);
            if (version is null)
                return;

            string message = $"Environment version `v{version.Value.ToString(CultureInfo.InvariantCulture)}` for environment `{GetEnvId(ns, name, null)}` doesn't exist.";

            // sorted(key=int(version or -1)): v0 sorts with the unversioned, a stable sort keeps their registry order.
            List<EnvSpec> envSpecs = Registry.Values
                .Where(envSpec => envSpec.Namespace == ns && envSpec.Name == name)
                .OrderBy(envSpec => envSpec.Version is null or 0 ? -1 : envSpec.Version.Value)
                .ToList();

            List<EnvSpec> defaultSpec = envSpecs.Where(envSpec => envSpec.Version is null).ToList();
            if (defaultSpec.Count > 0) {
                message += $" It provides the default version `{defaultSpec[0].Id}`.";
                if (envSpecs.Count == 1)
                    throw new DeprecatedEnv(message);
            }

            // Process the possible versioned environments; max keeps the first of equal versions.
            EnvSpec? latestSpec = null;
            foreach (EnvSpec envSpec in envSpecs.Where(envSpec => envSpec.Version is not null)) {
                if (latestSpec is null || envSpec.Version > latestSpec.Version)
                    latestSpec = envSpec;
            }

            if (latestSpec is not null && version > latestSpec.Version) {
                string versionListMsg = string.Join(", ", envSpecs.Select(envSpec => $"`v{PyBuiltins.Str(envSpec.Version)}`"));
                message += $" It provides versioned environments: [ {versionListMsg} ].";
                throw new VersionNotFound(message);
            }

            if (latestSpec is not null && version < latestSpec.Version)
                throw new DeprecatedEnv($"Environment version v{version.Value.ToString(CultureInfo.InvariantCulture)} for `{GetEnvId(ns, name, null)}` is deprecated. Please use `{latestSpec.Id}` instead.");
        }

        /// <summary>
        ///     Python's <c>_check_spec_register</c>: a name is registered either unversioned or with versions, never both, so
        ///     an unversioned <c>make</c> is unambiguous.
        /// </summary>
        /// <param name="testingSpec">The spec about to be registered.</param>
        /// <exception cref="RegistrationError">Thrown when it would mix versioned and unversioned ids of one name, with Gymnasium's messages.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L431">gymnasium/envs/registration.py</see>.</remarks>
        private static void CheckSpecRegister(EnvSpec testingSpec) {
            EnvSpec? latestVersionedSpec = null;
            EnvSpec? unversionedSpec = null;
            foreach (EnvSpec envSpec in Registry.Values) {
                if (envSpec.Namespace != testingSpec.Namespace || envSpec.Name != testingSpec.Name)
                    continue;
                if (envSpec.Version is not null) {
                    if (latestVersionedSpec is null || envSpec.Version > latestVersionedSpec.Version)
                        latestVersionedSpec = envSpec;
                } else {
                    unversionedSpec ??= envSpec;
                }
            }

            if (unversionedSpec is not null && testingSpec.Version is not null)
                throw new RegistrationError($"Can't register the versioned environment `{testingSpec.Id}` when the unversioned environment `{unversionedSpec.Id}` of the same name already exists.");
            if (latestVersionedSpec is not null && testingSpec.Version is null)
                throw new RegistrationError($"Can't register the unversioned environment `{testingSpec.Id}` when the versioned environment `{latestVersionedSpec.Id}` of the same name already exists. Note: the default behavior is that `gym.make` with the unversioned environment will return the latest versioned environment");
        }

        /// <summary>
        ///     Python's <c>item in container</c>: a substring test in a string, a key test in a dict, an equality test
        ///     (<see cref="PyEquality"/>) over any other iterable.
        /// </summary>
        /// <param name="item">The item.</param>
        /// <param name="container">The container.</param>
        /// <returns><see langword="true"/> when the item is in it.</returns>
        /// <exception cref="TypeError">
        ///     Thrown for a container that is not iterable (<c>argument of type 'int' is not iterable</c>), or a non-string
        ///     looked for in a string (<c>'in &lt;string&gt;' requires string as left operand, not int</c>).
        /// </exception>
        private static bool PyIn(object? item, object container) {
            switch (container) {
                case string text:
                    return item is string part
                        ? text.Contains(part, StringComparison.Ordinal)
                        : throw new TypeError($"'in <string>' requires string as left operand, not {PyBuiltins.TypeName(item)}");
                case NDArray:
                    // NumSharp has no string arrays, so an array never holds a render mode.
                    return false;
            }

            if (Dict.Of(container) is { } entries)
                return item is string key && entries.ContainsKey(key);
            if (container is IEnumerable sequence) {
                foreach (object? element in sequence) {
                    if (ReferenceEquals(element, item) || PyEquality.AreEqual(element, item))
                        return true;
                }

                return false;
            }

            throw new TypeError($"argument of type '{PyBuiltins.TypeName(container)}' is not iterable");
        }

        /// <summary>A string's length in code points, Python's <c>len(str)</c>.</summary>
        /// <param name="text">The string.</param>
        /// <returns>The length.</returns>
        private static int CodePointLength(string text) {
            int length = 0;
            foreach (Rune _ in text.EnumerateRunes())
                length++;
            return length;
        }

        /// <summary>Python's <c>s.rstrip(" ")</c> on a builder: drops trailing spaces only.</summary>
        /// <param name="text">The builder.</param>
        private static void TrimEndSpaces(StringBuilder text) {
            int end = text.Length;
            while (end > 0 && text[end - 1] == ' ')
                end--;
            text.Length = end;
        }

        /// <summary>A dict view of metadata-like values, Python's <c>isinstance(x, dict)</c> for the forms Gym.NET uses.</summary>
        private static class Dict {
            /// <summary>The value as a string-keyed dictionary, or <see langword="null"/> when it is no dict.</summary>
            /// <param name="value">The value.</param>
            /// <returns>The view.</returns>
            public static IReadOnlyDictionary<string, object?>? Of(object value) => value switch {
                IReadOnlyDictionary<string, object?> dictionary => dictionary,
                IDictionary dictionary when dictionary.Keys.Cast<object>().All(key => key is string) => dictionary.Keys.Cast<string>().ToDictionary(key => key, key => dictionary[key]),
                _ => null,
            };

            /// <summary>Python's <c>d.get(key)</c> on a dict-like value.</summary>
            /// <param name="value">The dict.</param>
            /// <param name="key">The key.</param>
            /// <returns>The value, or <see langword="null"/>.</returns>
            public static object? Get(object value, string key) => Of(value) is { } dictionary && dictionary.TryGetValue(key, out object? item) ? item : null;
        }

        /// <summary>The scope <see cref="Namespace"/> returns: restores the namespace that was current before.</summary>
        private sealed class NamespaceScope : IDisposable {
            /// <summary>The namespace to restore.</summary>
            private readonly string? _old;

            /// <summary>Whether the scope was already disposed.</summary>
            private bool _disposed;

            /// <summary>Remembers the namespace to restore.</summary>
            /// <param name="old">The namespace current before the scope.</param>
            public NamespaceScope(string? old) => _old = old;

            /// <summary>
            ///     Restores the namespace, once. Disposed on the flow that opened it (the <see langword="using"/> pattern),
            ///     it restores that flow's value; disposed on another thread, it would set that thread's instead.
            /// </summary>
            public void Dispose() {
                if (_disposed)
                    return;
                _disposed = true;
                CurrentNamespace = _old;
            }
        }
    }
}
