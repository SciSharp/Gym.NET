using System;
using System.Collections.Generic;
using System.Reflection;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Utils;
using NumSharp.Interop.PythonNet;
using Python.Runtime;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Parity.Bridge {
    /// <summary>
    ///     The reverse bridge (the interop design's §8, completion plan P3.5): Gym.NET envs as Gymnasium envs, so Gymnasium's
    ///     own tools (<c>check_env</c>, <c>check_environments_match</c>, <c>make</c>'s wrappers and passive checker) run on
    ///     them unchanged, rather than only Gym.NET's ports of those tools.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The Python side is one class, <c>GymNetShim(gymnasium.Env)</c> (<see cref="ShimStatements"/>), whose every method
    ///     crosses to a <see cref="BridgedEnv"/>. <see cref="ShimClass"/> subclasses it once per .NET env class, named after
    ///     that class (<c>CartPoleEnv</c>, module <c>Gym.Environments.Envs.Classic</c>) and carrying the class's metadata, so
    ///     the subclass is an entry point <c>gymnasium.make</c> takes like any env class: its kwargs reach the Gym.NET
    ///     constructor by Python name, and <c>str(env)</c> reads as Gymnasium's own. <see cref="Spec"/> mirrors a Gym.NET
    ///     spec with that entry point, so <c>gymnasium.make(spec)</c> builds Gymnasium's stack over a Gym.NET env, and
    ///     <c>spec.make(render_mode=...)</c> inside <c>check_env</c> builds further bridged envs.
    ///     </para>
    ///     <para>
    ///     Spaces are mirrored as the Gymnasium spaces they equal (<see cref="Space"/>), so <c>env_a.action_space ==
    ///     env_b.action_space</c> holds against a Gymnasium env, and actions are sampled on the Python side. The env's
    ///     generator is mirrored after every call (<see cref="BridgedEnv.NpRandomState"/>); observations cross as zero-copy
    ///     views, so a buffer Gym.NET reuses is a buffer Gymnasium's data-reuse checks see reused.
    ///     </para>
    ///     <para>
    ///     Ownership: every .NET env built through this bridge, by a test or by Gymnasium's tools (<c>check_env</c> makes one
    ///     per render mode), belongs to it and is disposed with it. Python objects may outlive it (a shim kept in Python's
    ///     garbage); calls on them then fail with <see cref="ObjectDisposedException"/>. Every member that touches Python
    ///     must run under the GIL (<see cref="PythonRuntime.Run(Action)"/>, <see cref="PyRef.Of"/>).
    ///     </para>
    ///     <para>
    ///     Scope: the shim exposes Gymnasium's <c>Env</c> surface; attributes of the .NET env beyond it (CartPole's
    ///     <c>state</c>, <c>tau</c>) stay on the .NET side (<see cref="Envs"/>). Spaces mirror for <c>Box</c>,
    ///     <c>Discrete</c>, <c>Tuple</c> and <c>Dict</c>, which every env in this run uses; the others are refused by name.
    ///     </para>
    /// </remarks>
    internal sealed class ReverseBridge : IDisposable {
        /// <summary>
        ///     The Gymnasium-side env class, a snippet (the pythonic skill's flavour B) since it is a Python class definition,
        ///     line by line as the Python it is. <c>reset</c> has Gymnasium's keyword-only
        ///     <c>seed</c>/<c>options</c> signature because <c>check_env</c> inspects it; seeding happens on the .NET side, so
        ///     the shim never calls <c>super().reset(seed=seed)</c> (that would make a Python generator the env does not draw
        ///     from) and mirrors the .NET generator instead, a new NumPy generator exactly when the env made a new one.
        /// </summary>
        private const string ShimStatements =
            "import numpy as np\n" +
            "import gymnasium as gym\n" +
            "\n" +
            "\n" +
            "class GymNetShim(gym.Env):\n" +
            "    _factory = None\n" +
            "\n" +
            "    def __init__(self, **kwargs):\n" +
            "        self._env = type(self)._factory.Create(kwargs)\n" +
            "        self._spec = None\n" +
            "        self._np_random_generation = None\n" +
            "        self.metadata = self._env.Metadata()\n" +
            "        self.render_mode = self._env.RenderMode()\n" +
            "        self.observation_space = self._env.ObservationSpace()\n" +
            "        self.action_space = self._env.ActionSpace()\n" +
            "        self._sync_np_random()\n" +
            "\n" +
            "    def reset(self, *, seed=None, options=None):\n" +
            "        result = self._env.Reset(seed, options)\n" +
            "        self._sync_np_random()\n" +
            "        return result\n" +
            "\n" +
            "    def step(self, action):\n" +
            "        result = self._env.Step(action)\n" +
            "        self._sync_np_random()\n" +
            "        return result\n" +
            "\n" +
            "    def render(self):\n" +
            "        return self._env.Render()\n" +
            "\n" +
            "    def close(self):\n" +
            "        self._env.Close()\n" +
            "\n" +
            "    @property\n" +
            "    def spec(self):\n" +
            "        return self._spec\n" +
            "\n" +
            "    @spec.setter\n" +
            "    def spec(self, value):\n" +
            "        self._spec = value\n" +
            "        if value is None:\n" +
            "            self._env.ClearSpec()\n" +
            "        else:\n" +
            "            self._env.SetSpec(value.id, value.reward_threshold, value.nondeterministic, value.max_episode_steps,\n" +
            "                              value.order_enforce, value.disable_env_checker, value.kwargs)\n" +
            "\n" +
            "    @property\n" +
            "    def np_random(self):\n" +
            "        self._env.CreateNpRandom()\n" +
            "        self._sync_np_random()\n" +
            "        return self._np_random\n" +
            "\n" +
            "    @np_random.setter\n" +
            "    def np_random(self, value):\n" +
            "        raise AttributeError(\"GymNetShim mirrors its Gym.NET env's generator; assign IEnv.NpRandom on the .NET side\")\n" +
            "\n" +
            "    @property\n" +
            "    def np_random_seed(self):\n" +
            "        seed = self._env.NpRandomSeed()\n" +
            "        self._sync_np_random()\n" +
            "        return seed\n" +
            "\n" +
            "    def _sync_np_random(self):\n" +
            "        mirror = self._env.NpRandomState()\n" +
            "        if mirror is None:\n" +
            "            self._np_random = self._np_random_seed = self._np_random_generation = None\n" +
            "            return\n" +
            "        generation, seed, state = mirror\n" +
            "        if generation != self._np_random_generation:\n" +
            "            self._np_random = np.random.Generator(np.random.PCG64())\n" +
            "            self._np_random_generation = generation\n" +
            "        self._np_random.bit_generator.state = state\n" +
            "        self._np_random_seed = seed\n";

        /// <summary>A Gymnasium <c>EnvSpec</c> built from bound fields: its constructor is keyword-heavy, so it is a snippet (as the facade's <c>EnvSpec</c> notes).</summary>
        private const string SpecStatements =
            "from gymnasium.envs.registration import EnvSpec\n" +
            "spec = EnvSpec(id, entry_point=entry_point, reward_threshold=reward_threshold, nondeterministic=nondeterministic,\n" +
            "               max_episode_steps=max_episode_steps, order_enforce=order_enforce, disable_env_checker=disable_env_checker,\n" +
            "               kwargs=kwargs)\n";

        /// <summary>Guards the creation of <see cref="_shimBase"/>.</summary>
        private static readonly object ShimLock = new();

        /// <summary>
        ///     <c>GymNetShim</c>, defined once per process and never released: a process-scoped class object, like the cached
        ///     modules (the pythonic skill's long-lived singletons).
        /// </summary>
        private static PyObject? _shimBase;

        /// <summary>Every env built through this bridge, in order; guarded by itself.</summary>
        private readonly List<BridgedEnv> _envs = new();

        /// <summary>Set by <see cref="Dispose"/>.</summary>
        private bool _disposed;

        /// <summary>The .NET envs built through this bridge so far, oldest first (by a test or by Gymnasium's tools).</summary>
        internal IReadOnlyList<IEnv> Envs {
            get {
                lock (_envs)
                    return _envs.ConvertAll(bridged => bridged.Env);
            }
        }

        /// <summary>
        ///     The Gymnasium-side class standing for <paramref name="envType"/>: a <c>GymNetShim</c> subclass named after it,
        ///     whose constructor builds a .NET env (owned by this bridge) from its kwargs. Must run under the GIL.
        /// </summary>
        /// <param name="envType">The Gym.NET env class.</param>
        /// <param name="create">
        ///     How to build the env from the kwargs; <see langword="null"/> (the default) constructs <paramref name="envType"/>
        ///     with them by Python name (<see cref="PyKwargs.Construct"/>), as Gym.NET's <c>make</c> does. A test plants a defect
        ///     through it.
        /// </param>
        /// <returns>A new owned reference to the class; a new class per call, with its own factory.</returns>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        /// <exception cref="PythonException">Thrown when the class cannot be created.</exception>
        internal PyObject ShimClass(Type envType, Func<Info, IEnv>? create = null) {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ReverseBridge));
            var factory = new BridgedEnvFactory(envType, create ?? (kwargs => (IEnv)PyKwargs.Construct(envType, kwargs)), Register);
            using var bases = new PyTuple(new[] { ShimBase });
            using var body = new PyDict();
            using PyObject factoryObject = factory.ToPython();
            using PyObject metadata = PyConvert.ToPython(ClassMetadataOf(envType));
            using var module = new PyString(envType.Namespace ?? "");
            body["_factory"] = factoryObject;
            body["metadata"] = metadata;
            // The .NET namespace as the module, so the class reads as the .NET class it stands for: Gym.Environments.Envs.Classic.CartPoleEnv.
            body["__module__"] = module;
            return Python.builtins.type(PyAttributes.PyTypeName(envType), bases, body);
        }

        /// <summary>
        ///     A Gymnasium-side env over a new .NET env, built by calling <see cref="ShimClass"/>'s class with
        ///     <paramref name="kwargs"/> (no spec, no wrappers: Gymnasium's <c>CartPoleEnv(render_mode=...)</c>). Must run under
        ///     the GIL.
        /// </summary>
        /// <param name="envType">The Gym.NET env class.</param>
        /// <param name="kwargs">The constructor's keyword arguments by Python name, or <see langword="null"/>; converted by <see cref="PyConvert.ToPython"/>.</param>
        /// <param name="create">How to build the env, as for <see cref="ShimClass"/>.</param>
        /// <returns>A new owned reference to the env.</returns>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        /// <exception cref="PythonException">Thrown with what the construction raised in Python (a <c>TypeError</c> for an unknown keyword).</exception>
        internal PyObject Shim(Type envType, IReadOnlyDictionary<string, object?>? kwargs = null, Func<Info, IEnv>? create = null) {
            using PyObject shimClass = ShimClass(envType, create);
            using PyDict arguments = PyConvert.ToKwargs(kwargs);
            // Calling the class this bridge just made: a run-time object, so a raw call rather than a facade row.
            return shimClass.Invoke(Array.Empty<PyObject>(), arguments);
        }

        /// <summary>
        ///     A Gymnasium <c>EnvSpec</c> mirroring <paramref name="spec"/> (id, reward threshold, determinism, step limit, order
        ///     enforcement, checker, kwargs), whose entry point is <see cref="ShimClass"/>'s class for the spec's entry point:
        ///     <c>gymnasium.make(mirror)</c> then builds Gymnasium's own stack over a Gym.NET env. Must run under the GIL.
        /// </summary>
        /// <param name="spec">The Gym.NET spec; its entry point is a class or a <c>"Namespace:Class"</c> string.</param>
        /// <param name="create">How to build the env, as for <see cref="ShimClass"/>.</param>
        /// <returns>A new owned reference to the Gymnasium spec.</returns>
        /// <exception cref="ObjectDisposedException">Thrown after <see cref="Dispose"/>.</exception>
        /// <exception cref="NotSupportedException">Thrown for an entry point that is a creator delegate (no class to name) or a spec with additional wrappers.</exception>
        /// <exception cref="PythonException">Thrown when Gymnasium refuses the mirrored fields.</exception>
        internal PyObject Spec(EnvSpec spec, Func<Info, IEnv>? create = null) {
            Type envType = spec.EntryPoint switch {
                Type type => type,
                string name => EnvRegistry.LoadEnvCreator(name),
                _ => throw new NotSupportedException($"The bridge mirrors specs whose entry point is a class; {spec.Id}'s is {spec.EntryPoint?.GetType().Name ?? "None"}."),
            };
            if (spec.AdditionalWrappers.Count > 0)
                throw new NotSupportedException($"The bridge mirrors specs without additional wrappers; {spec.Id} has {spec.AdditionalWrappers.Count}.");

            using PyObject entryPoint = ShimClass(envType, create);
            using var id = new PyString(spec.Id);
            using PyObject rewardThreshold = PyConvert.ToPython(spec.RewardThreshold);
            using PyObject nondeterministic = spec.Nondeterministic.ToPython();
            using PyObject maxEpisodeSteps = PyConvert.ToPython(spec.MaxEpisodeSteps);
            using PyObject orderEnforce = spec.OrderEnforce.ToPython();
            using PyObject disableEnvChecker = spec.DisableEnvChecker.ToPython();
            using PyObject kwargs = PyConvert.ToPython(spec.Kwargs);
            return Python.evalStmts(SpecStatements, "spec", ("id", id), ("entry_point", entryPoint), ("reward_threshold", rewardThreshold),
                ("nondeterministic", nondeterministic), ("max_episode_steps", maxEpisodeSteps), ("order_enforce", orderEnforce),
                ("disable_env_checker", disableEnvChecker), ("kwargs", kwargs));
        }

        /// <summary>
        ///     The Gymnasium space <paramref name="space"/> equals: same type, bounds, dtype and structure, so <c>==</c> holds
        ///     against Gymnasium's and the <c>repr</c> is Gymnasium's. The mirror is unseeded: it samples from its own Python
        ///     generator. Must run under the GIL.
        /// </summary>
        /// <param name="space">The Gym.NET space; a typed env space's <c>ObjectSpace</c> view is mirrored as the space it views.</param>
        /// <returns>A new owned reference.</returns>
        /// <exception cref="NotSupportedException">Thrown for a space type the bridge does not mirror (only <c>Box</c>, <c>Discrete</c>, <c>Tuple</c> and <c>Dict</c> are).</exception>
        /// <exception cref="PythonException">Thrown when Gymnasium refuses the mirrored parameters.</exception>
        internal static PyObject Space(Space space) {
            switch (space) {
                case ObjectSpace view:
                    return Space(view.Inner);
                case Discrete discrete: {
                    using PyObject dtype = Python.numpy.dtype(discrete.DType.str);
                    return Python.gymnasium.spaces.Discrete(discrete.N, null, discrete.Start, dtype);
                }
                case Box box: {
                    // Copies: Gymnasium's Box keeps its bounds, which must not alias the .NET space's memory.
                    using PyObject low = box.Low.ToNumpyCopy();
                    using PyObject high = box.High.ToNumpyCopy();
                    using PyObject dtype = Python.numpy.dtype(box.DType.str);
                    return Python.gymnasium.spaces.Box(low, high, dtype);
                }
                case Tuple tuple: {
                    var items = new PyObject[tuple.Count];
                    try {
                        for (int i = 0; i < items.Length; i++)
                            items[i] = Space(tuple.Spaces[i]);
                        using var subspaces = new PyTuple(items);
                        return Python.gymnasium.spaces.Tuple(subspaces);
                    } finally {
                        // PyTuple takes its own references to the items.
                        foreach (PyObject? item in items)
                            item?.Dispose();
                    }
                }
                case Dict dict: {
                    // (key, space) pairs keep Gym.NET's key order, which is final (already sorted or not, per sort_keys).
                    using var pairs = new PyList();
                    foreach (string key in dict.Keys) {
                        using PyObject subspace = Space(dict.Spaces[key]);
                        using var name = new PyString(key);
                        using var pair = new PyTuple(new[] { name, subspace });
                        pairs.Append(pair);
                    }

                    return Python.gymnasium.spaces.Dict(pairs, null, dict.SortKeys);
                }
                default:
                    throw new NotSupportedException($"The bridge mirrors Box, Discrete, Tuple and Dict spaces; {space.GetType().Name} is not mirrored yet.");
            }
        }

        /// <summary>
        ///     A Gymnasium outcome on a bridged env, presented as <see cref="ByteContract"/>'s Gym.NET side, for cells that hold
        ///     a Gymnasium tool's verdict on a bridged env to its verdict on Gymnasium's own. The contract maps the left
        ///     (Gymnasium) side's raised Python type to the .NET type it expects on the right; there the right side is Python too
        ///     (the bridge raises what Gym.NET threw as the Python type it stands for), so its type is mapped the same way and
        ///     the two compare like with like. A type with no mapping is left as it is and fails the comparison, named.
        /// </summary>
        /// <param name="bridged">The outcome of a Gymnasium call on the bridged env.</param>
        /// <returns>The same outcome, its raised type (if any) in .NET spelling.</returns>
        internal static Outcome AsBridgedSide(Outcome bridged) =>
            bridged.Raised is { } raised ? Outcome.Threw(raised with { Type = ExceptionMap.DotNetTypeFor(raised.Type) ?? raised.Type }) : bridged;

        /// <summary>Disposes every env built through this bridge (each releases its handed-off object under the GIL). Safe to call twice.</summary>
        public void Dispose() {
            if (_disposed)
                return;
            _disposed = true;
            BridgedEnv[] envs;
            lock (_envs) {
                envs = _envs.ToArray();
                _envs.Clear();
            }

            foreach (BridgedEnv env in envs)
                env.Dispose();
        }

        /// <summary><c>GymNetShim</c>, defined on first use. Must run under the GIL; the returned object is borrowed.</summary>
        /// <exception cref="PythonException">Thrown when the class definition fails.</exception>
        private static PyObject ShimBase {
            get {
                lock (ShimLock)
                    return _shimBase ??= Python.evalStmts(ShimStatements, "GymNetShim");
            }
        }

        /// <summary>Takes ownership of an env a factory built.</summary>
        /// <param name="env">The env.</param>
        private void Register(BridgedEnv env) {
            lock (_envs)
                _envs.Add(env);
        }

        /// <summary>
        ///     The env class's class-level metadata (its static <c>ClassMetadata</c>, the most derived declaration), which
        ///     <c>gymnasium.make</c> reads from the entry point before the env exists, as Gym.NET's own <c>make</c> does.
        /// </summary>
        /// <param name="type">The env class.</param>
        /// <returns>The metadata, or Gymnasium's default <c>{"render_modes": []}</c> when the class declares none.</returns>
        /// <remarks>
        ///     Walks the hierarchy declaration by declaration: an env class hides <c>Env</c>'s property with <c>new</c>, so a
        ///     flattened lookup would find two. The same walk as Gym.NET's registry (whose helper is private).
        /// </remarks>
        private static object ClassMetadataOf(Type type) {
            const BindingFlags flags = BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly;
            for (Type? level = type; level != null; level = level.BaseType) {
                if (level.GetProperty("ClassMetadata", flags) is { } property && property.GetIndexParameters().Length == 0)
                    return property.GetValue(null) ?? new Metadata { [Gym.Envs.Metadata.RenderModesKey] = new List<string>() };
                if (level.GetField("ClassMetadata", flags) is { } field)
                    return field.GetValue(null) ?? new Metadata { [Gym.Envs.Metadata.RenderModesKey] = new List<string>() };
            }

            return new Metadata { [Gym.Envs.Metadata.RenderModesKey] = new List<string>() };
        }
    }
}
