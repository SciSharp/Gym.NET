#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     The base of every vector env (DR-21): <see cref="NumEnvs"/> sub-envs stepped in lockstep, typed by the batched
    ///     observation and action types (<see cref="NDArray"/> for array spaces, <see langword="object"/>?[] for a
    ///     <c>Tuple</c> space, <see cref="Info"/> for a <c>Dict</c> space). Rewards are a float64 array and the flags bool
    ///     arrays, one element per sub-env; infos are merged by key with Gymnasium's <c>_key</c> masks (<see cref="AddInfo"/>).
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, an element of <see cref="ObservationSpace"/>.</typeparam>
    /// <typeparam name="TAct">The batched action type, an element of <see cref="ActionSpace"/>.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Implementing a vector env: set the four spaces, <see cref="NumEnvs"/> and <see cref="Metadata"/> (with its
    ///     <c>autoreset_mode</c>) in the constructor; begin <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/>
    ///     with <see cref="SeedNpRandom"/> when the env draws itself (Gymnasium's <c>super().reset(seed=seed)</c>). Python
    ///     attributes map to properties by snake_case (<c>num_envs</c>, <c>single_action_space</c>).
    ///     </para>
    ///     <para>
    ///     Deviations that follow from C#: <see cref="Metadata"/> is per instance (Python's class attribute is shared), and
    ///     an integer seed and a list of seeds are two overloads of <c>Reset</c> where Python takes one argument.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L45">gymnasium/vector/vector_env.py</see>.</para>
    /// </remarks>
    public abstract class VectorEnv<TObs, TAct> : IVectorEnv {
        /// <summary>The metadata; Gymnasium's class attribute is an empty dict until a vector env sets its own.</summary>
        private Metadata _metadata = new();

        /// <summary>The batched observation space, <see langword="null"/> while unset.</summary>
        private Space<TObs>? _observationSpace;

        /// <summary>The batched action space, <see langword="null"/> while unset.</summary>
        private Space<TAct>? _actionSpace;

        /// <summary>One sub-env's observation space, <see langword="null"/> while unset.</summary>
        private Space? _singleObservationSpace;

        /// <summary>One sub-env's action space, <see langword="null"/> while unset.</summary>
        private Space? _singleActionSpace;

        /// <summary>The number of sub-envs, <see langword="null"/> while unset (Python's annotated-only <c>num_envs</c>).</summary>
        private int? _numEnvs;

        /// <summary>Python's <c>_np_random</c>: <see langword="null"/> until first used or seeded.</summary>
        private Generator? _npRandom;

        /// <summary>Python's <c>_np_random_seed</c>: <see langword="null"/> until first used or seeded, -1 once a generator was assigned.</summary>
        private BigInteger? _npRandomSeed;

        /// <summary>The metadata (<c>metadata</c>); a vector env sets <c>autoreset_mode</c> in it, which <c>MakeVec</c> and the observation wrappers check.</summary>
        /// <exception cref="ArgumentNullException">Thrown when assigning <see langword="null"/>.</exception>
        public virtual Metadata Metadata {
            get => _metadata;
            set => _metadata = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>The spec <c>MakeVec</c> attached (<c>spec</c>), which the vector env's <c>repr</c> names.</summary>
        public virtual EnvSpec? Spec { get; set; }

        /// <summary>The render mode (<c>render_mode</c>), the sub-envs' for <see cref="SyncVectorEnv{TObs, TAct}"/>.</summary>
        public virtual string? RenderMode { get; protected set; }

        /// <summary>Whether <see cref="Close"/> ran (<c>closed</c>).</summary>
        public virtual bool Closed { get; set; }

        /// <summary>The number of sub-envs (<c>num_envs</c>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset, as Python's annotated-only attribute does.</exception>
        public virtual int NumEnvs {
            get => _numEnvs ?? throw MissingAttribute("num_envs");
            protected set => _numEnvs = value;
        }

        /// <summary>
        ///     The batched observation space (<c>observation_space</c>), typed by the batched observation type (DR-21). The vector
        ///     env owns and disposes it, so the setter refuses a space another env not yet disposed owns (DR-40).
        /// </summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="InvalidOperationException">Thrown by the setter for a space another env not yet disposed owns.</exception>
        public virtual Space<TObs> ObservationSpace {
            get => _observationSpace ?? throw MissingAttribute("observation_space");
            set => _observationSpace = Space.Assign(this, value, _observationSpace, _actionSpace, "observation_space");
        }

        /// <summary>The batched action space (<c>action_space</c>), typed by the batched action type; owned as <see cref="ObservationSpace"/> is.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="InvalidOperationException">Thrown by the setter for a space another env not yet disposed owns.</exception>
        public virtual Space<TAct> ActionSpace {
            get => _actionSpace ?? throw MissingAttribute("action_space");
            set => _actionSpace = Space.Assign(this, value, _actionSpace, _observationSpace, "action_space");
        }

        /// <summary>One sub-env's observation space (<c>single_observation_space</c>), untyped: its elements may be of another type than the batch's.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        public virtual Space SingleObservationSpace {
            get => _singleObservationSpace ?? throw MissingAttribute("single_observation_space");
            set => _singleObservationSpace = value;
        }

        /// <summary>One sub-env's action space (<c>single_action_space</c>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        public virtual Space SingleActionSpace {
            get => _singleActionSpace ?? throw MissingAttribute("single_action_space");
            set => _singleActionSpace = value;
        }

        /// <summary>The batched observation space, untyped.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="TypeError">Thrown by the setter for a space whose elements are not <typeparamref name="TObs"/>.</exception>
        Space IVectorEnv.ObservationSpace {
            get => ObservationSpace;
            set => ObservationSpace = Typed<TObs>(value, "observation_space");
        }

        /// <summary>The batched action space, untyped.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="TypeError">Thrown by the setter for a space whose elements are not <typeparamref name="TAct"/>.</exception>
        Space IVectorEnv.ActionSpace {
            get => ActionSpace;
            set => ActionSpace = Typed<TAct>(value, "action_space");
        }

        /// <summary>
        ///     The vector env's own generator (<c>np_random</c>), created from OS entropy on first read as Gymnasium does; a
        ///     seeded <c>Reset</c> of a vector env that draws itself replaces it. Assigning one shares it and makes
        ///     <see cref="NpRandomSeed"/> -1.
        /// </summary>
        public virtual Generator NpRandom {
            get {
                if (_npRandom is null)
                    (_npRandom, _npRandomSeed) = Seeding.NpRandom();
                return _npRandom;
            }
            set {
                _npRandom = value;
                // A generator's seed cannot be recovered, so it is marked unknown.
                _npRandomSeed = -1;
            }
        }

        /// <summary>The seed of <see cref="NpRandom"/> (<c>np_random_seed</c>), creating the generator from entropy on first read; -1 after one was assigned.</summary>
        public virtual BigInteger NpRandomSeed {
            get {
                if (_npRandomSeed is null)
                    (_npRandom, _npRandomSeed) = Seeding.NpRandom();
                return _npRandomSeed.Value;
            }
        }

        /// <summary>This vector env: the innermost one under its wrappers (<c>unwrapped</c>).</summary>
        public virtual IVectorEnv Unwrapped => this;

        /// <summary>Python's <c>_np_random</c>, read without creating a generator.</summary>
        [PyName("_np_random")]
        protected virtual Generator? RawNpRandom {
            get => _npRandom;
            set => _npRandom = value;
        }

        /// <summary>Python's <c>_np_random_seed</c>, read without creating a generator.</summary>
        [PyName("_np_random_seed")]
        protected virtual BigInteger? RawNpRandomSeed {
            get => _npRandomSeed;
            set => _npRandomSeed = value;
        }

        /// <summary>Resets every sub-env (<c>reset(seed=seed, options=options)</c>).</summary>
        /// <param name="seed">The seed, or <see langword="null"/> to keep the generators.</param>
        /// <param name="options">The reset options.</param>
        /// <returns>The batched first observations and the merged infos; the caller owns the arrays.</returns>
        public abstract VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null);

        /// <summary>
        ///     Resets every sub-env with a seed each (<c>reset(seed=[...])</c>). The base vector env takes only an integer seed:
        ///     Gymnasium's passes the list to <c>seeding.np_random</c>, which refuses it, so this raises that error unless a
        ///     vector env (<see cref="SyncVectorEnv{TObs, TAct}"/>) seeds its sub-envs one by one.
        /// </summary>
        /// <param name="seeds">One seed per sub-env.</param>
        /// <param name="options">The reset options.</param>
        /// <returns>The batched first observations and the merged infos.</returns>
        /// <exception cref="Error">Thrown here: <c>Seed must be a python integer, actual type: &lt;class 'list'&gt;</c>.</exception>
        public virtual VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) =>
            throw new Error("Seed must be a python integer, actual type: <class 'list'>");

        /// <summary>Steps every sub-env with its action (<c>step(actions)</c>).</summary>
        /// <param name="actions">One action per sub-env.</param>
        /// <returns>The batched transition; the caller owns its arrays.</returns>
        public abstract VectorStepResult<TObs> Step(TAct actions);

        /// <summary>Renders every sub-env (<c>render()</c>). The base vector env does not render.</summary>
        /// <returns>One result per sub-env, or <see langword="null"/>.</returns>
        /// <exception cref="NotImplementedError">Thrown here: <c>&lt;repr&gt; render function is not implemented.</c></exception>
        public virtual IReadOnlyList<RenderResult>? Render() => throw new NotImplementedError($"{this} render function is not implemented.");

        /// <summary>Closes the vector env once (<c>close()</c>): <see cref="CloseExtras"/>, then <see cref="Closed"/>; a second call does nothing.</summary>
        public virtual void Close() {
            if (Closed)
                return;
            CloseExtras();
            Closed = true;
        }

        /// <summary>What a vector env releases on close (<c>close_extras</c>): its sub-envs, its windows. Nothing here.</summary>
        protected virtual void CloseExtras() { }

        /// <summary>
        ///     <see cref="CloseExtras"/> for a vector wrapper, which forwards its own <c>close_extras</c> to the env it wraps
        ///     (Python's <c>self.env.close_extras()</c>) but cannot reach another instance's protected member.
        /// </summary>
        internal void CloseExtrasCore() => CloseExtras();

        /// <summary>Closes the vector env and disposes the batched spaces it owns.</summary>
        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///     Closes the vector env and, when disposing, the batched spaces: a vector env builds them by batching, so it owns
        ///     them; the single spaces belong to the sub-envs and are left to them.
        /// </summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Dispose()"/>.</param>
        protected virtual void Dispose(bool disposing) {
            Close();
            if (!disposing)
                return;
            _observationSpace?.Dispose();
            _actionSpace?.Dispose();
        }

        /// <summary>
        ///     Python's <c>repr</c>: <c>SyncVectorEnv(num_envs=3)</c>, or with the spec's id, <c>SyncVectorEnv(CartPole-v1,
        ///     num_envs=3)</c>.
        /// </summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            string name = PyAttributes.PyTypeName(GetType());
            return Spec is null ? $"{name}(num_envs={NumEnvs})" : $"{name}({Spec.Id}, num_envs={NumEnvs})";
        }

        /// <summary>
        ///     Seeds <see cref="NpRandom"/> as Gymnasium's base <c>reset</c> does: only when a seed is given; without one the
        ///     generator keeps its stream.
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        protected void SeedNpRandom(BigInteger? seed) {
            if (seed is not null)
                (_npRandom, _npRandomSeed) = Seeding.NpRandom(seed);
        }

        /// <summary>
        ///     Gymnasium's <c>_add_info</c>: merges one sub-env's info into the vector infos. Each key gets a batch of the
        ///     sub-envs' values and a <c>_key</c> mask of the sub-envs that reported it; a nested dict is merged recursively,
        ///     and <c>final_obs</c> collects the sub-envs' observations unbatched.
        /// </summary>
        /// <param name="vectorInfos">The infos merged so far; updated in place and returned.</param>
        /// <param name="envInfo">The sub-env's info.</param>
        /// <param name="envNum">The sub-env's index.</param>
        /// <returns><paramref name="vectorInfos"/>.</returns>
        /// <remarks>
        ///     <para>
        ///     A key's batch is created from the first value seen, as NumPy does: a number makes a <c>(num_envs,)</c> array of
        ///     its type (a .NET integer is a Python <c>int</c>, so int64; a <see langword="double"/> float64; a
        ///     <see langword="bool"/> bool; a 0-d array, NumPy's scalar, its own dtype, except a bool one, which NumPy does not
        ///     count as a number), an array of one or more dimensions makes a <c>(num_envs, *shape)</c> array of its dtype,
        ///     and anything else an <see langword="object"/>?[] (NumPy's object array). Later values are cast into the batch
        ///     as NumPy's assignment casts them: a float into an integer batch truncates, so mixing a reset info's
        ///     <c>prob=1</c> with steps' <c>prob=0.33</c> in one call loses the fraction, exactly as in Gymnasium.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L284">gymnasium/vector/vector_env.py</see>.</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vectorInfos"/> or <paramref name="envInfo"/> is <see langword="null"/>.</exception>
        protected Info AddInfo(Info vectorInfos, Info envInfo, int envNum) => VectorInfos.Add(vectorInfos, envInfo, envNum, NumEnvs);

        /// <summary>
        ///     The untyped step's action conversion: the actions as given when they are <typeparamref name="TAct"/> (or the
        ///     batched type is <see langword="object"/>), a .NET array as an <see cref="NDArray"/> for an array batch.
        /// </summary>
        /// <param name="actions">The actions as given.</param>
        /// <returns>The typed actions: the ones given, or a new array the step releases.</returns>
        /// <exception cref="TypeError">Thrown for actions of another type, naming both.</exception>
        protected virtual TAct ConvertActions(object? actions) {
            if (actions is TAct typed)
                return typed;
            if (typeof(TAct) == typeof(NDArray) && actions is Array array)
                return (TAct)(object)np.array(array);
            throw new TypeError($"{PyAttributes.PyTypeName(GetType())} takes actions of type {typeof(TAct).Name}, not {PyBuiltins.Repr(actions)} ({PyBuiltins.TypeRepr(actions)})");
        }

        /// <summary>The untyped reset, the observations boxed.</summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <param name="options">The reset options.</param>
        /// <returns>The result.</returns>
        VectorResetResult<object?> IVectorEnv.Reset(BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            VectorResetResult<TObs> result = Reset(seed, options);
            return new VectorResetResult<object?>(result.Observations, result.Infos);
        }

        /// <summary>The untyped reset with one seed per sub-env, the observations boxed.</summary>
        /// <param name="seeds">The seeds.</param>
        /// <param name="options">The reset options.</param>
        /// <returns>The result.</returns>
        VectorResetResult<object?> IVectorEnv.Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options) {
            VectorResetResult<TObs> result = Reset(seeds, options);
            return new VectorResetResult<object?>(result.Observations, result.Infos);
        }

        /// <summary>The untyped step: converts the actions (<see cref="ConvertActions"/>), steps, and boxes the observations.</summary>
        /// <param name="actions">The actions.</param>
        /// <returns>The result.</returns>
        VectorStepResult<object?> IVectorEnv.Step(object? actions) {
            TAct typed = ConvertActions(actions);
            try {
                VectorStepResult<TObs> result = Step(typed);
                return new VectorStepResult<object?>(result.Observations, result.Rewards, result.Terminations, result.Truncations, result.Infos);
            } finally {
                // An array made by the conversion is this call's own; the caller's actions are left alone.
                if (!ReferenceEquals(typed, actions) && typed is IDisposable disposable)
                    disposable.Dispose();
            }
        }

        /// <summary>A space assigned through the untyped view, checked against the batched type.</summary>
        /// <typeparam name="T">The batched element type.</typeparam>
        /// <param name="space">The space.</param>
        /// <param name="attribute">The Python attribute, for the error.</param>
        /// <returns>The typed space.</returns>
        /// <exception cref="TypeError">Thrown for a space of other elements.</exception>
        private Space<T> Typed<T>(Space? space, string attribute) {
            if (space is Space<T> typed)
                return typed;
            if (space is not null && typeof(T) == typeof(object))
                return (Space<T>)(object)new ObjectSpace(space);
            throw new TypeError($"{PyAttributes.PyTypeName(GetType())}.{attribute} must be a space of {typeof(T).Name}, not {(space is null ? "None" : space.GetType().Name)}");
        }

        /// <summary>The error Python raises for an attribute that was declared but never assigned.</summary>
        /// <param name="name">The attribute.</param>
        /// <returns>The error to throw.</returns>
        private AttributeError MissingAttribute(string name) => new($"'{PyAttributes.PyTypeName(GetType())}' object has no attribute '{name}'");
    }

    /// <summary>
    ///     Gymnasium's <c>_add_info</c> on Gym.NET's values (<see cref="VectorEnv{TObs, TAct}.AddInfo"/>), shared with the
    ///     vector wrappers that merge infos the same way.
    /// </summary>
    internal static class VectorInfos {
        /// <summary>Merges one sub-env's info into the vector infos; see <see cref="VectorEnv{TObs, TAct}.AddInfo"/>.</summary>
        /// <param name="vectorInfos">The infos merged so far; updated in place and returned.</param>
        /// <param name="envInfo">The sub-env's info.</param>
        /// <param name="envNum">The sub-env's index.</param>
        /// <param name="numEnvs">The number of sub-envs, the batches' length.</param>
        /// <returns><paramref name="vectorInfos"/>.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="vectorInfos"/> or <paramref name="envInfo"/> is <see langword="null"/>.</exception>
        internal static Info Add(Info vectorInfos, Info envInfo, int envNum, int numEnvs) {
            if (vectorInfos is null)
                throw new ArgumentNullException(nameof(vectorInfos));
            if (envInfo is null)
                throw new ArgumentNullException(nameof(envInfo));

            foreach (KeyValuePair<string, object?> entry in envInfo) {
                string key = entry.Key;
                object? value = entry.Value;
                object array;
                if (key == "final_obs") {
                    // It is easier for users to access their final observations unbatched, in an object array.
                    object?[] observations = vectorInfos.TryGetValue("final_obs", out object? existing) && existing is object?[] kept ? kept : new object?[numEnvs];
                    observations[envNum] = value;
                    array = observations;
                } else if (value is Info nested) {
                    // A nested dict is merged recursively, into the dict this key already holds or a new one.
                    array = Add(vectorInfos.Get(key) as Info ?? new Info(), nested, envNum, numEnvs);
                } else {
                    // The base case: the batch exists or is created from this value's type, then the value goes in its slot.
                    array = vectorInfos.TryGetValue(key, out object? batch) && batch is not null ? batch : CreateBatch(value, numEnvs);
                    Assign(array, envNum, value);
                }

                // The mask of the sub-envs that reported the key, created all False on first sight.
                NDArray mask = vectorInfos.Get($"_{key}") as NDArray ?? np.zeros(new Shape(numEnvs), NPTypeCode.Boolean);
                mask.SetValue(true, envNum);
                vectorInfos[key] = array;
                vectorInfos[$"_{key}"] = mask;
            }

            return vectorInfos;
        }

        /// <summary>
        ///     The batch Gymnasium creates for a key's first value: <c>np.zeros(num_envs, dtype=type(value))</c> for a number,
        ///     <c>np.zeros((num_envs, *value.shape), dtype=value.dtype)</c> for an array, and <c>np.full(num_envs, None,
        ///     dtype=object)</c> for anything else.
        /// </summary>
        /// <param name="value">The first value.</param>
        /// <param name="numEnvs">The batch's length.</param>
        /// <returns>An <see cref="NDArray"/> of zeros, or an <see langword="object"/>?[] of <see langword="null"/>s.</returns>
        private static object CreateBatch(object? value, int numEnvs) {
            switch (value) {
                case bool:
                    return np.zeros(new Shape(numEnvs), NPTypeCode.Boolean);
                case double:
                    return np.zeros(new Shape(numEnvs), NPTypeCode.Double);
                case float:
                    return np.zeros(new Shape(numEnvs), NPTypeCode.Single);
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    // A .NET integer stands for a Python int, whose dtype is int64.
                    return np.zeros(new Shape(numEnvs), NPTypeCode.Int64);
                case NDArray { ndim: 0 } scalar when scalar.typecode != NPTypeCode.Boolean:
                    // A 0-d array stands for a NumPy scalar (DR-08): an np.number makes a batch of its own dtype.
                    return np.zeros(new Shape(numEnvs), scalar.typecode);
                case NDArray { ndim: > 0 } array: {
                    var dims = new long[array.ndim + 1];
                    dims[0] = numEnvs;
                    for (int d = 0; d < array.ndim; d++)
                        dims[d + 1] = array.shape[d];
                    return np.zeros(new Shape(dims), array.typecode);
                }
                default:
                    // Strings, None, lists and NumPy's bool scalar (not an np.number) go into an object array.
                    return new object?[numEnvs];
            }
        }

        /// <summary><c>array[env_num] = value</c>: the value cast into a numeric batch as NumPy's assignment casts it, or stored in an object batch.</summary>
        /// <param name="array">The batch.</param>
        /// <param name="envNum">The slot.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="TypeError">Thrown for a value NumPy could not store in a numeric batch (a string into a float batch).</exception>
        private static void Assign(object array, int envNum, object? value) {
            switch (array) {
                case object?[] objects:
                    objects[envNum] = value;
                    return;
                case NDArray batch:
                    switch (value) {
                        case NDArray { ndim: 0 } scalar:
                            batch[envNum] = scalar.astype(batch.dtype);
                            return;
                        case NDArray row:
                            batch[envNum] = row;
                            return;
                        case bool flag:
                            batch.SetValue(flag, envNum);
                            return;
                        case double real:
                            batch.SetValue(real, envNum);
                            return;
                        case float single:
                            batch.SetValue(single, envNum);
                            return;
                        case BigInteger big:
                            batch.SetValue((long)big, envNum);
                            return;
                        case sbyte or byte or short or ushort or int or uint or long:
                            batch.SetValue(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture), envNum);
                            return;
                        case ulong unsigned:
                            batch.SetValue(unsigned, envNum);
                            return;
                        default:
                            throw new TypeError($"cannot store {PyBuiltins.Repr(value)} ({PyBuiltins.TypeRepr(value)}) in a {batch.dtype.name} info batch");
                    }

                default:
                    throw new TypeError($"cannot store an info value in {PyBuiltins.TypeRepr(array)}");
            }
        }
    }
}
