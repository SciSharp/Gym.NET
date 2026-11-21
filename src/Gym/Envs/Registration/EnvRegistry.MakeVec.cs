#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Environments;
using Gym.Exceptions;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Envs.Registration {
    public static partial class EnvRegistry {
        /// <summary>
        ///     Builds a vector env of <paramref name="numEnvs"/> copies of a registered env (<c>gymnasium.make_vec</c>): its own
        ///     vectorized class when the spec has a <c>vector_entry_point</c> (<c>CartPoleVectorEnv</c>), else a
        ///     <c>SyncVectorEnv</c> of <c>Make</c>'s stacks. The result is untyped (observations and actions as
        ///     <see langword="object"/>s); <see cref="MakeVec{TObs, TAct}(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/> types it.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs; a spec made by <c>MakeVec</c> records its own, which wins.</param>
        /// <param name="vectorizationMode">A <see cref="VectorizeMode"/> or its value (<c>"sync"</c>, <c>"async"</c>, <c>"vector_entry_point"</c>); <see langword="null"/> picks the vector entry point when the spec has one, else sync.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name (<c>autoreset_mode</c>, <c>copy</c>, <c>observation_mode</c>; for async also <c>shared_memory</c>, <c>max_concurrency</c>, <c>execution_mode</c>, <c>thread_count</c>); sync and async only.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env after <c>Make</c>, innermost first; sync and async only.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, over the spec's.</param>
        /// <returns>The vector env, its spec recording how it was made; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="id"/> is <see langword="null"/>, or the env cannot be built from the kwargs.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode, with Gymnasium's message.</exception>
        /// <exception cref="Error">Thrown when the id is not registered, has no (vector) entry point, or the vector entry point is asked for with vector kwargs or wrappers.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/registration.py#L835">gymnasium/envs/registration.py</see>.</remarks>
        public static IVectorEnv MakeVec(string id, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null,
            object? kwargs = null) {
            if (id is null)
                throw new TypeError($"Invalid id type: {PyBuiltins.TypeRepr(null)}. Expected `str` or `EnvSpec`");
            return MakeVecCore(FindSpec(id), id, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs, null);
        }

        /// <summary>Builds a vector env from a spec (<c>gymnasium.make_vec(env_spec)</c>); see <see cref="MakeVec(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.</summary>
        /// <param name="spec">The spec: a registered one, or one a previous <c>MakeVec</c> recorded (its <c>num_envs</c>, mode, vector kwargs and wrappers win).</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/>, or the env cannot be built.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point, or the vector entry point is asked for with vector kwargs or wrappers.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static IVectorEnv MakeVec(EnvSpec spec, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null, IEnumerable<Func<IEnv, IEnv>>? wrappers = null,
            object? kwargs = null) {
            if (spec is null)
                throw new TypeError($"Invalid id type: {PyBuiltins.TypeRepr(null)}. Expected `str` or `EnvSpec`");
            return MakeVecCore(spec, spec, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs, null);
        }

        /// <summary>
        ///     <see cref="MakeVec(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/> typed (DR-21): the vector env
        ///     comes back as <c>VectorEnv&lt;TObs, TAct&gt;</c> of the batched types, a sync one built as
        ///     <c>SyncVectorEnv&lt;TObs, TAct&gt;</c>.
        /// </summary>
        /// <typeparam name="TObs">The batched observation type (<see cref="NDArray"/> for array spaces).</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="id">The id.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the vector env's batched types are not the ones asked for, or as the untyped form throws.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the id is not registered or has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(string id, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null,
            IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null) {
            if (id is null)
                throw new TypeError($"Invalid id type: {PyBuiltins.TypeRepr(null)}. Expected `str` or `EnvSpec`");
            return (VectorEnv<TObs, TAct>)MakeVecCore(FindSpec(id), id, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs, (typeof(TObs), typeof(TAct)));
        }

        /// <summary>The spec form of <see cref="MakeVec{TObs, TAct}(string, int, object, object, IEnumerable{Func{IEnv, IEnv}}, object)"/>.</summary>
        /// <typeparam name="TObs">The batched observation type.</typeparam>
        /// <typeparam name="TAct">The batched action type.</typeparam>
        /// <param name="spec">The spec.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spec"/> is <see langword="null"/> or the batched types differ.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the spec has no (vector) entry point.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        public static VectorEnv<TObs, TAct> MakeVec<TObs, TAct>(EnvSpec spec, int numEnvs = 1, object? vectorizationMode = null, object? vectorKwargs = null,
            IEnumerable<Func<IEnv, IEnv>>? wrappers = null, object? kwargs = null) {
            if (spec is null)
                throw new TypeError($"Invalid id type: {PyBuiltins.TypeRepr(null)}. Expected `str` or `EnvSpec`");
            return (VectorEnv<TObs, TAct>)MakeVecCore(spec, spec, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs, (typeof(TObs), typeof(TAct)));
        }

        /// <summary>
        ///     <c>make_vec</c> with a viewer factory (DR-89): what the <c>Gymnasium</c> facade's overloads call, the spec found
        ///     from an id or given, the factory handed to every sub-env's window (see <see cref="MakeVecCore"/>).
        /// </summary>
        /// <param name="idOrSpec">The id (a string) or the spec.</param>
        /// <param name="numEnvs">The number of sub-envs.</param>
        /// <param name="vectorizationMode">The vectorization mode (a <see cref="VectorizeMode"/> or its value), or <see langword="null"/> for the spec's default.</param>
        /// <param name="vectorKwargs">The vector env's constructor arguments by Python name.</param>
        /// <param name="wrappers">Wrappers applied to every sub-env, innermost first.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name.</param>
        /// <param name="viewerFactory">The viewer factory, or <see langword="null"/> for the app-wide one.</param>
        /// <param name="typed">The batched types of the typed form, or <see langword="null"/>.</param>
        /// <returns>The vector env; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="idOrSpec"/> is neither, the env cannot be built, or the typed form's types differ.</exception>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown when the id is not registered, has no (vector) entry point, or the vector entry point is asked for with vector kwargs or wrappers.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        internal static IVectorEnv MakeVecWithViewer(object idOrSpec, int numEnvs, object? vectorizationMode, object? vectorKwargs, IEnumerable<Func<IEnv, IEnv>>? wrappers,
            object? kwargs, EnvViewerFactory? viewerFactory, (Type Obs, Type Act)? typed) =>
            idOrSpec switch {
                string id => MakeVecCore(FindSpec(id), id, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs, typed, viewerFactory),
                EnvSpec spec => MakeVecCore(spec, spec, numEnvs, vectorizationMode, vectorKwargs, wrappers, kwargs, typed, viewerFactory),
                _ => throw new TypeError($"Invalid id type: {PyBuiltins.TypeRepr(idOrSpec)}. Expected `str` or `EnvSpec`"),
            };

        /// <summary>
        ///     <c>make_vec</c>'s body: the spec copied with its kwargs taken out (a recorded <c>num_envs</c>, mode, vector kwargs
        ///     and wrappers among them win over the arguments), the mode resolved, the vector env built, and a spec recording
        ///     how attached to it.
        /// </summary>
        /// <param name="registered">The spec found or given.</param>
        /// <param name="id">The id argument as given, for the message that quotes it.</param>
        /// <param name="numEnvs">The number of sub-envs asked for.</param>
        /// <param name="vectorizationMode">The mode asked for, or <see langword="null"/>.</param>
        /// <param name="vectorKwargs">The vector kwargs asked for.</param>
        /// <param name="wrappers">The wrappers asked for.</param>
        /// <param name="kwargs">The env kwargs asked for.</param>
        /// <param name="typed">The batched types of the typed form, or <see langword="null"/>.</param>
        /// <param name="viewerFactory">
        ///     The viewer factory the facade's overloads were given (DR-89), or <see langword="null"/>: each sub-env's
        ///     <c>make</c> gets it (sync and async modes), or the vector entry point's constructor as <c>viewer_factory</c>; the
        ///     spec never records it.
        /// </param>
        /// <returns>The vector env.</returns>
        /// <exception cref="ValueError">Thrown for an unknown vectorization mode.</exception>
        /// <exception cref="Error">Thrown for the entry-point and argument conflicts Gymnasium refuses.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit the batch (sync and async modes).</exception>
        /// <exception cref="TypeError">Thrown when the typed form's types do not fit, or the vector entry point builds no vector env.</exception>
        private static IVectorEnv MakeVecCore(EnvSpec registered, object id, int numEnvs, object? vectorizationMode, object? vectorKwargs, IEnumerable<Func<IEnv, IEnv>>? wrappers,
            object? kwargs, (Type Obs, Type Act)? typed, EnvViewerFactory? viewerFactory = null) {
            Info vectorArguments = vectorKwargs as Info ?? PyKwargs.From(vectorKwargs);
            IReadOnlyList<Func<IEnv, IEnv>> wrapperList = wrappers?.ToArray() ?? Array.Empty<Func<IEnv, IEnv>>();

            EnvSpec envSpec = registered.DeepCopy();
            Info envSpecKwargs = envSpec.Kwargs;
            // For sync or async, these parameters should be passed in `make(..., **kwargs)` rather than in the env spec
            // kwargs, therefore, the spec's kwargs are reset.
            envSpec.Kwargs = new Info();

            // env_spec_kwargs.pop(key, default): a value the spec recorded wins over the argument.
            if (Pop(envSpecKwargs, "num_envs", out object? recordedNumEnvs))
                numEnvs = Convert.ToInt32(recordedNumEnvs, CultureInfo.InvariantCulture);
            if (Pop(envSpecKwargs, "vectorization_mode", out object? recordedMode))
                vectorizationMode = recordedMode;
            if (Pop(envSpecKwargs, "vector_kwargs", out object? recordedVectorKwargs))
                vectorArguments = recordedVectorKwargs as Info ?? PyKwargs.From(recordedVectorKwargs);
            if (Pop(envSpecKwargs, "wrappers", out object? recordedWrappers))
                wrapperList = ((IEnumerable<Func<IEnv, IEnv>>?)recordedWrappers)?.ToArray() ?? Array.Empty<Func<IEnv, IEnv>>();

            envSpecKwargs.Update(PyKwargs.From(kwargs));

            // Specify the vectorization mode if None or update to a `VectorizeMode`.
            VectorizeMode mode;
            if (vectorizationMode is null) {
                mode = envSpec.VectorEntryPoint is not null ? VectorizeMode.VectorEntryPoint : VectorizeMode.Sync;
            } else {
                try {
                    // VectorizeMode(vectorization_mode): a member as it is (a cast integer the enum does not declare is no
                    // member, as Python's lookup refuses any non-member), a string by its value, anything else refused.
                    mode = vectorizationMode switch {
                        VectorizeMode member when Enum.IsDefined(member) => member,
                        string value => VectorizeModes.Of(value),
                        _ => throw new ValueError($"{PyBuiltins.Repr(vectorizationMode)} is not a valid VectorizeMode"),
                    };
                } catch (ValueError e) {
                    throw new ValueError($"Invalid vectorization mode: {PyBuiltins.Repr(vectorizationMode)}, valid modes: {PyBuiltins.Repr(VectorizeModes.All.Select(m => (object?)VectorizeModes.Value(m)).ToList())}", e);
                }
            }

            // One Make stack per sub-env, each wrapped by the wrappers in order (the sync and async modes' create_single_env).
            IEnv CreateSingleEnv() {
                // make(env_spec, **env_spec_kwargs.copy()): a shallow copy per sub-env, the values shared as in Python; the
                // viewer factory (DR-89) goes to each sub-env's window, never into its spec.
                IEnv singleEnv = MakeCore(envSpec, new Info(envSpecKwargs), null, null, null, viewerFactory);
                foreach (Func<IEnv, IEnv> wrapper in wrapperList)
                    singleEnv = wrapper(singleEnv);
                return singleEnv;
            }

            IVectorEnv env;
            if (mode == VectorizeMode.Sync) {
                if (envSpec.EntryPoint is null)
                    throw new Error($"Cannot create vectorized environment for {envSpec.Id} because it doesn't have an entry point defined.");
                Type syncType = typed is { } types ? typeof(SyncVectorEnv<,>).MakeGenericType(types.Obs, types.Act) : typeof(SyncVectorEnv);
                var arguments = new Info { ["env_fns"] = Enumerable.Repeat<Func<IEnv>>(CreateSingleEnv, numEnvs).ToList() };
                arguments.Update(vectorArguments);
                env = (IVectorEnv)PyKwargs.Construct(syncType, arguments);
            } else if (mode == VectorizeMode.Async) {
                if (envSpec.EntryPoint is null)
                    throw new Error($"Cannot create vectorized environment for {envSpec.Id} because it doesn't have an entry point defined.");
                // The workers call create_single_env on their own threads (DR-22), as Gymnasium's processes do.
                Type asyncType = typed is { } types ? typeof(AsyncVectorEnv<,>).MakeGenericType(types.Obs, types.Act) : typeof(AsyncVectorEnv);
                var arguments = new Info { ["env_fns"] = Enumerable.Repeat<Func<IEnv>>(CreateSingleEnv, numEnvs).ToList() };
                arguments.Update(vectorArguments);
                env = (IVectorEnv)PyKwargs.Construct(asyncType, arguments);
            } else {
                if (vectorArguments.Count > 0)
                    throw new Error($"Custom vector environment can be passed arguments only through kwargs and `vector_kwargs` is not empty ({PyBuiltins.Str(vectorArguments)})");
                if (wrapperList.Count > 0)
                    throw new Error($"Cannot use `vector_entry_point` vectorization mode with the wrappers argument ({PyBuiltins.Str(wrapperList.ToList())}).");
                if (envSpec.AdditionalWrappers.Count > 0)
                    throw new Error($"Cannot use `vector_entry_point` vectorization mode with the additional_wrappers parameter in spec being not empty ({PyBuiltins.Str(envSpec.AdditionalWrappers.Cast<object?>().ToArray())}).");

                Type envCreator = envSpec.VectorEntryPoint switch {
                    null => throw new Error($"Cannot create vectorized environment for {PyBuiltins.Str(id)} because it doesn't have a vector entry point defined."),
                    Type type => type,
                    string name => LoadEnvCreator(name),
                    _ => throw new TypeError($"A vector entry point is a class or a \"Namespace:Class\" string, not {PyBuiltins.TypeRepr(envSpec.VectorEntryPoint)}"),
                };

                if (envSpec.MaxEpisodeSteps is not null && !envSpecKwargs.ContainsKey("max_episode_steps"))
                    envSpecKwargs["max_episode_steps"] = envSpec.MaxEpisodeSteps;
                var arguments = new Info { ["num_envs"] = numEnvs };
                arguments.Update(envSpecKwargs);
                // The constructor's arguments only: the spec below records envSpecKwargs, without the delegate (DR-89).
                if (viewerFactory is not null)
                    arguments["viewer_factory"] = viewerFactory;
                object created = PyKwargs.Construct(envCreator, arguments);
                env = created as IVectorEnv ?? throw new TypeError($"The vector entry point of {envSpec.Id} built {PyBuiltins.TypeRepr(created)}, not a vector env");
            }

            try {
                // Copies the environment creation specification and kwargs to add to the environment specification details.
                EnvSpec copiedIdSpec = envSpec.DeepCopy();
                copiedIdSpec.Kwargs = new Info(envSpecKwargs);
                if (numEnvs != 1)
                    copiedIdSpec.Kwargs["num_envs"] = (long)numEnvs;
                copiedIdSpec.Kwargs["vectorization_mode"] = VectorizeModes.Value(mode);
                if (vectorArguments.Count > 0)
                    copiedIdSpec.Kwargs["vector_kwargs"] = vectorArguments;
                if (wrapperList.Count > 0)
                    copiedIdSpec.Kwargs["wrappers"] = wrapperList.ToList();
                env.Unwrapped.Spec = copiedIdSpec;

                if (!env.Metadata.TryGetValue("autoreset_mode", out object? autoresetMode))
                    Logger.Warn($"The VectorEnv ({env}) is missing AutoresetMode metadata, metadata={PyBuiltins.Str(env.Metadata)}");
                else if (autoresetMode is not AutoresetMode)
                    Logger.Warn($"The VectorEnv ({env}) metadata['autoreset_mode'] is not an instance of AutoresetMode, {PyBuiltins.TypeRepr(autoresetMode)}.");

                if (typed is { } final && !IsVectorEnvOf(env, final))
                    throw new TypeError($"{envSpec.Id} vectorizes to {PyAttributes.PyTypeName(env.GetType())}, not the VectorEnv<{final.Obs.Name}, {final.Act.Name}> asked for; use the untyped MakeVec or ask for its batched types");
                return env;
            } catch {
                // Nobody else holds the vector env yet: release it before the error propagates.
                env.Dispose();
                throw;
            }
        }

        /// <summary>Python's <c>dict.pop(key)</c> when the key is there: removes it and gives its value.</summary>
        /// <param name="dict">The dict.</param>
        /// <param name="key">The key.</param>
        /// <param name="value">The value removed, or <see langword="null"/>.</param>
        /// <returns><see langword="true"/> when the key was there.</returns>
        private static bool Pop(Info dict, string key, out object? value) {
            if (!dict.TryGetValue(key, out value))
                return false;
            dict.Remove(key);
            return true;
        }

        /// <summary>Whether a vector env is a <c>VectorEnv&lt;TObs, TAct&gt;</c> of the given batched types.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="types">The batched types.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        private static bool IsVectorEnvOf(IVectorEnv env, (Type Obs, Type Act) types) {
            for (Type? level = env.GetType(); level is not null; level = level.BaseType) {
                if (level.IsGenericType && level.GetGenericTypeDefinition() == typeof(VectorEnv<,>)) {
                    Type[] arguments = level.GetGenericArguments();
                    return arguments[0] == types.Obs && arguments[1] == types.Act;
                }
            }

            return false;
        }
    }
}
