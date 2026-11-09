#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.utils.env_checker</c>: the checks an env must pass to follow the API. Today it holds
    ///     <see cref="DataEquivalence"/>, which tests and checkers use to compare observations, actions and infos; the
    ///     env checks themselves arrive with the registry (P3).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py">gymnasium/utils/env_checker.py</see>.</remarks>
    public static class EnvChecker {
        /// <summary>
        ///     Whether two values are equivalent (Gymnasium's <c>data_equivalence</c>): of the same Python type, dicts with the
        ///     same keys and equivalent values, tuples and lists of equivalent items, arrays of the same shape and dtype with
        ///     close (or, when <paramref name="exact"/>, equal) entries, and anything else equal by Python's <c>==</c>.
        /// </summary>
        /// <param name="data1">The first value.</param>
        /// <param name="data2">The second value.</param>
        /// <param name="exact">Whether arrays must be equal entry for entry rather than within <c>np.allclose(rtol=1e-5, atol=1e-5)</c>.</param>
        /// <returns><see langword="true"/> when equivalent.</returns>
        /// <remarks>
        ///     <para>
        ///     The type test is on the values' Python counterparts (<see cref="PyBuiltins"/>): every .NET integer is an
        ///     <c>int</c>, an <see langword="object"/>?[] or <see cref="ITuple"/> a <c>tuple</c>, a typed array or list a
        ///     <c>list</c>, a 0-d array the NumPy scalar it stands for (DR-08), so <c>1</c> and <c>1L</c> are equivalent but a
        ///     tuple never matches a list, as in Gymnasium.
        ///     </para>
        ///     <para>
        ///     A 0-d array compares with <c>==</c>, exactly, as Gymnasium compares NumPy scalars (only arrays of one or more
        ///     dimensions go through <c>allclose</c>); a NaN therefore never equals itself.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L36">gymnasium/utils/env_checker.py (data_equivalence)</see>.
        ///     </para>
        /// </remarks>
        public static bool DataEquivalence(object? data1, object? data2, bool exact = false) {
            if (PyBuiltins.TypeName(data1) != PyBuiltins.TypeName(data2))
                return false;
            if (Entries(data1) is { } left) {
                Dictionary<string, object?> right = Entries(data2)!;
                // dict_keys compare as sets: the same keys in any order.
                return left.Count == right.Count && left.Keys.All(right.ContainsKey) && left.All(pair => DataEquivalence(pair.Value, right[pair.Key], exact));
            }

            if (Items(data1) is { } leftItems) {
                IReadOnlyList<object?> rightItems = Items(data2)!;
                return leftItems.Count == rightItems.Count && leftItems.Zip(rightItems).All(pair => DataEquivalence(pair.First, pair.Second, exact));
            }

            if (data1 is NDArray first && first.ndim > 0) {
                var second = (NDArray)data2!;
                if (!first.Shape.dimensions.SequenceEqual(second.Shape.dimensions) || first.dtype != second.dtype)
                    return false;
                if (exact) {
                    using NDArray equal = first == second;
                    return np.all(equal);
                }

                return np.allclose(first, second, rtol: 1e-5, atol: 1e-5);
            }

            // NumPy scalars (0-d arrays), numbers, strings, None: Python's ==, where NaN equals nothing.
            if (data1 is NDArray scalar1 && data2 is NDArray scalar2) {
                using NDArray equal = scalar1 == scalar2;
                return equal.GetBoolean();
            }

            return PyEquality.AreEqual(data1, data2) && !(data1 is double d && double.IsNaN(d)) && !(data1 is float f && float.IsNaN(f));
        }

        /// <summary>A dict's entries by key, for <see cref="DataEquivalence"/>; <see langword="null"/> for anything that is not a dict.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The entries, or <see langword="null"/>.</returns>
        private static Dictionary<string, object?>? Entries(object? value) {
            switch (value) {
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    return pairs.ToDictionary(pair => pair.Key, pair => pair.Value);
                case IDictionary dictionary: {
                    var entries = new Dictionary<string, object?>();
                    foreach (DictionaryEntry entry in dictionary)
                        entries[Convert.ToString(entry.Key, System.Globalization.CultureInfo.InvariantCulture)!] = entry.Value;
                    return entries;
                }
                default:
                    return null;
            }
        }

        /// <summary>A tuple's or list's items, for <see cref="DataEquivalence"/>; <see langword="null"/> for anything else.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The items, or <see langword="null"/>.</returns>
        private static IReadOnlyList<object?>? Items(object? value) {
            switch (value) {
                case string:
                case NDArray:
                    return null;
                case ITuple tuple: {
                    // GraphInstance too: a NamedTuple is a tuple to isinstance(data_1, (tuple, list)).
                    var items = new object?[tuple.Length];
                    for (int i = 0; i < items.Length; i++)
                        items[i] = tuple[i];
                    return items;
                }
                case IList list:
                    return list.Cast<object?>().ToArray();
                default:
                    return null;
            }
        }

        /// <summary>
        ///     Checks that an env follows Gymnasium's API (Gymnasium's <c>check_env</c>): its spaces, then reset (return type,
        ///     seed determinism, options), the passive checks of a reset and a step, step determinism, fresh data on every
        ///     call, every declared render mode, and a double close. Problems that make the env unusable are assertions;
        ///     questionable choices are warnings through <see cref="Logger"/>.
        /// </summary>
        /// <param name="env">The env, ideally unwrapped (a wrapper is checked too, with a warning).</param>
        /// <param name="warn">Ignored, as in Gymnasium; giving it warns that it is.</param>
        /// <param name="skipRenderCheck">Skip the render checks, which build one env per declared render mode.</param>
        /// <param name="skipCloseCheck">Skip building an env from the spec to close it twice.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>, or the render checks find a <c>render_modes</c> that is not iterable.</exception>
        /// <exception cref="AttributeError">Thrown when the env has no action or observation space.</exception>
        /// <exception cref="AssertionError">Thrown for the first violation found, with Gymnasium's message.</exception>
        /// <exception cref="KeyError">Thrown when the render checks run and the metadata has no <c>render_modes</c>.</exception>
        /// <remarks>
        ///     <para>
        ///     The render checks build an env for every mode in <c>metadata["render_modes"]</c> through the env's spec, so
        ///     <c>human</c> needs a viewer (<c>Gymnasium.HumanViewerFactory</c>, DR-20) as Gymnasium's needs pygame; pass
        ///     <paramref name="skipRenderCheck"/> where there is none. Gymnasium's JAX and PyTorch conversions do not apply:
        ///     Gym.NET's envs return NumSharp arrays.
        ///     </para>
        ///     <para>
        ///     The env is reset and stepped many times and left in whatever state the last check put it. The observations the
        ///     checks collect are left to the garbage collector rather than disposed: an env that wrongly hands out its own
        ///     buffer, which <see cref="CheckReturnedDataNotReused"/> reports, must not have it disposed under it first.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L442">gymnasium/utils/env_checker.py</see>.</para>
        /// </remarks>
        public static void CheckEnv(IEnv env, bool? warn = null, bool skipRenderCheck = false, bool skipCloseCheck = false) {
            if (warn is not null)
                Logger.Warn("`check_env(warn=...)` parameter is now ignored.");

            if (env is null)
                throw new TypeError($"The environment must inherit from the gymnasium.Env class, actual class: {PyBuiltins.TypeRepr(null)}. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.");
            if (!ReferenceEquals(env.Unwrapped, env))
                Logger.Warn($"The environment ({env}) is different from the unwrapped version ({env.Unwrapped}). This could effect the environment checker as the environment most likely has a wrapper applied to it. We recommend using the raw environment for `check_env` using `env.unwrapped`.");

            // ============= Check the spaces (observation and action) ================
            Space actionSpace = RequireSpace(() => env.ActionSpace, "action");
            PassiveChecker.CheckActionSpace(actionSpace);
            CheckSpaceLimit(actionSpace, "action");

            Space observationSpace = RequireSpace(() => env.ObservationSpace, "observation");
            PassiveChecker.CheckObservationSpace(observationSpace);
            CheckSpaceLimit(observationSpace, "observation");

            // ==== Check the reset method ====
            CheckSeedDeprecation(env);
            CheckResetReturnInfoDeprecation(env);
            CheckResetReturnType(env);
            CheckResetSeedDeterminism(env);
            CheckResetOptions(env);

            // ============ Check the returned values ===============
            // NDW012: the results are left to the garbage collector on purpose (see CheckEnv's remarks): an env that wrongly hands out its own buffer must not have it disposed under it.
#pragma warning disable NDW012
            PassiveChecker.EnvResetPassiveChecker(env);
            PassiveChecker.EnvStepPassiveChecker(env, env.ActionSpace.Sample());
#pragma warning restore NDW012

            // ==== Check the step method ====
            CheckStepDeterminism(env);
            CheckReturnedDataNotReused(env);

            // ==== Check the render method and the declared render modes ====
            if (!skipRenderCheck) {
                if (env.RenderMode is not null)
                    PassiveChecker.EnvRenderPassiveChecker(env).Dispose();

                if (env.Spec is { } spec) {
                    object? renderModes = env.Metadata[Metadata.RenderModesKey];
                    foreach (object? renderMode in renderModes as IEnumerable ?? throw new TypeError($"'{PyBuiltins.TypeName(renderModes)}' object is not iterable")) {
                        using IEnv newEnv = spec.Make(new Info { ["render_mode"] = renderMode });
                        // NDW012: the results are left to the garbage collector on purpose (see CheckEnv's remarks): an env that wrongly hands out its own buffer must not have it disposed under it.
#pragma warning disable NDW012
                        newEnv.Reset();
#pragma warning restore NDW012
                        PassiveChecker.EnvRenderPassiveChecker(newEnv).Dispose();
                        newEnv.Close();
                    }
                } else {
                    Logger.Warn("Not able to test alternative render modes due to the environment not having a spec. Try instantiating the environment through `gymnasium.make`");
                }
            }

            if (!skipCloseCheck && env.Spec is { } closeSpec) {
                using IEnv newEnv = closeSpec.Make();
                newEnv.Close();
                try {
                    newEnv.Close();
                } catch (Exception e) {
                    Logger.Warn($"Calling `env.close()` on the closed environment should be allowed, but it raised an exception: {e.Message}");
                }
            }
        }

        /// <summary>
        ///     Checks that seeded resets are deterministic and reseed the env's generator (Gymnasium's
        ///     <c>check_reset_seed_determinism</c>): two <c>reset(seed=123)</c> give equivalent observations and the same
        ///     generator state, the unseeded resets after them agree too, and <c>reset(seed=456)</c> moves the generator elsewhere.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <exception cref="AssertionError">
        ///     Thrown for an observation outside the space, a reset that leaves no generator or does not reseed it
        ///     (<c>super().reset(seed=seed)</c> missing), or observations that differ for the same seed, with Gymnasium's
        ///     messages; and for a <see cref="TypeError"/> raised by the resets, which Gymnasium turns into one.
        /// </exception>
        /// <remarks>
        ///     The determinism of the observations is asserted only for an env whose spec says it is deterministic, as in
        ///     Gymnasium; the generator checks always run. Gymnasium's checks of the signature have no counterpart: every
        ///     <c>Reset</c> takes a seed, and C# allows a <see cref="BigInteger"/>? parameter no default but <c>null</c>, so the
        ///     warning about a seed default other than <c>None</c> cannot arise.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L78">gymnasium/utils/env_checker.py</see>.
        /// </remarks>
        public static void CheckResetSeedDeterminism(IEnv env) {
            try {
                // Some environments reuse an internal observation buffer: keep a snapshot before the next reset can mutate it.
                object? obs1 = PyCopy.DeepCopy(env.Reset(123).Observation);
                Assert(env.ObservationSpace.Contains(obs1), "The observation returned by `env.reset(seed=123)` is not within the observation space.");
                BitGeneratorState seed123Rng1 = RawGenerator(env)?.bit_generator.state
                    ?? throw new AssertionError("Expects the random number generator to have been generated given a seed was passed to reset. Most likely the environment reset function does not call `super().reset(seed=seed)`.");

                object? obs2 = PyCopy.DeepCopy(env.Reset().Observation);
                Assert(env.ObservationSpace.Contains(obs2), "The observation returned by `env.reset()` is not within the observation space.");

                object? obs3 = env.Reset(123).Observation;
                Assert(env.ObservationSpace.Contains(obs3), "The observation returned by `env.reset(seed=123)` is not within the observation space.");
                BitGeneratorState? seed123Rng3 = RawGenerator(env)?.bit_generator.state;

                object? obs4 = env.Reset().Observation;
                Assert(env.ObservationSpace.Contains(obs4), "The observation returned by `env.reset()` is not within the observation space.");

                if (env.Spec is { Nondeterministic: false }) {
                    Assert(DataEquivalence(obs1, obs3), "Using `env.reset(seed=123)` is non-deterministic as the observations are not equivalent.");
                    Assert(DataEquivalence(obs2, obs4), "Using `env.reset(seed=123)` then `env.reset()` is non-deterministic as the observations are not equivalent.");
                    if (!DataEquivalence(obs1, obs3, exact: true))
                        Logger.Warn("Using `env.reset(seed=123)` observations are not equal although similar.");
                    if (!DataEquivalence(obs2, obs4, exact: true))
                        Logger.Warn("Using `env.reset(seed=123)` then `env.reset()` observations are not equal although similar.");
                }

                Assert(StatesEqual(seed123Rng1, seed123Rng3),
                    "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random generates are not same when the same seeds are passed to `env.reset`.");

                object? obs5 = env.Reset(456).Observation;
                Assert(env.ObservationSpace.Contains(obs5), "The observation returned by `env.reset(seed=456)` is not within the observation space.");
                Assert(!StatesEqual(RawGenerator(env)?.bit_generator.state, seed123Rng1),
                    "Most likely the environment reset function does not call `super().reset(seed=seed)` as the random number generators are not different when different seeds are passed to `env.reset`.");
            } catch (TypeError e) {
                throw new AssertionError($"The environment cannot be reset with a random seed, even though `seed` or `kwargs` appear in the signature. This should never happen, please report this issue. The error was: {e.Message}", e);
            }
        }

        /// <summary>Checks that the env can be reset with options (Gymnasium's <c>check_reset_options</c>): <c>reset(options={})</c> must not raise <see cref="TypeError"/>.</summary>
        /// <param name="env">The env.</param>
        /// <exception cref="AssertionError">Thrown when the reset raises <see cref="TypeError"/>, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L176">gymnasium/utils/env_checker.py</see>.</remarks>
        public static void CheckResetOptions(IEnv env) {
            try {
                // NDW012: the results are left to the garbage collector on purpose (see CheckEnv's remarks): an env that wrongly hands out its own buffer must not have it disposed under it.
#pragma warning disable NDW012
                env.Reset(options: new Info());
#pragma warning restore NDW012
            } catch (TypeError e) {
                throw new AssertionError($"The environment cannot be reset with options, even though `options` or `**kwargs` appear in the signature. This should never happen, please report this issue. The error was: {e.Message}", e);
            }
        }

        /// <summary>
        ///     Checks that one step after a seeded reset is deterministic (Gymnasium's <c>check_step_determinism</c>): the same
        ///     seed and action give the same generator state, equivalent observation, reward and info, the same termination,
        ///     and no truncation. An env whose spec is nondeterministic is skipped.
        /// </summary>
        /// <param name="env">The env, which must already pass the reset checks.</param>
        /// <param name="seed">The seed of the action space and of both resets.</param>
        /// <exception cref="AssertionError">Thrown for the first difference, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L204">gymnasium/utils/env_checker.py</see>.</remarks>
        public static void CheckStepDeterminism(IEnv env, long seed = 123) {
            if (env.Spec is { Nondeterministic: true })
                return;

            env.ActionSpace.Seed(seed);
            object? action = env.ActionSpace.Sample();

            // NDW012: the results are left to the garbage collector on purpose (see CheckEnv's remarks): an env that wrongly hands out its own buffer must not have it disposed under it.
#pragma warning disable NDW012
            env.Reset(seed);
#pragma warning restore NDW012
            (object? obs0, double rew0, bool term0, bool trunc0, Info info0) = env.Step(action);
            // Preserve the first transition before the second reset and step can mutate a reused observation buffer.
            obs0 = PyCopy.DeepCopy(obs0);

            BitGeneratorState seededRng = RawGenerator(env)?.bit_generator.state ?? throw new AssertionError("env.reset() should have initialized env._np_random");

#pragma warning disable NDW012
            env.Reset(seed);
#pragma warning restore NDW012
            (object? obs1, double rew1, bool term1, bool trunc1, Info info1) = env.Step(action);

            // A seeded reset can replace the environment's random number generator.
            BitGeneratorState currentRng = RawGenerator(env)?.bit_generator.state ?? throw new AssertionError("env.reset() should have initialized env._np_random");
            Assert(StatesEqual(currentRng, seededRng), "The `.np_random` is not properly been updated after step.");

            Assert(DataEquivalence(obs0, obs1), "Deterministic step observations are not equivalent for the same seed and action");
            if (!DataEquivalence(obs0, obs1, exact: true))
                Logger.Warn("Step observations are not equal although similar given the same seed and action");

            Assert(DataEquivalence(rew0, rew1), "Deterministic step rewards are not equivalent for the same seed and action");
            if (!DataEquivalence(rew0, rew1, exact: true))
                Logger.Warn("Step rewards are not equal although similar given the same seed and action");

            Assert(DataEquivalence(term0, term1, exact: true), "Deterministic step termination are not equivalent for the same seed and action");
            Assert(!trunc0 && !trunc1, "Environment truncates after 1 step, something has gone very wrong.");

            Assert(DataEquivalence(info0, info1), "Deterministic step info are not equivalent for the same seed and action");
            if (!DataEquivalence(info0, info1, exact: true))
                Logger.Warn("Step info are not equal although similar given the same seed and action");
        }

        /// <summary>
        ///     Checks that every reset and step returns new observation and info data (Gymnasium's
        ///     <c>check_returned_data_not_reused</c>): a reset, up to two steps and another reset, compared pairwise with
        ///     <see cref="PassiveChecker.DataSharesObjects(object?, object?)"/>.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <param name="seed">The seed of the action space and of both resets.</param>
        /// <exception cref="AssertionError">Thrown when two calls returned data sharing an object, naming both calls.</exception>
        /// <remarks>
        ///     Identity, not value, is compared, so even an array an env means to be constant must be a new array in every
        ///     result: neither the user nor this check can know a later call will not change it.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L272">gymnasium/utils/env_checker.py</see>.
        /// </remarks>
        public static void CheckReturnedDataNotReused(IEnv env, long seed = 123) {
            env.ActionSpace.Seed(seed);

            ResetResult<object?> first = env.Reset(seed);
            var calls = new List<(string Source, object? Obs, Info Info)> { ("env.reset() (call 1)", first.Observation, first.Info) };
            for (int i = 0; i < 2; i++) {
                (object? obs, _, bool terminated, bool truncated, Info info) = env.Step(env.ActionSpace.Sample());
                calls.Add(($"env.step() (call {(calls.Count + 1).ToString(CultureInfo.InvariantCulture)})", obs, info));
                if (terminated || truncated)
                    break;
            }

            // AutoresetMode.SAME_STEP resets a sub-environment as it terminates or truncates.
            ResetResult<object?> last = env.Reset(seed);
            calls.Add(($"env.reset() (call {(calls.Count + 1).ToString(CultureInfo.InvariantCulture)})", last.Observation, last.Info));

            for (int i = 0; i < calls.Count; i++) {
                for (int j = i + 1; j < calls.Count; j++) {
                    Assert(!PassiveChecker.DataSharesObjects(calls[i].Obs, calls[j].Obs),
                        $"The observations returned by `{calls[i].Source}` and `{calls[j].Source}` share an object, therefore, modifying one will modify the other. Environments must return new observation data on every call as users keep the data returned to them.");
                    Assert(!PassiveChecker.DataSharesObjects(calls[i].Info, calls[j].Info),
                        $"The infos returned by `{calls[i].Source}` and `{calls[j].Source}` share an object, therefore, modifying one will modify the other. Environments must return new info data on every call as users keep the data returned to them.");
                }
            }
        }

        /// <summary>
        ///     Warns when the env's reset still takes the pre-0.26 <c>return_info</c> argument (Gymnasium's
        ///     <c>check_reset_return_info_deprecation</c>): a public <c>Reset</c> overload with a parameter of that Python name.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L309">gymnasium/utils/env_checker.py</see>.</remarks>
        public static void CheckResetReturnInfoDeprecation(IEnv env) {
            bool takesReturnInfo = env.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Where(method => method.Name == "Reset")
                .Any(method => method.GetParameters().Any(parameter => PyKwargs.ParameterName(parameter) == "return_info"));
            // Gymnasium's message, its missing spaces included: the literals are concatenated without them.
            if (takesReturnInfo)
                Logger.Warn("`return_info` is deprecated as an optional argument to `reset`. `reset`should now always return `obs, info` where `obs` is an observation, and `info` is a dictionarycontaining additional information.");
        }

        /// <summary>
        ///     Warns when the env still has the pre-0.26 <c>seed</c> function (Gymnasium's <c>check_seed_deprecation</c>, which
        ///     tests <c>callable(getattr(env, "seed", None))</c>): a public instance method named <c>Seed</c>, or a <c>seed</c>
        ///     attribute set at run time to a function. Seeding is <c>Reset(seed: s)</c>'s job.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <remarks>
        ///     A run-time <c>seed</c> attribute that is not a function (a number, a list) is not one, as in Python.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L326">gymnasium/utils/env_checker.py</see>.
        /// </remarks>
        public static void CheckSeedDeprecation(IEnv env) {
            bool seedMethod = env.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance).Any(method => method.Name == "Seed");
            // Methods are not attributes to PyAttributes, so a declared Seed and a run-time `seed` function are checked apart.
            bool seedFunction = PyAttributes.HasAttr(env, "seed") && PyAttributes.GetAttr(env, "seed") is Delegate;
            if (seedMethod || seedFunction)
                Logger.Warn("Official support for the `seed` function is dropped. Standard practice is to reset gymnasium environments using `env.reset(seed=<desired seed>)`");
        }

        /// <summary>
        ///     Checks what an unseeded reset returns (Gymnasium's <c>check_reset_return_type</c>): an observation within the
        ///     space and an info dict. The 2-tuple form Gymnasium also asserts is a <see cref="ResetResult{TObs}"/> by type.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <exception cref="AssertionError">Thrown for an observation outside the space or a <see langword="null"/> info, with Gymnasium's messages.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L342">gymnasium/utils/env_checker.py</see>.</remarks>
        public static void CheckResetReturnType(IEnv env) {
            (object? obs, Info info) = env.Reset();
            Assert(env.ObservationSpace.Contains(obs), "The first element returned by `env.reset()` is not within the observation space.");
            Assert(info is not null, $"The second element returned by `env.reset()` was not a dictionary, actual type: {PyBuiltins.TypeRepr(null)}");
        }

        /// <summary>
        ///     Warns about box bounds that make learning hard (Gymnasium's <c>check_space_limit</c>, run only by
        ///     <see cref="CheckEnv"/>): infinite bounds, and for a 1-d action box any range other than a symmetric or
        ///     <c>[0, 1]</c> one within <c>[-1, 1]</c>. Tuples and dicts are checked subspace by subspace.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <param name="spaceType"><c>observation</c> or <c>action</c>.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/env_checker.py#L366">gymnasium/utils/env_checker.py</see>.</remarks>
        public static void CheckSpaceLimit(Space space, string spaceType) {
            switch (space is ObjectSpace view ? view.Inner : space) {
                case Box box: {
                    using (NDArray lowInfinite = np.equal(box.Low, double.NegativeInfinity)) {
                        if (np.any(lowInfinite))
                            Logger.Warn($"A Box {spaceType} space minimum value is -infinity. This is probably too low.");
                    }

                    using (NDArray highInfinite = np.equal(box.High, double.PositiveInfinity)) {
                        if (np.any(highInfinite))
                            Logger.Warn($"A Box {spaceType} space maximum value is infinity. This is probably too high.");
                    }

                    // Check that the Box space is normalized: vector action boxes only.
                    if (spaceType == "action" && box.Shape.dimensions.Length == 1) {
                        using NDArray zeros = np.zeros_like(box.Low);
                        using NDArray nonZeroLow = box.Low != zeros;
                        using NDArray absLow = np.abs(box.Low);
                        using NDArray absHigh = np.abs(box.High);
                        using NDArray asymmetric = absLow != absHigh;
                        using NDArray unnormalized = np.logical_and(nonZeroLow, asymmetric);
                        using NDArray belowMinusOne = box.Low < -1;
                        using NDArray aboveOne = box.High > 1;
                        if (np.any(unnormalized) || np.any(belowMinusOne) || np.any(aboveOne))
                            Logger.Warn("For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.");
                    }

                    break;
                }
                case Spaces.Tuple tuple:
                    foreach (Space subspace in tuple.Spaces)
                        CheckSpaceLimit(subspace, spaceType);
                    break;
                case Dict dict:
                    foreach (Space subspace in dict.Values)
                        CheckSpaceLimit(subspace, spaceType);
                    break;
            }
        }

        /// <summary>Python's <c>assert condition, message</c>.</summary>
        /// <param name="condition">The asserted condition.</param>
        /// <param name="message">The message when it fails.</param>
        /// <exception cref="AssertionError">Thrown when <paramref name="condition"/> is <see langword="false"/>.</exception>
        private static void Assert(bool condition, string message) {
            if (!condition)
                throw new AssertionError(message);
        }

        /// <summary>
        ///     <c>hasattr(env, "action_space")</c> for <see cref="CheckEnv"/>: reads a space, turning "no such attribute" into
        ///     <c>check_env</c>'s message (which, unlike <c>PassiveEnvChecker</c>'s, ends with "See ... for more info.").
        /// </summary>
        /// <param name="read">Reads the space.</param>
        /// <param name="kind"><c>action</c> or <c>observation</c>.</param>
        /// <returns>The space.</returns>
        /// <exception cref="AttributeError">Thrown when the env has no such space.</exception>
        private static Space RequireSpace(Func<Space> read, string kind) {
            try {
                return read();
            } catch (AttributeError) {
                throw new AttributeError($"The environment must specify an {kind} space. See https://gymnasium.farama.org/introduction/create_custom_env/ for more info.");
            }
        }

        /// <summary>
        ///     The unwrapped env's own generator, Python's <c>env.unwrapped._np_random</c>: <see langword="null"/> while none
        ///     was created. Never <c>np_random</c>, whose read would create one and hide a reset that forgot to seed.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <returns>The generator, or <see langword="null"/>.</returns>
        private static Generator? RawGenerator(IEnv env) => PyAttributes.GetAttr(env.Unwrapped, "_np_random") as Generator;

        /// <summary>
        ///     Python's <c>state_1 == state_2</c> on two <c>bit_generator.state</c> dicts: the same generator class and equal
        ///     fields. A missing state equals only a missing state.
        /// </summary>
        /// <param name="left">The first snapshot.</param>
        /// <param name="right">The second snapshot.</param>
        /// <returns><see langword="true"/> when the states are the same.</returns>
        private static bool StatesEqual(BitGeneratorState? left, BitGeneratorState? right) {
            if (left is null || right is null)
                return left is null && right is null;
            if (left.GetType() != right.GetType())
                return false;
            foreach (PropertyInfo property in left.GetType().GetProperties(BindingFlags.Public | BindingFlags.Instance)) {
                if (property.GetIndexParameters().Length > 0)
                    continue;
                object? a = property.GetValue(left), b = property.GetValue(right);
                bool same = a is Array first && b is Array second
                    ? first.Length == second.Length && first.Cast<object?>().SequenceEqual(second.Cast<object?>())
                    : Equals(a, b);
                if (!same)
                    return false;
            }

            return true;
        }
    }
}
