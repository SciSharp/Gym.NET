#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Utils {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.utils.passive_env_checker</c>: checks of an env's spaces, observations, rewards and renders
    ///     that warn (or assert, for what cannot work) without changing what the env returns. <c>PassiveEnvChecker</c>, which
    ///     <c>make</c> adds unless <c>disable_env_checker</c>, runs them on the first reset, step and render, so a broken env
    ///     is reported where it breaks rather than deep inside a training loop.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Checks that Python needs because it is dynamically typed cannot fire here, and are left out: a step result always
    ///     has five items with boolean signals, a reward is always a <see langword="double"/>, and <c>reset</c> always takes
    ///     <c>seed</c> and <c>options</c>. What remains is what an env can still get wrong in C#: observations outside the
    ///     space or of another type, a NaN or infinite reward, a <see langword="null"/> info, metadata that does not describe
    ///     the render modes, render results of the wrong form, and data reused between calls.
    ///     </para>
    ///     <para>
    ///     An <see cref="ObjectSpace"/>, the view through which an <c>Env&lt;object?, object?&gt;</c> holds any space, is
    ///     checked as the space it views, as Python sees only that space.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py">gymnasium/utils/passive_env_checker.py</see>.</para>
    /// </remarks>
    public static class PassiveChecker {
        /// <summary>
        ///     Checks a <see cref="Box"/> observation space (Gymnasium's <c>_check_box_observation_space</c>): asserts that the
        ///     bounds have the space's shape, then warns when a bound pair is equal or inverted, either of which makes a
        ///     dimension constant or empty.
        /// </summary>
        /// <param name="observationSpace">The box.</param>
        /// <exception cref="AssertionError">
        ///     Thrown when <c>low</c> or <c>high</c> has another shape than the space, with Gymnasium's messages (the
        ///     <c>high</c> one says "have have", as Gymnasium's does).
        /// </exception>
        /// <remarks>
        ///     The bound shapes always match for a box built by its constructor; the assertion guards against bounds replaced
        ///     afterwards. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L25">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static void CheckBoxObservationSpace(Box observationSpace) {
            if (!SameShape(observationSpace.Low.shape, observationSpace.Shape))
                throw new AssertionError($"The Box observation space shape and low shape have different shapes, low shape: {Space.ShapeRepr(observationSpace.Low.Shape)}, box shape: {Space.ShapeRepr(observationSpace.Shape)}");
            if (!SameShape(observationSpace.High.shape, observationSpace.Shape))
                throw new AssertionError($"The Box observation space shape and high shape have have different shapes, high shape: {Space.ShapeRepr(observationSpace.High.Shape)}, box shape: {Space.ShapeRepr(observationSpace.Shape)}");

            using NDArray equal = observationSpace.Low == observationSpace.High;
            if (np.any(equal)) {
                Logger.Warn("A Box observation space maximum and minimum values are equal.");
            } else {
                using NDArray inverted = observationSpace.High < observationSpace.Low;
                if (np.any(inverted))
                    Logger.Warn("A Box observation space low value is greater than a high value.");
            }
        }

        /// <summary>
        ///     Checks a <see cref="Box"/> action space (Gymnasium's <c>_check_box_action_space</c>): asserts the bound shapes,
        ///     then warns when a bound pair is equal, which leaves the agent no choice in that dimension.
        /// </summary>
        /// <param name="actionSpace">The box.</param>
        /// <exception cref="AssertionError">
        ///     Thrown when <c>low</c> or <c>high</c> has another shape than the space, with Gymnasium's messages (here the
        ///     <c>low</c> one says "have have").
        /// </exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L44">gymnasium/utils/passive_env_checker.py</see>.</remarks>
        public static void CheckBoxActionSpace(Box actionSpace) {
            if (!SameShape(actionSpace.Low.shape, actionSpace.Shape))
                throw new AssertionError($"The Box action space shape and low shape have have different shapes, low shape: {Space.ShapeRepr(actionSpace.Low.Shape)}, box shape: {Space.ShapeRepr(actionSpace.Shape)}");
            if (!SameShape(actionSpace.High.shape, actionSpace.Shape))
                throw new AssertionError($"The Box action space shape and high shape have different shapes, high shape: {Space.ShapeRepr(actionSpace.High.Shape)}, box shape: {Space.ShapeRepr(actionSpace.Shape)}");

            using NDArray equal = actionSpace.Low == actionSpace.High;
            if (np.any(equal))
                Logger.Warn("A Box action space maximum and minimum values are equal.");
        }

        /// <summary>
        ///     Checks a space of either kind (Gymnasium's <c>check_space</c>): what makes it unusable is asserted (a
        ///     non-positive count, a shape that disagrees with <c>nvec</c>, an empty composite), boxes go to
        ///     <paramref name="checkBoxSpace"/>, and tuples and dicts are checked subspace by subspace. Other spaces pass.
        /// </summary>
        /// <param name="space">The space; <see langword="null"/> is refused as Python refuses a non-space.</param>
        /// <param name="spaceType"><c>observation</c> or <c>action</c>, for the messages.</param>
        /// <param name="checkBoxSpace">The box check of that kind: <see cref="CheckBoxObservationSpace"/> or <see cref="CheckBoxActionSpace"/>.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="space"/> is <see langword="null"/>: <c>{spaceType} space does not inherit from `gymnasium.spaces.Space`, actual type: &lt;class 'NoneType'&gt;</c>.</exception>
        /// <exception cref="AssertionError">Thrown for a space that cannot work, with Gymnasium's message.</exception>
        /// <remarks>
        ///     Gym.NET's spaces validate themselves on construction and cannot be reshaped afterwards, so the assertions only
        ///     fire for a space whose state was changed by reflection or a subclass. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L61">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static void CheckSpace(Space? space, string spaceType, Action<Box> checkBoxSpace) {
            switch (Unview(space)) {
                case null:
                    throw new TypeError($"{spaceType} space does not inherit from `gymnasium.spaces.Space`, actual type: {PyBuiltins.TypeRepr(null)}");
                case Box box:
                    checkBoxSpace(box);
                    break;
                case Discrete discrete:
                    if (!(0 < discrete.N))
                        throw new AssertionError($"Discrete {spaceType} space's number of elements must be positive, actual number of elements: {PyBuiltins.Str(discrete.N)}");
                    if (Dimensions(discrete.Shape).Length != 0)
                        throw new AssertionError($"Discrete {spaceType} space's shape should be empty, actual shape: {Space.ShapeRepr(discrete.Shape)}");
                    break;
                case MultiDiscrete multiDiscrete: {
                    if (!SameShape(multiDiscrete.Nvec.shape, multiDiscrete.Shape))
                        throw new AssertionError($"Multi-discrete {spaceType} space's shape must be equal to the nvec shape, space shape: {Space.ShapeRepr(multiDiscrete.Shape)}, nvec shape: {Space.ShapeRepr(multiDiscrete.Nvec.Shape)}");
                    using NDArray positive = multiDiscrete.Nvec > 0;
                    if (!np.all(positive))
                        throw new AssertionError($"Multi-discrete {spaceType} space's all nvec elements must be greater than 0, actual nvec: {PyBuiltins.Str(multiDiscrete.Nvec)}");
                    break;
                }
                case MultiBinary binary:
                    // np.all(0 < np.asarray(space.shape)): an empty shape passes.
                    if (!Dimensions(binary.Shape).All(d => 0 < d))
                        throw new AssertionError($"Multi-binary {spaceType} space's all shape elements must be greater than 0, actual shape: {Space.ShapeRepr(binary.Shape)}");
                    break;
                case Tuple tuple:
                    if (!(0 < tuple.Count))
                        throw new AssertionError($"An empty Tuple {spaceType} space is not allowed.");
                    foreach (Space subspace in tuple.Spaces)
                        CheckSpace(subspace, spaceType, checkBoxSpace);
                    break;
                case Dict dict:
                    if (!(0 < dict.Count))
                        throw new AssertionError($"An empty Dict {spaceType} space is not allowed.");
                    foreach (Space subspace in dict.Values)
                        CheckSpace(subspace, spaceType, checkBoxSpace);
                    break;
            }
        }

        /// <summary>Checks an observation space (Gymnasium's <c>check_observation_space</c>): <see cref="CheckSpace"/> with the observation box check.</summary>
        /// <param name="space">The space.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <exception cref="AssertionError">Thrown for a space that cannot work, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L109">gymnasium/utils/passive_env_checker.py</see>.</remarks>
        public static void CheckObservationSpace(Space? space) => CheckSpace(space, "observation", CheckBoxObservationSpace);

        /// <summary>Checks an action space (Gymnasium's <c>check_action_space</c>): <see cref="CheckSpace"/> with the action box check.</summary>
        /// <param name="space">The space.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <exception cref="AssertionError">Thrown for a space that cannot work, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L114">gymnasium/utils/passive_env_checker.py</see>.</remarks>
        public static void CheckActionSpace(Space? space) => CheckSpace(space, "action", CheckBoxActionSpace);

        /// <summary>
        ///     Checks that an observation matches its space (Gymnasium's <c>check_obs</c>): warns when it is of another type or
        ///     dtype than the space's elements, recurses into tuples and dicts, and warns when it is not a member (or when the
        ///     membership test throws). It never changes the observation.
        /// </summary>
        /// <param name="obs">The observation.</param>
        /// <param name="observationSpace">The space it should belong to.</param>
        /// <param name="methodName"><c>reset</c> or <c>step</c>, for the messages (<c>The obs returned by the `reset()` method ...</c>).</param>
        /// <exception cref="AssertionError">
        ///     Thrown for a tuple observation of another length than the space, or a dict observation that is not a dict or has
        ///     other keys, with Gymnasium's messages.
        /// </exception>
        /// <exception cref="TypeError">
        ///     Thrown when a <see cref="Tuple"/> space's observation has no length (<c>object of type 'int' has no len()</c>),
        ///     Python's error from <c>len(obs)</c>.
        /// </exception>
        /// <remarks>
        ///     An <c>int</c> here is any .NET integer or enum, as Python's <c>int</c>; <c>np.int64</c> is a 0-d int64 array. An
        ///     <see cref="ObjectSpace"/> is checked as the space it views. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L131">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static void CheckObs(object? obs, Space observationSpace, string methodName) {
            string pre = $"The obs returned by the `{methodName}()` method";
            switch (Unview(observationSpace)) {
                case Discrete:
                    if (!(obs is NDArray { ndim: 0 } scalar && scalar.typecode == NPTypeCode.Int64 || IsPythonInt(obs)))
                        Logger.Warn($"{pre} should be an int or np.int64, actual type: {PyBuiltins.TypeRepr(obs)}");
                    break;
                case Box box:
                    // A scalar box accepts any number; only a shaped one expects an array of its dtype.
                    if (Dimensions(box.Shape).Length != 0) {
                        if (!IsNdArray(obs))
                            Logger.Warn($"{pre} was expecting a numpy array, actual type: {PyBuiltins.TypeRepr(obs)}");
                        else if (((NDArray)obs!).dtype != box.DType)
                            Logger.Warn($"{pre} was expecting numpy array dtype to be {Discrete.DTypeName(box.DType!)}, actual type: {Discrete.DTypeName(((NDArray)obs!).dtype)}");
                    }

                    break;
                case MultiBinary or MultiDiscrete:
                    if (!IsNdArray(obs))
                        Logger.Warn($"{pre} was expecting a numpy array, actual type: {PyBuiltins.TypeRepr(obs)}");
                    break;
                case Tuple tuple: {
                    if (!IsTuple(obs))
                        Logger.Warn($"{pre} was expecting a tuple, actual type: {PyBuiltins.TypeRepr(obs)}");
                    object?[] items = SizedItems(obs);
                    try {
                        if (items.Length != tuple.Count)
                            throw new AssertionError($"{pre} length is not same as the observation space length, obs length: {items.Length}, space length: {tuple.Count}");
                        for (int i = 0; i < items.Length; i++)
                            CheckObs(items[i], tuple[i], methodName);
                    } finally {
                        // An array's rows are views made here and released here; any other value's items are the caller's.
                        if (obs is NDArray)
                            foreach (object? row in items)
                                (row as IDisposable)?.Dispose();
                    }

                    break;
                }
                case Dict dict: {
                    IReadOnlyDictionary<string, object?> entries = Dict.AsDict(obs) ?? throw new AssertionError($"{pre} must be a dict, actual type: {PyBuiltins.TypeRepr(obs)}");
                    // obs.keys() == space.keys() compares as sets: the order may differ, the members may not.
                    if (entries.Count != dict.Count || !dict.Keys.All(entries.ContainsKey))
                        throw new AssertionError($"{pre} observation keys is not same as the observation space keys, obs keys: {PyBuiltins.Repr(entries.Keys.ToList())}, space keys: {PyBuiltins.Repr(dict.Keys.ToList())}");
                    foreach (string key in dict.Keys)
                        CheckObs(entries[key], dict[key], methodName);
                    break;
                }
            }

            try {
                if (!observationSpace.Contains(obs))
                    Logger.Warn($"{pre} is not within the observation space.");
            } catch (Exception e) {
                // A membership test that throws is reported, not raised: the check stays passive.
                Logger.Warn($"{pre} is not within the observation space with exception: {e.Message}");
            }
        }

        /// <summary>
        ///     Whether two calls' data share a mutable object (Gymnasium's <c>data_shares_objects</c>): the same dict, or arrays
        ///     over common memory, in matching positions of matching structures. Where <see cref="EnvChecker.DataEquivalence"/>
        ///     compares values, this compares identity. Immutable values (numbers, strings, NumPy scalars) and unknown ones
        ///     (lists included, as in Gymnasium) never count.
        /// </summary>
        /// <param name="data1">The earlier data.</param>
        /// <param name="data2">The later data, of the same structure.</param>
        /// <returns><see langword="true"/> when they share a mutable object.</returns>
        /// <exception cref="TooHardError">Thrown when NumPy's exact overlap problem for two arrays exceeds its solver, as <c>np.shares_memory</c> raises.</exception>
        /// <remarks>
        ///     Both values must be alive: comparing with arrays the caller has since disposed is meaningless, as their memory
        ///     may already belong to another array. To compare with data across calls, keep it with <see cref="Retain"/> and use
        ///     <see cref="DataSharesObjects(RetainedData?, object?)"/>.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L179">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static bool DataSharesObjects(object? data1, object? data2) {
            using RetainedData retained = Retain(data1);
            return DataSharesObjects(retained, data2);
        }

        /// <summary>
        ///     <see cref="DataSharesObjects(object?, object?)"/> against data kept by <see cref="Retain"/>, whose arrays are
        ///     still alive however the caller treated the originals.
        /// </summary>
        /// <param name="data1">The earlier data, retained; <see langword="null"/> shares nothing.</param>
        /// <param name="data2">The later data.</param>
        /// <returns><see langword="true"/> when they share a mutable object.</returns>
        /// <exception cref="TooHardError">Thrown when NumPy's exact overlap problem for two arrays exceeds its solver.</exception>
        public static bool DataSharesObjects(RetainedData? data1, object? data2) {
            // type(data_1) is not type(data_2), on the Python counterparts: an Info and a Dictionary are both dicts.
            if (data1 is null || data1.TypeName != PyBuiltins.TypeName(data2))
                return false;
            switch (data1.Kind) {
                case RetainedKind.Dict:
                    // data_1 is data_2, or any common key whose values share.
                    if (ReferenceEquals(data1.Original, data2))
                        return true;
                    return Dict.AsDict(data2) is { } entries && data1.Entries!.Any(pair => entries.TryGetValue(pair.Key, out object? value) && DataSharesObjects(pair.Value, value));
                case RetainedKind.Tuple:
                    // A tuple is immutable, its items need not be; zip without strict, as the lengths may differ.
                    return data1.Items!.Zip(TupleItems(data2)).Any(pair => DataSharesObjects(pair.First, pair.Second));
                case RetainedKind.Array:
                    // shares_memory also catches a new view of a reused buffer (buffer[:]), which identity would miss.
                    return data2 is NDArray array && np.shares_memory(data1.View!, array);
                default:
                    return false;
            }
        }

        /// <summary>
        ///     Keeps what <see cref="DataSharesObjects(RetainedData?, object?)"/> needs of data the caller now owns: containers
        ///     by reference (for identity), every array as a view. The view keeps the array's memory alive even after the caller
        ///     disposes the array, so that memory cannot be handed to the next call's fresh array and mistaken for sharing.
        /// </summary>
        /// <param name="data">The observation or info.</param>
        /// <returns>The retained data; dispose it to release the views (and with them the memory they keep).</returns>
        public static RetainedData Retain(object? data) {
            string typeName = PyBuiltins.TypeName(data);
            switch (data) {
                case NDArray array when typeName == "numpy.ndarray":
                    return new RetainedData(RetainedKind.Array, typeName, null) { View = array.view() };
                case NDArray or string:
                    // A NumPy scalar or a string is immutable: never shared.
                    return new RetainedData(RetainedKind.Other, typeName, null);
                default:
                    if (Dict.AsDict(data) is { } entries)
                        return new RetainedData(RetainedKind.Dict, typeName, data) { Entries = entries.ToDictionary(pair => pair.Key, pair => Retain(pair.Value)) };
                    if (IsTuple(data))
                        return new RetainedData(RetainedKind.Tuple, typeName, data) { Items = TupleItems(data).Select(Retain).ToList() };
                    return new RetainedData(RetainedKind.Other, typeName, null);
            }
        }

        /// <summary>
        ///     Warns when two calls returned observations or infos that share an object (Gymnasium's
        ///     <c>env_data_reuse_passive_checker</c>): users keep the data of every step (replay and rollout buffers), and an
        ///     env that edits and returns the same buffer corrupts what they hold.
        /// </summary>
        /// <param name="previousData">The earlier call's <c>(obs, info)</c>; <see langword="null"/> fails as Python's unpacking of <c>None</c> does.</param>
        /// <param name="data">The later call's <c>(obs, info)</c>.</param>
        /// <param name="previousSource">The earlier call, <c>reset</c> or <c>step</c>.</param>
        /// <param name="source">The later call.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="previousData"/> is <see langword="null"/>: <c>cannot unpack non-iterable NoneType object</c>.</exception>
        /// <exception cref="TooHardError">Thrown when NumPy's exact overlap problem for two arrays exceeds its solver.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L221">gymnasium/utils/passive_env_checker.py</see>.</remarks>
        public static void EnvDataReusePassiveChecker((object? Obs, object? Info)? previousData, (object? Obs, object? Info) data, string previousSource, string source) {
            if (previousData is not { } previous)
                throw new TypeError("cannot unpack non-iterable NoneType object");
            using RetainedData previousObs = Retain(previous.Obs);
            using RetainedData previousInfo = Retain(previous.Info);
            EnvDataReusePassiveChecker(previousObs, previousInfo, data, previousSource, source);
        }

        /// <summary>
        ///     <see cref="EnvDataReusePassiveChecker(ValueTuple{object, object}?, ValueTuple{object, object}, string, string)"/>
        ///     against data kept by <see cref="Retain"/> since the earlier call, as <c>PassiveEnvChecker</c> keeps it.
        /// </summary>
        /// <param name="previousObs">The earlier observation, retained.</param>
        /// <param name="previousInfo">The earlier info, retained.</param>
        /// <param name="data">The later call's <c>(obs, info)</c>.</param>
        /// <param name="previousSource">The earlier call, <c>reset</c> or <c>step</c>.</param>
        /// <param name="source">The later call.</param>
        /// <exception cref="TooHardError">Thrown when NumPy's exact overlap problem for two arrays exceeds its solver.</exception>
        public static void EnvDataReusePassiveChecker(RetainedData previousObs, RetainedData previousInfo, (object? Obs, object? Info) data, string previousSource, string source) {
            if (DataSharesObjects(previousObs, data.Obs))
                Logger.Warn($"The observations returned by `{previousSource}` and the following `{source}` share an object, therefore, modifying one will modify the other. Environments should return new observation data on every call as users commonly keep the data returned to them.");
            if (DataSharesObjects(previousInfo, data.Info))
                Logger.Warn($"The infos returned by `{previousSource}` and the following `{source}` share an object, therefore, modifying one will modify the other. Environments should return new info data on every call as users commonly keep the data returned to them.");
        }

        /// <summary>
        ///     Resets the env and checks what it returned (Gymnasium's <c>env_reset_passive_checker</c>): the observation
        ///     against the observation space, and that an info came back. The result is returned unchanged.
        /// </summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <typeparam name="TAct">The action type.</typeparam>
        /// <param name="env">The env.</param>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The reset's result.</returns>
        /// <exception cref="AssertionError">
        ///     Thrown for a malformed composite observation (<see cref="CheckObs"/>), or a <see langword="null"/> info: <c>The
        ///     second element returned by `env.reset()` was not a dictionary, actual type: &lt;class 'NoneType'&gt;</c>.
        /// </exception>
        /// <exception cref="TypeError">Thrown by <see cref="CheckObs"/> for a tuple-space observation without a length.</exception>
        /// <remarks>
        ///     Gymnasium's checks of the reset signature (a <c>seed</c> and <c>options</c> parameter, a <c>None</c> default) and
        ///     of the result's tuple form cannot fail in C#. Whatever the env's reset throws propagates.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L250">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static ResetResult<TObs> EnvResetPassiveChecker<TObs, TAct>(Env<TObs, TAct> env, BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> result = env.Reset(seed, options);
            CheckObs(result.Observation, env.ObservationSpace, "reset");
            if (result.Info is null)
                throw new AssertionError($"The second element returned by `env.reset()` was not a dictionary, actual type: {PyBuiltins.TypeRepr(null)}");
            return result;
        }

        /// <summary>
        ///     <see cref="EnvResetPassiveChecker{TObs, TAct}"/> through the untyped view, for code that holds envs of any type
        ///     (<c>check_env</c>).
        /// </summary>
        /// <param name="env">The env.</param>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The reset's result.</returns>
        /// <exception cref="AssertionError">Thrown for a malformed composite observation, or a <see langword="null"/> info.</exception>
        /// <exception cref="TypeError">Thrown by <see cref="CheckObs"/> for a tuple-space observation without a length.</exception>
        public static ResetResult<object?> EnvResetPassiveChecker(IEnv env, BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<object?> result = env.Reset(seed, options);
            CheckObs(result.Observation, env.ObservationSpace, "reset");
            if (result.Info is null)
                throw new AssertionError($"The second element returned by `env.reset()` was not a dictionary, actual type: {PyBuiltins.TypeRepr(null)}");
            return result;
        }

        /// <summary>
        ///     <see cref="EnvStepPassiveChecker{TObs, TAct}"/> through the untyped view: the env converts the action to its
        ///     own type first.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <param name="action">The action, of any type the env converts.</param>
        /// <returns>The step's result.</returns>
        /// <exception cref="AssertionError">Thrown for a malformed composite observation, or a <see langword="null"/> info.</exception>
        /// <exception cref="TypeError">Thrown by <see cref="CheckObs"/> for a tuple-space observation without a length.</exception>
        public static StepResult<object?> EnvStepPassiveChecker(IEnv env, object? action) {
            StepResult<object?> result = env.Step(action);
            CheckObs(result.Observation, env.ObservationSpace, "step");
            if (double.IsNaN(result.Reward))
                Logger.Warn("The reward is a NaN value.");
            if (double.IsInfinity(result.Reward))
                Logger.Warn("The reward is an inf value.");
            if (result.Info is null)
                throw new AssertionError($"The `info` returned by `step()` must be a python dictionary, actual type: {PyBuiltins.TypeRepr(null)}");
            return result;
        }

        /// <summary>
        ///     Steps the env and checks what it returned (Gymnasium's <c>env_step_passive_checker</c>): the observation, a NaN
        ///     or infinite reward, and that an info came back. The action is not checked: some envs accept out-of-bounds
        ///     actions on purpose. The result is returned unchanged.
        /// </summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <typeparam name="TAct">The action type.</typeparam>
        /// <param name="env">The env.</param>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The step's result.</returns>
        /// <exception cref="AssertionError">
        ///     Thrown for a malformed composite observation (<see cref="CheckObs"/>), or a <see langword="null"/> info: <c>The
        ///     `info` returned by `step()` must be a python dictionary, actual type: &lt;class 'NoneType'&gt;</c>.
        /// </exception>
        /// <exception cref="TypeError">Thrown by <see cref="CheckObs"/> for a tuple-space observation without a length.</exception>
        /// <remarks>
        ///     Gymnasium's checks of the result's length (the 4-tuple of the old API) and of the signal and reward types cannot
        ///     fail in C#. Whatever the env's step throws propagates.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L293">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static StepResult<TObs> EnvStepPassiveChecker<TObs, TAct>(Env<TObs, TAct> env, TAct action) {
            StepResult<TObs> result = env.Step(action);
            CheckObs(result.Observation, env.ObservationSpace, "step");
            if (double.IsNaN(result.Reward))
                Logger.Warn("The reward is a NaN value.");
            if (double.IsInfinity(result.Reward))
                Logger.Warn("The reward is an inf value.");
            if (result.Info is null)
                throw new AssertionError($"The `info` returned by `step()` must be a python dictionary, actual type: {PyBuiltins.TypeRepr(null)}");
            return result;
        }

        /// <summary>
        ///     Checks the env's render metadata, then renders and checks the result's form against the render mode (Gymnasium's
        ///     <c>env_render_passive_checker</c>): missing or malformed <c>render_modes</c> and <c>render_fps</c> warn, a render
        ///     mode the metadata does not list asserts. The result is returned unchanged.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <returns>The render's result, which the caller owns as from <c>render()</c>.</returns>
        /// <exception cref="AssertionError">
        ///     Thrown, with Gymnasium's messages, for a <c>render_fps</c> that is not positive, a render mode set although
        ///     <c>render_modes</c> is empty, or a render mode <c>render_modes</c> does not list.
        /// </exception>
        /// <exception cref="TypeError">
        ///     Thrown when <c>render_modes</c> has no length (<c>object of type 'int' has no len()</c>), Python's error from
        ///     <c>len(render_modes)</c>.
        /// </exception>
        /// <remarks>
        ///     Metadata values are read as their Python counterparts: a typed array or <see cref="List{T}"/> is a list, an
        ///     <see langword="object"/>[] a tuple, a string a string (whose <c>in</c> is a substring test, as in Python).
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L405">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        public static RenderResult EnvRenderPassiveChecker(IEnv env) {
            object? renderModes = env.Metadata.Get(Metadata.RenderModesKey);
            if (renderModes is null) {
                Logger.Warn("No render modes was declared in the environment (env.metadata['render_modes'] is None or not defined), you may have trouble when calling `.render()`.");
            } else {
                string modesType = PyBuiltins.TypeName(renderModes);
                if (modesType is not ("list" or "tuple")) {
                    Logger.Warn($"Expects the render_modes to be a sequence (i.e. list, tuple), actual type: {PyBuiltins.TypeRepr(renderModes)}");
                } else {
                    object?[] modes = SizedItems(renderModes);
                    if (!modes.All(mode => mode is string))
                        Logger.Warn($"Expects all render modes to be strings, actual types: [{string.Join(", ", modes.Select(PyBuiltins.TypeRepr))}]");
                }

                int count = PyLen(renderModes);
                object? renderFps = env.Metadata.Get(Metadata.RenderFpsKey);
                // render_fps is only required when rendering is implemented.
                if (count > 0) {
                    if (renderFps is null)
                        Logger.Warn("No render fps was declared in the environment (env.metadata['render_fps'] is None or not defined), rendering may occur at inconsistent fps.");
                    else if (!IsNumber(renderFps))
                        Logger.Warn($"Expects the `env.metadata['render_fps']` to be an integer or a float, actual type: {PyBuiltins.TypeRepr(renderFps)}");
                    else if (!IsPositive(renderFps))
                        throw new AssertionError($"Expects the `env.metadata['render_fps']` to be greater than zero, actual value: {PyBuiltins.Str(renderFps)}");
                }

                if (count == 0) {
                    if (env.RenderMode is not null)
                        throw new AssertionError($"With no render_modes, expects the Env.render_mode to be None, actual value: {env.RenderMode}");
                } else if (env.RenderMode is not null && !PyIn(env.RenderMode, renderModes)) {
                    throw new AssertionError($"The environment was initialized successfully however with an unsupported render mode. Render mode: {env.RenderMode}, modes: {PyBuiltins.Str(renderModes)}");
                }
            }

            RenderResult result = env.Render();
            if (env.RenderMode is not null)
                CheckRenderReturn(env.RenderMode, result);
            return result;
        }

        /// <summary>
        ///     Gymnasium's <c>_check_render_return</c>: warns when a render result does not have the form its mode promises
        ///     (nothing for <c>human</c>, a <c>(h, w, 3)</c> uint8 array for <c>rgb_array</c>, a 2-d array for
        ///     <c>depth_array</c>, text for <c>ansi</c>, a list of such for the <c>*_list</c> modes). Other modes pass.
        /// </summary>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="renderReturn">What <c>render()</c> returned.</param>
        /// <remarks>
        ///     An RGB and a depth frame are both NumPy arrays to Python, so either case meets the array test of either mode and
        ///     is then judged by its dtype and axes. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/passive_env_checker.py#L353">gymnasium/utils/passive_env_checker.py</see>.
        /// </remarks>
        private static void CheckRenderReturn(string renderMode, RenderResult renderReturn) {
            NDArray? frame = renderReturn switch {
                RenderResult.Rgb rgb => rgb.Frame,
                RenderResult.Depth depth => depth.Frame,
                _ => null,
            };
            switch (renderMode) {
                case "human":
                    if (!renderReturn.IsNone)
                        Logger.Warn($"Human rendering should return `None`, got {RenderTypeRepr(renderReturn)}");
                    break;
                case "rgb_array":
                    if (frame is null) {
                        Logger.Warn($"RGB-array rendering should return a numpy array, got {RenderTypeRepr(renderReturn)}");
                    } else {
                        if (frame.dtype != np.uint8)
                            Logger.Warn($"RGB-array rendering should return a numpy array with dtype uint8, got {Discrete.DTypeName(frame.dtype)}");
                        if (frame.ndim != 3)
                            Logger.Warn($"RGB-array rendering should return a numpy array with three axes, got {frame.ndim}");
                        if (frame.ndim == 3 && frame.shape[2] != 3)
                            Logger.Warn($"RGB-array rendering should return a numpy array in which the last axis has three dimensions, got {frame.shape[2]}");
                    }

                    break;
                case "depth_array":
                    if (frame is null)
                        Logger.Warn($"Depth-array rendering should return a numpy array, got {RenderTypeRepr(renderReturn)}");
                    else if (frame.ndim != 2)
                        Logger.Warn($"Depth-array rendering should return a numpy array with two axes, got {frame.ndim}");
                    break;
                case "ansi" or "ascii":
                    if (renderReturn is not RenderResult.Ansi)
                        Logger.Warn($"ANSI/ASCII rendering should produce a string, got {RenderTypeRepr(renderReturn)}");
                    break;
                default:
                    if (renderMode.EndsWith("_list", StringComparison.Ordinal)) {
                        if (renderReturn is not RenderResult.Frames frames) {
                            Logger.Warn($"Render mode `{renderMode}` should produce a list, got {RenderTypeRepr(renderReturn)}");
                        } else {
                            // Each item must match the base render mode.
                            string baseRenderMode = renderMode.Substring(0, renderMode.Length - "_list".Length);
                            foreach (RenderResult item in frames.Items)
                                CheckRenderReturn(baseRenderMode, item);
                        }
                    }

                    break;
            }
        }

        /// <summary>The Python type of a render result's counterpart, as <c>type(render_return)</c> prints it.</summary>
        /// <param name="result">The render result.</param>
        /// <returns><c>&lt;class 'NoneType'&gt;</c>, <c>&lt;class 'numpy.ndarray'&gt;</c>, <c>&lt;class 'tuple'&gt;</c>, <c>&lt;class 'str'&gt;</c> or <c>&lt;class 'list'&gt;</c>.</returns>
        private static string RenderTypeRepr(RenderResult result) => result switch {
            RenderResult.Rgb or RenderResult.Depth => "<class 'numpy.ndarray'>",
            RenderResult.RgbDepth => "<class 'tuple'>",
            RenderResult.Ansi => "<class 'str'>",
            RenderResult.Frames => "<class 'list'>",
            _ => "<class 'NoneType'>",
        };

        /// <summary>The space an <see cref="ObjectSpace"/> views, or the space itself: the space Python would see.</summary>
        /// <param name="space">The space.</param>
        /// <returns>The underlying space, <see langword="null"/> for <see langword="null"/>.</returns>
        private static Space? Unview(Space? space) => space is ObjectSpace view ? view.Inner : space;

        /// <summary>Python's <c>array.shape == space.shape</c>: equal dimensions, and never equal to a missing shape.</summary>
        /// <param name="arrayShape">An array's dimensions.</param>
        /// <param name="spaceShape">A space's shape, <see langword="null"/> for <c>None</c>.</param>
        /// <returns><see langword="true"/> when they are the same tuple.</returns>
        private static bool SameShape(long[] arrayShape, Shape? spaceShape) =>
            spaceShape is { } shape && arrayShape.SequenceEqual(Dimensions(shape));

        /// <summary>A shape's dimensions, empty for a scalar shape whose array NumSharp leaves unset.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The dimensions, Python's shape tuple.</returns>
        private static long[] Dimensions(Shape shape) => shape.dimensions ?? Array.Empty<long>();

        /// <summary>Whether a value is a NumPy array to Python (<c>isinstance(x, np.ndarray)</c>), not a NumPy scalar.</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for an array of one or more dimensions (or a 0-d array of a dtype NumPy has no scalar for).</returns>
        private static bool IsNdArray(object? value) => value is NDArray && PyBuiltins.TypeName(value) == "numpy.ndarray";

        /// <summary>Whether a value is a Python <c>int</c>: a .NET integer, a <see cref="BigInteger"/>, an enum or a <see langword="bool"/> (a subclass of <c>int</c>).</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for an int.</returns>
        private static bool IsPythonInt(object? value) => value is bool || value is not null && PyBuiltins.TypeName(value) == "int";

        /// <summary>
        ///     Whether a value is an integer or floating number to NumPy (<c>np.issubdtype(type(x), np.integer)</c> or
        ///     <c>np.floating</c>): Python ints and floats and NumPy's numeric scalars, but not bools.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for a number.</returns>
        private static bool IsNumber(object? value) => value switch {
            bool => false,
            NDArray { ndim: 0 } scalar => scalar.dtype.kind is 'i' or 'u' or 'f',
            double or float or Half => true,
            _ => PyBuiltins.TypeName(value) == "int",
        };

        /// <summary>Python's <c>x &gt; 0</c> for a number <see cref="IsNumber"/> accepts; NaN is not positive.</summary>
        /// <param name="number">The number.</param>
        /// <returns><see langword="true"/> when it is greater than zero.</returns>
        private static bool IsPositive(object number) => number switch {
            NDArray scalar => IsPositive(scalar.GetAtIndex(0)),
            BigInteger big => big.Sign > 0,
            double d => d > 0,
            float f => f > 0,
            Half h => (float)h > 0,
            // Every other number is a .NET integer or enum, which decimal holds exactly whatever its width.
            _ => Convert.ToDecimal(number, CultureInfo.InvariantCulture) > 0,
        };

        /// <summary>Whether a value is a Python tuple's counterpart: an <see cref="ITuple"/> or exactly <see langword="object"/>[] (DR-13).</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for a tuple.</returns>
        private static bool IsTuple(object? value) => value is ITuple || value is object?[] array && PyBuiltins.IsTupleArray(array);

        /// <summary>A tuple's items.</summary>
        /// <param name="value">A value <see cref="IsTuple"/> accepts.</param>
        /// <returns>The items; empty for anything else.</returns>
        private static object?[] TupleItems(object? value) => value switch {
            object?[] array => array,
            ITuple tuple => Tuple.TupleItems(tuple),
            _ => Array.Empty<object?>(),
        };

        /// <summary>
        ///     Python's <c>len(x)</c> and iteration together: the items of a value that has a length, in Python's order (an
        ///     array by its first axis, a string by code point, a dict by key, a sequence or set by item).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>Its items; their count is <c>len(x)</c>.</returns>
        /// <exception cref="TypeError">
        ///     Thrown, with Python's text, for a value without a length: <c>object of type 'int' has no len()</c>, or
        ///     <c>len() of unsized object</c> for a 0-d array NumPy would keep as an array.
        /// </exception>
        private static object?[] SizedItems(object? value) {
            switch (value) {
                case NDArray { ndim: 0 } scalar:
                    throw new TypeError(PyBuiltins.TypeName(scalar) == "numpy.ndarray" ? "len() of unsized object" : $"object of type '{PyBuiltins.TypeName(scalar)}' has no len()");
                case NDArray array:
                    // Rows are views of the observation the caller owns; they live as long as it does.
                    return Enumerable.Range(0, checked((int)array.shape[0])).Select(i => (object?)array[i.ToString(CultureInfo.InvariantCulture)]).ToArray();
                case string text:
                    return Text.CodePoints(text).Select(c => (object?)c).ToArray();
                case object?[] array:
                    return array;
                case ITuple tuple:
                    return Tuple.TupleItems(tuple);
            }

            if (Dict.AsDict(value) is { } entries)
                return entries.Keys.Select(k => (object?)k).ToArray();
            if (value is IEnumerable sequence && IsSized(value))
                return sequence.Cast<object?>().ToArray();
            throw new TypeError($"object of type '{PyBuiltins.TypeName(value)}' has no len()");
        }

        /// <summary>Whether a sequence knows its length (a collection, generic or not), as Python's sized containers do; a lazy enumeration does not.</summary>
        /// <param name="value">The sequence.</param>
        /// <returns><see langword="true"/> for a collection.</returns>
        private static bool IsSized(object value) =>
            value is ICollection || value.GetType().GetInterfaces().Any(i => i.IsGenericType && (i.GetGenericTypeDefinition() == typeof(ICollection<>) || i.GetGenericTypeDefinition() == typeof(IReadOnlyCollection<>)));

        /// <summary>Python's <c>len(x)</c>, without making the row views <see cref="SizedItems"/> makes of an array.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The length.</returns>
        /// <exception cref="TypeError">Thrown, with Python's text, for a value without a length (see <see cref="SizedItems"/>).</exception>
        private static int PyLen(object? value) =>
            value is NDArray { ndim: > 0 } array ? checked((int)array.shape[0]) : SizedItems(value).Length;

        /// <summary>
        ///     Python's <c>mode in container</c> for a render mode: a substring test in a string, a member test in anything else
        ///     with items. NumSharp has no string arrays, so an array never contains a mode.
        /// </summary>
        /// <param name="mode">The render mode.</param>
        /// <param name="container">The <c>render_modes</c> value, whose length was already taken.</param>
        /// <returns><see langword="true"/> when the mode is in it.</returns>
        /// <exception cref="TypeError">Thrown for a container without a length (see <see cref="SizedItems"/>).</exception>
        private static bool PyIn(string mode, object container) => container switch {
            string text => text.Contains(mode, StringComparison.Ordinal),
            NDArray => false,
            _ => SizedItems(container).Any(item => item is string s && string.Equals(s, mode, StringComparison.Ordinal)),
        };
    }

    /// <summary>What <see cref="PassiveChecker.Retain"/> kept of a value, which decides how it is compared.</summary>
    public enum RetainedKind {
        /// <summary>An immutable or unknown value (a number, a string, a NumPy scalar, a list), never shared.</summary>
        Other,

        /// <summary>A dict: the container itself, for identity, and its entries.</summary>
        Dict,

        /// <summary>A tuple: its items.</summary>
        Tuple,

        /// <summary>An array: a view that keeps its memory alive.</summary>
        Array,
    }

    /// <summary>
    ///     The part of an observation or info that <see cref="PassiveChecker.DataSharesObjects(RetainedData?, object?)"/>
    ///     compares across calls: containers by reference, arrays as views that keep their memory alive. Dispose it to
    ///     release the views, which otherwise hold that memory until they are collected.
    /// </summary>
    public sealed class RetainedData : IDisposable {
        /// <summary>Records a retained value.</summary>
        /// <param name="kind">What was retained.</param>
        /// <param name="typeName">The value's Python type name, compared first.</param>
        /// <param name="original">The container itself, for identity; <see langword="null"/> for arrays and leaves.</param>
        internal RetainedData(RetainedKind kind, string typeName, object? original) {
            Kind = kind;
            TypeName = typeName;
            Original = original;
        }

        /// <summary>What was retained.</summary>
        public RetainedKind Kind { get; }

        /// <summary>The value's Python type name (<c>dict</c>, <c>tuple</c>, <c>numpy.ndarray</c>, ...), which must match the other value's.</summary>
        public string TypeName { get; }

        /// <summary>The container, for identity; it is not owned.</summary>
        internal object? Original { get; }

        /// <summary>A view of the array, owned: it keeps the array's memory alive until disposed.</summary>
        internal NDArray? View { get; init; }

        /// <summary>The retained entries of a dict, owned.</summary>
        internal Dictionary<string, RetainedData>? Entries { get; init; }

        /// <summary>The retained items of a tuple, owned.</summary>
        internal List<RetainedData>? Items { get; init; }

        /// <summary>Releases the views, nested ones included; the original data is untouched.</summary>
        public void Dispose() {
            View?.Dispose();
            if (Entries is not null)
                foreach (RetainedData entry in Entries.Values)
                    entry.Dispose();
            if (Items is not null)
                foreach (RetainedData item in Items)
                    item.Dispose();
        }
    }
}
