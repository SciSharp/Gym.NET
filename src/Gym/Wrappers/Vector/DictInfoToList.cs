#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Wrappers.Vector {
    /// <summary>
    ///     A vector step whose infos are one dict per sub-env (<see cref="DictInfoToList{TObs, TAct}"/>'s result).
    /// </summary>
    /// <typeparam name="TObs">The batched observation type.</typeparam>
    /// <param name="Observations">The batched observations.</param>
    /// <param name="Rewards">The rewards, float64.</param>
    /// <param name="Terminations">Which sub-envs terminated.</param>
    /// <param name="Truncations">Which sub-envs were truncated.</param>
    /// <param name="Infos">The i-th sub-env's info at index i.</param>
    public readonly record struct VectorListStepResult<TObs>(TObs Observations, NDArray Rewards, NDArray Terminations, NDArray Truncations, IReadOnlyList<Info> Infos)
        : INDArrayCarrier {
        /// <summary>Yields every array the result holds to a scope (D19), so none is reclaimed with the scope's temporaries.</summary>
        /// <param name="scope">The scope.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => Gym.Observations.ResultEquality.Yield(scope, Observations, Rewards, Terminations, Truncations, Infos);
    }

    /// <summary>A vector reset whose infos are one dict per sub-env (<see cref="DictInfoToList{TObs, TAct}"/>'s result).</summary>
    /// <typeparam name="TObs">The batched observation type.</typeparam>
    /// <param name="Observations">The batched first observations.</param>
    /// <param name="Infos">The i-th sub-env's info at index i.</param>
    public readonly record struct VectorListResetResult<TObs>(TObs Observations, IReadOnlyList<Info> Infos) : INDArrayCarrier {
        /// <summary>Yields every array the result holds to a scope (D19), so none is reclaimed with the scope's temporaries.</summary>
        /// <param name="scope">The scope.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => Gym.Observations.ResultEquality.Yield(scope, Observations, Infos);
    }

    /// <summary>
    ///     Converts the vector env's dict of batched infos into a list of one info dict per sub-env, as Gymnasium's
    ///     <c>DictInfoToList</c> does: each key's array is split by sub-env, the <c>_key</c> masks say which sub-envs have the
    ///     key, and nested dicts convert likewise.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     A vector env's results carry a dict of infos (<see cref="VectorStepResult{TObs}"/>), so the list comes from this
    ///     class's own <see cref="Step"/> and <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/>, which
    ///     hide the base ones: called on a <see cref="DictInfoToList{TObs, TAct}"/> they return Gymnasium's lists; called
    ///     through <see cref="IVectorEnv"/> or <see cref="VectorEnv{TObs, TAct}"/> they return the dict unchanged. Put it
    ///     outermost, as Gymnasium's wrappers that read infos expect a dict too. Each element is a copy (NumPy's scalar of
    ///     the batch), so the list outlives the batch.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/dict_info_to_list.py#L13">gymnasium/wrappers/vector/dict_info_to_list.py</see>.</para>
    /// </remarks>
    public class DictInfoToList<TObs, TAct> : VectorWrapper<TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public DictInfoToList(VectorEnv<TObs, TAct> env) : base(env) { }

        /// <summary>Steps the env and converts its infos to a list (Gymnasium's <c>step</c>).</summary>
        /// <param name="actions">The actions, passed on.</param>
        /// <returns>The transition with one info per sub-env.</returns>
        /// <exception cref="AssertionError">Thrown when an info value is neither a dict nor an array of one entry per sub-env.</exception>
        public new VectorListStepResult<TObs> Step(TAct actions) {
            VectorStepResult<TObs> result = base.Step(actions);
            return new VectorListStepResult<TObs>(result.Observations, result.Rewards, result.Terminations, result.Truncations, ConvertInfoToList(result.Infos));
        }

        /// <summary>Resets the env and converts its infos to a list (Gymnasium's <c>reset</c>).</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The first observations with one info per sub-env.</returns>
        /// <exception cref="AssertionError">Thrown when an info value is neither a dict nor an array of one entry per sub-env.</exception>
        public new VectorListResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = base.Reset(seed, options);
            return new VectorListResetResult<TObs>(result.Observations, ConvertInfoToList(result.Infos));
        }

        /// <summary>
        ///     The dict of batched infos as one info per sub-env (<c>_convert_info_to_list</c>): keys starting with <c>_</c> are
        ///     masks and are skipped, a key's mask (when present) keeps it only for the sub-envs it marks, and dicts convert
        ///     recursively.
        /// </summary>
        /// <param name="vectorInfos">The env's infos.</param>
        /// <returns>The list, one new info per sub-env.</returns>
        /// <exception cref="AssertionError">Thrown when a value is neither a dict nor an array, or its length is not the number of sub-envs, with Gymnasium's messages.</exception>
        public IReadOnlyList<Info> ConvertInfoToList(IReadOnlyDictionary<string, object?> vectorInfos) {
            var listInfo = Enumerable.Range(0, NumEnvs).Select(_ => new Info()).ToList();
            foreach (KeyValuePair<string, object?> entry in vectorInfos) {
                string key = entry.Key;
                if (key.StartsWith('_'))
                    continue;
                string binaryKey = $"_{key}";
                vectorInfos.TryGetValue(binaryKey, out object? mask);
                bool hasMask = vectorInfos.ContainsKey(binaryKey);

                if (entry.Value is IReadOnlyDictionary<string, object?> nested) {
                    IReadOnlyList<Info> valueListInfo = ConvertInfoToList(nested);
                    if (valueListInfo.Count != NumEnvs)
                        throw new AssertionError($"Expects {PyBuiltins.Repr(valueListInfo)} to have length equal to the num-envs ({NumEnvs}), actual length is {valueListInfo.Count}");
                    if (hasMask) {
                        bool[] present = Flags(mask, binaryKey);
                        for (int envNum = 0; envNum < NumEnvs; envNum++) {
                            if (present[envNum])
                                listInfo[envNum][key] = valueListInfo[envNum];
                        }
                    } else {
                        for (int envNum = 0; envNum < NumEnvs; envNum++)
                            listInfo[envNum][key] = valueListInfo[envNum];
                    }

                    continue;
                }

                IReadOnlyList<object?> values = Items(entry.Value);
                if (values.Count != NumEnvs)
                    throw new AssertionError($"Expects {PyBuiltins.Repr(entry.Value)} to have length equal to the num-envs ({NumEnvs}), actual length is {values.Count}");
                if (hasMask) {
                    bool[] present = Flags(mask, binaryKey);
                    for (int envNum = 0; envNum < NumEnvs; envNum++) {
                        if (present[envNum])
                            listInfo[envNum][key] = values[envNum];
                    }
                } else {
                    for (int envNum = 0; envNum < NumEnvs; envNum++)
                        listInfo[envNum][key] = values[envNum];
                }
            }

            return listInfo;
        }

        /// <summary>A mask's flags, one per sub-env.</summary>
        /// <param name="mask">The mask, a bool array.</param>
        /// <param name="binaryKey">The mask's key, for the message.</param>
        /// <returns>The flags.</returns>
        /// <exception cref="AssertionError">Thrown when the mask's length is not the number of sub-envs.</exception>
        private bool[] Flags(object? mask, string binaryKey) {
            IReadOnlyList<object?> items = Items(mask);
            if (items.Count != NumEnvs)
                throw new AssertionError($"Expects {PyBuiltins.Repr(mask)} to have length equal to the num-envs ({NumEnvs}), actual length is {items.Count}");
            return items.Select(item => item switch {
                bool flag => flag,
                NDArray array => (bool)array,
                _ => item is not null,
            }).ToArray();
        }

        /// <summary>
        ///     The per-sub-env entries of an info value: an array's elements (each a copy, as NumPy's scalar is), or an object
        ///     array's (<see langword="object"/>?[]) items.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The entries.</returns>
        /// <exception cref="AssertionError">Thrown for a value that is no array (<c>assert isinstance(value, np.ndarray)</c>).</exception>
        private static IReadOnlyList<object?> Items(object? value) {
            switch (value) {
                case NDArray array when array.ndim > 0: {
                    var items = new object?[array.shape[0]];
                    for (int i = 0; i < items.Length; i++) {
                        using NDArray element = array[i];
                        items[i] = element.copy();
                    }

                    return items;
                }
                case object?[] items:
                    return items;
                default:
                    throw new AssertionError("");
            }
        }
    }
}
