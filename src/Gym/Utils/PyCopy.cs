#nullable enable
using System;
using System.Collections;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Python's <c>copy.deepcopy</c> for the values Gymnasium deep-copies: specs (<c>deepcopy(env.spec)</c> before a
    ///     wrapper edits it) and recorded constructor arguments (<c>RecordConstructorArgs</c>). The copy shares nothing
    ///     mutable with the original, so editing one never shows through the other.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Copied: <see cref="Info"/> (and <see cref="Metadata"/>), <see cref="EnvSpec"/>, <see cref="WrapperSpec"/>,
    ///     <see cref="NDArray"/> (a new buffer), .NET arrays, and any <see cref="IList"/> or <see cref="IDictionary"/> with a
    ///     public parameterless constructor (<c>List&lt;T&gt;</c>, <c>Dictionary&lt;TKey, TValue&gt;</c>), element by element.
    ///     Shared: immutable values (strings, numbers, enums, <see langword="null"/>) and every other object, which Python
    ///     would copy through its <c>__deepcopy__</c>/pickle protocol and .NET has no general way to; a mutable object of
    ///     another type inside a spec's kwargs is therefore shared by the copy.
    ///     </para>
    ///     <para>
    ///     Python's memo, which copies an object reachable twice only once and survives cycles, is not reproduced: a value
    ///     reachable twice is copied twice, and a cycle recurses without end.
    ///     </para>
    /// </remarks>
    public static class PyCopy {
        /// <summary>Deep-copies a value as the class remarks describe.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The copy, or <paramref name="value"/> itself when it is immutable or of a type that is shared.</returns>
        /// <exception cref="System.Reflection.TargetInvocationException">
        ///     Thrown when the parameterless constructor of a copied collection type throws (the copy is created through it).
        /// </exception>
        public static object? DeepCopy(object? value) {
            switch (value) {
                case null:
                case string:
                case bool or char or sbyte or byte or short or ushort or int or uint or long or ulong:
                case float or double or decimal or Half or BigInteger:
                case Enum:
                    return value;
                case NDArray array:
                    return array.copy();
                case Info info:
                    return info.DeepCopy();
                case EnvSpec spec:
                    return spec.DeepCopy();
                case WrapperSpec wrapperSpec:
                    return wrapperSpec.DeepCopy();
                case Array array: {
                    var copy = Array.CreateInstance(array.GetType().GetElementType()!, LengthsOf(array));
                    CopyElements(array, copy);
                    return copy;
                }
                case IList list when HasParameterlessConstructor(list.GetType()): {
                    var copy = (IList)Activator.CreateInstance(list.GetType())!;
                    foreach (object? item in list)
                        copy.Add(DeepCopy(item));
                    return copy;
                }
                case IDictionary dictionary when HasParameterlessConstructor(dictionary.GetType()): {
                    var copy = (IDictionary)Activator.CreateInstance(dictionary.GetType())!;
                    // Keys are hashable, hence immutable in Python; only the values are copied.
                    foreach (DictionaryEntry entry in dictionary)
                        copy.Add(entry.Key, DeepCopy(entry.Value));
                    return copy;
                }
                default:
                    return value;
            }
        }

        /// <summary>The length of every dimension of an array, so a multi-dimensional .NET array is copied with its shape.</summary>
        /// <param name="array">The array.</param>
        /// <returns>One length per rank.</returns>
        private static int[] LengthsOf(Array array) {
            var lengths = new int[array.Rank];
            for (int i = 0; i < lengths.Length; i++)
                lengths[i] = array.GetLength(i);
            return lengths;
        }

        /// <summary>Deep-copies every element of <paramref name="source"/> into the same position of <paramref name="target"/>.</summary>
        /// <param name="source">The original array.</param>
        /// <param name="target">An array of the same element type and shape.</param>
        private static void CopyElements(Array source, Array target) {
            if (source.Length == 0)
                return;
            // Walk the indices of any rank in row-major order, as the enumeration of a multi-dimensional array does.
            var index = new int[source.Rank];
            for (int n = 0; n < source.Length; n++) {
                target.SetValue(DeepCopy(source.GetValue(index)), index);
                for (int d = source.Rank - 1; d >= 0; d--) {
                    if (++index[d] < source.GetLength(d))
                        break;
                    index[d] = 0;
                }
            }
        }

        /// <summary>Whether a collection type can be created empty and filled, which the element-wise copy needs.</summary>
        /// <param name="type">The collection's type.</param>
        /// <returns><see langword="true"/> for a non-abstract type with a public parameterless constructor.</returns>
        private static bool HasParameterlessConstructor(Type type) => !type.IsAbstract && type.GetConstructor(Type.EmptyTypes) != null;
    }
}
