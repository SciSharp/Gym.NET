#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using SpaceTuple = Gym.Spaces.Tuple;

namespace Gym.Vector {
    public static partial class VectorUtils {
        /// <summary>
        ///     Allocates the buffer an <see cref="AsyncVectorEnv{TObs, TAct}"/> with <c>shared_memory</c> collects its sub-envs'
        ///     observations in (Gymnasium's <c>create_shared_memory</c>): for an array space, one flat zeroed array of
        ///     <c>n * prod(shape)</c> elements of the space's dtype; a tuple of buffers for a <see cref="SpaceTuple"/>, a dict of
        ///     them for a <see cref="Dict"/>, int32 character indices for a <see cref="Text"/>, and for a <see cref="OneOf"/> an
        ///     int64 array of the chosen subspaces followed by every subspace's buffer.
        /// </summary>
        /// <param name="space">One sub-env's observation space.</param>
        /// <param name="n">The number of sub-envs.</param>
        /// <returns>The buffer; the caller owns its arrays (<see cref="ReleaseSharedMemory"/> disposes them).</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <exception cref="TypeError">Thrown for a <see cref="Graph"/> or <see cref="Sequence"/>, whose shape varies, with Gymnasium's message.</exception>
        /// <exception cref="CustomSpaceError">Thrown for a space of any other class, with Gymnasium's message.</exception>
        /// <remarks>
        ///     <para>
        ///     Gymnasium's buffer is a <c>multiprocessing</c> array its worker processes write into. Gym.NET's workers are
        ///     threads of the same process, which share every array, so the buffer is an ordinary NumSharp array: what stays is
        ///     the layout, the dtypes (a float16 or bool space gets a float16 or bool buffer, where Python needs raw bytes or
        ///     <c>c_bool</c>) and the refusals, which decide whether <c>AsyncVectorEnv(..., shared_memory=True)</c> accepts a
        ///     space. Untyped <see cref="ObjectSpace"/> views are read through to the space they view.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/utils/shared_memory.py#L39">gymnasium/vector/utils/shared_memory.py</see>.</para>
        /// </remarks>
        public static object CreateSharedMemory(Space space, int n = 1) {
            if (space is null)
                throw new ArgumentNullException(nameof(space));
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case Box or Discrete or MultiDiscrete or MultiBinary: {
                    long size = n * ShapeSize(space.Shape!.Value);
                    return np.zeros(new Shape(size), space.DType!);
                }
                case SpaceTuple tuple: {
                    var memories = new object?[tuple.Spaces.Count];
                    try {
                        for (int i = 0; i < memories.Length; i++)
                            memories[i] = CreateSharedMemory(tuple.Spaces[i], n);
                    } catch {
                        // A later subspace refused: the buffers made so far have no other owner.
                        ReleaseSharedMemory(memories);
                        throw;
                    }

                    return memories;
                }
                case Dict dict: {
                    var memories = new Info();
                    try {
                        foreach (KeyValuePair<string, Space> entry in dict.Spaces)
                            memories[entry.Key] = CreateSharedMemory(entry.Value, n);
                    } catch {
                        ReleaseSharedMemory(memories);
                        throw;
                    }

                    return memories;
                }
                case Text text:
                    return np.zeros(new Shape(n * text.MaxLength), NPTypeCode.Int32);
                case OneOf oneOf: {
                    var memories = new object?[oneOf.Spaces.Count + 1];
                    try {
                        memories[0] = np.zeros(new Shape(n), NPTypeCode.Int64);
                        for (int i = 0; i < oneOf.Spaces.Count; i++)
                            memories[i + 1] = CreateSharedMemory(oneOf.Spaces[i], n);
                    } catch {
                        ReleaseSharedMemory(memories);
                        throw;
                    }

                    return memories;
                }
                case Graph or Sequence:
                    throw new TypeError($"As {space} has a dynamic shape so its not possible to make a static shared memory. For `AsyncVectorEnv`, disable `shared_memory`.");
                default:
                    throw new CustomSpaceError($"Space of type `{PyBuiltins.TypeRepr(space)}` doesn't have an registered `create_shared_memory` function. Register `{PyBuiltins.TypeRepr(space)}` for `create_shared_memory` to support it.");
            }
        }

        /// <summary>
        ///     The batch of observations a shared buffer holds (Gymnasium's <c>read_from_shared_memory</c>): for an array
        ///     space, a <c>(n, *shape)</c> view of the buffer, so later writes show through it; tuples and dicts of those for
        ///     <see cref="SpaceTuple"/> and <see cref="Dict"/>; for a <see cref="Text"/> the decoded strings (a new tuple); for a
        ///     <see cref="OneOf"/> a tuple of <c>(index, sample)</c> pairs.
        /// </summary>
        /// <param name="space">One sub-env's observation space.</param>
        /// <param name="sharedMemory">The buffer <see cref="CreateSharedMemory"/> made for <paramref name="space"/>.</param>
        /// <param name="n">The number of sub-envs.</param>
        /// <returns>The batch: arrays that are views of the buffer (dispose them; the buffer stays its owner's), or <see langword="object"/>?[] / <see cref="Info"/> of them.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <exception cref="CustomSpaceError">Thrown for a space with no shared buffer, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown when a tuple buffer holds another number of members than the space (Python's <c>zip(strict=True)</c> text).</exception>
        /// <exception cref="KeyError">Thrown for a <see cref="OneOf"/> whose chosen subspace is a <see cref="Dict"/>: Gymnasium indexes its batch by position, which a dict refuses.</exception>
        /// <exception cref="IndexError">Thrown for a <see cref="OneOf"/> whose chosen subspace is a tuple with fewer members than the sub-env's index, as in Gymnasium.</exception>
        /// <exception cref="InvalidCastException">Thrown when <paramref name="sharedMemory"/> is not the buffer of <paramref name="space"/>.</exception>
        /// <remarks>
        ///     <para>
        ///     A <see cref="OneOf"/> pair holds the chosen index as a 0-d int64 array (NumPy's <c>np.int64</c>) and the
        ///     <c>index</c>-th element of that subspace's batch: a row view for an array subspace, and, as in Gymnasium, the
        ///     <c>index</c>-th member of a tuple subspace's batch (not that sub-env's sample).
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/utils/shared_memory.py#L140">gymnasium/vector/utils/shared_memory.py</see>.</para>
        /// </remarks>
        public static object? ReadFromSharedMemory(Space space, object sharedMemory, int n = 1) {
            if (space is null)
                throw new ArgumentNullException(nameof(space));
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case Box or Discrete or MultiDiscrete or MultiBinary: {
                    long[] dimensions = new long[] { n }.Concat(space.Shape!.Value.dimensions).ToArray();
                    // np.frombuffer(...).reshape((n,) + shape): a view of the buffer, never a copy.
                    return ((NDArray)sharedMemory).reshape(new Shape(dimensions));
                }
                case SpaceTuple tuple: {
                    var memories = (object?[])sharedMemory;
                    if (memories.Length != tuple.Spaces.Count)
                        throw new ValueError(memories.Length < tuple.Spaces.Count ? "zip() argument 2 is longer than argument 1" : "zip() argument 2 is shorter than argument 1");
                    return tuple.Spaces.Select((subspace, i) => ReadFromSharedMemory(subspace, memories[i]!, n)).ToArray();
                }
                case Dict dict: {
                    var memories = (IReadOnlyDictionary<string, object?>)sharedMemory;
                    var batch = new Info();
                    foreach (KeyValuePair<string, Space> entry in dict.Spaces)
                        batch[entry.Key] = ReadFromSharedMemory(entry.Value, memories[entry.Key]!, n);
                    return batch;
                }
                case Text text: {
                    var codes = (NDArray)sharedMemory;
                    int setSize = text.CharacterSet.Count;
                    var strings = new object?[n];
                    for (int i = 0; i < n; i++) {
                        var characters = new System.Text.StringBuilder();
                        for (long j = 0; j < text.MaxLength; j++) {
                            int value = codes.GetInt32(i * text.MaxLength + j);
                            // Indices at or past the set's size are the padding flatten writes.
                            if (value < setSize)
                                characters.Append(text.CharacterList[value]);
                        }

                        strings[i] = characters.ToString();
                    }

                    return strings;
                }
                case OneOf oneOf: {
                    var memories = (object?[])sharedMemory;
                    var indexes = (NDArray)memories[0]!;
                    object?[] subspaceSamples = oneOf.Spaces.Select((subspace, i) => ReadFromSharedMemory(subspace, memories[i + 1]!, n)).ToArray();
                    var pairs = new object?[n];
                    for (int index = 0; index < n; index++) {
                        long sampleIndex = indexes.GetInt64(index);
                        pairs[index] = new object?[] { NDArray.Scalar(sampleIndex), ItemAt(subspaceSamples[sampleIndex], index) };
                    }

                    // The other subspaces' batches were read only to pick from; release the views nobody received.
                    var kept = new HashSet<object?>(pairs.Select(pair => ((object?[])pair!)[1]), ReferenceEqualityComparer.Instance);
                    foreach (object? batch in subspaceSamples)
                        ReleaseUnkept(batch, kept);
                    return pairs;
                }
                default:
                    throw new CustomSpaceError($"Space of type `{PyBuiltins.TypeRepr(space)}` doesn't have an registered `read_from_shared_memory` function. Register `{PyBuiltins.TypeRepr(space)}` for `read_from_shared_memory` to support it.");
            }
        }

        /// <summary>
        ///     Writes one sub-env's observation into its slot of a shared buffer (Gymnasium's <c>write_to_shared_memory</c>):
        ///     an array observation is cast to the space's dtype and flattened into elements
        ///     <c>[index * size, (index + 1) * size)</c>; tuples and dicts write member by member; a text writes its flattened
        ///     character indices; a <see cref="OneOf"/> pair writes its index and, into that subspace's buffer only, its sample.
        /// </summary>
        /// <param name="space">One sub-env's observation space.</param>
        /// <param name="index">The sub-env, in <c>[0, n)</c>.</param>
        /// <param name="value">Its observation; read, never kept.</param>
        /// <param name="sharedMemory">The buffer <see cref="CreateSharedMemory"/> made for <paramref name="space"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when a tuple observation's length differs from the space's (Python's <c>zip(strict=True)</c> text), a <see cref="OneOf"/> value is not a pair, or the observation does not fit the slot.</exception>
        /// <exception cref="TypeError">Thrown when a tuple or <see cref="OneOf"/> observation cannot be unpacked.</exception>
        /// <exception cref="CustomSpaceError">Thrown for a space with no shared buffer, with Gymnasium's message.</exception>
        /// <remarks>
        ///     Every sub-env writes its own disjoint slot, which is what lets Gymnasium's worker processes (and Gym.NET's worker
        ///     threads) write the same buffer at once without a lock.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/utils/shared_memory.py#L251">gymnasium/vector/utils/shared_memory.py</see>.
        /// </remarks>
        public static void WriteToSharedMemory(Space space, int index, object? value, object sharedMemory) {
            if (space is null)
                throw new ArgumentNullException(nameof(space));
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case Box or Discrete or MultiDiscrete or MultiBinary: {
                    long size = ShapeSize(space.Shape!.Value);
                    var destination = (NDArray)sharedMemory;
                    // np.asarray(value, dtype=space.dtype).flatten(): an unsafe cast to the space's dtype, then a flat copy.
                    NDArray given = value as NDArray ?? np.asanyarray(value);
                    try {
                        using NDArray cast = given.astype(space.DType!);
                        using NDArray flat = cast.flatten();
                        using NDArray slot = destination[$"{index * size}:{(index + 1) * size}"];
                        np.copyto(slot, flat);
                    } finally {
                        // Only an array made here from a non-array value is this method's; the observation stays its owner's.
                        if (!ReferenceEquals(given, value))
                            given.Dispose();
                    }

                    return;
                }
                case SpaceTuple tuple: {
                    IReadOnlyList<object?> values = Items(value);
                    var memories = (object?[])sharedMemory;
                    // zip(values, shared_memory, space.spaces, strict=True): the members before a length mismatch are
                    // written, then the mismatch raises with the text of the first argument found too short or too long.
                    for (int i = 0;; i++) {
                        if (i == values.Count) {
                            if (i < memories.Length)
                                throw new ValueError("zip() argument 2 is longer than argument 1");
                            if (i < tuple.Spaces.Count)
                                throw new ValueError("zip() argument 3 is longer than arguments 1-2");
                            return;
                        }

                        if (i == memories.Length)
                            throw new ValueError("zip() argument 2 is shorter than argument 1");
                        if (i == tuple.Spaces.Count)
                            throw new ValueError("zip() argument 3 is shorter than arguments 1-2");
                        WriteToSharedMemory(tuple.Spaces[i], index, values[i], memories[i]!);
                    }
                }
                case Dict dict: {
                    var values = (IReadOnlyDictionary<string, object?>)value!;
                    var memories = (IReadOnlyDictionary<string, object?>)sharedMemory;
                    foreach (KeyValuePair<string, Space> entry in dict.Spaces)
                        WriteToSharedMemory(entry.Value, index, values[entry.Key], memories[entry.Key]!);
                    return;
                }
                case Text text: {
                    long size = text.MaxLength;
                    var destination = (NDArray)sharedMemory;
                    using var flat = (NDArray)SpaceUtils.Flatten(text, value)!;
                    using NDArray slot = destination[$"{index * size}:{(index + 1) * size}"];
                    np.copyto(slot, flat);
                    return;
                }
                case OneOf oneOf: {
                    IReadOnlyList<object?> pair = Items(value);
                    if (pair.Count != 2)
                        throw new ValueError($"too many values to unpack (expected 2, got {pair.Count})");
                    long subspaceIndex = Convert.ToInt64(pair[0] is NDArray scalar ? scalar.GetAtIndex(0) : pair[0], System.Globalization.CultureInfo.InvariantCulture);
                    var memories = (object?[])sharedMemory;
                    var indexes = (NDArray)memories[0]!;
                    indexes.SetInt64(subspaceIndex, index);
                    // Only the chosen subspace's buffer is written; the others keep whatever they held, as in Gymnasium.
                    WriteToSharedMemory(oneOf.Spaces[(int)subspaceIndex], index, pair[1], memories[1 + subspaceIndex]!);
                    return;
                }
                default:
                    throw new CustomSpaceError($"Space of type `{PyBuiltins.TypeRepr(space)}` doesn't have an registered `write_to_shared_memory` function. Register `{PyBuiltins.TypeRepr(space)}` for `write_to_shared_memory` to support it.");
            }
        }

        /// <summary>Disposes every array of a shared buffer: an array, or the arrays inside its tuples and dicts.</summary>
        /// <param name="sharedMemory">The buffer, or <see langword="null"/>.</param>
        /// <remarks>Gymnasium's buffer is freed by the garbage collector; Gym.NET's owner releases it when it closes (D19).</remarks>
        public static void ReleaseSharedMemory(object? sharedMemory) {
            switch (sharedMemory) {
                case NDArray array:
                    array.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        ReleaseSharedMemory(item);
                    break;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? item in dict.Values)
                        ReleaseSharedMemory(item);
                    break;
            }
        }

        /// <summary>The number of elements of one observation of a shape, <c>int(np.prod(shape))</c> (1 for a scalar shape).</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The product of its dimensions.</returns>
        private static long ShapeSize(Shape shape) => shape.dimensions.Aggregate(1L, (product, dimension) => product * dimension);

        /// <summary>A tuple-like value's items: an <see langword="object"/>?[], a list, or an <see cref="System.Runtime.CompilerServices.ITuple"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The items.</returns>
        /// <exception cref="TypeError">Thrown for anything else, as Python's unpacking of a non-iterable.</exception>
        private static IReadOnlyList<object?> Items(object? value) => value switch {
            object?[] items => items,
            System.Collections.IList list and not NDArray => list.Cast<object?>().ToArray(),
            System.Runtime.CompilerServices.ITuple tuple => Enumerable.Range(0, tuple.Length).Select(i => tuple[i]).ToArray(),
            _ => throw new TypeError($"cannot unpack non-iterable {PyBuiltins.TypeName(value)} object"),
        };

        /// <summary>The <paramref name="index"/>-th element of a batch read from a buffer (<c>batch[index]</c>): a row view of an array, a member of a tuple.</summary>
        /// <param name="batch">The batch.</param>
        /// <param name="index">The element.</param>
        /// <returns>The element.</returns>
        /// <exception cref="TypeError">Thrown for a batch that cannot be indexed, as Python's subscript fails.</exception>
        /// <exception cref="KeyError">Thrown for a dict batch, which has no positional keys.</exception>
        /// <exception cref="IndexError">Thrown for a tuple batch with fewer members than <paramref name="index"/>.</exception>
        private static object? ItemAt(object? batch, int index) => batch switch {
            NDArray array => array[index],
            // A tuple batch indexed past its members: Python's IndexError, as Gymnasium's read of such a OneOf raises.
            object?[] items => index < items.Length ? items[index] : throw new IndexError("tuple index out of range"),
            // A dict indexed by a position finds no such key, as Python's dict[index] does not.
            IReadOnlyDictionary<string, object?> => throw new KeyError(PyBuiltins.Repr(index)),
            _ => throw new TypeError($"'{PyBuiltins.TypeName(batch)}' object is not subscriptable"),
        };

        /// <summary>Disposes the arrays of a read batch that no pair kept (the views of the subspaces not chosen).</summary>
        /// <param name="batch">The batch.</param>
        /// <param name="kept">The objects the pairs hold, compared by reference.</param>
        private static void ReleaseUnkept(object? batch, HashSet<object?> kept) {
            // A kept object is a pair's sample, whole: a nested OneOf's pair keeps its index and sample alive with it, so
            // nothing inside a kept container is released either.
            if (batch is null || kept.Contains(batch))
                return;
            switch (batch) {
                case NDArray array:
                    array.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        ReleaseUnkept(item, kept);
                    break;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? item in dict.Values)
                        ReleaseUnkept(item, kept);
                    break;
            }
        }
    }
}
