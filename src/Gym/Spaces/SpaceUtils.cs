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
using Gym.Utils;
using Gym.Vector;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Spaces {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.spaces.utils</c>: flattening a space's elements into one array for learning code
    ///     (<see cref="Flatten"/>), back (<see cref="Unflatten"/>), the space they flatten into (<see cref="FlattenSpace"/>)
    ///     and its size (<see cref="Flatdim"/>), plus the dtype-and-shape equivalence vector envs check
    ///     (<see cref="IsSpaceDtypeShapeEquiv"/>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Each function dispatches on the space's type, as Gymnasium's <c>singledispatch</c> does: a subclass of a Gym.NET
    ///     space takes its base's path, and an unknown custom space raises <see cref="NotImplementedError"/>. Spaces that
    ///     cannot become one array (a <see cref="Graph"/>, a <see cref="Sequence"/>, a composite holding one) flatten to the
    ///     same structure with flattened parts.
    ///     </para>
    ///     <para>
    ///     Every array returned is new and owned by the caller, also where Gymnasium returns a view of the input
    ///     (<c>unflatten</c> of a box reshapes the input itself): the values are Gymnasium's, the memory is never shared.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/utils.py">gymnasium/spaces/utils.py</see>.
    ///     </para>
    /// </remarks>
    public static class SpaceUtils {
        /// <summary>Gymnasium's message for a space that holds a <c>Graph</c> or <c>Sequence</c> and so cannot be flattened to an array.</summary>
        /// <param name="space">The space.</param>
        /// <returns>The text.</returns>
        private static string NotFlattenable(Space space) =>
            $"{space} cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace";

        /// <summary>
        ///     The number of entries the flattened space has (Gymnasium's <c>flatdim</c>): the element count of a box or
        ///     <see cref="MultiBinary"/>, <c>n</c> of a <see cref="Discrete"/> (one-hot), the sum of <c>nvec</c> of a
        ///     <see cref="MultiDiscrete"/>, the sum over a composite's subspaces, <c>max_length</c> of a <see cref="Text"/>,
        ///     and one more than the largest subspace for a <see cref="OneOf"/> (its index comes first).
        /// </summary>
        /// <param name="space">The space.</param>
        /// <returns>The flattened size.</returns>
        /// <exception cref="ValueError">Thrown for a space that cannot be flattened to an array (a graph, a sequence, or a composite holding one).</exception>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space flattenable but unknown to this function.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        public static long Flatdim(Space space) {
            // An untyped env's space is a view (ObjectSpace): dispatch on the space it shows, as Python sees only that one.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new ArgumentNullException(nameof(space));
                case Box or MultiBinary:
                    return space.Shape!.Value.dimensions.Aggregate(1L, (product, dimension) => product * dimension);
                case Discrete discrete:
                    return discrete.N;
                case MultiDiscrete multiDiscrete: {
                    using NDArray total = np.sum(multiDiscrete.Nvec);
                    return Convert.ToInt64(total.GetAtIndex(0), CultureInfo.InvariantCulture);
                }
                case Tuple tuple:
                    if (!tuple.IsNpFlattenable)
                        throw new ValueError(NotFlattenable(space));
                    return tuple.Spaces.Sum(Flatdim);
                case Dict dict:
                    if (!dict.IsNpFlattenable)
                        throw new ValueError(NotFlattenable(space));
                    return dict.Values.Sum(Flatdim);
                case Graph:
                    throw new ValueError("Cannot get flattened size as the Graph Space has a dynamic size.");
                case Text text:
                    return text.MaxLength;
                case OneOf oneOf:
                    return 1 + oneOf.Spaces.Max(Flatdim);
                default:
                    // Gymnasium's base function: a space that says it cannot be flattened gets the ValueError first.
                    if (!space.IsNpFlattenable)
                        throw new ValueError(NotFlattenable(space));
                    throw new NumSharp.NotImplementedError($"Unknown space: `{space}`");
            }
        }

        /// <summary>
        ///     Flattens an element of <paramref name="space"/> (Gymnasium's <c>flatten</c>): a box or <see cref="MultiBinary"/>
        ///     element raveled in its dtype, a <see cref="Discrete"/> or <see cref="MultiDiscrete"/> element one-hot encoded,
        ///     a <see cref="Text"/> as character indexes padded with the charset's size, a <see cref="OneOf"/> pair as its
        ///     index followed by the padded flattened value, and a composite's parts concatenated.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The element: an array, a 0-d array or integer, a tuple (<see langword="object"/>?[]), a dict, a string, a <see cref="GraphInstance"/>.</param>
        /// <returns>
        ///     A new array for a flattenable space; for a <see cref="Graph"/> a new <see cref="GraphInstance"/> of flattened
        ///     nodes and edges; for a <see cref="Sequence"/> a tuple of flattened items (or their stacked array); for a
        ///     composite that cannot become an array, a tuple or <see cref="Info"/> of flattened parts. The caller owns it.
        /// </returns>
        /// <exception cref="IndexError">Thrown for a discrete value outside the space, or a text longer than <c>max_length</c>, with NumPy's message.</exception>
        /// <exception cref="KeyError">Thrown for a dict element missing a key, or a character outside a text space's charset.</exception>
        /// <exception cref="ValueError">Thrown for a tuple element of another length than the space (Python's <c>zip(strict=True)</c>).</exception>
        /// <exception cref="TypeError">Thrown for a dict space given something that is not a mapping, or a text space given something that is not a string, with Python's messages.</exception>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        public static object? Flatten(Space space, object? x) {
            // An untyped env's space is a view (ObjectSpace): dispatch on the space it shows, as Python sees only that one.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new ArgumentNullException(nameof(space));
                case Box or MultiBinary: {
                    using NDArray converted = Box.AsArray(x, space.DType!);
                    return converted.flatten();
                }
                case Discrete discrete:
                    return FlattenDiscrete(discrete, x);
                case MultiDiscrete multiDiscrete:
                    return FlattenMultiDiscrete(multiDiscrete, x);
                case Tuple tuple: {
                    object?[] parts = ZipStrict(Items(x), tuple.Spaces);
                    object?[] flattened = parts.Select((part, i) => Flatten(tuple[i], part)).ToArray();
                    // Not one ?: expression: with an NDArray arm and an object?[] arm it would be typed NDArray, and NumSharp's
                    // implicit Array -> NDArray conversion would turn the tuple of parts into an object array.
                    if (tuple.IsNpFlattenable)
                        return ConcatenateOwned(flattened);
                    return flattened;
                }
                case Dict dict: {
                    IReadOnlyDictionary<string, object?> element = Dict.AsDict(x) ?? throw new TypeError($"'{PyBuiltins.TypeName(x)}' object is not subscriptable");
                    var flattened = new List<KeyValuePair<string, object?>>();
                    foreach (KeyValuePair<string, Space> pair in dict) {
                        if (!element.TryGetValue(pair.Key, out object? part))
                            throw new KeyError(PyBuiltins.Repr(pair.Key));
                        flattened.Add(new KeyValuePair<string, object?>(pair.Key, Flatten(pair.Value, part)));
                    }

                    if (dict.IsNpFlattenable)
                        return ConcatenateOwned(flattened.Select(p => p.Value).ToArray());
                    var info = new Info();
                    foreach (KeyValuePair<string, object?> pair in flattened)
                        info[pair.Key] = pair.Value;
                    return info;
                }
                case Graph graph:
                    return FlattenGraph(graph, (GraphInstance)x!);
                case Text text:
                    return FlattenText(text, x as string ?? throw new TypeError($"'{PyBuiltins.TypeName(x)}' object is not iterable"));
                case Sequence sequence:
                    return FlattenSequence(sequence, x);
                case OneOf oneOf:
                    return FlattenOneOf(oneOf, x);
                default:
                    throw new NumSharp.NotImplementedError($"Unknown space: `{space}`");
            }
        }

        /// <summary>
        ///     Reverses <see cref="Flatten"/> for <paramref name="space"/> (Gymnasium's <c>unflatten</c>); the space must be
        ///     the one the element was flattened with.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The flattened element: an array, or the tuple, dict or graph a non-flattenable space flattens to.</param>
        /// <returns>An element of <paramref name="space"/>, new and owned by the caller.</returns>
        /// <exception cref="ValueError">Thrown for a flattened discrete element with no one-hot entry, with Gymnasium's message.</exception>
        /// <exception cref="AssertionError">Thrown for a composite given the wrong flattened form (an array for a non-flattenable space or the reverse), with Gymnasium's messages.</exception>
        /// <exception cref="KeyError">Thrown for a non-flattenable dict space's flattened dict missing a key.</exception>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Not every array of the flattened space unflattens: a one-hot block with no 1 has no element, as in Gymnasium.
        /// </remarks>
        public static object? Unflatten(Space space, object? x) {
            // An untyped env's space is a view (ObjectSpace): dispatch on the space it shows, as Python sees only that one.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new ArgumentNullException(nameof(space));
                case Box or MultiBinary: {
                    using NDArray converted = Box.AsArray(x, space.DType!);
                    return converted.reshape(space.Shape!.Value).copy();
                }
                case Discrete discrete:
                    return UnflattenDiscrete(discrete, AsArray(x));
                case MultiDiscrete multiDiscrete:
                    return UnflattenMultiDiscrete(multiDiscrete, AsArray(x));
                case Tuple tuple:
                    if (tuple.IsNpFlattenable) {
                        if (x is not NDArray array)
                            throw new AssertionError($"{tuple} is numpy-flattenable. Thus, you should only unflatten numpy arrays for this space. Got a {PyBuiltins.TypeRepr(x)}");
                        return SplitByFlatdim(array, tuple.Spaces, (subspace, part) => Unflatten(subspace, part)).ToArray();
                    } else {
                        if (!IsTuple(x))
                            throw new AssertionError($"{tuple} is not numpy-flattenable. Thus, you should only unflatten tuples for this space. Got a {PyBuiltins.TypeRepr(x)}");
                        object?[] parts = ZipStrict(Items(x), tuple.Spaces);
                        return parts.Select((part, i) => Unflatten(tuple[i], part)).ToArray();
                    }
                case Dict dict: {
                    var result = new Info();
                    if (dict.IsNpFlattenable) {
                        if (x is not NDArray array)
                            throw new AssertionError("x must be a numpy array when unflattening a numpy-flattenable space");
                        string[] keys = dict.Keys.ToArray();
                        object?[] values = SplitByFlatdim(array, keys.Select(k => dict[k]).ToArray(), (subspace, part) => Unflatten(subspace, part)).ToArray();
                        for (int i = 0; i < keys.Length; i++)
                            result[keys[i]] = values[i];
                        return result;
                    }

                    IReadOnlyDictionary<string, object?> element = Dict.AsDict(x)
                                                                   ?? throw new AssertionError($"{dict} is not numpy-flattenable. Thus, you should only unflatten dictionary for this space. Got a {PyBuiltins.TypeRepr(x)}");
                    foreach (KeyValuePair<string, Space> pair in dict) {
                        if (!element.TryGetValue(pair.Key, out object? part))
                            throw new KeyError(PyBuiltins.Repr(pair.Key));
                        result[pair.Key] = Unflatten(pair.Value, part);
                    }

                    return result;
                }
                case Graph graph:
                    return UnflattenGraph(graph, (GraphInstance)x!);
                case Text text:
                    return UnflattenText(text, AsArray(x));
                case Sequence sequence:
                    return UnflattenSequence(sequence, x);
                case OneOf oneOf:
                    return UnflattenOneOf(oneOf, AsArray(x));
                default:
                    throw new NumSharp.NotImplementedError($"Unknown space: `{space}`");
            }
        }

        /// <summary>
        ///     The space <see cref="Flatten"/> maps <paramref name="space"/>'s elements into (Gymnasium's <c>flatten_space</c>):
        ///     a flat <see cref="Box"/> for every space that flattens to an array, and for the others the same structure
        ///     (<see cref="Tuple"/>, <see cref="Dict"/>, <see cref="Graph"/>, <see cref="Sequence"/>) over flattened subspaces.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <returns>A new space, owned by the caller, unseeded.</returns>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     Sampling the flattened space is not sampling the original: a flattened <see cref="Discrete"/> is a 0/1 box
        ///     whose samples are rarely one-hot, as Gymnasium warns.
        /// </remarks>
        public static Space FlattenSpace(Space space) {
            // An untyped env's space is a view (ObjectSpace): dispatch on the space it shows, as Python sees only that one.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new ArgumentNullException(nameof(space));
                case Box box: {
                    using NDArray low = box.Low.flatten();
                    using NDArray high = box.High.flatten();
                    return new Box(low, high, dtype: box.DType);
                }
                case Discrete or MultiBinary or MultiDiscrete:
                    // Gymnasium's Box(low=0, high=1, ...) with int literals: a 0/1 box of the space's integer dtype.
                    return new Box(0, 1, new Shape(Flatdim(space)), space.DType);
                case Tuple tuple:
                    if (tuple.IsNpFlattenable)
                        return ConcatenatedBox(tuple.Spaces);
                    return new Tuple(tuple.Spaces.Select(FlattenSpace).ToArray());
                case Dict dict:
                    if (dict.IsNpFlattenable)
                        return ConcatenatedBox(dict.Values.ToArray());
                    // Through the mapping constructor with the space's sort_keys, as Gymnasium passes a dict.
                    var flattened = new Dictionary<string, Space>();
                    foreach (KeyValuePair<string, Space> pair in dict)
                        flattened.Add(pair.Key, FlattenSpace(pair.Value));
                    return new Dict(flattened, sortKeys: dict.SortKeys);
                case Graph graph:
                    return new Graph(FlattenSpace(graph.NodeSpace), graph.EdgeSpace is null ? null : FlattenSpace(graph.EdgeSpace));
                case Text text:
                    return new Box(0, text.CharacterSet.Count, new Shape(text.MaxLength), np.int32);
                case Sequence sequence:
                    return new Sequence(FlattenSpace(sequence.FeatureSpace), stack: sequence.Stack);
                case OneOf oneOf:
                    return FlattenSpaceOneOf(oneOf);
                default:
                    throw new NumSharp.NotImplementedError($"Unknown space: `{space}`");
            }
        }

        /// <summary>
        ///     Whether two spaces share a dtype and shape and whatever else decides an element's layout (Gymnasium's
        ///     <c>is_space_dtype_shape_equiv</c>): vector envs require it of their sub-envs' spaces. Simple spaces compare
        ///     their exact type, shape and dtype; a <see cref="Text"/> its length and charset; composites recurse.
        /// </summary>
        /// <param name="space1">The first space; its type decides the comparison, as Gymnasium dispatches on it.</param>
        /// <param name="space2">The second space.</param>
        /// <returns><see langword="true"/> when equivalent.</returns>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom first space, with Gymnasium's message.</exception>
        /// <exception cref="TypeError">Thrown when either argument is <see langword="null"/> (Gymnasium's bare <c>TypeError()</c>).</exception>
        public static bool IsSpaceDtypeShapeEquiv(Space space1, Space space2) {
            // An untyped env's space is a view (ObjectSpace): dispatch on the space it shows, as Python sees only that one.
            space1 = ObjectSpace.Unview(space1)!;
            space2 = ObjectSpace.Unview(space2)!;
            switch (space1) {
                case null:
                    throw new TypeError("");
                case Box or Discrete or MultiDiscrete or MultiBinary:
                    // The exact type: singledispatch only looked at the first argument.
                    return space2 is not null && space1.GetType() == space2.GetType() && SameShape(space1.Shape, space2.Shape) && space1.DType == space2.DType;
                case Text text:
                    return space2 is Text other && text.MaxLength == other.MaxLength && text.CharacterSet.SetEquals(other.CharacterSet);
                case Dict dict:
                    // dict_keys compare as sets: the order does not matter.
                    return space2 is Dict otherDict && dict.Count == otherDict.Count && dict.Keys.All(otherDict.ContainsKey) &&
                           dict.Keys.All(key => IsSpaceDtypeShapeEquiv(dict[key], otherDict[key]));
                case Tuple tuple:
                    return space2 is Tuple otherTuple && tuple.Count == otherTuple.Count &&
                           Enumerable.Range(0, tuple.Count).All(i => IsSpaceDtypeShapeEquiv(tuple[i], otherTuple[i]));
                case Graph graph:
                    return space2 is Graph otherGraph && IsSpaceDtypeShapeEquiv(graph.NodeSpace, otherGraph.NodeSpace) &&
                           (graph.EdgeSpace is null && otherGraph.EdgeSpace is null ||
                            graph.EdgeSpace is not null && otherGraph.EdgeSpace is not null && IsSpaceDtypeShapeEquiv(graph.EdgeSpace, otherGraph.EdgeSpace));
                case OneOf oneOf:
                    return space2 is OneOf otherOneOf && oneOf.Count == otherOneOf.Count &&
                           Enumerable.Range(0, oneOf.Count).All(i => IsSpaceDtypeShapeEquiv(oneOf[i], otherOneOf[i]));
                case Sequence sequence:
                    return space2 is Sequence otherSequence && sequence.Stack == otherSequence.Stack &&
                           IsSpaceDtypeShapeEquiv(sequence.FeatureSpace, otherSequence.FeatureSpace);
                default:
                    if (space2 is null)
                        throw new TypeError("");
                    throw new NumSharp.NotImplementedError("`check_dtype_shape_equivalence` doesn't support Generic Gymnasium Spaces, ");
            }
        }

        /// <summary>
        ///     The dtype of the box a space flattens to (Gymnasium's <c>_flat_dtype</c>), computed without building it:
        ///     <see cref="Flatten"/> and <see cref="Unflatten"/> need it for every step of a flattening wrapper.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <returns>The dtype.</returns>
        /// <exception cref="ValueError">Thrown for a space with no dtype (a graph, a sequence), with Gymnasium's message.</exception>
        internal static DType FlatDType(Space space) {
            switch (space) {
                case Text:
                    // A text flattens to character indexes, not to its own (string) dtype.
                    return np.int32;
                case Tuple tuple:
                    return np.result_type(tuple.Spaces.Select(FlatDType).ToArray());
                case Dict dict:
                    return np.result_type(dict.Values.Select(FlatDType).ToArray());
                case OneOf oneOf: {
                    DType dtype = np.result_type(oneOf.Spaces.Select(FlatDType).ToArray());
                    // The index is stored in the same dtype, so it must fit: v1.4.0 promotes a bool or a too-narrow integer.
                    long maxIndex = oneOf.Count - 1;
                    if (dtype.kind == 'b' && maxIndex > 1 || dtype.kind is 'i' or 'u' && new BigInteger(maxIndex) > MaxOf(dtype))
                        dtype = np.result_type(dtype, np.min_scalar_type(maxIndex));
                    return dtype;
                }
                default:
                    return space.DType ?? throw new ValueError($"{space} does not have a dtype, therefore, it cannot be flattened to a numpy array, probably because it contains a `Graph` or `Sequence` subspace.");
            }
        }

        /// <summary>The largest value of an integer dtype (NumPy's <c>np.iinfo(dtype).max</c>).</summary>
        /// <param name="dtype">The integer dtype.</param>
        /// <returns>The maximum.</returns>
        private static BigInteger MaxOf(DType dtype) => BigInteger.Parse(np.iinfo(dtype).max.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);

        /// <summary>Gymnasium's <c>_flatten_discrete</c>: a one-hot vector of the space's dtype.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The element: an integer or a 0-d integer array.</param>
        /// <returns>A new array of <c>n</c> entries.</returns>
        /// <exception cref="IndexError">Thrown when <c>x - start</c> is outside the vector (negative indexes count from the end, as in NumPy), or not an integer.</exception>
        private static NDArray FlattenDiscrete(Discrete space, object? x) {
            if (x is bool || !Discrete.TryToInt64(x, out long value))
                throw new IndexError("only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices");
            NDArray onehot = np.zeros(new Shape(space.N), space.DType);
            try {
                // onehot[x - start] = 1: np.put has NumPy's index wrap and out-of-bounds message. Since NumSharp 0.81.0
                // np.put takes arrays only (NumPy's signature), so the index and the 1 are 0-d arrays this method owns.
                using NDArray index = np.array(value - space.Start);
                using NDArray one = np.array(1L);
                np.put(onehot, index, one);
                return onehot;
            } catch {
                onehot.Dispose();
                throw;
            }
        }

        /// <summary>Gymnasium's <c>_flatten_multidiscrete</c>: one one-hot block per entry, concatenated in C order.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The element, an integer array of the space's shape.</param>
        /// <returns>A new array of <c>sum(nvec)</c> entries.</returns>
        /// <exception cref="IndexError">Thrown when an entry is outside its block's range, with NumPy's message.</exception>
        [MethodImpl(MethodImplOptions.NoInlining)]
        private static NDArray FlattenMultiDiscrete(MultiDiscrete space, object? x) {
            long[] nvec = Longs(space.Nvec);
            long[] start = Longs(space.Start);
            long[] values;
            using (NDArray element = Box.AsArray(x, np.int64))
                values = Longs(element);
            // offsets = [0, cumsum(nvec)...]: Gymnasium's int32 offsets array, summed in int64 here.
            var offsets = new long[nvec.Length + 1];
            for (int i = 0; i < nvec.Length; i++)
                offsets[i + 1] = offsets[i] + nvec[i];
            NDArray onehot = np.zeros(new Shape(offsets[nvec.Length]), space.DType);
            try {
                // onehot[offsets[:-1] + (x - start).flatten()] = 1: the fancy index broadcasts, wraps and bounds-checks.
                if (values.Length != nvec.Length && values.Length != 1)
                    throw new ValueError($"operands could not be broadcast together with shapes ({nvec.Length},) ({values.Length},) ");
                // np.put takes arrays only since NumSharp 0.81.0: one 0-d 1 for every block, and a 0-d index per block
                using NDArray one = np.array(1L);
                for (int i = 0; i < nvec.Length; i++) {
                    using NDArray index = np.array(offsets[i] + values[values.Length == 1 ? 0 : i] - start[i]);
                    np.put(onehot, index, one);
                }

                return onehot;
            } catch {
                onehot.Dispose();
                throw;
            }
        }

        /// <summary>Gymnasium's <c>_flatten_text</c>: each code point's charset index, padded with the charset's size.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The text.</param>
        /// <returns>A new int32 array of <c>max_length</c> entries.</returns>
        /// <exception cref="IndexError">Thrown for a text longer than <c>max_length</c>.</exception>
        /// <exception cref="KeyError">Thrown for a character outside the charset.</exception>
        private static NDArray FlattenText(Text space, string x) {
            NDArray indexes = np.full(new Shape(space.MaxLength), space.CharacterSet.Count, np.int32);
            try {
                int position = 0;
                foreach (string character in Text.CodePoints(x)) {
                    if (position >= space.MaxLength)
                        throw new IndexError($"index {position} is out of bounds for axis 0 with size {space.MaxLength}");
                    indexes.SetAtIndex(space.CharacterIndex(character), position);
                    position++;
                }

                return indexes;
            } catch {
                indexes.Dispose();
                throw;
            }
        }

        /// <summary>Gymnasium's <c>_flatten_graph</c>: the nodes and edges flattened one by one and stacked again.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The graph.</param>
        /// <returns>A new graph; its links are a copy of <paramref name="x"/>'s (Gymnasium shares them).</returns>
        /// <exception cref="AssertionError">Thrown for a graph with edges in a space without an edge space.</exception>
        private static GraphInstance FlattenGraph(Graph space, GraphInstance x) {
            object? nodes = FlattenBatch(FlattenSpace(space.NodeSpace), space.NodeSpace, space.BatchNodeSpace, x.Nodes);
            object? edges = null;
            if (x.Edges is not null) {
                if (space.EdgeSpace is null)
                    throw new AssertionError();
                edges = FlattenBatch(FlattenSpace(space.EdgeSpace), space.EdgeSpace, space.BatchEdgeSpace!, x.Edges);
            }

            // GraphInstance owns its arrays, so the links are copied rather than shared with the input graph.
            return new GraphInstance(nodes, edges, x.EdgeLinks?.copy());
        }

        /// <summary>Flattens each item of a batch and stacks the results into the flattened space's empty batch.</summary>
        /// <param name="flatSpace">The flattened item space; disposed here.</param>
        /// <param name="itemSpace">The item space.</param>
        /// <param name="batchSpace">The batched item space, to iterate the batch.</param>
        /// <param name="batch">The batch.</param>
        /// <returns>The stacked flattened items.</returns>
        private static object? FlattenBatch(Space flatSpace, Space itemSpace, Space batchSpace, object? batch) {
            using (flatSpace) {
                object?[] flattened = VectorUtils.Iterate(batchSpace, batch).Select(item => Flatten(itemSpace, item)).ToArray();
                object? @out = VectorUtils.CreateEmptyArray(flatSpace, flattened.Length);
                try {
                    return VectorUtils.Concatenate(flatSpace, flattened, @out);
                } finally {
                    // Concatenate copied array items into @out; a tuple-batched space holds the items themselves.
                    if (@out is NDArray)
                        foreach (object? item in flattened)
                            (item as IDisposable)?.Dispose();
                }
            }
        }

        /// <summary>Gymnasium's <c>_flatten_sequence</c>: each item flattened, stacked again for a stacked space.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The sequence: a tuple, or the stacked batch.</param>
        /// <returns>A tuple of flattened items, or their stacked array.</returns>
        private static object? FlattenSequence(Sequence space, object? x) {
            if (space.Stack)
                return FlattenBatch(FlattenSpace(space.FeatureSpace), space.FeatureSpace, space.StackedFeatureSpace!, x);
            return Items(x).Select(item => Flatten(space.FeatureSpace, item)).ToArray();
        }

        /// <summary>
        ///     Gymnasium's <c>_flatten_oneof</c>: the subspace index, then the subspace's flattened value padded with its own
        ///     first entry to the widest subspace, all in the shared flat dtype.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The <c>(index, value)</c> pair.</param>
        /// <returns>A new array of <c>flatdim(space)</c> entries.</returns>
        /// <exception cref="ValueError">Thrown for a value that does not unpack into two items (Python's unpacking error).</exception>
        /// <exception cref="TypeError">Thrown for an index that is not an integer, with Python's tuple-indexing message.</exception>
        private static NDArray FlattenOneOf(OneOf space, object? x) {
            object?[] pair = Items(x);
            if (pair.Length != 2)
                throw new ValueError(pair.Length > 2 ? "too many values to unpack (expected 2)" : $"not enough values to unpack (expected 2, got {pair.Length})");
            if (pair[0] is bool || !Discrete.TryToInt64(pair[0], out long index))
                throw new TypeError("tuple indices must be integers or slices, not " + PyBuiltins.TypeName(pair[0]));
            Space subspace = space[(int)index];
            var parts = new List<NDArray>();
            try {
                var flat = (NDArray)Flatten(subspace, pair[1])!;
                parts.Add(flat);
                long widest = Flatdim(space) - 1;
                if (flat.size < widest) {
                    // Padded with the sample's own first entry in its own dtype, as Gymnasium pads.
                    using NDArray first = flat["0"];
                    parts.Add(np.full(new Shape(widest - flat.size), first.GetAtIndex(0), flat.dtype));
                }

                DType dtype = FlatDType(space);
                using NDArray indexScalar = NDArray.Scalar(index);
                using NDArray indexVector = indexScalar.reshape(new Shape(1L));
                using NDArray indexArray = indexVector.astype(dtype);
                // The parts are this method's own arrays (disposed in finally), borrowed for the concatenation.
#pragma warning disable NDW012
                using NDArray padded = parts.Count == 1 ? parts[0].copy() : np.concatenate(parts.ToArray());
#pragma warning restore NDW012
                using NDArray cast = padded.astype(dtype);
                return np.concatenate(new[] { indexArray, cast });
            } finally {
                foreach (NDArray part in parts)
                    part.Dispose();
            }
        }

        /// <summary>Gymnasium's <c>_unflatten_discrete</c>: <c>start</c> plus the index of the first non-zero entry.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The one-hot vector; disposed here when this class created it.</param>
        /// <returns>A new 0-d array of the space's dtype.</returns>
        /// <exception cref="ValueError">Thrown when no entry is non-zero, with Gymnasium's message.</exception>
        private static NDArray UnflattenDiscrete(Discrete space, Borrowed x) {
            using (x) {
                var nonzero = np.nonzero(x.Array);
                try {
                    if (nonzero[0].size == 0)
                        throw new ValueError($"{PyBuiltins.Str(x.Array)} is not a valid one-hot encoded vector and can not be unflattened to space {space}. Not all valid samples in a flattened space can be unflattened.");
                    long first = Convert.ToInt64(nonzero[0].GetAtIndex(0), CultureInfo.InvariantCulture);
                    using NDArray start = NDArray.Scalar(space.Start).astype(space.DType);
                    using NDArray offset = NDArray.Scalar(first).astype(space.DType);
                    return start + offset;
                } finally {
                    foreach (NDArray index in nonzero)
                        index.Dispose();
                }
            }
        }

        /// <summary>Gymnasium's <c>_unflatten_multidiscrete</c>: each non-zero entry's offset into its block, plus <c>start</c>.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The concatenated one-hot blocks; disposed here when this class created it.</param>
        /// <returns>A new array of the space's shape and dtype.</returns>
        /// <exception cref="ValueError">Thrown when no entry is non-zero (Gymnasium's message), or the non-zero count is not the block count (NumPy's broadcast error).</exception>
        private static NDArray UnflattenMultiDiscrete(MultiDiscrete space, Borrowed x) {
            using (x) {
                var nonzero = np.nonzero(x.Array);
                try {
                    NDArray indices = nonzero[0];
                    if (indices.size == 0)
                        throw new ValueError($"{PyBuiltins.Str(x.Array)} is not a concatenation of one-hot encoded vectors and can not be unflattened to space {space}. Not all valid samples in a flattened space can be unflattened.");
                    // offsets = zeros(nvec.size + 1, dtype); offsets[1:] = cumsum(nvec.flatten()), in the space's dtype.
                    using NDArray flatNvec = space.Nvec.flatten();
                    using NDArray sums = np.cumsum(flatNvec);
                    using NDArray zero = np.zeros(new Shape(1L), space.DType);
                    using NDArray castSums = sums.astype(space.DType);
                    using NDArray offsets = np.concatenate(new[] { zero, castSums });
                    using NDArray leading = offsets[$":{offsets.size - 1}"];
                    using NDArray positions = indices - leading;
                    using NDArray cast = positions.astype(space.DType);
                    using NDArray shaped = cast.reshape(space.Shape);
                    return shaped + space.Start;
                } finally {
                    foreach (NDArray index in nonzero)
                        index.Dispose();
                }
            }
        }

        /// <summary>Gymnasium's <c>_unflatten_text</c>: the characters of the indexes below the charset's size.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The indexes; disposed here when this class created them.</param>
        /// <returns>The text.</returns>
        /// <exception cref="IndexError">Thrown for a negative index past the charset's start (Python's tuple indexing).</exception>
        /// <exception cref="TypeError">
        ///     Thrown when an index below the charset's size is not an integer (a float array, as a text flattened inside a
        ///     tuple with a float box becomes): Python's <c>tuple indices must be integers or slices, not numpy.float64</c>.
        /// </exception>
        private static string UnflattenText(Text space, Borrowed x) {
            using (x) {
                int size = space.CharacterSet.Count;
                IReadOnlyList<string> characters = space.CharacterList;
                bool integral = x.Array.dtype.kind is 'i' or 'u';
                var text = new System.Text.StringBuilder();
                for (long i = 0; i < x.Array.size; i++) {
                    // val < len(character_set) filters first; only the values that pass index the character tuple.
                    double magnitude = Convert.ToDouble(x.Array.GetAtIndex(i), CultureInfo.InvariantCulture);
                    if (!(magnitude < size))
                        continue;
                    if (!integral)
                        throw new TypeError($"tuple indices must be integers or slices, not numpy.{Discrete.DTypeName(x.Array.dtype)}");
                    long value = Convert.ToInt64(x.Array.GetAtIndex(i), CultureInfo.InvariantCulture);
                    // Python indexes the character tuple, so a negative index counts from its end.
                    long position = value < 0 ? value + characters.Count : value;
                    if (position < 0 || position >= characters.Count)
                        throw new IndexError("tuple index out of range");
                    text.Append(characters[(int)position]);
                }

                return text.ToString();
            }
        }

        /// <summary>Gymnasium's <c>_unflatten_graph</c>: the flattened nodes and edges unflattened one by one and stacked again.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The flattened graph.</param>
        /// <returns>A new graph; its links are a copy of <paramref name="x"/>'s.</returns>
        /// <exception cref="AssertionError">Thrown for a graph with edges in a space without an edge space.</exception>
        private static GraphInstance UnflattenGraph(Graph space, GraphInstance x) {
            object? nodes = UnflattenBatch(space.NodeSpace, x.Nodes);
            object? edges = null;
            if (x.Edges is not null) {
                if (space.EdgeSpace is null)
                    throw new AssertionError();
                edges = UnflattenBatch(space.EdgeSpace, x.Edges);
            }

            return new GraphInstance(nodes, edges, x.EdgeLinks?.copy());
        }

        /// <summary>Unflattens each item of a flattened batch and stacks the results into the item space's empty batch.</summary>
        /// <param name="itemSpace">The item space.</param>
        /// <param name="batch">The flattened batch.</param>
        /// <returns>The stacked items.</returns>
        private static object? UnflattenBatch(Space itemSpace, object? batch) {
            using Space flatSpace = FlattenSpace(itemSpace);
            object?[] items = VectorUtils.Iterate(flatSpace, batch).Select(item => Unflatten(itemSpace, item)).ToArray();
            object? @out = VectorUtils.CreateEmptyArray(itemSpace, items.Length);
            try {
                return VectorUtils.Concatenate(itemSpace, items, @out);
            } finally {
                if (@out is NDArray)
                    foreach (object? item in items)
                        (item as IDisposable)?.Dispose();
            }
        }

        /// <summary>Gymnasium's <c>_unflatten_sequence</c>: each flattened item unflattened, stacked again for a stacked space.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">A tuple of flattened items, or their stacked array.</param>
        /// <returns>The sequence.</returns>
        private static object? UnflattenSequence(Sequence space, object? x) {
            if (space.Stack)
                return UnflattenBatch(space.FeatureSpace, x);
            return Items(x).Select(item => Unflatten(space.FeatureSpace, item)).ToArray();
        }

        /// <summary>
        ///     Gymnasium's <c>_unflatten_oneof</c>: the index (<c>np.int64(x[0])</c>, truncating a float), then the subspace's
        ///     own entries cast back to its flat dtype and unflattened.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The flattened pair; disposed here when this class created it.</param>
        /// <returns>The <c>(index, value)</c> pair, its index a new 0-d int64 array.</returns>
        /// <exception cref="ValueError">Thrown for a NaN index (NumPy's conversion error).</exception>
        private static object?[] UnflattenOneOf(OneOf space, Borrowed x) {
            using (x) {
                long index = TruncateToInt64(x.Array.GetAtIndex(0));
                Space subspace = space[(int)index];
                long size = Flatdim(subspace);
                using NDArray trimmed = x.Array[$"1:{1 + size}"];
                using NDArray cast = trimmed.astype(FlatDType(subspace));
                return new object?[] { NDArray.Scalar(index), Unflatten(subspace, cast) };
            }
        }

        /// <summary>
        ///     Gymnasium's <c>_flatten_space_oneof</c>: a box whose first entry is the index range and whose others span the
        ///     lowest and highest bound of every flattened subspace, in the shared flat dtype.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <returns>A new box.</returns>
        private static Box FlattenSpaceOneOf(OneOf space) {
            int count = space.Count;
            long width = space.Spaces.Max(Flatdim) + 1;
            var lowParts = new List<NDArray>();
            var highParts = new List<NDArray>();
            try {
                foreach (Space subspace in space.Spaces) {
                    using var flat = (Box)FlattenSpace(subspace);
                    lowParts.Add(np.min(flat.Low));
                    highParts.Add(np.max(flat.High));
                }

                // np.array([np.min(...) ...]) of NumPy scalars: their promoted dtype, as np.stack gives it. The parts are
                // this method's own arrays (disposed in finally), borrowed for the stack.
#pragma warning disable NDW012
                using NDArray lows = np.stack(lowParts.ToArray());
                using NDArray highs = np.stack(highParts.ToArray());
#pragma warning restore NDW012
                using NDArray overallLow = np.min(lows);
                using NDArray overallHigh = np.max(highs);
                using NDArray zero = np.array(new long[] { 0 });
                using NDArray last = np.array(new long[] { count - 1 });
                using NDArray lowFill = np.full(new Shape(width - 1), overallLow.GetAtIndex(0), overallLow.dtype);
                using NDArray highFill = np.full(new Shape(width - 1), overallHigh.GetAtIndex(0), overallHigh.dtype);
                using NDArray low = np.concatenate(new[] { zero, lowFill });
                using NDArray high = np.concatenate(new[] { last, highFill });
                return new Box(low, high, new Shape(width), FlatDType(space));
            } finally {
                foreach (NDArray part in lowParts.Concat(highParts))
                    part.Dispose();
            }
        }

        /// <summary>
        ///     Gymnasium's flattenable <c>Tuple</c>/<c>Dict</c> space: the subspaces' flattened bounds concatenated, in their
        ///     promoted dtype.
        /// </summary>
        /// <param name="spaces">The subspaces, in order.</param>
        /// <returns>A new box.</returns>
        private static Box ConcatenatedBox(IReadOnlyList<Space> spaces) {
            var flattened = spaces.Select(s => (Box)FlattenSpace(s)).ToList();
            try {
                // The bounds are the flattened boxes' own arrays, borrowed for the concatenation.
#pragma warning disable NDW012
                using NDArray low = np.concatenate(flattened.Select(b => b.Low).ToArray());
                using NDArray high = np.concatenate(flattened.Select(b => b.High).ToArray());
#pragma warning restore NDW012
                return new Box(low, high, dtype: np.result_type(flattened.Select(b => b.DType).ToArray()));
            } finally {
                foreach (Box box in flattened)
                    box.Dispose();
            }
        }

        /// <summary>Concatenates flattened parts (all arrays) into one new array and disposes the parts.</summary>
        /// <param name="parts">The parts; disposed here.</param>
        /// <returns>The concatenation, in the parts' promoted dtype.</returns>
        private static NDArray ConcatenateOwned(object?[] parts) {
            NDArray[] arrays = parts.Select(p => (NDArray)p!).ToArray();
            try {
                return np.concatenate(arrays);
            } finally {
                foreach (NDArray array in arrays)
                    array.Dispose();
            }
        }

        /// <summary>
        ///     Splits a flat array into one piece per subspace (Gymnasium's <c>np.split(x, np.cumsum(dims[:-1]))</c>) and maps
        ///     each piece.
        /// </summary>
        /// <param name="array">The flat array.</param>
        /// <param name="spaces">The subspaces, in order.</param>
        /// <param name="map">Unflattens one piece.</param>
        /// <returns>The mapped pieces.</returns>
        private static List<object?> SplitByFlatdim(NDArray array, IReadOnlyList<Space> spaces, Func<Space, NDArray, object?> map) {
            var results = new List<object?>(spaces.Count);
            long offset = 0;
            for (int i = 0; i < spaces.Count; i++) {
                long size = Flatdim(spaces[i]);
                // The last piece runs to the end, as np.split's last section does.
                long end = i == spaces.Count - 1 ? Math.Max(offset, array.size) : Math.Min(offset + size, array.size);
                using NDArray piece = array[$"{Math.Min(offset, array.size)}:{end}"];
                results.Add(map(spaces[i], piece));
                offset += size;
            }

            return results;
        }

        /// <summary>A flattened value as an array, converting a list; the wrapper disposes only what it created.</summary>
        /// <param name="x">The value.</param>
        /// <returns>The array, borrowed or owned.</returns>
        private static Borrowed AsArray(object? x) => x is NDArray array ? new Borrowed(array, owned: false) : new Borrowed(Box.AsArray(x, np.float64), owned: true);

        /// <summary>An array that is disposed only when this class created it.</summary>
        private readonly struct Borrowed : IDisposable {
            /// <summary>Wraps an array.</summary>
            /// <param name="array">The array.</param>
            /// <param name="owned">Whether disposing the wrapper disposes the array.</param>
            public Borrowed(NDArray array, bool owned) {
                Array = array;
                Owned = owned;
            }

            /// <summary>The array.</summary>
            public NDArray Array { get; }

            /// <summary>Whether the wrapper owns the array.</summary>
            private bool Owned { get; }

            /// <summary>Disposes the array when owned.</summary>
            public void Dispose() {
                if (Owned)
                    Array.Dispose();
            }
        }

        /// <summary>The items of a tuple-like value: an <see langword="object"/>?[], an <see cref="ITuple"/>, a list, or an array's first axis (views).</summary>
        /// <param name="x">The value.</param>
        /// <returns>The items.</returns>
        /// <exception cref="TypeError">Thrown for a value that cannot be iterated, with Python's message.</exception>
        private static object?[] Items(object? x) => x switch {
            object?[] array => array,
            ITuple tuple => Tuple.TupleItems(tuple),
            NDArray { ndim: > 0 } array => Enumerable.Range(0, (int)array.shape[0]).Select(i => (object?)array[i.ToString(CultureInfo.InvariantCulture)]).ToArray(),
            string text => Text.CodePoints(text).Select(c => (object?)c).ToArray(),
            IList list => list.Cast<object?>().ToArray(),
            _ => throw new TypeError($"'{PyBuiltins.TypeName(x)}' object is not iterable"),
        };

        /// <summary>Whether a value is a Python tuple's counterpart (an <see cref="ITuple"/> or an exact <see langword="object"/>?[]).</summary>
        /// <param name="x">The value.</param>
        /// <returns><see langword="true"/> for a tuple.</returns>
        private static bool IsTuple(object? x) => x is ITuple || x is object?[] array && PyBuiltins.IsTupleArray(array);

        /// <summary>Python's <c>zip(x, spaces, strict=True)</c>: the items, checked to be one per subspace.</summary>
        /// <param name="items">The element's items.</param>
        /// <param name="spaces">The subspaces.</param>
        /// <returns>The items.</returns>
        /// <exception cref="ValueError">Thrown when the counts differ, with Python's message.</exception>
        private static object?[] ZipStrict(object?[] items, IReadOnlyList<Space> spaces) {
            if (items.Length < spaces.Count)
                throw new ValueError("zip() argument 2 is longer than argument 1");
            if (items.Length > spaces.Count)
                throw new ValueError("zip() argument 2 is shorter than argument 1");
            return items;
        }

        /// <summary>Every element of an integer array as a <see langword="long"/>, in C order.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The values.</returns>
        private static long[] Longs(NDArray array) {
            var values = new long[array.size];
            using NDArray flat = array.flatten();
            for (long i = 0; i < values.Length; i++)
                values[i] = Convert.ToInt64(flat.GetAtIndex(i), CultureInfo.InvariantCulture);
            return values;
        }

        /// <summary>NumPy's <c>np.int64(value)</c> of a scalar: an integer as is, a float truncated toward zero.</summary>
        /// <param name="value">The boxed element.</param>
        /// <returns>The integer.</returns>
        /// <exception cref="ValueError">Thrown for NaN: <c>cannot convert float NaN to integer</c>.</exception>
        /// <exception cref="OverflowError">Thrown for an infinity, as NumPy's <c>OverflowError</c>.</exception>
        private static long TruncateToInt64(object? value) {
            switch (value) {
                case double or float or Half: {
                    double d = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                    if (double.IsNaN(d))
                        throw new ValueError("cannot convert float NaN to integer");
                    if (double.IsInfinity(d))
                        throw new OverflowError("cannot convert float infinity to integer");
                    return (long)Math.Truncate(d);
                }
                case bool flag:
                    return flag ? 1 : 0;
                default:
                    return Convert.ToInt64(value, CultureInfo.InvariantCulture);
            }
        }

        /// <summary>Whether two optional shapes are equal dimension for dimension (both absent counts as equal).</summary>
        /// <param name="left">The first shape.</param>
        /// <param name="right">The second shape.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        private static bool SameShape(Shape? left, Shape? right) =>
            left is null ? right is null : right is not null && left.Value.dimensions.SequenceEqual(right.Value.dimensions);
    }
}
