#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Vector {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.vector.utils</c> space functions: batching a space for <c>n</c> envs, iterating over and
    ///     concatenating batched samples, and allocating an empty batch. Vector envs use them for their batched spaces and
    ///     observations; <see cref="Graph"/> and a stacked <see cref="Sequence"/> sample through them.
    /// </summary>
    /// <remarks>
    ///     Each function dispatches on the space's type, as Gymnasium's <c>singledispatch</c> does: a subclass of a known space
    ///     takes its base's path, and an unknown custom space takes Gymnasium's fallback (a tuple of copies for
    ///     <see cref="BatchSpace"/>, an error for <see cref="Iterate"/>). Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/utils/space_utils.py">gymnasium/vector/utils/space_utils.py</see>.
    /// </remarks>
    public static partial class VectorUtils {
        /// <summary>
        ///     The space of <paramref name="n"/> stacked elements of <paramref name="space"/> (Gymnasium's <c>batch_space</c>):
        ///     a <see cref="Box"/> gains a leading axis, a <see cref="Discrete"/> becomes a <see cref="MultiDiscrete"/>, a
        ///     <see cref="MultiDiscrete"/> or <see cref="MultiBinary"/> a <see cref="Box"/>, composites batch their subspaces, and
        ///     any other space becomes a <see cref="Tuple"/> of <paramref name="n"/> independently seeded copies.
        /// </summary>
        /// <param name="space">The space of one env's elements.</param>
        /// <param name="n">The number of envs.</param>
        /// <returns>A new space, owned by the caller; it starts from a copy of <paramref name="space"/>'s generator.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="space"/> is <see langword="null"/>, with Gymnasium's message.</exception>
        /// <remarks>
        ///     Reading the space's generator creates it from entropy when the space was never seeded, as Gymnasium's
        ///     <c>deepcopy(space.np_random)</c> does. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/utils/space_utils.py#L50">gymnasium/vector/utils/space_utils.py (batch_space)</see>.
        /// </remarks>
        public static Space BatchSpace(Space space, int n = 1) {
            // An untyped env's spaces are ObjectSpace views, Gym.NET's own wrapper: the space inside is the one Gymnasium's function gets.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new TypeError("The space provided to `batch_space` is not a gymnasium Space instance, type: <class 'NoneType'>, None");
                case Box box: {
                    long[] repeats = new[] { (long)n }.Concat(Enumerable.Repeat(1L, box.Low.ndim)).ToArray();
                    using NDArray low = np.tile(box.Low, repeats);
                    using NDArray high = np.tile(box.High, repeats);
                    return new Box(low, high, dtype: box.DType, seed: Space.CopyGenerator(box.NpRandom));
                }
                case Discrete discrete: {
                    using NDArray nvec = np.full(new Shape((long)n), discrete.N, discrete.DType);
                    using NDArray start = np.full(new Shape((long)n), discrete.Start, discrete.DType);
                    return new MultiDiscrete(nvec, discrete.DType, Space.CopyGenerator(discrete.NpRandom), start);
                }
                case MultiDiscrete multiDiscrete: {
                    long[] repeats = new[] { (long)n }.Concat(Enumerable.Repeat(1L, multiDiscrete.Nvec.ndim)).ToArray();
                    using NDArray low = np.tile(multiDiscrete.Start, repeats);
                    using NDArray counts = np.tile(multiDiscrete.Nvec, repeats);
                    using NDArray sum = low + counts;
                    using NDArray one = NDArray.Scalar(1L).astype(multiDiscrete.DType);
                    using NDArray high = sum - one;
                    return new Box(low, high, dtype: multiDiscrete.DType, seed: Space.CopyGenerator(multiDiscrete.NpRandom));
                }
                case MultiBinary multiBinary:
                    return new Box(0, 1, new Shape(new[] { (long)n }.Concat(multiBinary.Shape.dimensions).ToArray()), multiBinary.DType, Space.CopyGenerator(multiBinary.NpRandom));
                case Tuple tuple:
                    return new Tuple(tuple.Spaces.Select(s => BatchSpace(s, n)).ToArray(), Space.CopyGenerator(tuple.NpRandom));
                case Dict dict: {
                    var batched = new Dictionary<string, Space>();
                    foreach (KeyValuePair<string, Space> pair in dict)
                        batched.Add(pair.Key, BatchSpace(pair.Value, n));
                    // Through the mapping constructor with the space's sort_keys, as Gymnasium re-sorts a batched Dict.
                    return new Dict(new OrderedSpaces(batched, dict.Keys), Space.CopyGenerator(dict.NpRandom), dict.SortKeys);
                }
                default: {
                    // Graph, Text, Sequence, OneOf and custom spaces: copies, re-seeded from a copy of the space's generator so
                    // sampling the batch never moves the original.
                    var batched = new Tuple(Enumerable.Range(0, n).Select(_ => space.DeepCopy()).ToArray(), Space.CopyGenerator(space.NpRandom));
                    Generator spaceRng = Space.CopyGenerator(space.NpRandom);
                    using NDArray drawn = spaceRng.integers(0, 100_000_000L, size: new Shape((long)n));
                    object?[] seeds = Enumerable.Range(0, n).Select(i => (object?)new BigInteger(Convert.ToInt64(drawn.GetAtIndex(i), CultureInfo.InvariantCulture))).ToArray();
                    batched.Seed(seeds);
                    return batched;
                }
            }
        }

        /// <summary>
        ///     One space for <paramref name="spaces"/> that differ only in parameters (Gymnasium's <c>batch_differing_spaces</c>):
        ///     boxes stack their bounds, discrete spaces become a <see cref="MultiDiscrete"/>, composites batch per position or
        ///     key, and <see cref="Graph"/>, <see cref="Text"/>, <see cref="Sequence"/> and <see cref="OneOf"/> become a
        ///     <see cref="Tuple"/> of copies.
        /// </summary>
        /// <param name="spaces">The spaces, all of one type.</param>
        /// <returns>A new space, owned by the caller, starting from a copy of the first space's generator.</returns>
        /// <exception cref="ValueError">Thrown for an empty list, or spaces whose dtypes, shapes or keys differ, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown for spaces of different types, or a type with no batching.</exception>
        public static Space BatchDifferingSpaces(IReadOnlyList<Space> spaces) {
            if (spaces is null || spaces.Count == 0)
                throw new ValueError("Expects a non-empty list of spaces");
            // An untyped env's spaces are ObjectSpace views, Gym.NET's own wrapper: the spaces inside are the ones Gymnasium's
            // function gets (as in BatchSpace), so SyncVectorEnv's "different" mode batches untyped envs too.
            spaces = spaces.Select(space => ObjectSpace.Unview(space) ?? space).ToArray();
            Type first = spaces[0].GetType();
            if (!spaces.All(s => first.IsInstanceOfType(s)))
                throw new TypeError($"Expects all spaces to be of the same type, actual types: [{string.Join(", ", spaces.Select(s => PyBuiltins.TypeRepr(s)))}]");
            Generator seed = Space.CopyGenerator(spaces[0].NpRandom);
            switch (spaces[0]) {
                case Box: {
                    Box[] boxes = spaces.Cast<Box>().ToArray();
                    if (!boxes.All(b => b.DType == boxes[0].DType))
                        throw new ValueError($"Expected all dtypes to be equal, actually [{string.Join(", ", boxes.Select(b => $"dtype('{Discrete.DTypeName(b.DType)}')"))}]");
                    if (!boxes.All(b => b.Low.Shape.dimensions.SequenceEqual(boxes[0].Low.Shape.dimensions)))
                        throw new ValueError($"Expected all Box.low shape to be equal, actually [{string.Join(", ", boxes.Select(b => Space.ShapeRepr(b.Low.Shape)))}]");
                    if (!boxes.All(b => b.High.Shape.dimensions.SequenceEqual(boxes[0].High.Shape.dimensions)))
                        throw new ValueError($"Expected all Box.high shape to be equal, actually [{string.Join(", ", boxes.Select(b => Space.ShapeRepr(b.High.Shape)))}]");
                    // The bounds are the boxes' own arrays, borrowed for the stack, never created here.
#pragma warning disable NDW012
                    using NDArray low = np.stack(boxes.Select(b => b.Low).ToArray());
                    using NDArray high = np.stack(boxes.Select(b => b.High).ToArray());
#pragma warning restore NDW012
                    return new Box(low, high, dtype: boxes[0].DType, seed: seed);
                }
                case Discrete: {
                    Discrete[] discretes = spaces.Cast<Discrete>().ToArray();
                    // The "largest" dtype by item size fits the others, Gymnasium's choice.
                    DType largest = discretes.Select(d => d.DType).Aggregate((a, b) => b.itemsize > a.itemsize ? b : a);
                    using NDArray nvec = np.array(discretes.Select(d => d.N).ToArray());
                    using NDArray start = np.array(discretes.Select(d => d.Start).ToArray());
                    return new MultiDiscrete(nvec, largest, seed, start);
                }
                case MultiDiscrete: {
                    MultiDiscrete[] multis = spaces.Cast<MultiDiscrete>().ToArray();
                    if (!multis.All(m => m.DType == multis[0].DType))
                        throw new ValueError($"Expected all dtypes to be equal, actually [{string.Join(", ", multis.Select(m => $"dtype('{Discrete.DTypeName(m.DType)}')"))}]");
                    if (!multis.All(m => m.Nvec.Shape.dimensions.SequenceEqual(multis[0].Nvec.Shape.dimensions)))
                        throw new ValueError($"Expects all MultiDiscrete.nvec shape, actually [{string.Join(", ", multis.Select(m => Space.ShapeRepr(m.Nvec.Shape)))}]");
                    if (!multis.All(m => m.Start.Shape.dimensions.SequenceEqual(multis[0].Start.Shape.dimensions)))
                        throw new ValueError($"Expects all MultiDiscrete.start shape, actually [{string.Join(", ", multis.Select(m => Space.ShapeRepr(m.Start.Shape)))}]");
                    // The starts are the spaces' own arrays, borrowed for the stack, never created here.
#pragma warning disable NDW012
                    using NDArray low = np.stack(multis.Select(m => m.Start).ToArray());
#pragma warning restore NDW012
                    NDArray[] ends = multis.Select(m => m.Start + m.Nvec).ToArray();
                    try {
                        using NDArray stacked = np.stack(ends);
                        using NDArray one = NDArray.Scalar(1L).astype(multis[0].DType);
                        using NDArray high = stacked - one;
                        return new Box(low, high, dtype: multis[0].DType, seed: seed);
                    } finally {
                        foreach (NDArray end in ends)
                            end.Dispose();
                    }
                }
                case MultiBinary: {
                    MultiBinary[] binaries = spaces.Cast<MultiBinary>().ToArray();
                    if (!binaries.All(b => b.Shape.dimensions.SequenceEqual(binaries[0].Shape.dimensions)))
                        throw new ValueError($"Expected all MultiBinary shapes to be equal, actually [{string.Join(", ", binaries.Select(b => Space.ShapeRepr(b.Shape)))}]");
                    return new Box(0, 1, new Shape(new[] { (long)binaries.Length }.Concat(binaries[0].Shape.dimensions).ToArray()), binaries[0].DType, seed);
                }
                case Tuple: {
                    Tuple[] tuples = spaces.Cast<Tuple>().ToArray();
                    int count = tuples[0].Count;
                    if (tuples.Any(t => t.Count != count))
                        throw new ValueError("zip() argument 2 is shorter than argument 1");
                    return new Tuple(Enumerable.Range(0, count).Select(i => BatchDifferingSpaces(tuples.Select(t => t[i]).ToArray())).ToArray(), seed);
                }
                case Dict: {
                    Dict[] dicts = spaces.Cast<Dict>().ToArray();
                    if (!dicts.All(d => d.Keys.SequenceEqual(dicts[0].Keys)))
                        throw new ValueError($"Expected all Dict spaces to have the same keys, actually [{string.Join(", ", dicts.Select(d => PyBuiltins.Repr(d.Keys.ToList())))}]");
                    var batched = new Dictionary<string, Space>();
                    foreach (string key in dicts[0].Keys)
                        batched.Add(key, BatchDifferingSpaces(dicts.Select(d => d[key]).ToArray()));
                    return new Dict(new OrderedSpaces(batched, dicts[0].Keys), seed, dicts[0].SortKeys);
                }
                case Graph:
                case Text:
                case Sequence:
                case OneOf:
                    return new Tuple(spaces.Select(s => s.DeepCopy()).ToArray(), seed);
                default:
                    throw new TypeError($"Requires the Space type to have a registered `batch_differing_space`, current list: {first.Name}");
            }
        }

        /// <summary>
        ///     The elements of a batched sample, one per env (Gymnasium's <c>iterate</c>): an array's first axis, a batched
        ///     tuple's or dict's per-env tuples or dicts.
        /// </summary>
        /// <param name="space">The batched space (from <see cref="BatchSpace"/>).</param>
        /// <param name="items">A batched sample.</param>
        /// <returns>The per-env elements; array elements are views into <paramref name="items"/>.</returns>
        /// <exception cref="TypeError">Thrown for a <see cref="Discrete"/> space, or items that cannot be iterated.</exception>
        /// <exception cref="CustomSpaceError">Thrown for a space type with no iteration.</exception>
        public static IEnumerable<object?> Iterate(Space space, object? items) {
            // An untyped env's spaces are ObjectSpace views, Gym.NET's own wrapper: the space inside is the one Gymnasium's function gets.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new TypeError("The space provided to `iterate` is not a gymnasium Space instance, type: <class 'NoneType'>, None");
                case Discrete:
                    throw new TypeError("Unable to iterate over a space of type `Discrete`.");
                case Box:
                case MultiDiscrete:
                case MultiBinary:
                    return IterateBase(items);
                case Tuple tuple when tuple.Spaces.All(IsRegistered): {
                    object?[] parts = AsTupleItems(items);
                    IEnumerable<object?>[] columns = tuple.Spaces.Select((s, i) => Iterate(s, parts[i])).ToArray();
                    return Zip(columns).Select(row => (object?)row);
                }
                case Tuple tuple:
                    try {
                        return PyIter(items);
                    } catch (Exception e) {
                        // Gymnasium names the subspaces that lack an iterate function and quotes iter()'s own error.
                        string unregistered = string.Join(", ", tuple.Spaces.Where(s => !IsRegistered(s)).Select(s => PyBuiltins.TypeRepr(s)));
                        throw new CustomSpaceError($"Could not iterate through {tuple} as no custom iterate function is registered for [{unregistered}] and `iter(items)` raised the following error: {e.Message}.");
                    }
                case Dict dict:
                    return IterateDict(dict, items);
                default:
                    throw new CustomSpaceError($"Space of type `{PyBuiltins.TypeRepr(space)}` doesn't have an registered `iterate` function. Register `{PyBuiltins.TypeRepr(space)}` for `iterate` to support it.");
            }
        }

        /// <summary>
        ///     Stacks per-env elements into a batch (Gymnasium's <c>concatenate</c>): arrays into <paramref name="out"/> along a
        ///     new first axis, tuples and dicts per position or key, and other spaces into a tuple of the items.
        /// </summary>
        /// <param name="space">The space of one env's elements.</param>
        /// <param name="items">The elements, one per env.</param>
        /// <param name="out">The destination (from <see cref="CreateEmptyArray"/>): an array, a tuple or dict of them, or <see langword="null"/> for other spaces.</param>
        /// <returns><paramref name="out"/> filled, a new tuple or dict holding the filled parts, or a tuple of the items.</returns>
        /// <exception cref="TypeError">
        ///     Thrown when <paramref name="space"/> is <see langword="null"/>, or an array space's item is neither an array nor
        ///     a Python number (<c>Expected an array to concatenate, got &lt;class 'str'&gt;</c>).
        /// </exception>
        /// <exception cref="OverflowError">Thrown, as Python's <c>OverflowError</c>, for an int beyond 64 bits, which NumPy would stack as an object.</exception>
        /// <remarks>
        ///     An array space's items may be Python numbers, as <c>np.stack</c> takes them: an int stacks as int64 (uint64 above
        ///     <see cref="long.MaxValue"/>), a float as float64, a bool as bool, a .NET <see langword="float"/> as float32 (the
        ///     NumPy scalar it stands for); the batch then takes <paramref name="out"/>'s dtype.
        /// </remarks>
        public static object? Concatenate(Space space, IEnumerable<object?> items, object? @out) {
            // An untyped env's spaces are ObjectSpace views, Gym.NET's own wrapper: the space inside is the one Gymnasium's function gets.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new TypeError("The space provided to `concatenate` is not a gymnasium Space instance, type: <class 'NoneType'>, None");
                case Box:
                case Discrete:
                case MultiDiscrete:
                case MultiBinary: {
                    // np.stack takes whatever np.asanyarray takes: the caller's arrays, borrowed for the stack, and Python
                    // numbers (an env's Discrete observation as an int), which become 0-d arrays of NumPy's default dtypes
                    // here and are released once stacked.
                    var created = new List<NDArray>();
                    try {
#pragma warning disable NDW012
                        NDArray[] arrays = items.Select(item => item as NDArray ?? Track(created, ScalarArray(item))).ToArray();
#pragma warning restore NDW012
                        using NDArray stacked = np.stack(arrays);
                        if (@out is not NDArray destination)
                            return stacked.copy();
                        // np.stack(..., out=out): the batch is written into the caller's buffer, which the caller keeps.
                        np.copyto(destination, stacked);
                        return destination;
                    } finally {
                        foreach (NDArray array in created)
                            array.Dispose();
                    }
                }
                case Tuple tuple: {
                    object?[] list = items.ToArray();
                    object?[] outParts = AsTupleItems(@out);
                    return tuple.Spaces.Select((s, i) => Concatenate(s, list.Select(item => AsTupleItems(item)[i]), outParts[i])).ToArray();
                }
                case Dict dict: {
                    object?[] list = items.ToArray();
                    IReadOnlyDictionary<string, object?> outParts = Dict.AsDict(@out) ?? throw new TypeError("Expected a dict output to concatenate a Dict space into.");
                    var result = new Info();
                    foreach (KeyValuePair<string, Space> pair in dict)
                        result[pair.Key] = Concatenate(pair.Value, list.Select(item => Dict.AsDict(item)![pair.Key]), outParts[pair.Key]);
                    return result;
                }
                default:
                    return items.ToArray();
            }
        }

        /// <summary>
        ///     An empty batch for <paramref name="n"/> envs (Gymnasium's <c>create_empty_array</c>): arrays of shape
        ///     <c>(n,) + space.shape</c> from <paramref name="fn"/>, tuples and dicts of them, and placeholder tuples for spaces
        ///     whose batches are tuples.
        /// </summary>
        /// <param name="space">The space of one env's elements.</param>
        /// <param name="n">The number of envs.</param>
        /// <param name="fn">Creates an array of a shape and dtype; <see langword="null"/> is <c>np.zeros</c>, Gymnasium's default.</param>
        /// <returns>The empty batch, owned by the caller; <see langword="null"/> for an unknown custom space, as in Gymnasium.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        public static object? CreateEmptyArray(Space space, int n = 1, Func<Shape, DType, NDArray>? fn = null) {
            fn ??= (shape, dtype) => np.zeros(shape, dtype);
            // An untyped env's spaces are ObjectSpace views, Gym.NET's own wrapper: the space inside is the one Gymnasium's function gets.
            space = ObjectSpace.Unview(space)!;
            switch (space) {
                case null:
                    throw new TypeError("The space provided to `create_empty_array` is not a gymnasium Space instance, type: <class 'NoneType'>, None");
                case Box or Discrete or MultiDiscrete or MultiBinary:
                    return fn(new Shape(new[] { (long)n }.Concat(space.Shape!.Value.dimensions).ToArray()), space.DType!);
                case Tuple tuple:
                    return tuple.Spaces.Select(s => CreateEmptyArray(s, n, fn)).ToArray();
                case Dict dict: {
                    var result = new Info();
                    foreach (KeyValuePair<string, Space> pair in dict)
                        result[pair.Key] = CreateEmptyArray(pair.Value, n, fn);
                    return result;
                }
                case Graph graph:
                    return Enumerable.Range(0, n).Select(_ => (object?)new GraphInstance(
                        CreateEmptyArray(graph.NodeSpace, 1, fn),
                        graph.EdgeSpace is null ? null : CreateEmptyArray(graph.EdgeSpace, 1, fn),
                        graph.EdgeSpace is null ? null : fn(new Shape(1L, 2L), np.int64))).ToArray();
                case Text text:
                    return Enumerable.Range(0, n).Select(_ => (object?)string.Concat(Enumerable.Repeat(Text.CodePoints(text.Characters).First(), (int)text.MinLength))).ToArray();
                case Sequence sequence:
                    return sequence.Stack
                        ? Enumerable.Range(0, n).Select(_ => CreateEmptyArray(sequence.FeatureSpace, 1, fn)).ToArray()
                        : Enumerable.Range(0, n).Select(_ => (object?)Array.Empty<object?>()).ToArray();
                case OneOf:
                    return Enumerable.Range(0, n).Select(_ => (object?)Array.Empty<object?>()).ToArray();
                default:
                    return null;
            }
        }

        /// <summary>
        ///     <c>np.asanyarray(item)</c> for a Python number <see cref="Concatenate"/> stacks: a new 0-d array of NumPy's dtype
        ///     for it (int64, or uint64 past <see cref="long.MaxValue"/>; float64; bool; float32 and float16 for the .NET floats
        ///     standing for NumPy's scalars).
        /// </summary>
        /// <param name="item">The item.</param>
        /// <returns>A new 0-d array the caller disposes.</returns>
        /// <exception cref="TypeError">Thrown for anything else: <c>Expected an array to concatenate, got &lt;class 'str'&gt;</c>.</exception>
        /// <exception cref="OverflowError">Thrown for an int beyond 64 bits.</exception>
        private static NDArray ScalarArray(object? item) {
            switch (item) {
                case bool flag:
                    return NDArray.Scalar(flag);
                case sbyte or byte or short or ushort or int or uint or long:
                    return NDArray.Scalar(Convert.ToInt64(item, CultureInfo.InvariantCulture));
                case ulong unsigned:
                    return unsigned <= long.MaxValue ? NDArray.Scalar((long)unsigned) : NDArray.Scalar(unsigned);
                case BigInteger integer when integer >= long.MinValue && integer <= long.MaxValue:
                    return NDArray.Scalar((long)integer);
                case BigInteger integer when integer >= 0 && integer <= ulong.MaxValue:
                    return NDArray.Scalar((ulong)integer);
                case BigInteger:
                    throw new OverflowError("Python int too large to convert to C long");
                case double d:
                    return NDArray.Scalar(d);
                case float f:
                    return NDArray.Scalar(f);
                case Half h:
                    return NDArray.Scalar(h);
                default:
                    throw new TypeError($"Expected an array to concatenate, got {PyBuiltins.TypeRepr(item)}");
            }
        }

        /// <summary>Records an array <see cref="Concatenate"/> created, so it is released after the stack.</summary>
        /// <param name="created">The arrays created so far.</param>
        /// <param name="array">The new array.</param>
        /// <returns><paramref name="array"/>.</returns>
        private static NDArray Track(List<NDArray> created, NDArray array) {
            created.Add(array);
            return array;
        }

        /// <summary>
        ///     Whether Gymnasium's <c>iterate</c> registry holds the space's exact type (its <c>Tuple</c> path zips its
        ///     subspaces only then): <see cref="Discrete"/>, <see cref="Box"/>, <see cref="MultiDiscrete"/>,
        ///     <see cref="MultiBinary"/>, <see cref="Tuple"/>, <see cref="Dict"/>.
        /// </summary>
        /// <param name="space">The subspace.</param>
        /// <returns><see langword="true"/> for a registered type.</returns>
        private static bool IsRegistered(Space space) {
            Type type = space.GetType();
            return type == typeof(Discrete) || type == typeof(Box) || type == typeof(MultiDiscrete) || type == typeof(MultiBinary) || type == typeof(Tuple) || type == typeof(Dict);
        }

        /// <summary>Gymnasium's <c>_iterate_base</c>: Python's <c>iter(items)</c>, its failure reworded.</summary>
        /// <param name="items">The batch.</param>
        /// <returns>Its items; an array's are views along its first axis.</returns>
        /// <exception cref="TypeError">Thrown for items that cannot be iterated: <c>Unable to iterate over the following elements: …</c>.</exception>
        private static IEnumerable<object?> IterateBase(object? items) {
            try {
                return PyIter(items);
            } catch (TypeError) {
                throw new TypeError($"Unable to iterate over the following elements: {PyBuiltins.Str(items)}");
            }
        }

        /// <summary>
        ///     Python's <c>iter(x)</c> over a value's Python counterpart: an array along its first axis (views), a string by
        ///     code point, a dict by key, a tuple or list by item.
        /// </summary>
        /// <param name="items">The value.</param>
        /// <returns>Its items.</returns>
        /// <exception cref="TypeError">
        ///     Thrown, with Python's text, for a value that is not iterable (<c>'int' object is not iterable</c>) or a 0-d array
        ///     (<c>iteration over a 0-d array</c>).
        /// </exception>
        private static IEnumerable<object?> PyIter(object? items) {
            switch (items) {
                case NDArray { ndim: 0 }:
                    throw new TypeError("iteration over a 0-d array");
                case NDArray array:
                    return Enumerable.Range(0, (int)array.shape[0]).Select(i => (object?)array[i.ToString(CultureInfo.InvariantCulture)]);
                case string text:
                    return Text.CodePoints(text).Select(c => (object?)c);
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    return pairs.Select(pair => (object?)pair.Key);
                case System.Collections.IDictionary dictionary:
                    return dictionary.Keys.Cast<object?>();
                case System.Runtime.CompilerServices.ITuple tuple:
                    return Tuple.TupleItems(tuple);
                case System.Collections.IEnumerable sequence:
                    return sequence.Cast<object?>();
                default:
                    throw new TypeError($"'{PyBuiltins.TypeName(items)}' object is not iterable");
            }
        }

        /// <summary>Gymnasium's <c>_iterate_dict</c>: one dict per env, zipping the subspaces' iterations.</summary>
        /// <param name="dict">The batched dict space.</param>
        /// <param name="items">The batched dict.</param>
        /// <returns>The per-env dicts.</returns>
        /// <exception cref="TypeError">Thrown when the items are not a dict.</exception>
        private static IEnumerable<object?> IterateDict(Dict dict, object? items) {
            IReadOnlyDictionary<string, object?> batch = Dict.AsDict(items) ?? throw new TypeError($"Unable to iterate over the following elements: {PyBuiltins.Str(items)}");
            string[] keys = dict.Keys.ToArray();
            IEnumerable<object?>[] columns = keys.Select(k => Iterate(dict[k], batch[k])).ToArray();
            foreach (object?[] row in Zip(columns)) {
                var element = new Info();
                for (int i = 0; i < keys.Length; i++)
                    element[keys[i]] = row[i];
                yield return element;
            }
        }

        /// <summary>Python's <c>zip(*columns, strict=True)</c>.</summary>
        /// <param name="columns">The sequences.</param>
        /// <returns>One array per position.</returns>
        /// <exception cref="ValueError">Thrown when the sequences differ in length.</exception>
        private static IEnumerable<object?[]> Zip(IEnumerable<object?>[] columns) {
            IEnumerator<object?>[] enumerators = columns.Select(c => c.GetEnumerator()).ToArray();
            try {
                while (true) {
                    bool[] moved = enumerators.Select(e => e.MoveNext()).ToArray();
                    if (moved.All(m => !m))
                        yield break;
                    if (moved.Any(m => !m))
                        throw new ValueError("zip() argument 2 is shorter than argument 1");
                    yield return enumerators.Select(e => e.Current).ToArray();
                }
            } finally {
                foreach (IEnumerator<object?> enumerator in enumerators)
                    enumerator.Dispose();
            }
        }

        /// <summary>A tuple value's items (an <see langword="object"/>?[], an <see cref="System.Runtime.CompilerServices.ITuple"/>, or a list).</summary>
        /// <param name="value">The value.</param>
        /// <returns>Its items.</returns>
        /// <exception cref="TypeError">Thrown for a value that is not a tuple.</exception>
        private static object?[] AsTupleItems(object? value) => value switch {
            object?[] array => array,
            System.Runtime.CompilerServices.ITuple tuple => Tuple.TupleItems(tuple),
            System.Collections.IList list => list.Cast<object?>().ToArray(),
            _ => throw new TypeError($"Expected a tuple, got {PyBuiltins.TypeRepr(value)}"),
        };

        /// <summary>
        ///     A dictionary that enumerates in a given key order, so a batched <see cref="Dict"/> sees the original's order
        ///     before its own <c>sort_keys</c> decides (Gymnasium passes a dict built in the original's order).
        /// </summary>
        private sealed class OrderedSpaces : IReadOnlyDictionary<string, Space> {
            /// <summary>The spaces by key.</summary>
            private readonly Dictionary<string, Space> _spaces;

            /// <summary>The key order.</summary>
            private readonly string[] _keys;

            /// <summary>Wraps the spaces in the given order.</summary>
            /// <param name="spaces">The spaces by key.</param>
            /// <param name="keys">The order.</param>
            public OrderedSpaces(Dictionary<string, Space> spaces, IEnumerable<string> keys) {
                _spaces = spaces;
                _keys = keys.ToArray();
            }

            /// <inheritdoc />
            public Space this[string key] => _spaces[key];

            /// <inheritdoc />
            public IEnumerable<string> Keys => _keys;

            /// <inheritdoc />
            public IEnumerable<Space> Values => _keys.Select(k => _spaces[k]);

            /// <inheritdoc />
            public int Count => _keys.Length;

            /// <inheritdoc />
            public bool ContainsKey(string key) => _spaces.ContainsKey(key);

            /// <inheritdoc />
            public bool TryGetValue(string key, out Space value) => _spaces.TryGetValue(key, out value!);

            /// <inheritdoc />
            public IEnumerator<KeyValuePair<string, Space>> GetEnumerator() => _keys.Select(k => new KeyValuePair<string, Space>(k, _spaces[k])).GetEnumerator();

            /// <inheritdoc />
            System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
        }
    }
}
