#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The cartesian product of discrete spaces: every element is an integer array whose entry <c>i</c> lies in
    ///     <c>[Start[i], Start[i] + Nvec[i])</c>, such as a game controller's several buttons each with its own options.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The space owns <see cref="Nvec"/> and <see cref="Start"/> and releases them in <see cref="Space.Dispose()"/>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/multi_discrete.py#L29">gymnasium/spaces/multi_discrete.py</see>.</para>
    /// </remarks>
    public class MultiDiscrete : Space<NDArray>, IEquatable<MultiDiscrete> {
        /// <summary>The flat space of <c>len(nvec)</c> discrete values (Gymnasium's <c>MultiDiscrete([5, 2, 2])</c>).</summary>
        /// <param name="nvec">The number of values of each entry; every one must be positive.</param>
        /// <param name="dtype">The integer dtype of the elements; <see langword="null"/> is int64, Gymnasium's default.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/> for a lazy entropy-seeded one.</param>
        /// <param name="start">Each entry's smallest value, or <see langword="null"/> for zeros.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for a non-integer dtype, a start of another shape, or a count that is not positive, with Gymnasium's messages.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="nvec"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public MultiDiscrete(long[] nvec, DType? dtype = null, SpaceSeed? seed = null, long[]? start = null)
            : this(NewArray(nvec ?? throw new ArgumentNullException(nameof(nvec))), dtype, seed, start is null ? null : NewArray(start), owned: true) { }

        /// <summary>A multi-dimensional space: <paramref name="nvec"/>'s shape is the elements' shape.</summary>
        /// <param name="nvec">The counts, any shape; read, never kept (the space stores a copy in its dtype).</param>
        /// <param name="dtype">The integer dtype; <see langword="null"/> is int64.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <param name="start">The smallest values, of <paramref name="nvec"/>'s shape, or <see langword="null"/> for zeros; read, never kept.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for a non-integer dtype, a start of another shape, or a count that is not positive.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="nvec"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public MultiDiscrete(NDArray nvec, DType? dtype = null, SpaceSeed? seed = null, NDArray? start = null)
            : this(nvec ?? throw new ArgumentNullException(nameof(nvec)), dtype, seed, start, owned: false) { }

        /// <summary>The shared constructor: Gymnasium's <c>MultiDiscrete.__init__</c>.</summary>
        /// <param name="nvec">The counts.</param>
        /// <param name="dtype">The dtype, or <see langword="null"/> for int64.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="start">The smallest values, or <see langword="null"/>.</param>
        /// <param name="owned">Whether <paramref name="nvec"/> and <paramref name="start"/> are temporaries this constructor made (and releases).</param>
        /// <exception cref="NumSharp.ValueError">Thrown for every invalid argument, with Gymnasium's message.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        private MultiDiscrete(NDArray nvec, DType? dtype, SpaceSeed? seed, NDArray? start, bool owned)
            : base(new Shape((long[])(nvec.Shape.dimensions ?? Array.Empty<long>()).Clone()), ValidateDType(dtype), null) {
            try {
                // np.array(nvec, dtype=dtype, copy=True): the space keeps its own copy in its dtype.
                Nvec = nvec.astype(DType, copy: true);
                Start = start is null ? np.zeros(Nvec.Shape, DType) : start.astype(DType, copy: true);
            } finally {
                if (owned) {
                    nvec.Dispose();
                    start?.Dispose();
                }
            }

            if (!SameDimensions(Start.Shape, Nvec.Shape))
                throw new ValueError($"start and nvec (counts) should have the same shape, got {ShapeRepr(Start.Shape)} and {ShapeRepr(Nvec.Shape)}");
            using (NDArray positive = Nvec > 0) {
                if (!np.all(positive))
                    throw new ValueError("nvec (counts) have to be positive");
            }

            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The number of values of each entry (Gymnasium's <c>nvec</c>), an array of the space's dtype and shape.</summary>
        public NDArray Nvec { get; }

        /// <summary>The smallest value of each entry (Gymnasium's <c>start</c>), an array of the space's dtype and shape.</summary>
        public NDArray Start { get; }

        /// <summary>The element dtype; never <see langword="null"/> for this space.</summary>
        public new DType DType => base.DType!;

        /// <summary>The element shape, <see cref="Nvec"/>'s.</summary>
        public new Shape Shape => base.Shape!.Value;

        /// <inheritdoc />
        public override bool IsNpFlattenable => true;

        /// <summary>
        ///     The number of entries along the first axis (Gymnasium's <c>len(space)</c>), with Gymnasium's warning for a
        ///     multi-dimensional space, whose rows are then counted.
        /// </summary>
        public long Count {
            get {
                if (Nvec.ndim >= 2)
                    Logger.Warn("Getting the length of a multi-dimensional MultiDiscrete space.");
                return Nvec.shape[0];
            }
        }

        /// <summary>
        ///     The subspace at <paramref name="index"/> (Gymnasium's <c>space[index]</c>): a <see cref="Discrete"/> for one entry
        ///     of a flat space, a <see cref="MultiDiscrete"/> for a row of a multi-dimensional one. The subspace starts from a
        ///     copy of this space's generator state, as in Gymnasium.
        /// </summary>
        /// <param name="index">The index; negative counts from the end, as in Python.</param>
        /// <returns>A new space, owned by the caller.</returns>
        /// <exception cref="NumSharp.IndexError">Thrown for an index outside the first axis.</exception>
        public Space<NDArray> this[int index] => Subspace(index.ToString(CultureInfo.InvariantCulture));

        /// <summary>The subspace at a multi-dimensional index (Gymnasium's <c>space[i, j]</c>).</summary>
        /// <param name="index">One index per leading axis.</param>
        /// <returns>A new space, owned by the caller.</returns>
        /// <exception cref="NumSharp.IndexError">Thrown for an index outside its axis.</exception>
        public Space<NDArray> this[params int[] index] => Subspace(string.Join(",", index.Select(i => i.ToString(CultureInfo.InvariantCulture))));

        /// <summary>
        ///     The subspace at a NumPy index written as NumSharp's index string (Gymnasium's <c>space[0:2, :]</c> is
        ///     <c>space["0:2, :"]</c>): an index that selects one entry gives a <see cref="Discrete"/>, any other a
        ///     <see cref="MultiDiscrete"/> of the selected counts and starts.
        /// </summary>
        /// <param name="index">The index: integers, slices (<c>"1:3"</c>, <c>":"</c>, <c>"::2"</c>), commas between axes.</param>
        /// <returns>A new space (never this one, even for <c>":"</c>), owned by the caller, starting from a copy of this space's generator state.</returns>
        /// <exception cref="NumSharp.IndexError">Thrown for an index outside its axis.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="index"/> is <see langword="null"/>.</exception>
        public Space<NDArray> this[string index] => Subspace(index ?? throw new ArgumentNullException(nameof(index)));

        /// <summary>
        ///     Draws one element (Gymnasium's <c>sample</c>): each entry uniformly, or under a per-entry mask or probability
        ///     mask given as nested tuples (<see langword="object"/>?[]) of 1-D arrays matching <see cref="Nvec"/>'s structure.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/>, or a tuple with one int8 array of length <c>nvec[i]</c> per entry (nested tuples for a
        ///     multi-dimensional space); an all-zero array draws that entry's start.
        /// </param>
        /// <param name="probability"><see langword="null"/>, or the same structure of float64 arrays, each summing to 1.</param>
        /// <returns>A new array of the space's dtype and shape, owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a malformed mask or probability, with Gymnasium's messages.</exception>
        /// <remarks>
        ///     Unmasked, <c>(random(nvec.shape) * nvec).astype(dtype) + start</c>, Gymnasium's draw and arithmetic; masked,
        ///     one <c>choice</c> per entry in C order. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/multi_discrete.py#L109">gymnasium/spaces/multi_discrete.py</see>.
        /// </remarks>
        public override NDArray Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");
            if (mask is not null || probability is not null) {
                var leaves = new List<long>();
                ApplyMask(mask ?? probability, Nvec, Start, mask is not null ? "mask" : "probability", leaves);
                using NDArray flat = np.array(leaves.ToArray());
                using NDArray shaped = flat.reshape(Shape);
                return shaped.astype(DType);
            }

            using NDArray uniform = NpRandom.random(Nvec.Shape);
            using NDArray scaled = uniform * Nvec;
            using NDArray truncated = scaled.astype(DType);
            return truncated + Start;
        }

        /// <summary>
        ///     Draws one element, typed (DR-86, DR-97): Gymnasium's <c>sample(mask, probability)</c> with one array per entry,
        ///     Gymnasium's tuple as a list. For a one-dimensional space the list is that tuple; for a multi-dimensional one it
        ///     holds the entries in C order (row by row), which this method groups into Gymnasium's nested tuples. A call on a
        ///     <see cref="MultiDiscrete"/> reference binds this overload unless an argument is typed <see langword="object"/>;
        ///     it draws exactly what <see cref="Sample(object, object)"/> draws with the tuples.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/>, or one int8 array per entry, of length <c>nvec[i]</c>, with 1 for the values that may be
        ///     drawn (all 0: that entry's start).
        /// </param>
        /// <param name="probability"><see langword="null"/>, or one float64 array per entry, of length <c>nvec[i]</c>, summing to 1.</param>
        /// <returns>A new array of the space's dtype and shape, owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">
        ///     Thrown when both masks are given, or when a multi-dimensional space gets a list whose length is not its number of
        ///     entries (one-dimensional spaces report a wrong length with Gymnasium's assertion).
        /// </exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a malformed array, with Gymnasium's messages.</exception>
        public NDArray Sample(IReadOnlyList<NDArray>? mask = null, IReadOnlyList<NDArray>? probability = null) =>
            // The tuples go to the untyped draw (the casts pick it), which checks them with Gymnasium's assertions.
            Sample((object?)Nest(mask, nameof(mask)), (object?)Nest(probability, nameof(probability)));

        /// <summary>
        ///     Reseeds the space's generator (Gymnasium's <c>seed</c>), typed (DR-86): a simple space's seed is one integer, so
        ///     it comes back as a <see cref="System.Numerics.BigInteger"/> where <see cref="Space.Seed(System.Numerics.BigInteger?)"/> returns it boxed.
        /// </summary>
        /// <param name="seed">Any non-negative integer, or <see langword="null"/> for a fresh generator from OS entropy.</param>
        /// <returns>The seed used: <paramref name="seed"/> itself, or the 128-bit entropy of an unseeded call, which reproduces it.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <exception cref="InvalidCastException">Thrown by a subclass whose seeding returns something other than one integer.</exception>
        /// <remarks>Footgun: a call through a <see cref="Space"/> reference binds the untyped base method, which returns the same seed as an <see langword="object"/>.</remarks>
        public new System.Numerics.BigInteger Seed(System.Numerics.BigInteger? seed = null) => (System.Numerics.BigInteger)base.Seed(seed);

        /// <summary>
        ///     Turns the typed overload's list into the tuple form the untyped draw reads: the list itself as Python's tuple for
        ///     a one-dimensional space, else nested tuples of <see cref="Nvec"/>'s shape filled in C order.
        /// </summary>
        /// <param name="arrays">The per-entry arrays, or <see langword="null"/>.</param>
        /// <param name="name">The parameter's name, for the message.</param>
        /// <returns>The tuple (<see langword="object"/>?[]), or <see langword="null"/> for no list.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when a multi-dimensional space gets a list of another length than its number of entries.</exception>
        private object?[]? Nest(IReadOnlyList<NDArray>? arrays, string name) {
            if (arrays is null)
                return null;
            long[] dims = Nvec.Shape.dimensions ?? Array.Empty<long>();
            // One dimension: the list is Gymnasium's tuple as it stands, so a wrong length meets Gymnasium's own assertion.
            if (dims.Length <= 1)
                return arrays.Select(a => (object?)a).ToArray();
            long entries = dims.Aggregate(1L, (product, d) => product * d);
            if (arrays.Count != entries)
                throw new ValueError($"`{name}` holds one array per entry of nvec in C order: expected {entries} arrays for nvec of shape {ShapeRepr(Nvec.Shape)}, got {arrays.Count}");
            int next = 0;
            return Level(0);

            // One tuple per axis level, the last level holding the arrays themselves, as Gymnasium's nested tuples do.
            object?[] Level(int axis) {
                var items = new object?[dims[axis]];
                // The (object?) cast types the conditional as object: typed NDArray, the inner tuple would go through
                // NDArray's implicit conversion from Array and become an array of objects, which NumSharp refuses.
                for (long i = 0; i < dims[axis]; i++)
                    items[i] = axis == dims.Length - 1 ? (object?)arrays[next++] : Level(axis + 1);
                return items;
            }
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): an array (a .NET array or list is
        ///     converted first) of the space's shape, a dtype that casts safely to the space's, and every entry between its start
        ///     and its largest value.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        /// <remarks>
        ///     Compares against <c>start + (nvec - 1)</c> rather than subtracting the start from <paramref name="x"/>, which could
        ///     overflow a narrow dtype, as Gymnasium's own comment explains.
        /// </remarks>
        public override bool Contains(object? x) {
            NDArray? owned = null;
            try {
                NDArray? array = x switch {
                    NDArray nd => nd,
                    Array a => owned = np.array(a),
                    System.Collections.IList list => owned = MultiBinary.SequenceToArray(list),
                    _ => null,
                };
                if (array is null || !SameDimensions(array.Shape, Shape) || !np.can_cast(array.dtype, DType))
                    return false;
                using NDArray atLeastStart = Start <= array;
                using NDArray one = NDArray.Scalar(1L).astype(DType);
                using NDArray lastOffset = Nvec - one;
                using NDArray largest = Start + lastOffset;
                using NDArray atMostLargest = array <= largest;
                return np.all(atLeastStart) && np.all(atMostLargest);
            } catch (Exception e) when (e is ArgumentException or InvalidCastException or NotSupportedException or ValueError or TypeError) {
                return false;
            } finally {
                owned?.Dispose();
            }
        }

        /// <summary>Converts samples to nested lists (Gymnasium's <c>sample.tolist()</c>).</summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>One nested list per sample.</returns>
        public override object ToJsonable(IEnumerable<NDArray> samples) => samples.Select(s => PythonList(s)).ToList();

        /// <summary>Converts nested lists back to arrays of the space's dtype (Gymnasium's <c>np.array(sample, dtype=dtype)</c>).</summary>
        /// <param name="jsonable">A list of nested lists.</param>
        /// <returns>New arrays, owned by the caller.</returns>
        /// <exception cref="ArgumentException">Thrown for a value no array can be made from.</exception>
        public override IList<NDArray> FromJsonable(object jsonable) => JsonList(jsonable).Select(v => {
            using NDArray made = v switch {
                NDArray nd => nd.copy(),
                Array a => np.array(a),
                System.Collections.IList list => MultiBinary.SequenceToArray(list),
                _ => throw new ArgumentException($"{PyBuiltins.Repr(v)} is not a MultiDiscrete sample.", nameof(jsonable)),
            };
            return made.astype(DType);
        }).ToList();

        /// <summary>Gymnasium's <c>repr</c>: <c>MultiDiscrete([2 3 4])</c>, with <c>start=…</c> when any start is not 0.</summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            using NDArray nonzero = Start != 0;
            if (np.any(nonzero))
                return $"MultiDiscrete({PyBuiltins.Str(Nvec)}, start={PyBuiltins.Str(Start)})";
            return $"MultiDiscrete({PyBuiltins.Str(Nvec)})";
        }

        /// <summary>Gymnasium's <c>==</c>: the same dtype, shape, counts and starts.</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(MultiDiscrete? other) {
            if (other is null)
                return false;
            if (ReferenceEquals(this, other))
                return true;
            if (DType != other.DType || !SameDimensions(Shape, other.Shape))
                return false;
            using NDArray sameCounts = Nvec == other.Nvec;
            using NDArray sameStarts = Start == other.Start;
            return np.all(sameCounts) && np.all(sameStarts);
        }

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is MultiDiscrete other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(ShapeRepr(Shape), Discrete.DTypeName(DType));

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                Nvec.Dispose();
                Start.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>
        ///     Gymnasium's <c>_apply_mask</c>: walks the mask and the counts together and draws each leaf entry, appending the
        ///     values in C order.
        /// </summary>
        /// <param name="subMask">The mask (a tuple) or, at a leaf, a 1-D array.</param>
        /// <param name="subNvec">The counts at this level: an array, or a 0-d array at a leaf.</param>
        /// <param name="subStart">The starts at this level, shaped like <paramref name="subNvec"/>.</param>
        /// <param name="maskType"><c>mask</c> or <c>probability</c>.</param>
        /// <param name="leaves">Where the drawn values go, in C order.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a malformed mask, with Gymnasium's messages.</exception>
        private void ApplyMask(object? subMask, NDArray subNvec, NDArray subStart, string maskType, List<long> leaves) {
            if (subNvec.ndim > 0) {
                // isinstance(sub_nvec, np.ndarray): the mask must be a tuple with one item per entry.
                if (subMask is not object?[] and not ITuple)
                    throw new AssertionError($"Expects the mask to be a tuple for sub_nvec ({PyBuiltins.Str(subNvec)}), actual type: {PyBuiltins.TypeRepr(subMask)}");
                object?[] items = subMask is object?[] array ? array : TupleItems((ITuple)subMask);
                long length = subNvec.shape[0];
                if (items.Length != length)
                    throw new AssertionError($"Expects the mask length to be equal to the number of actions, mask length: {items.Length}, nvec length: {length}");
                for (int i = 0; i < length; i++) {
                    using NDArray nvecItem = subNvec[i.ToString(CultureInfo.InvariantCulture)];
                    using NDArray startItem = subStart[i.ToString(CultureInfo.InvariantCulture)];
                    ApplyMask(items[i], nvecItem, startItem, maskType, leaves);
                }

                return;
            }

            long count = Convert.ToInt64(subNvec.GetAtIndex(0), CultureInfo.InvariantCulture);
            long start = Convert.ToInt64(subStart.GetAtIndex(0), CultureInfo.InvariantCulture);
            NDArray leafMask = subMask as NDArray ?? throw new AssertionError($"Expects the sub mask to be np.ndarray, actual type: {PyBuiltins.TypeRepr(subMask)}");
            long maskLength = leafMask.ndim == 0 ? throw new AssertionError("len() of unsized object") : leafMask.shape[0];
            if (maskLength != count)
                throw new AssertionError($"Expects the mask length to be equal to the number of actions, mask length: {maskLength}, action: {count}");

            if (maskType == "mask") {
                if (leafMask.dtype != np.int8)
                    throw new AssertionError($"Expects the mask dtype to be np.int8, actual dtype: {Discrete.DTypeName(leafMask.dtype)}");
                using NDArray valid = leafMask == 1;
                using NDArray zero = leafMask == 0;
                using NDArray either = np.logical_or(zero, valid);
                if (!np.all(either))
                    throw new AssertionError($"Expects all masks values to 0 or 1, actual values: {PyBuiltins.Str(leafMask)}");
                if (!np.any(valid)) {
                    leaves.Add(start);
                    return;
                }

                using NDArray candidates = np.nonzero(valid)[0];
                using NDArray chosen = NpRandom.choice(candidates);
                leaves.Add(Convert.ToInt64(chosen.GetAtIndex(0), CultureInfo.InvariantCulture) + start);
                return;
            }

            if (leafMask.dtype != np.float64)
                throw new AssertionError($"Expects the mask dtype to be np.float64, actual dtype: {Discrete.DTypeName(leafMask.dtype)}");
            using NDArray positive = leafMask > 0;
            using NDArray atMostOne = leafMask <= 1;
            using NDArray validProbability = np.logical_and(positive, atMostOne);
            using NDArray zeros = leafMask == 0;
            using NDArray allowed = np.logical_or(zeros, validProbability);
            if (!np.all(allowed))
                throw new AssertionError($"Expects all masks values to be between 0 and 1, actual values: {PyBuiltins.Str(leafMask)}");
            using NDArray total = np.sum(leafMask);
            using NDArray close = np.isclose(total, 1.0);
            if (!close.GetBoolean())
                throw new AssertionError($"Expects the sum of all mask values to be 1, actual sum: {PyBuiltins.Str(total)}");
            using NDArray normalized = leafMask / total;
            using NDArray indices = np.nonzero(validProbability)[0];
            using NDArray weights = normalized[validProbability];
            using NDArray picked = NpRandom.choice(indices, p: weights);
            leaves.Add(Convert.ToInt64(picked.GetAtIndex(0), CultureInfo.InvariantCulture) + start);
        }

        /// <summary>The subspace at an index, NumPy-style (Gymnasium's <c>__getitem__</c>).</summary>
        /// <param name="index">The index as NumSharp's string indexer reads it (<c>"1"</c>, <c>"0,1"</c>, <c>"-1"</c>).</param>
        /// <returns>A <see cref="Discrete"/> or a <see cref="MultiDiscrete"/> starting from this space's generator state.</returns>
        private Space<NDArray> Subspace(string index) {
            using NDArray nvec = Nvec[index];
            using NDArray start = Start[index];
            Space<NDArray> subspace = nvec.ndim == 0
                ? new Discrete(Convert.ToInt64(nvec.GetAtIndex(0), CultureInfo.InvariantCulture), start: Convert.ToInt64(start.GetAtIndex(0), CultureInfo.InvariantCulture), dtype: DType)
                : new MultiDiscrete(nvec, DType, start: start);
            // subspace.np_random.bit_generator.state = self.np_random.bit_generator.state: a copy of the state, not a shared
            // generator, so the two spaces draw the same next values and then go their own ways.
            subspace.NpRandom.bit_generator.state = NpRandom.bit_generator.state;
            return subspace;
        }

        /// <summary>The dtype a constructor was given, validated: <see langword="null"/> is int64; it must be an integer dtype.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The dtype to use.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for a dtype that is not an integer dtype.</exception>
        private static DType ValidateDType(DType? dtype) {
            DType resolved = dtype ?? np.int64;
            if (resolved.kind is not ('i' or 'u'))
                throw new ValueError($"Invalid MultiDiscrete dtype ({Discrete.DTypeName(resolved)}), must be an integer dtype");
            return resolved;
        }

        /// <summary>A .NET array of counts or starts as a 1-D int64 array.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array the caller releases.</returns>
        private static NDArray NewArray(long[] values) => np.array((long[])values.Clone());

        /// <summary>Whether two shapes have the same dimensions.</summary>
        /// <param name="left">A shape.</param>
        /// <param name="right">A shape.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        private static bool SameDimensions(Shape left, Shape right) =>
            (left.dimensions ?? Array.Empty<long>()).SequenceEqual(right.dimensions ?? Array.Empty<long>());

        /// <summary>A tuple's items.</summary>
        /// <param name="tuple">The tuple.</param>
        /// <returns>Its items in order.</returns>
        private static object?[] TupleItems(ITuple tuple) {
            var items = new object?[tuple.Length];
            for (int i = 0; i < items.Length; i++)
                items[i] = tuple[i];
            return items;
        }
    }
}
