#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     Sequences of any length whose items belong to one feature space: every element is a tuple (an
    ///     <see langword="object"/>?[]) of feature-space elements, or, with <see cref="Stack"/>, those elements stacked into one
    ///     batch (an array, or a tuple or dict of arrays).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The length of an unmasked sample is geometric with <c>p = 0.25</c> (Gymnasium's arbitrary choice), so a sample is
    ///     empty one time in four and rarely long.
    ///     </para>
    ///     <para>
    ///     The space owns its feature space. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/sequence.py#L18">gymnasium/spaces/sequence.py</see>.
    ///     </para>
    /// </remarks>
    public class Sequence : Space<object?>, IEquatable<Sequence> {
        /// <summary>Sequences of <paramref name="space"/>'s elements (Gymnasium's <c>Sequence(space, seed=seed, stack=stack)</c>).</summary>
        /// <param name="space">The feature space; the sequence space owns it from now on.</param>
        /// <param name="seed">
        ///     An integer seed (it seeds the sequence's generator, which draws the feature space's seed, then is seeded again
        ///     with the same value), a generator to share, or <see langword="null"/>.
        /// </param>
        /// <param name="stack">Whether samples are the items stacked into one batch rather than a tuple of them.</param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> feature space, with Gymnasium's message.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        /// <remarks>
        ///     A stacked space batches the feature space once (<c>batch_space(space, 1)</c>), which reads the feature space's
        ///     generator and so creates it from entropy when it was not seeded, as in Gymnasium.
        /// </remarks>
        public Sequence(Space space, SpaceSeed? seed = null, bool stack = false) : base(null, null, null) {
            FeatureSpace = space ?? throw new TypeError("Expects the feature space to be instance of a gym Space, actual type: <class 'NoneType'>");
            Stack = stack;
            if (stack)
                StackedFeatureSpace = VectorUtils.BatchSpace(FeatureSpace, 1);
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The space every item belongs to (Gymnasium's <c>feature_space</c>).</summary>
        public Space FeatureSpace { get; }

        /// <summary>Whether samples are stacked (Gymnasium's <c>stack</c>).</summary>
        public bool Stack { get; }

        /// <summary>The feature space batched once, for a stacked space's membership and JSON (Gymnasium's <c>stacked_feature_space</c>); <see langword="null"/> otherwise.</summary>
        public Space? StackedFeatureSpace { get; }

        /// <summary>A sequence never flattens to a box: its length varies.</summary>
        public override bool IsNpFlattenable => false;

        /// <summary>
        ///     Draws a sequence (Gymnasium's <c>sample</c>): a length (geometric, fixed, or drawn from an array of lengths), then
        ///     that many feature-space samples, stacked when <see cref="Stack"/> is on.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/>, or a pair <c>(length_mask, sample_mask)</c> (an <see langword="object"/>?[] or tuple):
        ///     the length mask is <see langword="null"/> (geometric), a non-negative integer (fixed), or a 1-D integer array of
        ///     lengths to choose from; the sample mask goes to every feature-space sample.
        /// </param>
        /// <param name="probability"><see langword="null"/>, or the same pair with a probability mask for the feature space.</param>
        /// <returns>A tuple of samples, or the stacked batch; arrays are owned by the caller.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a negative or malformed length mask, with Gymnasium's messages.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown for a length mask that is neither an integer nor an array.</exception>
        public override object? Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");
            object?[] values;
            if (mask is not null) {
                object?[] pair = Pair(mask);
                long length = GenerateSampleLength(pair[0], "mask");
                values = Enumerable.Range(0, (int)length).Select(_ => FeatureSpace.Sample(mask: pair[1])).ToArray();
            } else if (probability is not null) {
                object?[] pair = Pair(probability);
                long length = GenerateSampleLength(pair[0], "probability");
                values = Enumerable.Range(0, (int)length).Select(_ => FeatureSpace.Sample(probability: pair[1])).ToArray();
            } else {
                using NDArray drawn = NpRandom.geometric(0.25);
                long length = Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
                values = Enumerable.Range(0, (int)length).Select(_ => FeatureSpace.Sample()).ToArray();
            }

            if (!Stack)
                return values;
            object? @out = VectorUtils.CreateEmptyArray(FeatureSpace, values.Length);
            return VectorUtils.Concatenate(FeatureSpace, values, @out);
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): a tuple of feature-space members, or for
        ///     a stacked space a batch whose every item is one.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        /// <exception cref="NumSharp.TypeError">
        ///     Thrown by a stacked space for a candidate that cannot be iterated (<c>Unable to iterate over the following
        ///     elements: …</c>), as Gymnasium's does: unlike the other spaces, its membership test raises rather than answering
        ///     <see langword="false"/>.
        /// </exception>
        /// <exception cref="Gym.Exceptions.CustomSpaceError">Thrown by a stacked space whose feature space has no batched iteration.</exception>
        public override bool Contains(object? x) {
            if (Stack)
                // Not guarded: Gymnasium's contains lets iterate's errors through, and parity keeps them.
                return VectorUtils.Iterate(StackedFeatureSpace!, x).All(FeatureSpace.Contains);

            object?[]? items = x switch {
                object?[] array => array,
                System.Runtime.CompilerServices.ITuple tuple => Tuple.TupleItems(tuple),
                _ => null,
            };
            return items is not null && items.All(FeatureSpace.Contains);
        }

        /// <summary>Converts samples to Gymnasium's JSON form: the stacked space's, or one feature-space list per sample.</summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>The JSON data.</returns>
        public override object ToJsonable(IEnumerable<object?> samples) {
            if (Stack)
                return StackedFeatureSpace!.ToJsonable(samples);
            return samples.Select(sample => FeatureSpace.ToJsonable(((object?[])sample!).AsEnumerable())).Cast<object?>().ToList();
        }

        /// <summary>Converts Gymnasium's JSON form back to sequences (Gymnasium's <c>from_jsonable</c>).</summary>
        /// <param name="jsonable">The JSON data.</param>
        /// <returns>The samples.</returns>
        public override IList<object?> FromJsonable(object jsonable) {
            if (Stack)
                return StackedFeatureSpace!.FromJsonable(jsonable);
            return JsonList(jsonable).Select(sample => (object?)FeatureSpace.FromJsonable(sample!).ToArray()).ToList();
        }

        /// <summary>Gymnasium's <c>repr</c>: <c>Sequence(Discrete(4), stack=False)</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"Sequence({FeatureSpace}, stack={(Stack ? "True" : "False")})";

        /// <summary>Gymnasium's <c>==</c>: an equal feature space and the same stacking.</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Sequence? other) => other is not null && FeatureSpace.Equals(other.FeatureSpace) && Stack == other.Stack;

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Sequence other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(FeatureSpace.GetHashCode(), Stack);

        /// <summary>
        ///     Gymnasium's <c>Sequence.seed</c> for an integer or <see langword="null"/>: <see langword="null"/> seeds both
        ///     generators from entropy; an integer seeds the sequence's generator, draws the feature space's seed from it, and
        ///     seeds it again with the same value, so integer and list seeding leave the sequence's generator alike.
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>The pair of seeds used, as a list.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        protected override object SeedCore(BigInteger? seed) {
            if (seed is null)
                return new List<object> { base.SeedCore(null), FeatureSpace.Seed((BigInteger?)null) };
            object superSeed = base.SeedCore(seed);
            using NDArray drawn = NpRandom.integers(int.MaxValue);
            long featureSeed = Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
            base.SeedCore(seed);
            return new List<object> { superSeed, FeatureSpace.Seed(featureSeed) };
        }

        /// <summary>Gymnasium's <c>Sequence.seed</c> for a pair: the sequence's seed and the feature space's.</summary>
        /// <param name="seeds">Two seeds.</param>
        /// <returns>The pair of seeds used.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when the list does not have two items.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when the sequence's own seed is negative or not a Python <c>int</c>.</exception>
        protected override object SeedListCore(IReadOnlyList<object?> seeds) {
            if (seeds.Count != 2)
                throw new ValueError($"Expects the seed to have two elements for the Sequence and feature space, actual length: {seeds.Count}");
            object superSeed = SeedOwnGenerator(seeds[0]);
            return new List<object> { superSeed, FeatureSpace.SeedWith(seeds[1]) };
        }

        /// <summary>Gymnasium's <c>Sequence.seed</c> error for a seed that is not an integer, a pair or <see langword="null"/>.</summary>
        /// <param name="seed">The refused seed.</param>
        /// <returns><c>TypeError: Expected None, int, tuple of ints, actual type: &lt;class 'dict'&gt;</c>.</returns>
        protected override Exception SeedTypeError(object? seed) =>
            new TypeError($"Expected None, int, tuple of ints, actual type: {PyBuiltins.TypeRepr(seed)}");

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                FeatureSpace.Dispose();
                StackedFeatureSpace?.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>Gymnasium's <c>generate_sample_length</c>: the length a length mask gives.</summary>
        /// <param name="lengthMask"><see langword="null"/>, a non-negative integer, or a 1-D integer array of lengths.</param>
        /// <param name="maskType"><c>mask</c> or <c>probability</c>, for the messages.</param>
        /// <returns>The length.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a negative length or a malformed array.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown for a length mask that is neither an integer nor an array.</exception>
        private long GenerateSampleLength(object? lengthMask, string maskType) {
            if (lengthMask is null) {
                using NDArray drawn = NpRandom.geometric(0.25);
                return Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
            }

            if (lengthMask is NDArray array && array.ndim > 0) {
                if (array.ndim != 1)
                    throw new AssertionError($"Expects the shape of the length mask of `{maskType}` to be 1-dimensional, actual shape: {ShapeRepr(array.Shape)}");
                using (NDArray nonNegative = array >= 0) {
                    if (!np.all(nonNegative))
                        throw new AssertionError($"Expects all values in the length_mask of `{maskType}` to be greater than or equal to zero, actual values: {PyBuiltins.Str(array)}");
                }

                if (array.dtype.kind is not ('i' or 'u'))
                    throw new AssertionError($"Expects the length mask array of `{maskType}` to have dtype of np.integer, actual type: {Discrete.DTypeName(array.dtype)}");
                using NDArray chosen = NpRandom.choice(array);
                return Convert.ToInt64(chosen.GetAtIndex(0), CultureInfo.InvariantCulture);
            }

            if (lengthMask is not bool && Discrete.TryToInt64(lengthMask, out long length)) {
                if (length < 0)
                    throw new AssertionError($"Expects the length mask of `{maskType}` to be greater than or equal to zero, actual value: {length}");
                return length;
            }

            throw new TypeError($"Expects the type of length_mask of `{maskType}` to be an integer or a np.ndarray, actual type: {PyBuiltins.TypeRepr(lengthMask)}");
        }

        /// <summary>A two-item mask argument's items.</summary>
        /// <param name="argument">The pair.</param>
        /// <returns>Its two items.</returns>
        /// <exception cref="NumSharp.TypeError">Thrown for a value that is not a pair (Python's indexing error).</exception>
        /// <exception cref="NumSharp.IndexError">Thrown for a tuple with fewer than two items (Python's <c>tuple index out of range</c>).</exception>
        private static object?[] Pair(object argument) {
            object?[] items = argument switch {
                object?[] array => array,
                System.Runtime.CompilerServices.ITuple tuple => Tuple.TupleItems(tuple),
                _ => throw new TypeError($"'{PyBuiltins.TypeName(argument)}' object is not subscriptable"),
            };
            if (items.Length < 2)
                throw new IndexError("tuple index out of range");
            return items;
        }
    }
}
