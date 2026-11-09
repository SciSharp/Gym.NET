#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's space test catalog (<c>tests/spaces/utils.py</c>): <c>TESTING_FUNDAMENTAL_SPACES</c>,
    ///     <c>TESTING_COMPOSITE_SPACES</c> and the custom space, as factories, so every test builds its own spaces (they are
    ///     stateful and disposable) and the parametrized tests take a space by its position.
    /// </summary>
    /// <remarks>
    ///     A space is named by its <c>repr</c>, as Gymnasium's test ids are. Keyword-built dicts
    ///     (<c>Dict(position=..., velocity=...)</c>) are lists of pairs in Gym.NET: both keep the order written.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/utils.py">tests/spaces/utils.py</see>.
    /// </remarks>
    internal static class TestingSpaces {
        /// <summary><c>TESTING_FUNDAMENTAL_SPACES</c>.</summary>
        internal static readonly Func<Space>[] Fundamental = {
            () => new Discrete(3),
            () => new Discrete(3, start: -1),
            () => new Discrete(4, dtype: np.int32, start: 1),
            () => new Box(0.0, 1.0),
            () => new Box(0.0, double.PositiveInfinity, new Shape(2L, 2L)),
            () => BoxOf(new[] { -10.0, 0.0 }, new[] { 10.0, 10.0 }, np.float64),
            () => new Box(double.NegativeInfinity, 0.0, new Shape(2L, 1L)),
            () => new Box(0.0, double.PositiveInfinity, new Shape(2L, 1L)),
            () => new Box(0, 255, new Shape(2L, 2L, 3L), np.uint8),
            () => BoxOf(new long[] { 0, 0, 1 }, new long[] { 1, 0, 1 }, np.@bool),
            () => BoxOf(new[] { double.NegativeInfinity, double.NegativeInfinity, 0, -10 }, new[] { double.PositiveInfinity, 0, double.PositiveInfinity, 10 }, np.int32),
            () => new MultiDiscrete(new long[] { 2, 2 }),
            () => MultiDiscreteOf(new long[,] { { 2, 3 }, { 3, 2 } }),
            () => new MultiDiscrete(new long[] { 2, 2 }, start: new long[] { 10, 10 }),
            () => MultiDiscreteOf(new long[,] { { 2, 3 }, { 3, 2 } }, new long[,] { { 10, 20 }, { 30, 40 } }),
            () => new MultiDiscrete(new long[] { 2, 3 }, np.int8),
            () => new MultiDiscrete(new long[] { 2, 3 }, np.uint16),
            () => new MultiBinary(8),
            () => new MultiBinary(new long[] { 2, 3 }),
            () => new Text(6),
            () => new Text(6, minLength: 3),
            () => new Text(6, charset: "abcdef"),
        };

        /// <summary><c>TESTING_COMPOSITE_SPACES</c>; the graphs Gymnasium comments out are left out here too.</summary>
        internal static readonly Func<Space>[] Composite = {
            // Tuple spaces
            () => new Tuple(new Discrete(5), new Discrete(4)),
            () => new Tuple(new Discrete(5), VelocityBox()),
            () => new Tuple(new Discrete(5), new Tuple(new Box(0.0, 1.0, new Shape(3L)), new Discrete(2))),
            () => new Tuple(new Discrete(3), DictOf(("position", new Box(0.0, 1.0)), ("velocity", new Discrete(2)))),
            () => new Tuple(new OneOf(new Box(-1, 1, new Shape(2L)), new Box(-1, 1, new Shape(3L))), new Box(0.0, 1.0)),
            () => new Tuple(new Graph(new Box(-1, 1, new Shape(2L, 1L)), null), new Discrete(2)),
            // Dict spaces
            () => new Dict(new Dictionary<string, Space> { ["position"] = new Discrete(5), ["velocity"] = VelocityBox() }),
            () => DictOf(("position", new Discrete(6)), ("velocity", VelocityBox())),
            () => new Dict(new Dictionary<string, Space> {
                ["a"] = new Box(0, 1, new Shape(3L, 3L)),
                ["b"] = new Dict(new Dictionary<string, Space> { ["b_1"] = new Box(-100, 100, new Shape(2L)), ["b_2"] = new Box(-1, 1, new Shape(2L)) }),
                ["c"] = new Discrete(4),
            }),
            () => DictOf(
                ("a", DictOf(("a", new Graph(new Box(-100, 100, new Shape(2L, 2L)), null)), ("b", new Box(-100, 100, new Shape(2L, 2L))))),
                ("b", new Tuple(new Box(-100, 100, new Shape(2L)), new Box(-100, 100, new Shape(2L))))),
            // Graph spaces
            () => new Graph(new Box(-1, 1, new Shape(2L)), null),
            () => new Graph(new Box(-100, 100, new Shape(3L, 4L)), new Discrete(5)),
            () => new Graph(new Discrete(5), new Box(-100, 100, new Shape(3L, 4L))),
            () => new Graph(new Discrete(3), new Discrete(4)),
            () => new Graph(new Dict(new Dictionary<string, Space> { ["node_feature_1"] = new Discrete(5), ["node_feature_2"] = new MultiBinary(5) }), new Box(1, 4)),
            // Graph spaces with arbitrary node/edge spaces
            () => new Graph(new MultiBinary(4), null),
            () => new Graph(new MultiDiscrete(new long[] { 3, 4, 2 }), new Discrete(3)),
            () => new Graph(new Tuple(new Box(-1, 1, new Shape(2L)), new Discrete(3)), null),
            () => new Graph(new Discrete(4), new Tuple(new Box(0, 1, new Shape(2L)), new Discrete(3))),
            () => new Graph(new Discrete(3), new Dict(new Dictionary<string, Space> { ["weight"] = new Box(0, 1, Shape.Scalar), ["type"] = new Discrete(4) })),
            () => new Graph(
                new Dict(new Dictionary<string, Space> { ["pos"] = new Box(-10, 10, new Shape(3L)), ["label"] = new Discrete(5) }),
                new Dict(new Dictionary<string, Space> { ["weight"] = new Box(0, 1, Shape.Scalar), ["type"] = new Discrete(3) })),
            () => new Graph(
                new Tuple(new Dict(new Dictionary<string, Space> { ["a"] = new Discrete(3) }), new MultiBinary(2)),
                new Tuple(new Discrete(4), new Box(0, 1, new Shape(2L)))),
            // Sequence spaces
            () => new Sequence(new Discrete(4)),
            () => new Sequence(new Dict(new Dictionary<string, Space> { ["feature"] = new Box(0, 1, new Shape(3L)) })),
            () => new Sequence(new Graph(new Box(-100, 100, new Shape(2L, 2L)), new Discrete(4))),
            () => new Sequence(new Box(0.0, 1.0), stack: true),
            () => new Sequence(new Dict(new Dictionary<string, Space> { ["a"] = new Box(0, 1, new Shape(3L)), ["b"] = new Discrete(5) }), stack: true),
            // OneOf spaces
            () => new OneOf(new Discrete(3), new Box(0.0, 1.0)),
            () => new OneOf(new MultiBinary(2), new MultiDiscrete(new long[] { 2, 2 })),
            () => new OneOf(new Box(-1, 1, new Shape(2L)), new Box(-1, 1, new Shape(3L))),
            () => new OneOf(new Text(5), new Discrete(3)),
            () => new OneOf(new MultiBinary(3), new MultiBinary(new long[] { 2, 2 })),
            () => new OneOf(new Discrete(3), new Discrete(5, start: -2)),
            () => new OneOf(new Discrete(3), new Box(-1, 1, new Shape(2L))),
            () => new OneOf(new Text(5), new Box(-1, 1, new Shape(4L))),
            () => new OneOf(new OneOf(new Discrete(2), new MultiBinary(3)), new Box(-1, 1, new Shape(4L))),
            // More subspaces than the shared bool dtype can index (_flat_dtype)
            () => new OneOf(Enumerable.Range(0, 3).Select(_ => (Space)new Box(0, 1, new Shape(2L), np.@bool)).ToArray()),
        };

        /// <summary><c>TESTING_SPACES</c>: the fundamental spaces, then the composite ones.</summary>
        internal static readonly Func<Space>[] All = Fundamental.Concat(Composite).ToArray();

        /// <summary>
        ///     <c>TESTING_SPACE_SAMPLE_MASK</c>: one mask per fundamental space, <see langword="null"/> for the boxes, which take
        ///     none. Built per call: the masks are arrays a test disposes with its spaces.
        /// </summary>
        /// <returns>The masks, in <see cref="Fundamental"/>'s order.</returns>
        internal static object?[] FundamentalSampleMasks() {
            // SAMPLE_MASK_RNG: seeding.np_random(1), drawn in this order for the two default-charset text masks.
            (Generator rng, _) = Seeding.NpRandom(1);
            NDArray textMask1 = rng.integers(0, 2, size: new Shape(62L), dtype: np.int8);
            NDArray textMask2 = rng.integers(0, 2, size: new Shape(62L), dtype: np.int8);
            return new object?[] {
                // Discrete
                Int8(1, 1, 0),
                Int8(0, 0, 0),
                Int8(0, 0, 0, 1),
                // Box
                null, null, null, null, null, null, null, null,
                // Multi-discrete
                new object?[] { Int8(1, 1), Int8(0, 0) },
                new object?[] { new object?[] { Int8(1, 0), Int8(0, 1, 1) }, new object?[] { Int8(1, 1, 0), Int8(0, 1) } },
                new object?[] { Int8(1, 1), Int8(0, 0) },
                new object?[] { new object?[] { Int8(1, 0), Int8(0, 1, 1) }, new object?[] { Int8(1, 1, 0), Int8(0, 1) } },
                new object?[] { Int8(1, 1), Int8(0, 0, 0) },
                new object?[] { Int8(1, 1), Int8(0, 0, 0) },
                // Multi-binary
                Int8(0, 1, 0, 1, 0, 2, 1, 1),
                Int8Grid(new sbyte[] { 0, 1, 2, 0, 2, 1 }, 2, 3),
                // Text
                new object?[] { null, textMask1 },
                new object?[] { 4L, textMask2 },
                new object?[] { null, Int8(1, 1, 0, 1, 0, 0) },
            };
        }

        /// <summary>Every space's position, for <c>DynamicData</c> (MSTest shows the position; the message names the space).</summary>
        internal static IEnumerable<object[]> AllIndexes => Enumerable.Range(0, All.Length).Select(i => new object[] { i });

        /// <summary>Every fundamental space's position.</summary>
        internal static IEnumerable<object[]> FundamentalIndexes => Enumerable.Range(0, Fundamental.Length).Select(i => new object[] { i });

        /// <summary>A box with array bounds.</summary>
        /// <param name="low">The lower bounds.</param>
        /// <param name="high">The upper bounds.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The box.</returns>
        private static Box BoxOf(Array low, Array high, DType dtype) {
            using NDArray lowArray = np.array(low);
            using NDArray highArray = np.array(high);
            return new Box(lowArray, highArray, dtype: dtype);
        }

        /// <summary><c>Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float64)</c>.</summary>
        /// <returns>The box.</returns>
        private static Box VelocityBox() => BoxOf(new[] { 0.0, 0.0 }, new[] { 1.0, 5.0 }, np.float64);

        /// <summary>A 2-D <c>MultiDiscrete</c>.</summary>
        /// <param name="nvec">The counts.</param>
        /// <param name="start">The starts, or <see langword="null"/>.</param>
        /// <returns>The space.</returns>
        internal static MultiDiscrete MultiDiscreteOf(long[,] nvec, long[,]? start = null) {
            using NDArray counts = np.array(nvec);
            if (start is null)
                return new MultiDiscrete(counts);
            using NDArray starts = np.array(start);
            return new MultiDiscrete(counts, start: starts);
        }

        /// <summary>A dict built from pairs, in the order written (Python's keyword form, never sorted).</summary>
        /// <param name="pairs">The keys and spaces.</param>
        /// <returns>The space.</returns>
        internal static Dict DictOf(params (string Key, Space Space)[] pairs) =>
            new Dict(pairs.Select(p => new KeyValuePair<string, Space>(p.Key, p.Space)).ToList());

        /// <summary>An int8 vector.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        internal static NDArray Int8(params sbyte[] values) => np.array(values);

        /// <summary>An int8 matrix.</summary>
        /// <param name="values">The values in C order.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <returns>A new array.</returns>
        internal static NDArray Int8Grid(sbyte[] values, long rows, long columns) {
            using NDArray flat = np.array(values);
            return flat.reshape(new Shape(rows, columns)).copy();
        }
    }

    /// <summary>
    ///     <c>CustomSpace</c>: a space Gym.NET knows nothing about, which inherits Gymnasium's unimplemented base members
    ///     (<c>sample</c>, <c>contains</c>, <c>is_np_flattenable</c> raise <c>NotImplementedError</c>) and equals any other
    ///     custom space.
    /// </summary>
    internal sealed class CustomSpace : Space<object?> {
        /// <summary>A shapeless, dtype-less space.</summary>
        public CustomSpace() : base(null, null, null) { }

        /// <summary>Gymnasium's base <c>is_np_flattenable</c>.</summary>
        /// <exception cref="Gym.Exceptions.NotImplementedError">Always.</exception>
        public override bool IsNpFlattenable => throw new Gym.Exceptions.NotImplementedError("");

        /// <summary>Gymnasium's base <c>sample</c>.</summary>
        /// <param name="mask">Ignored.</param>
        /// <param name="probability">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="Gym.Exceptions.NotImplementedError">Always.</exception>
        public override object? Sample(object? mask = null, object? probability = null) => throw new Gym.Exceptions.NotImplementedError("");

        /// <summary>Gymnasium's base <c>contains</c>.</summary>
        /// <param name="x">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="Gym.Exceptions.NotImplementedError">Always.</exception>
        public override bool Contains(object? x) => throw new Gym.Exceptions.NotImplementedError("");

        /// <summary>Any custom space is equal to this one, as <c>CustomSpace.__eq__</c> says.</summary>
        /// <param name="obj">The other object.</param>
        /// <returns><see langword="true"/> for a <see cref="CustomSpace"/>.</returns>
        public override bool Equals(object? obj) => obj is CustomSpace;

        /// <inheritdoc />
        public override int GetHashCode() => typeof(CustomSpace).GetHashCode();
    }
}
