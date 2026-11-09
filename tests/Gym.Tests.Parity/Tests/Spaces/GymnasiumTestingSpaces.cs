using System.Collections.Generic;
using NumSharp;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     Gymnasium's own space test catalog (<c>TESTING_FUNDAMENTAL_SPACES</c> and <c>TESTING_COMPOSITE_SPACES</c>), as
    ///     specs, so every generic parity cell also runs over exactly the spaces Gymnasium tests itself with.
    /// </summary>
    /// <remarks>
    ///     Keyword-built dicts (<c>Dict(position=..., velocity=...)</c>) are given as lists of pairs: both keep the order
    ///     written and are never sorted. Integer literals (<c>Box(-1, 1, shape=(2,))</c>) stay Python ints.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/utils.py">tests/spaces/utils.py</see>.
    /// </remarks>
    internal static class GymnasiumTestingSpaces {
        /// <summary><c>TESTING_FUNDAMENTAL_SPACES</c>.</summary>
        internal static readonly IReadOnlyList<SpaceSpec> Fundamental = new SpaceSpec[] {
            new DiscreteSpec(3),
            new DiscreteSpec(3, start: -1),
            new DiscreteSpec(4, start: 1, dtype: "int32"),
            new BoxSpec(0.0, 1.0, new long[0]),
            new BoxSpec(0.0, double.PositiveInfinity, new long[] { 2, 2 }),
            new BoxArraySpec("Box(low=np.array([-10.0, 0.0]), high=np.array([10.0, 10.0]), dtype=np.float64)",
                () => np.array(new[] { -10.0, 0.0 }), () => np.array(new[] { 10.0, 10.0 }), "float64"),
            new BoxSpec(double.NegativeInfinity, 0.0, new long[] { 2, 1 }),
            new BoxSpec(0.0, double.PositiveInfinity, new long[] { 2, 1 }),
            new BoxSpec(0, 255, new long[] { 2, 2, 3 }, "uint8", intBounds: true),
            new BoxArraySpec("Box(low=np.array([0, 0, 1]), high=np.array([1, 0, 1]), dtype=np.bool_)",
                () => np.array(new long[] { 0, 0, 1 }), () => np.array(new long[] { 1, 0, 1 }), "bool"),
            new BoxArraySpec("Box(low=np.array([-np.inf, -np.inf, 0, -10]), high=np.array([np.inf, 0, np.inf, 10]), dtype=np.int32)",
                () => np.array(new[] { double.NegativeInfinity, double.NegativeInfinity, 0.0, -10.0 }),
                () => np.array(new[] { double.PositiveInfinity, 0.0, double.PositiveInfinity, 10.0 }), "int32"),
            new MultiDiscreteSpec(new long[] { 2, 2 }),
            new MultiDiscreteSpec(new long[] { 2, 3, 3, 2 }, new long[] { 2, 2 }),
            new MultiDiscreteSpec(new long[] { 2, 2 }, start: new long[] { 10, 10 }),
            new MultiDiscreteSpec(new long[] { 2, 3, 3, 2 }, new long[] { 2, 2 }, start: new long[] { 10, 20, 30, 40 }),
            new MultiDiscreteSpec(new long[] { 2, 3 }, dtype: "int8"),
            new MultiDiscreteSpec(new long[] { 2, 3 }, dtype: "uint16"),
            new MultiBinarySpec(8),
            new MultiBinarySpec(new long[] { 2, 3 }),
            new TextSpec(6),
            new TextSpec(6, 3),
            new TextSpec(6, 1, "abcdef"),
        };

        /// <summary><c>TESTING_COMPOSITE_SPACES</c>, the commented-out graphs of Gymnasium's list left out as Gymnasium leaves them.</summary>
        internal static readonly IReadOnlyList<SpaceSpec> Composite = new SpaceSpec[] {
            // Tuple spaces
            new TupleSpec(new DiscreteSpec(5), new DiscreteSpec(4)),
            new TupleSpec(new DiscreteSpec(5), VelocityBox()),
            new TupleSpec(new DiscreteSpec(5), new TupleSpec(new BoxSpec(0.0, 1.0, new long[] { 3 }), new DiscreteSpec(2))),
            new TupleSpec(new DiscreteSpec(3), new DictSpec(new[] { ("position", (SpaceSpec)new BoxSpec(0.0, 1.0, new long[0])), ("velocity", new DiscreteSpec(2)) }, mapping: false)),
            new TupleSpec(new OneOfSpec(IntBox(-1, 1, 2), IntBox(-1, 1, 3)), new BoxSpec(0.0, 1.0, new long[0])),
            new TupleSpec(new GraphSpec(IntBox(-1, 1, 2, 1), null), new DiscreteSpec(2)),
            // Dict spaces
            new DictSpec(new[] { ("position", (SpaceSpec)new DiscreteSpec(5)), ("velocity", VelocityBox()) }),
            new DictSpec(new[] { ("position", (SpaceSpec)new DiscreteSpec(6)), ("velocity", VelocityBox()) }, mapping: false),
            new DictSpec(new[] {
                ("a", (SpaceSpec)IntBox(0, 1, 3, 3)),
                ("b", new DictSpec(new[] { ("b_1", (SpaceSpec)IntBox(-100, 100, 2)), ("b_2", IntBox(-1, 1, 2)) })),
                ("c", new DiscreteSpec(4)),
            }),
            new DictSpec(new[] {
                ("a", (SpaceSpec)new DictSpec(new[] { ("a", (SpaceSpec)new GraphSpec(IntBox(-100, 100, 2, 2), null)), ("b", IntBox(-100, 100, 2, 2)) }, mapping: false)),
                ("b", new TupleSpec(IntBox(-100, 100, 2), IntBox(-100, 100, 2))),
            }, mapping: false),
            // Graph spaces
            new GraphSpec(IntBox(-1, 1, 2), null),
            new GraphSpec(IntBox(-100, 100, 3, 4), new DiscreteSpec(5)),
            new GraphSpec(new DiscreteSpec(5), IntBox(-100, 100, 3, 4)),
            new GraphSpec(new DiscreteSpec(3), new DiscreteSpec(4)),
            new GraphSpec(new DictSpec(new[] { ("node_feature_1", (SpaceSpec)new DiscreteSpec(5)), ("node_feature_2", new MultiBinarySpec(5)) }), IntBox(1, 4)),
            // Graph spaces with arbitrary node/edge spaces
            new GraphSpec(new MultiBinarySpec(4), null),
            new GraphSpec(new MultiDiscreteSpec(new long[] { 3, 4, 2 }), new DiscreteSpec(3)),
            new GraphSpec(new TupleSpec(IntBox(-1, 1, 2), new DiscreteSpec(3)), null),
            new GraphSpec(new DiscreteSpec(4), new TupleSpec(IntBox(0, 1, 2), new DiscreteSpec(3))),
            new GraphSpec(new DiscreteSpec(3), new DictSpec(new[] { ("weight", (SpaceSpec)IntBox(0, 1)), ("type", new DiscreteSpec(4)) })),
            new GraphSpec(
                new DictSpec(new[] { ("pos", (SpaceSpec)IntBox(-10, 10, 3)), ("label", new DiscreteSpec(5)) }),
                new DictSpec(new[] { ("weight", (SpaceSpec)IntBox(0, 1)), ("type", new DiscreteSpec(3)) })),
            new GraphSpec(
                new TupleSpec(new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)) }), new MultiBinarySpec(2)),
                new TupleSpec(new DiscreteSpec(4), IntBox(0, 1, 2))),
            // Sequence spaces
            new SequenceSpec(new DiscreteSpec(4)),
            new SequenceSpec(new DictSpec(new[] { ("feature", (SpaceSpec)IntBox(0, 1, 3)) })),
            new SequenceSpec(new GraphSpec(IntBox(-100, 100, 2, 2), new DiscreteSpec(4))),
            new SequenceSpec(new BoxSpec(0.0, 1.0, new long[0]), stack: true),
            new SequenceSpec(new DictSpec(new[] { ("a", (SpaceSpec)IntBox(0, 1, 3)), ("b", new DiscreteSpec(5)) }), stack: true),
            // OneOf spaces
            new OneOfSpec(new DiscreteSpec(3), new BoxSpec(0.0, 1.0, new long[0])),
            new OneOfSpec(new MultiBinarySpec(2), new MultiDiscreteSpec(new long[] { 2, 2 })),
            new OneOfSpec(IntBox(-1, 1, 2), IntBox(-1, 1, 3)),
            new OneOfSpec(new TextSpec(5), new DiscreteSpec(3)),
            new OneOfSpec(new MultiBinarySpec(3), new MultiBinarySpec(new long[] { 2, 2 })),
            new OneOfSpec(new DiscreteSpec(3), new DiscreteSpec(5, start: -2)),
            new OneOfSpec(new DiscreteSpec(3), IntBox(-1, 1, 2)),
            new OneOfSpec(new TextSpec(5), IntBox(-1, 1, 4)),
            new OneOfSpec(new OneOfSpec(new DiscreteSpec(2), new MultiBinarySpec(3)), IntBox(-1, 1, 4)),
            // More subspaces than the shared bool dtype can index (_flat_dtype)
            new OneOfSpec(BoolBox(), BoolBox(), BoolBox()),
        };

        /// <summary><c>Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float64)</c>.</summary>
        /// <returns>The spec.</returns>
        private static SpaceSpec VelocityBox() => new BoxArraySpec("Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 5.0]), dtype=np.float64)",
            () => np.array(new[] { 0.0, 0.0 }), () => np.array(new[] { 1.0, 5.0 }), "float64");

        /// <summary>A float32 box with Python-int bounds, <c>Box(low, high, shape=shape)</c>.</summary>
        /// <param name="low">The lower bound.</param>
        /// <param name="high">The upper bound.</param>
        /// <param name="shape">The shape; empty for <c>()</c>.</param>
        /// <returns>The spec.</returns>
        private static SpaceSpec IntBox(long low, long high, params long[] shape) => new BoxSpec(low, high, shape, intBounds: true);

        /// <summary><c>Box(0, 1, shape=(2,), dtype=np.bool_)</c>.</summary>
        /// <returns>The spec.</returns>
        private static SpaceSpec BoolBox() => new BoxSpec(0, 1, new long[] { 2 }, "bool", intBounds: true);
    }
}
