using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     The composite and multi spaces' details against Gymnasium's, live: masked and probability-weighted samples with
    ///     the generator states after them, seeding with lists, dicts and refused forms, membership matrices, and the errors
    ///     each space raises.
    /// </summary>
    /// <remarks>
    ///     A mask, seed or candidate is written once as a .NET value and crosses to Python through
    ///     <see cref="SpaceParity.ToPython"/> (tuples as tuples, arrays as NumPy copies), so both sides receive the same
    ///     bits; a NumPy scalar is built through the tree, because a 0-d array crosses as a NumPy array, not a scalar.
    /// </remarks>
    [TestClass]
    public sealed class SpaceDetailsLiveParityTests : ParityTestBase {
        /// <summary>
        ///     Masked and weighted sampling cases: the space, the seed, and the keyword arguments of each call, as .NET values
        ///     built fresh for each side.
        /// </summary>
        private static readonly Dictionary<string, (SpaceSpec Spec, Func<object?> Mask, Func<object?> Probability)> MaskCases = new() {
            ["MultiBinary(4) mask"] = (new MultiBinarySpec(4), () => Int8(0, 1, 2, 2), () => null),
            ["MultiBinary(4) probability"] = (new MultiBinarySpec(4), () => null, () => Float64(0.0, 1.0, 0.5, 0.25)),
            ["MultiBinary([2, 2]) mask"] = (new MultiBinarySpec(new long[] { 2, 2 }), () => Int8Grid(new sbyte[] { 2, 0, 1, 2 }, 2, 2), () => null),
            ["MultiDiscrete([3, 5]) mask"] = (new MultiDiscreteSpec(new long[] { 3, 5 }), () => new object?[] { Int8(1, 0, 1), Int8(0, 0, 1, 1, 0) }, () => null),
            ["MultiDiscrete([3, 5]) probability"] = (new MultiDiscreteSpec(new long[] { 3, 5 }), () => null, () => new object?[] { Float64(0.5, 0.0, 0.5), Float64(0.0, 0.0, 0.5, 0.5, 0.0) }),
            ["MultiDiscrete([3, 5], start=[-2, 4]) mask"] = (new MultiDiscreteSpec(new long[] { 3, 5 }, start: new long[] { -2, 4 }), () => new object?[] { Int8(0, 1, 1), Int8(1, 0, 0, 0, 1) }, () => null),
            ["MultiDiscrete([[3, 2], [4, 5]]) mask"] = (new MultiDiscreteSpec(new long[] { 3, 2, 4, 5 }, new long[] { 2, 2 }),
                () => new object?[] { new object?[] { Int8(1, 1, 0), Int8(0, 1) }, new object?[] { Int8(1, 0, 0, 1), Int8(0, 0, 0, 0, 1) } }, () => null),
            ["MultiDiscrete([3, 2]) all-masked"] = (new MultiDiscreteSpec(new long[] { 3, 2 }), () => new object?[] { Int8(0, 0, 0), Int8(0, 1) }, () => null),
            ["Tuple mask"] = (new TupleSpec(new DiscreteSpec(3), new MultiBinarySpec(2)), () => new object?[] { Int8(1, 0, 1), Int8(2, 1) }, () => null),
            ["Tuple probability with None"] = (new TupleSpec(new DiscreteSpec(3), new MultiBinarySpec(2)), () => null, () => new object?[] { Float64(0.2, 0.8, 0.0), null }),
            ["Dict mask"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)), ("b", new DiscreteSpec(2)) }),
                () => new Dictionary<string, object?> { ["a"] = Int8(0, 1, 1), ["b"] = null }, () => null),
            ["Dict probability"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)), ("b", new DiscreteSpec(2)) }),
                () => null, () => new Dictionary<string, object?> { ["a"] = null, ["b"] = Float64(0.3, 0.7) }),
            ["Text length and charset mask"] = (new TextSpec(5, 1, "abcd"), () => new object?[] { 3L, Int8(1, 0, 1, 1) }, () => null),
            ["Text charset mask only"] = (new TextSpec(5, 1, "abcd"), () => new object?[] { null, Int8(0, 1, 0, 1) }, () => null),
            ["Text charset probability"] = (new TextSpec(6, 2, "abcd"), () => null, () => new object?[] { null, Float64(0.1, 0.2, 0.3, 0.4) }),
            ["Text fixed length probability"] = (new TextSpec(6, 2, "abcd"), () => null, () => new object?[] { 4L, Float64(0.0, 0.5, 0.0, 0.5) }),
            ["Sequence fixed length and feature mask"] = (new SequenceSpec(new DiscreteSpec(4)), () => new object?[] { 3L, Int8(1, 1, 0, 0) }, () => null),
            ["Sequence length array"] = (new SequenceSpec(new DiscreteSpec(4)), () => new object?[] { np.array(new long[] { 1, 2, 5 }), null }, () => null),
            ["Sequence probability"] = (new SequenceSpec(new DiscreteSpec(4)), () => null, () => new object?[] { null, Float64(0.25, 0.25, 0.5, 0.0) }),
            ["Sequence stacked fixed length"] = (new SequenceSpec(new DiscreteSpec(3), stack: true), () => new object?[] { 4L, null }, () => null),
            ["OneOf mask"] = (new OneOfSpec(new DiscreteSpec(3), new MultiBinarySpec(2)), () => new object?[] { Int8(1, 0, 0), Int8(1, 2) }, () => null),
            ["OneOf probability"] = (new OneOfSpec(new DiscreteSpec(3), new MultiBinarySpec(2)), () => null, () => new object?[] { Float64(0.0, 0.0, 1.0), null }),
        };

        /// <summary>The masked sampling cases, by name.</summary>
        public static IEnumerable<object[]> MaskCaseNames => MaskCases.Keys.Select(k => new object[] { k });

        /// <summary>
        ///     Five samples under a mask or a probability mask are Gymnasium's, value for value, and so are the generator
        ///     states of the space and its subspaces afterwards.
        /// </summary>
        /// <param name="name">The case.</param>
        [TestMethod]
        [DynamicData(nameof(MaskCaseNames))]
        public void MaskedSamples_MatchGymnasium(string name) {
            (SpaceSpec spec, Func<object?> mask, Func<object?> probability) = MaskCases[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(7));
            using Space dotnet = spec.BuildDotNet(7);
            for (int i = 1; i <= 5; i++) {
                ByteContract.AssertSame($"{name}: {spec.PythonText}.sample(...) #{i}",
                    Outcome.CapturePython(() => {
                        object? m = mask(), p = probability();
                        try {
                            using PyObject? pythonMask = m is null ? null : SpaceParity.ToPython(m);
                            using PyObject? pythonProbability = p is null ? null : SpaceParity.ToPython(p);
                            return Python.gymnasium.spaces.space.Space.sample(python.Object, pythonMask, pythonProbability);
                        } finally {
                            Outcome.Release(m);
                            Outcome.Release(p);
                        }
                    }),
                    Outcome.Capture(() => {
                        object? m = mask(), p = probability();
                        try {
                            return dotnet.Sample(m, p);
                        } finally {
                            Outcome.Release(m);
                            Outcome.Release(p);
                        }
                    }));
            }

            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{name} after 5 samples");
        }

        /// <summary>
        ///     Mask errors raise Gymnasium's exception type and message, and leave the generator where Gymnasium's leaves it
        ///     (OneOf draws its subspace index before checking the mask).
        /// </summary>
        /// <param name="name">The case.</param>
        [TestMethod]
        [DynamicData(nameof(MaskErrorCaseNames))]
        public void MaskErrors_MatchGymnasium(string name) {
            (SpaceSpec spec, Func<object?> mask, Func<object?> probability) = MaskErrorCases[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(3));
            using Space dotnet = spec.BuildDotNet(3);
            ByteContract.AssertSame($"{name}: {spec.PythonText}.sample(...)",
                Outcome.CapturePython(() => {
                    object? m = mask(), p = probability();
                    try {
                        using PyObject? pythonMask = m is null ? null : SpaceParity.ToPython(m);
                        using PyObject? pythonProbability = p is null ? null : SpaceParity.ToPython(p);
                        return Python.gymnasium.spaces.space.Space.sample(python.Object, pythonMask, pythonProbability);
                    } finally {
                        Outcome.Release(m);
                        Outcome.Release(p);
                    }
                }),
                Outcome.Capture(() => {
                    object? m = mask(), p = probability();
                    try {
                        return dotnet.Sample(m, p);
                    } finally {
                        Outcome.Release(m);
                        Outcome.Release(p);
                    }
                }));
            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{name} after the refused sample");
        }

        /// <summary>Sampling calls Gymnasium refuses, with the error each raises.</summary>
        private static readonly Dictionary<string, (SpaceSpec Spec, Func<object?> Mask, Func<object?> Probability)> MaskErrorCases = new() {
            ["MultiBinary mask and probability"] = (new MultiBinarySpec(2), () => Int8(1, 1), () => Float64(0.5, 0.5)),
            ["MultiBinary mask of wrong dtype"] = (new MultiBinarySpec(2), () => Float64(1.0, 0.0), () => null),
            ["MultiBinary mask of wrong shape"] = (new MultiBinarySpec(2), () => Int8(1, 0, 1), () => null),
            ["MultiBinary mask value 3"] = (new MultiBinarySpec(2), () => Int8(1, 3), () => null),
            ["MultiDiscrete mask not a tuple"] = (new MultiDiscreteSpec(new long[] { 3, 2 }), () => Int8(1, 0, 1), () => null),
            ["MultiDiscrete mask of wrong length"] = (new MultiDiscreteSpec(new long[] { 3, 2 }), () => new object?[] { Int8(1, 0, 1) }, () => null),
            ["Tuple mask not a tuple"] = (new TupleSpec(new DiscreteSpec(2), new DiscreteSpec(3)), () => new List<object?> { null, null }, () => null),
            ["Tuple mask of wrong length"] = (new TupleSpec(new DiscreteSpec(2), new DiscreteSpec(3)), () => new object?[] { null }, () => null),
            ["Dict mask and probability"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)) }), () => new Dictionary<string, object?> { ["a"] = null }, () => new Dictionary<string, object?> { ["a"] = null }),
            ["Text mask of wrong dtype"] = (new TextSpec(4, 1, "ab"), () => new object?[] { 2L, Float64(1.0, 1.0) }, () => null),
            ["Text length out of range"] = (new TextSpec(4, 2, "ab"), () => new object?[] { 9L, null }, () => null),
            ["Sequence negative length"] = (new SequenceSpec(new DiscreteSpec(3)), () => new object?[] { -1L, null }, () => null),
            ["Sequence float length"] = (new SequenceSpec(new DiscreteSpec(3)), () => new object?[] { 1.5, null }, () => null),
            ["Sequence 2-D length array"] = (new SequenceSpec(new DiscreteSpec(3)), () => new object?[] { Int64Grid(new long[] { 1, 2, 3, 4 }, 2, 2), null }, () => null),
            ["OneOf mask not a tuple"] = (new OneOfSpec(new DiscreteSpec(2), new DiscreteSpec(3)), () => new List<object?> { null, null }, () => null),
            ["OneOf mask of wrong length"] = (new OneOfSpec(new DiscreteSpec(2), new DiscreteSpec(3)), () => new object?[] { null }, () => null),
            ["OneOf mask and probability"] = (new OneOfSpec(new DiscreteSpec(2), new DiscreteSpec(3)), () => new object?[] { null, null }, () => new object?[] { null, null }),
        };

        /// <summary>The mask error cases, by name.</summary>
        public static IEnumerable<object[]> MaskErrorCaseNames => MaskErrorCases.Keys.Select(k => new object[] { k });

        /// <summary>Seeding calls with every seed form: what each returns or raises, and the streams left behind.</summary>
        private static readonly Dictionary<string, (SpaceSpec Spec, Func<object?> Seed)> SeedCases = new() {
            ["Tuple list"] = (new TupleSpec(new DiscreteSpec(3), new BoxSpec(0.0, 1.0, new long[] { 2 })), () => new List<object?> { 1L, 2L }),
            ["Tuple nested list"] = (new TupleSpec(new DiscreteSpec(2), new TupleSpec(new DiscreteSpec(3), new DiscreteSpec(4))), () => new List<object?> { 5L, new List<object?> { 6L, 7L } }),
            ["Tuple tuple"] = (new TupleSpec(new DiscreteSpec(3), new DiscreteSpec(3)), () => new object?[] { 9L, 10L }),
            ["Tuple list of wrong length"] = (new TupleSpec(new DiscreteSpec(3), new DiscreteSpec(3)), () => new List<object?> { 1L }),
            ["Tuple dict"] = (new TupleSpec(new DiscreteSpec(3), new DiscreteSpec(3)), () => new Dictionary<string, object?> { ["a"] = 1L }),
            ["Tuple string"] = (new TupleSpec(new DiscreteSpec(3)), () => "42"),
            ["Dict dict"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)), ("b", new MultiBinarySpec(3)) }), () => new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 2L }),
            ["Dict nested dict"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DictSpec(new[] { ("x", (SpaceSpec)new DiscreteSpec(2)) })), ("b", new DiscreteSpec(2)) }),
                () => new Dictionary<string, object?> { ["a"] = new Dictionary<string, object?> { ["x"] = 4L }, ["b"] = 5L }),
            ["Dict dict with other keys"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)) }), () => new Dictionary<string, object?> { ["z"] = 1L }),
            ["Dict list"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)) }), () => new List<object?> { 1L }),
            ["Sequence pair"] = (new SequenceSpec(new DiscreteSpec(4)), () => new List<object?> { 3L, 4L }),
            ["Sequence pair of wrong length"] = (new SequenceSpec(new DiscreteSpec(4)), () => new List<object?> { 3L }),
            ["Sequence dict"] = (new SequenceSpec(new DiscreteSpec(4)), () => new Dictionary<string, object?> { ["a"] = 1L }),
            ["Graph triple"] = (new GraphSpec(new BoxSpec(0.0, 1.0, new long[] { 2 }), new DiscreteSpec(3)), () => new List<object?> { 1L, 2L, 3L }),
            ["Graph pair without edges"] = (new GraphSpec(new DiscreteSpec(3), null), () => new List<object?> { 1L, 2L }),
            ["Graph pair with edges"] = (new GraphSpec(new DiscreteSpec(3), new DiscreteSpec(2)), () => new List<object?> { 1L, 2L }),
            ["Graph dict"] = (new GraphSpec(new DiscreteSpec(3), null), () => new Dictionary<string, object?> { ["a"] = 1L }),
            ["OneOf list"] = (new OneOfSpec(new DiscreteSpec(2), new DiscreteSpec(5)), () => new List<object?> { 1L, 2L, 3L }),
            ["OneOf list of wrong length"] = (new OneOfSpec(new DiscreteSpec(2), new DiscreteSpec(5)), () => new List<object?> { 1L, 2L }),
            ["OneOf list with a list for its own seed"] = (new OneOfSpec(new DiscreteSpec(2)), () => new List<object?> { new List<object?> { 1L }, 2L }),
            ["Discrete list"] = (new DiscreteSpec(3), () => new List<object?> { 1L }),
            ["Discrete negative"] = (new DiscreteSpec(3), () => -4L),
            ["Discrete bool"] = (new DiscreteSpec(3), () => true),
            ["Discrete np.int64"] = (new DiscreteSpec(3), () => NDArray.Scalar(5L)),
            ["Tuple np.int64"] = (new TupleSpec(new DiscreteSpec(3)), () => NDArray.Scalar(5L)),
            ["MultiBinary dict"] = (new MultiBinarySpec(3), () => new Dictionary<string, object?> { ["a"] = 1L }),
        };

        /// <summary>The seeding cases, by name.</summary>
        public static IEnumerable<object[]> SeedCaseNames => SeedCases.Keys.Select(k => new object[] { k });

        /// <summary>
        ///     <c>space.seed(seed)</c> for a list, tuple, dict, nested seeds and refused forms returns or raises what Gymnasium's
        ///     does, and when it succeeds, three samples and every generator state afterwards are Gymnasium's.
        /// </summary>
        /// <param name="name">The case.</param>
        [TestMethod]
        [DynamicData(nameof(SeedCaseNames))]
        public void Seed_AnyForm_MatchesGymnasium(string name) {
            (SpaceSpec spec, Func<object?> seed) = SeedCases[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(null));
            using Space dotnet = spec.BuildDotNet(null);
            Outcome pythonSeeded = Outcome.CapturePython(() => {
                object? value = seed();
                try {
                    // A NumPy scalar must cross as one (a 0-d array would cross as an ndarray): build it through the tree.
                    using PyObject pythonSeed = value is NDArray { ndim: 0 } scalar ? Python.numpy.int64(Convert.ToInt64(scalar.GetAtIndex(0))) : SpaceParity.ToPython(value);
                    return Python.gymnasium.spaces.space.Space.seed(python.Object, pythonSeed);
                } finally {
                    Outcome.Release(value);
                }
            });
            Outcome dotnetSeeded = Outcome.Capture(() => {
                object? value = seed();
                try {
                    return dotnet.SeedWith(value);
                } finally {
                    Outcome.Release(value);
                }
            });
            ByteContract.AssertSame($"{name}: {spec.PythonText}.seed(...)", pythonSeeded, dotnetSeeded);
            if (pythonSeeded.Raised != null)
                return;
            for (int i = 1; i <= 3; i++)
                ByteContract.AssertSame($"{name}: sample() #{i} after seeding", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => dotnet.Sample()));
            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{name} after seeding and 3 samples");
        }

        /// <summary>Membership candidates per space: the Python spelling, Gymnasium's candidate, and Gym.NET's.</summary>
        private static readonly Dictionary<string, (SpaceSpec Spec, (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] Candidates)> ContainsCases = new() {
            ["MultiBinary(3)"] = (new MultiBinarySpec(3), new (string, Func<PyObject>, Func<object?>)[] {
                ("np.array([0, 1, 1], int8)", () => Numpy(Int8(0, 1, 1)), () => Int8(0, 1, 1)),
                ("np.array([0, 1, 1])", () => Numpy(Int64(0, 1, 1)), () => Int64(0, 1, 1)),
                ("np.array([0., 1., 1.])", () => Numpy(Float64(0, 1, 1)), () => Float64(0, 1, 1)),
                ("np.array([0, 2, 1], int8)", () => Numpy(Int8(0, 2, 1)), () => Int8(0, 2, 1)),
                ("np.array([0, 1], int8)", () => Numpy(Int8(0, 1)), () => Int8(0, 1)),
                ("[0, 1, 1]", () => SpaceParity.ToPython(new List<object?> { 0L, 1L, 1L }), () => new List<object?> { 0L, 1L, 1L }),
                ("(0, 1, 1)", () => SpaceParity.ToPython(new object?[] { 0L, 1L, 1L }), () => new object?[] { 0L, 1L, 1L }),
                ("None", () => PyConvert.ToPython(null), () => null),
                ("'011'", () => PyConvert.ToPython("011"), () => "011"),
            }),
            ["MultiDiscrete([3, 5])"] = (new MultiDiscreteSpec(new long[] { 3, 5 }), new (string, Func<PyObject>, Func<object?>)[] {
                ("np.array([2, 4])", () => Numpy(Int64(2, 4)), () => Int64(2, 4)),
                ("np.array([3, 4])", () => Numpy(Int64(3, 4)), () => Int64(3, 4)),
                ("np.array([-1, 0])", () => Numpy(Int64(-1, 0)), () => Int64(-1, 0)),
                ("np.array([2, 4], int32)", () => Numpy(Int32(2, 4)), () => Int32(2, 4)),
                ("np.array([2., 4.])", () => Numpy(Float64(2, 4)), () => Float64(2, 4)),
                ("np.array([2, 4, 0])", () => Numpy(Int64(2, 4, 0)), () => Int64(2, 4, 0)),
                ("[2, 4]", () => SpaceParity.ToPython(new List<object?> { 2L, 4L }), () => new List<object?> { 2L, 4L }),
                ("(2, 4)", () => SpaceParity.ToPython(new object?[] { 2L, 4L }), () => new object?[] { 2L, 4L }),
            }),
            ["Tuple((Discrete(3), Discrete(2)))"] = (new TupleSpec(new DiscreteSpec(3), new DiscreteSpec(2)), new (string, Func<PyObject>, Func<object?>)[] {
                ("(1, 1)", () => SpaceParity.ToPython(new object?[] { 1L, 1L }), () => new object?[] { 1L, 1L }),
                ("[1, 1]", () => SpaceParity.ToPython(new List<object?> { 1L, 1L }), () => new List<object?> { 1L, 1L }),
                ("(1,)", () => SpaceParity.ToPython(new object?[] { 1L }), () => new object?[] { 1L }),
                ("(3, 0)", () => SpaceParity.ToPython(new object?[] { 3L, 0L }), () => new object?[] { 3L, 0L }),
                ("np.array([1, 1])", () => Numpy(Int64(1, 1)), () => Int64(1, 1)),
                ("1", () => PyConvert.ToPython(1L), () => 1L),
            }),
            ["Dict({'a': Discrete(3), 'b': Discrete(2)})"] = (new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(3)), ("b", new DiscreteSpec(2)) }), new (string, Func<PyObject>, Func<object?>)[] {
                ("{'a': 1, 'b': 0}", () => SpaceParity.ToPython(new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 0L }), () => new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 0L }),
                ("{'b': 0, 'a': 1}", () => SpaceParity.ToPython(new Dictionary<string, object?> { ["b"] = 0L, ["a"] = 1L }), () => new Dictionary<string, object?> { ["b"] = 0L, ["a"] = 1L }),
                ("{'a': 1}", () => SpaceParity.ToPython(new Dictionary<string, object?> { ["a"] = 1L }), () => new Dictionary<string, object?> { ["a"] = 1L }),
                ("{'a': 1, 'b': 0, 'c': 0}", () => SpaceParity.ToPython(new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 0L, ["c"] = 0L }), () => new Dictionary<string, object?> { ["a"] = 1L, ["b"] = 0L, ["c"] = 0L }),
                ("{'a': 5, 'b': 0}", () => SpaceParity.ToPython(new Dictionary<string, object?> { ["a"] = 5L, ["b"] = 0L }), () => new Dictionary<string, object?> { ["a"] = 5L, ["b"] = 0L }),
                ("[1, 0]", () => SpaceParity.ToPython(new List<object?> { 1L, 0L }), () => new List<object?> { 1L, 0L }),
            }),
            ["Text(4, min_length=1, charset='ab')"] = (new TextSpec(4, 1, "ab"), new (string, Func<PyObject>, Func<object?>)[] {
                ("'ab'", () => PyConvert.ToPython("ab"), () => "ab"),
                ("''", () => PyConvert.ToPython(""), () => ""),
                ("'abc'", () => PyConvert.ToPython("abc"), () => "abc"),
                ("'aaaaa'", () => PyConvert.ToPython("aaaaa"), () => "aaaaa"),
                ("5", () => PyConvert.ToPython(5L), () => 5L),
            }),
            ["Sequence(Discrete(3))"] = (new SequenceSpec(new DiscreteSpec(3)), new (string, Func<PyObject>, Func<object?>)[] {
                ("(0, 1, 2)", () => SpaceParity.ToPython(new object?[] { 0L, 1L, 2L }), () => new object?[] { 0L, 1L, 2L }),
                ("()", () => SpaceParity.ToPython(Array.Empty<object?>()), () => Array.Empty<object?>()),
                ("[0, 1]", () => SpaceParity.ToPython(new List<object?> { 0L, 1L }), () => new List<object?> { 0L, 1L }),
                ("(3,)", () => SpaceParity.ToPython(new object?[] { 3L }), () => new object?[] { 3L }),
            }),
            ["Sequence(Discrete(3), stack=True)"] = (new SequenceSpec(new DiscreteSpec(3), stack: true), new (string, Func<PyObject>, Func<object?>)[] {
                ("np.array([0, 1, 2])", () => Numpy(Int64(0, 1, 2)), () => Int64(0, 1, 2)),
                ("np.array([0, 3])", () => Numpy(Int64(0, 3)), () => Int64(0, 3)),
                ("[0, 1]", () => SpaceParity.ToPython(new List<object?> { 0L, 1L }), () => new List<object?> { 0L, 1L }),
                ("5", () => PyConvert.ToPython(5L), () => 5L),
            }),
            ["OneOf((Discrete(2), Discrete(3)))"] = (new OneOfSpec(new DiscreteSpec(2), new DiscreteSpec(3)), new (string, Func<PyObject>, Func<object?>)[] {
                ("(0, 1)", () => SpaceParity.ToPython(new object?[] { 0L, 1L }), () => new object?[] { 0L, 1L }),
                ("(1, 2)", () => SpaceParity.ToPython(new object?[] { 1L, 2L }), () => new object?[] { 1L, 2L }),
                ("(np.int64(1), 2)", () => PythonPair(() => Python.numpy.int64(1), 2L), () => new object?[] { NDArray.Scalar(1L), 2L }),
                ("(np.int32(1), 2)", () => PythonPair(() => Python.numpy.int32(1), 2L), () => new object?[] { NDArray.Scalar(1), 2L }),
                ("(True, 2)", () => SpaceParity.ToPython(new object?[] { true, 2L }), () => new object?[] { true, 2L }),
                ("(2, 0)", () => SpaceParity.ToPython(new object?[] { 2L, 0L }), () => new object?[] { 2L, 0L }),
                ("(-1, 0)", () => SpaceParity.ToPython(new object?[] { -1L, 0L }), () => new object?[] { -1L, 0L }),
                ("[0, 1]", () => SpaceParity.ToPython(new List<object?> { 0L, 1L }), () => new List<object?> { 0L, 1L }),
                ("(0,)", () => SpaceParity.ToPython(new object?[] { 0L }), () => new object?[] { 0L }),
            }),
        };

        /// <summary>The membership cases, by name.</summary>
        public static IEnumerable<object[]> ContainsCaseNames => ContainsCases.Keys.Select(k => new object[] { k });

        /// <summary>Membership agrees with Gymnasium's <c>contains</c> for every candidate, including the errors it lets through.</summary>
        /// <param name="name">The space.</param>
        [TestMethod]
        [DynamicData(nameof(ContainsCaseNames))]
        public void Contains_MatchesGymnasium(string name) {
            (SpaceSpec spec, (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] candidates) = ContainsCases[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in candidates) {
                ByteContract.AssertSame($"{name}.contains({text})",
                    Outcome.CapturePython(() => {
                        using PyObject candidate = gymnasium();
                        return Python.gymnasium.spaces.space.Space.contains(python.Object, candidate);
                    }),
                    Outcome.Capture(() => {
                        object? candidate = value();
                        try {
                            return dotnet.Contains(candidate);
                        } finally {
                            Outcome.Release(candidate);
                        }
                    }));
            }
        }

        /// <summary>
        ///     <c>Graph.sample</c> with a fixed or drawn number of nodes and edges, and with node and edge masks: five graphs
        ///     and the generator states afterwards are Gymnasium's.
        /// </summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numEdges">The number of edges, or -1 to draw it.</param>
        /// <param name="masked">Whether to pass a node mask (one per node) and an edge mask.</param>
        [TestMethod]
        [DataRow(10L, -1L, false)]
        [DataRow(1L, -1L, false)]
        [DataRow(4L, 6L, false)]
        [DataRow(3L, 0L, false)]
        [DataRow(4L, -1L, true)]
        [DataRow(3L, 5L, true)]
        public void GraphSample_MatchesGymnasium(long numNodes, long numEdges, bool masked) {
            var spec = new GraphSpec(new DiscreteSpec(3), new DiscreteSpec(2));
            long? edges = numEdges < 0 ? null : numEdges;
            using PyRef python = PyRef.Of(() => spec.BuildPython(21));
            using var dotnet = (Graph)spec.BuildDotNet(21);
            string call = $"Graph(Discrete(3), Discrete(2)).sample(num_nodes={numNodes}, num_edges={edges?.ToString() ?? "None"}{(masked ? ", mask=..." : "")})";
            for (int i = 1; i <= 5; i++) {
                ByteContract.AssertSame($"{call} #{i}",
                    Outcome.CapturePython(() => {
                        object? mask = masked ? GraphMask(numNodes, edges is not null) : null;
                        try {
                            using PyObject? pythonMask = mask is null ? null : SpaceParity.ToPython(mask);
                            return Python.gymnasium.spaces.graph.Graph.sample(python.Object, pythonMask, null, numNodes, edges);
                        } finally {
                            Outcome.Release(mask);
                        }
                    }),
                    Outcome.Capture(() => {
                        object? mask = masked ? GraphMask(numNodes, edges is not null) : null;
                        try {
                            return dotnet.Sample(mask, null, numNodes, edges);
                        } finally {
                            Outcome.Release(mask);
                        }
                    }));
            }

            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{call} after 5 samples");
        }

        /// <summary><c>Graph.sample</c>'s argument errors are Gymnasium's.</summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="numEdges">The number of edges.</param>
        [TestMethod]
        [DataRow(0L, 2L)]
        [DataRow(-3L, 2L)]
        [DataRow(3L, -1L)]
        public void GraphSample_Errors_MatchGymnasium(long numNodes, long numEdges) {
            var spec = new GraphSpec(new BoxSpec(0.0, 1.0, new long[] { 2 }), new DiscreteSpec(2));
            using PyRef python = PyRef.Of(() => spec.BuildPython(1));
            using var dotnet = (Graph)spec.BuildDotNet(1);
            ByteContract.AssertSame($"Graph.sample(num_nodes={numNodes}, num_edges={numEdges})",
                Outcome.CapturePython(() => Python.gymnasium.spaces.graph.Graph.sample(python.Object, null, null, numNodes, numEdges)),
                Outcome.Capture(() => dotnet.Sample(null, null, numNodes, numEdges)));
        }

        /// <summary>Membership of graphs built on both sides: valid, with bad links, wrong dtype links, and without edges.</summary>
        [TestMethod]
        public void GraphContains_MatchesGymnasium() {
            var spec = new GraphSpec(new DiscreteSpec(3), new DiscreteSpec(2));
            var noEdges = new GraphSpec(new DiscreteSpec(3), null);
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using PyRef pythonNoEdges = PyRef.Of(() => noEdges.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            using Space dotnetNoEdges = noEdges.BuildDotNet(0);
            var cases = new (string Label, PyRef Python, Space DotNet, Func<NDArray> Nodes, Func<NDArray?> Edges, Func<NDArray?> Links)[] {
                ("valid", python, dotnet, () => Int64(0, 2, 1), () => Int64(1, 0), () => Int32Grid(new[] { 0, 1, 2, 2 }, 2, 2)),
                ("a link to a missing node", python, dotnet, () => Int64(0, 2, 1), () => Int64(1, 0), () => Int32Grid(new[] { 0, 3, 2, 2 }, 2, 2)),
                ("float links", python, dotnet, () => Int64(0, 2, 1), () => Int64(1, 0), () => Float64Grid(new[] { 0.0, 1.0, 2.0, 2.0 }, 2, 2)),
                ("links of the wrong shape", python, dotnet, () => Int64(0, 2, 1), () => Int64(1, 0), () => Int32Grid(new[] { 0, 1, 2 }, 3, 1)),
                ("a node out of range", python, dotnet, () => Int64(0, 5, 1), () => Int64(1, 0), () => Int32Grid(new[] { 0, 1, 2, 2 }, 2, 2)),
                ("edges without links", python, dotnet, () => Int64(0, 2, 1), () => Int64(1, 0), () => null),
                ("no edges in a graph with an edge space", python, dotnet, () => Int64(0, 2, 1), () => null, () => null),
                ("no edges in a graph without an edge space", pythonNoEdges, dotnetNoEdges, () => Int64(0, 2, 1), () => null, () => null),
                ("edges in a graph without an edge space", pythonNoEdges, dotnetNoEdges, () => Int64(0, 2, 1), () => Int64(1, 0), () => Int32Grid(new[] { 0, 1, 2, 2 }, 2, 2)),
            };
            foreach ((string label, PyRef pythonSpace, Space dotnetSpace, Func<NDArray> nodes, Func<NDArray?> edges, Func<NDArray?> links) in cases) {
                ByteContract.AssertSame($"Graph.contains: {label}",
                    Outcome.CapturePython(() => {
                        using NDArray n = nodes();
                        using NDArray? e = edges();
                        using NDArray? l = links();
                        using PyObject pythonNodes = SpaceParity.ToPython(n);
                        using PyObject pythonEdges = SpaceParity.ToPython(e);
                        using PyObject pythonLinks = SpaceParity.ToPython(l);
                        using PyObject graph = Python.gymnasium.spaces.GraphInstance(pythonNodes, pythonEdges, pythonLinks);
                        return Python.gymnasium.spaces.space.Space.contains(pythonSpace.Object, graph);
                    }),
                    Outcome.Capture(() => {
                        using var graph = new GraphInstance(nodes(), edges(), links());
                        return dotnetSpace.Contains(graph);
                    }));
            }
        }

        /// <summary>Constructor errors: Gymnasium's exception type and message for each refused argument.</summary>
        [TestMethod]
        public void ConstructorErrors_MatchGymnasium() {
            ByteContract.AssertSame("MultiBinary([2, 0])",
                Outcome.CapturePython(() => {
                    using PyObject n = SpaceParity.ToPython(new List<object?> { 2L, 0L });
                    return Python.gymnasium.spaces.MultiBinary(n);
                }),
                Outcome.Capture(() => new MultiBinary(new long[] { 2, 0 })));
            ByteContract.AssertSame("MultiBinary(-1)",
                Outcome.CapturePython(() => {
                    using PyObject n = PyConvert.ToPython(-1L);
                    return Python.gymnasium.spaces.MultiBinary(n);
                }),
                Outcome.Capture(() => new MultiBinary(-1)));
            ByteContract.AssertSame("MultiDiscrete([3, 0])",
                Outcome.CapturePython(() => {
                    using NDArray nvec = Int64(3, 0);
                    using PyObject pythonNvec = SpaceParity.ToPython(nvec);
                    return Python.gymnasium.spaces.MultiDiscrete(pythonNvec);
                }),
                Outcome.Capture(() => new MultiDiscrete(new long[] { 3, 0 })));
            ByteContract.AssertSame("Text(-1)",
                Outcome.CapturePython(() => Python.gymnasium.spaces.Text(-1)),
                Outcome.Capture(() => new Text(-1)));
            ByteContract.AssertSame("Text(3, min_length=5)",
                Outcome.CapturePython(() => Python.gymnasium.spaces.Text(3, 5)),
                Outcome.Capture(() => new Text(3, 5)));
            ByteContract.AssertSame("Text(3, min_length=-1)",
                Outcome.CapturePython(() => Python.gymnasium.spaces.Text(3, -1)),
                Outcome.Capture(() => new Text(3, -1)));
            ByteContract.AssertSame("OneOf(())",
                Outcome.CapturePython(() => {
                    using var empty = new PyTuple();
                    return Python.gymnasium.spaces.OneOf(empty);
                }),
                Outcome.Capture(() => new OneOf(Array.Empty<Space>())));
        }

        /// <summary>
        ///     <c>MultiDiscrete</c> indexing: <c>space[i]</c> of a 1-D space is a <c>Discrete</c> (its <c>repr</c> and the
        ///     generator state it starts from), of a 2-D space a row <c>MultiDiscrete</c>, and <c>len</c>.
        /// </summary>
        [TestMethod]
        public void MultiDiscreteIndexing_MatchesGymnasium() {
            var vector = new MultiDiscreteSpec(new long[] { 3, 5, 2 }, start: new long[] { 1, -2, 0 });
            var grid = new MultiDiscreteSpec(new long[] { 3, 2, 4, 5 }, new long[] { 2, 2 });
            foreach ((SpaceSpec spec, int index) in new[] { ((SpaceSpec)vector, 1), (vector, -1), (grid, 0), (grid, 1) }) {
                using PyRef python = PyRef.Of(() => spec.BuildPython(4));
                using var dotnet = (MultiDiscrete)spec.BuildDotNet(4);
                using PyRef pythonItem = PyRef.Of(() => {
                    using var position = new PyInt(index);
                    return Python.@operator.getitem(python.Object, position);
                });
                using Space dotnetItem = dotnet[index];
                ByteContract.AssertSame($"repr({spec.PythonText}[{index}])", Outcome.CapturePython(() => Python.builtins.repr(pythonItem.Object)), Outcome.Capture(() => dotnetItem.ToString()));
                ByteContract.AssertSame($"{spec.PythonText}[{index}] generator state", RngFacts.OfPythonSpace(pythonItem.Object), Outcome.Returned(RngFacts.Of(dotnetItem.NpRandom)));
                ByteContract.AssertSame($"{spec.PythonText}[{index}].sample()", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(pythonItem.Object)), Outcome.Capture(() => dotnetItem.Sample()));
                ByteContract.AssertSame($"len({spec.PythonText})", Outcome.CapturePython(() => Python.builtins.len(python.Object)), Outcome.Capture(() => (long)dotnet.Count));
            }
        }

        /// <summary>A node mask per node (a tuple, since the batched node space is a <c>MultiDiscrete</c>) and an edge mask.</summary>
        /// <param name="numNodes">The number of nodes.</param>
        /// <param name="fixedEdges">Whether the edge count is fixed, so the edge mask must be the per-edge tuple itself.</param>
        /// <returns>The <c>(node_mask, edge_mask)</c> pair.</returns>
        private static object?[] GraphMask(long numNodes, bool fixedEdges) {
            object?[] nodes = Enumerable.Range(0, (int)numNodes).Select(i => (object?)(i % 2 == 0 ? Int8(1, 0, 1) : Int8(0, 1, 0))).ToArray();
            // A drawn edge count repeats the single edge mask once per edge (Gymnasium's sample does it); a fixed count passes it
            // through as given, so it must already be the tuple of per-edge masks.
            // Each arm cast to object: with an object?[] arm and an NDArray arm, ?: would be typed NDArray (NumSharp's implicit
            // Array -> NDArray conversion) and turn the tuple of masks into an object array.
            object? edges = fixedEdges ? (object)Enumerable.Range(0, 5).Select(_ => (object?)Int8(0, 1)).ToArray() : (object)Int8(0, 1);
            return new[] { (object?)nodes, edges };
        }

        /// <summary>A Python pair whose first item is built through the tree (a NumPy scalar) and whose second is converted.</summary>
        /// <param name="first">Builds the first item.</param>
        /// <param name="second">The second item.</param>
        /// <returns>A new owned tuple.</returns>
        private static PyObject PythonPair(Func<PyObject> first, object? second) {
            using PyObject a = first();
            using PyObject b = SpaceParity.ToPython(second);
            return new PyTuple(new[] { a, b });
        }

        /// <summary>A NumPy copy of an array, disposing the .NET array.</summary>
        /// <param name="array">The array, consumed.</param>
        /// <returns>A new owned reference.</returns>
        private static PyObject Numpy(NDArray array) {
            using (array)
                return SpaceParity.ToPython(array);
        }

        /// <summary>An int8 vector.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int8(params sbyte[] values) => np.array(values);

        /// <summary>An int8 matrix.</summary>
        /// <param name="values">The values in C order.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int8Grid(sbyte[] values, long rows, long columns) {
            using NDArray flat = np.array(values);
            return flat.reshape(new Shape(rows, columns)).copy();
        }

        /// <summary>An int32 vector.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int32(params int[] values) => np.array(values);

        /// <summary>An int32 matrix.</summary>
        /// <param name="values">The values in C order.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int32Grid(int[] values, long rows, long columns) {
            using NDArray flat = np.array(values);
            return flat.reshape(new Shape(rows, columns)).copy();
        }

        /// <summary>An int64 vector.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int64(params long[] values) => np.array(values);

        /// <summary>An int64 matrix.</summary>
        /// <param name="values">The values in C order.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <returns>A new array.</returns>
        private static NDArray Int64Grid(long[] values, long rows, long columns) {
            using NDArray flat = np.array(values);
            return flat.reshape(new Shape(rows, columns)).copy();
        }

        /// <summary>A float64 vector.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Float64(params double[] values) => np.array(values);

        /// <summary>A float64 matrix.</summary>
        /// <param name="values">The values in C order.</param>
        /// <param name="rows">The rows.</param>
        /// <param name="columns">The columns.</param>
        /// <returns>A new array.</returns>
        private static NDArray Float64Grid(double[] values, long rows, long columns) {
            using NDArray flat = np.array(values);
            return flat.reshape(new Shape(rows, columns)).copy();
        }
    }
}
