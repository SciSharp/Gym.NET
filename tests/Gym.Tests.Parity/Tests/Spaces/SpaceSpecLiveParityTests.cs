using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     Every space of the catalog against Gymnasium's, live, through the same generic checks: <c>repr</c>, seeded samples
    ///     with the generator state of the space and of every subspace after them, what <c>seed(int)</c> returns and the
    ///     stream it leaves, membership of the space's own samples, <c>is_np_flattenable</c>, the JSON round trip, equality,
    ///     and <c>batch_space</c>'s space and samples.
    /// </summary>
    /// <remarks>
    ///     The catalog (<see cref="Catalog"/>) covers each composite alone, nested, and over the simple spaces, so a
    ///     divergence names the smallest space that shows it. Cells per space (masks, errors, membership matrices) live in
    ///     the per-space classes.
    /// </remarks>
    [TestClass]
    public sealed class SpaceSpecLiveParityTests : ParityTestBase {
        /// <summary>
        ///     The spaces every cell runs over, by their Python spelling: this suite's own 24 and Gymnasium's test catalog
        ///     (<see cref="GymnasiumTestingSpaces"/>); the vector-utils and spaces.utils cells reuse it.
        /// </summary>
        internal static readonly Dictionary<string, SpaceSpec> Catalog = new SpaceSpec[] {
            new MultiBinarySpec(5),
            new MultiBinarySpec(new long[] { 2, 3 }),
            new MultiDiscreteSpec(new long[] { 3, 5, 2 }),
            new MultiDiscreteSpec(new long[] { 3, 2, 4, 5 }, new long[] { 2, 2 }),
            new MultiDiscreteSpec(new long[] { 3, 5 }, start: new long[] { -2, 4 }),
            new MultiDiscreteSpec(new long[] { 3, 5 }, dtype: "int32"),
            new TupleSpec(new DiscreteSpec(3), new BoxSpec(-1.0, 1.0, new long[] { 2 })),
            new TupleSpec(new DiscreteSpec(2), new TupleSpec(new MultiBinarySpec(3), new DiscreteSpec(4, start: -1))),
            new DictSpec(new[] { ("b", (SpaceSpec)new DiscreteSpec(3)), ("a", new BoxSpec(0.0, 1.0, new long[] { 2 })) }),
            new DictSpec(new[] { ("b", (SpaceSpec)new DiscreteSpec(3)), ("a", new BoxSpec(0.0, 1.0, new long[] { 2 })) }, sortKeys: false),
            new DictSpec(new[] { ("b", (SpaceSpec)new DiscreteSpec(3)), ("a", new MultiBinarySpec(2)) }, mapping: false),
            new DictSpec(new[] { ("obs", (SpaceSpec)new TupleSpec(new DiscreteSpec(2), new DiscreteSpec(3))), ("goal", new MultiDiscreteSpec(new long[] { 4, 4 })) }),
            new TextSpec(5),
            new TextSpec(8, 2, "abc"),
            new TextSpec(3, 0, "01"),
            new SequenceSpec(new DiscreteSpec(4)),
            new SequenceSpec(new BoxSpec(0.0, 1.0, new long[] { 2 }), stack: true),
            new SequenceSpec(new DiscreteSpec(3), stack: true),
            new SequenceSpec(new TupleSpec(new DiscreteSpec(2), new BoxSpec(0.0, 1.0, new long[] { 1 }))),
            new GraphSpec(new BoxSpec(-1.0, 1.0, new long[] { 3 }), new DiscreteSpec(3)),
            new GraphSpec(new DiscreteSpec(4), null),
            new GraphSpec(new BoxSpec(0.0, 1.0, new long[] { 2 }), new BoxSpec(0.0, 1.0, new long[] { 1 })),
            new OneOfSpec(new DiscreteSpec(2), new BoxSpec(-1.0, 1.0, new long[] { 2 })),
            new OneOfSpec(new DiscreteSpec(3), new TextSpec(4), new MultiBinarySpec(2)),
        }.Concat(GymnasiumTestingSpaces.Fundamental).Concat(GymnasiumTestingSpaces.Composite)
         // Gymnasium's catalog repeats a few of the spaces above; one cell per spelling is enough.
         .GroupBy(spec => spec.PythonText, StringComparer.Ordinal)
         .ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);

        /// <summary>Every space of the catalog, by name.</summary>
        public static IEnumerable<object[]> Spaces => Catalog.Keys.Select(name => new object[] { name });

        /// <summary>Every space of the catalog with two seeds.</summary>
        public static IEnumerable<object[]> SeededSpaces => Catalog.Keys.SelectMany(name => new[] { new object[] { name, 0L }, new object[] { name, 42L } });

        /// <summary>Gymnasium's <c>repr(space)</c> is Gym.NET's <c>ToString()</c>.</summary>
        /// <param name="name">The space, by its Python spelling.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void Repr_MatchesGymnasium(string name) {
            SpaceSpec spec = Catalog[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            ByteContract.AssertSame($"repr({name})", Outcome.CapturePython(() => Python.builtins.repr(python.Object)), Outcome.Capture(() => dotnet.ToString()));
        }

        /// <summary>
        ///     Five seeded samples are Gymnasium's, value for value, and so is the generator state of the space and of each
        ///     subspace afterwards.
        /// </summary>
        /// <param name="name">The space, by its Python spelling.</param>
        /// <param name="seed">The constructor's seed.</param>
        [TestMethod]
        [DynamicData(nameof(SeededSpaces))]
        public void Samples_MatchGymnasium(string name, long seed) {
            SpaceSpec spec = Catalog[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(seed));
            using Space dotnet = spec.BuildDotNet(seed);
            for (int i = 1; i <= 5; i++)
                ByteContract.AssertSame($"{name} (seed={seed}).sample() #{i}", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => dotnet.Sample()));
            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{name} (seed={seed}) after 5 samples");
        }

        /// <summary>
        ///     <c>space.seed(seed)</c> on an unseeded space returns Gymnasium's seeds (an int, a tuple per subspace, a dict per
        ///     key) and leaves the streams Gymnasium's: three samples and every generator state agree.
        /// </summary>
        /// <param name="name">The space, by its Python spelling.</param>
        /// <param name="seed">The seed.</param>
        [TestMethod]
        [DynamicData(nameof(SeededSpaces))]
        public void Seed_ReturnsAndLeavesGymnasiumsSeeds(string name, long seed) {
            SpaceSpec spec = Catalog[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(null));
            using Space dotnet = spec.BuildDotNet(null);
            ByteContract.AssertSame($"{name}.seed({seed})", Outcome.CapturePython(() => {
                using var value = new PyInt(seed);
                return Python.gymnasium.spaces.space.Space.seed(python.Object, value);
            }), Outcome.Capture(() => dotnet.Seed(seed)));
            for (int i = 1; i <= 3; i++)
                ByteContract.AssertSame($"{name} after seed({seed}): sample() #{i}", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => dotnet.Sample()));
            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{name} after seed({seed}) and 3 samples");
        }

        /// <summary>Each side's own samples are members of its space, and <c>is_np_flattenable</c> agrees.</summary>
        /// <param name="name">The space, by its Python spelling.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void OwnSamples_AreMembers(string name) {
            SpaceSpec spec = Catalog[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(3));
            using Space dotnet = spec.BuildDotNet(3);
            for (int i = 1; i <= 3; i++) {
                ByteContract.AssertSame($"{name}: sample #{i} in space", Outcome.CapturePython(() => {
                    using PyObject sample = Python.gymnasium.spaces.space.Space.sample(python.Object);
                    return Python.gymnasium.spaces.space.Space.contains(python.Object, sample);
                }), Outcome.Capture(() => {
                    object? sample = dotnet.Sample();
                    try {
                        return dotnet.Contains(sample);
                    } finally {
                        Outcome.Release(sample);
                    }
                }));
            }

            ByteContract.AssertSame($"{name}.is_np_flattenable", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.is_np_flattenable(python.Object)), Outcome.Capture(() => dotnet.IsNpFlattenable));
        }

        /// <summary>
        ///     <c>to_jsonable</c> of three samples is Gymnasium's (Python floats, ints, strings, nested lists and dicts), and so
        ///     is <c>from_jsonable</c> of that JSON.
        /// </summary>
        /// <param name="name">The space, by its Python spelling.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void Jsonable_RoundTrip_MatchesGymnasium(string name) {
            SpaceSpec spec = Catalog[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(11));
            using Space dotnet = spec.BuildDotNet(11);
            using PyRef pythonSamples = PyRef.Of(() => {
                var list = new PyList();
                for (int i = 0; i < 3; i++) {
                    using PyObject sample = Python.gymnasium.spaces.space.Space.sample(python.Object);
                    list.Append(sample);
                }

                return list;
            });
            List<object?> dotnetSamples = Enumerable.Range(0, 3).Select(_ => dotnet.Sample()).ToList();
            try {
                ByteContract.AssertSame($"{name}.to_jsonable(3 samples)",
                    Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.to_jsonable(python.Object, pythonSamples.Object)),
                    Outcome.Capture(() => dotnet.ToJsonable(dotnetSamples)));
                ByteContract.AssertSame($"{name}.from_jsonable(to_jsonable(3 samples))",
                    Outcome.CapturePython(() => {
                        using PyObject json = Python.gymnasium.spaces.space.Space.to_jsonable(python.Object, pythonSamples.Object);
                        return Python.gymnasium.spaces.space.Space.from_jsonable(python.Object, json);
                    }),
                    Outcome.Capture(() => dotnet.FromJsonable(dotnet.ToJsonable(dotnetSamples))));
            } finally {
                Outcome.Release(dotnetSamples);
            }
        }

        /// <summary>A space equals a second one built from the same spec, on both sides, and not a space of another spec.</summary>
        /// <param name="name">The space, by its Python spelling.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void Equality_MatchesGymnasium(string name) {
            SpaceSpec spec = Catalog[name];
            SpaceSpec other = Catalog.Values.First(s => s.PythonText != name);
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using PyRef pythonTwin = PyRef.Of(() => spec.BuildPython(1));
            using PyRef pythonOther = PyRef.Of(() => other.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            using Space dotnetTwin = spec.BuildDotNet(1);
            using Space dotnetOther = other.BuildDotNet(0);
            ByteContract.AssertSame($"{name} == {name}", Outcome.CapturePython(() => Python.@operator.eq(python.Object, pythonTwin.Object)), Outcome.Capture(() => dotnet.Equals(dotnetTwin)));
            ByteContract.AssertSame($"{name} == {other.PythonText}", Outcome.CapturePython(() => Python.@operator.eq(python.Object, pythonOther.Object)), Outcome.Capture(() => dotnet.Equals(dotnetOther)));
        }

        /// <summary>
        ///     <c>batch_space(space, 3)</c> is Gymnasium's: its <c>repr</c>, three samples and its generator state, and the
        ///     original space's state after batching (batching copies generators, so it must not draw from the original's).
        /// </summary>
        /// <param name="name">The space, by its Python spelling.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void BatchSpace_MatchesGymnasium(string name) {
            SpaceSpec spec = Catalog[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(5));
            using Space dotnet = spec.BuildDotNet(5);
            using PyRef pythonBatch = PyRef.Of(() => Python.gymnasium.vector.utils.batch_space(python.Object, 3));
            using Space dotnetBatch = VectorUtils.BatchSpace(dotnet, 3);
            ByteContract.AssertSame($"repr(batch_space({name}, 3))", Outcome.CapturePython(() => Python.builtins.repr(pythonBatch.Object)), Outcome.Capture(() => dotnetBatch.ToString()));
            for (int i = 1; i <= 3; i++)
                ByteContract.AssertSame($"batch_space({name}, 3).sample() #{i}", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(pythonBatch.Object)), Outcome.Capture(() => dotnetBatch.Sample()));
            ByteContract.AssertSame($"batch_space({name}, 3) generator state", RngFacts.OfPythonSpace(pythonBatch.Object), Outcome.Returned(RngFacts.Of(dotnetBatch.NpRandom)));
            SpaceParity.AssertGeneratorStates(spec, python, dotnet, $"{name} after batch_space and sampling the batch");
        }
    }
}
