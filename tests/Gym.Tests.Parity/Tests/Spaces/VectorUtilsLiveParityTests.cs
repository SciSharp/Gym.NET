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
    ///     Gym.NET's <see cref="VectorUtils"/> against <c>gymnasium.vector.utils</c>, live: <c>create_empty_array</c>,
    ///     <c>concatenate</c> of seeded samples into it, <c>iterate</c> over a batched sample, and
    ///     <c>batch_differing_spaces</c> with its errors, over the space catalog of <see cref="SpaceSpecLiveParityTests"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/utils/space_utils.py">gymnasium/vector/utils/space_utils.py</see>.</remarks>
    [TestClass]
    public sealed class VectorUtilsLiveParityTests : ParityTestBase {
        /// <summary>Every space of the catalog, by name, plus <c>Discrete(3)</c> and a box.</summary>
        public static IEnumerable<object[]> Spaces =>
            SpaceSpecLiveParityTests.Catalog.Keys.Concat(Extra.Keys).Select(name => new object[] { name });

        /// <summary>The simple spaces the catalog leaves to their own cells, needed here as batch members.</summary>
        private static readonly Dictionary<string, SpaceSpec> Extra = new SpaceSpec[] {
            new DiscreteSpec(3, start: 2),
            new BoxSpec(-2.0, 2.0, new long[] { 2, 2 }, "float64"),
        }.ToDictionary(spec => spec.PythonText, StringComparer.Ordinal);

        /// <summary><c>create_empty_array(space, 3)</c> is Gymnasium's: zeros of the batched shape, or the placeholder tuples.</summary>
        /// <param name="name">The space.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void CreateEmptyArray_MatchesGymnasium(string name) {
            SpaceSpec spec = Spec(name);
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            ByteContract.AssertSame($"create_empty_array({name}, 3)",
                Outcome.CapturePython(() => Python.gymnasium.vector.utils.create_empty_array(python.Object, 3)),
                Outcome.Capture(() => VectorUtils.CreateEmptyArray(dotnet, 3)));
        }

        /// <summary>
        ///     <c>concatenate(space, samples, create_empty_array(space, 3))</c> of three seeded samples is Gymnasium's batch.
        /// </summary>
        /// <param name="name">The space.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void Concatenate_MatchesGymnasium(string name) {
            SpaceSpec spec = Spec(name);
            using PyRef python = PyRef.Of(() => spec.BuildPython(8));
            using Space dotnet = spec.BuildDotNet(8);
            ByteContract.AssertSame($"concatenate({name}, 3 samples, create_empty_array(...))",
                Outcome.CapturePython(() => {
                    using var samples = new PyList();
                    for (int i = 0; i < 3; i++) {
                        using PyObject sample = Python.gymnasium.spaces.space.Space.sample(python.Object);
                        samples.Append(sample);
                    }

                    using PyObject empty = Python.gymnasium.vector.utils.create_empty_array(python.Object, 3);
                    return Python.gymnasium.vector.utils.concatenate(python.Object, samples, empty);
                }),
                Outcome.Capture(() => {
                    object?[] samples = Enumerable.Range(0, 3).Select(_ => dotnet.Sample()).ToArray();
                    object? empty = VectorUtils.CreateEmptyArray(dotnet, 3);
                    object? batch = VectorUtils.Concatenate(dotnet, samples, empty);
                    // The samples were copied into the batch (or are held by it, for tuple-batched spaces): release what the
                    // batch does not hold, the per-env arrays of an array-batched space.
                    if (batch is not object?[])
                        Outcome.Release(samples);
                    return batch;
                }));
        }

        /// <summary>
        ///     <c>list(iterate(batch_space(space, 3), batch_space(space, 3).sample()))</c> is Gymnasium's per-env items, or
        ///     its error (a batched <c>Discrete</c> is a <c>MultiDiscrete</c>, so only a bare <c>Discrete</c> refuses).
        /// </summary>
        /// <param name="name">The space.</param>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void Iterate_MatchesGymnasium(string name) {
            SpaceSpec spec = Spec(name);
            using PyRef python = PyRef.Of(() => spec.BuildPython(13));
            using Space dotnet = spec.BuildDotNet(13);
            using PyRef pythonBatch = PyRef.Of(() => Python.gymnasium.vector.utils.batch_space(python.Object, 3));
            using Space dotnetBatch = VectorUtils.BatchSpace(dotnet, 3);
            Outcome pythonItems = Outcome.CapturePython(() => {
                using PyObject sample = Python.gymnasium.spaces.space.Space.sample(pythonBatch.Object);
                using PyObject iterator = Python.gymnasium.vector.utils.iterate(pythonBatch.Object, sample);
                return Python.builtins.list(iterator);
            });
            object? batchSample = dotnetBatch.Sample();
            List<object?>? items = null;
            // Described inside Capture while the batch sample is alive: the items of an array batch are views into it, so
            // they are released (views first) only afterwards.
            Outcome dotnetItems = Outcome.Capture(() => items = VectorUtils.Iterate(dotnetBatch, batchSample).ToList(), disposeResult: false);
            Outcome.Release(items);
            Outcome.Release(batchSample);
            ByteContract.AssertSame($"list(iterate(batch_space({name}, 3), sample))", pythonItems, dotnetItems);
        }

        /// <summary>
        ///     <c>create_shared_memory(space, n=3)</c>, three seeded samples written with <c>write_to_shared_memory</c> into slots
        ///     0 to 2, then <c>read_from_shared_memory(space, memory, n=3)</c>: Gymnasium's batch (views of the buffer, decoded
        ///     texts, <c>OneOf</c>'s index and sample pairs), or its refusal (a dynamic shape's <c>TypeError</c>).
        /// </summary>
        /// <param name="name">The space.</param>
        /// <remarks>
        ///     Gymnasium's buffer is a <c>multiprocessing</c> array and Gym.NET's an ordinary one, so the cell compares what
        ///     reading gives after the writes: the layout, the casts and the decoding, which the async vector env relies on.
        /// </remarks>
        [TestMethod]
        [DynamicData(nameof(Spaces))]
        public void SharedMemory_MatchesGymnasium(string name) {
            SpaceSpec spec = Spec(name);
            using PyRef python = PyRef.Of(() => spec.BuildPython(17));
            using Space dotnet = spec.BuildDotNet(17);
            Outcome pythonRead = Outcome.CapturePython(() => {
                using PyObject memory = Python.gymnasium.vector.utils.create_shared_memory(python.Object, 3);
                for (int i = 0; i < 3; i++) {
                    using PyObject sample = Python.gymnasium.spaces.space.Space.sample(python.Object);
                    using PyObject _ = Python.gymnasium.vector.utils.write_to_shared_memory(python.Object, i, sample, memory);
                }

                return Python.gymnasium.vector.utils.read_from_shared_memory(python.Object, memory, 3);
            });
            object? memory = null;
            Outcome dotnetRead = Outcome.Capture(() => {
                memory = VectorUtils.CreateSharedMemory(dotnet, 3);
                for (int i = 0; i < 3; i++) {
                    object? sample = dotnet.Sample();
                    VectorUtils.WriteToSharedMemory(dotnet, i, sample, memory);
                    // The write copied the sample into the buffer.
                    Outcome.Release(sample);
                }

                return VectorUtils.ReadFromSharedMemory(dotnet, memory, 3);
            });
            // The read's views were described and released by Capture; the buffer goes after them.
            VectorUtils.ReleaseSharedMemory(memory);
            ByteContract.AssertSame($"read_from_shared_memory({name}) after three writes", pythonRead, dotnetRead);
        }

        /// <summary><c>iterate</c> refuses a bare <c>Discrete</c> and a custom space as Gymnasium does.</summary>
        [TestMethod]
        public void Iterate_Refusals_MatchGymnasium() {
            var discrete = new DiscreteSpec(3);
            using PyRef python = PyRef.Of(() => discrete.BuildPython(0));
            using Space dotnet = discrete.BuildDotNet(0);
            ByteContract.AssertSame("iterate(Discrete(3), [1, 2])",
                Outcome.CapturePython(() => {
                    using PyObject items = SpaceParity.ToPython(new List<object?> { 1L, 2L });
                    using PyObject iterator = Python.gymnasium.vector.utils.iterate(python.Object, items);
                    return Python.builtins.list(iterator);
                }),
                Outcome.Capture(() => VectorUtils.Iterate(dotnet, new List<object?> { 1L, 2L }).ToList()));

            var text = new TextSpec(4);
            using PyRef pythonText = PyRef.Of(() => text.BuildPython(0));
            using Space dotnetText = text.BuildDotNet(0);
            ByteContract.AssertSame("iterate(Text(4), ['a'])",
                Outcome.CapturePython(() => {
                    using PyObject items = SpaceParity.ToPython(new List<object?> { "a" });
                    using PyObject iterator = Python.gymnasium.vector.utils.iterate(pythonText.Object, items);
                    return Python.builtins.list(iterator);
                }),
                Outcome.Capture(() => VectorUtils.Iterate(dotnetText, new List<object?> { "a" }).ToList()));

            var box = new BoxSpec(0.0, 1.0, new long[] { 2 });
            using PyRef pythonBox = PyRef.Of(() => box.BuildPython(0));
            using Space dotnetBox = box.BuildDotNet(0);
            ByteContract.AssertSame("iterate(Box(0.0, 1.0, (2,), float32), 5)",
                Outcome.CapturePython(() => {
                    using PyObject items = PyConvert.ToPython(5L);
                    using PyObject iterator = Python.gymnasium.vector.utils.iterate(pythonBox.Object, items);
                    return Python.builtins.list(iterator);
                }),
                Outcome.Capture(() => VectorUtils.Iterate(dotnetBox, 5L).ToList()));
        }

        /// <summary>Lists of spaces <c>batch_differing_spaces</c> takes or refuses, each space seeded with its position.</summary>
        private static readonly Dictionary<string, SpaceSpec[]> DifferingCases = new() {
            ["Discrete"] = new SpaceSpec[] { new DiscreteSpec(3), new DiscreteSpec(5), new DiscreteSpec(4, start: 1) },
            ["Box"] = new SpaceSpec[] { new BoxSpec(0.0, 1.0, new long[] { 2 }), new BoxSpec(-1.0, 2.0, new long[] { 2 }) },
            ["MultiDiscrete"] = new SpaceSpec[] { new MultiDiscreteSpec(new long[] { 2, 3 }), new MultiDiscreteSpec(new long[] { 4, 5 }) },
            ["MultiBinary"] = new SpaceSpec[] { new MultiBinarySpec(3), new MultiBinarySpec(3) },
            ["Tuple"] = new SpaceSpec[] { new TupleSpec(new DiscreteSpec(2), new DiscreteSpec(3)), new TupleSpec(new DiscreteSpec(4), new DiscreteSpec(5)) },
            ["Dict"] = new SpaceSpec[] {
                new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(2)), ("b", new BoxSpec(0.0, 1.0, new long[] { 1 })) }),
                new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(6)), ("b", new BoxSpec(0.0, 3.0, new long[] { 1 })) }),
            },
            ["Text"] = new SpaceSpec[] { new TextSpec(4), new TextSpec(6, 2, "xyz") },
            ["empty list"] = Array.Empty<SpaceSpec>(),
            ["mixed types"] = new SpaceSpec[] { new DiscreteSpec(2), new BoxSpec(0.0, 1.0, new long[] { 2 }) },
            ["Box shapes differ"] = new SpaceSpec[] { new BoxSpec(0.0, 1.0, new long[] { 2 }), new BoxSpec(0.0, 1.0, new long[] { 3 }) },
            ["Box dtypes differ"] = new SpaceSpec[] { new BoxSpec(0.0, 1.0, new long[] { 2 }), new BoxSpec(0.0, 1.0, new long[] { 2 }, "float64") },
            ["MultiBinary shapes differ"] = new SpaceSpec[] { new MultiBinarySpec(3), new MultiBinarySpec(4) },
            ["Dict keys differ"] = new SpaceSpec[] {
                new DictSpec(new[] { ("a", (SpaceSpec)new DiscreteSpec(2)) }),
                new DictSpec(new[] { ("b", (SpaceSpec)new DiscreteSpec(2)) }),
            },
        };

        /// <summary>The differing-space cases, by name.</summary>
        public static IEnumerable<object[]> DifferingCaseNames => DifferingCases.Keys.Select(k => new object[] { k });

        /// <summary>
        ///     <c>batch_differing_spaces(spaces)</c> is Gymnasium's: the batched space's <c>repr</c> and three samples from it
        ///     (it starts from a copy of the first space's generator), or the error it raises.
        /// </summary>
        /// <param name="name">The case.</param>
        [TestMethod]
        [DynamicData(nameof(DifferingCaseNames))]
        public void BatchDifferingSpaces_MatchesGymnasium(string name) {
            SpaceSpec[] specs = DifferingCases[name];
            using PyRef pythonSpaces = PyRef.Of(() => {
                var list = new PyList();
                for (int i = 0; i < specs.Length; i++) {
                    using PyObject space = specs[i].BuildPython(i + 1);
                    list.Append(space);
                }

                return list;
            });
            Space[] dotnetSpaces = specs.Select((spec, i) => spec.BuildDotNet(i + 1)).ToArray();
            try {
                Outcome pythonBatch = Outcome.CapturePython(() => {
                    using PyObject batch = Python.gymnasium.vector.utils.batch_differing_spaces(pythonSpaces.Object);
                    return Python.builtins.repr(batch);
                });
                Outcome dotnetBatch = Outcome.Capture(() => {
                    using Space batch = VectorUtils.BatchDifferingSpaces(dotnetSpaces);
                    return batch.ToString();
                });
                ByteContract.AssertSame($"repr(batch_differing_spaces({name}))", pythonBatch, dotnetBatch);
                if (pythonBatch.Raised != null)
                    return;

                using PyRef python = PyRef.Of(() => Python.gymnasium.vector.utils.batch_differing_spaces(pythonSpaces.Object));
                using Space dotnet = VectorUtils.BatchDifferingSpaces(dotnetSpaces);
                for (int i = 1; i <= 3; i++)
                    ByteContract.AssertSame($"batch_differing_spaces({name}).sample() #{i}", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => dotnet.Sample()));
            } finally {
                foreach (Space space in dotnetSpaces)
                    space.Dispose();
            }
        }

        /// <summary>A spec of the catalog or of <see cref="Extra"/>.</summary>
        /// <param name="name">The Python spelling.</param>
        /// <returns>The spec.</returns>
        private static SpaceSpec Spec(string name) => SpaceSpecLiveParityTests.Catalog.TryGetValue(name, out SpaceSpec? spec) ? spec : Extra[name];
    }
}
