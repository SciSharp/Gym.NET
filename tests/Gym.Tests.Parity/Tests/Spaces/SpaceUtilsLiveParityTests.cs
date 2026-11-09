using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     Gym.NET's <see cref="SpaceUtils"/> against <c>gymnasium.spaces.utils</c>, live, over the space catalog: <c>flatdim</c>,
    ///     <c>flatten_space</c>, <c>flatten</c> of seeded samples, <c>unflatten</c> of what <c>flatten</c> gave, membership of
    ///     the flattened sample in the flattened space, and <c>is_space_dtype_shape_equiv</c>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/utils.py">gymnasium/spaces/utils.py</see>.</remarks>
    [TestClass]
    public sealed class SpaceUtilsLiveParityTests : ParityTestBase {
        /// <summary>The spaces of the catalog, plus the simple ones and the flattening corner cases.</summary>
        private static readonly Dictionary<string, SpaceSpec> Spaces = SpaceSpecLiveParityTests.Catalog.Values.Concat(new SpaceSpec[] {
            new DiscreteSpec(3, start: 2),
            new BoxSpec(-2.0, 2.0, new long[] { 2, 2 }, "float64"),
            new BoxSpec(0.0, 9.0, new long[] { 3 }, "int32"),
            new TupleSpec(new DiscreteSpec(2), new BoxSpec(-1.0, 1.0, new long[] { 2 }), new TextSpec(3, 1, "ab")),
            new DictSpec(new[] { ("x", (SpaceSpec)new MultiBinarySpec(2)), ("y", new DiscreteSpec(3)) }),
            new TupleSpec(new DiscreteSpec(2), new SequenceSpec(new DiscreteSpec(3))),
            new DictSpec(new[] { ("g", (SpaceSpec)new GraphSpec(new DiscreteSpec(2), null)), ("d", new DiscreteSpec(2)) }),
            new OneOfSpec(new MultiBinarySpec(2), new MultiBinarySpec(3)),
            new OneOfSpec(new BoxSpec(0.0, 1.0, new long[] { 2 }, "uint8"), new DiscreteSpec(300)),
            new OneOfSpec(new TextSpec(2), new BoxSpec(-5.0, 5.0, new long[] { 3 }, "float64")),
        }).GroupBy(spec => spec.PythonText, StringComparer.Ordinal).ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);

        /// <summary>Every space, by name.</summary>
        public static IEnumerable<object[]> SpaceNames => Spaces.Keys.Select(name => new object[] { name });

        /// <summary>A spec's space type, the class name its Python spelling starts with.</summary>
        /// <param name="spec">The spec.</param>
        /// <returns>The type name (<c>Box</c>, <c>Tuple</c>).</returns>
        private static string Kind(SpaceSpec spec) => spec.PythonText.Substring(0, spec.PythonText.IndexOf('('));

        /// <summary><c>flatdim(space)</c> and <c>flatten_space(space)</c> (its <c>repr</c>) are Gymnasium's, values or errors.</summary>
        /// <param name="name">The space.</param>
        [TestMethod]
        [DynamicData(nameof(SpaceNames))]
        public void FlatdimAndFlattenSpace_MatchGymnasium(string name) {
            SpaceSpec spec = Spaces[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            ByteContract.AssertSame($"flatdim({name})", Outcome.CapturePython(() => Python.gymnasium.spaces.utils.flatdim(python.Object)), Outcome.Capture(() => SpaceUtils.Flatdim(dotnet)));
            ByteContract.AssertSame($"repr(flatten_space({name}))",
                Outcome.CapturePython(() => {
                    using PyObject flat = Python.gymnasium.spaces.utils.flatten_space(python.Object);
                    return Python.builtins.repr(flat);
                }),
                Outcome.Capture(() => {
                    using Space flat = SpaceUtils.FlattenSpace(dotnet);
                    return flat.ToString();
                }));
        }

        /// <summary>
        ///     For three seeded samples: <c>flatten(space, sample)</c>, membership of that in <c>flatten_space(space)</c>, and
        ///     <c>unflatten(space, flatten(space, sample))</c> are Gymnasium's.
        /// </summary>
        /// <param name="name">The space.</param>
        [TestMethod]
        [DynamicData(nameof(SpaceNames))]
        public void FlattenRoundTrip_MatchesGymnasium(string name) {
            SpaceSpec spec = Spaces[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(17));
            using Space dotnet = spec.BuildDotNet(17);
            using PyRef pythonFlatSpace = PyRef.Of(() => Python.gymnasium.spaces.utils.flatten_space(python.Object));
            using Space dotnetFlatSpace = SpaceUtils.FlattenSpace(dotnet);
            for (int i = 1; i <= 3; i++) {
                using PyRef pythonSample = PyRef.Of(() => Python.gymnasium.spaces.space.Space.sample(python.Object));
                object? dotnetSample = dotnet.Sample();
                try {
                    ByteContract.AssertSame($"flatten({name}, sample #{i})",
                        Outcome.CapturePython(() => Python.gymnasium.spaces.utils.flatten(python.Object, pythonSample.Object)),
                        Outcome.Capture(() => SpaceUtils.Flatten(dotnet, dotnetSample)));
                    ByteContract.AssertSame($"flatten({name}, sample #{i}) in flatten_space({name})",
                        Outcome.CapturePython(() => {
                            using PyObject flat = Python.gymnasium.spaces.utils.flatten(python.Object, pythonSample.Object);
                            return Python.gymnasium.spaces.space.Space.contains(pythonFlatSpace.Object, flat);
                        }),
                        Outcome.Capture(() => {
                            object? flat = SpaceUtils.Flatten(dotnet, dotnetSample);
                            try {
                                return dotnetFlatSpace.Contains(flat);
                            } finally {
                                Outcome.Release(flat);
                            }
                        }));
                    ByteContract.AssertSame($"unflatten({name}, flatten({name}, sample #{i}))",
                        Outcome.CapturePython(() => {
                            using PyObject flat = Python.gymnasium.spaces.utils.flatten(python.Object, pythonSample.Object);
                            return Python.gymnasium.spaces.utils.unflatten(python.Object, flat);
                        }),
                        Outcome.Capture(() => {
                            object? flat = SpaceUtils.Flatten(dotnet, dotnetSample);
                            try {
                                return SpaceUtils.Unflatten(dotnet, flat);
                            } finally {
                                Outcome.Release(flat);
                            }
                        }));
                } finally {
                    Outcome.Release(dotnetSample);
                }
            }
        }

        /// <summary>
        ///     <c>is_space_dtype_shape_equiv</c> of a space with a twin built from the same spec (true), with every space of
        ///     its type in the catalog, and with one space of each other type (true or false, or the error Gymnasium raises),
        ///     is Gymnasium's.
        /// </summary>
        /// <param name="name">The first space.</param>
        [TestMethod]
        [DynamicData(nameof(SpaceNames))]
        public void IsSpaceDtypeShapeEquiv_MatchesGymnasium(string name) {
            SpaceSpec spec = Spaces[name];
            using PyRef python = PyRef.Of(() => spec.BuildPython(0));
            using Space dotnet = spec.BuildDotNet(0);
            // Same-type pairs are where the function compares anything; one space of each other type checks the refusals.
            string kind = Kind(spec);
            IEnumerable<SpaceSpec> others = Spaces.Values.Where(o => Kind(o) == kind)
                .Concat(Spaces.Values.GroupBy(Kind).Where(g => g.Key != kind).Select(g => g.First()));
            foreach (SpaceSpec other in others) {
                using PyRef pythonOther = PyRef.Of(() => other.BuildPython(1));
                using Space dotnetOther = other.BuildDotNet(1);
                ByteContract.AssertSame($"is_space_dtype_shape_equiv({name}, {other.PythonText})",
                    Outcome.CapturePython(() => Python.gymnasium.spaces.utils.is_space_dtype_shape_equiv(python.Object, pythonOther.Object)),
                    Outcome.Capture(() => SpaceUtils.IsSpaceDtypeShapeEquiv(dotnet, dotnetOther)));
            }
        }
    }
}
