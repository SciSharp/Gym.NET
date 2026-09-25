using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     Gym.NET's <see cref="Box"/> against <c>gymnasium.spaces.Box</c> (<c>gymnasium/spaces/box.py</c> @ 9e04324), live:
    ///     the bounds and their classification, <c>is_bounded</c>, seeded samples in every bound category and dtype family
    ///     with the generator state behind them, and membership.
    /// </summary>
    [TestClass]
    public sealed class BoxLiveParityTests : ParityTestBase {
        /// <summary>
        ///     The boxes compared: Gymnasium's constructor in Python spelling (with <c>{seed}</c> to fill in, for reports), the
        ///     same call through the <see cref="Python"/> tree, and the Gym.NET construction of the same box. They cover every
        ///     bound category of <c>Box.sample</c> (bounded, unbounded, low-bounded, upper-bounded, all four at once) and the
        ///     float, signed and unsigned integer dtype families. Integer boxes receive Python ints and float boxes Python
        ///     floats, as a Gymnasium user writes them.
        /// </summary>
        private static readonly (string Python, Func<long, PyObject> Gymnasium, Func<long, Box> DotNet)[] Boxes = {
            ("spaces.Box(-1.0, 1.0, (3,), np.float32, seed={seed})",
                seed => ScalarBox(-1.0, 1.0, new long[] { 3 }, "float32", seed),
                seed => new Box(-1f, 1f, new Shape(3), np.float32, seed: seed)),
            ("spaces.Box(-np.inf, np.inf, (3,), np.float64, seed={seed})",
                seed => ScalarBox(double.NegativeInfinity, double.PositiveInfinity, new long[] { 3 }, "float64", seed),
                seed => new Box(float.NegativeInfinity, float.PositiveInfinity, new Shape(3), np.float64, seed: seed)),
            ("spaces.Box(0.0, np.inf, (3,), np.float64, seed={seed})",
                seed => ScalarBox(0.0, double.PositiveInfinity, new long[] { 3 }, "float64", seed),
                seed => new Box(0f, float.PositiveInfinity, new Shape(3), np.float64, seed: seed)),
            ("spaces.Box(-np.inf, 0.0, (3,), np.float64, seed={seed})",
                seed => ScalarBox(double.NegativeInfinity, 0.0, new long[] { 3 }, "float64", seed),
                seed => new Box(float.NegativeInfinity, 0f, new Shape(3), np.float64, seed: seed)),
            ("spaces.Box(np.array([-1.0, 0.0, -np.inf, -np.inf]), np.array([1.0, np.inf, 0.0, np.inf]), dtype=np.float64, seed={seed})",
                seed => {
                    using NDArray low = MixedLow();
                    using NDArray high = MixedHigh();
                    using PyObject pythonLow = low.ToNumpyCopy();
                    using PyObject pythonHigh = high.ToNumpyCopy();
                    using PyObject float64 = Python.numpy.dtype("float64");
                    return Python.gymnasium.spaces.Box(pythonLow, pythonHigh, float64, seed);
                },
                seed => {
                    using NDArray low = MixedLow();
                    using NDArray high = MixedHigh();
                    return new Box(low, high, np.float64, seed: seed);
                }),
            ("spaces.Box(0, 5, (8,), np.int32, seed={seed})",
                seed => ScalarBox(0L, 5L, new long[] { 8 }, "int32", seed),
                seed => new Box(0, 5, new Shape(8), np.int32, seed: seed)),
            ("spaces.Box(-5, 5, (4,), np.int8, seed={seed})",
                seed => ScalarBox(-5L, 5L, new long[] { 4 }, "int8", seed),
                seed => new Box(-5, 5, new Shape(4), np.int8, seed: seed)),
            ("spaces.Box(0, 255, (2, 3), np.uint8, seed={seed})",
                seed => ScalarBox(0L, 255L, new long[] { 2, 3 }, "uint8", seed),
                seed => new Box(0, 255, new Shape(2, 3), np.uint8, seed: seed)),
        };

        /// <summary>Every box of <see cref="Boxes"/> under the seeds of A.6.</summary>
        public static IEnumerable<object[]> Cases =>
            from box in Enumerable.Range(0, Boxes.Length)
            from seed in new[] { 0L, 1L, 42L, 1000L }
            select new object[] { box, seed };

        /// <summary>
        ///     The bounds as Gymnasium stores them (<c>low</c>, <c>high</c>: dtype, shape, bytes), their classification
        ///     (<c>bounded_below</c>, <c>bounded_above</c>), the dtype and shape, and <c>is_bounded</c> in each manner.
        /// </summary>
        /// <param name="box">The index into <see cref="Boxes"/>.</param>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(4)]
        [DataRow(5)]
        [DataRow(6)]
        [DataRow(7)]
        public void Bounds_MatchGymnasium(int box) {
            (string constructor, Func<long, PyObject> gymnasium, Func<long, Box> create) = Boxes[box];
            string code = constructor.Replace("{seed}", "0");
            using PyRef python = PyRef.Of(() => gymnasium(0));
            Box dotnet = create(0);
            ByteContract.AssertSame($"{code}.low", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.low(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.Low)));
            ByteContract.AssertSame($"{code}.high", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.high(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.High)));
            ByteContract.AssertSame($"{code}.bounded_below", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.bounded_below(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.BoundedLow)));
            ByteContract.AssertSame($"{code}.bounded_above", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.bounded_above(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.BoundedHigh)));
            ByteContract.AssertSame($"{code}.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.DType)));
            ByteContract.AssertSame($"{code}.shape", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.shape(python.Object)), Outcome.Returned(DiscreteLiveParityTests.ShapeFacts(dotnet.Shape)));
            foreach ((string manner, BoundedMannerEnum value) in new[] { ("both", BoundedMannerEnum.Both), ("below", BoundedMannerEnum.Below), ("above", BoundedMannerEnum.Above) })
                ByteContract.AssertSame($"{code}.is_bounded('{manner}')", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.is_bounded(python.Object, manner)), Outcome.Capture(() => dotnet.IsBounded(value)));
        }

        /// <summary>
        ///     Five seeded samples (dtype, shape, bytes) and the generator state after them are Gymnasium's: the port of
        ///     <c>box.py:404</c> makes the same draws in the same order, casts, floors and clips alike.
        /// </summary>
        /// <param name="box">The index into <see cref="Boxes"/>.</param>
        /// <param name="seed">The box's seed.</param>
        [TestMethod]
        [DynamicData(nameof(Cases))]
        public void Sample_MatchesGymnasium(int box, long seed) {
            (string constructor, Func<long, PyObject> gymnasium, Func<long, Box> create) = Boxes[box];
            string code = constructor.Replace("{seed}", seed.ToString(CultureInfo.InvariantCulture));
            using PyRef python = PyRef.Of(() => gymnasium(seed));
            Box dotnet = create(seed);
            for (int i = 1; i <= 5; i++)
                ByteContract.AssertSame($"{code}.sample() #{i}", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => dotnet.Sample()));
            ByteContract.AssertSame($"{code} generator state after 5 samples", RngFacts.OfPythonSpace(python.Object), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
        }

        /// <summary>
        ///     Membership agrees for points of the box's own dtype: inside, on the bounds, outside, of the wrong shape, and
        ///     with a NaN; and a sample of the box is a member on both sides. Each point reaches Gymnasium as a NumPy copy of
        ///     the very array Gym.NET receives.
        /// </summary>
        [TestMethod]
        public void Contains_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => ScalarBox(-1.0, 1.0, new long[] { 3 }, "float32", 3));
            var dotnet = new Box(-1f, 1f, new Shape(3), np.float32, seed: 3);
            var candidates = new (string Python, float[] DotNet)[] {
                ("np.array([0.0, 0.5, -1.0], dtype=np.float32)", new[] { 0f, 0.5f, -1f }),
                ("np.array([1.0, 1.0, 1.0], dtype=np.float32)", new[] { 1f, 1f, 1f }),
                ("np.array([0.0, 1.5, 0.0], dtype=np.float32)", new[] { 0f, 1.5f, 0f }),
                ("np.array([0.0, 0.5], dtype=np.float32)", new[] { 0f, 0.5f }),
                ("np.array([np.nan, 0.0, 0.0], dtype=np.float32)", new[] { float.NaN, 0f, 0f }),
            };
            foreach ((string candidate, float[] values) in candidates) {
                using NDArray point = np.array(values);
                ByteContract.AssertSame($"Box(-1, 1, (3,), float32).contains({candidate})", Contains(python, point), Outcome.Capture(() => dotnet.Contains(point)));
            }

            ByteContract.AssertSame("Box(-1, 1, (3,), float32).contains(sample())", Outcome.CapturePython(() => {
                using PyObject sample = Python.gymnasium.spaces.space.Space.sample(python.Object);
                return Python.gymnasium.spaces.space.Space.contains(python.Object, sample);
            }), Outcome.Capture(() => {
                using NDArray sample = dotnet.Sample();
                return dotnet.Contains(sample);
            }));
        }

        /// <summary>
        ///     Gymnasium also requires <c>np.can_cast(x.dtype, self.dtype)</c>, so a float64 point is not a member of a float32
        ///     box even when its values are in range; Gym.NET accepts it.
        /// </summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Contains_RequiresCastableDtypes_LikeGymnasium() {
            using PyRef python = PyRef.Of(() => ScalarBox(-1.0, 1.0, new long[] { 3 }, "float32", null));
            var dotnet = new Box(-1f, 1f, new Shape(3), np.float32);
            using NDArray point = np.array(new[] { 0.0, 0.5, -1.0 });
            ByteContract.AssertSame("Box(-1, 1, (3,), float32).contains(float64 point)", Contains(python, point), Outcome.Capture(() => dotnet.Contains(point)));
        }

        /// <summary>
        ///     Gymnasium converts a point that is not an array (a list) with <c>np.asarray(x, dtype=self.dtype)</c> and answers;
        ///     Gym.NET throws <see cref="NotSupportedException"/>.
        /// </summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Contains_AcceptsAList_LikeGymnasium() {
            using PyRef python = PyRef.Of(() => ScalarBox(-1.0, 1.0, new long[] { 3 }, "float32", null));
            var dotnet = new Box(-1f, 1f, new Shape(3), np.float32);
            double[] point = { 0.0, 0.5, -1.0 };
            ByteContract.AssertSame("Box(-1, 1, (3,), float32).contains([0.0, 0.5, -1.0])", Outcome.CapturePython(() => {
                using PyObject list = PyConvert.ToList(point.Cast<object?>());
                return Python.gymnasium.spaces.space.Space.contains(python.Object, list);
            }), Outcome.Capture(() => dotnet.Contains(point)));
        }

        /// <summary>
        ///     A float64 box keeps its bounds in double precision: Gymnasium's <c>Box(-0.1, 0.1, (2,), np.float64)</c> stores
        ///     -0.1 exactly as the double -0.1. Gym.NET's scalar constructor takes <see cref="float"/> bounds, so it stores
        ///     -0.10000000149011612, and every sample shifts with it.
        /// </summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Bounds_OfAFloat64Box_KeepDoublePrecision() {
            using PyRef python = PyRef.Of(() => ScalarBox(-0.1, 0.1, new long[] { 2 }, "float64", null));
            var dotnet = new Box(-0.1f, 0.1f, new Shape(2), np.float64);
            ByteContract.AssertSame("Box(-0.1, 0.1, (2,), float64).low", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.low(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.Low)));
        }

        /// <summary><c>spaces.Box(low, high, shape, np.dtype(dtype), seed=seed)</c> with scalar bounds. Must run under the GIL.</summary>
        /// <param name="low">The lower bound: a <see langword="double"/> for a Python float, a <see langword="long"/> for a Python int.</param>
        /// <param name="high">The upper bound, likewise.</param>
        /// <param name="shape">The shape.</param>
        /// <param name="dtype">The NumPy dtype's name.</param>
        /// <param name="seed">The seed, or <see langword="null"/> for an unseeded box.</param>
        /// <returns>A new owned reference to the box.</returns>
        /// <exception cref="PythonException">Thrown when the constructor raises.</exception>
        private static PyObject ScalarBox(object low, object high, long[] shape, string dtype, long? seed) {
            using PyObject type = Python.numpy.dtype(dtype);
            return Python.gymnasium.spaces.Box(low, high, shape, type, seed);
        }

        /// <summary>The low bounds of the mixed box: one of each bound category.</summary>
        /// <returns>A new float64 array the caller disposes.</returns>
        private static NDArray MixedLow() => np.array(new[] { -1.0, 0.0, double.NegativeInfinity, double.NegativeInfinity });

        /// <summary>The high bounds of the mixed box.</summary>
        /// <returns>A new float64 array the caller disposes.</returns>
        private static NDArray MixedHigh() => np.array(new[] { 1.0, double.PositiveInfinity, 0.0, double.PositiveInfinity });

        /// <summary><c>space.contains(point)</c> on Gymnasium's side, with the point crossed as a NumPy copy of the .NET one.</summary>
        /// <param name="space">The box.</param>
        /// <param name="point">The point both sides test.</param>
        /// <returns>The described answer, or what the check raised.</returns>
        private static Outcome Contains(PyRef space, NDArray point) {
            return Outcome.CapturePython(() => {
                using PyObject pythonPoint = point.ToNumpyCopy();
                return Python.gymnasium.spaces.space.Space.contains(space.Object, pythonPoint);
            });
        }
    }
}
