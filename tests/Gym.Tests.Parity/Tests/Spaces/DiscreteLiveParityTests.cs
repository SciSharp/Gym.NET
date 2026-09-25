using System;
using System.Collections.Generic;
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
    ///     Gym.NET's <see cref="Discrete"/> against <c>gymnasium.spaces.Discrete</c>, live: seeded samples with and without a
    ///     mask and the generator state behind them, membership over a matrix of candidates, the attributes, the shape, the
    ///     <c>repr</c> and the constructor's validation.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/discrete.py#L27">gymnasium/spaces/discrete.py</see>.</remarks>
    [TestClass]
    public sealed class DiscreteLiveParityTests : ParityTestBase {
        /// <summary>Sampling cases: <c>(n, start, seed)</c>, including an <c>n</c> above 2^31 and one of 2^40.</summary>
        public static IEnumerable<object[]> SampleCases => new[] {
            new object[] { 4L, 0L, 0L },
            new object[] { 4L, 0L, 42L },
            new object[] { 3L, 5L, 1L },
            new object[] { 3L, -2L, 1000L },
            new object[] { 1L, 0L, 7L },
            new object[] { 3_000_000_000L, 0L, 0L },
            new object[] { 1L << 40, 0L, 5L },
        };

        /// <summary>
        ///     Membership candidates that must agree today, for <c>Discrete(3, start=5)</c>: the Python spelling (for reports),
        ///     Gymnasium's candidate built through the <see cref="Python"/> tree (or converted by <see cref="PyConvert"/>), and
        ///     the .NET value Gym.NET's <see cref="Discrete.Contains(object)"/> receives.
        /// </summary>
        private static readonly (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] PortableCandidates = {
            ("5", () => PyConvert.ToPython(5L), () => 5L),
            ("7", () => PyConvert.ToPython(7L), () => 7L),
            ("8", () => PyConvert.ToPython(8L), () => 8L),
            ("4", () => PyConvert.ToPython(4L), () => 4L),
            ("-1", () => PyConvert.ToPython(-1L), () => -1L),
            ("True", () => PyConvert.ToPython(true), () => true),
            ("np.int64(6)", () => Python.numpy.int64(6), () => NDArray.Scalar<long>(6L)),
            ("np.int32(6)", () => Python.numpy.int32(6), () => NDArray.Scalar<int>(6)),
            ("np.uint8(6)", () => Python.numpy.uint8(6), () => NDArray.Scalar<byte>(6)),
            ("np.float64(6.0)", () => Python.numpy.float64(6.0), () => NDArray.Scalar<double>(6.0)),
            ("6.0", () => PyConvert.ToPython(6.0), () => 6.0),
            ("np.array([6])", () => {
                using NDArray six = np.array(new[] { 6L });
                return six.ToNumpyCopy();
            }, () => np.array(new[] { 6L })),
            ("'6'", () => PyConvert.ToPython("6"), () => "6"),
            ("None", () => PyConvert.ToPython(null), () => null),
            ("2**63", () => PyConvert.ToPython(1UL << 63), () => 1UL << 63),
        };

        /// <summary>
        ///     <c>Discrete(n, seed=seed, start=start).sample()</c> ten times: every sample (dtype <c>int64</c>, value) and the
        ///     space's generator state afterwards are Gymnasium's.
        /// </summary>
        /// <param name="n">The number of elements.</param>
        /// <param name="start">The smallest element.</param>
        /// <param name="seed">The space's seed.</param>
        [TestMethod]
        [DynamicData(nameof(SampleCases))]
        public void Sample_MatchesGymnasium(long n, long start, long seed) {
            string constructor = $"spaces.Discrete({n}, seed={seed}, start={start})";
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(n, seed: seed, start: start));
            var dotnet = new Discrete(n, seed: seed, start: start);
            for (int i = 1; i <= 10; i++)
                ByteContract.AssertSame($"{constructor}.sample() #{i}", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => dotnet.Sample()));
            ByteContract.AssertSame($"{constructor} generator state after 10 samples", RngFacts.OfPythonSpace(python.Object), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
        }

        /// <summary>
        ///     <c>sample(mask=[1, 0, 1, 1])</c> five times draws among the allowed elements exactly as Gymnasium's
        ///     <c>np_random.choice(np.where(mask == 1)[0])</c> does, value and generator state.
        /// </summary>
        /// <param name="seed">The space's seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1000L)]
        public void MaskedSample_MatchesGymnasium(long seed) {
            string constructor = $"spaces.Discrete(4, seed={seed})";
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(4, seed: seed));
            var dotnet = new Discrete(4, seed: seed);
            using NDArray mask = np.array(new sbyte[] { 1, 0, 1, 1 });
            for (int i = 1; i <= 5; i++)
                ByteContract.AssertSame($"{constructor}.sample(mask=[1, 0, 1, 1]) #{i}", SampleWithMask(python, mask), Outcome.Capture(() => dotnet.Sample(mask)));
            ByteContract.AssertSame($"{constructor} generator state after 5 masked samples", RngFacts.OfPythonSpace(python.Object), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
        }

        /// <summary>
        ///     A mask that allows nothing returns <c>start</c> without drawing: the value and the untouched generator state are
        ///     Gymnasium's.
        /// </summary>
        [TestMethod]
        public void MaskedSample_AllowingNothing_ReturnsStartWithoutDrawing() {
            const string constructor = "spaces.Discrete(3, seed=9, start=5)";
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(3, seed: 9, start: 5));
            var dotnet = new Discrete(3, seed: 9, start: 5);
            using NDArray mask = np.zeros(new Shape(3), np.int8);
            ByteContract.AssertSame($"{constructor}.sample(mask=[0, 0, 0])", SampleWithMask(python, mask), Outcome.Capture(() => dotnet.Sample(mask)));
            ByteContract.AssertSame($"{constructor} generator state", RngFacts.OfPythonSpace(python.Object), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
        }

        /// <summary>
        ///     Membership agrees with Gymnasium's <c>contains</c> for integers inside and outside
        ///     <c>[start, start + n)</c>, booleans, NumPy integer scalars that can be cast to int64, floats, arrays, strings,
        ///     <c>None</c>, and an integer beyond int64.
        /// </summary>
        [TestMethod]
        public void Contains_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(3, start: 5));
            var dotnet = new Discrete(3, start: 5);
            foreach ((string candidate, Func<PyObject> gymnasium, Func<object?> value) in PortableCandidates)
                ByteContract.AssertSame($"Discrete(3, start=5).contains({candidate})", Contains(python, gymnasium), Outcome.Capture(() => dotnet.Contains(DisposeAfterUse(value()))));
        }

        /// <summary>
        ///     Gymnasium also requires <c>np.can_cast(x.dtype, self.dtype)</c>, so an in-range <c>np.uint64</c> is not a member of
        ///     an int64 space; Gym.NET accepts it.
        /// </summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Contains_RequiresCastableDtypes_LikeGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(3, start: 5));
            var dotnet = new Discrete(3, start: 5);
            ByteContract.AssertSame("Discrete(3, start=5).contains(np.uint64(6))", Contains(python, () => Python.numpy.uint64(6)), Outcome.Capture(() => dotnet.Contains(DisposeAfterUse(NDArray.Scalar<ulong>(6UL)))));
        }

        /// <summary>The attributes Gymnasium's users read: <c>n</c> and <c>start</c> (both <c>np.int64</c>) and the dtype.</summary>
        /// <param name="n">The number of elements.</param>
        /// <param name="start">The smallest element.</param>
        [TestMethod]
        [DataRow(4L, 0L)]
        [DataRow(3L, 5L)]
        [DataRow(3_000_000_000L, -7L)]
        public void Attributes_MatchGymnasium(long n, long start) {
            string constructor = $"spaces.Discrete({n}, start={start})";
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(n, start: start));
            var dotnet = new Discrete(n, start: start);
            ByteContract.AssertSame($"{constructor}.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.N)));
            ByteContract.AssertSame($"{constructor}.start", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.start(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.Start)));
            ByteContract.AssertSame($"{constructor}.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(python.Object)), Outcome.Returned(ValueFacts.Of(dotnet.DType)));
        }

        /// <summary>Gymnasium's <c>Discrete</c> is 0-d, <c>shape == ()</c>; Gym.NET's is <c>(n,)</c> (B-03).</summary>
        [TestMethod]
        [OpenBugs("B-03")]
        public void Shape_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(4));
            var dotnet = new Discrete(4);
            ByteContract.AssertSame("Discrete(4).shape", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.shape(python.Object)), Outcome.Returned(ShapeFacts(dotnet.Shape)));
        }

        /// <summary><c>repr</c> of the plain space, Gymnasium's <c>Discrete(4)</c>, is also Gym.NET's <c>ToString()</c>.</summary>
        [TestMethod]
        public void Repr_OfAPlainSpace_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(4));
            ByteContract.AssertSame("repr(Discrete(4))", Outcome.CapturePython(() => Python.builtins.repr(python.Object)), Outcome.Returned(ValueFacts.FromText(new Discrete(4).ToString())));
        }

        /// <summary>Gymnasium's <c>repr</c> names a non-zero start (<c>Discrete(3, start=5)</c>); Gym.NET's <c>ToString()</c> does not.</summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Repr_WithAStart_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => Python.gymnasium.spaces.Discrete(3, start: 5));
            ByteContract.AssertSame("repr(Discrete(3, start=5))", Outcome.CapturePython(() => Python.builtins.repr(python.Object)), Outcome.Returned(ValueFacts.FromText(new Discrete(3, start: 5).ToString())));
        }

        /// <summary>
        ///     Gymnasium rejects a non-positive <c>n</c> with <c>ValueError("n (counts) have to be positive, actual value: 0")</c>;
        ///     Gym.NET constructs the space (B-10's counterpart for <c>Discrete</c>).
        /// </summary>
        [TestMethod]
        [OpenBugs("B-10", "P2")]
        public void Constructor_RejectsANonPositiveN_LikeGymnasium() {
            ByteContract.AssertSame("Discrete(0)", Outcome.CapturePython(() => Python.gymnasium.spaces.Discrete(0)), Outcome.Capture(() => new Discrete(0)));
        }

        /// <summary>Describes a NumSharp shape the way Python describes <c>space.shape</c>: a tuple of Python ints.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The description.</returns>
        internal static ValueFacts ShapeFacts(Shape shape) {
            return ValueFacts.FromSequence(shape.Dimensions.Select(d => ValueFacts.FromInteger(d, "", "int")), "tuple");
        }

        /// <summary><c>space.sample(mask=mask)</c> on Gymnasium's side, with the mask crossed as a NumPy copy of the .NET one.</summary>
        /// <param name="space">The space.</param>
        /// <param name="mask">The mask both sides sample with.</param>
        /// <returns>The described sample, or what sampling raised.</returns>
        private static Outcome SampleWithMask(PyRef space, NDArray mask) {
            return Outcome.CapturePython(() => {
                using PyObject pythonMask = mask.ToNumpyCopy();
                return Python.gymnasium.spaces.space.Space.sample(space.Object, pythonMask);
            });
        }

        /// <summary><c>space.contains(candidate)</c> on Gymnasium's side, releasing the candidate afterwards.</summary>
        /// <param name="space">The space.</param>
        /// <param name="candidate">Builds the candidate under the GIL; it returns a new reference.</param>
        /// <returns>The described answer, or what the check raised.</returns>
        private static Outcome Contains(PyRef space, Func<PyObject> candidate) {
            return Outcome.CapturePython(() => {
                using PyObject value = candidate();
                return Python.gymnasium.spaces.space.Space.contains(space.Object, value);
            });
        }

        /// <summary>
        ///     Hands a membership candidate to <see cref="Discrete.Contains(object)"/> and releases it afterwards when it is an
        ///     array: the candidates are made per call and owned by the test.
        /// </summary>
        /// <param name="candidate">The candidate.</param>
        /// <returns>The candidate, registered for disposal after the current test's comparison.</returns>
        private object? DisposeAfterUse(object? candidate) {
            if (candidate is IDisposable disposable)
                _owned.Add(disposable);
            return candidate;
        }

        /// <summary>Candidates to release in <see cref="ReleaseCandidates"/>.</summary>
        private readonly List<IDisposable> _owned = new();

        /// <summary>Releases the arrays the membership candidates allocated.</summary>
        [TestCleanup]
        public void ReleaseCandidates() {
            foreach (IDisposable disposable in _owned)
                disposable.Dispose();
            _owned.Clear();
        }
    }
}
