using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Spaces;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     Gym.NET's <see cref="Discrete"/> against <c>gymnasium.spaces.Discrete</c> (<c>gymnasium/spaces/discrete.py</c> @
    ///     9e04324), live: seeded samples with and without a mask and the generator state behind them, membership over a
    ///     matrix of candidates, the attributes, the shape, the <c>repr</c> and the constructor's validation.
    /// </summary>
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
        ///     Membership candidates that must agree today: Python source for Gymnasium's side and the .NET value Gym.NET's
        ///     <see cref="Discrete.Contains(object)"/> receives, for <c>Discrete(3, start=5)</c>.
        /// </summary>
        private static readonly (string Python, Func<object?> DotNet)[] PortableCandidates = {
            ("5", () => 5L),
            ("7", () => 7L),
            ("8", () => 8L),
            ("4", () => 4L),
            ("-1", () => -1L),
            ("True", () => true),
            ("np.int64(6)", () => NDArray.Scalar<long>(6L)),
            ("np.int32(6)", () => NDArray.Scalar<int>(6)),
            ("np.uint8(6)", () => NDArray.Scalar<byte>(6)),
            ("np.float64(6.0)", () => NDArray.Scalar<double>(6.0)),
            ("6.0", () => 6.0),
            ("np.array([6])", () => np.array(new[] { 6L })),
            ("'6'", () => "6"),
            ("None", () => null),
            ("2**63", () => 1UL << 63),
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
            using PyHandle python = PyGym.Evaluate(constructor);
            var dotnet = new Discrete(n, seed: seed, start: start);
            for (int i = 1; i <= 10; i++)
                ByteContract.AssertSame($"{constructor}.sample() #{i}", python.Invoke("sample"), Outcome.Capture(() => dotnet.Sample()));
            ByteContract.AssertSame($"{constructor} generator state after 10 samples", python.Evaluate("x.np_random.bit_generator.state"), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
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
            using PyHandle python = PyGym.Evaluate(constructor);
            var dotnet = new Discrete(4, seed: seed);
            using NDArray mask = np.array(new sbyte[] { 1, 0, 1, 1 });
            for (int i = 1; i <= 5; i++)
                ByteContract.AssertSame($"{constructor}.sample(mask=[1, 0, 1, 1]) #{i}", python.Invoke("sample", mask), Outcome.Capture(() => dotnet.Sample(mask)));
            ByteContract.AssertSame($"{constructor} generator state after 5 masked samples", python.Evaluate("x.np_random.bit_generator.state"), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
        }

        /// <summary>
        ///     A mask that allows nothing returns <c>start</c> without drawing: the value and the untouched generator state are
        ///     Gymnasium's.
        /// </summary>
        [TestMethod]
        public void MaskedSample_AllowingNothing_ReturnsStartWithoutDrawing() {
            const string constructor = "spaces.Discrete(3, seed=9, start=5)";
            using PyHandle python = PyGym.Evaluate(constructor);
            var dotnet = new Discrete(3, seed: 9, start: 5);
            using NDArray mask = np.zeros(new Shape(3), np.int8);
            ByteContract.AssertSame($"{constructor}.sample(mask=[0, 0, 0])", python.Invoke("sample", mask), Outcome.Capture(() => dotnet.Sample(mask)));
            ByteContract.AssertSame($"{constructor} generator state", python.Evaluate("x.np_random.bit_generator.state"), Outcome.Returned(GeneratorProbe.StateOf(dotnet)));
        }

        /// <summary>
        ///     Membership agrees with Gymnasium's <c>contains</c> for integers inside and outside
        ///     <c>[start, start + n)</c>, booleans, NumPy integer scalars that can be cast to int64, floats, arrays, strings,
        ///     <c>None</c>, and an integer beyond int64.
        /// </summary>
        [TestMethod]
        public void Contains_MatchesGymnasium() {
            using PyHandle python = PyGym.Evaluate("spaces.Discrete(3, start=5)");
            var dotnet = new Discrete(3, start: 5);
            foreach ((string candidate, Func<object?> value) in PortableCandidates)
                ByteContract.AssertSame($"Discrete(3, start=5).contains({candidate})", python.Invoke("contains", new PyExpression(candidate)), Outcome.Capture(() => dotnet.Contains(DisposeAfterUse(value()))));
        }

        /// <summary>
        ///     Gymnasium also requires <c>np.can_cast(x.dtype, self.dtype)</c>, so an in-range <c>np.uint64</c> is not a member of
        ///     an int64 space; Gym.NET accepts it.
        /// </summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Contains_RequiresCastableDtypes_LikeGymnasium() {
            using PyHandle python = PyGym.Evaluate("spaces.Discrete(3, start=5)");
            var dotnet = new Discrete(3, start: 5);
            ByteContract.AssertSame("Discrete(3, start=5).contains(np.uint64(6))", python.Invoke("contains", new PyExpression("np.uint64(6)")), Outcome.Capture(() => dotnet.Contains(DisposeAfterUse(NDArray.Scalar<ulong>(6UL)))));
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
            using PyHandle python = PyGym.Evaluate(constructor);
            var dotnet = new Discrete(n, start: start);
            ByteContract.AssertSame($"{constructor}.n", python.Get("n"), Outcome.Returned(ValueFacts.Of(dotnet.N)));
            ByteContract.AssertSame($"{constructor}.start", python.Get("start"), Outcome.Returned(ValueFacts.Of(dotnet.Start)));
            ByteContract.AssertSame($"{constructor}.dtype.str", python.Evaluate("x.dtype.str"), Outcome.Returned(ValueFacts.FromText(dotnet.DType.str)));
        }

        /// <summary>Gymnasium's <c>Discrete</c> is 0-d, <c>shape == ()</c>; Gym.NET's is <c>(n,)</c> (B-03).</summary>
        [TestMethod]
        [OpenBugs("B-03")]
        public void Shape_MatchesGymnasium() {
            using PyHandle python = PyGym.Evaluate("spaces.Discrete(4)");
            var dotnet = new Discrete(4);
            ByteContract.AssertSame("Discrete(4).shape", python.Get("shape"), Outcome.Returned(ShapeFacts(dotnet.Shape)));
        }

        /// <summary><c>repr</c> of the plain space, Gymnasium's <c>Discrete(4)</c>, is also Gym.NET's <c>ToString()</c>.</summary>
        [TestMethod]
        public void Repr_OfAPlainSpace_MatchesGymnasium() {
            using PyHandle python = PyGym.Evaluate("spaces.Discrete(4)");
            ByteContract.AssertSame("repr(Discrete(4))", python.Evaluate("repr(x)"), Outcome.Returned(ValueFacts.FromText(new Discrete(4).ToString())));
        }

        /// <summary>Gymnasium's <c>repr</c> names a non-zero start (<c>Discrete(3, start=5)</c>); Gym.NET's <c>ToString()</c> does not.</summary>
        [TestMethod]
        [OpenBugs("P2")]
        public void Repr_WithAStart_MatchesGymnasium() {
            using PyHandle python = PyGym.Evaluate("spaces.Discrete(3, start=5)");
            ByteContract.AssertSame("repr(Discrete(3, start=5))", python.Evaluate("repr(x)"), Outcome.Returned(ValueFacts.FromText(new Discrete(3, start: 5).ToString())));
        }

        /// <summary>
        ///     Gymnasium rejects a non-positive <c>n</c> with <c>ValueError("n (counts) have to be positive, actual value: 0")</c>;
        ///     Gym.NET constructs the space (B-10's counterpart for <c>Discrete</c>).
        /// </summary>
        [TestMethod]
        [OpenBugs("B-10", "P2")]
        public void Constructor_RejectsANonPositiveN_LikeGymnasium() {
            ByteContract.AssertSame("Discrete(0)", PyGym.Invoke("spaces.Discrete(0)"), Outcome.Capture(() => new Discrete(0)));
        }

        /// <summary>Describes a NumSharp shape the way Python describes <c>space.shape</c>: a tuple of Python ints.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The description.</returns>
        internal static ValueFacts ShapeFacts(Shape shape) {
            return ValueFacts.FromSequence(shape.Dimensions.Select(d => ValueFacts.FromInteger(d, "", "int")), "tuple");
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
