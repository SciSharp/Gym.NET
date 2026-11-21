#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

// Outside namespace Gym on purpose: this is where an app stands, and where `using Gym.Spaces;` makes `Spaces` the static
// class (inside Gym.* the simple name is a namespace, the footgun the class's docs name).
namespace GymConsumer {
    /// <summary>
    ///     Pins DR-79, DR-81, DR-86 and DR-97 as an app uses them: the static <see cref="Spaces"/> class mirrors
    ///     <c>gymnasium.spaces</c>, building each space exactly as its constructor does; the typed <c>Sample</c> overloads take
    ///     Gymnasium's whole parameter list and draw what the untyped entry draws; the simple spaces' <c>Seed</c> returns the
    ///     seed as a <see cref="BigInteger"/>; <see cref="Space.Clone"/> is <see cref="Space.DeepCopy"/>.
    /// </summary>
    [TestClass]
    public class SpacesFacadeTests {
        /// <summary>Asserts two spaces are equal and draw the same five samples, so a factory built exactly what its constructor builds.</summary>
        /// <param name="expected">The constructor's space.</param>
        /// <param name="actual">The factory's space.</param>
        /// <param name="what">The case, for the messages.</param>
        /// <param name="reseed">
        ///     Whether to seed both before drawing: for the <c>params</c> forms, which take no seed, so both spaces start from
        ///     OS entropy. The seeded cases draw as built, which checks that the factory passed the seed on.
        /// </param>
        private static void AssertSameSpace(Space expected, Space actual, string what, bool reseed = false) {
            using (expected)
            using (actual) {
                Assert.AreEqual(expected.GetType(), actual.GetType(), what);
                Assert.AreEqual(expected, actual, what);
                if (reseed) {
                    expected.Seed(0);
                    actual.Seed(0);
                }

                for (int i = 0; i < 5; i++)
                    Assert.IsTrue(Gym.Utils.PyEquality.AreEqual(expected.Sample(), actual.Sample()), $"{what}: sample {i}");
            }
        }

        /// <summary>Every factory builds the class of its name, equal to the constructor's space and drawing the same values with the same seed.</summary>
        [TestMethod]
        public void EveryFactory_BuildsWhatItsConstructorBuilds() {
            using NDArray low = np.array(new[] { -1.0, -2.0 });
            using NDArray high = np.array(new[] { 1.0, 2.0 });
            AssertSameSpace(new Box(-1, 1, new Shape(3), np.float32, seed: 0), Spaces.Box(-1, 1, new Shape(3), np.float32, seed: 0), "Box(double, double)");
            AssertSameSpace(new Box(low, high, seed: 0), Spaces.Box(low, high, seed: 0), "Box(NDArray, NDArray)");
            AssertSameSpace(new Box(low, 3.0, seed: 0), Spaces.Box(low, 3.0, seed: 0), "Box(NDArray, double)");
            AssertSameSpace(new Box(-3.0, high, seed: 0), Spaces.Box(-3.0, high, seed: 0), "Box(double, NDArray)");
            AssertSameSpace(new Discrete(5, seed: 0, start: -2), Spaces.Discrete(5, seed: 0, start: -2), "Discrete");
            AssertSameSpace(new MultiBinary(4, seed: 0), Spaces.MultiBinary(4, seed: 0), "MultiBinary(long)");
            AssertSameSpace(new MultiBinary(new long[] { 2, 3 }, seed: 0), Spaces.MultiBinary(new long[] { 2, 3 }, seed: 0), "MultiBinary(long[])");
            AssertSameSpace(new MultiDiscrete(new long[] { 3, 2 }, seed: 0), Spaces.MultiDiscrete(new long[] { 3, 2 }, seed: 0), "MultiDiscrete(long[])");
            using (NDArray nvec = np.array(new long[,] { { 2, 3 }, { 4, 5 } }))
                AssertSameSpace(new MultiDiscrete(nvec, seed: 0), Spaces.MultiDiscrete(nvec, seed: 0), "MultiDiscrete(NDArray)");
            AssertSameSpace(new Gym.Spaces.Tuple(new Space[] { new Discrete(2), new Discrete(3) }, seed: 0),
                Spaces.Tuple(new Space[] { new Discrete(2), new Discrete(3) }, seed: 0), "Tuple(IEnumerable)");
            AssertSameSpace(new Gym.Spaces.Tuple(new Discrete(2), new Discrete(3)), Spaces.Tuple(new Discrete(2), new Discrete(3)), "Tuple(params)", reseed: true);
            AssertSameSpace(new Dict(new[] { KeyValuePair.Create<string, Space>("b", new Discrete(2)), KeyValuePair.Create<string, Space>("a", new Discrete(3)) }, seed: 0),
                Spaces.Dict(new[] { KeyValuePair.Create<string, Space>("b", new Discrete(2)), KeyValuePair.Create<string, Space>("a", new Discrete(3)) }, seed: 0), "Dict(pairs)");
            AssertSameSpace(new Dict(new Dictionary<string, Space> { ["b"] = new Discrete(2), ["a"] = new Discrete(3) }, seed: 0),
                Spaces.Dict(new Dictionary<string, Space> { ["b"] = new Discrete(2), ["a"] = new Discrete(3) }, seed: 0), "Dict(mapping)");
            AssertSameSpace(new Text(5, seed: 0), Spaces.Text(5, seed: 0), "Text(string charset)");
            AssertSameSpace(new Text(4, new HashSet<string> { "x", "a", "q" }, seed: 0), Spaces.Text(4, new HashSet<string> { "x", "a", "q" }, seed: 0), "Text(set charset)");
            AssertSameSpace(new Sequence(new Discrete(3), seed: 0), Spaces.Sequence(new Discrete(3), seed: 0), "Sequence");
            AssertSameSpace(new Graph(new Box(0, 1, new Shape(2)), new Discrete(3), seed: 0), Spaces.Graph(new Box(0, 1, new Shape(2)), new Discrete(3), seed: 0), "Graph");
            AssertSameSpace(new OneOf(new Space[] { new Discrete(2), new Discrete(3) }, seed: 0), Spaces.OneOf(new Space[] { new Discrete(2), new Discrete(3) }, seed: 0), "OneOf(IEnumerable)");
            AssertSameSpace(new OneOf(new Discrete(2), new Discrete(3)), Spaces.OneOf(new Discrete(2), new Discrete(3)), "OneOf(params)", reseed: true);

            using GraphInstance built = Spaces.GraphInstance(np.zeros(new Shape(2, 2)), null, null);
            using var expected = new GraphInstance(np.zeros(new Shape(2, 2)), null, null);
            Assert.AreEqual(expected, built, "GraphInstance");
        }

        /// <summary>The module's four functions are <see cref="SpaceUtils"/>'s: the same size, flattening, flattened space and round trip.</summary>
        [TestMethod]
        public void TheFlatteningFunctions_AreSpaceUtils() {
            using var space = new Gym.Spaces.Tuple(new Discrete(3), new Box(-1, 1, new Shape(2)));
            Assert.AreEqual(SpaceUtils.Flatdim(space), Spaces.Flatdim(space));
            Assert.AreEqual(5L, Spaces.Flatdim(space));
            object?[] element = { NDArray.Scalar(2L), np.array(new[] { 0.5f, -0.5f }) };
            using var flat = (NDArray)Spaces.Flatten(space, element)!;
            using var expectedFlat = (NDArray)SpaceUtils.Flatten(space, element)!;
            Assert.IsTrue(np.array_equal(expectedFlat, flat));
            using Space flatSpace = Spaces.FlattenSpace(space);
            Assert.AreEqual(SpaceUtils.FlattenSpace(space), flatSpace);
            Assert.IsTrue(Gym.Utils.PyEquality.AreEqual(element, Spaces.Unflatten(space, flat)), "unflatten undoes flatten");
        }

        /// <summary>
        ///     Each typed <c>Sample</c> overload has Gymnasium's whole parameter list (a method group conversion proves the
        ///     signature) and draws, from the same seed, exactly what the untyped entry draws (DR-97).
        /// </summary>
        [TestMethod]
        public void TheTypedSampleOverloads_DrawWhatTheUntypedEntryDraws() {
            using var discrete = new Discrete(5, seed: 1);
            using var discreteTwin = new Discrete(5, seed: 1);
            Func<NDArray?, NDArray?, NDArray> discreteTyped = discrete.Sample;
            using NDArray mask = np.array(new sbyte[] { 0, 1, 0, 1, 1 });
            using NDArray probability = np.array(new[] { 0.1, 0.2, 0.3, 0.2, 0.2 });
            for (int i = 0; i < 5; i++) {
                AssertSameDraw(discreteTwin.Sample((object)mask), discreteTyped(mask, null), "Discrete mask");
                AssertSameDraw(discreteTwin.Sample(null, (object)probability), discrete.Sample(probability: probability), "Discrete probability");
                AssertSameDraw(discreteTwin.Sample((object?)null), discrete.Sample(), "Discrete unmasked");
            }

            using var binary = new MultiBinary(new long[] { 2, 2 }, seed: 1);
            using var binaryTwin = new MultiBinary(new long[] { 2, 2 }, seed: 1);
            Func<NDArray?, NDArray?, NDArray> binaryTyped = binary.Sample;
            using NDArray binaryMask = np.array(new sbyte[,] { { 2, 0 }, { 1, 2 } });
            for (int i = 0; i < 5; i++)
                AssertSameDraw(binaryTwin.Sample((object)binaryMask), binaryTyped(binaryMask, null), "MultiBinary mask");

            using var box = new Box(-1, 1, new Shape(3), seed: 1);
            using var boxTwin = new Box(-1, 1, new Shape(3), seed: 1);
            Func<NDArray> boxTyped = box.Sample;
            for (int i = 0; i < 5; i++)
                AssertSameDraw(boxTwin.Sample((object?)null), boxTyped(), "Box");
        }

        /// <summary>Asserts two draws are the same array (values and dtype as <c>np.array_equal</c> reads them), then releases both.</summary>
        /// <param name="expected">The untyped entry's draw.</param>
        /// <param name="actual">The typed overload's draw.</param>
        /// <param name="what">The case, for the message.</param>
        private static void AssertSameDraw(NDArray expected, NDArray actual, string what) {
            using (expected)
            using (actual)
                Assert.IsTrue(np.array_equal(expected, actual), what);
        }

        /// <summary>
        ///     MultiDiscrete's typed overload takes Gymnasium's tuple as a list: the list itself for a one-dimensional space
        ///     (a wrong length meets Gymnasium's own assertion), the entries in C order for a multi-dimensional one (a wrong
        ///     count is refused before any draw).
        /// </summary>
        [TestMethod]
        public void MultiDiscrete_TypedMasks_AreGymnasiumsTuples() {
            using var flat = new MultiDiscrete(new long[] { 3, 2 }, seed: 1);
            using var flatTwin = new MultiDiscrete(new long[] { 3, 2 }, seed: 1);
            Func<IReadOnlyList<NDArray>?, IReadOnlyList<NDArray>?, NDArray> typed = flat.Sample;
            NDArray[] masks = { np.array(new sbyte[] { 0, 1, 1 }), np.array(new sbyte[] { 1, 0 }) };
            for (int i = 0; i < 5; i++)
                AssertSameDraw(flatTwin.Sample(new object?[] { masks[0], masks[1] }), typed(masks, null), "one-dimensional mask");
            var tooShort = Assert.ThrowsExactly<AssertionError>(() => flat.Sample(new[] { masks[0] }));
            StringAssert.Contains(tooShort.Message, "Expects the mask length to be equal to the number of actions");

            using NDArray nvec = np.array(new long[,] { { 2, 3 }, { 2, 2 } });
            using var grid = new MultiDiscrete(nvec, seed: 2);
            using var gridTwin = new MultiDiscrete(nvec, seed: 2);
            NDArray[] cOrder = {
                np.array(new sbyte[] { 0, 1 }), np.array(new sbyte[] { 1, 0, 1 }), np.array(new sbyte[] { 1, 1 }), np.array(new sbyte[] { 0, 0 }),
            };
            object?[] nested = { new object?[] { cOrder[0], cOrder[1] }, new object?[] { cOrder[2], cOrder[3] } };
            for (int i = 0; i < 5; i++)
                AssertSameDraw(gridTwin.Sample(nested), grid.Sample(cOrder), "multi-dimensional mask in C order");
            var wrongCount = Assert.ThrowsExactly<ValueError>(() => grid.Sample(new[] { cOrder[0], cOrder[1] }));
            StringAssert.Contains(wrongCount.Message, "expected 4 arrays");
            foreach (NDArray m in masks)
                m.Dispose();
            foreach (NDArray m in cOrder)
                m.Dispose();
        }

        /// <summary>
        ///     The simple spaces' <c>Seed</c> returns the seed as a <see cref="BigInteger"/>; through a <see cref="Space"/>
        ///     reference the same seed comes boxed; an unseeded call returns the entropy, which reproduces the draws.
        /// </summary>
        [TestMethod]
        public void TheSimpleSpaces_SeedTyped() {
            using var discrete = new Discrete(10);
            BigInteger used = discrete.Seed(5);
            Assert.AreEqual(new BigInteger(5), used);
            Assert.AreEqual((object)new BigInteger(5), ((Space)discrete).Seed(5), "the untyped base method returns the same seed, boxed");
            using var box = new Box(-1, 1, new Shape(2));
            using var multiBinary = new MultiBinary(3);
            using var multiDiscrete = new MultiDiscrete(new long[] { 2, 3 });
            using var text = new Text(4);
            Assert.AreEqual(new BigInteger(7), box.Seed(7));
            Assert.AreEqual(new BigInteger(7), multiBinary.Seed(7));
            Assert.AreEqual(new BigInteger(7), multiDiscrete.Seed(7));
            Assert.AreEqual(new BigInteger(7), text.Seed(7));

            BigInteger entropy = discrete.Seed();
            using NDArray first = discrete.Sample();
            discrete.Seed(entropy);
            using NDArray again = discrete.Sample();
            Assert.IsTrue(np.array_equal(first, again), "the entropy an unseeded call returns reproduces its draws");
        }

        /// <summary><see cref="Space.Clone"/> is <see cref="Space.DeepCopy"/>: an equal, independent space, whose draws never move the original's generator.</summary>
        [TestMethod]
        public void Clone_IsTheDeepCopy() {
            using var space = new Discrete(100, seed: 3);
            using var twin = new Discrete(100, seed: 3);
            using Space clone = space.Clone();
            Assert.AreNotSame(space, clone);
            Assert.AreEqual(space, clone);
            Assert.AreEqual(typeof(Discrete), clone.GetType(), "the copy is the space's own class");

            // The copy starts where the original stands; drawing from it must leave the original's generator where it was.
            using NDArray fromTwin = twin.Sample();
            using NDArray fromClone = ((Discrete)clone).Sample();
            Assert.IsTrue(np.array_equal(fromTwin, fromClone), "the copy continues the original's stream");
            using NDArray fromOriginal = space.Sample();
            Assert.IsTrue(np.array_equal(fromTwin, fromOriginal), "drawing from the copy did not move the original");
        }
    }
}
