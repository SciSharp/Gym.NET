#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Text</c> tests (<c>tests/spaces/test_text.py</c>): masked and weighted samples (length and characters)
    ///     with their errors, the character order of set and string charsets, and multi-character charset elements refused.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A charset given as a list (<c>["ab", "cd"]</c>) is Python-only: C# takes a string or a set. <c>test_deterministic_across_hash_seeds</c>
    ///     guards against Python's hash randomization, which .NET's ordinal sorting of a set does not have.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_text.py">tests/spaces/test_text.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestText {
        /// <summary>A length mask fixes the length; a character mask restricts the characters; an all-zero mask with a positive minimum is refused.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_text.py#L12">test_sample_mask</see>.</remarks>
        [TestMethod]
        public void TestSampleMask() {
            using (var space = new Text(5)) {
                string sample = (string)space.Sample(new object?[] { 3L, null })!;
                Assert.IsTrue(space.Contains(sample));
                Assert.AreEqual(3, sample.Length);

                sample = (string)space.Sample()!;
                Assert.IsTrue(space.Contains(sample));
                Assert.IsTrue(sample.Length is >= 1 and <= 5);

                using NDArray zeros = np.zeros(new Shape((long)space.CharacterSet.Count), np.int8);
                Assert.AreEqual("Trying to sample with a minimum length > 0 (actual minimum length=1) but the character mask is all zero meaning that no character could be sampled.",
                    Assert.ThrowsExactly<ValueError>(() => space.Sample(new object?[] { 3L, zeros })).Message);
            }

            using (var space = new Text(5, minLength: 0)) {
                using NDArray zeros = np.zeros(new Shape((long)space.CharacterSet.Count), np.int8);
                string sample = (string)space.Sample(new object?[] { null, zeros })!;
                Assert.IsTrue(space.Contains(sample));
                Assert.AreEqual("", sample);

                sample = (string)space.Sample(new object?[] { 0L, null })!;
                Assert.IsTrue(space.Contains(sample));
                Assert.AreEqual("", sample);
            }

            using (var space = new Text(5, charset: "abcd"))
            using (NDArray onlyB = np.array(new sbyte[] { 0, 1, 0, 0 })) {
                string sample = (string)space.Sample(new object?[] { 3L, onlyB })!;
                Assert.IsTrue(space.Contains(sample));
                Assert.AreEqual("bbb", sample);
            }
        }

        /// <summary>A probability mask weights the characters; one that sums to 0 is refused.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_text.py#L51">test_sample_probability</see>.</remarks>
        [TestMethod]
        public void TestSampleProbability() {
            using (var space = new Text(5)) {
                string sample = (string)space.Sample(probability: new object?[] { 3L, null })!;
                Assert.IsTrue(space.Contains(sample));
                Assert.AreEqual(3, sample.Length);

                sample = (string)space.Sample(probability: null)!;
                Assert.IsTrue(space.Contains(sample));
                Assert.IsTrue(sample.Length is >= 1 and <= 5);

                using NDArray zeros = np.zeros(new Shape((long)space.CharacterSet.Count), np.float64);
                StringAssert.Contains(Assert.ThrowsExactly<AssertionError>(() => space.Sample(probability: new object?[] { 3L, zeros })).Message,
                    "Expects the sum of the probability mask to be 1, actual sum: 0.0");
            }

            using (var space = new Text(5, charset: "abcd")) {
                using (NDArray onlyB = np.array(new[] { 0.0, 1, 0, 0 })) {
                    string sample = (string)space.Sample(probability: new object?[] { 3L, onlyB })!;
                    Assert.IsTrue(space.Contains(sample));
                    Assert.AreEqual("bbb", sample);
                }

                using (NDArray aOrB = np.array(new[] { 0.5, 0.5, 0, 0 })) {
                    string sample = (string)space.Sample(probability: new object?[] { 2L, aOrB })!;
                    Assert.IsTrue(space.Contains(sample));
                    CollectionAssert.Contains(new[] { "aa", "bb", "ab", "ba" }, sample);
                }
            }
        }

        /// <summary>A set charset is sorted; a string charset keeps its order and drops duplicates.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_text.py#L87">test_charset_ordering</see>.</remarks>
        [TestMethod]
        public void TestCharsetOrdering() {
            using (var space = new Text(5, new HashSet<string> { "d", "c", "b", "a" })) {
                CollectionAssert.AreEqual(new[] { "a", "b", "c", "d" }, space.CharacterList.ToArray());
                CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, "abcd".Select(c => space.CharacterIndex(c.ToString())).ToArray());
            }

            using (var space = new Text(5, charset: "dcbad")) {
                CollectionAssert.AreEqual(new[] { "d", "c", "b", "a" }, space.CharacterList.ToArray());
                CollectionAssert.AreEqual(new[] { 0, 1, 2, 3 }, "dcba".Select(c => space.CharacterIndex(c.ToString())).ToArray());
            }
        }

        /// <summary>Charset elements longer than one character (or empty) are refused, listed in order; single characters keep the contract.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_text.py#L99">test_multi_character_charset_raises</see>.</remarks>
        [TestMethod]
        public void TestMultiCharacterCharsetRaises() {
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => new Text(1, new HashSet<string> { "sit", "walk", "fly" })).Message,
                "Expects all charset elements to be a single character, actual invalid elements: ['fly', 'sit', 'walk']");
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => new Text(3, new HashSet<string> { "a", "" })).Message,
                "actual invalid elements: ['']");

            foreach (Func<Text> create in new Func<Text>[] { () => new Text(3, new HashSet<string> { "a", "b", "c" }), () => new Text(3, charset: "abc") }) {
                using Text space = create();
                for (int i = 0; i < 10; i++) {
                    string sample = (string)space.Sample()!;
                    Assert.IsTrue(space.Contains(sample));
                    Assert.IsTrue(sample.Length <= space.MaxLength);
                }
            }
        }
    }
}
