#nullable enable
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Envs {
    /// <summary>
    ///     <see cref="KeyCombination"/>, the key of <c>get_keys_to_action</c>'s typed mapping (DR-45): Python's tuple of key
    ///     codes, its order kept, compared and printed as the tuple.
    /// </summary>
    [TestClass]
    public class KeyCombinationTests {
        /// <summary>Python's tuple <c>repr</c>: the empty tuple, the one-element tuple's trailing comma, and the rest.</summary>
        [TestMethod]
        public void ToString_IsPythonsTupleRepr() {
            Assert.AreEqual("()", KeyCombination.None.ToString());
            Assert.AreEqual("(276,)", new KeyCombination(276).ToString());
            Assert.AreEqual("(275, 276)", new KeyCombination(275, 276).ToString());
            Assert.AreEqual("(275, 276)", PyBuiltins.Repr(new KeyCombination(275, 276)), "Gym.NET's repr reads it as the tuple it is");
        }

        /// <summary>Equality compares the codes in order, as Python's tuples do; the default value is the empty combination.</summary>
        [TestMethod]
        public void Equality_IsOrdered() {
            Assert.AreEqual(new KeyCombination(275, 276), new KeyCombination(275, 276));
            Assert.AreNotEqual(new KeyCombination(275, 276), new KeyCombination(276, 275));
            Assert.AreEqual(KeyCombination.None, new KeyCombination());
            Assert.AreEqual(new KeyCombination(1, 2).GetHashCode(), new KeyCombination(1, 2).GetHashCode());
            Assert.IsTrue(new KeyCombination(97) == KeyCombination.FromText("a"));
        }

        /// <summary><see cref="KeyCombination.FromText"/> is <c>play</c>'s reading of a string key: each character's code point, a surrogate pair as one.</summary>
        [TestMethod]
        public void FromText_TakesEachCharactersCodePoint() {
            Assert.AreEqual(new KeyCombination(119, 97), KeyCombination.FromText("wa"));
            Assert.AreEqual(new KeyCombination(0x1F600), KeyCombination.FromText("\U0001F600"));
            Assert.AreEqual(KeyCombination.None, KeyCombination.FromText(""));
        }

        /// <summary>As an <see cref="ITuple"/>, the combination's length and items are the codes, boxed.</summary>
        [TestMethod]
        public void IsATupleOfItsCodes() {
            ITuple tuple = new KeyCombination(275, 276);
            Assert.AreEqual(2, tuple.Length);
            Assert.AreEqual(275, tuple[0]);
            Assert.AreEqual(276, tuple[1]);
            Assert.AreEqual(0, ((ITuple)KeyCombination.None).Length);
        }

        /// <summary>The codes are copied in: changing the array afterwards does not change the combination.</summary>
        [TestMethod]
        public void Codes_AreCopiedIn() {
            int[] codes = { 1, 2 };
            var combination = new KeyCombination(codes);
            codes[0] = 9;
            Assert.AreEqual(1, combination[0]);
        }
    }
}
