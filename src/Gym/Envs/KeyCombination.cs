#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Gym.Envs {
    /// <summary>
    ///     Keys pressed together, as key codes: the key of a keyboard mapping (<see cref="Env{TObs, TAct}.GetKeysToAction"/>),
    ///     Python's tuple of ints such as MountainCar's <c>(275, 276)</c>. A code is an SDL keycode, which for a printable
    ///     key is Python's <c>ord</c> of its character (<c>97</c> for <c>a</c>), so a mapping written with characters
    ///     (<c>{"wa": 1}</c>, which <c>play</c> also accepts) becomes <see cref="FromText"/>'s codes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The codes keep the order given, as a Python tuple does, and equality compares them in that order; <c>play</c> sorts
    ///     the codes of every key when it looks a held set up, so the order never changes which action fires. The default
    ///     value is the empty combination <c>()</c>, nothing pressed (<see cref="None"/>).
    ///     </para>
    ///     <para>
    ///     An <see cref="ITuple"/>, so every tuple reader in Gym.NET (<c>play</c>'s key handling, Python's <c>repr</c>, the
    ///     parity harness) reads it as the tuple it stands for. Immutable: the codes are copied in.
    ///     </para>
    /// </remarks>
    public readonly struct KeyCombination : IEquatable<KeyCombination>, ITuple, IReadOnlyList<int> {
        /// <summary>The codes, or <see langword="null"/> for the empty combination (the default value).</summary>
        private readonly int[]? _codes;

        /// <summary>The combination of <paramref name="codes"/>, in the order given.</summary>
        /// <param name="codes">The key codes; copied, so the caller may reuse the array.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="codes"/> is <see langword="null"/>.</exception>
        public KeyCombination(params int[] codes) {
            if (codes is null)
                throw new ArgumentNullException(nameof(codes));
            _codes = codes.Length == 0 ? null : (int[])codes.Clone();
        }

        /// <summary>The empty combination <c>()</c>: nothing pressed.</summary>
        public static KeyCombination None => default;

        /// <summary>How many keys the combination holds.</summary>
        public int Count => _codes?.Length ?? 0;

        /// <summary>The code at <paramref name="index"/>.</summary>
        /// <param name="index">The position.</param>
        /// <returns>The key code.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown for a position outside the combination.</exception>
        public int this[int index] => (_codes ?? Array.Empty<int>())[index];

        /// <summary>The combination's length, read as a tuple's.</summary>
        int ITuple.Length => Count;

        /// <summary>The code at <paramref name="index"/>, boxed, read as a tuple's item.</summary>
        /// <param name="index">The position.</param>
        /// <returns>The key code.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown for a position outside the combination.</exception>
        object? ITuple.this[int index] => this[index];

        /// <summary>
        ///     The combination of a text's characters, one code each (Python's <c>tuple(ord(c) for c in keys)</c>, what
        ///     <c>play</c> makes of a string key): <c>FromText("wa")</c> is <c>(119, 97)</c>. A surrogate pair is one character.
        /// </summary>
        /// <param name="keys">The characters.</param>
        /// <returns>The combination.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="keys"/> is <see langword="null"/>.</exception>
        public static KeyCombination FromText(string keys) {
            if (keys is null)
                throw new ArgumentNullException(nameof(keys));
            var codes = new List<int>(keys.Length);
            for (int i = 0; i < keys.Length; i++) {
                // A character outside the Basic Multilingual Plane is two UTF-16 units and one code point, as Python's ord sees it.
                codes.Add(char.ConvertToUtf32(keys, i));
                if (char.IsHighSurrogate(keys[i]))
                    i++;
            }

            return new KeyCombination(codes.ToArray());
        }

        /// <summary>The codes in order.</summary>
        /// <returns>An enumerator over the codes.</returns>
        public IEnumerator<int> GetEnumerator() => ((IEnumerable<int>)(_codes ?? Array.Empty<int>())).GetEnumerator();

        /// <summary>The codes in order, untyped.</summary>
        /// <returns>An enumerator over the codes.</returns>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Equal when both hold the same codes in the same order, as Python's tuples compare.</summary>
        /// <param name="other">The other combination.</param>
        /// <returns><see langword="true"/> for the same codes in order.</returns>
        public bool Equals(KeyCombination other) {
            if (Count != other.Count)
                return false;
            for (int i = 0; i < Count; i++) {
                if (this[i] != other[i])
                    return false;
            }

            return true;
        }

        /// <summary>Equal to another combination with the same codes in the same order.</summary>
        /// <param name="obj">The other value.</param>
        /// <returns><see langword="true"/> for an equal combination.</returns>
        public override bool Equals(object? obj) => obj is KeyCombination other && Equals(other);

        /// <summary>A hash of the codes in order, equal for equal combinations.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() {
            var hash = new HashCode();
            foreach (int code in this)
                hash.Add(code);
            return hash.ToHashCode();
        }

        /// <summary>Whether two combinations hold the same codes in the same order.</summary>
        /// <param name="left">One combination.</param>
        /// <param name="right">The other.</param>
        /// <returns><see langword="true"/> when they are equal.</returns>
        public static bool operator ==(KeyCombination left, KeyCombination right) => left.Equals(right);

        /// <summary>Whether two combinations differ.</summary>
        /// <param name="left">One combination.</param>
        /// <param name="right">The other.</param>
        /// <returns><see langword="true"/> when they are not equal.</returns>
        public static bool operator !=(KeyCombination left, KeyCombination right) => !left.Equals(right);

        /// <summary>Python's <c>repr</c> of the tuple: <c>()</c>, <c>(276,)</c>, <c>(275, 276)</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => Count == 1 ? $"({this[0]},)" : $"({string.Join(", ", this)})";
    }
}
