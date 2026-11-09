#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The strings of a character set whose length lies in <c>[MinLength, MaxLength]</c>, such as a command or a name.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Characters are Unicode code points, as Python's are, not UTF-16 code units: a character outside the BMP (an emoji)
    ///     counts once toward a string's length, and a charset may hold it.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/text.py#L16">gymnasium/spaces/text.py</see>.</para>
    /// </remarks>
    public class Text : Space<string>, IEquatable<Text> {
        /// <summary>Gymnasium's default charset, <c>alphanumeric</c> (ASCII letters and digits), as the sorted list a set becomes.</summary>
        public const string Alphanumeric = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

        /// <summary>The characters in sampling order (Gymnasium's <c>character_list</c>).</summary>
        private readonly string[] _charList;

        /// <summary>Each character's index in <see cref="_charList"/> (Gymnasium's <c>_char_index</c>).</summary>
        private readonly Dictionary<string, int> _charIndex;

        /// <summary>
        ///     Strings of <paramref name="charset"/> with <paramref name="minLength"/> to <paramref name="maxLength"/> characters
        ///     (Gymnasium's <c>Text(max_length, min_length=1, charset=alphanumeric, seed=seed)</c> with a string charset, whose
        ///     order is kept and duplicates dropped).
        /// </summary>
        /// <param name="maxLength">The longest length, inclusive.</param>
        /// <param name="minLength">The shortest length, inclusive; 1 by default, so no empty string.</param>
        /// <param name="charset">The characters, in sampling order; <see langword="null"/> is <see cref="Alphanumeric"/> (Gymnasium's default set, sorted).</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for a negative minimum or a minimum above the maximum, with Gymnasium's messages.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Text(long maxLength, long minLength = 1, string? charset = null, SpaceSeed? seed = null)
            : this(maxLength, minLength, CodePoints(charset ?? Alphanumeric).Distinct().ToArray(), seed) { }

        /// <summary>
        ///     Strings of a character set (Gymnasium's <c>Text(..., charset=frozenset(...))</c>): the characters are sorted, as
        ///     Gymnasium sorts a set so sampling does not depend on hash order.
        /// </summary>
        /// <param name="maxLength">The longest length, inclusive.</param>
        /// <param name="charset">The characters; each must be a single code point.</param>
        /// <param name="minLength">The shortest length, inclusive.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for invalid lengths, or a charset element that is not one character.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="charset"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public Text(long maxLength, ISet<string> charset, long minLength = 1, SpaceSeed? seed = null)
            : this(maxLength, minLength, (charset ?? throw new ArgumentNullException(nameof(charset))).OrderBy(c => c, StringComparer.Ordinal).ToArray(), seed) { }

        /// <summary>The shared constructor: Gymnasium's validation, then the character tables.</summary>
        /// <param name="maxLength">The longest length.</param>
        /// <param name="minLength">The shortest length.</param>
        /// <param name="charList">The characters in sampling order.</param>
        /// <param name="seed">The seed.</param>
        /// <exception cref="NumSharp.ValueError">Thrown for invalid lengths or characters.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        private Text(long maxLength, long minLength, string[] charList, SpaceSeed? seed) : base(null, StringDType(), null) {
            if (minLength < 0)
                throw new ValueError($"Minimum text length must be non-negative, actual value: {minLength}");
            if (minLength > maxLength)
                throw new ValueError($"The min_length must be less than or equal to the max_length, min_length: {minLength}, max_length: {maxLength}");
            MinLength = minLength;
            MaxLength = maxLength;
            string[] invalid = charList.Where(c => CodePoints(c).Count() != 1).ToArray();
            if (invalid.Length > 0)
                throw new ValueError($"Expects all charset elements to be a single character, actual invalid elements: {PyBuiltins.Repr(invalid.ToList())}");
            _charList = charList;
            _charIndex = new Dictionary<string, int>(StringComparer.Ordinal);
            for (int i = 0; i < charList.Length; i++)
                _charIndex[charList[i]] = i;
            Characters = string.Concat(charList.OrderBy(c => c, StringComparer.Ordinal));
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The shortest length, inclusive (Gymnasium's <c>min_length</c>).</summary>
        public long MinLength { get; }

        /// <summary>The longest length, inclusive (Gymnasium's <c>max_length</c>).</summary>
        public long MaxLength { get; }

        /// <summary>The characters as a set (Gymnasium's <c>character_set</c>).</summary>
        public IReadOnlySet<string> CharacterSet => new HashSet<string>(_charList, StringComparer.Ordinal);

        /// <summary>The characters in sampling order (Gymnasium's <c>character_list</c>).</summary>
        public IReadOnlyList<string> CharacterList => _charList;

        /// <summary>Every character, sorted, in one string (Gymnasium's <c>characters</c>); the <c>repr</c> prints it.</summary>
        public string Characters { get; }

        /// <summary>A text flattens to the indices of its characters, padded to the longest length.</summary>
        public override bool IsNpFlattenable => true;

        /// <summary>A character's index in <see cref="CharacterList"/> (Gymnasium's <c>character_index</c>, an <c>np.int32</c>).</summary>
        /// <param name="character">The character.</param>
        /// <returns>Its index.</returns>
        /// <exception cref="NumSharp.KeyError">Thrown for a character outside the set.</exception>
        public int CharacterIndex(string character) => _charIndex.TryGetValue(character, out int index) ? index : throw new KeyError(PyBuiltins.Repr(character));

        /// <summary>
        ///     Draws a string (Gymnasium's <c>sample</c>): a length drawn from <c>[MinLength, MaxLength]</c> (or given), then each
        ///     character drawn from the charset, uniformly or under a mask or probability mask.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/>, or a tuple <c>(length, charlist_mask)</c>: a fixed length (or <see langword="null"/>)
        ///     and an int8 array over <see cref="CharacterList"/> whose 1s allow characters (or <see langword="null"/>).
        /// </param>
        /// <param name="probability"><see langword="null"/>, or a tuple <c>(length, probabilities)</c> with a float64 array summing to 1.</param>
        /// <returns>The string.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given, or when a mask allows no character and the minimum length is above 0.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for a malformed mask, with Gymnasium's messages.</exception>
        /// <remarks>
        ///     The characters are drawn by <c>choice(len(charset), size=length, p=p)</c>, which is exactly NumPy's
        ///     <c>choice(character_list, size=length, p=p)</c> draw (choice picks indices, then the items), so a text space seeded
        ///     like Gymnasium's draws the same strings. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/text.py#L103">gymnasium/spaces/text.py</see>.
        /// </remarks>
        public override string Sample(object? mask = null, object? probability = null) {
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");

            long? length = null;
            NDArray? weights = null;
            var owned = new List<NDArray>();
            try {
                if (mask is not null) {
                    (length, NDArray? charlistMask) = ValidateMask(mask, np.int8, "mask");
                    if (charlistMask is not null) {
                        using NDArray zero = charlistMask == 0;
                        using NDArray one = charlistMask == 1;
                        using NDArray binary = np.logical_or(zero, one);
                        if (!np.all(binary))
                            throw new AssertionError($"Expects all mask values to 0 or 1, actual values: {PyBuiltins.Str(charlistMask)}");
                        using NDArray total = np.sum(charlistMask);
                        // normalise the mask to use as a probability
                        if (Convert.ToDouble(total.GetAtIndex(0), CultureInfo.InvariantCulture) > 0) {
                            weights = charlistMask / total;
                            owned.Add(weights);
                        } else {
                            weights = charlistMask;
                        }
                    }
                } else if (probability is not null) {
                    (length, NDArray? charlistMask) = ValidateMask(probability, np.float64, "probability");
                    if (charlistMask is not null) {
                        using NDArray atLeastZero = charlistMask >= 0;
                        using NDArray atMostOne = charlistMask <= 1;
                        using NDArray inRange = np.logical_and(atLeastZero, atMostOne);
                        if (!np.all(inRange))
                            throw new AssertionError($"Expects all probability mask values to be within 0 and 1, actual values: {PyBuiltins.Str(charlistMask)}");
                        using NDArray total = np.sum(charlistMask);
                        using NDArray close = np.isclose(total, 1.0);
                        if (!close.GetBoolean())
                            throw new AssertionError($"Expects the sum of the probability mask to be 1, actual sum: {PyBuiltins.Str(total)}");
                        weights = charlistMask;
                    }
                }

                if (length is null) {
                    using NDArray drawn = NpRandom.integers(MinLength, MaxLength + 1);
                    length = Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
                }

                if (weights is null) {
                    // np.ones(len(self.character_set)) / len(self.character_set): float64 1/n each.
                    using NDArray ones = np.ones(new Shape((long)_charList.Length));
                    weights = ones / (double)_charList.Length;
                    owned.Add(weights);
                }

                using (NDArray allZero = weights == 0) {
                    if (np.all(allZero)) {
                        if (MinLength == 0)
                            return "";
                        throw new ValueError($"Trying to sample with a minimum length > 0 (actual minimum length={MinLength}) but the character mask is all zero meaning that no character could be sampled.");
                    }
                }

                NDArray p = weights.dtype == np.float64 ? weights : weights.astype(np.float64);
                if (!ReferenceEquals(p, weights))
                    owned.Add(p);
                using NDArray indices = NpRandom.choice(_charList.Length, new Shape(length.Value), p: p);
                var text = new StringBuilder();
                for (long i = 0; i < length.Value; i++)
                    text.Append(_charList[Convert.ToInt32(indices.GetAtIndex(i), CultureInfo.InvariantCulture)]);
                return text.ToString();
            } finally {
                foreach (NDArray array in owned)
                    array.Dispose();
            }
        }

        /// <summary>Whether <paramref name="x"/> is a string of the right length (in code points) made of the charset's characters.</summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        public override bool Contains(object? x) {
            if (x is not string text)
                return false;
            string[] characters = CodePoints(text).ToArray();
            return MinLength <= characters.Length && characters.Length <= MaxLength && characters.All(_charIndex.ContainsKey);
        }

        /// <summary>Gymnasium's <c>repr</c>: <c>Text(1, 10, charset=0123456789ABC…xyz)</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"Text({MinLength}, {MaxLength}, charset={Characters})";

        /// <summary>Gymnasium's <c>==</c>: the same lengths and character set (order aside).</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Text? other) => other is not null && MinLength == other.MinLength && MaxLength == other.MaxLength && Characters == other.Characters;

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Text other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(MinLength, MaxLength, StringComparer.Ordinal.GetHashCode(Characters));

        /// <summary>Gymnasium's <c>_validate_mask</c>: a <c>(length, charlist_mask)</c> tuple with the right types and shape.</summary>
        /// <param name="mask">The mask or probability argument.</param>
        /// <param name="expectedDType">int8 for a mask, float64 for a probability.</param>
        /// <param name="maskType"><c>mask</c> or <c>probability</c>, for the messages.</param>
        /// <returns>The fixed length (or <see langword="null"/>) and the character mask (or <see langword="null"/>).</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a malformed argument, with Gymnasium's messages.</exception>
        private (long? Length, NDArray? CharlistMask) ValidateMask(object mask, DType expectedDType, string maskType) {
            object?[] items = mask switch {
                object?[] array => array,
                System.Runtime.CompilerServices.ITuple tuple => Tuple.TupleItems(tuple),
                _ => throw new AssertionError($"Expects the `{maskType}` type to be a tuple, actual type: {PyBuiltins.TypeRepr(mask)}"),
            };
            if (items.Length != 2)
                throw new AssertionError($"Expects the `{maskType}` length to be two, actual length: {items.Length}");
            long? length = null;
            if (items[0] is not null) {
                if (!Discrete.TryToInt64(items[0], out long given) || items[0] is bool)
                    throw new AssertionError($"Expects the Text sample length to be an integer, actual type: {PyBuiltins.TypeRepr(items[0])}");
                if (given < MinLength || given > MaxLength)
                    throw new AssertionError($"Expects the Text sample length be between {MinLength} and {MaxLength}, actual length: {given}");
                length = given;
            }

            NDArray? charlistMask = null;
            if (items[1] is not null) {
                charlistMask = items[1] as NDArray ?? throw new AssertionError($"Expects the Text sample `{maskType}` to be an np.ndarray, actual type: {PyBuiltins.TypeRepr(items[1])}");
                if (charlistMask.dtype != expectedDType)
                    throw new AssertionError($"Expects the Text sample `{maskType}` to be type {DTypeClassRepr(expectedDType)}, actual dtype: {Discrete.DTypeName(charlistMask.dtype)}");
                if (charlistMask.ndim != 1 || charlistMask.shape[0] != _charList.Length)
                    throw new AssertionError($"expects the Text sample `{maskType}` to be ({_charList.Length},), actual shape: {ShapeRepr(charlistMask.Shape)}");
            }

            return (length, charlistMask);
        }

        /// <summary>How Python prints a NumPy scalar class, <c>&lt;class 'numpy.int8'&gt;</c>, as Gymnasium's message formats <c>expected_dtype</c>.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The text.</returns>
        private static string DTypeClassRepr(DType dtype) => $"<class 'numpy.{Discrete.DTypeName(dtype)}'>";

        /// <summary>A string's code points, each as a string (Python's iteration over a <c>str</c>).</summary>
        /// <param name="text">The string.</param>
        /// <returns>One string per code point.</returns>
        internal static IEnumerable<string> CodePoints(string text) => text.EnumerateRunes().Select(r => r.ToString());

        /// <summary>NumPy's <c>np.dtype(str)</c>, the unsized unicode dtype Gymnasium gives a text space, or <see langword="null"/> where NumSharp has none.</summary>
        /// <returns>The dtype, or <see langword="null"/>.</returns>
        private static DType? StringDType() {
            try {
                return np.dtype("<U0");
            } catch (Exception e) when (e is TypeError or ValueError or NotSupportedException or ArgumentException) {
                // The dtype only describes the space; nothing samples or flattens through it.
                return null;
            }
        }
    }
}
