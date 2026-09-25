using System;
using System.Globalization;
using System.Numerics;
using Gym.Exceptions;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Port of <c>gymnasium/utils/seeding.py</c> @ 9e04324: the one way Gym.NET's envs and spaces create their random
    ///     number generators. Every generator is NumPy's <c>Generator(PCG64(SeedSequence(seed)))</c>, NumSharp's bit-exact
    ///     port of it, so a Gym.NET space or env seeded with <c>s</c> draws exactly what Gymnasium's draws for seed
    ///     <c>s</c>, provided it makes the same calls.
    /// </summary>
    /// <remarks>
    ///     A generator is owned by whoever stores it and is not synchronized: give every env and space its own instance, as
    ///     Gymnasium gives each its own <c>np_random</c>. It replaces the former <c>LegacySeeding</c>, which built NumPy's
    ///     legacy <c>RandomState</c> (MT19937), a different generator family that Gymnasium no longer uses.
    /// </remarks>
    public static class Seeding {
        /// <summary>
        ///     Port of <c>seeding.np_random(seed)</c>: a new generator, and the seed it was built from. Seeded with a value,
        ///     it is reproducible; unseeded (<see langword="null"/>), it draws fresh OS entropy, as Gymnasium does when a
        ///     seed is not given.
        /// </summary>
        /// <param name="seed">
        ///     Any non-negative integer, of any size, like Gymnasium's Python-int seeds (every C# integer type converts
        ///     implicitly); <see langword="null"/> seeds from OS entropy.
        /// </param>
        /// <returns>
        ///     The generator, and <c>seed_seq.entropy</c>: <paramref name="seed"/> itself, or for an unseeded generator the
        ///     128-bit entropy it was drawn from, which is why it is a <see cref="BigInteger"/>. Passing that value back
        ///     rebuilds the same generator, except in the 1-in-2^32 case where the entropy's top 32-bit word is zero: NumPy
        ///     would then derive the pool from three words, while NumSharp drew four.
        /// </returns>
        /// <exception cref="Error">
        ///     Thrown when <paramref name="seed"/> is negative, with Gymnasium's message
        ///     "Seed must be greater or equal to zero, actual value: …".
        /// </exception>
        /// <remarks>
        ///     Mirrors <c>seeding.py:10</c>: validate, build <c>SeedSequence(seed)</c>, report <c>seed_seq.entropy</c>, and
        ///     wrap <c>PCG64(seed_seq)</c> in a <c>Generator</c>. Gymnasium's other check, "Seed must be a python integer",
        ///     cannot fail for a typed integer.
        /// </remarks>
        public static (Generator Rng, BigInteger Seed) NpRandom(BigInteger? seed = null) {
            if (seed is { Sign: < 0 })
                throw new Error($"Seed must be greater or equal to zero, actual value: {seed.Value.ToString(CultureInfo.InvariantCulture)}");
            // NumSharp's SeedSequence(BigInteger) splits the integer into the same little-endian uint32 words as NumPy's
            // _int_to_uint32_array, whatever its size, so one path serves every seed.
            var seedSeq = seed.HasValue ? new SeedSequence(seed.Value) : new SeedSequence();
            return (new Generator(new PCG64(seedSeq)), Entropy(seedSeq));
        }

        /// <summary>
        ///     NumPy's <c>SeedSequence.entropy</c> as an integer. NumSharp keeps the entropy in the form it was given (a
        ///     <see langword="long"/>, a <see cref="BigInteger"/>), except for an unseeded sequence, where it keeps the four
        ///     32-bit words it drew; NumPy holds the integer those words encode, least significant word first.
        /// </summary>
        /// <param name="seedSeq">The sequence a generator was built from.</param>
        /// <returns>The entropy as a non-negative integer.</returns>
        /// <exception cref="InvalidOperationException">Thrown for an entropy form this port never creates.</exception>
        private static BigInteger Entropy(SeedSequence seedSeq) {
            switch (seedSeq.entropy) {
                case long l:
                    return l;
                case ulong ul:
                    return ul;
                case BigInteger b:
                    return b;
                case uint[] words:
                    var value = BigInteger.Zero;
                    for (int i = words.Length - 1; i >= 0; i--)
                        value = (value << 32) | words[i];
                    return value;
                default:
                    throw new InvalidOperationException($"Unexpected SeedSequence entropy of type {seedSeq.entropy?.GetType().Name ?? "null"}.");
            }
        }
    }
}
