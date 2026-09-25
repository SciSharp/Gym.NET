using System;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     The single factory for the legacy generators (NumPy's <c>RandomState</c>, MT19937) that Gym.NET's envs and spaces
    ///     draw from until F2 replaces them with <c>Generator(PCG64)</c>. Envs and spaces never touch <c>np.random</c> or
    ///     <c>np.random.RandomState(...)</c> directly: <c>RandomState(int)</c> cannot take the valid seeds above
    ///     <see cref="int.MaxValue"/> and validates nothing (<c>RandomState(-1)</c> silently seeds 4294967295), the global
    ///     <c>np.random</c> is shared and not thread-safe, and an unseeded <c>RandomState()</c> repeats the stream of every
    ///     other instance created in the same millisecond.
    /// </summary>
    /// <remarks>
    ///     A generator from this factory is owned by whoever stores it and is not thread-safe: NumSharp's MT19937, unlike
    ///     NumPy's <c>RandomState</c>, takes no lock. Give every env and space its own instance, as Gymnasium gives each its
    ///     own <c>np_random</c>.
    /// </remarks>
    public static class LegacySeeding {
        /// <summary>
        ///     Creates a generator whose stream is bit-identical to <c>numpy.random.RandomState(seed)</c>.
        /// </summary>
        /// <param name="seed">
        ///     A Python-int seed in <c>[0, 2**32 - 1]</c>. It is <see langword="long"/> because the values above
        ///     <see cref="int.MaxValue"/> are valid seeds.
        /// </param>
        /// <returns>A new generator that no other instance shares.</returns>
        /// <exception cref="ValueError">
        ///     Thrown when <paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c>, with NumPy's message
        ///     "Seed must be between 0 and 2**32 - 1".
        /// </exception>
        /// <remarks>
        ///     NumSharp offers only <c>RandomState(int)</c>, so a fresh instance is reseeded through <c>seed(long)</c>, the
        ///     overload that range-checks with NumPy's message (upstream item U3 in
        ///     docs/plans/NUMSHARP_MIGRATION_PLAN.md §14). Do not read the generator's <c>Seed</c> property back: it is an
        ///     <see cref="int"/> and wraps for seeds of 2^31 and above (U7).
        /// </remarks>
        public static NumPyRandom RandomState(long seed) {
            var rng = np.random.RandomState();
            // Binds to seed(long), the overload that accepts [2^31, 2^32 - 1] and rejects everything outside [0, 2^32 - 1]
            // with NumPy's ValueError; the unseeded instance it replaces is TickCount-seeded and never drawn from.
            rng.seed(seed);
            return rng;
        }

        /// <summary>
        ///     Creates a generator seeded from the operating system's CSPRNG, as NumPy does for <c>RandomState(None)</c>: its
        ///     stream is independent of every other instance and deliberately not reproducible. It is the fallback of every env
        ///     and space created without a seed or a generator.
        /// </summary>
        /// <returns>A new generator that no other instance shares.</returns>
        /// <remarks>
        ///     <para>
        ///     Not <c>np.random.RandomState()</c>: NumSharp seeds that from <see cref="Environment.TickCount"/>, so
        ///     instances created within the same millisecond share a stream (1,000 of 1,000 back-to-back pairs in the trial,
        ///     docs/plans/NUMSHARP_MIGRATION_PLAN.md §9.4). An env's action and observation spaces, or every env a loop
        ///     creates, would then draw identical values (upstream item U2).
        ///     </para>
        ///     <para>
        ///     Seeds MT19937 through <c>init_by_array</c> over four OS-random 32-bit words (128 bits), the routine NumPy uses
        ///     for an array seed. NumPy instead fills the whole key from <c>SeedSequence()</c> entropy
        ///     (<c>numpy/random/_mt19937.pyx:130</c>); neither stream can be reproduced, so only independence matters.
        ///     </para>
        /// </remarks>
        public static NumPyRandom FromEntropy() {
            var key = new uint[4];
            RandomNumberGenerator.Fill(MemoryMarshal.AsBytes(key.AsSpan()));
            var rng = np.random.RandomState();
            // seed(uint[]) runs init_by_array, which rebuilds the whole 624-word state from the key, so nothing of the
            // TickCount seeding the fresh instance started with survives.
            rng.seed(key);
            return rng;
        }
    }
}
