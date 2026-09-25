using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     The single factory for the legacy generators (NumPy's <c>RandomState</c>, MT19937) that Gym.NET's envs and spaces
    ///     draw from until F2 replaces them with <c>Generator(PCG64)</c>. Use it instead of <c>np.random.RandomState(int)</c>:
    ///     that overload takes only <see cref="int"/>, so the valid seeds above <see cref="int.MaxValue"/> cannot be passed,
    ///     and it validates nothing (<c>RandomState(-1)</c> silently seeds 4294967295).
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
    }
}
