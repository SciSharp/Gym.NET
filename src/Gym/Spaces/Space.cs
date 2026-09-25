using System;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The set an env's actions or observations belong to: it records their <see cref="Shape"/> and <see cref="DType"/>,
    ///     tests membership (<see cref="Contains"/>) and draws random members (<see cref="Sample"/>) from a generator of its
    ///     own, so seeding a space never disturbs the env's stream.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/space.py#L18">gymnasium/spaces/space.py</see>.</remarks>
    public abstract class Space {
        public Shape Shape { get; protected set; }

        /// <summary>
        ///     The element type of every value in this space, as NumSharp's <see cref="NumSharp.DType"/> descriptor
        ///     (NumPy's <c>space.dtype</c>). Compare it with <c>==</c>, which also accepts a <see cref="Type"/>,
        ///     an <see cref="NPTypeCode"/> or a NumPy dtype string (<c>space.DType == typeof(float)</c> is structural and true
        ///     for float32).
        /// </summary>
        /// <remarks>
        ///     This was a <see cref="Type"/> while Gym.NET used NumSharp.Lite. <c>Type t = space.DType;</c> no longer compiles,
        ///     because NumSharp makes <c>DType</c> → <c>Type</c> explicit; write <c>(Type)space.DType</c> or
        ///     <c>space.DType.type</c>. Beware <c>np.@byte</c>: it is int8 (NumPy's C <c>char</c>), not uint8 as in Lite;
        ///     uint8 is <c>np.uint8</c>.
        /// </remarks>
        public DType DType { get; protected set; }

        /// <summary>
        ///     Records the shape and dtype every subclass must describe. Subclasses resolve their own defaults first
        ///     (float32 for <see cref="Box"/>), because a <see langword="null"/> <paramref name="dType"/> is stored as-is
        ///     and later reads of <see cref="DType"/> would then be <see langword="null"/>.
        /// </summary>
        /// <param name="shape">The shape of every sample; <see cref="NumSharp.Shape"/> is a struct, so it cannot be <see langword="null"/>.</param>
        /// <param name="dType">The element type of every sample.</param>
        protected Space(Shape shape, DType dType) {
            Shape = shape;
            DType = dType;
        }

        public abstract NDArray Sample(NDArray mask = null);
        public abstract bool Contains(object ndArray);

        /// <summary>
        ///     Gives this space a fresh generator seeded with <paramref name="seed"/>, so the samples that follow are
        ///     reproducible and independent of every other space and env, including any that shared its previous generator.
        /// </summary>
        /// <param name="seed">
        ///     Any non-negative seed, as in Gymnasium (a Python int with no upper bound), so an env's 128-bit
        ///     <c>np_random_seed</c> can seed a space (<c>space.seed(env.np_random_seed)</c>); every C# integer converts
        ///     implicitly.
        /// </param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <remarks>
        ///     The generator is Gymnasium's, <c>Generator(PCG64(SeedSequence(seed)))</c> (<see cref="Gym.Utils.Seeding"/>).
        ///     This is the only way to make an unseeded space reproducible: seeding the process-wide <c>np.random</c> has no
        ///     effect on spaces, each of which draws from its own generator (entropy-seeded unless one was given), as in
        ///     Gymnasium. The parameter was a <see langword="long"/> until the core-API work (P1); overrides must take a
        ///     <see cref="System.Numerics.BigInteger"/> now. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/spaces/space.py#L112">gymnasium/spaces/space.py</see>.
        /// </remarks>
        public abstract void Seed(System.Numerics.BigInteger seed);
    }
}