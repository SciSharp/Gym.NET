using System;
using NumSharp;

namespace Gym.Spaces {
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
        public abstract void Seed(int seed);
    }
}