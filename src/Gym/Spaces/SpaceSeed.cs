#nullable enable
using System;
using System.Globalization;
using System.Numerics;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     What a space constructor's <c>seed</c> accepts, Gymnasium's <c>int | np.random.Generator | None</c>: an integer seed
    ///     for a fresh generator of the space's own, or an existing generator the space then shares. Callers never name this
    ///     type: <c>seed: 42</c>, <c>seed: someBigInteger</c> and <c>seed: generator</c> convert implicitly, and a
    ///     <see cref="Nullable{T}"/> <see langword="null"/> is <c>None</c>.
    /// </summary>
    /// <remarks>
    ///     Sharing a generator couples two streams (each draw by one moves the other), which is why Gymnasium allows it only
    ///     on purpose; generators are not synchronized, so a shared one must stay on one thread.
    /// </remarks>
    public readonly struct SpaceSeed : IEquatable<SpaceSeed> {
        /// <summary>Creates a seed from an integer.</summary>
        /// <param name="value">The seed; validated (non-negative) when the space seeds itself, as Gymnasium does.</param>
        private SpaceSeed(BigInteger value) {
            Value = value;
            Generator = null;
        }

        /// <summary>Creates a seed that shares an existing generator.</summary>
        /// <param name="generator">The generator.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="generator"/> is <see langword="null"/>; pass no seed for <c>None</c>.</exception>
        private SpaceSeed(Generator generator) {
            Value = default;
            Generator = generator ?? throw new ArgumentNullException(nameof(generator));
        }

        /// <summary>The integer seed, when <see cref="Generator"/> is <see langword="null"/>.</summary>
        public BigInteger Value { get; }

        /// <summary>The generator to share, or <see langword="null"/> for an integer seed.</summary>
        public Generator? Generator { get; }

        /// <summary>An integer seed.</summary>
        /// <param name="value">The seed.</param>
        public static implicit operator SpaceSeed(BigInteger value) => new(value);

        /// <summary>An integer seed.</summary>
        /// <param name="value">The seed.</param>
        public static implicit operator SpaceSeed(long value) => new(value);

        /// <summary>An integer seed.</summary>
        /// <param name="value">The seed.</param>
        public static implicit operator SpaceSeed(ulong value) => new(value);

        /// <summary>An integer seed.</summary>
        /// <param name="value">The seed.</param>
        public static implicit operator SpaceSeed(int value) => new(value);

        /// <summary>
        ///     A shared generator, or no seed for a <see langword="null"/> one: <c>seed: maybeGenerator</c> then means
        ///     Gymnasium's <c>seed=None</c> when the variable is unset.
        /// </summary>
        /// <param name="generator">The generator, or <see langword="null"/>.</param>
        /// <remarks>
        ///     The conversion targets <see cref="Nullable{T}"/> on purpose. Into the struct itself, a <see langword="null"/>
        ///     generator would have to throw, and C# also routes a bare <see langword="null"/> literal through this operator
        ///     wherever <c>SpaceSeed</c> is the natural type (<c>x is null ? null : (SpaceSeed)x</c>), which then threw on the
        ///     branch meant to be "no seed".
        /// </remarks>
        public static implicit operator SpaceSeed?(Generator? generator) => generator is null ? null : new SpaceSeed(generator);

        /// <inheritdoc />
        public bool Equals(SpaceSeed other) => Value.Equals(other.Value) && ReferenceEquals(Generator, other.Generator);

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is SpaceSeed other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => Generator is null ? Value.GetHashCode() : Generator.GetHashCode();

        /// <inheritdoc />
        public override string ToString() => Generator is null ? Value.ToString(CultureInfo.InvariantCulture) : "Generator(PCG64)";
    }
}
