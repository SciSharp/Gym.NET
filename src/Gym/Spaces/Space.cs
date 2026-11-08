#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The untyped view of every space: the set an env's actions or observations belong to. It records their
    ///     <see cref="Shape"/> and <see cref="DType"/> (when its elements are arrays), tests membership (<see cref="Contains"/>)
    ///     and draws random members (<see cref="Sample"/>) from a generator of its own, so seeding a space never disturbs the
    ///     env's stream. Code that knows the element type uses <see cref="Space{T}"/>, which every space derives from.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A space owns the arrays it holds (a <see cref="Box"/>'s bounds) and releases them in <see cref="Dispose()"/>; an
    ///     env disposes its spaces when it is disposed (D19). Samples are new arrays the caller owns.
    ///     </para>
    ///     <para>
    ///     Not thread-safe: the generator is not synchronized, so sample a space from one thread at a time, as Gymnasium's
    ///     single-threaded envs do. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/space.py#L18">gymnasium/spaces/space.py</see>.
    ///     </para>
    /// </remarks>
    public abstract class Space : IDisposable {
        /// <summary>The generator, created on first use (Gymnasium's <c>_np_random</c>); <see langword="null"/> until then.</summary>
        private Generator? _npRandom;

        /// <summary>Whether <see cref="Dispose()"/> already ran, so a second call is a no-op.</summary>
        private bool _disposed;

        /// <summary>
        ///     Records the shape and dtype and seeds the generator when a seed is given; only <see cref="Space{T}"/> derives
        ///     from this class.
        /// </summary>
        /// <param name="shape">The shape of every element, or <see langword="null"/> for spaces whose elements are not arrays (Python's <c>None</c>).</param>
        /// <param name="dtype">The element dtype, or <see langword="null"/> for spaces whose elements are not arrays.</param>
        /// <param name="seed">
        ///     <see langword="null"/> to create the generator lazily from OS entropy on first use; an integer to seed one now;
        ///     a generator to share.
        /// </param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        private protected Space(Shape? shape, DType? dtype, SpaceSeed? seed) {
            Shape = shape;
            DType = dtype;
            if (seed is { } given) {
                if (given.Generator != null)
                    _npRandom = given.Generator;
                else
                    // Through the virtual Seed, as Gymnasium's __init__ calls self.seed(seed): a composite seeds its subspaces.
                    SeedCore(given.Value);
            }
        }

        /// <summary>
        ///     The shape of every element (Gymnasium's <c>shape</c>), or <see langword="null"/> for spaces whose elements are not
        ///     arrays (<c>Tuple</c>, <c>Dict</c>, <c>Text</c>, ...).
        /// </summary>
        public Shape? Shape { get; }

        /// <summary>
        ///     The element type of every element, as NumSharp's <see cref="NumSharp.DType"/> (Gymnasium's <c>dtype</c>), or
        ///     <see langword="null"/> for spaces whose elements are not arrays. Compare it with <c>==</c>, which also accepts a
        ///     <see cref="Type"/>, an <see cref="NPTypeCode"/> or a NumPy dtype string.
        /// </summary>
        /// <remarks>
        ///     Beware <c>np.@byte</c>: it is int8 (NumPy's C <c>char</c>), not uint8; uint8 is <c>np.uint8</c>.
        /// </remarks>
        public DType? DType { get; }

        /// <summary>
        ///     The generator this space samples from (Gymnasium's <c>np_random</c>), created from OS entropy on first use when
        ///     none was seeded or shared. Assigning one shares it, as Gymnasium's <c>space._np_random = rng</c> does.
        /// </summary>
        /// <remarks>
        ///     Lazily created because seeding costs a <c>SeedSequence</c>, which a space that is never sampled should not pay;
        ///     reading it is therefore not free the first time, and it draws OS entropy.
        /// </remarks>
        public virtual Generator NpRandom {
            get {
                if (_npRandom is null)
                    SeedCore(null);
                // A composite's seed seeds its subspaces and may leave its own generator unset (Gymnasium's comment in
                // Space.np_random), so one is drawn from entropy here rather than returning null.
                return _npRandom ??= Seeding.NpRandom().Rng;
            }
            set => _npRandom = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>
        ///     The generator if one exists, without creating it (Gymnasium's <c>_np_random</c>); for code that must not draw OS
        ///     entropy just to look, such as a parity check of an unseeded space.
        /// </summary>
        public virtual Generator? NpRandomOrNull => _npRandom;

        /// <summary>Whether this space flattens to a <see cref="Box"/> (Gymnasium's <c>is_np_flattenable</c>).</summary>
        public abstract bool IsNpFlattenable { get; }

        /// <summary>
        ///     Draws a random element, untyped: what <see cref="Space{T}.Sample"/> returns, boxed. A mask restricts which
        ///     elements are drawn and a probability mask weights them; each space documents the forms it accepts.
        /// </summary>
        /// <param name="mask">A space-specific mask (an int8 array for <see cref="Discrete"/>), or <see langword="null"/>.</param>
        /// <param name="probability">A space-specific probability mask (a float64 array for <see cref="Discrete"/>), or <see langword="null"/>.</param>
        /// <returns>The element; arrays are new and owned by the caller.</returns>
        public object? Sample(object? mask = null, object? probability = null) => SampleObject(mask, probability);

        /// <summary>The typed sample, boxed; implemented once by <see cref="Space{T}"/>.</summary>
        /// <param name="mask">The mask.</param>
        /// <param name="probability">The probability mask.</param>
        /// <returns>The element.</returns>
        private protected abstract object? SampleObject(object? mask, object? probability);

        /// <summary>
        ///     Reseeds the generator (Gymnasium's <c>seed</c>): with a value, a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>
        ///     that draws what Gymnasium's space draws after <c>space.seed(seed)</c>; without one, a fresh generator from OS
        ///     entropy. A generator the space shared is replaced, never reseeded, so the env's stream is left untouched.
        /// </summary>
        /// <param name="seed">Any non-negative integer (Gymnasium's seeds are unbounded Python ints), or <see langword="null"/>.</param>
        /// <returns>
        ///     The seed used: <paramref name="seed"/> itself, or the 128-bit entropy of an unseeded call, which reproduces it.
        ///     Composite spaces return one seed per subspace (a list, or a dict for <c>Dict</c>), as Gymnasium's do.
        /// </returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/space.py#L112">gymnasium/spaces/space.py</see>.</remarks>
        public object Seed(BigInteger? seed = null) => SeedCore(seed);

        /// <summary>
        ///     The seeding behind <see cref="Seed"/>; composite spaces override it to seed their subspaces. Called from the
        ///     constructor for an integer seed, so an override must not rely on state its own constructor sets after the base
        ///     call (composites pass their subspaces to the base first for this reason).
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/> for entropy.</param>
        /// <returns>The seed(s) used.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative.</exception>
        protected virtual object SeedCore(BigInteger? seed) {
            (Generator rng, BigInteger used) = Seeding.NpRandom(seed);
            _npRandom = rng;
            return used;
        }

        /// <summary>Replaces the generator with one the caller made, for composite spaces seeding themselves; never shares by accident.</summary>
        /// <param name="generator">The generator; <see langword="null"/> forgets it, so the next use draws entropy.</param>
        protected void SetNpRandom(Generator? generator) => _npRandom = generator;

        /// <summary>
        ///     Whether <paramref name="x"/> is an element of this space (Gymnasium's <c>contains</c> and <c>x in space</c>).
        ///     Values of the wrong kind return <see langword="false"/> rather than throwing, as in Gymnasium.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        public abstract bool Contains(object? x);

        /// <summary>
        ///     Converts samples to JSON-friendly values (Gymnasium's <c>to_jsonable</c>), untyped: lists of numbers, strings and
        ///     nested lists or dictionaries.
        /// </summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>One JSON-friendly value per sample.</returns>
        /// <exception cref="InvalidCastException">Thrown when a sample is not of this space's element type.</exception>
        public IList<object?> ToJsonable(IEnumerable<object?> samples) => ToJsonableObject(samples);

        /// <summary>Converts JSON-friendly values back to samples (Gymnasium's <c>from_jsonable</c>), untyped.</summary>
        /// <param name="jsonable">Values <see cref="ToJsonable"/> produced.</param>
        /// <returns>The samples, boxed.</returns>
        public IList<object?> FromJsonable(IList<object?> jsonable) => FromJsonableObject(jsonable);

        /// <summary>The typed conversion, boxed; implemented once by <see cref="Space{T}"/>.</summary>
        /// <param name="samples">The samples.</param>
        /// <returns>The JSON-friendly values.</returns>
        private protected abstract IList<object?> ToJsonableObject(IEnumerable<object?> samples);

        /// <summary>The typed conversion, boxed; implemented once by <see cref="Space{T}"/>.</summary>
        /// <param name="jsonable">The JSON-friendly values.</param>
        /// <returns>The samples, boxed.</returns>
        private protected abstract IList<object?> FromJsonableObject(IList<object?> jsonable);

        /// <summary>Releases the arrays this space owns; idempotent. The space must not be used afterwards.</summary>
        public void Dispose() {
            if (_disposed)
                return;
            _disposed = true;
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>Releases what a derived space owns (a <see cref="Box"/>'s bounds); called once, by <see cref="Dispose()"/>.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Dispose()"/>; no space has a finalizer.</param>
        protected virtual void Dispose(bool disposing) { }

        /// <summary>A shape as Python prints the tuple: <c>()</c>, <c>(2,)</c>, <c>(3, 4)</c>; <c>None</c> for no shape.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The text.</returns>
        internal static string ShapeRepr(Shape? shape) {
            if (shape is not { } s)
                return "None";
            long[] dimensions = s.dimensions ?? Array.Empty<long>();
            if (dimensions.Length == 1)
                return $"({dimensions[0].ToString(CultureInfo.InvariantCulture)},)";
            return "(" + string.Join(", ", dimensions.Select(d => d.ToString(CultureInfo.InvariantCulture))) + ")";
        }
    }

    /// <summary>
    ///     A space whose elements are <typeparamref name="T"/>: <see cref="Box"/> and <see cref="Discrete"/> are
    ///     <c>Space&lt;NDArray&gt;</c> (a discrete element is a 0-d int64 array, NumPy's <c>np.int64</c>, DR-08). An env's
    ///     <c>ActionSpace</c> and <c>ObservationSpace</c> are typed this way (DR-07), and custom spaces derive from it.
    /// </summary>
    /// <typeparam name="T">The element type, what <see cref="Sample"/> returns.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/space.py#L18">gymnasium/spaces/space.py</see>.</remarks>
    public abstract class Space<T> : Space {
        /// <summary>Records the shape and dtype and seeds the generator when a seed is given.</summary>
        /// <param name="shape">The shape of every element, or <see langword="null"/> when elements are not arrays.</param>
        /// <param name="dtype">The element dtype, or <see langword="null"/> when elements are not arrays.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/> for a lazy entropy-seeded one.</param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        protected Space(Shape? shape, DType? dtype, SpaceSeed? seed) : base(shape, dtype, seed) { }

        /// <summary>
        ///     Draws a random element (Gymnasium's <c>sample</c>). A mask restricts which elements are drawn and a probability
        ///     mask weights them; the two cannot be combined, and each space documents the forms it accepts.
        /// </summary>
        /// <param name="mask">A space-specific mask, or <see langword="null"/>.</param>
        /// <param name="probability">A space-specific probability mask, or <see langword="null"/>.</param>
        /// <returns>The element; arrays are new and owned by the caller.</returns>
        public new abstract T Sample(object? mask = null, object? probability = null);

        /// <summary>
        ///     Converts samples to JSON-friendly values (Gymnasium's <c>to_jsonable</c>); the default keeps them as they are,
        ///     as Gymnasium's base does.
        /// </summary>
        /// <param name="samples">The samples.</param>
        /// <returns>One JSON-friendly value per sample.</returns>
        public virtual IList<object?> ToJsonable(IEnumerable<T> samples) => samples.Select(s => (object?)s).ToList();

        /// <summary>Converts JSON-friendly values back to samples (Gymnasium's <c>from_jsonable</c>); the default casts them.</summary>
        /// <param name="jsonable">The values.</param>
        /// <returns>The samples.</returns>
        /// <exception cref="InvalidCastException">Thrown by the default when a value is not a <typeparamref name="T"/>.</exception>
        public virtual IList<T> FromJsonable(IList<object?> jsonable) => jsonable.Select(v => (T)v!).ToList();

        /// <inheritdoc />
        private protected sealed override object? SampleObject(object? mask, object? probability) => Sample(mask, probability);

        /// <inheritdoc />
        private protected sealed override IList<object?> ToJsonableObject(IEnumerable<object?> samples) => ToJsonable(samples.Select(s => (T)s!));

        /// <inheritdoc />
        private protected sealed override IList<object?> FromJsonableObject(IList<object?> jsonable) =>
            FromJsonable(jsonable).Select(s => (object?)s).ToList();
    }
}
