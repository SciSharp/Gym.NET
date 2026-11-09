#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     Any space seen as a <c>Space&lt;object?&gt;</c>, for an env typed <c>Env&lt;object?, object?&gt;</c> whose spaces
    ///     vary (Gymnasium's generic test envs take any space). Everything forwards to <see cref="Inner"/>: sampling, membership,
    ///     seeding, the generator, <c>repr</c> and equality, so the view behaves as the space itself.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gym.NET's spaces are typed by their elements (DR-07), and a <c>Box</c> is a <c>Space&lt;NDArray&gt;</c>, which C#
    ///     cannot treat as a <c>Space&lt;object?&gt;</c> (classes are not covariant). This view bridges the two;
    ///     <c>((IEnv)env).ActionSpace = box</c> wraps automatically for an env whose action type is <see cref="object"/>.
    ///     </para>
    ///     <para>
    ///     The view owns the space it wraps: disposing the view (as its env does) disposes <see cref="Inner"/>.
    ///     </para>
    /// </remarks>
    public sealed class ObjectSpace : Space<object?>, IEquatable<ObjectSpace> {
        /// <summary>Wraps a space.</summary>
        /// <param name="inner">The space; owned by the view from now on.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="inner"/> is <see langword="null"/>.</exception>
        public ObjectSpace(Space inner) : base((inner ?? throw new ArgumentNullException(nameof(inner))).Shape, inner.DType, null) {
            // A view of a view would add a layer and nothing else.
            Inner = inner is ObjectSpace view ? view.Inner : inner;
        }

        /// <summary>The wrapped space.</summary>
        public Space Inner { get; }

        /// <summary>The wrapped space's generator, never one of the view's own.</summary>
        public override Generator NpRandom {
            get => Inner.NpRandom;
            set => Inner.NpRandom = value;
        }

        /// <inheritdoc />
        public override Generator? NpRandomOrNull => Inner.NpRandomOrNull;

        /// <inheritdoc />
        public override bool IsNpFlattenable => Inner.IsNpFlattenable;

        /// <summary>The wrapped space's sample, as it returns it.</summary>
        /// <param name="mask">The wrapped space's mask.</param>
        /// <param name="probability">The wrapped space's probability mask.</param>
        /// <returns>The element.</returns>
        public override object? Sample(object? mask = null, object? probability = null) => Inner.Sample(mask, probability);

        /// <inheritdoc />
        public override bool Contains(object? x) => Inner.Contains(x);

        /// <summary>The wrapped space's JSON conversion.</summary>
        /// <param name="samples">Samples of the wrapped space.</param>
        /// <returns>Their JSON-friendly values.</returns>
        public override object ToJsonable(IEnumerable<object?> samples) => Inner.ToJsonable(samples);

        /// <summary>The wrapped space's conversion back from JSON-friendly values.</summary>
        /// <param name="jsonable">The values.</param>
        /// <returns>The samples.</returns>
        public override IList<object?> FromJsonable(object jsonable) => Inner.FromJsonable(jsonable);

        /// <summary>Seeds the wrapped space.</summary>
        /// <param name="seed">The seed, or <see langword="null"/> for entropy.</param>
        /// <returns>The wrapped space's seed(s).</returns>
        protected override object SeedCore(BigInteger? seed) => Inner.Seed(seed);

        /// <summary>The wrapped space's <c>repr</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => Inner.ToString() ?? "";

        /// <summary>Equal to another view of an equal space, or to an equal space itself.</summary>
        /// <param name="other">The other view.</param>
        /// <returns><see langword="true"/> when the wrapped spaces are equal.</returns>
        public bool Equals(ObjectSpace? other) => other is not null && Inner.Equals(other.Inner);

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is ObjectSpace view ? Equals(view) : obj is Space space && Inner.Equals(space);

        /// <inheritdoc />
        public override int GetHashCode() => Inner.GetHashCode();

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing)
                Inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
