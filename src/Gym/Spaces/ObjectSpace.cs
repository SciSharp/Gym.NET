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

        /// <summary>
        ///     The space a value of an env's space really is: the wrapped space for a view, the space itself otherwise. Code
        ///     that dispatches on the space's class (<c>isinstance(env.observation_space, spaces.Box)</c>) must look through
        ///     the view, or an untyped env's <c>Box</c> would read as no <c>Box</c> at all.
        /// </summary>
        /// <param name="space">A space, a view of one, or <see langword="null"/>.</param>
        /// <returns><see cref="Inner"/> for a view, else <paramref name="space"/> unchanged (<see langword="null"/> stays <see langword="null"/>).</returns>
        public static Space? Unview(Space? space) => space is ObjectSpace view ? view.Inner : space;

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

        /// <summary>
        ///     Takes the view and the space it wraps for <paramref name="owner"/> (DR-40): the view disposes that space, so
        ///     whoever owns the view owns it too, and an env viewing a space another env owns is refused as sharing it would be.
        /// </summary>
        /// <param name="owner">The env taking the view.</param>
        /// <param name="attribute">The slot, in Python spelling, for the error.</param>
        /// <exception cref="InvalidOperationException">Thrown when another env that has not been disposed owns the view or the wrapped space.</exception>
        internal override void TakeOwnership(object owner, string attribute) {
            // The view's own refusal first, then the wrapped space's, so a refused take leaves the wrapped space untouched.
            object? viewOwner = Owner;
            if (viewOwner is not null && !ReferenceEquals(viewOwner, owner))
                base.TakeOwnership(owner, attribute);
            Inner.TakeOwnership(owner, attribute);
            base.TakeOwnership(owner, attribute);
        }

        /// <summary>Ends <paramref name="owner"/>'s ownership of the view and of the space it wraps.</summary>
        /// <param name="owner">The env letting the view go.</param>
        internal override void ReleaseOwnership(object owner) {
            base.ReleaseOwnership(owner);
            Inner.ReleaseOwnership(owner);
        }

        /// <summary>
        ///     Whether <paramref name="owner"/> owns the view or the space it wraps: a new view of an env's space counts as that
        ///     env's, so a wrapper handed one passes it through instead of disposing the env's space.
        /// </summary>
        /// <param name="owner">The env asked about.</param>
        /// <returns><see langword="true"/> when <paramref name="owner"/> owns either.</returns>
        internal override bool IsOwnedBy(object owner) => base.IsOwnedBy(owner) || Inner.IsOwnedBy(owner);

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing)
                Inner.Dispose();
            base.Dispose(disposing);
        }
    }
}
