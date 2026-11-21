#nullable enable
using System;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     The base of the typed info records (DR-91, DR-94, DR-95): a built-in env's <c>info</c> read as a record with a
    ///     .NET-typed property per key, <c>HalfCheetahInfoV5 info = step.Info;</c> then <c>info.XPosition</c>, while the dict's
    ///     own values stay readable through the indexer, <c>(double)(NDArray)info["x_position"]</c>, as on
    ///     <c>step.Info</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The records are generated from positional records marked <c>[GymDict]</c> that derive from this class, as the env
    ///     classes derive from <c>Env</c> (the MuJoCo envs' from <c>MujocoInfo</c>, the toy_text envs' from
    ///     <c>ToyTextInfo</c>). A member is <see langword="double"/> for every scalar (a float32 or int value widens exactly),
    ///     <see langword="bool"/> for a flag, <see cref="NDArray"/> for an array, and nullable when the env does not always send
    ///     its key (a reset sends fewer keys than a step); <see cref="InfoExtensions.Get{T}"/> converts each value.
    ///     </para>
    ///     <para>
    ///     A record read from a dict keeps that dict's values (DR-94): the indexer and <see cref="ToInfo"/> give the values
    ///     as the env sent them, a float32 scalar as float32. A record built in code, or by a <c>with</c> expression, has no
    ///     dict inside: its indexer reads its own members, so a changed copy never shows stale values.
    ///     </para>
    ///     <para>
    ///     Equality (DR-92): <c>==</c> and <see cref="object.Equals(object)"/> compare the members, as records do (two records
    ///     with equal members are equal whatever dtypes their dicts held); <see cref="PyEquals"/> is Python's <c>==</c> with
    ///     a dict or another record.
    ///     </para>
    ///     <para>
    ///     Footgun: array members are the step result's own arrays, shared, not copied: disposing the result (DR-51)
    ///     disposes them too. Scalar members are copies and outlive it.
    ///     </para>
    /// </remarks>
    public abstract record EnvInfo {
        /// <summary>A record built in code, without a dict inside.</summary>
        protected EnvInfo() { }

        /// <summary>
        ///     The copy a <c>with</c> expression makes: everything but the dict inside, so a copy whose members changed reads
        ///     its own members through the indexer instead of the original dict's values.
        /// </summary>
        /// <param name="original">The record copied.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="original"/> is <see langword="null"/>.</exception>
        protected EnvInfo(EnvInfo original) {
            if (original is null)
                throw new ArgumentNullException(nameof(original));
            // Source is deliberately left unset: the copy's members may differ from the dict it would otherwise share.
        }

        /// <summary>
        ///     The dict the record was read from (DR-94), or <see langword="null"/> for a record built in code or by
        ///     <c>with</c>. The generated conversion from <see cref="Info"/> sets it; the record shares it, not a copy.
        /// </summary>
        protected Info? Source { get; init; }

        /// <summary>
        ///     The value under a key as the env sent it (a NumPy scalar as a 0-d array, a Python int as a <see langword="long"/>),
        ///     read from the dict inside; for a record without one, its member's value under that key's Python name.
        /// </summary>
        /// <param name="key">The key, Gymnasium's spelling (<c>"x_position"</c>).</param>
        /// <returns>The value; an array is the dict's own.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="KeyError">Thrown, with Python's message, for a key the info does not hold.</exception>
        public object? this[string key] => (Source ?? ToInfoCore())[key];

        /// <summary>Whether the info holds a key: the dict inside's keys, or a record's members that are not <see langword="null"/>.</summary>
        /// <param name="key">The key.</param>
        /// <returns><see langword="true"/> when the key is present.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="key"/> is <see langword="null"/>.</exception>
        public bool ContainsKey(string key) => (Source ?? ToInfoCore()).ContainsKey(key);

        /// <summary>
        ///     The record as an info dict: a shallow copy of the dict it was read from, the env's values and key order kept,
        ///     else a new dict of its members by Python name, in declaration order, the <see langword="null"/> ones left out.
        /// </summary>
        /// <returns>A new dict the caller owns; array values are shared, not copied.</returns>
        public Info ToInfo() => Source?.Copy() ?? ToInfoCore();

        /// <summary>
        ///     The record's members as a new dict by Python name, in declaration order, a <see langword="null"/> member left out
        ///     (its key absent): what a record without a dict inside reads, and what equality compares. Generated.
        /// </summary>
        /// <returns>A new dict.</returns>
        protected abstract Info ToInfoCore();

        /// <summary>
        ///     Python's <c>==</c> (DR-92) between this info and a dict (an <see cref="Info"/>, a <see cref="System.Collections.Generic.Dictionary{TKey, TValue}"/>)
        ///     or another record: the same keys with equal values (<see cref="PyEquality"/>), whatever the order and the
        ///     numbers' types.
        /// </summary>
        /// <param name="other">The other value; <see langword="null"/> (Python's <c>None</c>) equals no dict.</param>
        /// <returns><see langword="true"/> when the two are equal dicts.</returns>
        public bool PyEquals(object? other) => other is not null && PyEquality.AreEqual(ToInfo(), other is EnvInfo record ? record.ToInfo() : other);

        /// <summary>
        ///     The base's share of record equality: the same record type. Each generated record compares its members on top,
        ///     never the dict inside, so records of equal members are equal whatever their dicts held.
        /// </summary>
        /// <param name="other">The other record.</param>
        /// <returns><see langword="true"/> when both are of the same record type.</returns>
        public virtual bool Equals(EnvInfo? other) => other is not null && EqualityContract == other.EqualityContract;

        /// <summary>A hash consistent with <see cref="Equals(EnvInfo)"/>: the record type's.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => EqualityContract.GetHashCode();
    }
}
