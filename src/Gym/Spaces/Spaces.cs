#nullable enable
using System;
using System.Collections.Generic;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.spaces</c> module as one static class (DR-79, DR-86): with <c>using Gym.Spaces;</c>,
    ///     <c>Spaces.Discrete(4, seed: 0)</c> reads as <c>spaces.Discrete(4, seed=0)</c> after <c>from gymnasium import
    ///     spaces</c>. Each factory builds the class of its name exactly as its constructor does, and the module's
    ///     functions are here too: <see cref="Flatdim"/>, <see cref="Flatten"/>, <see cref="FlattenSpace"/>,
    ///     <see cref="Unflatten"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The constructors stay the primary way in C# (<c>new Discrete(4)</c>); this class is for code that reads like
    ///     Gymnasium's, or that wants one entry point to discover the spaces from. It mirrors the module's
    ///     <c>__all__</c>: the ten spaces, <c>GraphInstance</c> and the four flattening functions (<c>Space</c>, the abstract
    ///     base, has no factory).
    ///     </para>
    ///     <para>
    ///     Every space a factory returns is the caller's: an env takes over a space assigned to it (DR-40), anything else is
    ///     disposed by whoever created it. A composite factory takes ownership of the subspaces it is given, as the
    ///     constructor does.
    ///     </para>
    ///     <para>
    ///     Footgun: the class shares its namespace's name. Outside namespace <c>Gym</c> that is harmless; inside namespace
    ///     <c>Gym.Spaces</c> the simple name <c>Spaces</c> is this class, and inside any other <c>Gym.*</c> namespace it is
    ///     the namespace, so write <c>Gym.Spaces.Spaces.Discrete(4)</c> there, or <c>new Discrete(4)</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/__init__.py">gymnasium/spaces/__init__.py</see>.</para>
    /// </remarks>
    public static class Spaces {
        /// <summary>A box whose every entry lies in [<paramref name="low"/>, <paramref name="high"/>] (scalar bounds): <c>spaces.Box(low, high, shape, dtype, seed)</c>.</summary>
        /// <param name="low">The lower bound of every entry; <c>double.NegativeInfinity</c> for none.</param>
        /// <param name="high">The upper bound of every entry; <c>double.PositiveInfinity</c> for none.</param>
        /// <param name="shape">The shape; <see langword="null"/> is <c>(1,)</c>, as for Gymnasium's scalar bounds.</param>
        /// <param name="dtype">The dtype: integer, floating or bool; <see langword="null"/> is float32, Gymnasium's default.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/> for a lazy entropy-seeded one.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown for a NaN bound, a bound outside the dtype, an infinite bound of an unsigned or bool dtype, <c>low &gt; high</c>, or a dtype that is not integer, floating or bool, with Gymnasium's messages.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Box Box(double low, double high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null) => new(low, high, shape, dtype, seed);

        /// <summary>A box with per-entry bounds: <c>spaces.Box(low_array, high_array, ...)</c>; the shape is the arrays' unless <paramref name="shape"/> is given.</summary>
        /// <param name="low">The lower bounds; read, never kept (the box stores a copy in its dtype).</param>
        /// <param name="high">The upper bounds; read, never kept.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to take the bounds'.</param>
        /// <param name="dtype">The dtype; <see langword="null"/> is float32.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown for shapes that differ, a NaN bound, a bound outside the dtype, an infinite bound of an unsigned or bool dtype, or <c>low &gt; high</c>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Box Box(NDArray low, NDArray high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null) => new(low, high, shape, dtype, seed);

        /// <summary>A box with per-entry lower bounds and a scalar upper bound.</summary>
        /// <param name="low">The lower bounds; read, never kept.</param>
        /// <param name="high">The upper bound of every entry.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to take <paramref name="low"/>'s.</param>
        /// <param name="dtype">The dtype; <see langword="null"/> is float32.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown as for the other boxes.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Box Box(NDArray low, double high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null) => new(low, high, shape, dtype, seed);

        /// <summary>A box with a scalar lower bound and per-entry upper bounds.</summary>
        /// <param name="low">The lower bound of every entry.</param>
        /// <param name="high">The upper bounds; read, never kept.</param>
        /// <param name="shape">The shape, or <see langword="null"/> to take <paramref name="high"/>'s.</param>
        /// <param name="dtype">The dtype; <see langword="null"/> is float32.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown as for the other boxes.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Box Box(double low, NDArray high, Shape? shape = null, DType? dtype = null, SpaceSeed? seed = null) => new(low, high, shape, dtype, seed);

        /// <summary>The integers <c>start</c> to <c>start + n - 1</c>: <c>spaces.Discrete(n, seed=seed, start=start, dtype=dtype)</c>.</summary>
        /// <param name="n">The number of elements; must be positive.</param>
        /// <param name="seed">An integer seed (the space then samples what Gymnasium's does), a generator to share, or <see langword="null"/>.</param>
        /// <param name="start">The smallest element.</param>
        /// <param name="dtype">The integer dtype of the elements; <see langword="null"/> is int64, Gymnasium's default. Samples are drawn in it.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown when <paramref name="n"/> is not positive.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="dtype"/> is not an integer dtype.</exception>
        /// <exception cref="OverflowError">Thrown when <paramref name="n"/> or <paramref name="start"/> does not fit <paramref name="dtype"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Discrete Discrete(long n, SpaceSeed? seed = null, long start = 0, DType? dtype = null) => new(n, seed, start, dtype);

        /// <summary>A flat array of <paramref name="n"/> binary values: <c>spaces.MultiBinary(n)</c>.</summary>
        /// <param name="n">The number of values; must be positive.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown when <paramref name="n"/> is not positive.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static MultiBinary MultiBinary(long n, SpaceSeed? seed = null) => new(n, seed);

        /// <summary>Binary arrays of the shape <paramref name="n"/>: <c>spaces.MultiBinary([3, 2])</c>.</summary>
        /// <param name="n">The shape; every dimension must be positive.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown when a dimension is not positive.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="n"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static MultiBinary MultiBinary(long[] n, SpaceSeed? seed = null) => new(n, seed);

        /// <summary>One discrete choice per entry of a flat array: <c>spaces.MultiDiscrete(nvec, dtype, seed, start)</c>.</summary>
        /// <param name="nvec">The number of choices of each entry; read, never kept.</param>
        /// <param name="dtype">The integer dtype; <see langword="null"/> is int64.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <param name="start">The smallest value of each entry, or <see langword="null"/> for zeros; read, never kept.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown for a non-integer dtype, a start of another length, or a count that is not positive.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="nvec"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static MultiDiscrete MultiDiscrete(long[] nvec, DType? dtype = null, SpaceSeed? seed = null, long[]? start = null) => new(nvec, dtype, seed, start);

        /// <summary>One discrete choice per entry of an array of any shape, <paramref name="nvec"/>'s.</summary>
        /// <param name="nvec">The counts, any shape; read, never kept (the space stores a copy in its dtype).</param>
        /// <param name="dtype">The integer dtype; <see langword="null"/> is int64.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <param name="start">The smallest values, of <paramref name="nvec"/>'s shape, or <see langword="null"/> for zeros; read, never kept.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown for a non-integer dtype, a start of another shape, or a count that is not positive.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="nvec"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static MultiDiscrete MultiDiscrete(NDArray nvec, DType? dtype = null, SpaceSeed? seed = null, NDArray? start = null) => new(nvec, dtype, seed, start);

        /// <summary>The product of <paramref name="spaces"/>: <c>spaces.Tuple(spaces, seed=seed)</c>; the new space owns the subspaces.</summary>
        /// <param name="spaces">The subspaces, in order.</param>
        /// <param name="seed">An integer seed (the tuple's generator then draws one <c>int32</c> seed per subspace), a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Tuple Tuple(IEnumerable<Space> spaces, SpaceSeed? seed = null) => new(spaces, seed);

        /// <summary>The product of the given subspaces: <c>spaces.Tuple((a, b))</c>; the new space owns them.</summary>
        /// <param name="spaces">The subspaces, in order.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        public static Tuple Tuple(params Space[] spaces) => new(spaces);

        /// <summary>
        ///     A dict space from key and subspace pairs, which keep their order: <c>spaces.Dict([("a", a), ("b", b)])</c>; the
        ///     new space owns the subspaces.
        /// </summary>
        /// <param name="spaces">The pairs, in order.</param>
        /// <param name="seed">An integer seed (the dict's generator then draws one <c>int32</c> seed per subspace), a generator to share, or <see langword="null"/>.</param>
        /// <param name="sortKeys">Recorded but not applied: Gymnasium sorts only an unordered mapping.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="ArgumentException">Thrown for a repeated key.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Dict Dict(IEnumerable<KeyValuePair<string, Space>> spaces, SpaceSeed? seed = null, bool sortKeys = true) => new(spaces, seed, sortKeys);

        /// <summary>
        ///     A dict space from a mapping: <c>spaces.Dict({"a": a, "b": b})</c>; with <paramref name="sortKeys"/> (Gymnasium's
        ///     default) the keys are sorted ordinally. The new space owns the subspaces.
        /// </summary>
        /// <param name="spaces">The mapping.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <param name="sortKeys">Whether to sort the keys, Gymnasium's behaviour for an unordered dict.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="spaces"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Dict Dict(IReadOnlyDictionary<string, Space> spaces, SpaceSeed? seed = null, bool sortKeys = true) => new(spaces, seed, sortKeys);

        /// <summary>Strings of a character set, in a length range: <c>spaces.Text(max_length, min_length=1, charset=alphanumeric, seed=seed)</c>.</summary>
        /// <param name="maxLength">The longest length, inclusive.</param>
        /// <param name="minLength">The shortest length, inclusive; 1 by default, so no empty string.</param>
        /// <param name="charset">The characters, in sampling order (duplicates dropped); <see langword="null"/> is Gymnasium's alphanumeric set.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown for a negative minimum or a minimum above the maximum.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Text Text(long maxLength, long minLength = 1, string? charset = null, SpaceSeed? seed = null) => new(maxLength, minLength, charset, seed);

        /// <summary>Strings of a set of characters, which is sorted as Gymnasium sorts a <c>frozenset</c> charset.</summary>
        /// <param name="maxLength">The longest length, inclusive.</param>
        /// <param name="charset">The characters; each must be a single code point.</param>
        /// <param name="minLength">The shortest length, inclusive.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="ValueError">Thrown for invalid lengths, or a charset element that is not one character.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="charset"/> is <see langword="null"/>.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Text Text(long maxLength, ISet<string> charset, long minLength = 1, SpaceSeed? seed = null) => new(maxLength, charset, minLength, seed);

        /// <summary>Sequences of any length of a feature space's elements: <c>spaces.Sequence(space, seed=seed, stack=stack)</c>; the new space owns the feature space.</summary>
        /// <param name="space">The feature space.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <param name="stack">Whether samples are the items stacked into one batch rather than a tuple of them.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> feature space.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Sequence Sequence(Space space, SpaceSeed? seed = null, bool stack = false) => new(space, seed, stack);

        /// <summary>Graphs with node and edge features: <c>spaces.Graph(node_space, edge_space, seed=seed)</c>; the new space owns both subspaces.</summary>
        /// <param name="nodeSpace">The space of one node's features, usually a box or a discrete space.</param>
        /// <param name="edgeSpace">The space of one edge's features, or <see langword="null"/> for graphs without edges.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> node space.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static Graph Graph(Space nodeSpace, Space? edgeSpace, SpaceSeed? seed = null) => new(nodeSpace, edgeSpace, seed);

        /// <summary>One element of one of the subspaces, with its index: <c>spaces.OneOf(spaces, seed=seed)</c>; the new space owns the subspaces.</summary>
        /// <param name="spaces">The subspaces, in order; at least one.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/>.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when <paramref name="spaces"/> or a subspace is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for no subspaces.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        public static OneOf OneOf(IEnumerable<Space> spaces, SpaceSeed? seed = null) => new(spaces, seed);

        /// <summary>One element of one of the given subspaces: <c>spaces.OneOf((a, b))</c>; the new space owns them.</summary>
        /// <param name="spaces">The subspaces, in order; at least one.</param>
        /// <returns>The new space; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown for a <see langword="null"/> subspace.</exception>
        /// <exception cref="ValueError">Thrown for no subspaces.</exception>
        public static OneOf OneOf(params Space[] spaces) => new(spaces);

        /// <summary>A graph element from its parts: <c>spaces.GraphInstance(nodes, edges, edge_links)</c>; the instance takes ownership of them.</summary>
        /// <param name="nodes">The node features, <c>(n, ...)</c>, one row per node.</param>
        /// <param name="edges">The edge features, <c>(m, ...)</c>, or <see langword="null"/> for a graph without edge features.</param>
        /// <param name="edgeLinks">The <c>(m, 2)</c> integer array of the node indices each edge connects, or <see langword="null"/> when there are no edges.</param>
        /// <returns>The new instance; the caller owns it, and disposing it releases the parts.</returns>
        public static GraphInstance GraphInstance(object? nodes, object? edges, NDArray? edgeLinks) => new(nodes, edges, edgeLinks);

        /// <summary>
        ///     The number of entries a space's elements flatten to: <c>spaces.flatdim(space)</c> (a box's element count, a
        ///     discrete space's <c>n</c>, one-hot, a composite's sum).
        /// </summary>
        /// <param name="space">The space.</param>
        /// <returns>The flattened size.</returns>
        /// <exception cref="ValueError">Thrown for a space that cannot be flattened to an array (a graph, a sequence, or a composite holding one).</exception>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space flattenable but unknown to the function.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        public static long Flatdim(Space space) => SpaceUtils.Flatdim(space);

        /// <summary>
        ///     An element flattened (<c>spaces.flatten(space, x)</c>): a new flat array for a flattenable space, and the same
        ///     structure of flattened parts for a graph, a sequence or a composite holding one.
        /// </summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The element.</param>
        /// <returns>The flattened element, new and owned by the caller.</returns>
        /// <exception cref="IndexError">Thrown for a discrete value outside the space, or a text longer than <c>max_length</c>.</exception>
        /// <exception cref="KeyError">Thrown for a dict element missing a key, or a character outside a text space's charset.</exception>
        /// <exception cref="ValueError">Thrown for a tuple element of another length than the space.</exception>
        /// <exception cref="TypeError">Thrown for an element of the wrong kind for a dict or text space.</exception>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        public static object? Flatten(Space space, object? x) => SpaceUtils.Flatten(space, x);

        /// <summary>The space <see cref="Flatten"/> maps a space's elements into: <c>spaces.flatten_space(space)</c>, a flat box for every flattenable space.</summary>
        /// <param name="space">The space.</param>
        /// <returns>A new space, owned by the caller, unseeded.</returns>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        /// <remarks>Footgun: sampling the flattened space is not sampling the original; a flattened discrete space is a 0/1 box whose samples are rarely one-hot.</remarks>
        public static Space FlattenSpace(Space space) => SpaceUtils.FlattenSpace(space);

        /// <summary>Reverses <see cref="Flatten"/>: <c>spaces.unflatten(space, x)</c>; the space must be the one the element was flattened with.</summary>
        /// <param name="space">The space.</param>
        /// <param name="x">The flattened element.</param>
        /// <returns>An element of <paramref name="space"/>, new and owned by the caller.</returns>
        /// <exception cref="ValueError">Thrown for a flattened discrete element with no one-hot entry.</exception>
        /// <exception cref="AssertionError">Thrown for a composite given the wrong flattened form.</exception>
        /// <exception cref="KeyError">Thrown for a non-flattenable dict space's flattened dict missing a key.</exception>
        /// <exception cref="NumSharp.NotImplementedError">Thrown for a custom space.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="space"/> is <see langword="null"/>.</exception>
        public static object? Unflatten(Space space, object? x) => SpaceUtils.Unflatten(space, x);
    }
}
