#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     Graphs whose nodes and edges carry features from two spaces: every element is a <see cref="GraphInstance"/> of
    ///     node features, edge features, and the <c>(m, 2)</c> links saying which two nodes each edge joins. The number of
    ///     nodes and edges varies per sample.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A sample draws its node and edge features from batches of the subspaces built for that sample, from copies of the
    ///     subspaces' generators; each subspace's own generator then advances by one <c>random()</c> draw so the next sample
    ///     differs. The links come from the graph's own generator.
    ///     </para>
    ///     <para>
    ///     The space owns its subspaces. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/graph.py#L30">gymnasium/spaces/graph.py</see>.
    ///     </para>
    /// </remarks>
    public class Graph : Space<GraphInstance>, IEquatable<Graph> {
        /// <summary>
        ///     Graphs with features from <paramref name="nodeSpace"/> and <paramref name="edgeSpace"/> (Gymnasium's
        ///     <c>Graph(node_space, edge_space, seed=seed)</c>).
        /// </summary>
        /// <param name="nodeSpace">The space of one node's features, usually a <see cref="Box"/> or <see cref="Discrete"/>; owned by the graph from now on.</param>
        /// <param name="edgeSpace">The space of one edge's features, or <see langword="null"/> for graphs without edge features (and so without edges).</param>
        /// <param name="seed">
        ///     An integer seed (it seeds the graph's generator, which draws the subspaces' seeds, then is seeded again with the
        ///     same value), a generator to share, or <see langword="null"/>.
        /// </param>
        /// <exception cref="NumSharp.TypeError">Thrown for a <see langword="null"/> node space, with <c>batch_space</c>'s message as in Gymnasium.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        /// <remarks>
        ///     Building the batched subspaces reads the subspaces' generators, creating them from entropy when they were not
        ///     seeded, as Gymnasium's <c>batch_space</c> does.
        /// </remarks>
        public Graph(Space nodeSpace, Space? edgeSpace, SpaceSeed? seed = null) : base(null, null, null) {
            NodeSpace = nodeSpace!;
            EdgeSpace = edgeSpace;
            // batch_space raises the TypeError for a null node space, as Gymnasium's constructor does.
            BatchNodeSpace = VectorUtils.BatchSpace(nodeSpace!, 1);
            BatchEdgeSpace = edgeSpace is null ? null : VectorUtils.BatchSpace(edgeSpace, 1);
            if (seed is { } given) {
                if (given.Generator != null)
                    NpRandom = given.Generator;
                else
                    Seed(given.Value);
            }
        }

        /// <summary>The space of one node's features (Gymnasium's <c>node_space</c>).</summary>
        public Space NodeSpace { get; }

        /// <summary>The space of one edge's features (Gymnasium's <c>edge_space</c>), or <see langword="null"/>.</summary>
        public Space? EdgeSpace { get; }

        /// <summary>The node space batched once (Gymnasium's <c>batch_node_space</c>), for membership and JSON.</summary>
        public Space BatchNodeSpace { get; }

        /// <summary>The edge space batched once (Gymnasium's <c>batch_edge_space</c>), or <see langword="null"/> without an edge space.</summary>
        public Space? BatchEdgeSpace { get; }

        /// <summary>A graph never flattens to a box: its size varies.</summary>
        public override bool IsNpFlattenable => false;

        /// <summary>Draws a graph of 10 nodes and a random number of edges (Gymnasium's <c>sample()</c> defaults).</summary>
        /// <param name="mask"><see langword="null"/>, or a pair <c>(node_mask, edge_mask)</c>; see <see cref="Sample(object, object, long, long?)"/>.</param>
        /// <param name="probability"><see langword="null"/>, or a pair <c>(node_probability, edge_probability)</c>.</param>
        /// <returns>The graph; the caller owns it and its arrays.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given.</exception>
        public override GraphInstance Sample(object? mask = null, object? probability = null) => Sample(mask, probability, 10, null);

        /// <summary>
        ///     Draws a graph (Gymnasium's <c>sample</c>): <paramref name="numNodes"/> node features, the edges (a random count
        ///     below <c>n·(n-1)</c> unless <paramref name="numEdges"/> fixes it) and their links.
        /// </summary>
        /// <param name="mask">
        ///     <see langword="null"/>, or a pair <c>(node_mask, edge_mask)</c> (an <see langword="object"/>?[], a tuple or a list
        ///     of two): the node mask goes to the batched node space; the edge mask to the batched edge space, repeated once per
        ///     edge when the edge count is drawn. <see cref="Box"/> subspaces take no mask.
        /// </param>
        /// <param name="probability"><see langword="null"/>, or the same pair of probability masks.</param>
        /// <param name="numNodes">The number of nodes, at least 1 (Gymnasium's <c>num_nodes</c>, default 10).</param>
        /// <param name="numEdges">The number of edges, or <see langword="null"/> to draw it (Gymnasium's <c>num_edges</c>).</param>
        /// <returns>
        ///     The graph; the caller owns it and its arrays. Its edges and links are <see langword="null"/> when there is no edge
        ///     space or no edge.
        /// </returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for <paramref name="numNodes"/> below 1 or a negative <paramref name="numEdges"/>, with Gymnasium's messages.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when both masks are given, or a mask is not a pair (Python's unpacking error).</exception>
        /// <exception cref="NumSharp.TypeError">Thrown for a mask that cannot be unpacked at all.</exception>
        /// <remarks>
        ///     A fixed <paramref name="numEdges"/> with no edge space only warns, as in Gymnasium, and the graph has no edges.
        /// </remarks>
        public GraphInstance Sample(object? mask = null, object? probability = null, long numNodes = 10, long? numEdges = null) {
            if (numNodes <= 0)
                throw new AssertionError($"The number of nodes is expected to be greater than 0, actual value: {numNodes}");

            object? nodeMask = null, edgeMask = null;
            string? maskType = null;
            if (mask is not null && probability is not null)
                throw new ValueError($"Only one of `mask` or `probability` can be provided, actual values: mask={PyBuiltins.Str(mask)}, probability={PyBuiltins.Str(probability)}");
            if (mask is not null) {
                (nodeMask, edgeMask) = UnpackPair(mask);
                maskType = "mask";
            } else if (probability is not null) {
                (nodeMask, edgeMask) = UnpackPair(probability);
                maskType = "probability";
            }

            long edges;
            if (numEdges is null) {
                if (numNodes > 1) {
                    // At most n*(n-1) edges: self loops and both directions are allowed.
                    using NDArray drawn = NpRandom.integers(numNodes * (numNodes - 1));
                    edges = Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
                } else {
                    edges = 0;
                }

                if (edgeMask is not null) {
                    object? perEdge = edgeMask;
                    edgeMask = Enumerable.Range(0, (int)edges).Select(_ => perEdge).ToArray();
                }
            } else {
                if (EdgeSpace is null)
                    Logger.Warn($"The number of edges is set ({numEdges.Value}) but the edge space is None.");
                if (numEdges.Value < 0)
                    throw new AssertionError($"Expects the number of edges to be greater than 0, actual value: {numEdges.Value}");
                edges = numEdges.Value;
            }

            // The batched node space is rebuilt per sample because the node count changes; it samples from a copy of the
            // node space's generator, which then advances by one draw so the next sample differs (Gymnasium's comment).
            object? sampledNodes;
            using (Space batchNodes = VectorUtils.BatchSpace(NodeSpace, checked((int)numNodes)))
                sampledNodes = SampleWith(batchNodes, maskType, nodeMask);
            AdvanceGenerator(NodeSpace);

            object? sampledEdges = null;
            try {
                // A single node with one edge (a self loop) is valid.
                if (numNodes >= 1 && edges >= 1 && EdgeSpace is not null) {
                    using (Space batchEdges = VectorUtils.BatchSpace(EdgeSpace, checked((int)edges)))
                        sampledEdges = SampleWith(batchEdges, maskType, edgeMask);
                    AdvanceGenerator(EdgeSpace);
                }

                NDArray? links = null;
                if (sampledEdges is not null && edges > 0)
                    links = NpRandom.integers(0, numNodes, size: new Shape(edges, 2L), dtype: np.int32);
                return new GraphInstance(sampledNodes, sampledEdges, links);
            } catch {
                // Nothing reaches the caller on failure, so the drawn features are released here.
                new GraphInstance(sampledNodes, sampledEdges, null).Dispose();
                throw;
            }
        }

        /// <summary>
        ///     Whether <paramref name="x"/> is an element (Gymnasium's <c>contains</c>): every node in the node space and, when
        ///     the graph has edges, every edge in the edge space with integer links of shape <c>(m, 2)</c> that point at
        ///     existing nodes; a graph without edges must have neither edges nor links.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        /// <exception cref="NumSharp.TypeError">Thrown, as Gymnasium's does, when the nodes or edges cannot be iterated as the batched subspace's elements.</exception>
        public override bool Contains(object? x) {
            if (x is not GraphInstance graph || graph.Nodes is null)
                return false;
            object?[] nodes = VectorUtils.Iterate(BatchNodeSpace, graph.Nodes).ToArray();
            if (!nodes.All(NodeSpace.Contains))
                return false;
            if (graph.Edges is null || graph.EdgeLinks is null)
                return graph.Edges is null && graph.EdgeLinks is null;
            if (EdgeSpace is null)
                return false;
            object?[] edges = VectorUtils.Iterate(BatchEdgeSpace!, graph.Edges).ToArray();
            if (!edges.All(EdgeSpace.Contains))
                return false;
            NDArray links = graph.EdgeLinks;
            if (links.dtype.kind is not ('i' or 'u'))
                return false;
            if (links.ndim != 2 || links.shape[0] != edges.Length || links.shape[1] != 2)
                return false;
            using NDArray atLeastZero = links >= 0;
            using NDArray nodeCount = NDArray.Scalar((long)nodes.Length);
            using NDArray belowCount = links < nodeCount;
            using NDArray valid = np.logical_and(atLeastZero, belowCount);
            return np.all(valid);
        }

        /// <summary>
        ///     Converts samples to Gymnasium's JSON form: one dict per sample with <c>"nodes"</c> (the batched node space's
        ///     JSON of the one sample) and, when the sample has edges, <c>"edges"</c> and <c>"edge_links"</c> (nested lists).
        /// </summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>The list of dicts.</returns>
        /// <exception cref="NullReferenceException">Thrown for a sample with edges in a graph without an edge space, as Gymnasium fails on <c>None</c>.</exception>
        public override object ToJsonable(IEnumerable<GraphInstance> samples) {
            var json = new List<object?>();
            foreach (GraphInstance sample in samples) {
                var entry = new Dictionary<string, object?> { ["nodes"] = BatchNodeSpace.ToJsonable(new[] { sample.Nodes }) };
                if (sample.Edges is not null && sample.EdgeLinks is not null) {
                    entry["edges"] = (BatchEdgeSpace ?? throw new NullReferenceException("'NoneType' object has no attribute 'to_jsonable'")).ToJsonable(new[] { sample.Edges });
                    entry["edge_links"] = PythonList(sample.EdgeLinks);
                }

                json.Add(entry);
            }

            return json;
        }

        /// <summary>Converts Gymnasium's JSON dicts back to graphs (Gymnasium's <c>from_jsonable</c>); links become int32 arrays.</summary>
        /// <param name="jsonable">The list of dicts.</param>
        /// <returns>The graphs, owned by the caller.</returns>
        /// <exception cref="ArgumentException">Thrown when an entry is not a dict with <c>"nodes"</c>.</exception>
        /// <exception cref="NumSharp.AssertionError">Thrown for an entry with edges in a graph without an edge space (Gymnasium's bare <c>assert</c>).</exception>
        public override IList<GraphInstance> FromJsonable(object jsonable) {
            var graphs = new List<GraphInstance>();
            foreach (object? item in JsonList(jsonable)) {
                IReadOnlyDictionary<string, object?> entry = Dict.AsDict(item) ?? throw new ArgumentException("Expected a dict with \"nodes\" for each graph.", nameof(jsonable));
                object? nodes = BatchNodeSpace.FromJsonable(entry["nodes"]!)[0];
                if (entry.ContainsKey("edges")) {
                    if (EdgeSpace is null)
                        throw new AssertionError();
                    object? edges = BatchEdgeSpace!.FromJsonable(entry["edges"]!)[0];
                    graphs.Add(new GraphInstance(nodes, edges, Box.AsArray(entry["edge_links"], np.int32)));
                } else {
                    graphs.Add(new GraphInstance(nodes, null, null));
                }
            }

            return graphs;
        }

        /// <summary>Gymnasium's <c>repr</c>: <c>Graph(Box(-1.0, 1.0, (3,), float32), Discrete(3))</c>, the edge space <c>None</c> when absent.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"Graph({NodeSpace}, {(EdgeSpace is null ? "None" : EdgeSpace.ToString())})";

        /// <summary>Gymnasium's <c>==</c>: equal node spaces and equal (or both absent) edge spaces.</summary>
        /// <param name="other">The other space.</param>
        /// <returns><see langword="true"/> when equal.</returns>
        public bool Equals(Graph? other) =>
            other is not null && NodeSpace.Equals(other.NodeSpace) && (EdgeSpace is null ? other.EdgeSpace is null : EdgeSpace.Equals(other.EdgeSpace));

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is Graph other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() => HashCode.Combine(NodeSpace.GetHashCode(), EdgeSpace?.GetHashCode() ?? 0);

        /// <summary>
        ///     Gymnasium's <c>Graph.seed</c> for an integer or <see langword="null"/>: <see langword="null"/> seeds the graph
        ///     and its subspaces from entropy; an integer seeds the graph's generator, draws the subspaces' seeds (one, or two
        ///     with an edge space, below <c>int32</c>'s maximum), and seeds the generator again with the same value.
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>The graph's seed followed by the node space's (and the edge space's), as a list.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        protected override object SeedCore(BigInteger? seed) {
            if (seed is null) {
                var entropySeeds = new List<object> { base.SeedCore(null), NodeSpace.Seed((BigInteger?)null) };
                if (EdgeSpace is not null)
                    entropySeeds.Add(EdgeSpace.Seed((BigInteger?)null));
                return entropySeeds;
            }

            object superSeed = base.SeedCore(seed);
            if (EdgeSpace is null) {
                long nodeSeed;
                using (NDArray drawn = NpRandom.integers(int.MaxValue))
                    nodeSeed = Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
                base.SeedCore(seed);
                return new List<object> { superSeed, NodeSpace.Seed(nodeSeed) };
            }

            long nodeSubseed, edgeSubseed;
            using (NDArray drawn = NpRandom.integers(int.MaxValue, size: new Shape(2L))) {
                nodeSubseed = Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture);
                edgeSubseed = Convert.ToInt64(drawn.GetAtIndex(1), CultureInfo.InvariantCulture);
            }

            base.SeedCore(seed);
            return new List<object> { superSeed, NodeSpace.Seed(nodeSubseed), EdgeSpace.Seed(edgeSubseed) };
        }

        /// <summary>Gymnasium's <c>Graph.seed</c> for a list: the graph's seed, the node space's, and the edge space's when there is one.</summary>
        /// <param name="seeds">Two seeds, or three with an edge space.</param>
        /// <returns>The seeds used.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for a list of the wrong length, with Gymnasium's messages.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when the graph's own seed is negative or not a Python <c>int</c>.</exception>
        protected override object SeedListCore(IReadOnlyList<object?> seeds) {
            if (EdgeSpace is null) {
                if (seeds.Count != 2)
                    throw new ValueError($"Expects a tuple of two values for Graph and node space, actual length: {seeds.Count}");
                return new List<object> { SeedOwnGenerator(seeds[0]), NodeSpace.SeedWith(seeds[1]) };
            }

            if (seeds.Count != 3)
                throw new ValueError($"Expects a tuple of three values for Graph, node and edge space, actual length: {seeds.Count}");
            return new List<object> { SeedOwnGenerator(seeds[0]), NodeSpace.SeedWith(seeds[1]), EdgeSpace.SeedWith(seeds[2]) };
        }

        /// <summary>Gymnasium's <c>Graph.seed</c> error for a seed that is not an integer, a list, a tuple or <see langword="null"/>.</summary>
        /// <param name="seed">The refused seed.</param>
        /// <returns><c>TypeError: Expects `None`, int or tuple of ints, actual type: &lt;class 'dict'&gt;</c>.</returns>
        protected override Exception SeedTypeError(object? seed) =>
            new TypeError($"Expects `None`, int or tuple of ints, actual type: {PyBuiltins.TypeRepr(seed)}");

        /// <inheritdoc />
        protected override void Dispose(bool disposing) {
            if (disposing) {
                NodeSpace.Dispose();
                EdgeSpace?.Dispose();
                BatchNodeSpace.Dispose();
                BatchEdgeSpace?.Dispose();
            }

            base.Dispose(disposing);
        }

        /// <summary>Samples a batched subspace under the mask kind in use, as Gymnasium's <c>sample(**{mask_type: mask})</c>.</summary>
        /// <param name="space">The batched subspace.</param>
        /// <param name="maskType"><c>mask</c>, <c>probability</c>, or <see langword="null"/> for none.</param>
        /// <param name="mask">The mask for that kind, possibly <see langword="null"/>.</param>
        /// <returns>The batched sample.</returns>
        private static object? SampleWith(Space space, string? maskType, object? mask) => maskType switch {
            "mask" => space.Sample(mask: mask),
            "probability" => space.Sample(probability: mask),
            _ => space.Sample(),
        };

        /// <summary>
        ///     Advances a subspace's generator by one <c>random()</c> draw, as Gymnasium does after sampling from a batch built
        ///     on a copy of it, so consecutive samples differ.
        /// </summary>
        /// <param name="space">The subspace.</param>
        private static void AdvanceGenerator(Space space) {
            using NDArray discarded = space.NpRandom.random();
        }

        /// <summary>Python's <c>a, b = pair</c> for a mask argument.</summary>
        /// <param name="pair">The argument.</param>
        /// <returns>Its two items.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for a sequence that does not hold exactly two items, with Python's unpacking messages.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown for a value that is not a sequence: <c>cannot unpack non-iterable int object</c>.</exception>
        private static (object? First, object? Second) UnpackPair(object pair) {
            object?[] items = pair switch {
                object?[] array => array,
                System.Runtime.CompilerServices.ITuple tuple => Tuple.TupleItems(tuple),
                NDArray { ndim: > 0 } array => Enumerable.Range(0, (int)array.shape[0]).Select(i => (object?)array[i.ToString(CultureInfo.InvariantCulture)]).ToArray(),
                string text => Text.CodePoints(text).Select(c => (object?)c).ToArray(),
                IList list => list.Cast<object?>().ToArray(),
                _ => throw new TypeError($"cannot unpack non-iterable {PyBuiltins.TypeName(pair)} object"),
            };
            if (items.Length > 2)
                throw new ValueError("too many values to unpack (expected 2)");
            if (items.Length < 2)
                throw new ValueError($"not enough values to unpack (expected 2, got {items.Length})");
            return (items[0], items[1]);
        }
    }
}
