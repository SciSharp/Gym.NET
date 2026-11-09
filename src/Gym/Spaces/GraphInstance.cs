#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     One element of a <see cref="Graph"/> space: the node features, the edge features and the edge links, as the
    ///     <c>(nodes, edges, edge_links)</c> tuple Gymnasium's NamedTuple is. It deconstructs like one:
    ///     <c>var (nodes, edges, links) = graph.Sample();</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     For a <see cref="Box"/> or <see cref="Discrete"/> node space (Gymnasium's intended use) <see cref="Nodes"/> is an
    ///     array of shape <c>(n, ...)</c> and <see cref="Edges"/> one of shape <c>(m, ...)</c>; other subspaces give their
    ///     batched form (a tuple, a dict), which is why both are typed <see langword="object"/>.
    ///     </para>
    ///     <para>
    ///     The instance owns the arrays it holds: <see cref="Dispose"/> releases them, nested ones included. Equality is
    ///     Python's tuple <c>==</c> through <see cref="PyEquality"/> (arrays equal when <c>np.array_equal</c> says so).
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/graph.py#L16">gymnasium/spaces/graph.py (GraphInstance)</see>.
    ///     </para>
    /// </remarks>
    public sealed class GraphInstance : ITuple, IEquatable<GraphInstance>, INDArrayCarrier, IDisposable {
        /// <summary>A graph from its parts (Gymnasium's <c>GraphInstance(nodes, edges, edge_links)</c>); the instance takes ownership of them.</summary>
        /// <param name="nodes">The node features: <c>(n, ...)</c>, one row per node.</param>
        /// <param name="edges">The edge features, <c>(m, ...)</c>, or <see langword="null"/> for a graph without edge features.</param>
        /// <param name="edgeLinks">
        ///     The <c>(m, 2)</c> integer array of the two node indices each edge connects, or <see langword="null"/> when there
        ///     are no edges.
        /// </param>
        public GraphInstance(object? nodes, object? edges, NDArray? edgeLinks) {
            Nodes = nodes;
            Edges = edges;
            EdgeLinks = edgeLinks;
        }

        /// <summary>The node features (Gymnasium's <c>nodes</c>), one row per node.</summary>
        public object? Nodes { get; }

        /// <summary>The edge features (Gymnasium's <c>edges</c>), one row per edge, or <see langword="null"/>.</summary>
        public object? Edges { get; }

        /// <summary>The <c>(m, 2)</c> node indices each edge connects (Gymnasium's <c>edge_links</c>), or <see langword="null"/>.</summary>
        public NDArray? EdgeLinks { get; }

        /// <summary>The number of fields, 3, as <c>len(graph_instance)</c> is.</summary>
        int ITuple.Length => 3;

        /// <summary>The field at <paramref name="index"/>, as Python's tuple indexing of a NamedTuple reads it.</summary>
        /// <param name="index">0 for the nodes, 1 for the edges, 2 for the edge links.</param>
        /// <returns>The field.</returns>
        /// <exception cref="NumSharp.IndexError">Thrown for any other index: <c>tuple index out of range</c>.</exception>
        object? ITuple.this[int index] => index switch {
            0 => Nodes,
            1 => Edges,
            2 => EdgeLinks,
            _ => throw new IndexError("tuple index out of range"),
        };

        /// <summary>Unpacks the instance, as Python's <c>nodes, edges, edge_links = graph_instance</c> does.</summary>
        /// <param name="nodes">The node features.</param>
        /// <param name="edges">The edge features, or <see langword="null"/>.</param>
        /// <param name="edgeLinks">The edge links, or <see langword="null"/>.</param>
        public void Deconstruct(out object? nodes, out object? edges, out NDArray? edgeLinks) {
            nodes = Nodes;
            edges = Edges;
            edgeLinks = EdgeLinks;
        }

        /// <summary>Python's tuple <c>==</c> of the three fields (see the remarks for arrays).</summary>
        /// <param name="other">The other instance.</param>
        /// <returns><see langword="true"/> when all three fields are equal.</returns>
        public bool Equals(GraphInstance? other) =>
            other is not null && (ReferenceEquals(this, other) ||
                                  PyEquality.AreEqual(Nodes, other.Nodes) && PyEquality.AreEqual(Edges, other.Edges) && PyEquality.AreEqual(EdgeLinks, other.EdgeLinks));

        /// <inheritdoc />
        public override bool Equals(object? obj) => obj is GraphInstance other && Equals(other);

        /// <summary>
        ///     A hash consistent with <see cref="Equals(GraphInstance)"/> that never reads array data (array fields contribute
        ///     their shapes). Python cannot hash an instance holding arrays at all.
        /// </summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() =>
            HashCode.Combine(ResultEquality.ObservationHash(Nodes), ResultEquality.ObservationHash(Edges), ResultEquality.ObservationHash(EdgeLinks));

        /// <summary>Gymnasium's NamedTuple <c>repr</c>: <c>GraphInstance(nodes=array([...]), edges=None, edge_links=None)</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() =>
            $"GraphInstance(nodes={PyBuiltins.Repr(Nodes)}, edges={PyBuiltins.Repr(Edges)}, edge_links={PyBuiltins.Repr(EdgeLinks)})";

        /// <summary>Hands the instance's arrays to an <see cref="NDScope"/> (D19), so an <c>[NDScoped]</c> method can return it.</summary>
        /// <param name="scope">The scope the arrays leave through.</param>
        void INDArrayCarrier.YieldTo(NDScope scope) => ResultEquality.Yield(scope, Nodes, Edges, EdgeLinks);

        /// <summary>Releases the arrays the instance holds, nested ones (a tuple's or dict's parts) included; idempotent.</summary>
        public void Dispose() {
            DisposeArrays(Nodes, depth: 0);
            DisposeArrays(Edges, depth: 0);
            EdgeLinks?.Dispose();
        }

        /// <summary>Disposes every array in a value, walking tuples, lists and dicts to a fixed depth.</summary>
        /// <param name="value">The value.</param>
        /// <param name="depth">How deep the walk is; containers below 16 levels are left alone, so a self-containing value cannot recurse forever.</param>
        private static void DisposeArrays(object? value, int depth) {
            if (depth > 16)
                return;
            switch (value) {
                case NDArray array:
                    array.Dispose();
                    break;
                case IDisposable disposable:
                    // A nested GraphInstance (or another owner of arrays) releases its own.
                    disposable.Dispose();
                    break;
                case string:
                    break;
                case IEnumerable<KeyValuePair<string, object?>> pairs:
                    foreach (KeyValuePair<string, object?> pair in pairs)
                        DisposeArrays(pair.Value, depth + 1);
                    break;
                case ITuple tuple:
                    for (int i = 0; i < tuple.Length; i++)
                        DisposeArrays(tuple[i], depth + 1);
                    break;
                case IEnumerable items:
                    foreach (object? item in items)
                        DisposeArrays(item, depth + 1);
                    break;
            }
        }
    }
}
