#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Spaces {
    /// <summary>
    ///     Gymnasium's <c>Graph</c> tests (<c>tests/spaces/test_graph.py</c>): node and edge masks, the edge-count warning and
    ///     error, graphs that are not members, and probability-weighted nodes and edges.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Two Python-only checks have no C# counterpart: assigning a string as the node space (<c>NodeSpace</c> is a typed,
    ///     read-only property) and a string where a graph belongs (<c>"abc" in space</c> is simply <see langword="false"/>, as
    ///     the first membership case here shows).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py">tests/spaces/test_graph.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestGraph {
        /// <summary>A per-node mask (one per node) fixes the nodes; an edge count without an edge space only warns.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py#L9">test_node_space_sample</see>.</remarks>
        [TestMethod]
        public void TestNodeSpaceSample() {
            using var space = new Graph(new Discrete(3), null);
            space.Seed(0);

            object?[] fiveMasks = Enumerable.Range(0, 5).Select(_ => (object?)np.array(new sbyte[] { 0, 1, 0 })).ToArray();
            try {
                using GraphInstance sample = space.Sample(new object?[] { fiveMasks, null }, numNodes: 5);
                Assert.IsTrue(space.Contains(sample));
                Assert.IsTrue(((NDArray)sample.Nodes!).ToArray<long>().All(v => v == 1));
            } finally {
                Release(fiveMasks);
            }

            object?[] twoMasks = { np.array(new sbyte[] { 1, 0, 0 }), np.array(new sbyte[] { 0, 1, 0 }) };
            try {
                using GraphInstance sample = space.Sample(new object?[] { twoMasks, null }, numNodes: 2);
                Assert.IsTrue(space.Contains(sample));
                CollectionAssert.AreEqual(new long[] { 0, 1 }, ((NDArray)sample.Nodes!).ToArray<long>());
            } finally {
                Release(twoMasks);
            }

            using (WarningCatcher caught = Warnings.Catch()) {
                using GraphInstance sample = space.Sample(numEdges: 5);
                Assert.IsTrue(space.Contains(sample));
                Assert.IsTrue(caught.Records.Any(w => w.Category == WarningCategory.UserWarning && w.Message.Contains("The number of edges is set (5) but the edge space is None.")));
            }
        }

        /// <summary>Edge counts, edge masks (repeated per drawn edge, or one per fixed edge), the negative count, and the warning's colours.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py#L47">test_edge_space_sample</see>.</remarks>
        [TestMethod]
        public void TestEdgeSpaceSample() {
            using (var space = new Graph(new Discrete(3), new Discrete(3))) {
                space.Seed(0);
                // With one node there are no edges.
                using (GraphInstance oneNode = space.Sample(numNodes: 1))
                    Assert.IsNull(oneNode.Edges);
                using (GraphInstance threeEdges = space.Sample(numEdges: 3)) {
                    long count = ((NDArray)threeEdges.Edges!).size;
                    Assert.IsTrue(count >= 0 && count < 6);
                }

                using (NDArray onlyOne = np.array(new sbyte[] { 0, 1, 0 }))
                using (GraphInstance sample = space.Sample(new object?[] { null, onlyOne }))
                    Assert.IsTrue(sample.Edges is null || ((NDArray)sample.Edges).ToArray<long>().All(v => v == 1));

                object?[] edgeMasks = { np.array(new sbyte[] { 1, 0, 0 }), np.array(new sbyte[] { 0, 1, 0 }), np.array(new sbyte[] { 0, 0, 1 }) };
                try {
                    using GraphInstance sample = space.Sample(new object?[] { null, edgeMasks }, numEdges: 3);
                    CollectionAssert.AreEqual(new long[] { 0, 1, 2 }, ((NDArray)sample.Edges!).ToArray<long>());
                } finally {
                    Release(edgeMasks);
                }

                Assert.AreEqual("Expects the number of edges to be greater than 0, actual value: -1",
                    Assert.ThrowsExactly<AssertionError>(() => space.Sample(numEdges: -1)).Message);
            }

            using (var space = new Graph(new Discrete(3), null))
            using (WarningCatcher caught = Warnings.Catch()) {
                using GraphInstance sample = space.Sample(numEdges: 5);
                Assert.IsTrue(caught.Records.Any(w => w.Message.Contains("\u001b[33mWARN: The number of edges is set (5) but the edge space is None.\u001b[0m")));
                Assert.IsNull(sample.Edges);
            }
        }

        /// <summary>Graphs with missing, out-of-range, mismatched or malformed parts are not members.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py#L94">test_not_contains</see>.</remarks>
        [TestMethod]
        public void TestNotContains() {
            using var space = new Graph(new Discrete(2), new Discrete(2));
            Assert.IsFalse(space.Contains("abc"));
            var cases = new Func<GraphInstance>[] {
                () => new GraphInstance(null, Ints(0, 1), Links(new long[,] { { 0, 1 }, { 1, 0 } })),
                () => new GraphInstance(Ints(10, 1, 0), Ints(0, 1), Links(new long[,] { { 0, 1 }, { 1, 0 } })),
                () => new GraphInstance(Ints(0, 1), null, Links(new long[,] { { 0, 1 }, { 1, 0 } })),
                () => new GraphInstance(Ints(0, 1), Ints(0, 1), null),
                () => new GraphInstance(Ints(1, 2), Ints(10, 1), Links(new long[,] { { 0, 1 }, { 1, 0 } })),
                () => new GraphInstance(Ints(1, 2), Ints(0, 1), np.array(new[,] { { 0.5, 1.0 }, { 2.0, 1.0 } })),
                () => new GraphInstance(Ints(1, 2), Ints(10, 1), Ints(0, 1)),
                () => new GraphInstance(Ints(1, 2), Ints(0, 1), np.array(new long[,,] { { { 0 }, { 1 } }, { { 0 }, { 0 } } })),
                () => new GraphInstance(Ints(1, 2), Ints(0, 1), Links(new long[,] { { 10, 1 }, { 0, 0 } })),
                () => new GraphInstance(Ints(1, 2), Ints(0, 1), Links(new long[,] { { -10, 1 }, { 0, 0 } })),
            };
            for (int i = 0; i < cases.Length; i++) {
                using GraphInstance graph = cases[i]();
                Assert.IsFalse(space.Contains(graph), $"case {i + 1}: {graph}");
            }
        }

        /// <summary>1000 single-node samples weighted by a node probability mask match its distribution.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py#L139">test_probability_node_sampling</see>.</remarks>
        [TestMethod]
        public void TestProbabilityNodeSampling() {
            using var space = new Graph(new Discrete(3), null);
            space.Seed(42);
            double[] probability = { 0.7, 0.2, 0.1 };
            using NDArray weights = np.array(probability);
            var counts = new double[3];
            for (int i = 0; i < 1000; i++) {
                using GraphInstance sample = space.Sample(probability: new object?[] { new object?[] { weights }, null }, numNodes: 1);
                counts[Convert.ToInt64(((NDArray)sample.Nodes!).GetAtIndex(0))]++;
            }

            AssertDistribution(counts, probability, "nodes");
        }

        /// <summary>1000 single-edge samples weighted by an edge probability mask match its distribution.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py#L168">test_probability_edge_sampling</see>.</remarks>
        [TestMethod]
        public void TestProbabilityEdgeSampling() {
            using var space = new Graph(new Discrete(3), new Discrete(3));
            space.Seed(42);
            double[] probability = { 0.5, 0.3, 0.2 };
            using NDArray weights = np.array(probability);
            var counts = new double[3];
            for (int i = 0; i < 1000; i++) {
                using GraphInstance sample = space.Sample(probability: new object?[] { null, new object?[] { weights } }, numEdges: 1);
                counts[Convert.ToInt64(((NDArray)sample.Edges!).GetAtIndex(0))]++;
            }

            AssertDistribution(counts, probability, "edges");
        }

        /// <summary>Node and edge probability masks together, each matching its distribution.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/spaces/test_graph.py#L197">test_probability_node_and_edge_sampling</see>.</remarks>
        [TestMethod]
        public void TestProbabilityNodeAndEdgeSampling() {
            using var space = new Graph(new Discrete(3), new Discrete(3));
            space.Seed(42);
            double[] nodeProbability = { 0.6, 0.3, 0.1 };
            double[] edgeProbability = { 0.4, 0.4, 0.2 };
            using NDArray nodeWeights = np.array(nodeProbability);
            using NDArray edgeWeights = np.array(edgeProbability);
            var nodeCounts = new double[3];
            var edgeCounts = new double[3];
            for (int i = 0; i < 1000; i++) {
                using GraphInstance sample = space.Sample(probability: new object?[] { new object?[] { nodeWeights }, new object?[] { edgeWeights } }, numNodes: 1, numEdges: 1);
                nodeCounts[Convert.ToInt64(((NDArray)sample.Nodes!).GetAtIndex(0))]++;
                edgeCounts[Convert.ToInt64(((NDArray)sample.Edges!).GetAtIndex(0))]++;
            }

            AssertDistribution(nodeCounts, nodeProbability, "nodes");
            AssertDistribution(edgeCounts, edgeProbability, "edges");
        }

        /// <summary>Gymnasium's <c>np.allclose(empirical, probability, atol=0.05)</c>.</summary>
        /// <param name="counts">The counts of 1000 draws.</param>
        /// <param name="probability">The expected distribution.</param>
        /// <param name="what">What was drawn, for messages.</param>
        private static void AssertDistribution(double[] counts, double[] probability, string what) {
            for (int i = 0; i < counts.Length; i++)
                Assert.IsTrue(Math.Abs(counts[i] / 1000 - probability[i]) <= 0.05 + 1e-5 * Math.Abs(probability[i]), $"{what}: value {i} frequency {counts[i] / 1000}, expected {probability[i]}");
        }

        /// <summary>An int64 vector.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private static NDArray Ints(params long[] values) => np.array(values);

        /// <summary>An int64 matrix of edge links.</summary>
        /// <param name="links">The links.</param>
        /// <returns>A new array.</returns>
        private static NDArray Links(long[,] links) => np.array(links);

        /// <summary>Disposes the arrays of a mask tuple.</summary>
        /// <param name="masks">The masks.</param>
        private static void Release(object?[] masks) {
            foreach (object? mask in masks)
                (mask as IDisposable)?.Dispose();
        }
    }
}
