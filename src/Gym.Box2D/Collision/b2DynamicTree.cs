/*
* Copyright (c) 2009 Erin Catto http://www.box2d.org
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2DynamicTree.h and b2DynamicTree.cpp of pybox2d
// 2.3.10's bundled Box2D 2.3.0, translated to C#. The node pool is a managed array grown by doubling with the same free
// list, so proxy ids (and so the broad-phase's pair order) are the C++ ones; the parent/next union is one field with two
// names; the templated Query and RayCast take a callback through a generic interface constraint. m_path (written,
// never read in 2.3) and RebuildBottomUp (no caller in the engine) are left out.
using System;
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>A node of the dynamic tree: a leaf holds a proxy, an internal node the union of its children's boxes.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2DynamicTree.h#L28">Box2D/Collision/b2DynamicTree.h</see>.</remarks>
    internal struct b2TreeNode {
        /// <summary>The enlarged box.</summary>
        public b2AABB aabb;

        /// <summary>The proxy's user data (the broad-phase stores a <see cref="b2FixtureProxy"/>).</summary>
        public object? userData;

        /// <summary>The parent node, or, for a free node, the next free node (C++'s union of parent and next).</summary>
        public int parent;

        /// <summary>The first child, <see cref="b2DynamicTree.b2_nullNode"/> for a leaf.</summary>
        public int child1;

        /// <summary>The second child.</summary>
        public int child2;

        /// <summary>The height: 0 for a leaf, -1 for a free node.</summary>
        public int height;

        /// <summary>The next free node: the same storage as <see cref="parent"/>.</summary>
        public int next {
            readonly get => parent;
            set => parent = value;
        }

        /// <summary>Whether this is a leaf.</summary>
        /// <returns>True when it has no children.</returns>
        public readonly bool IsLeaf() => child1 == b2DynamicTree.b2_nullNode;
    }

    /// <summary>The broad-phase tree's query callback (C++'s templated <c>QueryCallback</c>).</summary>
    internal interface Ib2TreeQueryCallback {
        /// <summary>Called for each proxy whose fat box overlaps the query box.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <returns>False to stop the query.</returns>
        bool QueryCallback(int proxyId);
    }

    /// <summary>The broad-phase tree's ray-cast callback (C++'s templated <c>RayCastCallback</c>).</summary>
    internal interface Ib2TreeRayCastCallback {
        /// <summary>Called for each proxy the ray's box and separating axis don't rule out.</summary>
        /// <param name="input">The ray, clipped to the current max fraction.</param>
        /// <param name="proxyId">The proxy.</param>
        /// <returns>0 to stop, a fraction to clip the ray to, or -1 to ignore the proxy.</returns>
        float RayCastCallback(in b2RayCastInput input, int proxyId);
    }

    /// <summary>
    ///     A dynamic AABB tree of fattened proxies (inspired by Nathanael Presson's btDbvt): moving a proxy only re-inserts
    ///     it when its box leaves the fat box, insertion uses the surface-area heuristic, and rotations keep it balanced.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2DynamicTree.cpp">Box2D/Collision/b2DynamicTree.cpp</see>.</remarks>
    internal sealed class b2DynamicTree {
        /// <summary>The null node id.</summary>
        public const int b2_nullNode = -1;

        /// <summary>The root node, or <see cref="b2_nullNode"/> for an empty tree.</summary>
        private int m_root;

        /// <summary>The node pool.</summary>
        private b2TreeNode[] m_nodes;

        /// <summary>The number of allocated nodes.</summary>
        private int m_nodeCount;

        /// <summary>The pool's capacity.</summary>
        private int m_nodeCapacity;

        /// <summary>The head of the free list.</summary>
        private int m_freeList;

        /// <summary>The number of leaf insertions (statistics; nothing in the engine reads it).</summary>
        private int m_insertionCount;

        /// <summary>The number of leaf insertions so far, a statistic of the tree's churn.</summary>
        public int InsertionCount => m_insertionCount;

        /// <summary>An empty tree with a pool of 16 nodes, all on the free list.</summary>
        public b2DynamicTree() {
            m_root = b2_nullNode;

            m_nodeCapacity = 16;
            m_nodeCount = 0;
            m_nodes = new b2TreeNode[m_nodeCapacity];

            // Build a linked list for the free list.
            for (int i = 0; i < m_nodeCapacity - 1; ++i) {
                m_nodes[i].next = i + 1;
                m_nodes[i].height = -1;
            }

            m_nodes[m_nodeCapacity - 1].next = b2_nullNode;
            m_nodes[m_nodeCapacity - 1].height = -1;
            m_freeList = 0;

            m_insertionCount = 0;
        }

        /// <summary>Allocates a node from the pool, doubling the pool when the free list is empty.</summary>
        /// <returns>The node id.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown if the free list and the node count disagree.</exception>
        private int AllocateNode() {
            // Expand the node pool as needed.
            if (m_freeList == b2_nullNode) {
                b2Assert(m_nodeCount == m_nodeCapacity);

                // The free list is empty. Rebuild a bigger pool.
                m_nodeCapacity *= 2;
                Array.Resize(ref m_nodes, m_nodeCapacity);

                // Build a linked list for the free list. The parent
                // pointer becomes the "next" pointer.
                for (int i = m_nodeCount; i < m_nodeCapacity - 1; ++i) {
                    m_nodes[i].next = i + 1;
                    m_nodes[i].height = -1;
                }

                m_nodes[m_nodeCapacity - 1].next = b2_nullNode;
                m_nodes[m_nodeCapacity - 1].height = -1;
                m_freeList = m_nodeCount;
            }

            // Peel a node off the free list.
            int nodeId = m_freeList;
            m_freeList = m_nodes[nodeId].next;
            m_nodes[nodeId].parent = b2_nullNode;
            m_nodes[nodeId].child1 = b2_nullNode;
            m_nodes[nodeId].child2 = b2_nullNode;
            m_nodes[nodeId].height = 0;
            m_nodes[nodeId].userData = null;
            ++m_nodeCount;
            return nodeId;
        }

        /// <summary>Returns a node to the pool.</summary>
        /// <param name="nodeId">The node.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for an id outside the pool or an empty pool.</exception>
        private void FreeNode(int nodeId) {
            b2Assert(0 <= nodeId && nodeId < m_nodeCapacity);
            b2Assert(0 < m_nodeCount);
            m_nodes[nodeId].next = m_freeList;
            m_nodes[nodeId].height = -1;
            // A freed node keeps no reference alive (C++ leaves the stale pointer, which nothing reads).
            m_nodes[nodeId].userData = null;
            m_freeList = nodeId;
            --m_nodeCount;
        }

        /// <summary>Creates a leaf proxy for a box fattened by <c>b2_aabbExtension</c>.</summary>
        /// <param name="aabb">The tight box.</param>
        /// <param name="userData">The proxy's user data.</param>
        /// <returns>The proxy id, stable until the proxy is destroyed.</returns>
        public int CreateProxy(in b2AABB aabb, object? userData) {
            int proxyId = AllocateNode();

            // Fatten the aabb.
            b2Vec2 r = new b2Vec2(b2_aabbExtension, b2_aabbExtension);
            m_nodes[proxyId].aabb.lowerBound = aabb.lowerBound - r;
            m_nodes[proxyId].aabb.upperBound = aabb.upperBound + r;
            m_nodes[proxyId].userData = userData;
            m_nodes[proxyId].height = 0;

            InsertLeaf(proxyId);

            return proxyId;
        }

        /// <summary>Destroys a leaf proxy.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for an id outside the pool or a non-leaf node.</exception>
        public void DestroyProxy(int proxyId) {
            b2Assert(0 <= proxyId && proxyId < m_nodeCapacity);
            b2Assert(m_nodes[proxyId].IsLeaf());

            RemoveLeaf(proxyId);
            FreeNode(proxyId);
        }

        /// <summary>
        ///     Moves a proxy: when the new box leaves the fat box, the proxy is re-inserted with a box fattened by the
        ///     extension and stretched by twice the displacement.
        /// </summary>
        /// <param name="proxyId">The proxy.</param>
        /// <param name="aabb">The new tight box.</param>
        /// <param name="displacement">The movement that predicts the next boxes.</param>
        /// <returns>True when the proxy was re-inserted (the broad-phase must look for new pairs).</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an id outside the pool or a non-leaf node.</exception>
        public bool MoveProxy(int proxyId, in b2AABB aabb, in b2Vec2 displacement) {
            b2Assert(0 <= proxyId && proxyId < m_nodeCapacity);

            b2Assert(m_nodes[proxyId].IsLeaf());

            if (m_nodes[proxyId].aabb.Contains(aabb))
                return false;

            RemoveLeaf(proxyId);

            // Extend AABB.
            b2AABB b = aabb;
            b2Vec2 r = new b2Vec2(b2_aabbExtension, b2_aabbExtension);
            b.lowerBound = b.lowerBound - r;
            b.upperBound = b.upperBound + r;

            // Predict AABB displacement.
            b2Vec2 d = b2_aabbMultiplier * displacement;

            if (d.x < 0.0f)
                b.lowerBound.x += d.x;
            else
                b.upperBound.x += d.x;

            if (d.y < 0.0f)
                b.lowerBound.y += d.y;
            else
                b.upperBound.y += d.y;

            m_nodes[proxyId].aabb = b;

            InsertLeaf(proxyId);
            return true;
        }

        /// <summary>A proxy's user data.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <returns>The user data.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an id outside the pool.</exception>
        public object? GetUserData(int proxyId) {
            b2Assert(0 <= proxyId && proxyId < m_nodeCapacity);
            return m_nodes[proxyId].userData;
        }

        /// <summary>A proxy's fat box.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <returns>The fat box.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an id outside the pool.</exception>
        public b2AABB GetFatAABB(int proxyId) {
            b2Assert(0 <= proxyId && proxyId < m_nodeCapacity);
            return m_nodes[proxyId].aabb;
        }

        /// <summary>Reports every proxy whose fat box overlaps <paramref name="aabb"/>, in the order of a depth-first stack walk.</summary>
        /// <typeparam name="T">The callback type; a struct callback is copied, so keep its mutable state in referenced objects.</typeparam>
        /// <param name="callback">The callback; returning false stops the query.</param>
        /// <param name="aabb">The query box.</param>
        public void Query<T>(T callback, in b2AABB aabb) where T : Ib2TreeQueryCallback {
            b2GrowableStack stack = new b2GrowableStack(256);
            stack.Push(m_root);

            while (stack.GetCount() > 0) {
                int nodeId = stack.Pop();
                if (nodeId == b2_nullNode)
                    continue;

                ref b2TreeNode node = ref m_nodes[nodeId];

                if (b2Collision.b2TestOverlap(node.aabb, aabb)) {
                    if (node.IsLeaf()) {
                        bool proceed = callback.QueryCallback(nodeId);
                        if (proceed == false)
                            return;
                    } else {
                        stack.Push(node.child1);
                        stack.Push(node.child2);
                    }
                }
            }
        }

        /// <summary>
        ///     Casts a ray through the tree, reporting the proxies its box and separating axis don't rule out; the callback
        ///     clips or stops the ray.
        /// </summary>
        /// <typeparam name="T">The callback type; a struct callback is copied, so keep its mutable state in referenced objects.</typeparam>
        /// <param name="callback">The callback.</param>
        /// <param name="input">The ray.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown for a ray of zero length.</exception>
        public void RayCast<T>(T callback, in b2RayCastInput input) where T : Ib2TreeRayCastCallback {
            b2Vec2 p1 = input.p1;
            b2Vec2 p2 = input.p2;
            b2Vec2 r = p2 - p1;
            b2Assert(r.LengthSquared() > 0.0f);
            r.Normalize();

            // v is perpendicular to the segment.
            b2Vec2 v = b2Cross(1.0f, r);
            b2Vec2 abs_v = b2Abs(v);

            // Separating axis for segment (Gino, p80).
            // |dot(v, p1 - c)| > dot(|v|, h)

            float maxFraction = input.maxFraction;

            // Build a bounding box for the segment.
            b2AABB segmentAABB;
            {
                b2Vec2 t = p1 + maxFraction * (p2 - p1);
                segmentAABB.lowerBound = b2Min(p1, t);
                segmentAABB.upperBound = b2Max(p1, t);
            }

            b2GrowableStack stack = new b2GrowableStack(256);
            stack.Push(m_root);

            while (stack.GetCount() > 0) {
                int nodeId = stack.Pop();
                if (nodeId == b2_nullNode)
                    continue;

                ref b2TreeNode node = ref m_nodes[nodeId];

                if (b2Collision.b2TestOverlap(node.aabb, segmentAABB) == false)
                    continue;

                // Separating axis for segment (Gino, p80).
                // |dot(v, p1 - c)| > dot(|v|, h)
                b2Vec2 c = node.aabb.GetCenter();
                b2Vec2 h = node.aabb.GetExtents();
                float separation = b2Abs(b2Dot(v, p1 - c)) - b2Dot(abs_v, h);
                if (separation > 0.0f)
                    continue;

                if (node.IsLeaf()) {
                    b2RayCastInput subInput;
                    subInput.p1 = input.p1;
                    subInput.p2 = input.p2;
                    subInput.maxFraction = maxFraction;

                    float value = callback.RayCastCallback(subInput, nodeId);

                    if (value == 0.0f)
                        // The client has terminated the ray cast.
                        return;

                    if (value > 0.0f) {
                        // Update segment bounding box.
                        maxFraction = value;
                        b2Vec2 t = p1 + maxFraction * (p2 - p1);
                        segmentAABB.lowerBound = b2Min(p1, t);
                        segmentAABB.upperBound = b2Max(p1, t);
                    }
                } else {
                    stack.Push(node.child1);
                    stack.Push(node.child2);
                }
            }
        }

        /// <summary>
        ///     Inserts a leaf: descends by the surface-area cost (a new parent here, or the cheaper child), creates the new
        ///     parent, then walks up re-balancing and refitting.
        /// </summary>
        /// <param name="leaf">The leaf.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown if an internal node lacks a child.</exception>
        private void InsertLeaf(int leaf) {
            ++m_insertionCount;

            if (m_root == b2_nullNode) {
                m_root = leaf;
                m_nodes[m_root].parent = b2_nullNode;
                return;
            }

            // Find the best sibling for this node
            b2AABB leafAABB = m_nodes[leaf].aabb;
            int index = m_root;
            while (m_nodes[index].IsLeaf() == false) {
                int child1 = m_nodes[index].child1;
                int child2 = m_nodes[index].child2;

                float area = m_nodes[index].aabb.GetPerimeter();

                b2AABB combinedAABB = default;
                combinedAABB.Combine(m_nodes[index].aabb, leafAABB);
                float combinedArea = combinedAABB.GetPerimeter();

                // Cost of creating a new parent for this node and the new leaf
                float cost = 2.0f * combinedArea;

                // Minimum cost of pushing the leaf further down the tree
                float inheritanceCost = 2.0f * (combinedArea - area);

                // Cost of descending into child1
                float cost1;
                if (m_nodes[child1].IsLeaf()) {
                    b2AABB aabb = default;
                    aabb.Combine(leafAABB, m_nodes[child1].aabb);
                    cost1 = aabb.GetPerimeter() + inheritanceCost;
                } else {
                    b2AABB aabb = default;
                    aabb.Combine(leafAABB, m_nodes[child1].aabb);
                    float oldArea = m_nodes[child1].aabb.GetPerimeter();
                    float newArea = aabb.GetPerimeter();
                    cost1 = (newArea - oldArea) + inheritanceCost;
                }

                // Cost of descending into child2
                float cost2;
                if (m_nodes[child2].IsLeaf()) {
                    b2AABB aabb = default;
                    aabb.Combine(leafAABB, m_nodes[child2].aabb);
                    cost2 = aabb.GetPerimeter() + inheritanceCost;
                } else {
                    b2AABB aabb = default;
                    aabb.Combine(leafAABB, m_nodes[child2].aabb);
                    float oldArea = m_nodes[child2].aabb.GetPerimeter();
                    float newArea = aabb.GetPerimeter();
                    cost2 = newArea - oldArea + inheritanceCost;
                }

                // Descend according to the minimum cost.
                if (cost < cost1 && cost < cost2)
                    break;

                // Descend
                if (cost1 < cost2)
                    index = child1;
                else
                    index = child2;
            }

            int sibling = index;

            // Create a new parent.
            int oldParent = m_nodes[sibling].parent;
            int newParent = AllocateNode();
            m_nodes[newParent].parent = oldParent;
            m_nodes[newParent].userData = null;
            m_nodes[newParent].aabb.Combine(leafAABB, m_nodes[sibling].aabb);
            m_nodes[newParent].height = m_nodes[sibling].height + 1;

            if (oldParent != b2_nullNode) {
                // The sibling was not the root.
                if (m_nodes[oldParent].child1 == sibling)
                    m_nodes[oldParent].child1 = newParent;
                else
                    m_nodes[oldParent].child2 = newParent;

                m_nodes[newParent].child1 = sibling;
                m_nodes[newParent].child2 = leaf;
                m_nodes[sibling].parent = newParent;
                m_nodes[leaf].parent = newParent;
            } else {
                // The sibling was the root.
                m_nodes[newParent].child1 = sibling;
                m_nodes[newParent].child2 = leaf;
                m_nodes[sibling].parent = newParent;
                m_nodes[leaf].parent = newParent;
                m_root = newParent;
            }

            // Walk back up the tree fixing heights and AABBs
            index = m_nodes[leaf].parent;
            while (index != b2_nullNode) {
                index = Balance(index);

                int child1 = m_nodes[index].child1;
                int child2 = m_nodes[index].child2;

                b2Assert(child1 != b2_nullNode);
                b2Assert(child2 != b2_nullNode);

                m_nodes[index].height = 1 + b2Max(m_nodes[child1].height, m_nodes[child2].height);
                m_nodes[index].aabb.Combine(m_nodes[child1].aabb, m_nodes[child2].aabb);

                index = m_nodes[index].parent;
            }
        }

        /// <summary>Removes a leaf: its parent is freed and the sibling takes its place; ancestors are re-balanced and refitted.</summary>
        /// <param name="leaf">The leaf.</param>
        private void RemoveLeaf(int leaf) {
            if (leaf == m_root) {
                m_root = b2_nullNode;
                return;
            }

            int parent = m_nodes[leaf].parent;
            int grandParent = m_nodes[parent].parent;
            int sibling;
            if (m_nodes[parent].child1 == leaf)
                sibling = m_nodes[parent].child2;
            else
                sibling = m_nodes[parent].child1;

            if (grandParent != b2_nullNode) {
                // Destroy parent and connect sibling to grandParent.
                if (m_nodes[grandParent].child1 == parent)
                    m_nodes[grandParent].child1 = sibling;
                else
                    m_nodes[grandParent].child2 = sibling;
                m_nodes[sibling].parent = grandParent;
                FreeNode(parent);

                // Adjust ancestor bounds.
                int index = grandParent;
                while (index != b2_nullNode) {
                    index = Balance(index);

                    int child1 = m_nodes[index].child1;
                    int child2 = m_nodes[index].child2;

                    m_nodes[index].aabb.Combine(m_nodes[child1].aabb, m_nodes[child2].aabb);
                    m_nodes[index].height = 1 + b2Max(m_nodes[child1].height, m_nodes[child2].height);

                    index = m_nodes[index].parent;
                }
            } else {
                m_root = sibling;
                m_nodes[sibling].parent = b2_nullNode;
                FreeNode(parent);
            }
        }

        /// <summary>Performs a left or right rotation if node A is imbalanced by more than one level.</summary>
        /// <param name="iA">The node.</param>
        /// <returns>The index of the subtree's new root.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a null node or children outside the pool.</exception>
        private int Balance(int iA) {
            b2Assert(iA != b2_nullNode);

            ref b2TreeNode A = ref m_nodes[iA];
            if (A.IsLeaf() || A.height < 2)
                return iA;

            int iB = A.child1;
            int iC = A.child2;
            b2Assert(0 <= iB && iB < m_nodeCapacity);
            b2Assert(0 <= iC && iC < m_nodeCapacity);

            ref b2TreeNode B = ref m_nodes[iB];
            ref b2TreeNode C = ref m_nodes[iC];

            int balance = C.height - B.height;

            // Rotate C up
            if (balance > 1) {
                int iF = C.child1;
                int iG = C.child2;
                ref b2TreeNode F = ref m_nodes[iF];
                ref b2TreeNode G = ref m_nodes[iG];
                b2Assert(0 <= iF && iF < m_nodeCapacity);
                b2Assert(0 <= iG && iG < m_nodeCapacity);

                // Swap A and C
                C.child1 = iA;
                C.parent = A.parent;
                A.parent = iC;

                // A's old parent should point to C
                if (C.parent != b2_nullNode) {
                    if (m_nodes[C.parent].child1 == iA) {
                        m_nodes[C.parent].child1 = iC;
                    } else {
                        b2Assert(m_nodes[C.parent].child2 == iA, "m_nodes[C->parent].child2 == iA");
                        m_nodes[C.parent].child2 = iC;
                    }
                } else {
                    m_root = iC;
                }

                // Rotate
                if (F.height > G.height) {
                    C.child2 = iF;
                    A.child2 = iG;
                    G.parent = iA;
                    A.aabb.Combine(B.aabb, G.aabb);
                    C.aabb.Combine(A.aabb, F.aabb);

                    A.height = 1 + b2Max(B.height, G.height);
                    C.height = 1 + b2Max(A.height, F.height);
                } else {
                    C.child2 = iG;
                    A.child2 = iF;
                    F.parent = iA;
                    A.aabb.Combine(B.aabb, F.aabb);
                    C.aabb.Combine(A.aabb, G.aabb);

                    A.height = 1 + b2Max(B.height, F.height);
                    C.height = 1 + b2Max(A.height, G.height);
                }

                return iC;
            }

            // Rotate B up
            if (balance < -1) {
                int iD = B.child1;
                int iE = B.child2;
                ref b2TreeNode D = ref m_nodes[iD];
                ref b2TreeNode E = ref m_nodes[iE];
                b2Assert(0 <= iD && iD < m_nodeCapacity);
                b2Assert(0 <= iE && iE < m_nodeCapacity);

                // Swap A and B
                B.child1 = iA;
                B.parent = A.parent;
                A.parent = iB;

                // A's old parent should point to B
                if (B.parent != b2_nullNode) {
                    if (m_nodes[B.parent].child1 == iA) {
                        m_nodes[B.parent].child1 = iB;
                    } else {
                        b2Assert(m_nodes[B.parent].child2 == iA, "m_nodes[B->parent].child2 == iA");
                        m_nodes[B.parent].child2 = iB;
                    }
                } else {
                    m_root = iB;
                }

                // Rotate
                if (D.height > E.height) {
                    B.child2 = iD;
                    A.child1 = iE;
                    E.parent = iA;
                    A.aabb.Combine(C.aabb, E.aabb);
                    B.aabb.Combine(A.aabb, D.aabb);

                    A.height = 1 + b2Max(C.height, E.height);
                    B.height = 1 + b2Max(A.height, D.height);
                } else {
                    B.child2 = iE;
                    A.child1 = iD;
                    D.parent = iA;
                    A.aabb.Combine(C.aabb, D.aabb);
                    B.aabb.Combine(A.aabb, E.aabb);

                    A.height = 1 + b2Max(C.height, D.height);
                    B.height = 1 + b2Max(A.height, E.height);
                }

                return iB;
            }

            return iA;
        }

        /// <summary>The tree's height (0 for an empty tree).</summary>
        /// <returns>The height.</returns>
        public int GetHeight() {
            if (m_root == b2_nullNode)
                return 0;

            return m_nodes[m_root].height;
        }

        /// <summary>The ratio of the sum of the node perimeters to the root perimeter, a quality metric.</summary>
        /// <returns>The ratio.</returns>
        public float GetAreaRatio() {
            if (m_root == b2_nullNode)
                return 0.0f;

            float rootArea = m_nodes[m_root].aabb.GetPerimeter();

            float totalArea = 0.0f;
            for (int i = 0; i < m_nodeCapacity; ++i) {
                if (m_nodes[i].height < 0)
                    // Free node in pool
                    continue;

                totalArea += m_nodes[i].aabb.GetPerimeter();
            }

            return totalArea / rootArea;
        }

        /// <summary>Computes the height of a subtree by recursion (for validation).</summary>
        /// <param name="nodeId">The subtree's root.</param>
        /// <returns>The height.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for an id outside the pool.</exception>
        private int ComputeHeight(int nodeId) {
            b2Assert(0 <= nodeId && nodeId < m_nodeCapacity);
            if (m_nodes[nodeId].IsLeaf())
                return 0;

            int height1 = ComputeHeight(m_nodes[nodeId].child1);
            int height2 = ComputeHeight(m_nodes[nodeId].child2);
            return 1 + b2Max(height1, height2);
        }

        /// <summary>Computes the tree's height by recursion (for validation).</summary>
        /// <returns>The height.</returns>
        private int ComputeHeight() => ComputeHeight(m_root);

        /// <summary>Checks the parent/child links below a node.</summary>
        /// <param name="index">The node.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown on a broken link.</exception>
        private void ValidateStructure(int index) {
            if (index == b2_nullNode)
                return;

            if (index == m_root)
                b2Assert(m_nodes[index].parent == b2_nullNode);

            int child1 = m_nodes[index].child1;
            int child2 = m_nodes[index].child2;

            if (m_nodes[index].IsLeaf()) {
                b2Assert(child1 == b2_nullNode);
                b2Assert(child2 == b2_nullNode);
                b2Assert(m_nodes[index].height == 0, "node->height == 0");
                return;
            }

            b2Assert(0 <= child1 && child1 < m_nodeCapacity);
            b2Assert(0 <= child2 && child2 < m_nodeCapacity);

            b2Assert(m_nodes[child1].parent == index);
            b2Assert(m_nodes[child2].parent == index);

            ValidateStructure(child1);
            ValidateStructure(child2);
        }

        /// <summary>Checks the heights and boxes below a node.</summary>
        /// <param name="index">The node.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown on a wrong height or box.</exception>
        private void ValidateMetrics(int index) {
            if (index == b2_nullNode)
                return;

            int child1 = m_nodes[index].child1;
            int child2 = m_nodes[index].child2;

            if (m_nodes[index].IsLeaf()) {
                b2Assert(child1 == b2_nullNode);
                b2Assert(child2 == b2_nullNode);
                b2Assert(m_nodes[index].height == 0, "node->height == 0");
                return;
            }

            b2Assert(0 <= child1 && child1 < m_nodeCapacity);
            b2Assert(0 <= child2 && child2 < m_nodeCapacity);

            int height1 = m_nodes[child1].height;
            int height2 = m_nodes[child2].height;
            int height;
            height = 1 + b2Max(height1, height2);
            b2Assert(m_nodes[index].height == height, "node->height == height");

            b2AABB aabb = default;
            aabb.Combine(m_nodes[child1].aabb, m_nodes[child2].aabb);

            b2Assert(aabb.lowerBound == m_nodes[index].aabb.lowerBound, "aabb.lowerBound == node->aabb.lowerBound");
            b2Assert(aabb.upperBound == m_nodes[index].aabb.upperBound, "aabb.upperBound == node->aabb.upperBound");

            ValidateMetrics(child1);
            ValidateMetrics(child2);
        }

        /// <summary>Validates the whole tree: structure, metrics, the free list and the node count.</summary>
        /// <exception cref="NumSharp.AssertionError">Thrown on any inconsistency.</exception>
        public void Validate() {
            ValidateStructure(m_root);
            ValidateMetrics(m_root);

            int freeCount = 0;
            int freeIndex = m_freeList;
            while (freeIndex != b2_nullNode) {
                b2Assert(0 <= freeIndex && freeIndex < m_nodeCapacity);
                freeIndex = m_nodes[freeIndex].next;
                ++freeCount;
            }

            b2Assert(GetHeight() == ComputeHeight());

            b2Assert(m_nodeCount + freeCount == m_nodeCapacity);
        }

        /// <summary>The largest height difference between sibling subtrees.</summary>
        /// <returns>The maximum balance.</returns>
        public int GetMaxBalance() {
            int maxBalance = 0;
            for (int i = 0; i < m_nodeCapacity; ++i) {
                if (m_nodes[i].height <= 1)
                    continue;

                b2Assert(m_nodes[i].IsLeaf() == false, "node->IsLeaf() == false");

                int child1 = m_nodes[i].child1;
                int child2 = m_nodes[i].child2;
                int balance = b2Abs(m_nodes[child2].height - m_nodes[child1].height);
                maxBalance = b2Max(maxBalance, balance);
            }

            return maxBalance;
        }

        /// <summary>Shifts every box by a new world origin.</summary>
        /// <param name="newOrigin">The new origin in the old frame.</param>
        public void ShiftOrigin(in b2Vec2 newOrigin) {
            // Build array of leaves. Free the rest.
            for (int i = 0; i < m_nodeCapacity; ++i) {
                m_nodes[i].aabb.lowerBound -= newOrigin;
                m_nodes[i].aabb.upperBound -= newOrigin;
            }
        }
    }
}
