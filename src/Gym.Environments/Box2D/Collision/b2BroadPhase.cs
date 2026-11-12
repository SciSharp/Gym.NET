/*
* Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2BroadPhase.h and b2BroadPhase.cpp of pybox2d
// 2.3.10's bundled Box2D 2.3.0, translated to C#. The move and pair buffers are managed arrays grown by doubling; the
// pair sort is Array.Sort under b2PairLessThan's order, which yields the same sequence as std::sort because equal pairs
// are identical; the templated client callback is the Ib2PairCallback interface.
using System;
using System.Collections.Generic;
using static Gym.Box2D.b2Math;

namespace Gym.Box2D {
    /// <summary>A pair of overlapping proxies, ordered so <see cref="proxyIdA"/> is the smaller id.</summary>
    internal struct b2Pair {
        /// <summary>The smaller proxy id.</summary>
        public int proxyIdA;

        /// <summary>The larger proxy id.</summary>
        public int proxyIdB;
    }

    /// <summary>The broad-phase's client, told about each new pair of overlapping proxies (C++'s templated <c>AddPair</c>).</summary>
    internal interface Ib2PairCallback {
        /// <summary>Called once per overlapping pair, in ascending order of (proxyIdA, proxyIdB).</summary>
        /// <param name="proxyUserDataA">The user data of the smaller proxy id.</param>
        /// <param name="proxyUserDataB">The user data of the larger proxy id.</param>
        void AddPair(object? proxyUserDataA, object? proxyUserDataB);
    }

    /// <summary>
    ///     The broad-phase: proxies in a dynamic tree, a buffer of moved proxies, and the pairs those moves create. New pairs
    ///     reach the client sorted by proxy id, which fixes the order contacts are created in.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2BroadPhase.cpp">Box2D/Collision/b2BroadPhase.cpp</see>.</remarks>
    internal sealed class b2BroadPhase : Ib2TreeQueryCallback {
        /// <summary>The null proxy id.</summary>
        public const int e_nullProxy = -1;

        /// <summary>The proxies' tree.</summary>
        private readonly b2DynamicTree m_tree = new b2DynamicTree();

        /// <summary>The number of proxies.</summary>
        private int m_proxyCount;

        /// <summary>The proxies moved since the last pair update (destroyed ones become <see cref="e_nullProxy"/>).</summary>
        private int[] m_moveBuffer;

        /// <summary>The number of moved proxies.</summary>
        private int m_moveCount;

        /// <summary>The pairs found by the current update.</summary>
        private b2Pair[] m_pairBuffer;

        /// <summary>The number of pairs found.</summary>
        private int m_pairCount;

        /// <summary>The proxy being queried for pairs.</summary>
        private int m_queryProxyId;

        /// <summary>An empty broad-phase with room for 16 moves and 16 pairs before its buffers grow.</summary>
        public b2BroadPhase() {
            m_proxyCount = 0;

            m_pairCount = 0;
            m_pairBuffer = new b2Pair[16];

            m_moveCount = 0;
            m_moveBuffer = new int[16];
        }

        /// <summary>Creates a proxy with a fattened box; it is buffered as moved, so its pairs appear at the next update.</summary>
        /// <param name="aabb">The tight box.</param>
        /// <param name="userData">The proxy's user data.</param>
        /// <returns>The proxy id.</returns>
        public int CreateProxy(in b2AABB aabb, object? userData) {
            int proxyId = m_tree.CreateProxy(aabb, userData);
            ++m_proxyCount;
            BufferMove(proxyId);
            return proxyId;
        }

        /// <summary>Destroys a proxy; the client destroys its pairs.</summary>
        /// <param name="proxyId">The proxy.</param>
        public void DestroyProxy(int proxyId) {
            UnBufferMove(proxyId);
            --m_proxyCount;
            m_tree.DestroyProxy(proxyId);
        }

        /// <summary>Moves a proxy; when it leaves its fat box it is buffered for the next pair update.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <param name="aabb">The new tight box.</param>
        /// <param name="displacement">The movement since the last move.</param>
        public void MoveProxy(int proxyId, in b2AABB aabb, in b2Vec2 displacement) {
            bool buffer = m_tree.MoveProxy(proxyId, aabb, displacement);
            if (buffer)
                BufferMove(proxyId);
        }

        /// <summary>Buffers a proxy as moved without moving it, so its pairs are re-examined.</summary>
        /// <param name="proxyId">The proxy.</param>
        public void TouchProxy(int proxyId) => BufferMove(proxyId);

        /// <summary>A proxy's fat box.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <returns>The fat box.</returns>
        public b2AABB GetFatAABB(int proxyId) => m_tree.GetFatAABB(proxyId);

        /// <summary>A proxy's user data.</summary>
        /// <param name="proxyId">The proxy.</param>
        /// <returns>The user data.</returns>
        public object? GetUserData(int proxyId) => m_tree.GetUserData(proxyId);

        /// <summary>Whether two proxies' fat boxes overlap.</summary>
        /// <param name="proxyIdA">The first proxy.</param>
        /// <param name="proxyIdB">The second proxy.</param>
        /// <returns>True when they overlap.</returns>
        public bool TestOverlap(int proxyIdA, int proxyIdB) {
            b2AABB aabbA = m_tree.GetFatAABB(proxyIdA);
            b2AABB aabbB = m_tree.GetFatAABB(proxyIdB);
            return b2Collision.b2TestOverlap(aabbA, aabbB);
        }

        /// <summary>The number of proxies.</summary>
        /// <returns>The count.</returns>
        public int GetProxyCount() => m_proxyCount;

        /// <summary>
        ///     Finds the pairs of the proxies moved since the last update and sends each new one to the client once, sorted by
        ///     (proxyIdA, proxyIdB); clears the move buffer.
        /// </summary>
        /// <typeparam name="T">The client type.</typeparam>
        /// <param name="callback">The client.</param>
        public void UpdatePairs<T>(T callback) where T : Ib2PairCallback {
            // Reset pair buffer
            m_pairCount = 0;

            // Perform tree queries for all moving proxies.
            for (int i = 0; i < m_moveCount; ++i) {
                m_queryProxyId = m_moveBuffer[i];
                if (m_queryProxyId == e_nullProxy)
                    continue;

                // We have to query the tree with the fat AABB so that
                // we don't fail to create a pair that may touch later.
                b2AABB fatAABB = m_tree.GetFatAABB(m_queryProxyId);

                // Query tree, create pairs and add them pair buffer.
                m_tree.Query(this, fatAABB);
            }

            // Reset move buffer
            m_moveCount = 0;

            // Sort the pair buffer to expose duplicates.
            Array.Sort(m_pairBuffer, 0, m_pairCount, b2PairComparer.Instance);

            // Send the pairs back to the client.
            int pairIndex = 0;
            while (pairIndex < m_pairCount) {
                b2Pair primaryPair = m_pairBuffer[pairIndex];
                object? userDataA = m_tree.GetUserData(primaryPair.proxyIdA);
                object? userDataB = m_tree.GetUserData(primaryPair.proxyIdB);

                callback.AddPair(userDataA, userDataB);
                ++pairIndex;

                // Skip any duplicate pairs.
                while (pairIndex < m_pairCount) {
                    b2Pair pair = m_pairBuffer[pairIndex];
                    if (pair.proxyIdA != primaryPair.proxyIdA || pair.proxyIdB != primaryPair.proxyIdB)
                        break;
                    ++pairIndex;
                }
            }

            // Try to keep the tree balanced.
            //m_tree.Rebalance(4);
        }

        /// <summary>Reports the proxies whose fat boxes overlap a box.</summary>
        /// <typeparam name="T">The callback type.</typeparam>
        /// <param name="callback">The callback; returning false stops the query.</param>
        /// <param name="aabb">The query box.</param>
        public void Query<T>(T callback, in b2AABB aabb) where T : Ib2TreeQueryCallback => m_tree.Query(callback, aabb);

        /// <summary>Casts a ray against the proxies.</summary>
        /// <typeparam name="T">The callback type.</typeparam>
        /// <param name="callback">The callback, which clips or stops the ray.</param>
        /// <param name="input">The ray.</param>
        public void RayCast<T>(T callback, in b2RayCastInput input) where T : Ib2TreeRayCastCallback => m_tree.RayCast(callback, input);

        /// <summary>The tree's height.</summary>
        /// <returns>The height.</returns>
        public int GetTreeHeight() => m_tree.GetHeight();

        /// <summary>The tree's largest sibling height difference.</summary>
        /// <returns>The balance.</returns>
        public int GetTreeBalance() => m_tree.GetMaxBalance();

        /// <summary>The tree's area ratio, a quality metric.</summary>
        /// <returns>The ratio.</returns>
        public float GetTreeQuality() => m_tree.GetAreaRatio();

        /// <summary>Shifts the world origin.</summary>
        /// <param name="newOrigin">The new origin in the old frame.</param>
        public void ShiftOrigin(in b2Vec2 newOrigin) => m_tree.ShiftOrigin(newOrigin);

        /// <summary>Appends a proxy to the move buffer, doubling it when full.</summary>
        /// <param name="proxyId">The proxy.</param>
        private void BufferMove(int proxyId) {
            if (m_moveCount == m_moveBuffer.Length)
                Array.Resize(ref m_moveBuffer, m_moveBuffer.Length * 2);

            m_moveBuffer[m_moveCount] = proxyId;
            ++m_moveCount;
        }

        /// <summary>Nulls a destroyed proxy's entries in the move buffer.</summary>
        /// <param name="proxyId">The proxy.</param>
        private void UnBufferMove(int proxyId) {
            for (int i = 0; i < m_moveCount; ++i) {
                if (m_moveBuffer[i] == proxyId)
                    m_moveBuffer[i] = e_nullProxy;
            }
        }

        /// <summary>Records a candidate pair with the proxy being queried (called by the tree during <see cref="UpdatePairs{T}"/>).</summary>
        /// <param name="proxyId">The overlapping proxy.</param>
        /// <returns>Always true, to keep querying.</returns>
        public bool QueryCallback(int proxyId) {
            // A proxy cannot form a pair with itself.
            if (proxyId == m_queryProxyId)
                return true;

            // Grow the pair buffer as needed.
            if (m_pairCount == m_pairBuffer.Length)
                Array.Resize(ref m_pairBuffer, m_pairBuffer.Length * 2);

            m_pairBuffer[m_pairCount].proxyIdA = b2Min(proxyId, m_queryProxyId);
            m_pairBuffer[m_pairCount].proxyIdB = b2Max(proxyId, m_queryProxyId);
            ++m_pairCount;

            return true;
        }

        /// <summary>The order of <c>b2PairLessThan</c>: by proxyIdA, then proxyIdB.</summary>
        private sealed class b2PairComparer : IComparer<b2Pair> {
            /// <summary>The comparer.</summary>
            public static readonly b2PairComparer Instance = new b2PairComparer();

            /// <summary>Compares two pairs by proxyIdA, then proxyIdB.</summary>
            /// <param name="pair1">The first pair.</param>
            /// <param name="pair2">The second pair.</param>
            /// <returns>Negative, zero or positive.</returns>
            public int Compare(b2Pair pair1, b2Pair pair2) {
                if (pair1.proxyIdA != pair2.proxyIdA)
                    return pair1.proxyIdA < pair2.proxyIdA ? -1 : 1;
                return pair1.proxyIdB.CompareTo(pair2.proxyIdB);
            }
        }
    }
}
