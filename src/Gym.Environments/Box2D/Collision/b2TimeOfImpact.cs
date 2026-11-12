/*
* Copyright (c) 2007-2009 Erin Catto http://www.box2d.org
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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Collision/b2TimeOfImpact.h and b2TimeOfImpact.cpp of pybox2d
// 2.3.10's bundled Box2D 2.3.0, translated to C#. The timing and iteration statistics (b2_toiTime, b2_toiCalls, ...) are
// left out: nothing reads them, and they would be shared mutable state across the threads of vector envs.
using static Gym.Box2D.b2Math;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>The input of a time-of-impact query: two proxies, their sweeps, and the interval <c>[0, tMax]</c>.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2TimeOfImpact.h">Box2D/Collision/b2TimeOfImpact.h</see>.</remarks>
    internal struct b2TOIInput {
        /// <summary>The first proxy.</summary>
        public b2DistanceProxy proxyA;

        /// <summary>The second proxy.</summary>
        public b2DistanceProxy proxyB;

        /// <summary>The first proxy's sweep.</summary>
        public b2Sweep sweepA;

        /// <summary>The second proxy's sweep.</summary>
        public b2Sweep sweepB;

        /// <summary>The end of the sweep interval.</summary>
        public float tMax;
    }

    /// <summary>How a time-of-impact query ended (C++ <c>b2TOIOutput::State</c>).</summary>
    internal enum b2TOIOutputState {
        /// <summary>Not computed.</summary>
        e_unknown,

        /// <summary>The root finder failed (t holds its last good time).</summary>
        e_failed,

        /// <summary>The shapes overlap at t = 0.</summary>
        e_overlapped,

        /// <summary>The shapes touch at t.</summary>
        e_touching,

        /// <summary>The shapes stay apart through tMax.</summary>
        e_separated,
    }

    /// <summary>The output of a time-of-impact query.</summary>
    internal struct b2TOIOutput {
        /// <summary>How the query ended.</summary>
        public b2TOIOutputState state;

        /// <summary>The time of impact, or the time the query stopped at.</summary>
        public float t;
    }

    /// <summary>The kind of separating axis a <see cref="b2SeparationFunction"/> tracks.</summary>
    internal enum b2SeparationFunctionType {
        /// <summary>Between two points.</summary>
        e_points,

        /// <summary>Along a face of A.</summary>
        e_faceA,

        /// <summary>Along a face of B.</summary>
        e_faceB,
    }

    /// <summary>The separation of two swept proxies along an axis fixed in one of them, evaluated over time.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2TimeOfImpact.cpp#L34">Box2D/Collision/b2TimeOfImpact.cpp</see>.</remarks>
    internal struct b2SeparationFunction {
        /// <summary>Proxy A.</summary>
        public b2DistanceProxy m_proxyA;

        /// <summary>Proxy B.</summary>
        public b2DistanceProxy m_proxyB;

        /// <summary>Sweep A.</summary>
        public b2Sweep m_sweepA;

        /// <summary>Sweep B.</summary>
        public b2Sweep m_sweepB;

        /// <summary>The axis kind.</summary>
        public b2SeparationFunctionType m_type;

        /// <summary>The face midpoint, for face axes.</summary>
        public b2Vec2 m_localPoint;

        /// <summary>The axis, local to the face's proxy (world for points).</summary>
        public b2Vec2 m_axis;

        /// <summary>Builds the axis from a GJK cache at time <paramref name="t1"/>, oriented so the separation is positive.</summary>
        /// <param name="cache">The GJK cache of one or two points per proxy.</param>
        /// <param name="proxyA">Proxy A.</param>
        /// <param name="sweepA">Sweep A.</param>
        /// <param name="proxyB">Proxy B.</param>
        /// <param name="sweepB">Sweep B.</param>
        /// <param name="t1">The time.</param>
        /// <returns>The separation at <paramref name="t1"/>.</returns>
        /// <exception cref="NumSharp.AssertionError">Thrown for a cache of other than 1 or 2 vertices.</exception>
        public float Initialize(in b2SimplexCache cache, in b2DistanceProxy proxyA, in b2Sweep sweepA, in b2DistanceProxy proxyB, in b2Sweep sweepB, float t1) {
            m_proxyA = proxyA;
            m_proxyB = proxyB;
            int count = cache.count;
            b2Assert(0 < count && count < 3);

            m_sweepA = sweepA;
            m_sweepB = sweepB;

            m_sweepA.GetTransform(out b2Transform xfA, t1);
            m_sweepB.GetTransform(out b2Transform xfB, t1);

            if (count == 1) {
                m_type = b2SeparationFunctionType.e_points;
                b2Vec2 localPointA = m_proxyA.GetVertex(cache.indexA[0]);
                b2Vec2 localPointB = m_proxyB.GetVertex(cache.indexB[0]);
                b2Vec2 pointA = b2Mul(xfA, localPointA);
                b2Vec2 pointB = b2Mul(xfB, localPointB);
                m_axis = pointB - pointA;
                float s = m_axis.Normalize();
                return s;
            } else if (cache.indexA[0] == cache.indexA[1]) {
                // Two points on B and one on A.
                m_type = b2SeparationFunctionType.e_faceB;
                b2Vec2 localPointB1 = proxyB.GetVertex(cache.indexB[0]);
                b2Vec2 localPointB2 = proxyB.GetVertex(cache.indexB[1]);

                m_axis = b2Cross(localPointB2 - localPointB1, 1.0f);
                m_axis.Normalize();
                b2Vec2 normal = b2Mul(xfB.q, m_axis);

                m_localPoint = 0.5f * (localPointB1 + localPointB2);
                b2Vec2 pointB = b2Mul(xfB, m_localPoint);

                b2Vec2 localPointA = proxyA.GetVertex(cache.indexA[0]);
                b2Vec2 pointA = b2Mul(xfA, localPointA);

                float s = b2Dot(pointA - pointB, normal);
                if (s < 0.0f) {
                    m_axis = -m_axis;
                    s = -s;
                }

                return s;
            } else {
                // Two points on A and one or two points on B.
                m_type = b2SeparationFunctionType.e_faceA;
                b2Vec2 localPointA1 = m_proxyA.GetVertex(cache.indexA[0]);
                b2Vec2 localPointA2 = m_proxyA.GetVertex(cache.indexA[1]);

                m_axis = b2Cross(localPointA2 - localPointA1, 1.0f);
                m_axis.Normalize();
                b2Vec2 normal = b2Mul(xfA.q, m_axis);

                m_localPoint = 0.5f * (localPointA1 + localPointA2);
                b2Vec2 pointA = b2Mul(xfA, m_localPoint);

                b2Vec2 localPointB = m_proxyB.GetVertex(cache.indexB[0]);
                b2Vec2 pointB = b2Mul(xfB, localPointB);

                float s = b2Dot(pointB - pointA, normal);
                if (s < 0.0f) {
                    m_axis = -m_axis;
                    s = -s;
                }

                return s;
            }
        }

        /// <summary>The deepest points along the axis at time <paramref name="t"/> and their separation.</summary>
        /// <param name="indexA">Receives the witness vertex on A (-1 for a face of A).</param>
        /// <param name="indexB">Receives the witness vertex on B (-1 for a face of B).</param>
        /// <param name="t">The time.</param>
        /// <returns>The minimum separation.</returns>
        public readonly float FindMinSeparation(out int indexA, out int indexB, float t) {
            m_sweepA.GetTransform(out b2Transform xfA, t);
            m_sweepB.GetTransform(out b2Transform xfB, t);

            switch (m_type) {
                case b2SeparationFunctionType.e_points: {
                    b2Vec2 axisA = b2MulT(xfA.q, m_axis);
                    b2Vec2 axisB = b2MulT(xfB.q, -m_axis);

                    indexA = m_proxyA.GetSupport(axisA);
                    indexB = m_proxyB.GetSupport(axisB);

                    b2Vec2 localPointA = m_proxyA.GetVertex(indexA);
                    b2Vec2 localPointB = m_proxyB.GetVertex(indexB);

                    b2Vec2 pointA = b2Mul(xfA, localPointA);
                    b2Vec2 pointB = b2Mul(xfB, localPointB);

                    float separation = b2Dot(pointB - pointA, m_axis);
                    return separation;
                }

                case b2SeparationFunctionType.e_faceA: {
                    b2Vec2 normal = b2Mul(xfA.q, m_axis);
                    b2Vec2 pointA = b2Mul(xfA, m_localPoint);

                    b2Vec2 axisB = b2MulT(xfB.q, -normal);

                    indexA = -1;
                    indexB = m_proxyB.GetSupport(axisB);

                    b2Vec2 localPointB = m_proxyB.GetVertex(indexB);
                    b2Vec2 pointB = b2Mul(xfB, localPointB);

                    float separation = b2Dot(pointB - pointA, normal);
                    return separation;
                }

                case b2SeparationFunctionType.e_faceB: {
                    b2Vec2 normal = b2Mul(xfB.q, m_axis);
                    b2Vec2 pointB = b2Mul(xfB, m_localPoint);

                    b2Vec2 axisA = b2MulT(xfA.q, -normal);

                    indexB = -1;
                    indexA = m_proxyA.GetSupport(axisA);

                    b2Vec2 localPointA = m_proxyA.GetVertex(indexA);
                    b2Vec2 pointA = b2Mul(xfA, localPointA);

                    float separation = b2Dot(pointA - pointB, normal);
                    return separation;
                }

                default:
                    b2Assert(false);
                    indexA = -1;
                    indexB = -1;
                    return 0.0f;
            }
        }

        /// <summary>The separation of given witness vertices along the axis at time <paramref name="t"/>.</summary>
        /// <param name="indexA">The witness vertex on A.</param>
        /// <param name="indexB">The witness vertex on B.</param>
        /// <param name="t">The time.</param>
        /// <returns>The separation.</returns>
        public readonly float Evaluate(int indexA, int indexB, float t) {
            m_sweepA.GetTransform(out b2Transform xfA, t);
            m_sweepB.GetTransform(out b2Transform xfB, t);

            switch (m_type) {
                case b2SeparationFunctionType.e_points: {
                    b2Vec2 localPointA = m_proxyA.GetVertex(indexA);
                    b2Vec2 localPointB = m_proxyB.GetVertex(indexB);

                    b2Vec2 pointA = b2Mul(xfA, localPointA);
                    b2Vec2 pointB = b2Mul(xfB, localPointB);
                    float separation = b2Dot(pointB - pointA, m_axis);

                    return separation;
                }

                case b2SeparationFunctionType.e_faceA: {
                    b2Vec2 normal = b2Mul(xfA.q, m_axis);
                    b2Vec2 pointA = b2Mul(xfA, m_localPoint);

                    b2Vec2 localPointB = m_proxyB.GetVertex(indexB);
                    b2Vec2 pointB = b2Mul(xfB, localPointB);

                    float separation = b2Dot(pointB - pointA, normal);
                    return separation;
                }

                case b2SeparationFunctionType.e_faceB: {
                    b2Vec2 normal = b2Mul(xfB.q, m_axis);
                    b2Vec2 pointB = b2Mul(xfB, m_localPoint);

                    b2Vec2 localPointA = m_proxyA.GetVertex(indexA);
                    b2Vec2 pointA = b2Mul(xfA, localPointA);

                    float separation = b2Dot(pointA - pointB, normal);
                    return separation;
                }

                default:
                    b2Assert(false);
                    return 0.0f;
            }
        }
    }

    /// <summary>Box2D's continuous collision: the time of impact of two swept convex shapes.</summary>
    internal static class b2TimeOfImpactFn {
        /// <summary>
        ///     Computes the upper bound on the time before two swept shapes touch (local separating axis method: conservative
        ///     advancement along successive axes, each root found by alternating bisection and the secant rule).
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Collision/b2TimeOfImpact.cpp#L254">Box2D/Collision/b2TimeOfImpact.cpp</see>.</remarks>
        /// <param name="output">Receives the state and time.</param>
        /// <param name="input">The proxies, sweeps and interval.</param>
        /// <exception cref="NumSharp.AssertionError">Thrown when the shapes' radii leave no room for the target separation.</exception>
        public static void b2TimeOfImpact(out b2TOIOutput output, in b2TOIInput input) {
            output.state = b2TOIOutputState.e_unknown;
            output.t = input.tMax;

            ref readonly b2DistanceProxy proxyA = ref input.proxyA;
            ref readonly b2DistanceProxy proxyB = ref input.proxyB;

            b2Sweep sweepA = input.sweepA;
            b2Sweep sweepB = input.sweepB;

            // Large rotations can make the root finder fail, so we normalize the
            // sweep angles.
            sweepA.Normalize();
            sweepB.Normalize();

            float tMax = input.tMax;

            float totalRadius = proxyA.m_radius + proxyB.m_radius;
            float target = b2Max(b2_linearSlop, totalRadius - 3.0f * b2_linearSlop);
            float tolerance = 0.25f * b2_linearSlop;
            b2Assert(target > tolerance);

            float t1 = 0.0f;
            const int k_maxIterations = 20; // TODO_ERIN b2Settings
            int iter = 0;

            // Prepare input for distance query.
            b2SimplexCache cache = default;
            cache.count = 0;
            b2DistanceInput distanceInput = default;
            distanceInput.proxyA = input.proxyA;
            distanceInput.proxyB = input.proxyB;
            distanceInput.useRadii = false;

            // The outer loop progressively attempts to compute new separating axes.
            // This loop terminates when an axis is repeated (no progress is made).
            for (; ; ) {
                sweepA.GetTransform(out b2Transform xfA, t1);
                sweepB.GetTransform(out b2Transform xfB, t1);

                // Get the distance between shapes. We can also use the results
                // to get a separating axis.
                distanceInput.transformA = xfA;
                distanceInput.transformB = xfB;
                b2DistanceGJK.b2Distance(out b2DistanceOutput distanceOutput, ref cache, distanceInput);

                // If the shapes are overlapped, we give up on continuous collision.
                if (distanceOutput.distance <= 0.0f) {
                    // Failure!
                    output.state = b2TOIOutputState.e_overlapped;
                    output.t = 0.0f;
                    break;
                }

                if (distanceOutput.distance < target + tolerance) {
                    // Victory!
                    output.state = b2TOIOutputState.e_touching;
                    output.t = t1;
                    break;
                }

                // Initialize the separating axis.
                b2SeparationFunction fcn = default;
                fcn.Initialize(cache, proxyA, sweepA, proxyB, sweepB, t1);

                // Compute the TOI on the separating axis. We do this by successively
                // resolving the deepest point. This loop is bounded by the number of vertices.
                bool done = false;
                float t2 = tMax;
                int pushBackIter = 0;
                for (; ; ) {
                    // Find the deepest point at t2. Store the witness point indices.
                    float s2 = fcn.FindMinSeparation(out int indexA, out int indexB, t2);

                    // Is the final configuration separated?
                    if (s2 > target + tolerance) {
                        // Victory!
                        output.state = b2TOIOutputState.e_separated;
                        output.t = tMax;
                        done = true;
                        break;
                    }

                    // Has the separation reached tolerance?
                    if (s2 > target - tolerance) {
                        // Advance the sweeps
                        t1 = t2;
                        break;
                    }

                    // Compute the initial separation of the witness points.
                    float s1 = fcn.Evaluate(indexA, indexB, t1);

                    // Check for initial overlap. This might happen if the root finder
                    // runs out of iterations.
                    if (s1 < target - tolerance) {
                        output.state = b2TOIOutputState.e_failed;
                        output.t = t1;
                        done = true;
                        break;
                    }

                    // Check for touching
                    if (s1 <= target + tolerance) {
                        // Victory! t1 should hold the TOI (could be 0.0).
                        output.state = b2TOIOutputState.e_touching;
                        output.t = t1;
                        done = true;
                        break;
                    }

                    // Compute 1D root of: f(x) - target = 0
                    int rootIterCount = 0;
                    float a1 = t1, a2 = t2;
                    for (; ; ) {
                        // Use a mix of the secant rule and bisection.
                        float t;
                        if ((rootIterCount & 1) != 0)
                            // Secant rule to improve convergence.
                            t = a1 + (target - s1) * (a2 - a1) / (s2 - s1);
                        else
                            // Bisection to guarantee progress.
                            t = 0.5f * (a1 + a2);

                        ++rootIterCount;

                        float s = fcn.Evaluate(indexA, indexB, t);

                        if (b2Abs(s - target) < tolerance) {
                            // t2 holds a tentative value for t1
                            t2 = t;
                            break;
                        }

                        // Ensure we continue to bracket the root.
                        if (s > target) {
                            a1 = t;
                            s1 = s;
                        } else {
                            a2 = t;
                            s2 = s;
                        }

                        if (rootIterCount == 50)
                            break;
                    }

                    ++pushBackIter;

                    if (pushBackIter == b2_maxPolygonVertices)
                        break;
                }

                ++iter;

                if (done)
                    break;

                if (iter == k_maxIterations) {
                    // Root finder got stuck. Semi-victory.
                    output.state = b2TOIOutputState.e_failed;
                    output.t = t1;
                    break;
                }
            }
        }
    }
}
