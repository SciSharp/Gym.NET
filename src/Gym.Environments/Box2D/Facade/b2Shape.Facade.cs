/*
* pybox2d -- http://pybox2d.googlecode.com
*
* Copyright (c) 2010 Ken Lauer / sirkne at gmail dot com
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
// Altered source version (Gym.NET, 2026-11-10): the shape extensions of pybox2d 2.3.10's Box2D/Box2D_shapes.i,
// translated to C#: the polygon's vertices and box properties (vertices set one by one, then Box2D's hull through
// Set), the edge's two-to-four vertices, the circle's radius and position.
using System;
using System.Collections.Generic;
using NumSharp;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    public abstract partial class b2Shape {
        /// <summary>The shape kind (pybox2d's <c>type</c>).</summary>
        public b2ShapeType Type => m_type;

        /// <summary>The number of child primitives (pybox2d's <c>childCount</c>; 1 for every shape here).</summary>
        public int ChildCount => GetChildCount();

        /// <summary>The radius (pybox2d's <c>radius</c>): a circle's, or the polygon and edge skin, <c>b2_polygonRadius</c>.</summary>
        public float Radius {
            get => m_radius;
            set => m_radius = value;
        }
    }

    public sealed partial class b2PolygonShape {
        /// <summary>
        ///     A polygon from pybox2d's keyword arguments: <c>polygonShape(vertices=...)</c> or <c>polygonShape(box=(hx, hy))</c>.
        ///     A failure carries pybox2d's <c>Failed on kwargs for b2PolygonShape.vertices: </c> (or <c>.box</c>) prefix.
        /// </summary>
        /// <remarks>
        ///     Python applies keyword arguments in call order; here <paramref name="vertices"/> always goes first, which only
        ///     matters when both are given and both fail.
        /// </remarks>
        /// <param name="vertices">The vertices, or null.</param>
        /// <param name="box">The half-extents of an axis-aligned box centered on the origin, applied after <paramref name="vertices"/>.</param>
        /// <exception cref="ValueError">Thrown for a vertex count outside 2..16.</exception>
        /// <exception cref="AssertionError">Thrown for fewer than 3 vertices, a degenerate edge, or no area.</exception>
        /// <exception cref="OverflowException">Thrown for a half-extent beyond the float32 range.</exception>
        public b2PolygonShape(IReadOnlyList<b2Vec2>? vertices, (double hx, double hy)? box = null) : this() {
            if (vertices != null)
                Box2DConvert.Kwarg("b2PolygonShape", "vertices", () => Vertices = vertices);
            if (box.HasValue)
                Box2DConvert.Kwarg("b2PolygonShape", "box", () => Box = box.Value);
        }

        /// <summary>
        ///     The vertices (pybox2d's <c>vertices</c>), counter-clockwise from the hull's right-most lowest point. Setting them
        ///     stores each one, then builds the convex hull, normals and centroid (Box2D's <c>Set</c>); an empty list clears them.
        /// </summary>
        /// <exception cref="ValueError">Thrown by the setter for a count outside 2..16, with pybox2d's message.</exception>
        /// <exception cref="AssertionError">Thrown by the setter for fewer than 3 vertices, a degenerate edge, or no area.</exception>
        public IReadOnlyList<b2Vec2> Vertices {
            get {
                var list = new b2Vec2[m_count];
                Array.Copy(m_vertices, list, m_count);
                return list;
            }
            set {
                if (value == null || value.Count == 0) {
                    // pybox2d's __clear_vertices.
                    m_count = 0;
                    for (int i = 0; i < b2_maxPolygonVertices; ++i)
                        m_vertices[i].Set(0, 0);
                    return;
                }

                if (value.Count < 2 || value.Count > b2_maxPolygonVertices)
                    throw new ValueError($"Expected tuple or list of length >= 2 and less than b2_maxPolygonVertices={b2_maxPolygonVertices}, got length {value.Count}.");

                for (int i = 0; i < value.Count; ++i) {
                    m_vertices[i].Set(value[i].x, value[i].y);
                    // pybox2d follows along so an exception leaves the count of vertices set so far.
                    m_count = i + 1;
                }

                Set(m_vertices, m_count);
            }
        }

        /// <summary>The edge normals (pybox2d's <c>normals</c>).</summary>
        public IReadOnlyList<b2Vec2> Normals {
            get {
                var list = new b2Vec2[m_count];
                Array.Copy(m_normals, list, m_count);
                return list;
            }
        }

        /// <summary>The number of vertices (pybox2d's <c>vertexCount</c>).</summary>
        public int VertexCount => m_count;

        /// <summary>The centroid (pybox2d's <c>centroid</c>).</summary>
        public b2Vec2 Centroid => m_centroid;

        /// <summary>Makes the polygon an axis-aligned box of the given half-extents (pybox2d's write-only <c>box</c>).</summary>
        /// <exception cref="OverflowException">Thrown for a half-extent beyond the float32 range.</exception>
        public (double hx, double hy) Box {
            set => SetAsBox(Box2DConvert.ToFloat32(value.hx, "b2PolygonShape_SetAsBox", 2), Box2DConvert.ToFloat32(value.hy, "b2PolygonShape_SetAsBox", 3));
        }

        /// <summary>Makes the polygon an oriented box (pybox2d's <c>box = (hx, hy, center, angle)</c>).</summary>
        /// <param name="hx">The half-width.</param>
        /// <param name="hy">The half-height.</param>
        /// <param name="center">The box's center in local coordinates.</param>
        /// <param name="angle">The box's rotation in radians.</param>
        /// <exception cref="OverflowException">Thrown for a scalar beyond the float32 range.</exception>
        public void SetAsBox(double hx, double hy, b2Vec2 center, double angle) =>
            SetAsBox(Box2DConvert.ToFloat32(hx, "b2PolygonShape_SetAsBox", 2), Box2DConvert.ToFloat32(hy, "b2PolygonShape_SetAsBox", 3), center,
                     Box2DConvert.ToFloat32(angle, "b2PolygonShape_SetAsBox", 5));
    }

    public sealed partial class b2EdgeShape {
        /// <summary>
        ///     An edge from pybox2d's keyword arguments, <c>edgeShape(vertices=[v1, v2])</c>; a failure carries pybox2d's
        ///     <c>Failed on kwargs for b2EdgeShape.vertices: </c> prefix.
        /// </summary>
        /// <param name="vertices">Two to four vertices, as <see cref="Vertices"/> takes them.</param>
        /// <exception cref="ValueError">Thrown for other than 2 to 4 vertices.</exception>
        public b2EdgeShape(IReadOnlyList<b2Vec2?> vertices) : this() {
            Box2DConvert.Kwarg("b2EdgeShape", "vertices", () => Vertices = vertices);
        }

        /// <summary>
        ///     The edge's two vertices (pybox2d's <c>vertices</c>). Setting 2 vertices drops the ghosts; 3 set
        ///     <c>vertex0</c> (a null one leaves no ghost) and 4 set both ghosts, as pybox2d does.
        /// </summary>
        /// <exception cref="ValueError">Thrown by the setter for other than 2 to 4 vertices: <c>Expected from 2 to 4 vertices.</c></exception>
        public IReadOnlyList<b2Vec2?> Vertices {
            get => new b2Vec2?[] { m_vertex1, m_vertex2 };
            set {
                // A None vertex converts to (0, 0) through pybox2d's typemap.
                static b2Vec2 Or0(b2Vec2? v) => v ?? new b2Vec2(0.0f, 0.0f);
                switch (value.Count) {
                    case 2:
                        m_vertex1 = Or0(value[0]);
                        m_vertex2 = Or0(value[1]);
                        m_hasVertex0 = false;
                        m_hasVertex3 = false;
                        break;
                    case 3:
                        m_vertex0 = Or0(value[0]);
                        m_vertex1 = Or0(value[1]);
                        m_vertex2 = Or0(value[2]);
                        m_hasVertex0 = value[0] != null;
                        m_hasVertex3 = false;
                        break;
                    case 4:
                        m_vertex0 = Or0(value[0]);
                        m_vertex1 = Or0(value[1]);
                        m_vertex2 = Or0(value[2]);
                        m_vertex3 = Or0(value[3]);
                        m_hasVertex0 = value[0] != null;
                        m_hasVertex3 = true;
                        break;
                    default:
                        throw new ValueError("Expected from 2 to 4 vertices.");
                }
            }
        }

        /// <summary>The ghost vertex before the edge (pybox2d's <c>vertex0</c>).</summary>
        public b2Vec2 Vertex0 {
            get => m_vertex0;
            set => m_vertex0 = value;
        }

        /// <summary>The first vertex (pybox2d's <c>vertex1</c>).</summary>
        public b2Vec2 Vertex1 {
            get => m_vertex1;
            set => m_vertex1 = value;
        }

        /// <summary>The second vertex (pybox2d's <c>vertex2</c>).</summary>
        public b2Vec2 Vertex2 {
            get => m_vertex2;
            set => m_vertex2 = value;
        }

        /// <summary>The ghost vertex after the edge (pybox2d's <c>vertex3</c>).</summary>
        public b2Vec2 Vertex3 {
            get => m_vertex3;
            set => m_vertex3 = value;
        }

        /// <summary>Whether <see cref="Vertex0"/> is set (pybox2d's <c>hasVertex0</c>).</summary>
        public bool HasVertex0 {
            get => m_hasVertex0;
            set => m_hasVertex0 = value;
        }

        /// <summary>Whether <see cref="Vertex3"/> is set (pybox2d's <c>hasVertex3</c>).</summary>
        public bool HasVertex3 {
            get => m_hasVertex3;
            set => m_hasVertex3 = value;
        }
    }

    public sealed partial class b2CircleShape {
        /// <summary>
        ///     A circle from pybox2d's keyword arguments, <c>circleShape(radius=..., pos=...)</c>; a failure carries pybox2d's
        ///     <c>Failed on kwargs for b2CircleShape.radius: </c> prefix.
        /// </summary>
        /// <param name="radius">The radius (stored as float32).</param>
        /// <param name="pos">The center in the body's frame (origin when null).</param>
        /// <exception cref="OverflowException">Thrown for a radius beyond the float32 range.</exception>
        public b2CircleShape(double radius, b2Vec2? pos = null) : this() {
            Box2DConvert.Kwarg("b2CircleShape", "radius", () => m_radius = Box2DConvert.ToFloat32(radius, "b2Shape_radius_set", 2));
            if (pos.HasValue)
                m_p = pos.Value;
        }

        /// <summary>The center in the body's frame (pybox2d's <c>pos</c>).</summary>
        public b2Vec2 Pos {
            get => m_p;
            set => m_p = value;
        }
    }
}
