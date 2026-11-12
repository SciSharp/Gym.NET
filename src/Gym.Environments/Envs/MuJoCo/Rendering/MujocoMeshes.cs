using System;
using System.Collections.Generic;
using Gym.MuJoCo;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     A display list as MuJoCo compiles it: polygons (triangles or quads) whose vertices carry a position, a normal, a
    ///     texture coordinate and, for the haze, a colour, recorded through the immediate-mode calls MuJoCo's context code
    ///     makes (<see cref="Begin"/>, <see cref="Normal"/>, <see cref="TexCoord"/>, <see cref="Color"/>, <see cref="Vertex"/>).
    /// </summary>
    /// <remarks>
    ///     The current normal, texture coordinate and colour persist from vertex to vertex until set again, as OpenGL's do,
    ///     which is what lets the translated code below read like MuJoCo's. Values are float32: <c>glVertex3d</c> and
    ///     <c>glTexCoord2d</c> hand OpenGL doubles it stores as floats, and the cast here does the same.
    /// </remarks>
    internal sealed class GlMesh {
        /// <summary>Vertex positions, 3 per vertex.</summary>
        internal readonly List<float> Positions = new();

        /// <summary>Vertex normals, 3 per vertex.</summary>
        internal readonly List<float> Normals = new();

        /// <summary>Vertex texture coordinates, 2 per vertex.</summary>
        internal readonly List<float> TexCoords = new();

        /// <summary>Vertex colours, 4 per vertex; meaningful only when <see cref="HasColors"/>.</summary>
        internal readonly List<float> Colors = new();

        /// <summary>The vertex count of each polygon, in order: 3 for a triangle, 4 for a quad.</summary>
        internal readonly List<int> PolygonSizes = new();

        /// <summary>Whether the vertices carry colours (the haze), which then replace the geom's colour.</summary>
        internal bool HasColors;

        /// <summary>The polygon size of the open <c>glBegin</c>, 0 outside one.</summary>
        private int _mode;

        /// <summary>Vertices recorded since the open polygon started.</summary>
        private int _pending;

        /// <summary>The current normal.</summary>
        private float _nx, _ny, _nz = 1;

        /// <summary>The current texture coordinate.</summary>
        private float _u, _v;

        /// <summary>The current colour.</summary>
        private float _r = 1, _g = 1, _b = 1, _a = 1;

        /// <summary>Opens a run of polygons of one kind (<c>glBegin</c>).</summary>
        /// <param name="vertices">3 for <c>GL_TRIANGLES</c>, 4 for <c>GL_QUADS</c>.</param>
        internal void Begin(int vertices) {
            _mode = vertices;
            _pending = 0;
        }

        /// <summary>Closes the run (<c>glEnd</c>); a trailing incomplete polygon is dropped, as OpenGL drops it.</summary>
        internal void End() {
            // OpenGL ignores the vertices of an unfinished primitive
            int drop = _pending;
            Positions.RemoveRange(Positions.Count - 3 * drop, 3 * drop);
            Normals.RemoveRange(Normals.Count - 3 * drop, 3 * drop);
            TexCoords.RemoveRange(TexCoords.Count - 2 * drop, 2 * drop);
            Colors.RemoveRange(Colors.Count - 4 * drop, 4 * drop);
            _mode = 0;
            _pending = 0;
        }

        /// <summary>Sets the current normal (<c>glNormal3f</c>).</summary>
        /// <param name="x">The x component.</param>
        /// <param name="y">The y component.</param>
        /// <param name="z">The z component.</param>
        internal void Normal(float x, float y, float z) {
            _nx = x;
            _ny = y;
            _nz = z;
        }

        /// <summary>Sets the current texture coordinate (<c>glTexCoord2f</c>).</summary>
        /// <param name="u">The s coordinate.</param>
        /// <param name="v">The t coordinate.</param>
        internal void TexCoord(float u, float v) {
            _u = u;
            _v = v;
        }

        /// <summary>Sets the current colour (<c>glColor4f</c>), and marks the mesh as coloured.</summary>
        /// <param name="r">Red.</param>
        /// <param name="g">Green.</param>
        /// <param name="b">Blue.</param>
        /// <param name="a">Alpha.</param>
        internal void Color(float r, float g, float b, float a) {
            _r = r;
            _g = g;
            _b = b;
            _a = a;
            HasColors = true;
        }

        /// <summary>Emits a vertex with the current attributes (<c>glVertex3f</c>); a polygon is complete every 3 or 4 calls.</summary>
        /// <param name="x">The x coordinate.</param>
        /// <param name="y">The y coordinate.</param>
        /// <param name="z">The z coordinate.</param>
        /// <exception cref="InvalidOperationException">Outside <see cref="Begin"/>/<see cref="End"/>.</exception>
        internal void Vertex(float x, float y, float z) {
            if (_mode == 0)
                throw new InvalidOperationException("glVertex outside glBegin/glEnd.");
            Positions.Add(x);
            Positions.Add(y);
            Positions.Add(z);
            Normals.Add(_nx);
            Normals.Add(_ny);
            Normals.Add(_nz);
            TexCoords.Add(_u);
            TexCoords.Add(_v);
            Colors.Add(_r);
            Colors.Add(_g);
            Colors.Add(_b);
            Colors.Add(_a);
            if (++_pending == _mode) {
                PolygonSizes.Add(_mode);
                _pending = 0;
            }
        }

        /// <summary>The vertex count.</summary>
        internal int VertexCount => Positions.Count / 3;

        /// <summary>The positions as an array, after <see cref="Seal"/>.</summary>
        internal float[] P = [];

        /// <summary>The normals as an array, after <see cref="Seal"/>.</summary>
        internal float[] N = [];

        /// <summary>The texture coordinates as an array, after <see cref="Seal"/>.</summary>
        internal float[] T = [];

        /// <summary>The colours as an array, after <see cref="Seal"/>.</summary>
        internal float[] C = [];

        /// <summary>The polygon sizes as an array, after <see cref="Seal"/>.</summary>
        internal int[] Sizes = [];

        /// <summary>Whether <see cref="Seal"/> ran since the last vertex.</summary>
        private bool _sealed;

        /// <summary>
        ///     Copies the recorded lists into arrays once (<c>glEndList</c>): drawing indexes arrays, which is several times
        ///     faster than indexing lists in the vertex loop. Recording after sealing is not supported.
        /// </summary>
        internal void Seal() {
            if (_sealed)
                return;
            P = Positions.ToArray();
            N = Normals.ToArray();
            T = TexCoords.ToArray();
            C = Colors.ToArray();
            Sizes = PolygonSizes.ToArray();
            _sealed = true;
        }
    }

    /// <summary>
    ///     MuJoCo's builtin display lists (<c>makeBuiltin</c>, <c>makePlane</c>): the unit sphere, its halves, the open and
    ///     closed unit cylinder, the box, the default plane, each plane geom's grid, and the haze's truncated cone, at the
    ///     model's <c>vis.quality</c> resolution.
    /// </summary>
    /// <remarks>
    ///     A line-for-line reading of MuJoCo's context code; angles are computed as MuJoCo computes them (the
    ///     <c>2.0f*mjPI</c> products in double, then stored as float) and the trigonometry is float32's (<c>cosf</c>,
    ///     <c>sinf</c>). Reference:
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_context.c">render_context.c</see>.
    /// </remarks>
    internal sealed class MujocoMeshes {
        /// <summary>MuJoCo's <c>mjPI</c>, a double.</summary>
        private const double Pi = Math.PI;

        /// <summary>The most grid lines of a plane (<c>mjMAXPLANEGRID</c>).</summary>
        private const int MaxPlaneGrid = 200;

        /// <summary>The unit sphere.</summary>
        internal readonly GlMesh Sphere;

        /// <summary>The upper half sphere shifted to sit on z = 1 (a capsule's top cap).</summary>
        internal readonly GlMesh SphereTop;

        /// <summary>The lower half sphere shifted to hang below z = -1 (a capsule's bottom cap).</summary>
        internal readonly GlMesh SphereBottom;

        /// <summary>The closed unit cylinder, z from -1 to 1.</summary>
        internal readonly GlMesh Cylinder;

        /// <summary>The open unit cylinder (a capsule's body).</summary>
        internal readonly GlMesh CylinderOpen;

        /// <summary>The unit box, each face split into <c>numquads</c>² quads.</summary>
        internal readonly GlMesh Box;

        /// <summary>
        ///     The unit cone (<c>mjrCONE</c>): apex at z = 1, base of radius 1 at z = 0 closed by a disk facing down; the head
        ///     of the decorations' arrows (contact forces, DR-70).
        /// </summary>
        internal readonly GlMesh Cone;

        /// <summary>The haze's truncated cone.</summary>
        internal readonly GlMesh Haze;

        /// <summary>The default plane (geoms without a grid of their own).</summary>
        internal readonly GlMesh Plane;

        /// <summary>Each plane geom's grid, in the order of the model's plane geoms (a geom's <c>dataid</c>).</summary>
        internal readonly List<GlMesh> Planes = new();

        /// <summary>Builds every builtin list for a model.</summary>
        /// <param name="m">The model: its quality settings, haze, extent and plane geoms.</param>
        internal MujocoMeshes(MjModel m) {
            int numstacks = m.vis_numstacks, numslices = m.vis_numslices, numquads = m.vis_numquads;
            Sphere = new GlMesh();
            MakeSphere(Sphere, numslices, numstacks);
            SphereTop = new GlMesh();
            HalfSphere(SphereTop, +1, numslices, numstacks / 2);
            SphereBottom = new GlMesh();
            HalfSphere(SphereBottom, -1, numslices, numstacks / 2);
            Cylinder = new GlMesh();
            Disk(Cylinder, -1, numslices, numstacks / 2);
            MakeCylinder(Cylinder, numslices, numstacks);
            Disk(Cylinder, +1, numslices, numstacks / 2);
            CylinderOpen = new GlMesh();
            MakeCylinder(CylinderOpen, numslices, numstacks);
            Haze = new GlMesh();
            MakeHaze(Haze, numslices, m.vis_haze, m.vis_rgba_haze);
            Box = MakeBox(numquads);
            Cone = new GlMesh();
            MakeCone(Cone, numslices, numstacks);
            Disk(Cone, 0, numslices, numstacks);
            Plane = MakeDefaultPlane(numquads);
            MakePlanes(m);
        }

        /// <summary>One sphere vertex and its normal (<c>setVertexSphere</c>).</summary>
        /// <param name="v">The position, written.</param>
        /// <param name="n">The normal, written.</param>
        /// <param name="az">The azimuth.</param>
        /// <param name="el">The elevation.</param>
        /// <param name="sign">0 for the sphere, +1/-1 for a half sphere shifted by one along z.</param>
        private static void SetVertexSphere(Span<float> v, Span<float> n, float az, float el, int sign) {
            v[0] = MathF.Cos(az) * MathF.Cos(el);
            v[1] = MathF.Sin(az) * MathF.Cos(el);
            v[2] = sign + MathF.Sin(el);
            n[0] = v[0];
            n[1] = v[1];
            n[2] = v[2] - sign;
        }

        /// <summary>Half a unit sphere (<c>halfSphere</c>): +1 the top, -1 the bottom.</summary>
        /// <param name="mesh">The list to record into.</param>
        /// <param name="sign">Which half.</param>
        /// <param name="nSlice">Slices around.</param>
        /// <param name="nStack">Stacks from the equator to the pole.</param>
        private static void HalfSphere(GlMesh mesh, int sign, int nSlice, int nStack) {
            Span<float> v1 = stackalloc float[3], v2 = stackalloc float[3], v3 = stackalloc float[3], v4 = stackalloc float[3];
            Span<float> n1 = stackalloc float[3], n2 = stackalloc float[3], n3 = stackalloc float[3], n4 = stackalloc float[3];

            // pole: triangles
            mesh.Begin(3);
            float el1 = (float)(Pi / 2.0f * sign * (nStack - 1) / nStack);
            for (int j = 0; j < nSlice; j++) {
                float az1 = (float)(2.0f * Pi * (j + 0.0f) / nSlice);
                float az2 = (float)(2.0f * Pi * (j + 1.0f) / nSlice);
                SetVertexSphere(v1, n1, az1, el1, sign);
                SetVertexSphere(v2, n2, az2, el1, sign);
                v3[0] = v3[1] = 0;
                v3[2] = 2 * sign;
                n3[0] = n3[1] = 0;
                n3[2] = sign;

                // the texture coordinates, in float as MuJoCo computes them (its mjPI products in double)
                float u1 = (float)(az1 / (2.0f * Pi));
                float vEl1 = (float)(nStack - 1) / nStack;
                float u2 = (float)(az2 / (2.0f * Pi));
                float u3 = (float)((az1 + az2) / 2.0f / (2.0f * Pi));
                float v3Uv = 1.0f;
                if (sign > 0) {
                    vEl1 = 1.0f - vEl1;
                    v3Uv = 1.0f - v3Uv;
                }

                if (sign > 0) {
                    Emit(mesh, u1, vEl1, n1, v1);
                    Emit(mesh, u2, vEl1, n2, v2);
                    Emit(mesh, u3, v3Uv, n3, v3);
                } else {
                    Emit(mesh, u3, v3Uv, n3, v3);
                    Emit(mesh, u2, vEl1, n2, v2);
                    Emit(mesh, u1, vEl1, n1, v1);
                }
            }

            mesh.End();

            // the rest: quads
            mesh.Begin(4);
            for (int i = 0; i < nStack - 1; i++) {
                el1 = (float)(Pi / 2.0f * sign * (i + 0) / nStack);
                float el2 = (float)(Pi / 2.0f * sign * (i + 1) / nStack);
                float vEl1 = (float)(i + 0) / nStack;
                float vEl2 = (float)(i + 1) / nStack;
                if (sign > 0) {
                    vEl1 = 1.0f - vEl1;
                    vEl2 = 1.0f - vEl2;
                }

                for (int j = 0; j < nSlice; j++) {
                    float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                    float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                    SetVertexSphere(v1, n1, az1, el1, sign);
                    SetVertexSphere(v2, n2, az2, el1, sign);
                    SetVertexSphere(v3, n3, az2, el2, sign);
                    SetVertexSphere(v4, n4, az1, el2, sign);
                    float u1 = (float)(az1 / (2.0f * Pi));
                    float u2 = (float)(az2 / (2.0f * Pi));
                    if (sign > 0) {
                        Emit(mesh, u1, vEl1, n1, v1);
                        Emit(mesh, u2, vEl1, n2, v2);
                        Emit(mesh, u2, vEl2, n3, v3);
                        Emit(mesh, u1, vEl2, n4, v4);
                    } else {
                        Emit(mesh, u1, vEl2, n4, v4);
                        Emit(mesh, u2, vEl2, n3, v3);
                        Emit(mesh, u2, vEl1, n2, v2);
                        Emit(mesh, u1, vEl1, n1, v1);
                    }
                }
            }

            mesh.End();
        }

        /// <summary>The unit sphere (<c>sphere</c>).</summary>
        /// <param name="mesh">The list to record into.</param>
        /// <param name="nSlice">Slices around.</param>
        /// <param name="nStack">Stacks from pole to pole.</param>
        private static void MakeSphere(GlMesh mesh, int nSlice, int nStack) {
            Span<float> v1 = stackalloc float[3], v2 = stackalloc float[3], v3 = stackalloc float[3], v4 = stackalloc float[3];
            Span<float> n1 = stackalloc float[3], n2 = stackalloc float[3], n3 = stackalloc float[3], n4 = stackalloc float[3];

            // poles: triangles
            mesh.Begin(3);
            for (int sign = -1; sign <= 1; sign += 2) {
                float el1 = (float)(0.5 * Pi * sign * (nStack / 2 - 1) / (nStack / 2));
                float vEl1 = 0.5f - sign * (float)(nStack / 2 - 1) / nStack;
                float v3Uv = sign > 0 ? 0.0f : 1.0f;
                for (int j = 0; j < nSlice; j++) {
                    float az1 = (float)(2.0f * Pi * (j + 0.0f) / nSlice);
                    float az2 = (float)(2.0f * Pi * (j + 1.0f) / nSlice);
                    SetVertexSphere(v1, n1, az1, el1, 0);
                    SetVertexSphere(v2, n2, az2, el1, 0);
                    v3[0] = v3[1] = 0;
                    v3[2] = sign;
                    n3[0] = n3[1] = 0;
                    n3[2] = sign;
                    float u1 = (float)(az1 / (2.0f * Pi));
                    float u2 = (float)(az2 / (2.0f * Pi));
                    float u3 = (float)((az1 + az2) / 2.0f / (2.0f * Pi));
                    if (sign > 0) {
                        Emit(mesh, u1, vEl1, n1, v1);
                        Emit(mesh, u2, vEl1, n2, v2);
                        Emit(mesh, u3, v3Uv, n3, v3);
                    } else {
                        Emit(mesh, u3, v3Uv, n3, v3);
                        Emit(mesh, u2, vEl1, n2, v2);
                        Emit(mesh, u1, vEl1, n1, v1);
                    }
                }
            }

            mesh.End();

            // the rest: quads
            mesh.Begin(4);
            for (int sign = -1; sign <= 1; sign += 2) {
                for (int i = 0; i < nStack / 2 - 1; i++) {
                    float el1 = (float)(0.5 * Pi * sign * (i + 0) / (nStack / 2));
                    float el2 = (float)(0.5 * Pi * sign * (i + 1) / (nStack / 2));
                    float vEl1 = 0.5f - sign * (float)(i + 0) / nStack;
                    float vEl2 = 0.5f - sign * (float)(i + 1) / nStack;
                    for (int j = 0; j < nSlice; j++) {
                        float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                        float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                        SetVertexSphere(v1, n1, az1, el1, 0);
                        SetVertexSphere(v2, n2, az2, el1, 0);
                        SetVertexSphere(v3, n3, az2, el2, 0);
                        SetVertexSphere(v4, n4, az1, el2, 0);
                        float u1 = (float)(az1 / (2.0f * Pi));
                        float u2 = (float)(az2 / (2.0f * Pi));
                        if (sign > 0) {
                            Emit(mesh, u1, vEl1, n1, v1);
                            Emit(mesh, u2, vEl1, n2, v2);
                            Emit(mesh, u2, vEl2, n3, v3);
                            Emit(mesh, u1, vEl2, n4, v4);
                        } else {
                            Emit(mesh, u1, vEl2, n4, v4);
                            Emit(mesh, u2, vEl2, n3, v3);
                            Emit(mesh, u2, vEl1, n2, v2);
                            Emit(mesh, u1, vEl1, n1, v1);
                        }
                    }
                }
            }

            mesh.End();
        }

        /// <summary>One cone vertex and its normal (<c>setVertexCone</c>): radius <paramref name="r"/> at height 1 - r.</summary>
        /// <param name="v">The position, written.</param>
        /// <param name="n">The normal, written (the side's, at 45°).</param>
        /// <param name="az">The azimuth.</param>
        /// <param name="r">The radius, 0 at the apex.</param>
        private static void SetVertexCone(Span<float> v, Span<float> n, float az, float r) {
            float scale = 1.0f / MathF.Sqrt(2.0f);
            v[0] = MathF.Cos(az) * r;
            v[1] = MathF.Sin(az) * r;
            v[2] = 1 - r;
            n[0] = MathF.Cos(az) * scale;
            n[1] = MathF.Sin(az) * scale;
            n[2] = scale;
        }

        /// <summary>
        ///     The open cone's side (<c>cone</c>): triangles at the apex, each with the mean of its base normals, then quads
        ///     down to the base.
        /// </summary>
        /// <param name="mesh">The list to record into.</param>
        /// <param name="nSlice">Slices around.</param>
        /// <param name="nStack">Rings from the apex down.</param>
        private static void MakeCone(GlMesh mesh, int nSlice, int nStack) {
            Span<float> v1 = stackalloc float[3], v2 = stackalloc float[3], v3 = stackalloc float[3], v4 = stackalloc float[3];
            Span<float> n1 = stackalloc float[3], n2 = stackalloc float[3], n3 = stackalloc float[3], n4 = stackalloc float[3];

            // pole: triangles
            mesh.Begin(3);
            float r1 = 1.0f / nStack;
            for (int j = 0; j < nSlice; j++) {
                float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                SetVertexCone(v1, n1, az1, r1);
                SetVertexCone(v2, n2, az2, r1);
                v3[0] = v3[1] = 0;
                v3[2] = 1;
                n3[0] = n1[0] + n2[0];
                n3[1] = n1[1] + n2[1];
                n3[2] = n1[2] + n2[2];
                NormalizeVec(n3);
                float u1 = (float)j / nSlice, u2 = (float)(j + 1) / nSlice, u3 = (u1 + u2) / 2.0f;
                mesh.TexCoord(u1, r1);
                mesh.Normal(n1[0], n1[1], n1[2]);
                mesh.Vertex(v1[0], v1[1], v1[2]);
                mesh.TexCoord(u2, r1);
                mesh.Normal(n2[0], n2[1], n2[2]);
                mesh.Vertex(v2[0], v2[1], v2[2]);
                mesh.TexCoord(u3, 0.0f);
                mesh.Normal(n3[0], n3[1], n3[2]);
                mesh.Vertex(v3[0], v3[1], v3[2]);
            }

            mesh.End();

            // the rest: quads
            mesh.Begin(4);
            for (int i = 0; i < nStack - 1; i++) {
                float ra = (i + 1) / (float)nStack, rb = (i + 2) / (float)nStack;
                for (int j = 0; j < nSlice; j++) {
                    float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                    float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                    SetVertexCone(v1, n1, az1, rb);
                    SetVertexCone(v2, n2, az2, rb);
                    SetVertexCone(v3, n3, az2, ra);
                    SetVertexCone(v4, n4, az1, ra);
                    float u1 = (float)j / nSlice, u2 = (float)(j + 1) / nSlice;
                    mesh.TexCoord(u1, rb);
                    mesh.Normal(n1[0], n1[1], n1[2]);
                    mesh.Vertex(v1[0], v1[1], v1[2]);
                    mesh.TexCoord(u2, rb);
                    mesh.Normal(n2[0], n2[1], n2[2]);
                    mesh.Vertex(v2[0], v2[1], v2[2]);
                    mesh.TexCoord(u2, ra);
                    mesh.Normal(n3[0], n3[1], n3[2]);
                    mesh.Vertex(v3[0], v3[1], v3[2]);
                    mesh.TexCoord(u1, ra);
                    mesh.Normal(n4[0], n4[1], n4[2]);
                    mesh.Vertex(v4[0], v4[1], v4[2]);
                }
            }

            mesh.End();
        }

        /// <summary>Normalizes a float vector in place (<c>mjr_normalizeVec</c>): +z when it is shorter than 1e-10.</summary>
        /// <param name="v">The vector.</param>
        private static void NormalizeVec(Span<float> v) {
            float len = MathF.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (len < 1e-10f) {
                v[0] = 0;
                v[1] = 0;
                v[2] = 1;
                return;
            }

            float scl = 1.0f / len;
            v[0] *= scl;
            v[1] *= scl;
            v[2] *= scl;
        }

        /// <summary>A disk at z = sign (<c>disk</c>): +1 the top, -1 the bottom, 0 a disk at z = 0 facing down.</summary>
        /// <param name="mesh">The list to record into.</param>
        /// <param name="sign">Which disk.</param>
        /// <param name="nSlice">Slices around.</param>
        /// <param name="nStack">Rings from the centre out.</param>
        private static void Disk(GlMesh mesh, int sign, int nSlice, int nStack) {
            Span<float> v1 = stackalloc float[3], v2 = stackalloc float[3], v3 = stackalloc float[3], v4 = stackalloc float[3];
            float nz = sign == 0 ? -1 : sign;

            // centre: triangles
            mesh.Begin(3);
            mesh.Normal(0, 0, nz);
            float r1 = 1.0f / nStack;
            for (int j = 0; j < nSlice; j++) {
                float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                SetVertexDisk(v1, az1, r1, sign);
                SetVertexDisk(v2, az2, r1, sign);
                v3[0] = v3[1] = 0;
                v3[2] = sign;
                float uA = 0.5f + 0.5f * v1[0], vA = 0.5f + 0.5f * v1[1];
                float uB = 0.5f + 0.5f * v2[0], vB = 0.5f + 0.5f * v2[1];
                if (sign > 0) {
                    mesh.TexCoord(uA, vA);
                    mesh.Vertex(v1[0], v1[1], v1[2]);
                    mesh.TexCoord(uB, vB);
                    mesh.Vertex(v2[0], v2[1], v2[2]);
                    mesh.TexCoord(0.5f, 0.5f);
                    mesh.Vertex(v3[0], v3[1], v3[2]);
                } else {
                    mesh.TexCoord(0.5f, 0.5f);
                    mesh.Vertex(v3[0], v3[1], v3[2]);
                    mesh.TexCoord(uB, vB);
                    mesh.Vertex(v2[0], v2[1], v2[2]);
                    mesh.TexCoord(uA, vA);
                    mesh.Vertex(v1[0], v1[1], v1[2]);
                }
            }

            mesh.End();

            // the rest: quads
            mesh.Begin(4);
            mesh.Normal(0, 0, nz);
            for (int i = 0; i < nStack - 1; i++) {
                float ra = (i + 1) / (float)nStack;
                float rb = (i + 2) / (float)nStack;
                for (int j = 0; j < nSlice; j++) {
                    float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                    float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                    SetVertexDisk(v1, az1, rb, sign);
                    SetVertexDisk(v2, az2, rb, sign);
                    SetVertexDisk(v3, az2, ra, sign);
                    SetVertexDisk(v4, az1, ra, sign);
                    if (sign > 0) {
                        EmitDisk(mesh, v1);
                        EmitDisk(mesh, v2);
                        EmitDisk(mesh, v3);
                        EmitDisk(mesh, v4);
                    } else {
                        EmitDisk(mesh, v4);
                        EmitDisk(mesh, v3);
                        EmitDisk(mesh, v2);
                        EmitDisk(mesh, v1);
                    }
                }
            }

            mesh.End();

            static void EmitDisk(GlMesh mesh, Span<float> v) {
                mesh.TexCoord(0.5f + 0.5f * v[0], 0.5f + 0.5f * v[1]);
                mesh.Vertex(v[0], v[1], v[2]);
            }
        }

        /// <summary>One disk vertex (<c>setVertexDisk</c>).</summary>
        /// <param name="v">The position, written.</param>
        /// <param name="az">The azimuth.</param>
        /// <param name="r">The radius.</param>
        /// <param name="sign">The disk's height.</param>
        private static void SetVertexDisk(Span<float> v, float az, float r, int sign) {
            v[0] = MathF.Cos(az) * r;
            v[1] = MathF.Sin(az) * r;
            v[2] = sign;
        }

        /// <summary>The open unit cylinder, z from -1 to 1 (<c>cylinder</c>).</summary>
        /// <param name="mesh">The list to record into.</param>
        /// <param name="nSlice">Slices around.</param>
        /// <param name="nStack">Stacks along z.</param>
        private static void MakeCylinder(GlMesh mesh, int nSlice, int nStack) {
            Span<float> v1 = stackalloc float[3], v2 = stackalloc float[3], v3 = stackalloc float[3], v4 = stackalloc float[3];
            Span<float> n1 = stackalloc float[3], n2 = stackalloc float[3], n3 = stackalloc float[3], n4 = stackalloc float[3];
            mesh.Begin(4);
            for (int i = 0; i < nStack; i++) {
                float h1 = 2 * (i + 0) / (float)nStack - 1;
                float h2 = 2 * (i + 1) / (float)nStack - 1;
                float vEl1 = (1.0f - h1) / 2.0f;
                float vEl2 = (1.0f - h2) / 2.0f;
                for (int j = 0; j < nSlice; j++) {
                    float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                    float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                    SetVertexCylinder(v1, n1, az1, h1);
                    SetVertexCylinder(v2, n2, az2, h1);
                    SetVertexCylinder(v3, n3, az2, h2);
                    SetVertexCylinder(v4, n4, az1, h2);
                    float u1 = (float)j / nSlice;
                    float u2 = (float)(j + 1) / nSlice;
                    Emit(mesh, u1, vEl1, n1, v1);
                    Emit(mesh, u2, vEl1, n2, v2);
                    Emit(mesh, u2, vEl2, n3, v3);
                    Emit(mesh, u1, vEl2, n4, v4);
                }
            }

            mesh.End();
        }

        /// <summary>One cylinder vertex and its normal (<c>setVertexCylinder</c>).</summary>
        /// <param name="v">The position, written.</param>
        /// <param name="n">The normal, written.</param>
        /// <param name="az">The azimuth.</param>
        /// <param name="h">The height.</param>
        private static void SetVertexCylinder(Span<float> v, Span<float> n, float az, float h) {
            v[0] = MathF.Cos(az);
            v[1] = MathF.Sin(az);
            v[2] = h;
            float len = MathF.Sqrt(v[0] * v[0] + v[1] * v[1]);
            n[0] = v[0] / len;
            n[1] = v[1] / len;
            n[2] = 0;
        }

        /// <summary>The haze's truncated cone (<c>haze</c>): two stacks, opaque at the rim, clear at the top and bottom.</summary>
        /// <param name="mesh">The list to record into.</param>
        /// <param name="nSlice">Slices around.</param>
        /// <param name="r">The haze proportion (<c>vis.map.haze</c>).</param>
        /// <param name="rgba">The haze colour (<c>vis.rgba.haze</c>).</param>
        private static void MakeHaze(GlMesh mesh, int nSlice, float r, float[] rgba) {
            float alpha = MathF.Atan2(1, r);
            float beta = (float)(0.75 * Pi) - alpha;
            float h = MathF.Sqrt(0.5f) * r * MathF.Sin(alpha) / MathF.Sin(beta);
            mesh.Begin(4);
            mesh.Normal(0, 0, 1);
            for (int i = 0; i < 2; i++) {
                float h1 = i == 0 ? 0 : h;
                float h2 = i == 0 ? h : 1;
                for (int j = 0; j < nSlice; j++) {
                    float az1 = (float)(2.0f * Pi * (j + 0) / nSlice);
                    float az2 = (float)(2.0f * Pi * (j + 1) / nSlice);
                    float c1 = i == 1 ? 1 : 0;
                    float c2 = i == 0 ? 1 : 0;
                    mesh.Color(rgba[0], rgba[1], rgba[2], c1);
                    HazeVertex(mesh, az1, h1, r);
                    HazeVertex(mesh, az2, h1, r);
                    mesh.Color(rgba[0], rgba[1], rgba[2], c2);
                    HazeVertex(mesh, az2, h2, r);
                    HazeVertex(mesh, az1, h2, r);
                }
            }

            mesh.End();

            static void HazeVertex(GlMesh mesh, float az, float hh, float rr) {
                mesh.Vertex(MathF.Cos(az) * (1 - rr * (1 - hh)), MathF.Sin(az) * (1 - rr * (1 - hh)), hh);
            }
        }

        /// <summary>The unit box, each face <c>numquads</c>² quads (<c>makeBuiltin</c>'s box).</summary>
        /// <param name="numquads">Quads per side.</param>
        /// <returns>The list.</returns>
        private static GlMesh MakeBox(int numquads) {
            var mesh = new GlMesh();
            float d = 2.0f / numquads;
            double nq = numquads;
            mesh.Begin(4);
            for (int x = 0; x < numquads; x++) {
                for (int y = 0; y < numquads; y++) {
                    float u0t = (float)(x / nq), u1t = (float)((x + 1) / nq), v0t = (float)(1.0 - y / nq), v1t = (float)(1.0 - (y + 1) / nq);
                    float u0b = u0t, u1b = u1t, v0b = (float)(1.0 - (y + 1) / nq), v1b = (float)(1.0 - y / nq);
                    float a0 = d * (x + 0) - 1, a1 = d * (x + 1) - 1, b0 = d * (y + 0) - 1, b1 = d * (y + 1) - 1;

                    mesh.Normal(0, 0, 1);  // top
                    Quad(mesh, u0t, v0t, a0, b0, 1, u1t, v0t, a1, b0, 1, u1t, v1t, a1, b1, 1, u0t, v1t, a0, b1, 1);
                    mesh.Normal(0, 0, -1); // bottom
                    Quad(mesh, u0b, v0b, a0, b1, -1, u1b, v0b, a1, b1, -1, u1b, v1b, a1, b0, -1, u0b, v1b, a0, b0, -1);
                    mesh.Normal(1, 0, 0);  // right
                    Quad(mesh, u0t, v0t, 1, a0, b0, u1t, v0t, 1, a1, b0, u1t, v1t, 1, a1, b1, u0t, v1t, 1, a0, b1);
                    mesh.Normal(-1, 0, 0); // left
                    Quad(mesh, u0b, v0b, -1, a0, b1, u1b, v0b, -1, a1, b1, u1b, v1b, -1, a1, b0, u0b, v1b, -1, a0, b0);
                    mesh.Normal(0, -1, 0); // front
                    Quad(mesh, u0t, v0t, a0, -1, b0, u1t, v0t, a1, -1, b0, u1t, v1t, a1, -1, b1, u0t, v1t, a0, -1, b1);
                    mesh.Normal(0, 1, 0);  // back
                    Quad(mesh, u0b, v0b, a0, 1, b1, u1b, v0b, a1, 1, b1, u1b, v1b, a1, 1, b0, u0b, v1b, a0, 1, b0);
                }
            }

            mesh.End();
            return mesh;
        }

        /// <summary>Emits one textured quad with the current normal.</summary>
        /// <param name="mesh">The list.</param>
        /// <param name="ua">First vertex: u.</param>
        /// <param name="va">First vertex: v.</param>
        /// <param name="xa">First vertex: x.</param>
        /// <param name="ya">First vertex: y.</param>
        /// <param name="za">First vertex: z.</param>
        /// <param name="ub">Second vertex: u.</param>
        /// <param name="vb">Second vertex: v.</param>
        /// <param name="xb">Second vertex: x.</param>
        /// <param name="yb">Second vertex: y.</param>
        /// <param name="zb">Second vertex: z.</param>
        /// <param name="uc">Third vertex: u.</param>
        /// <param name="vc">Third vertex: v.</param>
        /// <param name="xc">Third vertex: x.</param>
        /// <param name="yc">Third vertex: y.</param>
        /// <param name="zc">Third vertex: z.</param>
        /// <param name="ud">Fourth vertex: u.</param>
        /// <param name="vd">Fourth vertex: v.</param>
        /// <param name="xd">Fourth vertex: x.</param>
        /// <param name="yd">Fourth vertex: y.</param>
        /// <param name="zd">Fourth vertex: z.</param>
        private static void Quad(GlMesh mesh, float ua, float va, float xa, float ya, float za, float ub, float vb, float xb, float yb, float zb,
                                 float uc, float vc, float xc, float yc, float zc, float ud, float vd, float xd, float yd, float zd) {
            mesh.TexCoord(ua, va);
            mesh.Vertex(xa, ya, za);
            mesh.TexCoord(ub, vb);
            mesh.Vertex(xb, yb, zb);
            mesh.TexCoord(uc, vc);
            mesh.Vertex(xc, yc, zc);
            mesh.TexCoord(ud, vd);
            mesh.Vertex(xd, yd, zd);
        }

        /// <summary>The default plane (<c>makePlane</c>'s list 0): <c>numquads</c>² quads over [-1, 1]².</summary>
        /// <param name="numquads">Quads per side.</param>
        /// <returns>The list.</returns>
        private static GlMesh MakeDefaultPlane(int numquads) {
            var mesh = new GlMesh();
            mesh.Begin(4);
            mesh.Normal(0, 0, 1);
            double d = 2.0 / numquads, nq = numquads;
            for (int x = 0; x < numquads; x++) {
                for (int y = 0; y < numquads; y++) {
                    mesh.TexCoord((float)(x / nq), (float)(1.0 - y / nq));
                    mesh.Vertex((float)(d * (x + 0) - 1), (float)(d * (y + 0) - 1), 0);
                    mesh.TexCoord((float)((x + 1) / nq), (float)(1.0 - y / nq));
                    mesh.Vertex((float)(d * (x + 1) - 1), (float)(d * (y + 0) - 1), 0);
                    mesh.TexCoord((float)((x + 1) / nq), (float)(1.0 - (y + 1) / nq));
                    mesh.Vertex((float)(d * (x + 1) - 1), (float)(d * (y + 1) - 1), 0);
                    mesh.TexCoord((float)(x / nq), (float)(1.0 - (y + 1) / nq));
                    mesh.Vertex((float)(d * (x + 0) - 1), (float)(d * (y + 1) - 1), 0);
                }
            }

            mesh.End();
            return mesh;
        }

        /// <summary>
        ///     Each plane geom's grid (<c>makePlane</c>'s lists 1..n): a finite side split into cells of the geom's third size
        ///     with padding at both ends; an infinite side (size 0) into <see cref="MaxPlaneGrid"/> cells spanning beyond the
        ///     skybox, with texture coordinates in units of half a texture repeat.
        /// </summary>
        /// <param name="m">The model: its plane geoms, their materials and the far plane.</param>
        private void MakePlanes(MjModel m) {
            double zfar = m.vis_zfar * m.extent;
            var grid = new double[2][];
            grid[0] = new double[MaxPlaneGrid + 1];
            grid[1] = new double[MaxPlaneGrid + 1];
            var nn = new int[2];
            for (int i = 0; i < m.ngeom; i++) {
                if (m.geom_type[i] != 0)
                    continue;
                double[] sz = [m.geom_size[3 * i], m.geom_size[3 * i + 1]];
                for (int k = 0; k < 2; k++) {
                    if (sz[k] > 0) {
                        // limit the number of grid lines, leave room for padding
                        double sz2 = Math.Max(m.geom_size[3 * i + 2], sz[k] / (MaxPlaneGrid - 2));
                        nn[k] = (int)Math.Floor(sz[k] / Math.Max(1e-15, sz2));
                        nn[k] -= nn[k] % 2;
                        double pad = sz[k] - nn[k] * sz2;
                        nn[k] += 2;
                        for (int x = 0; x < nn[k]; x++) {
                            double left, right;
                            if (x == 0) {
                                left = -sz[k];
                                right = left + pad;
                            } else {
                                left = -sz[k] + pad + (x - 1) * sz2 * 2;
                                right = left + (x == nn[k] - 1 ? pad : sz2 * 2);
                            }

                            grid[k][x] = left;
                            grid[k][x + 1] = Math.Max(left, right);
                        }
                    } else {
                        int matid = m.geom_matid.Length > i ? m.geom_matid[i] : -1;
                        double sX = matid >= 0 && m.mat_texrepeat[2 * matid + k] > 0
                            ? 2 / (double)m.mat_texrepeat[2 * matid + k]
                            : 2.1 * zfar / (MaxPlaneGrid - 2);
                        double d = (2.1 * zfar + 2 * sX) / MaxPlaneGrid;
                        for (int x = 0; x <= MaxPlaneGrid; x++)
                            grid[k][x] = d * x - d * (MaxPlaneGrid / 2);
                        nn[k] = MaxPlaneGrid;
                    }
                }

                var mesh = new GlMesh();
                mesh.Begin(4);
                mesh.Normal(0, 0, 1);
                for (int x = 0; x < nn[0]; x++) {
                    for (int y = 0; y < nn[1]; y++) {
                        double u0, u1, v0, v1;
                        if (sz[0] > 0) {
                            u0 = (grid[0][x] + sz[0]) / (2.0 * sz[0]);
                            u1 = (grid[0][x + 1] + sz[0]) / (2.0 * sz[0]);
                        } else {
                            u0 = 0.5 * grid[0][x];
                            u1 = 0.5 * grid[0][x + 1];
                        }

                        if (sz[1] > 0) {
                            v0 = 1.0 - (grid[1][y] + sz[1]) / (2.0 * sz[1]);
                            v1 = 1.0 - (grid[1][y + 1] + sz[1]) / (2.0 * sz[1]);
                        } else {
                            v0 = -0.5 * grid[1][y];
                            v1 = -0.5 * grid[1][y + 1];
                        }

                        mesh.TexCoord((float)u0, (float)v0);
                        mesh.Vertex((float)grid[0][x], (float)grid[1][y], 0);
                        mesh.TexCoord((float)u1, (float)v0);
                        mesh.Vertex((float)grid[0][x + 1], (float)grid[1][y], 0);
                        mesh.TexCoord((float)u1, (float)v1);
                        mesh.Vertex((float)grid[0][x + 1], (float)grid[1][y + 1], 0);
                        mesh.TexCoord((float)u0, (float)v1);
                        mesh.Vertex((float)grid[0][x], (float)grid[1][y + 1], 0);
                    }
                }

                mesh.End();
                Planes.Add(mesh);
            }
        }

        /// <summary>Emits one vertex with its own texture coordinate and normal.</summary>
        /// <param name="mesh">The list.</param>
        /// <param name="u">The s coordinate.</param>
        /// <param name="v">The t coordinate.</param>
        /// <param name="n">The normal.</param>
        /// <param name="p">The position.</param>
        private static void Emit(GlMesh mesh, float u, float v, Span<float> n, Span<float> p) {
            mesh.TexCoord(u, v);
            mesh.Normal(n[0], n[1], n[2]);
            mesh.Vertex(p[0], p[1], p[2]);
        }
    }
}
