using System;
using System.Collections.Generic;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>A framebuffer the context draws into: per-sample RGBA8 colour, float depth and an 8-bit stencil.</summary>
    /// <remarks>
    ///     Samples are stored together per pixel, pixel rows from the bottom (OpenGL's window coordinates), so sample
    ///     <c>s</c> of pixel (<c>x</c>, <c>y</c>) is at <c>(y * Width + x) * Samples + s</c>.
    /// </remarks>
    internal sealed class GlTarget {
        /// <summary>The width in pixels.</summary>
        internal readonly int Width;

        /// <summary>The height in pixels.</summary>
        internal readonly int Height;

        /// <summary>Samples per pixel: 1, or 4 for MuJoCo's multisampled offscreen buffer.</summary>
        internal readonly int Samples;

        /// <summary>RGBA per sample, or <see langword="null"/> for a depth-only target (the shadow map).</summary>
        internal readonly byte[]? Color;

        /// <summary>Depth per sample, reverse-Z in [0, 1] (1 at the near plane), as MuJoCo's clip control sets it.</summary>
        internal readonly float[] Depth;

        /// <summary>Stencil per sample, or <see langword="null"/>.</summary>
        internal readonly byte[]? Stencil;

        /// <summary>Whether depths are stored as 24-bit fixed point (the shadow map's <c>GL_DEPTH24_STENCIL8</c>).</summary>
        internal readonly bool Depth24;

        /// <summary>Allocates a target.</summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="samples">Samples per pixel.</param>
        /// <param name="color">Whether it has a colour buffer.</param>
        /// <param name="stencil">Whether it has a stencil buffer.</param>
        /// <param name="depth24">Whether depths are quantized to 24 bits.</param>
        internal GlTarget(int width, int height, int samples, bool color, bool stencil, bool depth24 = false) {
            Width = width;
            Height = height;
            Samples = samples;
            Color = color ? new byte[width * height * samples * 4] : null;
            Depth = new float[width * height * samples];
            Stencil = stencil ? new byte[width * height * samples] : null;
            Depth24 = depth24;
        }
    }

    /// <summary>The blend factors MuJoCo uses.</summary>
    internal enum GlBlend {
        /// <summary><c>GL_ONE</c>.</summary>
        One,

        /// <summary><c>GL_SRC_ALPHA</c>.</summary>
        SrcAlpha,

        /// <summary><c>GL_ONE_MINUS_SRC_ALPHA</c>.</summary>
        OneMinusSrcAlpha,
    }

    /// <summary>One fixed-function light (<c>GL_LIGHTi</c>), its position and direction in eye coordinates.</summary>
    internal sealed class GlLight {
        /// <summary>Whether it is enabled.</summary>
        internal bool Enabled;

        /// <summary>Ambient colour.</summary>
        internal float Ar, Ag, Ab;

        /// <summary>Diffuse colour.</summary>
        internal float Dr, Dg, Db;

        /// <summary>Specular colour.</summary>
        internal float Sr, Sg, Sb;

        /// <summary>Position in eye coordinates; w = 0 for a directional light (its direction toward the light).</summary>
        internal float Px, Py, Pz, Pw;

        /// <summary>Spot direction in eye coordinates.</summary>
        internal float SDx, SDy, SDz = -1;

        /// <summary>Spot exponent.</summary>
        internal float SpotExponent;

        /// <summary>Spot cutoff in degrees; 180 for no cone.</summary>
        internal float SpotCutoff = 180;

        /// <summary>Attenuation: constant, linear, quadratic.</summary>
        internal float K0 = 1, K1, K2;
    }

    /// <summary>
    ///     A software OpenGL for exactly the fixed-function state MuJoCo's classic renderer touches (<c>mjr_render</c>):
    ///     matrix stacks; per-vertex lighting (<c>GL_LIGHT_MODEL_LOCAL_VIEWER</c>, colour tracking ambient and diffuse,
    ///     <c>GL_NORMALIZE</c>); object-linear texture generation and the texture matrix on unit 0; eye-linear generation of
    ///     the shadow map's coordinates on unit 1 with its depth compare; near, far and user-plane clipping; back-face
    ///     culling; multisampled rasterization with reverse-Z depth (<c>GL_GEQUAL</c>), the stencil, blending, the alpha
    ///     test and polygon offset.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Arithmetic is float32, as the GPU's. What the GPU leaves to the driver is chosen the common way where it could
    ///     not be measured: vertex positions snapped to 1/256 of a pixel, depth interpolated per
    ///     sample. Each is a place where a frame can differ from the GPU's by a level or a pixel at an edge, which the frame
    ///     cells measure until it is closed (DR-71, DR-98).
    ///     </para>
    ///     <para>
    ///     Where the reference GPU (an NVIDIA RTX 4060 Ti, driver 616.56, in MuJoCo's own GL context) was measured, its
    ///     behaviour is reproduced instead: quads split into (0, 1, 2) and (0, 2, 3); the standard 4-sample positions; a
    ///     fragment shaded once per pixel, its colour at the pixel's centre or, on an edge pixel, at its first covered
    ///     sample, its texture coordinates at the centre; one level of detail per 2×2 pixel quad, from the differences at
    ///     the quad's lower-left pixel, with each derivative's length a fixed-point <c>max + 0.3442·min</c> (a fit) and
    ///     the level read from the GPU's own logarithm table in 1/256 steps; bilinear weights rounded to 1/256; the 12-bit
    ///     route a colour takes into the 8-bit buffer (<see cref="ToUnorm8"/>), integer blending, the resolve
    ///     (<see cref="MujocoFrame.ResolveFour"/>), the driver's matrix stack and the vertex transform
    ///     (<see cref="GlMath.MultiplyDriver"/>, <see cref="GlMath.TransformGpu"/>), the viewport stage's reciprocal
    ///     (<see cref="GlReciprocal"/>) and depths truncated where a cast rounds. <c>tools/parity/mujoco/gpu_probe.py</c>
    ///     re-measures each on the machine it runs on and scores these models against it.
    ///     </para>
    ///     <para>
    ///     Reference: the OpenGL 2.1 specification (lighting, §2.14; texture generation, §2.12.4; clipping, §2.17; polygon
    ///     offset, §3.5.5) as MuJoCo's <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_gl3.c">render_gl3.c</see> drives it.
    ///     </para>
    /// </remarks>
    internal sealed class GlContext {
        /// <summary>The standard 4-sample positions in a pixel, OpenGL's y-up convention.</summary>
        private static readonly float[] Samples4 = [0.375f, 0.125f, 0.875f, 0.375f, 0.125f, 0.625f, 0.625f, 0.875f];

        /// <summary>The single-sample position: the pixel's centre.</summary>
        private static readonly float[] Samples1 = [0.5f, 0.5f];

        /// <summary>The target drawn into.</summary>
        private GlTarget _target = null!;

        /// <summary>The viewport: left, bottom, width, height.</summary>
        private int _vx, _vy, _vw, _vh;

        /// <summary>The projection matrix, column-major.</summary>
        internal float[] Projection = GlMath.Identity();

        /// <summary>The modelview matrix, column-major.</summary>
        internal float[] Modelview = GlMath.Identity();

        /// <summary>Unit 0's texture matrix, column-major.</summary>
        internal float[] TextureMatrix = GlMath.Identity();

        /// <summary>Saved modelview matrices (<c>glPushMatrix</c>).</summary>
        private readonly Stack<float[]> _stack = new();

        /// <summary>The eight lights.</summary>
        internal readonly GlLight[] Lights = { new(), new(), new(), new(), new(), new(), new(), new() };

        /// <summary>Whether lighting is enabled (<c>GL_LIGHTING</c>).</summary>
        internal bool Lighting = true;

        /// <summary>The global ambient light (<c>GL_LIGHT_MODEL_AMBIENT</c>).</summary>
        internal float GlobalAmbient;

        /// <summary>The material's emission (<c>GL_EMISSION</c>).</summary>
        internal float Er, Eg, Eb;

        /// <summary>The material's specular intensity (<c>GL_SPECULAR</c>, equal channels).</summary>
        internal float MatSpecular = 1;

        /// <summary>The material's specular exponent (<c>GL_SHININESS</c>).</summary>
        internal float Shininess;

        /// <summary>Whether colour tracks ambient and diffuse (<c>GL_COLOR_MATERIAL</c>).</summary>
        internal bool ColorMaterial = true;

        /// <summary>The current colour (<c>glColor4f</c>).</summary>
        internal float Cr = 1, Cg = 1, Cb = 1, Ca = 1;

        /// <summary>Unit 0's texture, or <see langword="null"/> when texturing is off.</summary>
        internal GlTexture? Texture;

        /// <summary>Whether unit 0 generates its coordinates from object-linear planes (<c>GL_TEXTURE_GEN_*</c>).</summary>
        internal bool TexGen;

        /// <summary>Unit 0's object planes for s, t and r, 4 floats each.</summary>
        internal float[] TexPlanes = new float[12];

        /// <summary>The shadow map unit 1 compares against, or <see langword="null"/> when it is off.</summary>
        internal GlTarget? ShadowMap;

        /// <summary>Unit 1's eye planes for s, t, r and q, already in eye coordinates.</summary>
        private readonly float[] _shadowPlanes = new float[16];

        /// <summary>Whether back faces (or front faces, <see cref="CullFront"/>) are discarded.</summary>
        internal bool CullFace = true;

        /// <summary>Discard front faces instead of back faces (<c>glCullFace(GL_FRONT)</c>).</summary>
        internal bool CullFront;

        /// <summary>Whether counter-clockwise polygons are front-facing (<c>glFrontFace(GL_CCW)</c>).</summary>
        internal bool FrontCcw = true;

        /// <summary>Whether the depth test (<c>GL_GEQUAL</c>) runs; off, depth is neither tested nor written.</summary>
        internal bool DepthTest = true;

        /// <summary>Whether depth is written (<c>glDepthMask</c>).</summary>
        internal bool DepthMask = true;

        /// <summary>Whether colour is written (<c>glColorMask</c>).</summary>
        internal bool ColorMask = true;

        /// <summary>Whether the stencil test runs.</summary>
        internal bool StencilTest;

        /// <summary>The stencil test: <see langword="true"/> for <c>GL_EQUAL</c> to <see cref="StencilRef"/>, else <c>GL_ALWAYS</c>.</summary>
        internal bool StencilEqual;

        /// <summary>The stencil reference.</summary>
        internal byte StencilRef;

        /// <summary>Whether every stencil outcome replaces the value with the reference (<c>GL_REPLACE</c>), else keeps it.</summary>
        internal bool StencilReplace;

        /// <summary>Whether blending is on.</summary>
        internal bool Blend;

        /// <summary>The source blend factor.</summary>
        internal GlBlend BlendSrc = GlBlend.One;

        /// <summary>The destination blend factor.</summary>
        internal GlBlend BlendDst = GlBlend.One;

        /// <summary>Whether the alpha test (<c>GL_GEQUAL</c> 0.99, MuJoCo's) runs.</summary>
        internal bool AlphaTest;

        /// <summary>Whether user clip plane 0 is on.</summary>
        internal bool ClipPlane;

        /// <summary>The clip plane in eye coordinates.</summary>
        private readonly float[] _clipPlane = new float[4];

        /// <summary>Whether polygon offset applies to filled polygons.</summary>
        internal bool PolygonOffset;

        /// <summary>The polygon offset's slope factor and units.</summary>
        internal float OffsetFactor, OffsetUnits;

        /// <summary>Whether linear fog is on.</summary>
        internal bool Fog;

        /// <summary>Fog's start and end distance.</summary>
        internal float FogStart, FogEnd;

        /// <summary>Fog's colour.</summary>
        internal float FogR, FogG, FogB;

        /// <summary>Processed vertices of the mesh being drawn.</summary>
        private Vtx[] _vertices = new Vtx[1024];

        /// <summary>A processed vertex: clip coordinates and every attribute the rasterizer interpolates.</summary>
        private struct Vtx {
            /// <summary>Clip coordinates.</summary>
            public float X, Y, Z, W;

            /// <summary>The colour (lit, clamped).</summary>
            public float R, G, B, A;

            /// <summary>Unit 0's texture coordinates.</summary>
            public float S, T, U, Q;

            /// <summary>Unit 1's texture coordinates (the shadow map).</summary>
            public float S1, T1, U1, Q1;

            /// <summary>The user clip plane's distance.</summary>
            public float Clip;

            /// <summary>The eye-space distance for fog.</summary>
            public float FogZ;
        }

        /// <summary>Binds a target and sets the viewport to all of it.</summary>
        /// <param name="target">The target.</param>
        internal void Bind(GlTarget target) {
            _target = target;
            Viewport(0, 0, target.Width, target.Height);
        }

        /// <summary>Sets the viewport (<c>glViewport</c>).</summary>
        /// <param name="x">Left.</param>
        /// <param name="y">Bottom.</param>
        /// <param name="width">Width.</param>
        /// <param name="height">Height.</param>
        internal void Viewport(int x, int y, int width, int height) {
            _vx = x;
            _vy = y;
            _vw = width;
            _vh = height;
        }

        /// <summary>Clears the bound target (<c>glClear</c>): colour to 0, depth to 0 (reverse-Z's far), stencil to 0.</summary>
        /// <param name="color">Clear colour.</param>
        /// <param name="depth">Clear depth.</param>
        /// <param name="stencil">Clear stencil.</param>
        internal void Clear(bool color, bool depth, bool stencil) {
            if (color && _target.Color != null)
                Array.Clear(_target.Color);
            if (depth)
                Array.Clear(_target.Depth);
            if (stencil && _target.Stencil != null)
                Array.Clear(_target.Stencil);
        }

        /// <summary>Saves the modelview matrix (<c>glPushMatrix</c>).</summary>
        internal void PushMatrix() => _stack.Push((float[])Modelview.Clone());

        /// <summary>Restores the modelview matrix (<c>glPopMatrix</c>).</summary>
        internal void PopMatrix() => Modelview = _stack.Pop();

        /// <summary>
        ///     Right-multiplies the modelview matrix (<c>glMultMatrixf</c>, and <c>glTranslatef</c>, <c>glScalef</c> and
        ///     <c>glRotatef</c> given as their matrices) with the reference driver's pairwise sums (<see cref="GlMath.MultiplyDriver"/>).
        /// </summary>
        /// <param name="m">The matrix, column-major.</param>
        internal void MultModelview(float[] m) => Modelview = GlMath.MultiplyDriver(Modelview, m);

        /// <summary>Sets light <paramref name="i"/>'s position (<c>glLightfv(GL_POSITION)</c>), transformed by the current modelview.</summary>
        /// <param name="i">The light.</param>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="z">z.</param>
        /// <param name="w">w: 0 for a direction.</param>
        internal void SetLightPosition(int i, float x, float y, float z, float w) {
            GlMath.Transform(Modelview, x, y, z, w, out float ex, out float ey, out float ez, out float ew);
            GlLight l = Lights[i];
            l.Px = ex;
            l.Py = ey;
            l.Pz = ez;
            l.Pw = ew;
        }

        /// <summary>Sets light <paramref name="i"/>'s spot direction, transformed by the modelview's upper 3×3.</summary>
        /// <param name="i">The light.</param>
        /// <param name="x">x.</param>
        /// <param name="y">y.</param>
        /// <param name="z">z.</param>
        internal void SetSpotDirection(int i, float x, float y, float z) {
            GlMath.Transform(Modelview, x, y, z, 0, out float ex, out float ey, out float ez, out _);
            GlLight l = Lights[i];
            l.SDx = ex;
            l.SDy = ey;
            l.SDz = ez;
        }

        /// <summary>
        ///     Sets unit 1's eye planes (<c>glTexGenfv(GL_EYE_PLANE)</c>): each plane is multiplied by the inverse of the
        ///     current modelview, as OpenGL stores eye planes.
        /// </summary>
        /// <param name="rows">The four planes (the texture matrix's rows), 16 floats.</param>
        internal void SetShadowPlanes(float[] rows) {
            float[] inv = GlMath.Invert(Modelview);
            for (int p = 0; p < 4; p++)
                GlMath.PlaneTimes(rows, 4 * p, inv, _shadowPlanes, 4 * p);
        }

        /// <summary>Sets user clip plane 0 (<c>glClipPlane</c>), multiplied by the inverse of the current modelview.</summary>
        /// <param name="plane">The plane in object coordinates, as doubles.</param>
        internal void SetClipPlane(double[] plane) {
            float[] inv = GlMath.Invert(Modelview);
            float[] p = [(float)plane[0], (float)plane[1], (float)plane[2], (float)plane[3]];
            GlMath.PlaneTimes(p, 0, inv, _clipPlane, 0);
        }

        /// <summary>Draws a mesh with the current state and modelview (<c>glCallList</c>).</summary>
        /// <param name="mesh">The mesh.</param>
        internal void DrawMesh(GlMesh mesh) {
            int n = mesh.VertexCount;
            if (n == 0)
                return;
            mesh.Seal();
            if (_vertices.Length < n) {
                _vertices = new Vtx[Math.Max(n, 2 * _vertices.Length)];
                _outcodes = new int[_vertices.Length];
                _needed = new bool[_vertices.Length];
            }

            // phase 1: clip coordinates and outcodes of every vertex
            // the GPU's own order: the projection times the modelview on the driver's side, then fused per vertex
            float[] mv = Modelview, mvp = GlMath.MultiplyDriver(Projection, mv);
            float[] pos = mesh.P;
            for (int v = 0; v < n; v++) {
                ref Vtx o = ref _vertices[v];
                GlMath.TransformGpu(mvp, pos[3 * v], pos[3 * v + 1], pos[3 * v + 2], 1, out o.X, out o.Y, out o.Z, out o.W);
                int code = 0;
                if (o.X < -o.W)
                    code |= 1;
                if (o.X > o.W)
                    code |= 2;
                if (o.Y < -o.W)
                    code |= 4;
                if (o.Y > o.W)
                    code |= 8;
                if (o.Z < 0)
                    code |= 16;
                if (o.Z > o.W)
                    code |= 32;
                if (ClipPlane) {
                    GlMath.TransformGpu(mv, pos[3 * v], pos[3 * v + 1], pos[3 * v + 2], 1, out float ex, out float ey, out float ez, out float ew);
                    o.Clip = _clipPlane[0] * ex + _clipPlane[1] * ey + _clipPlane[2] * ez + _clipPlane[3] * ew;
                    if (o.Clip < 0)
                        code |= 64;
                }

                _outcodes[v] = code;
                _needed[v] = false;
            }

            // the diagnostic's record of every triangle as drawn, before the rejection below
            if (DebugClipTriangles != null) {
                int first = 0;
                foreach (int size in mesh.Sizes) {
                    RecordClip(first, first + 1, first + 2);
                    if (size == 4)
                        RecordClip(first, first + 2, first + 3);
                    first += size;
                }
            }

            // phase 2: a triangle wholly outside one half-space of the clip volume yields no pixel (for any w); only
            // the survivors' vertices are lit, which spares the far reaches of an infinite plane
            _triangles.Clear();
            int start = 0;
            foreach (int size in mesh.Sizes) {
                // a quad is the triangles (0, 1, 2) and (0, 2, 3)
                Keep(start, start + 1, start + 2);
                if (size == 4)
                    Keep(start, start + 2, start + 3);
                start += size;
            }

            if (_triangles.Count == 0)
                return;

            // phase 3: lighting and texture coordinates of the vertices in use
            ProcessVertices(mesh, n);

            // phase 4: clipping and rasterization, in the mesh's order
            Span<int> idx = stackalloc int[3];
            for (int i = 0; i < _triangles.Count; i += 3) {
                idx[0] = _triangles[i];
                idx[1] = _triangles[i + 1];
                idx[2] = _triangles[i + 2];
                Triangle(idx);
            }

            void RecordClip(int a, int b, int c) {
                foreach (int k in (ReadOnlySpan<int>)[a, b, c]) {
                    ref Vtx o = ref _vertices[k];
                    DebugClipTriangles.Add(o.X);
                    DebugClipTriangles.Add(o.Y);
                    DebugClipTriangles.Add(o.Z);
                    DebugClipTriangles.Add(o.W);
                }
            }

            void Keep(int a, int b, int c) {
                if ((_outcodes[a] & _outcodes[b] & _outcodes[c]) != 0)
                    return;
                _triangles.Add(a);
                _triangles.Add(b);
                _triangles.Add(c);
                _needed[a] = _needed[b] = _needed[c] = true;
            }
        }

        /// <summary>Each vertex's clip-volume outcode, from <see cref="DrawMesh"/>'s first phase.</summary>
        private int[] _outcodes = new int[1024];

        /// <summary>Whether each vertex belongs to a triangle that survived trivial rejection.</summary>
        private bool[] _needed = new bool[1024];

        /// <summary>The surviving triangles' vertex indices, three per triangle.</summary>
        private readonly List<int> _triangles = new();

        /// <summary>Lights and texture-maps the vertices of the surviving triangles (clip coordinates are already set).</summary>
        /// <param name="mesh">The mesh.</param>
        /// <param name="n">Its vertex count.</param>
        private void ProcessVertices(GlMesh mesh, int n) {
            float[] mv = Modelview;
            float[] nm = GlMath.NormalMatrix(mv);
            float[] pos = mesh.P, nrm = mesh.N, tex = mesh.T, col = mesh.C;
            for (int v = 0; v < n; v++) {
                if (!_needed[v])
                    continue;
                float x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
                ref Vtx o = ref _vertices[v];
                GlMath.TransformGpu(mv, x, y, z, 1, out float ex, out float ey, out float ez, out float ew);

                // colour: the vertex's own (the haze), else the current one; lit when lighting is on
                float cr = Cr, cg = Cg, cb = Cb, ca = Ca;
                if (mesh.HasColors) {
                    cr = col[4 * v];
                    cg = col[4 * v + 1];
                    cb = col[4 * v + 2];
                    ca = col[4 * v + 3];
                }

                if (Lighting) {
                    float nx = nm[0] * nrm[3 * v] + nm[3] * nrm[3 * v + 1] + nm[6] * nrm[3 * v + 2];
                    float ny = nm[1] * nrm[3 * v] + nm[4] * nrm[3 * v + 1] + nm[7] * nrm[3 * v + 2];
                    float nz = nm[2] * nrm[3 * v] + nm[5] * nrm[3 * v + 1] + nm[8] * nrm[3 * v + 2];
                    float len = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
                    if (len > 0) {
                        nx /= len;
                        ny /= len;
                        nz /= len;
                    }

                    Light(ex / ew, ey / ew, ez / ew, nx, ny, nz, cr, cg, cb, ca, out o.R, out o.G, out o.B, out o.A);
                } else {
                    o.R = cr;
                    o.G = cg;
                    o.B = cb;
                    o.A = ca;
                }

                // unit 0: object-linear planes or the mesh's coordinates, then the texture matrix
                float s, t, u;
                if (TexGen) {
                    s = TexPlanes[0] * x + TexPlanes[1] * y + TexPlanes[2] * z + TexPlanes[3];
                    t = TexPlanes[4] * x + TexPlanes[5] * y + TexPlanes[6] * z + TexPlanes[7];
                    u = TexPlanes[8] * x + TexPlanes[9] * y + TexPlanes[10] * z + TexPlanes[11];
                } else {
                    s = tex[2 * v];
                    t = tex[2 * v + 1];
                    u = 0;
                }

                GlMath.TransformGpu(TextureMatrix, s, t, u, 1, out o.S, out o.T, out o.U, out o.Q);

                // unit 1: eye-linear planes
                if (ShadowMap != null) {
                    float[] p = _shadowPlanes;
                    o.S1 = p[0] * ex + p[1] * ey + p[2] * ez + p[3] * ew;
                    o.T1 = p[4] * ex + p[5] * ey + p[6] * ez + p[7] * ew;
                    o.U1 = p[8] * ex + p[9] * ey + p[10] * ez + p[11] * ew;
                    o.Q1 = p[12] * ex + p[13] * ey + p[14] * ez + p[15] * ew;
                }

                if (!ClipPlane)
                    o.Clip = 1;
                o.FogZ = MathF.Abs(ez / ew);
            }
        }

        /// <summary>
        ///     OpenGL's lighting equation for one vertex (GL 2.1 §2.14.1) with a local viewer, colour tracking ambient and
        ///     diffuse, front materials only, each channel clamped to [0, 1].
        /// </summary>
        /// <param name="px">The eye-space position's x.</param>
        /// <param name="py">y.</param>
        /// <param name="pz">z.</param>
        /// <param name="nx">The unit eye-space normal's x.</param>
        /// <param name="ny">y.</param>
        /// <param name="nz">z.</param>
        /// <param name="cr">The colour (ambient and diffuse material): red.</param>
        /// <param name="cg">Green.</param>
        /// <param name="cb">Blue.</param>
        /// <param name="ca">Alpha.</param>
        /// <param name="r">The lit red, written.</param>
        /// <param name="g">Green, written.</param>
        /// <param name="b">Blue, written.</param>
        /// <param name="a">Alpha (the diffuse material's), written.</param>
        private void Light(float px, float py, float pz, float nx, float ny, float nz, float cr, float cg, float cb, float ca,
                           out float r, out float g, out float b, out float a) {
            float ar = ColorMaterial ? cr : 0.2f, ag = ColorMaterial ? cg : 0.2f, ab = ColorMaterial ? cb : 0.2f;
            float dr = ColorMaterial ? cr : 0.8f, dg = ColorMaterial ? cg : 0.8f, db = ColorMaterial ? cb : 0.8f;
            r = Er + GlobalAmbient * ar;
            g = Eg + GlobalAmbient * ag;
            b = Eb + GlobalAmbient * ab;

            // the viewer's direction: from the vertex to the eye at the origin
            float vl = MathF.Sqrt(px * px + py * py + pz * pz);
            float vx = vl > 0 ? -px / vl : 0, vy = vl > 0 ? -py / vl : 0, vz = vl > 0 ? -pz / vl : 1;
            foreach (GlLight l in Lights) {
                if (!l.Enabled)
                    continue;
                float lx, ly, lz, att = 1;
                if (l.Pw == 0) {
                    float ll = MathF.Sqrt(l.Px * l.Px + l.Py * l.Py + l.Pz * l.Pz);
                    lx = l.Px / ll;
                    ly = l.Py / ll;
                    lz = l.Pz / ll;
                } else {
                    float dx = l.Px / l.Pw - px, dy = l.Py / l.Pw - py, dz = l.Pz / l.Pw - pz;
                    float d = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                    lx = dx / d;
                    ly = dy / d;
                    lz = dz / d;
                    att = 1 / (l.K0 + l.K1 * d + l.K2 * d * d);
                }

                float spot = 1;
                if (l.SpotCutoff != 180) {
                    float sl = MathF.Sqrt(l.SDx * l.SDx + l.SDy * l.SDy + l.SDz * l.SDz);
                    float cosang = -(lx * l.SDx + ly * l.SDy + lz * l.SDz) / sl;
                    spot = cosang >= MathF.Cos(l.SpotCutoff * MathF.PI / 180) ? MathF.Pow(MathF.Max(cosang, 0), l.SpotExponent) : 0;
                }

                float ndotl = MathF.Max(nx * lx + ny * ly + nz * lz, 0);
                float sr = 0, sg = 0, sb = 0;
                if (ndotl > 0) {
                    float hx = lx + vx, hy = ly + vy, hz = lz + vz;
                    float hl = MathF.Sqrt(hx * hx + hy * hy + hz * hz);
                    float ndoth = hl > 0 ? MathF.Max((nx * hx + ny * hy + nz * hz) / hl, 0) : 0;
                    float f = MathF.Pow(ndoth, Shininess);
                    sr = f * l.Sr * MatSpecular;
                    sg = f * l.Sg * MatSpecular;
                    sb = f * l.Sb * MatSpecular;
                }

                float k = att * spot;
                r += k * (l.Ar * ar + ndotl * l.Dr * dr + sr);
                g += k * (l.Ag * ag + ndotl * l.Dg * dg + sg);
                b += k * (l.Ab * ab + ndotl * l.Db * db + sb);
            }

            r = Math.Clamp(r, 0, 1);
            g = Math.Clamp(g, 0, 1);
            b = Math.Clamp(b, 0, 1);
            a = Math.Clamp(ColorMaterial ? ca : 1, 0, 1);
        }

        /// <summary>
        ///     A diagnostic for the frame-parity work (DR-98): when set, the id of the triangle that last wrote each sample's
        ///     colour into <see cref="DebugTarget"/> (triangles numbered from 1 in drawing order), so a pixel that differs
        ///     from the GPU's can be told apart as an edge (samples from several triangles) or an interior (one triangle).
        ///     <see langword="null"/> in every product frame, which then pays one null check per written sample.
        /// </summary>
        internal int[]? DebugTriangleIds;

        /// <summary>The target <see cref="DebugTriangleIds"/> records, its length that target's sample count; others are not recorded.</summary>
        internal GlTarget? DebugTarget;

        /// <summary>The number of the triangle being drawn, counted while <see cref="DebugTriangleIds"/> is set.</summary>
        private int _debugTriangle;

        /// <summary>
        ///     A diagnostic for the frame-parity work (DR-98): when set, every triangle a mesh draws, before any rejection
        ///     or clipping, as its three vertices' clip coordinates (x, y, z, w each), in drawing order, the quads split as
        ///     the GPU splits them; compared with the vertices MuJoCo's own renderer sends, captured by transform feedback.
        ///     <see langword="null"/> in every product frame.
        /// </summary>
        internal List<float>? DebugClipTriangles;

        /// <summary>Clips one triangle against the near, far and user planes, culls it and rasterizes what remains.</summary>
        /// <param name="idx">Its three vertex indices.</param>
        private void Triangle(Span<int> idx) {
            // counted before any rejection, so ids stay stable across a change that culls or clips differently
            if (DebugTriangleIds != null)
                _debugTriangle++;
            Span<Vtx> poly = stackalloc Vtx[12];
            Span<Vtx> tmp = stackalloc Vtx[12];
            poly[0] = _vertices[idx[0]];
            poly[1] = _vertices[idx[1]];
            poly[2] = _vertices[idx[2]];
            int count = 3;

            // reverse-Z clip volume: 0 <= z <= w; the user plane keeps distance >= 0
            count = ClipAgainst(poly, count, tmp, 0);
            if (count < 3)
                return;
            count = ClipAgainst(tmp, count, poly, 1);
            if (count < 3)
                return;
            if (ClipPlane) {
                count = ClipAgainst(poly, count, tmp, 2);
                if (count < 3)
                    return;
                tmp[..count].CopyTo(poly);
            }

            // window coordinates
            Span<float> wx = stackalloc float[12], wy = stackalloc float[12], wz = stackalloc float[12], iw = stackalloc float[12];
            for (int i = 0; i < count; i++) {
                // the GPU's own reciprocal, not the rounded one: its window depth is z times this, rounded (DR-98)
                float invW = GlReciprocal.Of(poly[i].W);
                // snap to the rasterizer's 1/256 sub-pixel grid, ties to even, from ((x rcp(w) + 1) 0.5) size in
                // float32: where the reference GPU puts a vertex (gpu_probe.py's snap, 200 of 200 coordinates located;
                // measured with the viewport at the origin, as MuJoCo's frames have it)
                wx[i] = MathF.Round((_vx + (poly[i].X * invW + 1) * 0.5f * _vw) * 256) / 256;
                wy[i] = MathF.Round((_vy + (poly[i].Y * invW + 1) * 0.5f * _vh) * 256) / 256;
                wz[i] = poly[i].Z * invW;
                iw[i] = invW;
            }

            // facing: the polygon's signed area in window coordinates (y up)
            float area = 0;
            for (int i = 0; i < count; i++) {
                int j = (i + 1) % count;
                area += wx[i] * wy[j] - wx[j] * wy[i];
            }

            if (area == 0)
                return;
            bool front = (area > 0) == FrontCcw;
            if (CullFace && front == CullFront)
                return;

            // the clipped polygon as a fan
            for (int i = 1; i + 1 < count; i++)
                Rasterize(poly, wx, wy, wz, iw, 0, i, i + 1);
        }

        /// <summary>One Sutherland-Hodgman pass against a plane, interpolating every attribute in clip space.</summary>
        /// <param name="input">The polygon.</param>
        /// <param name="count">Its vertex count.</param>
        /// <param name="output">The clipped polygon, written.</param>
        /// <param name="plane">0: z ≥ 0 (far), 1: z ≤ w (near), 2: the user plane.</param>
        /// <returns>The clipped polygon's vertex count.</returns>
        private static int ClipAgainst(Span<Vtx> input, int count, Span<Vtx> output, int plane) {
            int n = 0;
            for (int i = 0; i < count; i++) {
                ref Vtx a = ref input[i];
                ref Vtx b = ref input[(i + 1) % count];
                float da = Distance(in a, plane), db = Distance(in b, plane);
                if (da >= 0)
                    output[n++] = a;
                if ((da >= 0) != (db >= 0)) {
                    float t = da / (da - db);
                    output[n++] = Lerp(in a, in b, t);
                }
            }

            return n;

            static float Distance(in Vtx v, int plane) => plane switch {
                0 => v.Z,
                1 => v.W - v.Z,
                _ => v.Clip,
            };
        }

        /// <summary>Interpolates every attribute of two vertices.</summary>
        /// <param name="a">The first vertex.</param>
        /// <param name="b">The second.</param>
        /// <param name="t">The weight of the second.</param>
        /// <returns>The vertex between.</returns>
        private static Vtx Lerp(in Vtx a, in Vtx b, float t) => new() {
            X = a.X + t * (b.X - a.X), Y = a.Y + t * (b.Y - a.Y), Z = a.Z + t * (b.Z - a.Z), W = a.W + t * (b.W - a.W),
            R = a.R + t * (b.R - a.R), G = a.G + t * (b.G - a.G), B = a.B + t * (b.B - a.B), A = a.A + t * (b.A - a.A),
            S = a.S + t * (b.S - a.S), T = a.T + t * (b.T - a.T), U = a.U + t * (b.U - a.U), Q = a.Q + t * (b.Q - a.Q),
            S1 = a.S1 + t * (b.S1 - a.S1), T1 = a.T1 + t * (b.T1 - a.T1), U1 = a.U1 + t * (b.U1 - a.U1), Q1 = a.Q1 + t * (b.Q1 - a.Q1),
            Clip = a.Clip + t * (b.Clip - a.Clip), FogZ = a.FogZ + t * (b.FogZ - a.FogZ),
        };

        /// <summary>
        ///     Rasterizes one triangle of a clipped polygon: coverage per sample (the left-or-bottom tie rule), depth per sample, the
        ///     fragment's colour once per pixel at its centre (perspective-correct), then the per-sample tests and writes.
        /// </summary>
        /// <param name="v">The polygon's vertices.</param>
        /// <param name="wx">Window x per vertex.</param>
        /// <param name="wy">Window y per vertex.</param>
        /// <param name="wz">Window depth per vertex.</param>
        /// <param name="iw">1/w per vertex.</param>
        /// <param name="i0">The first vertex.</param>
        /// <param name="i1">The second.</param>
        /// <param name="i2">The third.</param>
        private void Rasterize(Span<Vtx> v, Span<float> wx, Span<float> wy, Span<float> wz, Span<float> iw, int i0, int i1, int i2) {
            double x0 = wx[i0], y0 = wy[i0], x1 = wx[i1], y1 = wy[i1], x2 = wx[i2], y2 = wy[i2];
            double area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if (area == 0)
                return;
            // make the edge functions positive inside
            if (area < 0) {
                (i1, i2) = (i2, i1);
                (x1, y1, x2, y2) = (x2, y2, x1, y1);
                area = -area;
            }

            GlTarget tg = _target;
            int minX = Math.Max(_vx, (int)Math.Floor(Math.Min(x0, Math.Min(x1, x2))));
            int maxX = Math.Min(_vx + _vw - 1, (int)Math.Ceiling(Math.Max(x0, Math.Max(x1, x2))));
            int minY = Math.Max(_vy, (int)Math.Floor(Math.Min(y0, Math.Min(y1, y2))));
            int maxY = Math.Min(_vy + _vh - 1, (int)Math.Ceiling(Math.Max(y0, Math.Max(y1, y2))));
            minX = Math.Max(minX, 0);
            minY = Math.Max(minY, 0);
            maxX = Math.Min(maxX, tg.Width - 1);
            maxY = Math.Min(maxY, tg.Height - 1);
            if (minX > maxX || minY > maxY)
                return;

            var t = new Setup {
                X0 = x0, Y0 = y0, X1 = x1, Y1 = y1, X2 = x2, Y2 = y2, InvArea = 1 / area,
                Z0 = wz[i0], Z1 = wz[i1], Z2 = wz[i2], W0 = iw[i0], W1 = iw[i1], W2 = iw[i2],
                A = v[i0], B = v[i1], C = v[i2], MinX = minX, MaxX = maxX,
                // edge k runs from vertex k+1 to vertex k+2; a sample exactly on it counts for left and bottom edges
                Tl0 = LeftOrBottom(x1, y1, x2, y2), Tl1 = LeftOrBottom(x2, y2, x0, y0), Tl2 = LeftOrBottom(x0, y0, x1, y1),
                // only colour writes and the alpha test read the fragment's colour; depth and stencil passes skip it
                NeedsColor = (ColorMask && tg.Color != null) || AlphaTest,
            };

            // polygon offset: the depth slope times the factor plus the resolvable difference times the units
            if (PolygonOffset) {
                double dzdx = ((t.Z1 - t.Z0) * (y2 - y0) - (t.Z2 - t.Z0) * (y1 - y0)) * t.InvArea;
                double dzdy = ((t.Z2 - t.Z0) * (x1 - x0) - (t.Z1 - t.Z0) * (x2 - x0)) * t.InvArea;
                double m = Math.Max(Math.Abs(dzdx), Math.Abs(dzdy));
                double r = tg.Depth24 ? 1.0 / (1 << 24) : Math.Pow(2, Math.Floor(Math.Log2(Math.Max(t.Z0, Math.Max(t.Z1, t.Z2)))) - 23);
                t.Offset = (float)(m * OffsetFactor + r * OffsetUnits);
            }

            // rows never share a pixel, so a large triangle (the floor in the shadow map) is split into bands of rows
            // drawn in parallel; the result is the same as drawing the rows in order
            long pixels = (long)(maxX - minX + 1) * (maxY - minY + 1);
            if (pixels < ParallelPixels) {
                RasterRows(t, minY, maxY);
                return;
            }

            // bands of at least 32 rows: finer bands cost more in scheduling than they save
            int rows = maxY - minY + 1, bands = Math.Clamp(rows / 32, 1, Environment.ProcessorCount);
            System.Threading.Tasks.Parallel.For(0, bands, band => {
                int from = minY + (int)((long)rows * band / bands), to = minY + (int)((long)rows * (band + 1) / bands) - 1;
                RasterRows(t, from, to);
            });
        }

        /// <summary>Triangles covering fewer pixels than this are drawn on the calling thread.</summary>
        private const long ParallelPixels = 16384;

        /// <summary>One triangle's rasterization setup, shared read-only by the rows that draw it.</summary>
        private sealed class Setup {
            /// <summary>Window coordinates of the three vertices (counter-clockwise).</summary>
            public double X0, Y0, X1, Y1, X2, Y2;

            /// <summary>1 over the doubled area.</summary>
            public double InvArea;

            /// <summary>Window depths.</summary>
            public float Z0, Z1, Z2;

            /// <summary>1/w per vertex.</summary>
            public float W0, W1, W2;

            /// <summary>The vertices' attributes.</summary>
            public Vtx A, B, C;

            /// <summary>The columns to scan.</summary>
            public int MinX, MaxX;

            /// <summary>Whether a sample exactly on each edge counts as inside (<see cref="LeftOrBottom"/>).</summary>
            public bool Tl0, Tl1, Tl2;

            /// <summary>Whether the fragment's colour is needed.</summary>
            public bool NeedsColor;

            /// <summary>The polygon offset added to every depth.</summary>
            public float Offset;
        }

        /// <summary>
        ///     Draws rows <paramref name="fromY"/> to <paramref name="toY"/> of a triangle: coverage per sample with edge
        ///     functions stepped across the row (exact: every coordinate is a multiple of 1/256), depth per sample, the
        ///     fragment once per pixel, then the per-sample tests and writes.
        /// </summary>
        /// <param name="t">The setup.</param>
        /// <param name="fromY">The first row.</param>
        /// <param name="toY">The last row.</param>
        private void RasterRows(Setup t, int fromY, int toY) {
            GlTarget tg = _target;
            if (tg.Samples == 1 && !t.NeedsColor && tg.Stencil == null) {
                DepthSpans(t, fromY, toY);
                return;
            }

            float[] spos = tg.Samples == 4 ? Samples4 : Samples1;
            int ns = tg.Samples;
            double x0 = t.X0, y0 = t.Y0, x1 = t.X1, y1 = t.Y1, x2 = t.X2, y2 = t.Y2;
            Span<int> lo = stackalloc int[4], hi = stackalloc int[4];
            bool wide = t.MaxX - t.MinX >= 16;
            Span<bool> covered = stackalloc bool[4];
            Span<float> sampleZ = stackalloc float[4];
            for (int py = fromY; py <= toY; py++) {
                // a wide triangle: each sample's covered columns on this row, exact, and only their union is visited; a
                // narrow one (most of a capsule's) is cheaper tested pixel by pixel
                int ulo = t.MinX, uhi = t.MaxX;
                if (wide) {
                    ulo = int.MaxValue;
                    uhi = int.MinValue;
                    for (int s = 0; s < ns; s++) {
                        lo[s] = t.MinX;
                        hi[s] = t.MaxX;
                        SampleSpan(t, py + spos[2 * s + 1], spos[2 * s], ref lo[s], ref hi[s]);
                        if (lo[s] <= hi[s]) {
                            ulo = Math.Min(ulo, lo[s]);
                            uhi = Math.Max(uhi, hi[s]);
                        }
                    }
                }

                for (int px = ulo; px <= uhi; px++) {
                    bool any = false;
                    for (int s = 0; s < ns; s++) {
                        double sx = px + spos[2 * s], sy = py + spos[2 * s + 1];
                        double a0 = (x2 - x1) * (sy - y1) - (y2 - y1) * (sx - x1);
                        double a1 = (x0 - x2) * (sy - y2) - (y0 - y2) * (sx - x2);
                        double a2 = (x1 - x0) * (sy - y0) - (y1 - y0) * (sx - x0);
                        bool inside = wide
                            ? px >= lo[s] && px <= hi[s]
                            : (a0 > 0 || (a0 == 0 && t.Tl0)) && (a1 > 0 || (a1 == 0 && t.Tl1)) && (a2 > 0 || (a2 == 0 && t.Tl2));
                        covered[s] = inside;
                        if (!inside)
                            continue;
                        any = true;
                        // depth is affine in window space; the edge values are exact, so the depth is the same
                        // whichever way the sample was found
                        sampleZ[s] = TruncateToFloat((a0 * t.Z0 + a1 * t.Z1 + a2 * t.Z2) * t.InvArea) + t.Offset;
                    }

                    if (!any)
                        continue;

                    int pixel = py * tg.Width + px;
                    float fr = 0, fg = 0, fb = 0, fa = 0;
                    if (t.NeedsColor) {
                        // early rejection, as the GPU's: no covered sample passes the stencil and depth tests, so the
                        // fragment would change nothing (the tests are rerun in order below)
                        bool canPass = false;
                        for (int s = 0; s < ns && !canPass; s++)
                            canPass = covered[s] && WouldPass(pixel * ns + s, sampleZ[s]);
                        if (!canPass)
                            continue;

                        // the fragment, perspective-correct: texture coordinates at the pixel's centre, the colour at the
                        // centre too when every sample is covered, else at the first covered sample in index order (the
                        // GPU's centroid rule for its colour varyings, measured sample by sample: 100% of 11,000 edge
                        // pixels; texture coordinates stayed at the centre in every coverage pattern)
                        float cox = 0.5f, coy = 0.5f;
                        for (int s = 0; s < ns; s++) {
                            if (covered[s])
                                continue;
                            for (int f = 0; f < ns; f++) {
                                if (!covered[f])
                                    continue;
                                cox = spos[2 * f];
                                coy = spos[2 * f + 1];
                                break;
                            }

                            break;
                        }

                        Shade(px, py, cox, coy, x0, y0, x1, y1, x2, y2, t.InvArea, in t.A, in t.B, in t.C, t.W0, t.W1, t.W2,
                              out fr, out fg, out fb, out fa, out bool discard);
                        if (discard)
                            continue;
                    }

                    for (int s = 0; s < ns; s++) {
                        if (covered[s])
                            WriteSample(pixel * ns + s, sampleZ[s], fr, fg, fb, fa);
                    }
                }
            }
        }

        /// <summary>
        ///     A depth computed in double, cut to float32 toward zero, as the reference GPU stores an interpolated depth: it
        ///     truncates where a cast rounds to nearest (DR-98, measured 2026-11-21 on exact-slope triangles: truncating the
        ///     exact plane reproduced 94% to 100% of each triangle's samples, rounding about 47%; the rest is the GPU's
        ///     per-4×4-tile evaluation, not modelled yet).
        /// </summary>
        /// <param name="value">The depth.</param>
        /// <returns>The float32 nearest to <paramref name="value"/> not farther from zero; NaN stays NaN.</returns>
        private static float TruncateToFloat(double value) {
            float f = (float)value;
            // the cast rounded away from zero: step back one float toward zero
            if (Math.Abs((double)f) > Math.Abs(value))
                f = f > 0 ? MathF.BitDecrement(f) : MathF.BitIncrement(f);
            return f;
        }

        /// <summary>
        ///     Whether a sample could pass the stencil and depth tests, without changing anything: the early rejection
        ///     that spares shading fragments nothing would show (the skybox behind the floor, the reflection outside it).
        /// </summary>
        /// <param name="k">The sample.</param>
        /// <param name="z">The sample's depth.</param>
        /// <returns>False when the stencil or the depth test is certain to fail.</returns>
        private bool WouldPass(int k, float z) {
            GlTarget tg = _target;
            if (StencilTest && StencilEqual && tg.Stencil != null && tg.Stencil[k] != StencilRef)
                return false;
            if (!DepthTest)
                return true;
            float zc = Math.Clamp(z, 0, 1);
            if (tg.Depth24)
                zc = Quantize24(zc);
            return zc >= tg.Depth[k];
        }

        /// <summary>
        ///     Draws rows of a triangle into a single-sample, depth-only target (the shadow map): each row's covered span
        ///     from the edge functions, adjusted by exact evaluation at its ends, then only depth tests and writes.
        /// </summary>
        /// <param name="t">The setup.</param>
        /// <param name="fromY">The first row.</param>
        /// <param name="toY">The last row.</param>
        private void DepthSpans(Setup t, int fromY, int toY) {
            GlTarget tg = _target;
            float[] depth = tg.Depth;
            bool q24 = tg.Depth24, test = DepthTest, write = DepthTest && DepthMask;
            double x0 = t.X0, y0 = t.Y0, x1 = t.X1, y1 = t.Y1, x2 = t.X2, y2 = t.Y2;
            for (int py = fromY; py <= toY; py++) {
                double sy = py + 0.5;
                int lo = t.MinX, hi = t.MaxX;
                SampleSpan(t, sy, 0.5, ref lo, ref hi);
                if (lo > hi || !test)
                    continue;
                int row = py * tg.Width;
                // the depth plane at the span's first centre, stepped across the row (each step from the start, so no
                // error accumulates)
                double sx0 = lo + 0.5;
                double e0 = (x2 - x1) * (sy - y1) - (y2 - y1) * (sx0 - x1);
                double e1 = (x0 - x2) * (sy - y2) - (y0 - y2) * (sx0 - x2);
                double e2 = (x1 - x0) * (sy - y0) - (y1 - y0) * (sx0 - x0);
                double zStart = (e0 * t.Z0 + e1 * t.Z1 + e2 * t.Z2) * t.InvArea;
                double dz = (-(y2 - y1) * t.Z0 - (y0 - y2) * t.Z1 - (y1 - y0) * t.Z2) * t.InvArea;
                for (int px = lo; px <= hi; px++) {
                    float z = (float)(zStart + (px - lo) * dz) + t.Offset;
                    float zc = z < 0 ? 0 : z > 1 ? 1 : z;
                    if (q24)
                        zc = (float)((long)(zc * 16777215.0 + 0.5) * InvDepth24);
                    int k = row + px;
                    if (zc >= depth[k] && write)
                        depth[k] = zc;
                }
            }
        }

        /// <summary>
        ///     Narrows [<paramref name="lo"/>, <paramref name="hi"/>] to the pixels whose sample at
        ///     (<c>px + offX</c>, <paramref name="sy"/>) lies inside the triangle under the left-or-bottom tie rule: each edge bounds
        ///     the span from one side, its crossing found by division and then stepped to the exact boundary by evaluating
        ///     the edge function, which is exact (every coordinate is a multiple of 1/256).
        /// </summary>
        /// <param name="t">The setup.</param>
        /// <param name="sy">The sample row's y.</param>
        /// <param name="offX">The sample's offset in its pixel.</param>
        /// <param name="lo">The first column, narrowed.</param>
        /// <param name="hi">The last column, narrowed; below <paramref name="lo"/> when nothing is covered.</param>
        private static void SampleSpan(Setup t, double sy, double offX, ref int lo, ref int hi) {
            ClipEdge(t.X1, t.Y1, t.X2, t.Y2, t.Tl0, sy, offX, ref lo, ref hi);
            ClipEdge(t.X2, t.Y2, t.X0, t.Y0, t.Tl1, sy, offX, ref lo, ref hi);
            ClipEdge(t.X0, t.Y0, t.X1, t.Y1, t.Tl2, sy, offX, ref lo, ref hi);
        }

        /// <summary>Narrows a span to the pixels whose sample lies inside one edge (from (ax, ay) to (bx, by)).</summary>
        /// <param name="ax">The edge's start x.</param>
        /// <param name="ay">The edge's start y.</param>
        /// <param name="bx">The edge's end x.</param>
        /// <param name="by">The edge's end y.</param>
        /// <param name="topLeft">Whether a sample exactly on the edge counts (<see cref="LeftOrBottom"/>).</param>
        /// <param name="sy">The sample row's y.</param>
        /// <param name="offX">The sample's offset in its pixel.</param>
        /// <param name="lo">The first column, narrowed.</param>
        /// <param name="hi">The last column, narrowed.</param>
        private static void ClipEdge(double ax, double ay, double bx, double by, bool topLeft, double sy, double offX, ref int lo, ref int hi) {
            if (lo > hi)
                return;
            double dEdx = -(by - ay);
            if (dEdx == 0) {
                if (!Inside(lo))
                    hi = lo - 1;
                return;
            }

            double cross = (bx - ax) * (sy - ay) / (by - ay) + ax - offX;
            int b = (int)Math.Clamp(Math.Floor(cross), lo - 1, hi + 1);
            if (dEdx > 0) {
                // inside to the right of the crossing
                while (b >= lo && Inside(b))
                    b--;
                while (b <= hi && !Inside(b))
                    b++;
                lo = Math.Max(lo, b);
            } else {
                // inside to the left
                while (b <= hi && Inside(b))
                    b++;
                while (b >= lo && !Inside(b))
                    b--;
                hi = Math.Min(hi, b);
            }

            bool Inside(int x) {
                double e = (bx - ax) * (sy - ay) - (by - ay) * (x + offX - ax);
                return e > 0 || (e == 0 && topLeft);
            }
        }

        /// <summary>
        ///     Whether a sample lying exactly on an edge from (x0, y0) to (x1, y1) of a counter-clockwise triangle (window
        ///     coordinates, y up) belongs to the triangle: on a left edge, or on a bottom edge, as the reference GPU decides
        ///     ties in MuJoCo's lower-left window (its own top-left rule applied in its y-down raster, DR-98).
        /// </summary>
        /// <param name="x0">Start x.</param>
        /// <param name="y0">Start y.</param>
        /// <param name="x1">End x.</param>
        /// <param name="y1">End y.</param>
        /// <returns>True for a left edge (running down) or a bottom edge (horizontal, running right).</returns>
        /// <remarks>
        ///     Measured on 2026-11-22: pixel centres on a shared vertical edge go to the triangle on its right, on a shared
        ///     horizontal edge to the triangle above it, for either winding, and on sloped edges as this rule names
        ///     (<c>tools/parity/mujoco/gpu_probe.py</c>'s <c>ties</c>: 1,648 of 1,648 centres, 14 directions). MuJoCo's
        ///     capsules seen level put horizontal edges through sample rows, where the former top-left reading gave
        ///     Reacher's far rail 172 edge pixels of the wrong triangle.
        /// </remarks>
        private static bool LeftOrBottom(double x0, double y0, double x1, double y1) => y1 < y0 || (y1 == y0 && x1 > x0);

        /// <summary>
        ///     The fragment of one pixel: its colour, unit 0's texture modulating it, unit 1's shadow compare modulating it,
        ///     fog, and the alpha test.
        /// </summary>
        /// <param name="px">The pixel's x.</param>
        /// <param name="py">The pixel's y.</param>
        /// <param name="cox">Where in the pixel the colour is evaluated, x (0.5 at the centre; a covered sample's offset
        ///     on an edge pixel, the GPU's centroid rule for colours).</param>
        /// <param name="coy">Where in the pixel the colour is evaluated, y.</param>
        /// <param name="x0">Vertex 0's window x.</param>
        /// <param name="y0">Vertex 0's window y.</param>
        /// <param name="x1">Vertex 1's window x.</param>
        /// <param name="y1">Vertex 1's window y.</param>
        /// <param name="x2">Vertex 2's window x.</param>
        /// <param name="y2">Vertex 2's window y.</param>
        /// <param name="invArea">1 over the triangle's doubled area.</param>
        /// <param name="a">Vertex 0.</param>
        /// <param name="b">Vertex 1.</param>
        /// <param name="c">Vertex 2.</param>
        /// <param name="w0">Vertex 0's 1/w.</param>
        /// <param name="w1">Vertex 1's 1/w.</param>
        /// <param name="w2">Vertex 2's 1/w.</param>
        /// <param name="r">Red, written.</param>
        /// <param name="g">Green, written.</param>
        /// <param name="bl">Blue, written.</param>
        /// <param name="al">Alpha, written.</param>
        /// <param name="discard">Whether the alpha test discarded the fragment.</param>
        private void Shade(int px, int py, float cox, float coy, double x0, double y0, double x1, double y1, double x2, double y2,
                           double invArea, in Vtx a, in Vtx b, in Vtx c, float w0, float w1, float w2,
                           out float r, out float g, out float bl, out float al, out bool discard) {
            Weights(px + 0.5, py + 0.5, x0, y0, x1, y1, x2, y2, invArea, w0, w1, w2, out float l0, out float l1, out float l2);
            // the colour's own weights only on an edge pixel; everything else (texture, shadow and fog coordinates) is
            // evaluated at the centre
            float k0 = l0, k1 = l1, k2 = l2;
            if (cox != 0.5f || coy != 0.5f)
                Weights(px + cox, py + coy, x0, y0, x1, y1, x2, y2, invArea, w0, w1, w2, out k0, out k1, out k2);
            r = k0 * a.R + k1 * b.R + k2 * c.R;
            g = k0 * a.G + k1 * b.G + k2 * c.G;
            bl = k0 * a.B + k1 * b.B + k2 * c.B;
            al = k0 * a.A + k1 * b.A + k2 * c.A;
            discard = false;

            if (Texture != null) {
                float s = l0 * a.S + l1 * b.S + l2 * c.S, t = l0 * a.T + l1 * b.T + l2 * c.T;
                float u = l0 * a.U + l1 * b.U + l2 * c.U, q = l0 * a.Q + l1 * b.Q + l2 * c.Q;
                // one level of detail per 2×2 quad, as the GPU computes it (measured): the differences at the quad's
                // lower-left pixel, extrapolated on this triangle's plane where that pixel lies outside it, so all four
                // pixels of a quad share the level even though their own derivatives differ under perspective
                int qx = px & ~1, qy = py & ~1;
                float tr, tg, tb;
                if (!Texture.IsCube) {
                    Coord2D(qx, qy, x0, y0, x1, y1, x2, y2, invArea, in a, in b, in c, w0, w1, w2, out float sa, out float ta);
                    Coord2D(qx + 1, qy, x0, y0, x1, y1, x2, y2, invArea, in a, in b, in c, w0, w1, w2, out float sb, out float tb2);
                    Coord2D(qx, qy + 1, x0, y0, x1, y1, x2, y2, invArea, in a, in b, in c, w0, w1, w2, out float sd, out float td);
                    float dudx = (sb - sa) * Texture.Width, dvdx = (tb2 - ta) * Texture.Height;
                    float dudy = (sd - sa) * Texture.Width, dvdy = (td - ta) * Texture.Height;
                    float rho = MathF.Max(Length(dudx, dvdx), Length(dudy, dvdy));
                    Texture.Sample2D(s / q, t / q, GlTexture.LodSteps(rho), out tr, out tg, out tb);
                } else {
                    // a cube map is sampled at level 0: the skybox is always magnified on screen, and a cube-textured
                    // geom shrinks below a texel per pixel only far away (an approximation the frame cells' baseline covers)
                    Texture.SampleCube(s, t, u, 0, out tr, out tg, out tb);
                }

                r *= tr;
                g *= tg;
                bl *= tb;
            }

            if (ShadowMap != null) {
                float s1 = l0 * a.S1 + l1 * b.S1 + l2 * c.S1, t1 = l0 * a.T1 + l1 * b.T1 + l2 * c.T1;
                float u1 = l0 * a.U1 + l1 * b.U1 + l2 * c.U1, q1 = l0 * a.Q1 + l1 * b.Q1 + l2 * c.Q1;
                float lit = ShadowCompare(s1 / q1, t1 / q1, u1 / q1);
                r *= lit;
                g *= lit;
                bl *= lit;
                al *= lit;
            }

            if (Fog) {
                float fz = l0 * a.FogZ + l1 * b.FogZ + l2 * c.FogZ;
                float f = Math.Clamp((FogEnd - fz) / (FogEnd - FogStart), 0, 1);
                r = f * r + (1 - f) * FogR;
                g = f * g + (1 - f) * FogG;
                bl = f * bl + (1 - f) * FogB;
            }

            if (AlphaTest && !(al >= 0.99f))
                discard = true;
        }

        /// <summary>
        ///     A texture derivative's length as the GPU's level of detail takes it: <c>max + 0.3442·min</c> of its absolute
        ///     texel components, both truncated to 9 fractional bits at the larger one's exponent, not the Euclidean
        ///     length. OpenGL allows any function between the larger component and their sum (§8.14.1); this one was
        ///     fitted to the reference GPU's levels over 722 rotated derivatives (97% exact, the rest one step of 1/256
        ///     away), where it reaches 0.075 of a level above the Euclidean length near 20° and as far below it at 45°.
        /// </summary>
        /// <param name="du">The derivative's u component, in texels.</param>
        /// <param name="dv">The derivative's v component, in texels.</param>
        /// <returns>The length, in texels; 0 when both components are 0, NaN when either is.</returns>
        private static float Length(float du, float dv) {
            float a = MathF.Abs(du), b = MathF.Abs(dv);
            if (float.IsNaN(a) || float.IsNaN(b))
                return float.NaN;
            float hi = MathF.Max(a, b), lo = MathF.Min(a, b);
            // a zero, subnormal or infinite larger component needs no fixed point (and ILogB is only exact on normal
            // numbers): the level of detail treats those as magnified or as the smallest level anyway
            if (!(hi >= GlTexture.MinNormal) || float.IsPositiveInfinity(hi))
                return hi;
            float scale = MathF.ScaleB(1, 9 - MathF.ILogB(hi));
            return (MathF.Floor(hi * scale) + MathF.Floor(0.3442f * MathF.Floor(lo * scale))) / scale;
        }

        /// <summary>Unit 0's projected 2D coordinates at another pixel's centre, for the level of detail.</summary>
        /// <param name="x">The pixel's x.</param>
        /// <param name="y">The pixel's y.</param>
        /// <param name="x0">Vertex 0's window x.</param>
        /// <param name="y0">Vertex 0's window y.</param>
        /// <param name="x1">Vertex 1's window x.</param>
        /// <param name="y1">Vertex 1's window y.</param>
        /// <param name="x2">Vertex 2's window x.</param>
        /// <param name="y2">Vertex 2's window y.</param>
        /// <param name="invArea">1 over the doubled area.</param>
        /// <param name="a">Vertex 0.</param>
        /// <param name="b">Vertex 1.</param>
        /// <param name="c">Vertex 2.</param>
        /// <param name="w0">Vertex 0's 1/w.</param>
        /// <param name="w1">Vertex 1's 1/w.</param>
        /// <param name="w2">Vertex 2's 1/w.</param>
        /// <param name="ss">s / q, written.</param>
        /// <param name="tt">t / q, written.</param>
        private static void Coord2D(int x, int y, double x0, double y0, double x1, double y1, double x2, double y2, double invArea,
                                    in Vtx a, in Vtx b, in Vtx c, float w0, float w1, float w2, out float ss, out float tt) {
            Weights(x + 0.5, y + 0.5, x0, y0, x1, y1, x2, y2, invArea, w0, w1, w2, out float m0, out float m1, out float m2);
            float qq = m0 * a.Q + m1 * b.Q + m2 * c.Q;
            ss = (m0 * a.S + m1 * b.S + m2 * c.S) / qq;
            tt = (m0 * a.T + m1 * b.T + m2 * c.T) / qq;
        }

        /// <summary>Perspective-correct barycentric weights at a point.</summary>
        /// <param name="sx">The point's x.</param>
        /// <param name="sy">The point's y.</param>
        /// <param name="x0">Vertex 0's x.</param>
        /// <param name="y0">Vertex 0's y.</param>
        /// <param name="x1">Vertex 1's x.</param>
        /// <param name="y1">Vertex 1's y.</param>
        /// <param name="x2">Vertex 2's x.</param>
        /// <param name="y2">Vertex 2's y.</param>
        /// <param name="invArea">1 over the doubled area.</param>
        /// <param name="w0">Vertex 0's 1/w.</param>
        /// <param name="w1">Vertex 1's 1/w.</param>
        /// <param name="w2">Vertex 2's 1/w.</param>
        /// <param name="l0">Vertex 0's weight, written.</param>
        /// <param name="l1">Vertex 1's weight, written.</param>
        /// <param name="l2">Vertex 2's weight, written.</param>
        private static void Weights(double sx, double sy, double x0, double y0, double x1, double y1, double x2, double y2, double invArea,
                                    float w0, float w1, float w2, out float l0, out float l1, out float l2) {
            double b0 = ((x2 - x1) * (sy - y1) - (y2 - y1) * (sx - x1)) * invArea;
            double b1 = ((x0 - x2) * (sy - y2) - (y0 - y2) * (sx - x2)) * invArea;
            double b2 = 1 - b0 - b1;
            double p0 = b0 * w0, p1 = b1 * w1, p2 = b2 * w2, sum = p0 + p1 + p2;
            l0 = (float)(p0 / sum);
            l1 = (float)(p1 / sum);
            l2 = (float)(p2 / sum);
        }

        /// <summary>
        ///     The shadow map's compare (<c>GL_COMPARE_R_TO_TEXTURE</c>, <c>GL_GEQUAL</c>, <c>GL_NEAREST</c>,
        ///     <c>GL_CLAMP_TO_EDGE</c>): 1 when the fragment is at least as close to the light as the stored depth.
        /// </summary>
        /// <param name="s">The map's s.</param>
        /// <param name="t">The map's t.</param>
        /// <param name="r">The fragment's depth from the light.</param>
        /// <returns>1 lit, 0 shadowed.</returns>
        private float ShadowCompare(float s, float t, float r) {
            GlTarget map = ShadowMap!;
            int i = Math.Clamp((int)MathF.Floor(s * map.Width), 0, map.Width - 1);
            int j = Math.Clamp((int)MathF.Floor(t * map.Height), 0, map.Height - 1);
            float stored = map.Depth[j * map.Width + i];
            if (map.Depth24)
                r = Quantize24(Math.Clamp(r, 0, 1));
            return r >= stored ? 1 : 0;
        }

        /// <summary>The value of one step of a 24-bit fixed-point depth.</summary>
        private const double InvDepth24 = 1.0 / 16777215.0;

        /// <summary>A depth as a 24-bit fixed-point value reads back (rounded to the nearest of 2^24 - 1 steps).</summary>
        /// <param name="z">The depth in [0, 1].</param>
        /// <returns>The nearest representable depth.</returns>
        private static float Quantize24(float z) => (float)((long)(z * 16777215.0 + 0.5) * InvDepth24);

        /// <summary>The per-sample operations: stencil test, depth test, stencil update, blending and the writes.</summary>
        /// <param name="k">The sample's index.</param>
        /// <param name="z">The sample's depth.</param>
        /// <param name="r">The fragment's red.</param>
        /// <param name="g">Green.</param>
        /// <param name="b">Blue.</param>
        /// <param name="a">Alpha.</param>
        private void WriteSample(int k, float z, float r, float g, float b, float a) {
            GlTarget tg = _target;
            if (StencilTest && tg.Stencil != null) {
                bool pass = !StencilEqual || tg.Stencil[k] == StencilRef;
                if (StencilReplace)
                    tg.Stencil[k] = StencilRef;
                if (!pass)
                    return;
            }

            if (DepthTest) {
                float zc = Math.Clamp(z, 0, 1);
                if (tg.Depth24)
                    zc = Quantize24(zc);
                if (!(zc >= tg.Depth[k]))
                    return;
                if (DepthMask)
                    tg.Depth[k] = zc;
            }

            if (!ColorMask || tg.Color == null)
                return;
            if (DebugTriangleIds != null && ReferenceEquals(tg, DebugTarget))
                DebugTriangleIds[k] = _debugTriangle;
            int o = 4 * k;
            byte[] px = tg.Color;
            byte sr = ToUnorm8(r), sg = ToUnorm8(g), sb = ToUnorm8(b), sa = ToUnorm8(a);
            if (!Blend) {
                px[o] = sr;
                px[o + 1] = sg;
                px[o + 2] = sb;
                px[o + 3] = sa;
                return;
            }

            // the GPU blends in the buffer's own format (measured, 20,000 of 20,000 for both of MuJoCo's modes): the
            // source colour and alpha are stored to 8 bits first, then each channel is the exact integer combination of
            // the levels, rounded to the nearest level and clamped, the factors in units of 1/255
            int sf = Factor(BlendSrc, sa), df = Factor(BlendDst, sa);
            px[o] = BlendLevel(sr, px[o], sf, df);
            px[o + 1] = BlendLevel(sg, px[o + 1], sf, df);
            px[o + 2] = BlendLevel(sb, px[o + 2], sf, df);
            px[o + 3] = BlendLevel(sa, px[o + 3], sf, df);

            static int Factor(GlBlend f, int alpha) => f switch {
                GlBlend.One => 255,
                GlBlend.SrcAlpha => alpha,
                _ => 255 - alpha,
            };
        }

        /// <summary>
        ///     One channel of a blend as the reference GPU computes it in an 8-bit buffer: the exact integer combination of the
        ///     stored source and destination levels, rounded to the nearest level and clamped (measured, 20,000 of 20,000 for
        ///     both of MuJoCo's blend modes). <see cref="MujocoOverlay"/>'s half-black box uses it too.
        /// </summary>
        /// <param name="s">The source level, already stored to 8 bits (<see cref="ToUnorm8"/>).</param>
        /// <param name="d">The destination level.</param>
        /// <param name="sf">The source factor in units of 1/255 (255 for <c>GL_ONE</c>, the source's alpha level for <c>GL_SRC_ALPHA</c>).</param>
        /// <param name="df">The destination factor in units of 1/255.</param>
        /// <returns>The new level.</returns>
        internal static byte BlendLevel(int s, int d, int sf, int df) => (byte)Math.Min(255, (s * sf + d * df + 127) / 255);

        /// <summary>
        ///     A colour channel as the GPU stores it in an 8-bit buffer: clamped, truncated to 12 fractional bits, then
        ///     rounded to the nearest of the 255 steps with an exact half rounding down.
        /// </summary>
        /// <param name="c">The channel, clamped to [0, 1] here (NaN stores 0).</param>
        /// <returns>The level.</returns>
        /// <remarks>
        ///     Measured on the reference GPU (<c>gpu_probe.py</c>'s store probe, 2026-11-12): a flat colour
        ///     <c>(i + δ) / 255</c> stores <c>i + 1</c> exactly from <c>δ = ⌈(i + ½)·4096/255⌉·255/4096 − i</c>, which a
        ///     plain <c>round(c·255)</c> misses by up to 0.06 of a level; the same rule makes a texel blend of exactly ½
        ///     store 127, as the GPU does. Every fragment colour passes through it, so a mismatch here alone shows as a
        ///     one-level difference on near-tie pixels all over a frame.
        /// </remarks>
        internal static byte ToUnorm8(float c) {
            // !(c > 0) also catches NaN, which the GPU stores as 0
            if (!(c > 0))
                return 0;
            int c12 = c >= 1 ? 4096 : (int)(c * 4096);
            return (byte)((c12 * 255 + 2047) >> 12);
        }
    }
}
