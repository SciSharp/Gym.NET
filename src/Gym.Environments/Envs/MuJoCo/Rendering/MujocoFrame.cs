using System;
using System.Collections.Generic;
using Gym.MuJoCo;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     MuJoCo's classic renderer for one model (<c>mjrContext</c> and <c>mjr_render</c>): the builtin meshes, the
    ///     textures, the multisampled offscreen buffer and the shadow map, and the passes of a frame in MuJoCo's order (the
    ///     floor's reflection, the opaque geoms under the lights that cast no shadow, each shadow light's depth map and its
    ///     additive pass, the skybox with its haze, the transparent geoms), then the read-back (<c>mjr_readPixels</c>).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Drawn by <see cref="GlContext"/> in software, so a frame never needs a GPU, and repeats bit for bit (DR-71's
    ///     goldens); what remains different from the GPU frames Gymnasium draws is measured, not required equal (DR-67).
    ///     </para>
    ///     <para>
    ///     Reference: <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_gl3.c">render_gl3.c</see>,
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_gl2.c">render_gl2.c</see>
    ///     (<c>mjr_readPixels</c>).
    ///     </para>
    /// </remarks>
    internal sealed class MujocoFrame {
        /// <summary>The renders' modes for <see cref="RenderGeom"/>.</summary>
        private const int Normal = 0, ShadowMapMode = 1, ShadowCast = 2;

        /// <summary>The model.</summary>
        private readonly MjModel _m;

        /// <summary>The builtin meshes.</summary>
        private readonly MujocoMeshes _meshes;

        /// <summary>The model's textures, built on first use, by texture id.</summary>
        private readonly GlTexture?[] _textures;

        /// <summary>The software OpenGL.</summary>
        private readonly GlContext _gl = new();

        /// <summary>The offscreen buffer, sized on demand.</summary>
        private GlTarget? _off;

        /// <summary>The shadow map, allocated on the first frame with a shadow light.</summary>
        private GlTarget? _shadow;

        /// <summary>Creates the context's resources for a model (<c>mjr_makeContext</c>).</summary>
        /// <param name="m">The model.</param>
        internal MujocoFrame(MjModel m) {
            _m = m;
            _meshes = new MujocoMeshes(m);
            _textures = new GlTexture?[m.ntex];
        }

        /// <summary>A texture, decoded from its embedded PNG the first time it is needed.</summary>
        /// <param name="id">The texture.</param>
        /// <returns>The texture, or <see langword="null"/> when the model recorded no pixels for it.</returns>
        private GlTexture? Texture(int id) {
            if (_textures[id] != null)
                return _textures[id];
            string? png = _m.tex_png[id];
            if (png == null)
                return null;
            byte[] rgb = EnvAssets.Load("mujoco/" + png).ToRgbBytes();
            return _textures[id] = new GlTexture(_m.tex_type[id], _m.tex_width[id], _m.tex_height[id], rgb);
        }

        /// <summary>Renders a scene into a <paramref name="width"/>×<paramref name="height"/> frame (<c>mjr_render</c>).</summary>
        /// <param name="scn">The scene.</param>
        /// <param name="width">The frame's width.</param>
        /// <param name="height">The frame's height.</param>
        /// <exception cref="InvalidOperationException">The camera's near plane is degenerate while there are geoms to draw.</exception>
        internal void Render(MujocoScene scn, int width, int height) {
            int samples = _m.vis_offsamples > 1 ? 4 : 1;
            if (_off == null || _off.Width != width || _off.Height != height || _off.Samples != samples)
                _off = new GlTarget(width, height, samples, color: true, stencil: true);
            GlCamera cam = AverageCamera(scn.Camera[0], scn.Camera[1]);
            if (cam.FrustumNear < 1e-15) {
                if (scn.Geoms.Count > 0)
                    throw new InvalidOperationException("mjvScene frustum_near too small in mjr_render");
                return;
            }

            // the lights the fixed-function pipeline draws (directional and spot), at most eight
            var lightIds = new List<int>();
            for (int i = 0; i < scn.Lights.Count && lightIds.Count < 8; i++) {
                if (scn.Lights[i].Type is 0 or 1)
                    lightIds.Add(i);
            }

            double[] hposD = new double[3], hfwdD = new double[3];
            scn.CameraInModel(hposD, hfwdD);
            float[] headpos = [(float)hposD[0], (float)hposD[1], (float)hposD[2]];

            // transparent geoms, sorted by distance from the camera; a plane seen from behind is one
            var order = new List<int>();
            for (int i = 0; i < scn.Geoms.Count; i++) {
                SceneGeom g = scn.Geoms[i];
                if (g.Rgba[3] < 0.995 || (g.Type == 0 && IsBehind(headpos, g.Pos, g.Mat))) {
                    order.Add(i);
                    g.Transparent = true;
                    float dx = g.Pos[0] - headpos[0], dy = g.Pos[1] - headpos[1], dz = g.Pos[2] - headpos[2];
                    g.CamDist = MathF.Sqrt(dx * dx + dy * dy + dz * dz) - Rbound(g);
                    if (g.Type == 0)
                        g.CamDist = 1e10f;
                } else {
                    g.Transparent = false;
                }
            }

            // a stable sort, as mjSORT's merge sort is
            order.Sort((x, y) => {
                int c = scn.Geoms[x].CamDist.CompareTo(scn.Geoms[y].CamDist);
                return c != 0 ? c : x.CompareTo(y);
            });
            int reflective = -1;
            for (int i = 0; i < scn.Geoms.Count; i++) {
                SceneGeom g = scn.Geoms[i];
                if (g.Type is 0 or 6 && !g.Transparent && g.Reflectance > 0) {
                    reflective = i;
                    break;
                }
            }

            InitGl(scn, lightIds);
            _gl.Bind(_off);
            _gl.Clear(color: true, depth: true, stencil: false);
            SetView(cam, width, height, out float[] camProject, out float[] camView);

            if (scn.Flags[MujocoScene.Reflection] && reflective >= 0)
                ReflectionPass(scn, lightIds, headpos, order, reflective);

            // regular pass: the lights that cast no shadow, the reflecting geom added onto its reflection, then the rest
            bool shadows = scn.Flags[MujocoScene.Shadow];
            for (int i = 0; i < lightIds.Count; i++) {
                SceneLight l = scn.Lights[lightIds[i]];
                AdjustLight(l, i);
                _gl.Lights[i].Enabled = !l.CastShadow || !shadows;
            }

            _gl.Blend = true;
            _gl.BlendSrc = GlBlend.One;
            _gl.BlendDst = GlBlend.One;
            if (reflective >= 0)
                RenderGeom(scn, scn.Geoms[reflective], Normal, headpos);
            _gl.Blend = false;
            for (int i = 0; i < scn.Geoms.Count; i++) {
                if (!scn.Geoms[i].Transparent && i != reflective)
                    RenderGeom(scn, scn.Geoms[i], Normal, headpos);
            }

            foreach (GlLight l in _gl.Lights)
                l.Enabled = false;

            if (shadows)
                ShadowPasses(scn, lightIds, headpos, cam, camProject, camView);

            if (scn.Flags[MujocoScene.Skybox])
                SkyboxPass(scn, headpos, cam);

            // transparent geoms: front to back, then back to front, without depth writes
            for (int i = 0; i < lightIds.Count; i++)
                _gl.Lights[i].Enabled = true;
            _gl.DepthMask = false;
            _gl.Blend = true;
            _gl.BlendSrc = GlBlend.SrcAlpha;
            _gl.BlendDst = GlBlend.OneMinusSrcAlpha;
            foreach (int i in order)
                RenderGeom(scn, scn.Geoms[i], Normal, headpos);
            for (int i = order.Count - 1; i >= 0; i--)
                RenderGeom(scn, scn.Geoms[order[i]], Normal, headpos);
            _gl.DepthMask = true;
            _gl.Blend = false;
            foreach (GlLight l in _gl.Lights)
                l.Enabled = false;
        }

        /// <summary>
        ///     Reads the frame back (<c>mjr_readPixels</c> through the resolve blit): each pixel's samples averaged, and the
        ///     depth of the first sample turned back to 0 at the near plane (MuJoCo's <c>mjDEPTH_ZERONEAR</c>); rows top
        ///     first, as Gymnasium's flip leaves them.
        /// </summary>
        /// <param name="rgb">The RGB bytes, <c>height × width × 3</c>, written when not <see langword="null"/>.</param>
        /// <param name="depth">The depths, <c>height × width</c>, written when not <see langword="null"/>.</param>
        /// <exception cref="InvalidOperationException">Nothing was rendered yet.</exception>
        internal void ReadPixels(byte[]? rgb, float[]? depth) {
            GlTarget off = _off ?? throw new InvalidOperationException("Nothing was rendered yet.");
            int w = off.Width, h = off.Height, ns = off.Samples;
            for (int row = 0; row < h; row++) {
                int y = h - 1 - row;
                for (int x = 0; x < w; x++) {
                    int pixel = y * w + x, o = row * w + x;
                    if (rgb != null) {
                        for (int ch = 0; ch < 3; ch++) {
                            int sum = 0;
                            for (int s = 0; s < ns; s++)
                                sum += off.Color![4 * (pixel * ns + s) + ch];
                            // the average of the samples' levels as the GPU's resolve blit rounds it: a quarter up
                            // from three, a half down (measured on exact per-sample values: 99.3% of 6,350 pixels whose
                            // samples differ, every pixel whose samples agree; (sum + 2) / 4 got 31% of the former)
                            rgb[3 * o + ch] = (byte)(ns == 1 ? sum : (sum + ns / 2 - 1) / ns);
                        }
                    }

                    if (depth != null)
                        depth[o] = 1.0f - off.Depth[pixel * ns];
                }
            }
        }

        /// <summary>The OpenGL state <c>mjr_render</c> starts from (<c>initGL3</c>, <c>initLights</c>).</summary>
        /// <param name="scn">The scene: its flags and lights.</param>
        /// <param name="lightIds">The scene lights the pipeline draws, by slot.</param>
        private void InitGl(MujocoScene scn, List<int> lightIds) {
            GlContext gl = _gl;
            gl.Lighting = true;
            gl.Fog = scn.Flags[MujocoScene.Fog];
            gl.FogStart = (float)(_m.extent * _m.vis_fogstart);
            gl.FogEnd = (float)(_m.extent * _m.vis_fogend);
            gl.FogR = _m.vis_rgba_fog[0];
            gl.FogG = _m.vis_rgba_fog[1];
            gl.FogB = _m.vis_rgba_fog[2];
            gl.Blend = false;
            gl.DepthTest = true;
            gl.DepthMask = true;
            gl.CullFace = scn.Flags[MujocoScene.CullFace];
            gl.CullFront = false;
            gl.FrontCcw = true;
            gl.ColorMask = true;
            gl.StencilTest = false;
            gl.AlphaTest = false;
            gl.ClipPlane = false;
            gl.PolygonOffset = false;
            gl.ColorMaterial = true;
            gl.MatSpecular = 1;
            gl.Texture = null;
            gl.ShadowMap = null;
            gl.TextureMatrix = GlMath.Identity();
            gl.GlobalAmbient = lightIds.Count > 0 ? 0 : 0.3f;
            for (int i = 0; i < 8; i++) {
                GlLight gl8 = gl.Lights[i];
                gl8.Enabled = false;
                if (i >= lightIds.Count)
                    continue;
                SceneLight l = scn.Lights[lightIds[i]];
                (gl8.Ar, gl8.Ag, gl8.Ab) = (l.Ambient[0], l.Ambient[1], l.Ambient[2]);
                (gl8.Dr, gl8.Dg, gl8.Db) = (l.Diffuse[0], l.Diffuse[1], l.Diffuse[2]);
                (gl8.Sr, gl8.Sg, gl8.Sb) = (l.Specular[0], l.Specular[1], l.Specular[2]);
                if (l.Type == 1) {
                    gl8.SpotExponent = 0;
                    gl8.SpotCutoff = 180;
                    (gl8.K0, gl8.K1, gl8.K2) = (1, 0, 0);
                } else {
                    gl8.SpotExponent = l.Exponent;
                    gl8.SpotCutoff = l.Cutoff;
                    (gl8.K0, gl8.K1, gl8.K2) = (l.Attenuation[0], l.Attenuation[1], l.Attenuation[2]);
                }
            }
        }

        /// <summary>The projection and view of the averaged eye (<c>setView</c> for a mono frame).</summary>
        /// <param name="cam">The averaged camera.</param>
        /// <param name="width">The viewport's width.</param>
        /// <param name="height">The viewport's height.</param>
        /// <param name="camProject">The projection, written.</param>
        /// <param name="camView">The view, written.</param>
        private void SetView(GlCamera cam, int width, int height, out float[] camProject, out float[] camView) {
            float halfwidth = cam.FrustumWidth != 0 ? cam.FrustumWidth
                : 0.5f * width / height * (cam.FrustumTop - cam.FrustumBottom);
            float[] proj = cam.Orthographic
                ? GlMath.Ortho(cam.FrustumCenter - halfwidth, cam.FrustumCenter + halfwidth, cam.FrustumBottom, cam.FrustumTop, cam.FrustumNear, cam.FrustumFar)
                : GlMath.Frustum(cam.FrustumCenter - halfwidth, cam.FrustumCenter + halfwidth, cam.FrustumBottom, cam.FrustumTop, cam.FrustumNear, cam.FrustumFar);
            camProject = GlMath.Multiply(GlMath.ReverseZ(), proj);
            camView = GlMath.LookAt(cam.Pos, cam.Forward, cam.Up);
            _gl.Projection = camProject;
            _gl.Modelview = camView;
        }

        /// <summary>
        ///     The averaged eye of the stereo pair (<c>mjv_averageCamera</c>): positions averaged, gazes added and normalized,
        ///     the up direction made orthogonal to the gaze, the frusta averaged.
        /// </summary>
        /// <param name="a">The left eye.</param>
        /// <param name="b">The right eye.</param>
        /// <returns>The averaged camera.</returns>
        private static GlCamera AverageCamera(GlCamera a, GlCamera b) {
            var c = new GlCamera();
            double[] pos = new double[3], fwd = new double[3], up = new double[3];
            for (int k = 0; k < 3; k++) {
                pos[k] = ((double)a.Pos[k] + b.Pos[k]) * 0.5;
                fwd[k] = (double)a.Forward[k] + b.Forward[k];
                up[k] = (double)a.Up[k] + b.Up[k];
            }

            Normalize3(fwd);
            double projection = up[0] * fwd[0] + up[1] * fwd[1] + up[2] * fwd[2];
            for (int k = 0; k < 3; k++)
                up[k] -= projection * fwd[k];
            Normalize3(up);
            for (int k = 0; k < 3; k++) {
                c.Pos[k] = (float)pos[k];
                c.Forward[k] = (float)fwd[k];
                c.Up[k] = (float)up[k];
            }

            c.FrustumBottom = 0.5f * (a.FrustumBottom + b.FrustumBottom);
            c.FrustumTop = 0.5f * (a.FrustumTop + b.FrustumTop);
            c.FrustumCenter = 0.5f * (a.FrustumCenter + b.FrustumCenter);
            c.FrustumWidth = 0.5f * (a.FrustumWidth + b.FrustumWidth);
            c.FrustumNear = 0.5f * (a.FrustumNear + b.FrustumNear);
            c.FrustumFar = 0.5f * (a.FrustumFar + b.FrustumFar);
            c.Orthographic = a.Orthographic;
            return c;

            static void Normalize3(double[] v) {
                double n = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
                if (n < 1e-15) {
                    v[0] = 1;
                    v[1] = v[2] = 0;
                    return;
                }

                v[0] /= n;
                v[1] /= n;
                v[2] /= n;
            }
        }

        /// <summary>Whether the head is behind a geom's xy plane (<c>isBehind</c>).</summary>
        /// <param name="headpos">The head.</param>
        /// <param name="pos">The geom's position.</param>
        /// <param name="mat">The geom's orientation, row-major.</param>
        /// <returns>True when the head is on the plane's negative side.</returns>
        private static bool IsBehind(float[] headpos, float[] pos, float[] mat) =>
            (headpos[0] - pos[0]) * mat[2] + (headpos[1] - pos[1]) * mat[5] + (headpos[2] - pos[2]) * mat[8] < 0.0f;

        /// <summary>A geom's bounding radius for sorting (<c>mjv_rbound</c>): the model's, or the scaled shape's.</summary>
        /// <param name="g">The geom.</param>
        /// <returns>The radius.</returns>
        private static float Rbound(SceneGeom g) {
            if (g.ModelRbound > 0)
                return g.ModelRbound;
            float[] s = g.Size;
            return g.Type switch {
                2 => s[0],
                3 => s[0] + s[2],
                5 => MathF.Sqrt(s[0] * s[0] + s[2] * s[2]),
                _ => MathF.Max(s[0], MathF.Max(s[1], s[2])),
            };
        }

        /// <summary>Places light <paramref name="n"/> by the current modelview (<c>adjustLight</c>).</summary>
        /// <param name="l">The scene light.</param>
        /// <param name="n">The pipeline's slot.</param>
        private void AdjustLight(SceneLight l, int n) {
            if (l.Type == 1) {
                _gl.SetLightPosition(n, -l.Dir[0], -l.Dir[1], -l.Dir[2], 0);
            } else if (l.Type == 0) {
                _gl.SetSpotDirection(n, l.Dir[0], l.Dir[1], l.Dir[2]);
                _gl.SetLightPosition(n, l.Pos[0], l.Pos[1], l.Pos[2], 1);
            }
        }

        /// <summary>
        ///     The reflection of the scene in the reflective plane (or a box's top), masked by the plane through the stencil
        ///     and clipped to the plane's far side, each geom's colour scaled by the reflectance.
        /// </summary>
        /// <param name="scn">The scene.</param>
        /// <param name="lightIds">The drawn lights.</param>
        /// <param name="headpos">The head.</param>
        /// <param name="order">The sorted transparent geoms.</param>
        /// <param name="reflective">The reflecting geom.</param>
        private void ReflectionPass(MujocoScene scn, List<int> lightIds, float[] headpos, List<int> order, int reflective) {
            SceneGeom g = scn.Geoms[reflective];
            if (g.Type == 6) {
                // a box reflects in a plane matching its +z face
                var plane = new SceneGeom { Type = 0, DataId = g.DataId, Reflectance = g.Reflectance, MatId = g.MatId, TexId = g.TexId };
                Array.Copy(g.Size, plane.Size, 3);
                Array.Copy(g.Mat, plane.Mat, 9);
                Array.Copy(g.Rgba, plane.Rgba, 4);
                plane.Pos[0] = g.Pos[0] + g.Size[2] * g.Mat[2];
                plane.Pos[1] = g.Pos[1] + g.Size[2] * g.Mat[5];
                plane.Pos[2] = g.Pos[2] + g.Size[2] * g.Mat[8];
                g = plane;
            }

            if (IsBehind(headpos, g.Pos, g.Mat))
                return;

            // the plane into the stencil: no depth test, no colour
            _gl.DepthTest = false;
            _gl.ColorMask = false;
            _gl.StencilTest = true;
            _gl.Clear(color: false, depth: false, stencil: true);
            _gl.StencilEqual = false;
            _gl.StencilRef = 1;
            _gl.StencilReplace = true;
            RenderGeom(scn, g, Normal, headpos);

            // the scene where the stencil is 1, mirrored, clipped below the plane
            _gl.DepthTest = true;
            _gl.ColorMask = true;
            _gl.StencilEqual = true;
            _gl.StencilReplace = false;
            _gl.ClipPlane = true;
            _gl.SetClipPlane([
                -g.Mat[2], -g.Mat[5], -g.Mat[8],
                (double)(g.Pos[0] * g.Mat[2] + g.Pos[1] * g.Mat[5] + g.Pos[2] * g.Mat[8]),
            ]);
            _gl.FrontCcw = false;
            _gl.PushMatrix();
            _gl.MultModelview(GlMath.Reflect(g.Pos, g.Mat));
            for (int j = 0; j < lightIds.Count; j++) {
                AdjustLight(scn.Lights[lightIds[j]], j);
                _gl.Lights[j].Enabled = true;
            }

            for (int j = 0; j < scn.Geoms.Count; j++) {
                if (!scn.Geoms[j].Transparent && j != reflective)
                    RenderGeomReflection(scn, scn.Geoms[j], g.Reflectance, headpos);
            }

            _gl.DepthMask = false;
            _gl.Blend = true;
            _gl.BlendSrc = GlBlend.SrcAlpha;
            _gl.BlendDst = GlBlend.OneMinusSrcAlpha;
            foreach (int j in order) {
                if (j != reflective)
                    RenderGeomReflection(scn, scn.Geoms[j], g.Reflectance, headpos);
            }

            for (int k = order.Count - 1; k >= 0; k--) {
                if (order[k] != reflective)
                    RenderGeomReflection(scn, scn.Geoms[order[k]], g.Reflectance, headpos);
            }

            _gl.DepthMask = true;
            _gl.Blend = false;
            foreach (GlLight l in _gl.Lights)
                l.Enabled = false;
            _gl.StencilTest = false;
            _gl.ClipPlane = false;
            _gl.PopMatrix();
            _gl.FrontCcw = true;
        }

        /// <summary>Draws a geom in the reflection, its colour scaled by the reflectance (<c>renderGeomReflection</c>).</summary>
        /// <param name="scn">The scene.</param>
        /// <param name="g">The geom.</param>
        /// <param name="reflectance">The mirror's reflectance.</param>
        /// <param name="headpos">The head.</param>
        private void RenderGeomReflection(MujocoScene scn, SceneGeom g, float reflectance, float[] headpos) {
            float r = g.Rgba[0], gg = g.Rgba[1], b = g.Rgba[2];
            g.Rgba[0] *= reflectance;
            g.Rgba[1] *= reflectance;
            g.Rgba[2] *= reflectance;
            RenderGeom(scn, g, Normal, headpos);
            g.Rgba[0] = r;
            g.Rgba[1] = gg;
            g.Rgba[2] = b;
        }

        /// <summary>
        ///     Each shadow light's pass: the scene's depth from the light into the shadow map (polygon offset against acne),
        ///     then the opaque geoms again with only that light, added where the map says the light reaches.
        /// </summary>
        /// <param name="scn">The scene.</param>
        /// <param name="lightIds">The drawn lights.</param>
        /// <param name="headpos">The head.</param>
        /// <param name="cam">The averaged camera (its clipping distances bound the light's view).</param>
        /// <param name="camProject">The camera's projection, restored after each map.</param>
        /// <param name="camView">The camera's view, restored after each map.</param>
        private void ShadowPasses(MujocoScene scn, List<int> lightIds, float[] headpos, GlCamera cam, float[] camProject, float[] camView) {
            int size = _m.vis_shadowsize;
            if (size <= 0)
                return;
            for (int i = 0; i < lightIds.Count; i++) {
                SceneLight l = scn.Lights[lightIds[i]];
                if (!l.CastShadow)
                    continue;
                _shadow ??= new GlTarget(size, size, 1, color: false, stencil: false, depth24: true);
                Span<float> up = stackalloc float[3];
                GlMath.OrthoVec(up, l.Dir);
                float shadowClip = (float)(_m.extent * _m.vis_shadowclip);
                float[] proj = l.Type == 1
                    ? GlMath.Ortho(-shadowClip, shadowClip, -shadowClip, shadowClip, cam.FrustumNear, cam.FrustumFar)
                    : Perspective(Math.Min(2 * l.Cutoff * _m.vis_shadowscale, 160), 1, cam.FrustumNear, cam.FrustumFar);
                float[] lightProject = GlMath.Multiply(GlMath.ReverseZ(), proj);
                float[] lightView = GlMath.LookAt(l.Pos, l.Dir, up);

                // the depth map from the light
                GlTarget off = _off!;
                _gl.Bind(_shadow);
                _gl.Clear(color: false, depth: true, stencil: false);
                _gl.Viewport(1, 1, size - 2, size - 2);
                _gl.Projection = lightProject;
                _gl.Modelview = lightView;
                _gl.Lighting = false;
                _gl.ColorMask = false;
                bool cull = _gl.CullFace;
                _gl.CullFace = false;
                _gl.PolygonOffset = true;
                _gl.OffsetFactor = -1.5f;
                _gl.OffsetUnits = -4.0f;
                foreach (SceneGeom g in scn.Geoms)
                    RenderGeom(scn, g, ShadowCast, headpos);

                _gl.Bind(off);
                _gl.CullFace = cull;
                _gl.PolygonOffset = false;
                _gl.Lighting = true;
                _gl.ColorMask = true;
                _gl.Projection = camProject;
                _gl.Modelview = camView;

                // the camera-to-light mapping, through the bias that maps the light's clip space to the map
                float[] bias = [0.5f, 0, 0, 0, 0, 0.5f, 0, 0, 0, 0, 1, 0, 0.5f, 0.5f, 0, 1];
                float[] texture = GlMath.Multiply(bias, GlMath.Multiply(lightProject, lightView));
                float[] rows = new float[16];
                for (int r = 0; r < 4; r++) {
                    for (int c = 0; c < 4; c++)
                        rows[4 * r + c] = texture[r + 4 * c];
                }

                _gl.ShadowMap = _shadow;
                _gl.SetShadowPlanes(rows);
                _gl.AlphaTest = true;
                _gl.Blend = true;
                _gl.BlendSrc = GlBlend.One;
                _gl.BlendDst = GlBlend.One;
                _gl.Lights[i].Enabled = true;
                foreach (SceneGeom g in scn.Geoms) {
                    if (!g.Transparent)
                        RenderGeom(scn, g, ShadowMapMode, headpos);
                }

                _gl.Lights[i].Enabled = false;
                _gl.Blend = false;
                _gl.AlphaTest = false;
                _gl.ShadowMap = null;
            }
        }

        /// <summary><c>mjr_perspective</c>: a symmetric frustum from a field of view, built in double.</summary>
        /// <param name="fovy">The vertical field of view in degrees.</param>
        /// <param name="aspect">Width over height.</param>
        /// <param name="znear">The near distance.</param>
        /// <param name="zfar">The far distance.</param>
        /// <returns>A new matrix.</returns>
        private static float[] Perspective(float fovy, float aspect, float znear, float zfar) {
            double h = Math.Tan(fovy / 360.0 * Math.PI) * znear, w = h * aspect;
            return GlMath.Frustum(-w, w, -h, h, znear, zfar);
        }

        /// <summary>
        ///     The first skybox texture on a closed cylinder around the head, scaled to just under the far plane's reach,
        ///     then the haze's band where the first infinite plane meets it.
        /// </summary>
        /// <param name="scn">The scene.</param>
        /// <param name="headpos">The head.</param>
        /// <param name="cam">The averaged camera.</param>
        private void SkyboxPass(MujocoScene scn, float[] headpos, GlCamera cam) {
            for (int t = 0; t < _m.ntex; t++) {
                if (_m.tex_type[t] != 2)
                    continue;
                GlTexture? sky = Texture(t);
                if (sky == null)
                    return;
                _gl.Lighting = false;
                _gl.ColorMaterial = false;
                _gl.CullFront = true;
                _gl.PushMatrix();
                _gl.MultModelview(GlMath.Translate(headpos[0], headpos[1], headpos[2]));
                float skyboxdst = cam.FrustumFar * 0.70f;
                _gl.MultModelview(GlMath.Scale(skyboxdst, skyboxdst, skyboxdst));

                // the skybox's mapping: rotated 90 degrees about x
                _gl.Texture = sky;
                _gl.TexGen = true;
                _gl.TexPlanes = [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0];
                (_gl.Cr, _gl.Cg, _gl.Cb, _gl.Ca) = (1, 1, 1, 1);
                _gl.DrawMesh(_meshes.Cylinder);
                _gl.Texture = null;
                _gl.TexGen = false;

                if (scn.Flags[MujocoScene.Haze]) {
                    foreach (SceneGeom g in scn.Geoms) {
                        if (g.Type != 0 || g.Size[0] != 0 || g.Size[1] != 0)
                            continue;
                        float[] mat3 = g.Mat;
                        float elevation = ((headpos[0] - g.Pos[0]) * mat3[2] + (headpos[1] - g.Pos[1]) * mat3[5] +
                                           (headpos[2] - g.Pos[2]) * mat3[8]) / skyboxdst;
                        if (elevation < 0)
                            break;
                        _gl.MultModelview([mat3[0], mat3[3], mat3[6], 0, mat3[1], mat3[4], mat3[7], 0, mat3[2], mat3[5], mat3[8], 0, 0, 0, 0, 1]);
                        _gl.MultModelview(GlMath.Translate(0, 0, -elevation));
                        _gl.MultModelview(GlMath.Scale(1, 1, elevation));
                        _gl.Blend = true;
                        _gl.BlendSrc = GlBlend.SrcAlpha;
                        _gl.BlendDst = GlBlend.OneMinusSrcAlpha;
                        _gl.DrawMesh(_meshes.Haze);
                        _gl.Blend = false;
                        break;
                    }
                }

                _gl.PopMatrix();
                _gl.Lighting = true;
                _gl.ColorMaterial = true;
                _gl.CullFront = false;
                return;
            }
        }

        /// <summary>
        ///     Draws one geom (<c>renderGeom</c>): its texture and material, then its builtin mesh scaled and placed; in the
        ///     shadow map's pass without material, and not at all when it casts no shadow.
        /// </summary>
        /// <param name="scn">The scene (its options).</param>
        /// <param name="g">The geom.</param>
        /// <param name="mode">Normal, the shadow light's pass, or the depth map.</param>
        /// <param name="headpos">The head.</param>
        private void RenderGeom(MujocoScene scn, SceneGeom g, int mode, float[] headpos) {
            if (mode == ShadowCast && (!g.CastShadow || g.Category == 4))
                return;
            bool behind = IsBehind(headpos, g.Pos, g.Mat);
            bool textured = g.MatId >= 0 && mode != ShadowCast;
            if (textured)
                SetTexture(g, true);

            float alpha = g.Rgba[3];
            if (g.Type == 0 && behind && mode == Normal)
                alpha *= 0.3f;
            if (mode == Normal)
                (_gl.Er, _gl.Eg, _gl.Eb) = (g.Emission * g.Rgba[0], g.Emission * g.Rgba[1], g.Emission * g.Rgba[2]);
            else
                (_gl.Er, _gl.Eg, _gl.Eb) = (0, 0, 0);
            (_gl.Cr, _gl.Cg, _gl.Cb, _gl.Ca) = (g.Rgba[0], g.Rgba[1], g.Rgba[2], alpha);
            _gl.Shininess = g.Shininess * 128.0f;
            _gl.MatSpecular = g.Specular;

            _gl.PushMatrix();
            _gl.MultModelview(GlMath.Translate(g.Pos[0], g.Pos[1], g.Pos[2]));
            float[] m = g.Mat;
            _gl.MultModelview([m[0], m[3], m[6], 0, m[1], m[4], m[7], 0, m[2], m[5], m[8], 0, 0, 0, 0, 1]);
            float[] s = g.Size;
            switch (g.Type) {
                case 0: {
                    // a plane seen from behind shows its back face
                    bool swapped = behind && !_gl.CullFront;
                    if (swapped)
                        _gl.CullFront = true;
                    if (g.DataId >= 0 && g.DataId < _meshes.Planes.Count) {
                        _gl.DrawMesh(_meshes.Planes[g.DataId]);
                    } else {
                        _gl.MultModelview(GlMath.Scale(s[0], s[1], 1.0f));
                        _gl.DrawMesh(_meshes.Plane);
                    }

                    if (swapped)
                        _gl.CullFront = false;
                    break;
                }
                case 2:
                case 4:
                    _gl.MultModelview(GlMath.Scale(s[0], s[1], s[2]));
                    _gl.DrawMesh(_meshes.Sphere);
                    break;
                case 3: {
                    float av = (float)(0.5 * (s[0] + s[1]));
                    _gl.MultModelview(GlMath.Scale(s[0], s[1], s[2]));
                    _gl.DrawMesh(_meshes.CylinderOpen);
                    _gl.MultModelview(GlMath.Scale(1, 1, av / s[2]));
                    _gl.MultModelview(GlMath.Translate(0, 0, (s[2] - av) / av));
                    _gl.DrawMesh(_meshes.SphereTop);
                    _gl.MultModelview(GlMath.Translate(0, 0, 2 * (av - s[2]) / av));
                    _gl.DrawMesh(_meshes.SphereBottom);
                    break;
                }
                case 5:
                    _gl.MultModelview(GlMath.Scale(s[0], s[1], s[2]));
                    _gl.DrawMesh(_meshes.Cylinder);
                    break;
                case 6:
                    _gl.MultModelview(GlMath.Scale(s[0], s[1], s[2]));
                    _gl.DrawMesh(_meshes.Box);
                    break;
            }

            _gl.PopMatrix();
            if (textured)
                SetTexture(g, false);
        }

        /// <summary>
        ///     A geom's texture on unit 0 (<c>settexture</c> for <c>mjtexREGULAR</c>): a 2D texture through the mesh's own
        ///     coordinates scaled by the repetition (an infinite plane's shifted by half), or a cube texture through
        ///     object-linear planes (scaled by the size when uniform).
        /// </summary>
        /// <param name="g">The geom.</param>
        /// <param name="on">Enable, or disable and reset the texture matrix.</param>
        private void SetTexture(SceneGeom g, bool on) {
            int texid = g.TexId;
            if (!on || texid < 0) {
                _gl.Texture = null;
                _gl.TexGen = false;
                _gl.TextureMatrix = GlMath.Identity();
                return;
            }

            GlTexture? tex = Texture(texid);
            if (tex == null)
                return;
            _gl.Texture = tex;
            if (!tex.IsCube) {
                // the builtin shapes carry their own texture coordinates
                float scl0 = g.TexRepeat[0] > 0 ? g.TexRepeat[0] : 1.0f, scl1 = g.TexRepeat[1] > 0 ? g.TexRepeat[1] : 1.0f;
                if (g.TexUniform) {
                    if (g.Size[0] > 0)
                        scl0 *= g.Size[0];
                    if (g.Size[1] > 0)
                        scl1 *= g.Size[1];
                }

                float[] tm = GlMath.Identity();
                if (g.Type == 0 && (g.Size[0] <= 0 || g.Size[1] <= 0))
                    tm = GlMath.Multiply(tm, GlMath.Translate(-0.5f, -0.5f, 0.0f));
                _gl.TextureMatrix = GlMath.Multiply(tm, GlMath.Scale(scl0, scl1, 1.0f));
                _gl.TexGen = false;
                return;
            }

            _gl.TexGen = true;
            _gl.TexPlanes = [
                g.TexUniform ? g.Size[0] : 1, 0, 0, 0,
                0, g.TexUniform ? g.Size[1] : 1, 0, 0,
                0, 0, g.TexUniform ? g.Size[2] : 1, 0,
            ];
        }
    }
}
