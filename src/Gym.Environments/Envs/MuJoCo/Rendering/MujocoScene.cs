using System;
using System.Collections.Generic;
using Gym.MuJoCo;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     The abstract camera (<c>mjvCamera</c>): the free camera's look-at point, distance and angles, a tracked body, or
    ///     a model camera.
    /// </summary>
    /// <remarks>Defaults are <c>mjv_defaultCamera</c>'s; Gymnasium's viewer then sets the look-at point and the distance.</remarks>
    internal sealed class MjvCamera {
        /// <summary>The free camera (<c>mjCAMERA_FREE</c>).</summary>
        internal const int Free = 0;

        /// <summary>The camera tracking a body (<c>mjCAMERA_TRACKING</c>).</summary>
        internal const int Tracking = 1;

        /// <summary>A model camera (<c>mjCAMERA_FIXED</c>).</summary>
        internal const int Fixed = 2;

        /// <summary>The camera's type.</summary>
        internal int Type = Free;

        /// <summary>The model camera, for <see cref="Fixed"/>.</summary>
        internal int FixedCamId = -1;

        /// <summary>The tracked body, for <see cref="Tracking"/>.</summary>
        internal int TrackBodyId = -1;

        /// <summary>The point looked at.</summary>
        internal readonly double[] Lookat = new double[3];

        /// <summary>The distance from the look-at point.</summary>
        internal double Distance = 2;

        /// <summary>The azimuth in degrees.</summary>
        internal double Azimuth = 90;

        /// <summary>The elevation in degrees.</summary>
        internal double Elevation = -45;

        /// <summary>Whether the free camera projects orthographically.</summary>
        internal bool Orthographic;
    }

    /// <summary>One eye's OpenGL camera (<c>mjvGLCamera</c>), float32 as MuJoCo stores it.</summary>
    internal sealed class GlCamera {
        /// <summary>The eye's position.</summary>
        internal readonly float[] Pos = new float[3];

        /// <summary>The forward direction.</summary>
        internal readonly float[] Forward = new float[3];

        /// <summary>The up direction.</summary>
        internal readonly float[] Up = new float[3];

        /// <summary>The frustum's centre offset, half width (0: from the aspect ratio), top and bottom at the near plane.</summary>
        internal float FrustumCenter, FrustumWidth, FrustumTop, FrustumBottom;

        /// <summary>The near and far clipping distances.</summary>
        internal float FrustumNear, FrustumFar;

        /// <summary>Whether the projection is orthographic.</summary>
        internal bool Orthographic;
    }

    /// <summary>A geom of the scene (<c>mjvGeom</c>), float32 as MuJoCo stores it.</summary>
    internal sealed class SceneGeom {
        /// <summary>The geom type (MuJoCo's <c>mjtGeom</c>: 0 plane, 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box).</summary>
        internal int Type;

        /// <summary>The scaling of the unit mesh: a capsule's (radius, radius, half-length).</summary>
        internal readonly float[] Size = new float[3];

        /// <summary>The position.</summary>
        internal readonly float[] Pos = new float[3];

        /// <summary>The orientation, row-major 3×3.</summary>
        internal readonly float[] Mat = new float[9];

        /// <summary>The colour.</summary>
        internal readonly float[] Rgba = new float[4];

        /// <summary>The material's emission, specular intensity, shininess and reflectance.</summary>
        internal float Emission, Specular = 0.5f, Shininess = 0.5f, Reflectance;

        /// <summary>The material, or -1.</summary>
        internal int MatId = -1;

        /// <summary>The texture, or -1.</summary>
        internal int TexId = -1;

        /// <summary>Whether the texture repeats per unit length.</summary>
        internal bool TexUniform;

        /// <summary>The texture repetition.</summary>
        internal readonly float[] TexRepeat = new float[2];

        /// <summary>The plane's grid list (its index among the model's planes), or -1.</summary>
        internal int DataId = -1;

        /// <summary>Whether it casts shadows.</summary>
        internal bool CastShadow = true;

        /// <summary>Its category: 1 static, 2 dynamic, 4 decoration.</summary>
        internal int Category;

        /// <summary>Whether <c>mjr_render</c> draws it in the transparent pass.</summary>
        internal bool Transparent;

        /// <summary>Its distance to the camera, for sorting transparent geoms.</summary>
        internal float CamDist;

        /// <summary>The bounding sphere's radius in the model.</summary>
        internal float ModelRbound;
    }

    /// <summary>A light of the scene (<c>mjvLight</c>).</summary>
    internal sealed class SceneLight {
        /// <summary>MuJoCo's light type: 0 spot, 1 directional (only these two are drawn).</summary>
        internal int Type;

        /// <summary>Whether it is the headlight.</summary>
        internal bool Headlight;

        /// <summary>Whether it casts shadows.</summary>
        internal bool CastShadow;

        /// <summary>The position.</summary>
        internal readonly float[] Pos = new float[3];

        /// <summary>The direction.</summary>
        internal readonly float[] Dir = new float[3];

        /// <summary>The colours.</summary>
        internal readonly float[] Ambient = new float[3], Diffuse = new float[3], Specular = new float[3];

        /// <summary>A spot light's attenuation.</summary>
        internal readonly float[] Attenuation = [1, 0, 0];

        /// <summary>A spot light's cutoff and exponent.</summary>
        internal float Cutoff, Exponent;
    }

    /// <summary>
    ///     The scene MuJoCo's visualizer builds for a frame (<c>mjv_updateScene</c>): the model's visible geoms and sites
    ///     with their materials, the cameras, and the lights; kept from frame to frame as Gymnasium's viewer keeps its
    ///     <c>MjvScene</c>, because an infinite plane is re-centred on the previous frame's camera.
    /// </summary>
    /// <remarks>
    ///     Only what Gymnasium's default options draw: the decorations (contact forces, frames) belong to the human
    ///     window's toggles (DR-70). Reference:
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_vis_visualize.c">engine_vis_visualize.c</see>.
    /// </remarks>
    internal sealed class MujocoScene {
        /// <summary>The render flags' indices (<c>mjtRndFlag</c>).</summary>
        internal const int Shadow = 0, Reflection = 2, Skybox = 4, Fog = 5, Haze = 6, CullFace = 10;

        /// <summary>The geoms, in the order the visualizer adds them.</summary>
        internal readonly List<SceneGeom> Geoms = new();

        /// <summary>The lights: the headlight first when on, then the model's active lights.</summary>
        internal readonly List<SceneLight> Lights = new();

        /// <summary>The two eyes' cameras (left and right of the stereo pair); a mono frame averages them.</summary>
        internal readonly GlCamera[] Camera = [new(), new()];

        /// <summary>The render flags, <c>mjRNDSTRING</c>'s defaults: shadow, reflection, skybox, haze and face culling on.</summary>
        internal readonly bool[] Flags = [true, false, true, false, true, false, true, false, false, false, true, true];

        /// <summary>Which geom groups are drawn (groups 0, 1 and 2 by default).</summary>
        internal readonly bool[] GeomGroup = [true, true, true, false, false, false];

        /// <summary>Which site groups are drawn.</summary>
        internal readonly bool[] SiteGroup = [true, true, true, false, false, false];

        /// <summary>The visual flags (<c>mjVISSTRING</c>'s defaults), by <c>mjtVisFlag</c> index.</summary>
        internal readonly bool[] VisFlags = new bool[31];

        /// <summary>The visual flag of textures (<c>mjVIS_TEXTURE</c>).</summary>
        internal const int VisTexture = 1;

        /// <summary>The visual flag of transparent dynamic geoms (<c>mjVIS_TRANSPARENT</c>).</summary>
        internal const int VisTransparent = 18;

        /// <summary>The visual flag of static bodies (<c>mjVIS_STATIC</c>).</summary>
        internal const int VisStatic = 22;

        /// <summary>
        ///     The most geoms a frame holds (<c>maxgeom</c>, Gymnasium's <c>max_geom</c>): the model's geoms and sites past it
        ///     are not drawn, as MuJoCo's full buffer drops them.
        /// </summary>
        internal int MaxGeom = 1000;

        /// <summary>Whether the full-buffer warning was printed (<c>mjvScene.status</c>): once per scene, as MuJoCo does.</summary>
        private bool _fullWarned;

        /// <summary>The model cameras' global positions (<c>cam_xpos</c>), refreshed by <see cref="CamLight"/>.</summary>
        private double[] _camXpos = [];

        /// <summary>The model cameras' global orientations, row-major (<c>cam_xmat</c>).</summary>
        private double[] _camXmat = [];

        /// <summary>The model lights' global positions (<c>light_xpos</c>).</summary>
        private double[] _lightXpos = [];

        /// <summary>The model lights' global directions (<c>light_xdir</c>).</summary>
        private double[] _lightXdir = [];

        /// <summary>A scene with MuJoCo's default options.</summary>
        internal MujocoScene() {
            foreach (int on in new[] { 1, 7, 8, 13, 22, 23, 25, 27 })
                VisFlags[on] = true;
        }

        /// <summary>
        ///     Rebuilds the scene for the data's state (<c>mjv_updateScene</c> with <c>mjCAT_ALL</c>): geoms and sites first
        ///     (re-centring an infinite plane on the previous frame's camera), then the camera, then the lights (the headlight
        ///     follows the new camera).
        /// </summary>
        /// <param name="m">The model.</param>
        /// <param name="d">The data, after a forward pass.</param>
        /// <param name="cam">The abstract camera; a tracking camera's look-at point moves to the body.</param>
        internal void Update(MjModel m, MjData d, MjvCamera cam) {
            Geoms.Clear();
            CamLight(m, d);
            int catmask = 7;
            if (!VisFlags[VisStatic])
                catmask &= ~1;
            AddGeoms(m, d, catmask);
            AddSites(m, d, catmask);
            UpdateCamera(m, d, cam);
            MakeLights(m, d);
        }

        /// <summary>
        ///     The model cameras' and lights' global poses (<c>mj_camlight</c>), from the bodies' poses of the last forward
        ///     pass: fixed to their body, tracking it or its subtree's centre of mass with the reference orientation, or
        ///     aimed at a target body. The engine's step never needs them, so the renderer computes them for itself.
        /// </summary>
        /// <param name="m">The model.</param>
        /// <param name="d">The data.</param>
        private void CamLight(MjModel m, MjData d) {
            if (_camXpos.Length != 3 * m.ncam) {
                _camXpos = new double[3 * m.ncam];
                _camXmat = new double[9 * m.ncam];
            }

            if (_lightXpos.Length != 3 * m.nlight) {
                _lightXpos = new double[3 * m.nlight];
                _lightXdir = new double[3 * m.nlight];
            }

            Span<double> quat = stackalloc double[4];
            for (int i = 0; i < m.ncam; i++) {
                int id = m.cam_bodyid[i], id1 = m.cam_targetbodyid[i];
                Local2Global(d, id, m.cam_pos, 3 * i, _camXpos.AsSpan(3 * i, 3));
                Spatial.MulQuat(quat, d.xquat.AsSpan(4 * id, 4), m.cam_quat.AsSpan(4 * i, 4));
                Spatial.Quat2Mat(_camXmat.AsSpan(9 * i, 9), quat);
                switch (m.cam_mode[i]) {
                    case 1:
                    case 2:
                        // track the body or its subtree's centre of mass, keeping the reference orientation
                        Array.Copy(m.cam_mat0, 9 * i, _camXmat, 9 * i, 9);
                        for (int k = 0; k < 3; k++)
                            _camXpos[3 * i + k] = m.cam_mode[i] == 1
                                ? d.xpos[3 * id + k] + m.cam_pos0[3 * i + k]
                                : d.subtree_com[3 * id + k] + m.cam_poscom0[3 * i + k];
                        break;
                    case 3:
                    case 4:
                        if (id1 >= 0) {
                            double[] pos = new double[3];
                            for (int k = 0; k < 3; k++)
                                pos[k] = m.cam_mode[i] == 3 ? d.xpos[3 * id1 + k] : d.subtree_com[3 * id1 + k];
                            // z: from the target to the camera; x: orthogonal to z and to +z; y: z × x
                            double[] mt = new double[9];
                            for (int k = 0; k < 3; k++)
                                mt[6 + k] = _camXpos[3 * i + k] - pos[k];
                            Normalize3(mt.AsSpan(6, 3));
                            mt[3] = 0;
                            mt[4] = 0;
                            mt[5] = 1;
                            Cross(mt.AsSpan(0, 3), mt.AsSpan(3, 3), mt.AsSpan(6, 3));
                            Normalize3(mt.AsSpan(0, 3));
                            Cross(mt.AsSpan(3, 3), mt.AsSpan(6, 3), mt.AsSpan(0, 3));
                            Normalize3(mt.AsSpan(3, 3));
                            for (int r = 0; r < 3; r++) {
                                for (int c = 0; c < 3; c++)
                                    _camXmat[9 * i + 3 * r + c] = mt[3 * c + r];
                            }
                        }

                        break;
                }
            }

            for (int i = 0; i < m.nlight; i++) {
                int id = m.light_bodyid[i];
                Local2Global(d, id, m.light_pos, 3 * i, _lightXpos.AsSpan(3 * i, 3));
                Spatial.RotVecQuat(_lightXdir.AsSpan(3 * i, 3), m.light_dir.AsSpan(3 * i, 3), d.xquat.AsSpan(4 * id, 4));
                if (m.light_mode[i] is 1 or 2) {
                    Array.Copy(m.light_dir0, 3 * i, _lightXdir, 3 * i, 3);
                    for (int k = 0; k < 3; k++)
                        _lightXpos[3 * i + k] = m.light_mode[i] == 1
                            ? d.xpos[3 * id + k] + m.light_pos0[3 * i + k]
                            : d.subtree_com[3 * id + k] + m.light_poscom0[3 * i + k];
                } else if (m.light_mode[i] is 3 or 4 && m.light_targetbodyid[i] >= 0) {
                    int id1 = m.light_targetbodyid[i];
                    for (int k = 0; k < 3; k++)
                        _lightXdir[3 * i + k] = (m.light_mode[i] == 3 ? d.xpos[3 * id1 + k] : d.subtree_com[3 * id1 + k]) - _lightXpos[3 * i + k];
                    Normalize3(_lightXdir.AsSpan(3 * i, 3));
                }
            }

            static void Local2Global(MjData d, int body, double[] local, int offset, Span<double> result) {
                for (int r = 0; r < 3; r++) {
                    result[r] = d.xmat[9 * body + 3 * r] * local[offset] + d.xmat[9 * body + 3 * r + 1] * local[offset + 1] +
                                d.xmat[9 * body + 3 * r + 2] * local[offset + 2] + d.xpos[3 * body + r];
                }
            }

            static void Cross(Span<double> a, ReadOnlySpan<double> b, ReadOnlySpan<double> c) {
                a[0] = b[1] * c[2] - b[2] * c[1];
                a[1] = b[2] * c[0] - b[0] * c[2];
                a[2] = b[0] * c[1] - b[1] * c[0];
            }
        }

        /// <summary>MuJoCo's <c>mju_round</c>: the nearer integer, a tie going up.</summary>
        /// <param name="x">The value.</param>
        /// <returns>The integer, as a double.</returns>
        private static double RoundMujoco(double x) {
            double lower = Math.Floor(x), upper = Math.Ceiling(x);
            return x - lower < upper - x ? lower : upper;
        }

        /// <summary>A body's category (<c>bodycategory</c>): static when welded to the world.</summary>
        /// <param name="m">The model.</param>
        /// <param name="body">The body.</param>
        /// <returns>1 static, 2 dynamic.</returns>
        private static int BodyCategory(MjModel m, int body) => m.body_weldid[body] == 0 ? 1 : 2;

        /// <summary>
        ///     Sets a geom's type, size, pose and default material (<c>mjv_initGeom</c>): a sphere's one size on all three
        ///     axes, a capsule's or cylinder's radius on two.
        /// </summary>
        /// <param name="g">The geom.</param>
        /// <param name="type">The geom type.</param>
        /// <param name="size">The model's three sizes.</param>
        /// <param name="sizeOffset">Their offset.</param>
        /// <param name="pos">The global position.</param>
        /// <param name="posOffset">Its offset.</param>
        /// <param name="mat">The global orientation.</param>
        /// <param name="matOffset">Its offset.</param>
        private static void InitGeom(SceneGeom g, int type, double[] size, int sizeOffset, double[] pos, int posOffset, double[] mat, int matOffset) {
            g.Type = type;
            switch (type) {
                case 2:
                    g.Size[0] = g.Size[1] = g.Size[2] = (float)size[sizeOffset];
                    break;
                case 3:
                case 5:
                    g.Size[0] = g.Size[1] = (float)size[sizeOffset];
                    g.Size[2] = (float)size[sizeOffset + 1];
                    break;
                default:
                    for (int k = 0; k < 3; k++)
                        g.Size[k] = (float)size[sizeOffset + k];
                    break;
            }

            for (int k = 0; k < 3; k++)
                g.Pos[k] = (float)pos[posOffset + k];
            for (int k = 0; k < 9; k++)
                g.Mat[k] = (float)mat[matOffset + k];
            g.Rgba[0] = g.Rgba[1] = g.Rgba[2] = 0.5f;
            g.Rgba[3] = 1;
            g.DataId = g.MatId = g.TexId = -1;
            g.TexUniform = false;
            g.TexRepeat[0] = g.TexRepeat[1] = 0;
            g.Emission = 0;
            g.Specular = 0.5f;
            g.Shininess = 0.5f;
            g.Reflectance = 0;
            g.CastShadow = true;
            g.ModelRbound = 0;
        }

        /// <summary>
        ///     A geom's material (<c>setMaterial</c>): the material's colour and properties, the element's own colour when it
        ///     differs from the default grey or there is no material, the texture when textures are shown.
        /// </summary>
        /// <param name="m">The model.</param>
        /// <param name="g">The geom.</param>
        /// <param name="matid">The material, or -1.</param>
        /// <param name="rgba">The element's colours.</param>
        /// <param name="rgbaOffset">Its offset.</param>
        private void SetMaterial(MjModel m, SceneGeom g, int matid, float[] rgba, int rgbaOffset) {
            if (matid >= 0) {
                for (int k = 0; k < 4; k++)
                    g.Rgba[k] = m.mat_rgba[4 * matid + k];
                g.Emission = m.mat_emission[matid];
                g.Specular = m.mat_specular[matid];
                g.Shininess = m.mat_shininess[matid];
                g.Reflectance = m.mat_reflectance[matid];
            }

            float r = rgba[rgbaOffset], gg = rgba[rgbaOffset + 1], b = rgba[rgbaOffset + 2], a = rgba[rgbaOffset + 3];
            if (r != 0.5f || gg != 0.5f || b != 0.5f || a != 1.0f || matid < 0) {
                g.Rgba[0] = r;
                g.Rgba[1] = gg;
                g.Rgba[2] = b;
                g.Rgba[3] = a;
            }

            if (VisFlags[VisTexture] && matid >= 0) {
                g.MatId = matid;
                g.TexId = m.mat_texid[matid * MjModel.TexRoleCount + MjModel.TexRoleRgb];
                g.TexUniform = m.mat_texuniform[matid];
                g.TexRepeat[0] = m.mat_texrepeat[2 * matid];
                g.TexRepeat[1] = m.mat_texrepeat[2 * matid + 1];
            }

            if (VisFlags[VisTransparent] && g.Category == 2)
                g.Rgba[3] *= m.vis_alpha;
        }

        /// <summary>The model's visible geoms (<c>addGeomGeoms</c>), an infinite plane re-centred under the camera.</summary>
        /// <param name="m">The model.</param>
        /// <param name="d">The data.</param>
        /// <param name="catmask">The categories drawn.</param>
        private void AddGeoms(MjModel m, MjData d, int catmask) {
            int planeid = -1;
            for (int i = 0; i < m.ngeom; i++) {
                if (m.geom_type[i] == 0)
                    planeid++;
                int category = BodyCategory(m, m.geom_bodyid[i]);
                if ((category & catmask) == 0)
                    continue;
                if (!GeomGroup[Math.Clamp(m.geom_group[i], 0, 5)])
                    continue;
                var g = new SceneGeom { Category = category };
                InitGeom(g, m.geom_type[i], m.geom_size, 3 * i, d.geom_xpos, 3 * i, d.geom_xmat, 9 * i);
                g.ModelRbound = (float)m.geom_rbound[i];
                g.CastShadow = m.geom_castshadow[i];
                SetMaterial(m, g, m.geom_matid[i], m.geom_rgba, 4 * i);
                if (g.Rgba[3] == 0)
                    continue;
                if (m.geom_type[i] == 0) {
                    g.DataId = planeid;
                    double[] tmp = [d.geom_xpos[3 * i], d.geom_xpos[3 * i + 1], d.geom_xpos[3 * i + 2]];
                    if (m.geom_size[3 * i] <= 0 || m.geom_size[3 * i + 1] <= 0) {
                        // the camera of the previous update: geoms are added before the camera moves
                        double[] vec = new double[3];
                        for (int j = 0; j < 3; j++)
                            vec[j] = 0.5 * ((double)Camera[0].Pos[j] + Camera[1].Pos[j]) - d.geom_xpos[3 * i + j];
                        double zfar = m.vis_zfar * m.extent;
                        for (int k = 0; k < 2; k++) {
                            if (m.geom_size[3 * i + k] > 0)
                                continue;
                            int matid = m.geom_matid[i];
                            double sX = matid >= 0 && m.mat_texrepeat[2 * matid + k] > 0
                                ? 2 / (double)m.mat_texrepeat[2 * matid + k]
                                : 2.1 * zfar / (200 - 2);
                            // the axis is column k of the geom's frame
                            double ax0 = d.geom_xmat[9 * i + k], ax1 = d.geom_xmat[9 * i + 3 + k], ax2 = d.geom_xmat[9 * i + 6 + k];
                            double dX = vec[0] * ax0 + vec[1] * ax1 + vec[2] * ax2;
                            dX = 2 * sX * RoundMujoco(0.5 * dX / sX);
                            tmp[0] += ax0 * dX;
                            tmp[1] += ax1 * dX;
                            tmp[2] += ax2 * dX;
                        }
                    }

                    for (int k = 0; k < 3; k++)
                        g.Pos[k] = (float)tmp[k];
                }

                if (!TryAdd(g))
                    return;
            }
        }

        /// <summary>The model's visible sites (<c>addSiteGeoms</c>).</summary>
        /// <param name="m">The model.</param>
        /// <param name="d">The data.</param>
        /// <param name="catmask">The categories drawn.</param>
        private void AddSites(MjModel m, MjData d, int catmask) {
            for (int i = 0; i < m.nsite; i++) {
                int category = BodyCategory(m, m.site_bodyid[i]);
                if ((category & catmask) == 0)
                    continue;
                if (!SiteGroup[Math.Clamp(m.site_group[i], 0, 5)])
                    continue;
                var g = new SceneGeom { Category = category };
                InitGeom(g, m.site_type[i], m.site_size, 3 * i, d.site_xpos, 3 * i, d.site_xmat, 9 * i);
                SetMaterial(m, g, m.site_matid[i], m.site_rgba, 4 * i);
                if (g.Rgba[3] == 0)
                    continue;
                if (!TryAdd(g))
                    return;
            }
        }

        /// <summary>
        ///     Adds a geom when the scene has room for it (<c>acquireGeom</c>); a full scene drops it and, the first time,
        ///     prints MuJoCo's warning to the standard output as MuJoCo's default handler does (without its log file).
        /// </summary>
        /// <param name="g">The geom.</param>
        /// <returns>False when the scene was full: the caller stops adding, as MuJoCo's loops return on a full buffer.</returns>
        private bool TryAdd(SceneGeom g) {
            if (Geoms.Count >= MaxGeom) {
                if (!_fullWarned) {
                    Console.Out.WriteLine($"WARNING: Pre-allocated visual geom buffer is full. Increase maxgeom above {MaxGeom}.");
                    Console.Out.WriteLine();
                    _fullWarned = true;
                }

                return false;
            }

            Geoms.Add(g);
            return true;
        }

        /// <summary>
        ///     The cameras (<c>mjv_updateCamera</c>): the frustum (<c>mjv_cameraFrustum</c>), the eye's frame
        ///     (<c>mjv_cameraFrame</c>) and the two stereo eyes half the separation to each side.
        /// </summary>
        /// <param name="m">The model.</param>
        /// <param name="d">The data.</param>
        /// <param name="cam">The abstract camera.</param>
        /// <exception cref="ArgumentOutOfRangeException">A fixed camera or tracked body out of range.</exception>
        private void UpdateCamera(MjModel m, MjData d, MjvCamera cam) {
            if (cam.Type == MjvCamera.Tracking) {
                int bid = cam.TrackBodyId;
                if (bid < 0 || bid >= m.nbody)
                    throw new ArgumentOutOfRangeException(nameof(cam), "track body id is outside valid range");
                for (int k = 0; k < 3; k++)
                    cam.Lookat[k] = d.subtree_com[3 * bid + k];
            }

            // the frustum: field of view (or intrinsics) and the clipping distances
            double fovy;
            bool orthographic;
            float[]? intrinsic = null, sensorsize = null;
            int cid = cam.FixedCamId;
            if (cam.Type == MjvCamera.Fixed) {
                if (cid < 0 || cid >= m.ncam)
                    throw new ArgumentOutOfRangeException(nameof(cam), "fixed camera id is outside valid range");
                orthographic = m.cam_projection[cid] == 1;
                fovy = m.cam_fovy[cid];
                if (m.cam_sensorsize[2 * cid + 1] != 0) {
                    sensorsize = [m.cam_sensorsize[2 * cid], m.cam_sensorsize[2 * cid + 1]];
                    intrinsic = [m.cam_intrinsic[4 * cid], m.cam_intrinsic[4 * cid + 1], m.cam_intrinsic[4 * cid + 2], m.cam_intrinsic[4 * cid + 3]];
                }
            } else {
                orthographic = cam.Orthographic;
                fovy = m.vis_fovy;
            }

            float znear = (float)(m.vis_znear * m.extent);
            float[] zver = new float[2], zhor = new float[2];
            if (orthographic) {
                zver[0] = zver[1] = (float)(fovy / 2);
            } else if (intrinsic != null) {
                zhor[0] = znear / intrinsic[0] * (sensorsize![0] / 2f - intrinsic[2]);
                zhor[1] = znear / intrinsic[0] * (sensorsize[0] / 2f + intrinsic[2]);
                zver[0] = znear / intrinsic[1] * (sensorsize[1] / 2f - intrinsic[3]);
                zver[1] = znear / intrinsic[1] * (sensorsize[1] / 2f + intrinsic[3]);
            } else {
                zver[0] = zver[1] = (float)(znear * Math.Tan(fovy * Math.PI / 360.0));
            }

            float zfar = (float)(m.vis_zfar * m.extent);

            // the eye's frame
            double[] headpos = new double[3], forward = new double[3], up = new double[3], right = new double[3];
            CameraFrame(d, cam, headpos, forward, up, right);
            double ipd = cam.Type == MjvCamera.Fixed ? m.cam_ipd[cid] : m.vis_ipd;
            for (int view = 0; view < 2; view++) {
                GlCamera c = Camera[view];
                for (int i = 0; i < 3; i++) {
                    c.Pos[i] = (float)(headpos[i] + (view != 0 ? ipd : -ipd) * 0.5 * right[i]);
                    c.Forward[i] = (float)forward[i];
                    c.Up[i] = (float)up[i];
                }

                c.Orthographic = orthographic;
                c.FrustumTop = zver[0];
                c.FrustumBottom = -zver[1];
                c.FrustumCenter = (zhor[1] - zhor[0]) / 2;
                c.FrustumWidth = (zhor[1] + zhor[0]) / 2;
                c.FrustumNear = znear;
                c.FrustumFar = zfar;
            }
        }

        /// <summary>The eye's position and axes (<c>mjv_cameraFrame</c>).</summary>
        /// <param name="d">The data.</param>
        /// <param name="cam">The abstract camera.</param>
        /// <param name="headpos">The eye, written.</param>
        /// <param name="forward">Forward, written.</param>
        /// <param name="up">Up, written.</param>
        /// <param name="right">Right, written.</param>
        private void CameraFrame(MjData d, MjvCamera cam, double[] headpos, double[] forward, double[] up, double[] right) {
            if (cam.Type == MjvCamera.Fixed) {
                double[] mat = _camXmat;
                int c = cam.FixedCamId;
                forward[0] = -mat[9 * c + 2];
                forward[1] = -mat[9 * c + 5];
                forward[2] = -mat[9 * c + 8];
                up[0] = mat[9 * c + 1];
                up[1] = mat[9 * c + 4];
                up[2] = mat[9 * c + 7];
                right[0] = mat[9 * c + 0];
                right[1] = mat[9 * c + 3];
                right[2] = mat[9 * c + 6];
                for (int k = 0; k < 3; k++)
                    headpos[k] = _camXpos[3 * c + k];
                return;
            }

            double ca = Math.Cos(cam.Azimuth / 180.0 * Math.PI), sa = Math.Sin(cam.Azimuth / 180.0 * Math.PI);
            double ce = Math.Cos(cam.Elevation / 180.0 * Math.PI), se = Math.Sin(cam.Elevation / 180.0 * Math.PI);
            forward[0] = ce * ca;
            forward[1] = ce * sa;
            forward[2] = se;
            up[0] = -se * ca;
            up[1] = -se * sa;
            up[2] = ce;
            right[0] = sa;
            right[1] = -ca;
            right[2] = 0;
            for (int k = 0; k < 3; k++)
                headpos[k] = cam.Lookat[k] - cam.Distance * forward[k];
        }

        /// <summary>
        ///     The head's position and gaze in the model (<c>mjv_cameraInModel</c> with no model transformation): each eye's
        ///     frame through a quaternion and back, averaged.
        /// </summary>
        /// <param name="headpos">The head, written.</param>
        /// <param name="forward">The gaze, written (unit).</param>
        internal void CameraInModel(double[] headpos, double[] forward) {
            Array.Clear(headpos);
            Array.Clear(forward);
            for (int n = 0; n < 2; n++) {
                GlCamera c = Camera[n];
                double[] fwd = [c.Forward[0], c.Forward[1], c.Forward[2]], u = [c.Up[0], c.Up[1], c.Up[2]];
                Normalize3(fwd);
                Normalize3(u);
                double[] left = [u[1] * fwd[2] - u[2] * fwd[1], u[2] * fwd[0] - u[0] * fwd[2], u[0] * fwd[1] - u[1] * fwd[0]];
                Normalize3(left);
                double[] mat = [left[0], u[0], fwd[0], left[1], u[1], fwd[1], left[2], u[2], fwd[2]];
                double[] quat = Mat2Quat(mat);
                Span<double> modelmat = stackalloc double[9];
                Spatial.Quat2Mat(modelmat, quat);
                for (int k = 0; k < 3; k++)
                    headpos[k] += 0.5 * c.Pos[k];
                forward[0] += 0.5 * modelmat[2];
                forward[1] += 0.5 * modelmat[5];
                forward[2] += 0.5 * modelmat[8];
            }

            Normalize3(forward);
        }

        /// <summary>Normalizes a 3-vector in place (<c>mju_normalize3</c>: a tiny one becomes +x).</summary>
        /// <param name="v">The vector.</param>
        private static void Normalize3(double[] v) => Normalize3(v.AsSpan());

        /// <summary>Normalizes a 3-vector in place (<c>mju_normalize3</c>: a tiny one becomes +x).</summary>
        /// <param name="v">The vector.</param>
        private static void Normalize3(Span<double> v) {
            double norm = Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
            if (norm < 1e-15) {
                v[0] = 1;
                v[1] = v[2] = 0;
                return;
            }

            double inv = 1 / norm;
            v[0] *= inv;
            v[1] *= inv;
            v[2] *= inv;
        }

        /// <summary>A rotation matrix's unit quaternion (<c>mju_mat2Quat</c>).</summary>
        /// <param name="mat">The matrix, row-major.</param>
        /// <returns>The quaternion (w, x, y, z).</returns>
        private static double[] Mat2Quat(double[] mat) {
            var q = new double[4];
            if (mat[0] + mat[4] + mat[8] > 0) {
                q[0] = 0.5 * Math.Sqrt(1 + mat[0] + mat[4] + mat[8]);
                q[1] = 0.25 * (mat[7] - mat[5]) / q[0];
                q[2] = 0.25 * (mat[2] - mat[6]) / q[0];
                q[3] = 0.25 * (mat[3] - mat[1]) / q[0];
            } else if (mat[0] > mat[4] && mat[0] > mat[8]) {
                q[1] = 0.5 * Math.Sqrt(1 + mat[0] - mat[4] - mat[8]);
                q[0] = 0.25 * (mat[7] - mat[5]) / q[1];
                q[2] = 0.25 * (mat[1] + mat[3]) / q[1];
                q[3] = 0.25 * (mat[2] + mat[6]) / q[1];
            } else if (mat[4] > mat[8]) {
                q[2] = 0.5 * Math.Sqrt(1 - mat[0] + mat[4] - mat[8]);
                q[0] = 0.25 * (mat[2] - mat[6]) / q[2];
                q[1] = 0.25 * (mat[1] + mat[3]) / q[2];
                q[3] = 0.25 * (mat[5] + mat[7]) / q[2];
            } else {
                q[3] = 0.5 * Math.Sqrt(1 - mat[0] - mat[4] + mat[8]);
                q[0] = 0.25 * (mat[3] - mat[1]) / q[3];
                q[1] = 0.25 * (mat[2] + mat[6]) / q[3];
                q[2] = 0.25 * (mat[5] + mat[7]) / q[3];
            }

            double norm = Math.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
            for (int k = 0; k < 4; k++)
                q[k] /= norm;
            return q;
        }

        /// <summary>
        ///     The lights (<c>mjv_makeLights</c>): the headlight along the new camera's gaze when on, then each active model
        ///     light at its pose (<c>mj_camlight</c>'s).
        /// </summary>
        /// <param name="m">The model.</param>
        /// <param name="d">The data.</param>
        private void MakeLights(MjModel m, MjData d) {
            Lights.Clear();
            if (m.vis_headlight_active != 0) {
                var l = new SceneLight { Type = 1, Headlight = true };
                double[] hpos = new double[3], hfwd = new double[3];
                CameraInModel(hpos, hfwd);
                for (int k = 0; k < 3; k++) {
                    l.Pos[k] = (float)hpos[k];
                    l.Dir[k] = (float)hfwd[k];
                    l.Ambient[k] = m.vis_headlight_ambient[k];
                    l.Diffuse[k] = m.vis_headlight_diffuse[k];
                    l.Specular[k] = m.vis_headlight_specular[k];
                }

                Lights.Add(l);
            }

            for (int i = 0; i < m.nlight && Lights.Count < 100; i++) {
                if (!m.light_active[i])
                    continue;
                var l = new SceneLight { Type = m.light_type[i], CastShadow = m.light_castshadow[i] };
                if (l.Type == 0) {
                    for (int k = 0; k < 3; k++)
                        l.Attenuation[k] = m.light_attenuation[3 * i + k];
                    l.Exponent = m.light_exponent[i];
                    l.Cutoff = m.light_cutoff[i];
                }

                for (int k = 0; k < 3; k++) {
                    l.Ambient[k] = m.light_ambient[3 * i + k];
                    l.Diffuse[k] = m.light_diffuse[3 * i + k];
                    l.Specular[k] = m.light_specular[3 * i + k];
                    l.Pos[k] = (float)_lightXpos[3 * i + k];
                    l.Dir[k] = (float)_lightXdir[3 * i + k];
                }

                Lights.Add(l);
            }
        }
    }
}
