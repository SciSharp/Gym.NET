using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;

namespace Gym.MuJoCo {
    /// <summary>
    ///     What only the renderer reads (DR-62, DR-69): MuJoCo's <c>vis</c> and <c>stat</c> settings, the cameras, lights,
    ///     materials and textures, and the geoms' visual fields. The embedded models (DR-63) carry them; a trace dump does
    ///     not, so for one of those every array stays empty and every count 0, and nothing here is ever read.
    /// </summary>
    /// <remarks>
    ///     Values MuJoCo stores as float32 (<c>vis</c>, the light colours, the materials, <c>cam_intrinsic</c>) are kept as
    ///     <see langword="float"/>; the recording wrote them as doubles, so the casts back are exact. Positions and
    ///     orientations (<c>cam_pos</c>, <c>light_dir</c>, ...) are MuJoCo's doubles. Reference:
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/include/mujoco/mjmodel.h">mjmodel.h</see>.
    /// </remarks>
    public sealed partial class MjModel {
        /// <summary>The free camera's vertical field of view in degrees (<c>vis.global.fovy</c>).</summary>
        internal float vis_fovy = 45f;
        /// <summary>The stereo cameras' separation (<c>vis.global.ipd</c>); averaged away by a mono frame.</summary>
        internal float vis_ipd = 0.068f;
        /// <summary>The free camera's default azimuth in degrees (<c>vis.global.azimuth</c>).</summary>
        internal float vis_azimuth = 90f;
        /// <summary>The free camera's default elevation in degrees (<c>vis.global.elevation</c>).</summary>
        internal float vis_elevation = -45f;
        /// <summary>Slices of the builtin round meshes (<c>vis.quality.numslices</c>).</summary>
        internal int vis_numslices = 28;
        /// <summary>Stacks of the builtin round meshes (<c>vis.quality.numstacks</c>).</summary>
        internal int vis_numstacks = 16;
        /// <summary>Quads per side of the builtin plane and box (<c>vis.quality.numquads</c>).</summary>
        internal int vis_numquads = 4;
        /// <summary>Multisamples of the offscreen buffer (<c>vis.quality.offsamples</c>).</summary>
        internal int vis_offsamples = 4;
        /// <summary>The shadow map's side in texels (<c>vis.quality.shadowsize</c>).</summary>
        internal int vis_shadowsize = 4096;
        /// <summary>Whether the headlight is on (<c>vis.headlight.active</c>, any nonzero value).</summary>
        internal int vis_headlight_active = 1;
        /// <summary>The headlight's ambient colour.</summary>
        internal float[] vis_headlight_ambient = [0.1f, 0.1f, 0.1f];
        /// <summary>The headlight's diffuse colour.</summary>
        internal float[] vis_headlight_diffuse = [0.4f, 0.4f, 0.4f];
        /// <summary>The headlight's specular colour.</summary>
        internal float[] vis_headlight_specular = [0.5f, 0.5f, 0.5f];
        /// <summary>The near clipping plane, relative to the extent (<c>vis.map.znear</c>).</summary>
        internal float vis_znear = 0.01f;
        /// <summary>The far clipping plane, relative to the extent (<c>vis.map.zfar</c>).</summary>
        internal float vis_zfar = 50f;
        /// <summary>The haze band's proportion (<c>vis.map.haze</c>).</summary>
        internal float vis_haze = 0.3f;
        /// <summary>Fog's start, relative to the extent (<c>vis.map.fogstart</c>).</summary>
        internal float vis_fogstart = 3f;
        /// <summary>Fog's end, relative to the extent (<c>vis.map.fogend</c>).</summary>
        internal float vis_fogend = 10f;
        /// <summary>The alpha transparent dynamic geoms get (<c>vis.map.alpha</c>).</summary>
        internal float vis_alpha = 0.3f;
        /// <summary>A directional light's shadow clip, relative to the extent (<c>vis.map.shadowclip</c>).</summary>
        internal float vis_shadowclip = 1f;
        /// <summary>A spot light's shadow cone scale (<c>vis.map.shadowscale</c>).</summary>
        internal float vis_shadowscale = 0.6f;
        /// <summary>The haze's colour (<c>vis.rgba.haze</c>).</summary>
        internal float[] vis_rgba_haze = [1f, 1f, 1f, 1f];
        /// <summary>The fog's colour (<c>vis.rgba.fog</c>).</summary>
        internal float[] vis_rgba_fog = [0f, 0f, 0f, 1f];
        /// <summary>The model's centre (<c>stat.center</c>).</summary>
        internal double[] stat_center = new double[3];

        /// <summary>Each camera's tracking mode: 0 fixed, 1 track, 2 track the subtree's centre of mass, 3/4 target a body.</summary>
        internal int[] cam_mode = [];
        /// <summary>Each camera's body.</summary>
        internal int[] cam_bodyid = [];
        /// <summary>Each camera's target body, or -1.</summary>
        internal int[] cam_targetbodyid = [];
        /// <summary>Each camera's position in its body's frame.</summary>
        internal double[] cam_pos = [];
        /// <summary>Each camera's orientation in its body's frame.</summary>
        internal double[] cam_quat = [];
        /// <summary>Each camera's offset from its subtree's centre of mass in the reference configuration.</summary>
        internal double[] cam_poscom0 = [];
        /// <summary>Each camera's offset from its body in the reference configuration.</summary>
        internal double[] cam_pos0 = [];
        /// <summary>Each camera's global orientation in the reference configuration, row-major.</summary>
        internal double[] cam_mat0 = [];
        /// <summary>Each camera's projection: 0 perspective, 1 orthographic.</summary>
        internal int[] cam_projection = [];
        /// <summary>Each camera's vertical field of view in degrees.</summary>
        internal double[] cam_fovy = [];
        /// <summary>Each camera's stereo separation.</summary>
        internal double[] cam_ipd = [];
        /// <summary>Each camera's intrinsics (focal lengths and principal point), used when its sensor size is set.</summary>
        internal float[] cam_intrinsic = [];
        /// <summary>Each camera's sensor size; a zero height means none (the field of view applies).</summary>
        internal float[] cam_sensorsize = [];

        /// <summary>Number of lights.</summary>
        internal int nlight;
        /// <summary>Each light's tracking mode, as the cameras'.</summary>
        internal int[] light_mode = [];
        /// <summary>Each light's body.</summary>
        internal int[] light_bodyid = [];
        /// <summary>Each light's target body, or -1.</summary>
        internal int[] light_targetbodyid = [];
        /// <summary>Each light's type: 0 spot, 1 directional, 2 point, 3 image (only the first two are drawn).</summary>
        internal int[] light_type = [];
        /// <summary>Each light's texture, or -1.</summary>
        internal int[] light_texid = [];
        /// <summary>Whether each light casts shadows.</summary>
        internal bool[] light_castshadow = [];
        /// <summary>Whether each light is on.</summary>
        internal bool[] light_active = [];
        /// <summary>Each light's position in its body's frame.</summary>
        internal double[] light_pos = [];
        /// <summary>Each light's direction in its body's frame.</summary>
        internal double[] light_dir = [];
        /// <summary>Each light's offset from its subtree's centre of mass in the reference configuration.</summary>
        internal double[] light_poscom0 = [];
        /// <summary>Each light's offset from its body in the reference configuration.</summary>
        internal double[] light_pos0 = [];
        /// <summary>Each light's global direction in the reference configuration.</summary>
        internal double[] light_dir0 = [];
        /// <summary>Each spot light's attenuation: constant, linear, quadratic.</summary>
        internal float[] light_attenuation = [];
        /// <summary>Each spot light's cone cutoff in degrees.</summary>
        internal float[] light_cutoff = [];
        /// <summary>Each spot light's exponent.</summary>
        internal float[] light_exponent = [];
        /// <summary>Each light's ambient colour.</summary>
        internal float[] light_ambient = [];
        /// <summary>Each light's diffuse colour.</summary>
        internal float[] light_diffuse = [];
        /// <summary>Each light's specular colour.</summary>
        internal float[] light_specular = [];

        /// <summary>Number of materials.</summary>
        internal int nmat;
        /// <summary>Each material's colour.</summary>
        internal float[] mat_rgba = [];
        /// <summary>Each material's emission, a multiple of its colour.</summary>
        internal float[] mat_emission = [];
        /// <summary>Each material's specular intensity.</summary>
        internal float[] mat_specular = [];
        /// <summary>Each material's shininess in [0, 1] (OpenGL's exponent is 128 times it).</summary>
        internal float[] mat_shininess = [];
        /// <summary>Each material's reflectance: above 0, a plane or box mirrors the scene.</summary>
        internal float[] mat_reflectance = [];
        /// <summary>Each material's textures by role, <see cref="TexRoleCount"/> per material (<see cref="TexRoleRgb"/> is the colour).</summary>
        internal int[] mat_texid = [];
        /// <summary>Each material's texture repetition in x and y.</summary>
        internal float[] mat_texrepeat = [];
        /// <summary>Whether each material's texture repeats per unit of length rather than per object.</summary>
        internal bool[] mat_texuniform = [];

        /// <summary>Number of textures.</summary>
        internal int ntex;
        /// <summary>Each texture's type: 0 2D, 1 cube, 2 skybox.</summary>
        internal int[] tex_type = [];
        /// <summary>Each texture's width in texels.</summary>
        internal int[] tex_width = [];
        /// <summary>Each texture's height in texels (six faces stacked for a cube or skybox whose height is six widths).</summary>
        internal int[] tex_height = [];
        /// <summary>Each texture's embedded PNG, a path under the assets' <c>mujoco/</c> folder, or <see langword="null"/> when not recorded.</summary>
        internal string?[] tex_png = [];

        /// <summary>Each geom's material, or -1.</summary>
        internal int[] geom_matid = [];
        /// <summary>Each geom's visibility group (0 to 5; the default options show 0, 1 and 2).</summary>
        internal int[] geom_group = [];
        /// <summary>Whether each geom casts shadows.</summary>
        internal bool[] geom_castshadow = [];

        /// <summary>Each site's shape (a geom type).</summary>
        internal int[] site_type = [];
        /// <summary>Each site's size, 3 per site.</summary>
        internal double[] site_size = [];
        /// <summary>Each site's colour.</summary>
        internal float[] site_rgba = [];
        /// <summary>Each site's material, or -1.</summary>
        internal int[] site_matid = [];
        /// <summary>Each site's visibility group.</summary>
        internal int[] site_group = [];

        /// <summary>The number of texture roles per material (<c>mjNTEXROLE</c>).</summary>
        internal const int TexRoleCount = 10;

        /// <summary>The colour role of a material's textures (<c>mjTEXROLE_RGB</c>).</summary>
        internal const int TexRoleRgb = 1;

        /// <summary>
        ///     Reads the renderer's fields from an embedded model's JSON (<c>tools/parity/mujoco/record_models.py</c>'s
        ///     layout); a trace dump, which records none of them, leaves them at MuJoCo's defaults and empty.
        /// </summary>
        /// <param name="root">The JSON's root object.</param>
        /// <param name="arrays">The object holding the arrays (the root's <c>arrays</c>, or the root of a trace dump).</param>
        /// <param name="sizes">The <c>sizes</c> object.</param>
        /// <exception cref="InvalidOperationException">A field has the wrong JSON kind.</exception>
        /// <exception cref="KeyNotFoundException">A recorded section lacks one of its fields.</exception>
        private void LoadVisual(JsonElement root, JsonElement arrays, JsonElement sizes) {
            if (root.TryGetProperty("vis", out JsonElement vis)) {
                JsonElement global = vis.GetProperty("global_"), quality = vis.GetProperty("quality");
                JsonElement headlight = vis.GetProperty("headlight"), map = vis.GetProperty("map"), rgba = vis.GetProperty("rgba");
                vis_fovy = (float)global.GetProperty("fovy").GetDouble();
                vis_ipd = (float)global.GetProperty("ipd").GetDouble();
                vis_azimuth = (float)global.GetProperty("azimuth").GetDouble();
                vis_elevation = (float)global.GetProperty("elevation").GetDouble();
                vis_numslices = quality.GetProperty("numslices").GetInt32();
                vis_numstacks = quality.GetProperty("numstacks").GetInt32();
                vis_numquads = quality.GetProperty("numquads").GetInt32();
                vis_offsamples = quality.GetProperty("offsamples").GetInt32();
                vis_shadowsize = quality.GetProperty("shadowsize").GetInt32();
                vis_headlight_active = headlight.GetProperty("active").GetInt32();
                vis_headlight_ambient = Floats(headlight.GetProperty("ambient"));
                vis_headlight_diffuse = Floats(headlight.GetProperty("diffuse"));
                vis_headlight_specular = Floats(headlight.GetProperty("specular"));
                vis_znear = (float)map.GetProperty("znear").GetDouble();
                vis_zfar = (float)map.GetProperty("zfar").GetDouble();
                vis_haze = (float)map.GetProperty("haze").GetDouble();
                vis_fogstart = (float)map.GetProperty("fogstart").GetDouble();
                vis_fogend = (float)map.GetProperty("fogend").GetDouble();
                vis_alpha = (float)map.GetProperty("alpha").GetDouble();
                vis_shadowclip = (float)map.GetProperty("shadowclip").GetDouble();
                vis_shadowscale = (float)map.GetProperty("shadowscale").GetDouble();
                vis_rgba_haze = Floats(rgba.GetProperty("haze"));
                vis_rgba_fog = Floats(rgba.GetProperty("fog"));
            }

            if (root.TryGetProperty("stat", out JsonElement stat) && stat.TryGetProperty("center", out JsonElement center))
                stat_center = center.EnumerateArray().Select(e => e.GetDouble()).ToArray();

            // a trace dump records no visual arrays: everything below stays empty
            if (!arrays.TryGetProperty("geom_matid", out _))
                return;
            double[] D(string name) => arrays.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
            float[] F(string name) => arrays.GetProperty(name).EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            int[] I(string name) => arrays.GetProperty(name).EnumerateArray()
                .Select(e => e.ValueKind == JsonValueKind.True ? 1 : e.ValueKind == JsonValueKind.False ? 0 : e.GetInt32()).ToArray();
            int Size(string name) => sizes.TryGetProperty(name, out JsonElement e) ? e.GetInt32() : 0;

            geom_matid = I("geom_matid");
            geom_group = I("geom_group");
            geom_castshadow = I("geom_castshadow").Select(v => v != 0).ToArray();
            if (nsite > 0) {
                site_type = I("site_type");
                site_size = D("site_size");
                site_rgba = F("site_rgba");
                site_matid = I("site_matid");
                site_group = I("site_group");
            }
            if (ncam > 0) {
                cam_mode = I("cam_mode");
                cam_bodyid = I("cam_bodyid");
                cam_targetbodyid = I("cam_targetbodyid");
                cam_pos = D("cam_pos");
                cam_quat = D("cam_quat");
                cam_poscom0 = D("cam_poscom0");
                cam_pos0 = D("cam_pos0");
                cam_mat0 = D("cam_mat0");
                cam_projection = I("cam_projection");
                cam_fovy = D("cam_fovy");
                cam_ipd = D("cam_ipd");
                cam_intrinsic = F("cam_intrinsic");
                cam_sensorsize = F("cam_sensorsize");
            }

            nlight = Size("nlight");
            if (nlight > 0) {
                light_mode = I("light_mode");
                light_bodyid = I("light_bodyid");
                light_targetbodyid = I("light_targetbodyid");
                light_type = I("light_type");
                light_texid = I("light_texid");
                light_castshadow = I("light_castshadow").Select(v => v != 0).ToArray();
                light_active = I("light_active").Select(v => v != 0).ToArray();
                light_pos = D("light_pos");
                light_dir = D("light_dir");
                light_poscom0 = D("light_poscom0");
                light_pos0 = D("light_pos0");
                light_dir0 = D("light_dir0");
                light_attenuation = F("light_attenuation");
                light_cutoff = F("light_cutoff");
                light_exponent = F("light_exponent");
                light_ambient = F("light_ambient");
                light_diffuse = F("light_diffuse");
                light_specular = F("light_specular");
            }

            nmat = Size("nmat");
            if (nmat > 0) {
                mat_rgba = F("mat_rgba");
                mat_emission = F("mat_emission");
                mat_specular = F("mat_specular");
                mat_shininess = F("mat_shininess");
                mat_reflectance = F("mat_reflectance");
                mat_texid = I("mat_texid");
                mat_texrepeat = F("mat_texrepeat");
                mat_texuniform = I("mat_texuniform").Select(v => v != 0).ToArray();
            }

            ntex = Size("ntex");
            if (ntex > 0) {
                tex_type = I("tex_type");
                tex_width = I("tex_width");
                tex_height = I("tex_height");
                // the pixels themselves are PNGs beside the model, one per texture in index order
                tex_png = new string?[ntex];
                if (root.TryGetProperty("textures", out JsonElement textures)) {
                    int i = 0;
                    foreach (JsonElement texture in textures.EnumerateArray()) {
                        if (i < ntex && texture.TryGetProperty("png", out JsonElement png))
                            tex_png[i] = png.GetString();
                        i++;
                    }
                }
            }

            static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => (float)v.GetDouble()).ToArray();
        }
    }
}
