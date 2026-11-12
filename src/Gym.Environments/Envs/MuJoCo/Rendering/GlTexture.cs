using System;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     A texture as MuJoCo uploads it (<c>mjr_uploadTexture</c>): a 2D texture with <c>GL_REPEAT</c> wrapping, or a cube
    ///     map (cube and skybox textures) with <c>GL_CLAMP_TO_EDGE</c>; either filtered with <c>GL_LINEAR</c> when magnified
    ///     and <c>GL_LINEAR_MIPMAP_LINEAR</c> when minified, over a mipmap chain built as <c>glGenerateMipmap</c> builds it.
    /// </summary>
    /// <remarks>
    ///     Texels are unorm8, as the GPU stores <c>GL_RGB</c>; every level of the chain is stored back to 8 bits as the GPU
    ///     stores a colour. <c>glGenerateMipmap</c>'s filter is the driver's: here an area-weighted box over the texels each
    ///     smaller texel covers (the 2×2 average for even sizes), which the reference GPU's chain of Gymnasium's floor
    ///     checker follows but for one level-3 departure; the level of detail, its blend and the bilinear weights are the
    ///     reference GPU's, measured (<see cref="LodSteps"/>, <see cref="Bilinear"/>). Reference:
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_context.c#L1448">render_context.c</see>.
    /// </remarks>
    internal sealed class GlTexture {
        /// <summary>One mipmap level of one face: RGB texels, row 0 first (texture coordinate t = 0).</summary>
        private sealed class Level {
            /// <summary>The level's width.</summary>
            internal readonly int Width;

            /// <summary>The level's height.</summary>
            internal readonly int Height;

            /// <summary>The texels, 3 bytes each, row-major from t = 0.</summary>
            internal readonly byte[] Rgb;

            /// <summary>A level of the given size.</summary>
            /// <param name="width">The width.</param>
            /// <param name="height">The height.</param>
            /// <param name="rgb">The texels.</param>
            internal Level(int width, int height, byte[] rgb) {
                Width = width;
                Height = height;
                Rgb = rgb;
            }
        }

        /// <summary>Whether this is a cube map (cube or skybox) rather than a 2D texture.</summary>
        internal readonly bool IsCube;

        /// <summary>Each face's mipmap chain: one face for a 2D texture, six (+x, -x, +y, -y, +z, -z) for a cube map.</summary>
        private readonly Level[][] _faces;

        /// <summary>Builds a texture and its mipmap chain from MuJoCo's texture data.</summary>
        /// <param name="type">MuJoCo's texture type: 0 2D, 1 cube, 2 skybox.</param>
        /// <param name="width">The data's width.</param>
        /// <param name="height">The data's height (six widths for a cube map with separate faces).</param>
        /// <param name="rgb">The texels, 3 bytes each, row-major.</param>
        /// <exception cref="ArgumentException">The data is shorter than its size says.</exception>
        internal GlTexture(int type, int width, int height, byte[] rgb) {
            if (rgb.Length < 3 * width * height)
                throw new ArgumentException($"A {width}×{height} texture needs {3 * width * height} bytes, got {rgb.Length}.", nameof(rgb));
            IsCube = type != 0;
            if (!IsCube) {
                _faces = [Chain(width, height, rgb)];
                return;
            }

            // a square image repeats on all six faces; otherwise the faces are stacked, each width × width
            _faces = new Level[6][];
            for (int f = 0; f < 6; f++) {
                var face = new byte[3 * width * width];
                Array.Copy(rgb, width == height ? 0 : f * 3 * width * width, face, 0, face.Length);
                _faces[f] = Chain(width, width, face);
            }
        }

        /// <summary>The level-0 size of the first face, for the level-of-detail computation.</summary>
        internal int Width => _faces[0][0].Width;

        /// <summary>The level-0 height of the first face.</summary>
        internal int Height => _faces[0][0].Height;

        /// <summary>The mipmap chain of one image (<c>glGenerateMipmap</c>): halving each side, floored, down to 1×1.</summary>
        /// <param name="width">Level 0's width.</param>
        /// <param name="height">Level 0's height.</param>
        /// <param name="rgb">Level 0's texels.</param>
        /// <returns>The levels, level 0 first.</returns>
        private static Level[] Chain(int width, int height, byte[] rgb) {
            var levels = new System.Collections.Generic.List<Level> { new(width, height, rgb) };
            while (width > 1 || height > 1) {
                int w2 = Math.Max(1, width / 2), h2 = Math.Max(1, height / 2);
                levels.Add(Downsample(levels[^1], w2, h2));
                width = w2;
                height = h2;
            }

            return levels.ToArray();
        }

        /// <summary>One smaller level: each texel the area-weighted average of the texels it covers, rounded to unorm8.</summary>
        /// <param name="src">The larger level.</param>
        /// <param name="w2">The new width.</param>
        /// <param name="h2">The new height.</param>
        /// <returns>The new level.</returns>
        private static Level Downsample(Level src, int w2, int h2) {
            var dst = new byte[3 * w2 * h2];
            double sx = src.Width / (double)w2, sy = src.Height / (double)h2;
            for (int j = 0; j < h2; j++) {
                double y0 = j * sy, y1 = (j + 1) * sy;
                for (int i = 0; i < w2; i++) {
                    double x0 = i * sx, x1 = (i + 1) * sx;
                    double r = 0, g = 0, b = 0, area = 0;
                    for (int yy = (int)Math.Floor(y0); yy < Math.Min(src.Height, (int)Math.Ceiling(y1)); yy++) {
                        double wy = Math.Min(y1, yy + 1) - Math.Max(y0, yy);
                        for (int xx = (int)Math.Floor(x0); xx < Math.Min(src.Width, (int)Math.Ceiling(x1)); xx++) {
                            double wx = Math.Min(x1, xx + 1) - Math.Max(x0, xx);
                            double w = wx * wy;
                            int k = 3 * (yy * src.Width + xx);
                            r += w * src.Rgb[k];
                            g += w * src.Rgb[k + 1];
                            b += w * src.Rgb[k + 2];
                            area += w;
                        }
                    }

                    // stored as the GPU stores a colour, an exact half rounding down: reduced from the GPU's own
                    // levels so, every texel of the reference GPU's levels 4 to 6 of Gymnasium's 100×100 checker comes
                    // out. Its level 3 departs from the area box by one in 10 of 144 texels (its 25 → 12 filter stores
                    // 179 and 180 for equal area averages), and this chain carries that into 4 texels of level 4 and 2
                    // of level 5 (MujocoRendererTests pins the counts)
                    int o = 3 * (j * w2 + i);
                    dst[o] = GlContext.ToUnorm8((float)(r / area / 255));
                    dst[o + 1] = GlContext.ToUnorm8((float)(g / area / 255));
                    dst[o + 2] = GlContext.ToUnorm8((float)(b / area / 255));
                }
            }

            return new Level(w2, h2, dst);
        }

        /// <summary>
        ///     The reference GPU's base-2 logarithm of a mantissa, in 1/256 steps: entry <c>j</c> is the step a derivative
        ///     length of mantissa <c>1 + j/256</c> (truncated to 8 bits) reaches. Measured (2026-11-12) by sweeping one
        ///     octave of lengths in 131,072 steps: every threshold fell on a multiple of 1/256 of the mantissa, the same in
        ///     two octaves. It is <c>round(256·log2(1 + (j + ½)/256))</c> but one lower at 14 entries; with it, 2,048 of
        ///     2,048 isotropic levels agree, against 89% from the exact logarithm.
        /// </summary>
        private static readonly byte[] Log2Steps = [
            0, 2, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43,
            44, 45, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 81, 82,
            83, 84, 85, 86, 88, 89, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 109, 110, 111, 112, 113, 114, 115, 116, 117,
            118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
            150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 173, 174, 175, 176, 177, 178, 179,
            180, 181, 181, 182, 183, 184, 185, 186, 187, 188, 188, 189, 190, 191, 192, 193, 194, 195, 195, 196, 197, 198, 199, 200, 200, 201, 202, 203, 204, 205, 205, 206,
            207, 208, 209, 210, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218, 218, 219, 220, 221, 222, 222, 223, 224, 225, 226, 226, 227, 228, 229, 229, 230, 231, 232,
            233, 233, 234, 235, 236, 236, 237, 238, 239, 239, 240, 241, 242, 242, 243, 244, 245, 245, 246, 247, 248, 248, 249, 250, 251, 251, 252, 253, 253, 254, 255, 255,
        ];

        /// <summary>The smallest normal float (2^-126): below it, <see cref="MathF.ILogB"/> no longer gives the exponent the mantissa table needs.</summary>
        internal const float MinNormal = 1.17549435E-38f;

        /// <summary>
        ///     The level of detail of a derivative length as the reference GPU quantizes it, in 1/256 steps: the length's
        ///     exponent, plus <see cref="Log2Steps"/> of its mantissa truncated to 8 bits.
        /// </summary>
        /// <param name="rho">The larger derivative length, in texels per pixel.</param>
        /// <returns>
        ///     The level in 1/256 steps; at or below 0 the texture is magnified (so for 0, a subnormal length or NaN, a
        ///     degenerate derivative), <see cref="int.MaxValue"/> for an infinite length (the smallest level).
        /// </returns>
        internal static int LodSteps(float rho) {
            // !(rho >= MinNormal) sends 0, NaN and subnormals to magnification; ILogB is only exact on normal numbers
            if (!(rho >= MinNormal))
                return int.MinValue;
            if (float.IsPositiveInfinity(rho))
                return int.MaxValue;
            int e = MathF.ILogB(rho);
            float m = MathF.ScaleB(rho, -e);
            return 256 * e + Log2Steps[(int)((m - 1) * 256)];
        }

        /// <summary>
        ///     Samples a 2D texture at (<paramref name="s"/>, <paramref name="t"/>) at a level of detail
        ///     (<c>GL_LINEAR</c> magnified, <c>GL_LINEAR_MIPMAP_LINEAR</c> minified, <c>GL_REPEAT</c>).
        /// </summary>
        /// <param name="s">The s coordinate.</param>
        /// <param name="t">The t coordinate.</param>
        /// <param name="lod">The level of detail in 1/256 steps (<see cref="LodSteps"/>); at or below 0 the texture is
        ///     magnified.</param>
        /// <param name="r">Red, written in [0, 1].</param>
        /// <param name="g">Green, written.</param>
        /// <param name="b">Blue, written.</param>
        internal void Sample2D(float s, float t, int lod, out float r, out float g, out float b) =>
            SampleFace(_faces[0], s, t, lod, repeat: true, out r, out g, out b);

        /// <summary>
        ///     Samples a cube map along direction (<paramref name="x"/>, <paramref name="y"/>, <paramref name="z"/>): the face
        ///     of the major axis and its (s, t) as OpenGL's table defines them, then the face as a clamped 2D texture.
        /// </summary>
        /// <param name="x">The direction's x.</param>
        /// <param name="y">The direction's y.</param>
        /// <param name="z">The direction's z.</param>
        /// <param name="lod">The level of detail in 1/256 steps.</param>
        /// <param name="r">Red, written in [0, 1].</param>
        /// <param name="g">Green, written.</param>
        /// <param name="b">Blue, written.</param>
        internal void SampleCube(float x, float y, float z, int lod, out float r, out float g, out float b) {
            CubeFace(x, y, z, out int face, out float s, out float t);
            SampleFace(_faces[face], s, t, lod, repeat: false, out r, out g, out b);
        }

        /// <summary>The cube map face and its coordinates in [0, 1] for a direction (OpenGL's major-axis table).</summary>
        /// <param name="x">The direction's x.</param>
        /// <param name="y">The direction's y.</param>
        /// <param name="z">The direction's z.</param>
        /// <param name="face">The face: 0 +x, 1 -x, 2 +y, 3 -y, 4 +z, 5 -z.</param>
        /// <param name="s">The face's s coordinate.</param>
        /// <param name="t">The face's t coordinate.</param>
        internal static void CubeFace(float x, float y, float z, out int face, out float s, out float t) {
            float ax = MathF.Abs(x), ay = MathF.Abs(y), az = MathF.Abs(z);
            float sc, tc, ma;
            if (ax >= ay && ax >= az) {
                face = x >= 0 ? 0 : 1;
                sc = x >= 0 ? -z : z;
                tc = -y;
                ma = ax;
            } else if (ay >= az) {
                face = y >= 0 ? 2 : 3;
                sc = x;
                tc = y >= 0 ? z : -z;
                ma = ay;
            } else {
                face = z >= 0 ? 4 : 5;
                sc = z >= 0 ? x : -x;
                tc = -y;
                ma = az;
            }

            s = (sc / ma + 1) * 0.5f;
            t = (tc / ma + 1) * 0.5f;
        }

        /// <summary>Trilinear (or, magnified, bilinear) sampling of one face's mipmap chain.</summary>
        /// <param name="chain">The face's levels.</param>
        /// <param name="s">The s coordinate.</param>
        /// <param name="t">The t coordinate.</param>
        /// <param name="lod">The level of detail in 1/256 steps: the whole levels in the high bits, the blend toward the
        ///     next level in the low 8 (the GPU blends levels in exact 1/256 steps, measured).</param>
        /// <param name="repeat"><c>GL_REPEAT</c> wrapping, else <c>GL_CLAMP_TO_EDGE</c>.</param>
        /// <param name="r">Red, written.</param>
        /// <param name="g">Green, written.</param>
        /// <param name="b">Blue, written.</param>
        private static void SampleFace(Level[] chain, float s, float t, int lod, bool repeat, out float r, out float g, out float b) {
            if (lod <= 0) {
                Bilinear(chain[0], s, t, repeat, out r, out g, out b);
                return;
            }

            int max = chain.Length - 1;
            if (lod >= max * 256) {
                Bilinear(chain[max], s, t, repeat, out r, out g, out b);
                return;
            }

            int d1 = lod >> 8;
            float f = (lod & 255) * (1f / 256);
            Bilinear(chain[d1], s, t, repeat, out float r1, out float g1, out float b1);
            if (f == 0) {
                r = r1;
                g = g1;
                b = b1;
                return;
            }

            Bilinear(chain[d1 + 1], s, t, repeat, out float r2, out float g2, out float b2);
            r = r1 + f * (r2 - r1);
            g = g1 + f * (g2 - g1);
            b = b1 + f * (b2 - b1);
        }

        /// <summary>
        ///     Bilinear sampling of one level (texel centres at half-integers), the weights in steps of 1/256 as the GPU's
        ///     texture unit takes them.
        /// </summary>
        /// <param name="level">The level.</param>
        /// <param name="s">The s coordinate.</param>
        /// <param name="t">The t coordinate.</param>
        /// <param name="repeat">Repeat or clamp to the edge.</param>
        /// <param name="r">Red, written.</param>
        /// <param name="g">Green, written.</param>
        /// <param name="b">Blue, written.</param>
        /// <remarks>
        ///     Measured on the reference GPU: a two-texel ramp magnified 1024 times steps once every four pixels, each
        ///     weight the exact one rounded to the nearest 1/256 (with the 12-bit store of
        ///     <see cref="GlContext.ToUnorm8"/>, all 1024 pixels agree).
        /// </remarks>
        private static void Bilinear(Level level, float s, float t, bool repeat, out float r, out float g, out float b) {
            int w = level.Width, h = level.Height;
            float u = s * w - 0.5f, v = t * h - 0.5f;
            float fu = MathF.Floor(u), fv = MathF.Floor(v);
            // the weights rounded to 8 fractional bits; a weight that rounds up to 1 puts the sample on the next texel
            float a = MathF.Floor((u - fu) * 256 + 0.5f) * (1f / 256), c = MathF.Floor((v - fv) * 256 + 0.5f) * (1f / 256);
            int i0 = (int)fu, j0 = (int)fv;
            int i1 = i0 + 1, j1 = j0 + 1;
            if (repeat) {
                i0 = Mod(i0, w);
                i1 = Mod(i1, w);
                j0 = Mod(j0, h);
                j1 = Mod(j1, h);
            } else {
                i0 = Math.Clamp(i0, 0, w - 1);
                i1 = Math.Clamp(i1, 0, w - 1);
                j0 = Math.Clamp(j0, 0, h - 1);
                j1 = Math.Clamp(j1, 0, h - 1);
            }

            byte[] px = level.Rgb;
            int k00 = 3 * (j0 * w + i0), k10 = 3 * (j0 * w + i1), k01 = 3 * (j1 * w + i0), k11 = 3 * (j1 * w + i1);
            float w00 = (1 - a) * (1 - c), w10 = a * (1 - c), w01 = (1 - a) * c, w11 = a * c;
            const float inv = 1f / 255f;
            r = (w00 * px[k00] + w10 * px[k10] + w01 * px[k01] + w11 * px[k11]) * inv;
            g = (w00 * px[k00 + 1] + w10 * px[k10 + 1] + w01 * px[k01 + 1] + w11 * px[k11 + 1]) * inv;
            b = (w00 * px[k00 + 2] + w10 * px[k10 + 2] + w01 * px[k01 + 2] + w11 * px[k11 + 2]) * inv;

            static int Mod(int a, int n) {
                int m = a % n;
                return m < 0 ? m + n : m;
            }
        }
    }
}
