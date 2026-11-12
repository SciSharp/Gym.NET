using System;
using System.IO;
using System.Reflection;
using System.Text;
using System.Text.Json;

namespace Gym.Environments.Envs.MuJoCo.Rendering {
    /// <summary>
    ///     MuJoCo's text overlay (<c>mjr_overlay</c>), what Gymnasium's MuJoCo window draws its menu and its statistics with
    ///     (DR-70): lines of MuJoCo's own bitmap font in a box blended half with black, at one of eight grid positions of the
    ///     frame; two columns side by side when the second text is not empty, the first grey and the second white.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The font is <c>normal150</c>, the context's normal font at the 150% scale Gymnasium creates its context with,
    ///     embedded as MuJoCo 4.0.0 compiles it into the library (<c>Assets/mujoco/fonts/normal150.json</c>, converted by
    ///     <c>tools/parity/mujoco/convert_font.py</c>): MuJoCo's own glyph bits, so every text pixel is MuJoCo's. The box
    ///     and the glyphs are written straight into the frame's samples with the reference GPU's measured storage rules
    ///     (<see cref="GlContext.ToUnorm8"/>, <see cref="GlContext.BlendLevel"/>), as <c>glRecti</c>'s quad under
    ///     <c>GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA</c> and <c>glBitmap</c>'s fragments reach them.
    ///     </para>
    ///     <para>
    ///     A text is MuJoCo's C string: its UTF-8 bytes, cut at <see cref="MaxOverlay"/>, split at <c>'\n'</c>. Only the
    ///     printable ASCII characters the font holds (32 to 126) have glyphs; other bytes take no room and draw nothing, where
    ///     MuJoCo's C reads past its 127-entry width table and calls display lists it never filled.
    ///     </para>
    ///     <para>
    ///     Reference: <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_gl2.c#L477">render_gl2.c</see>
    ///     (<c>draw_overlay</c>, <c>mjr_overlay</c>, <c>mjr_textActual</c>),
    ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/render/classic/render_context.c#L1279">render_context.c</see>
    ///     (<c>makeFont</c>).
    ///     </para>
    /// </remarks>
    internal static class MujocoOverlay {
        /// <summary>The grid positions (<c>mjtGridPos</c>): the corners, then the middles of the four edges.</summary>
        internal const int TopLeft = 0, TopRight = 1, BottomLeft = 2, BottomRight = 3, Top = 4, Bottom = 5, Left = 6, Right = 7;

        /// <summary>The most bytes of a text drawn (<c>mjMAXOVERLAY</c>); the rest is cut off, as MuJoCo's <c>strlen</c> cap does.</summary>
        internal const int MaxOverlay = 500;

        /// <summary>The embedded resource the font ships as (<c>Assets/mujoco/fonts/normal150.json</c> in the project).</summary>
        internal const string ResourceName = "assets/mujoco/fonts/normal150.json";

        /// <summary>The padding around and between lines, in pixels (<c>PAD</c>).</summary>
        private const int Pad = 5;

        /// <summary>The embedded font, parsed once on first use; a failure is cached and rethrown on every later use.</summary>
        private static readonly Lazy<BitmapFont> NormalFont = new(LoadEmbedded);

        /// <summary>
        ///     Draws one overlay into a frame (<c>mjr_overlay</c>): with a non-empty <paramref name="text2"/>, two boxes side by
        ///     side, <paramref name="text1"/> in grey (0.7) and <paramref name="text2"/> in white, mirrored on the right-hand
        ///     positions; else one white box of <paramref name="text1"/>.
        /// </summary>
        /// <param name="target">The frame, after the scene was rendered into it.</param>
        /// <param name="gridpos">One of <see cref="TopLeft"/> ... <see cref="Right"/>.</param>
        /// <param name="left">The viewport's left edge, in pixels.</param>
        /// <param name="bottom">The viewport's bottom edge, in pixels from the frame's bottom row.</param>
        /// <param name="width">The viewport's width; nothing is drawn unless positive.</param>
        /// <param name="height">The viewport's height; nothing is drawn unless positive.</param>
        /// <param name="text1">The first column's lines, each ending with <c>'\n'</c> as Gymnasium's <c>add_overlay</c> leaves them.</param>
        /// <param name="text2">The second column's lines, or an empty string for one column.</param>
        /// <exception cref="ArgumentNullException">A null target or text.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A grid position outside <c>mjtGridPos</c>.</exception>
        /// <exception cref="InvalidOperationException">The assembly lacks the embedded font (a build that dropped the asset).</exception>
        /// <exception cref="InvalidDataException">The embedded font is not MuJoCo's layout.</exception>
        internal static void Draw(GlTarget target, int gridpos, int left, int bottom, int width, int height, string text1, string text2) {
            ArgumentNullException.ThrowIfNull(target);
            ArgumentNullException.ThrowIfNull(text1);
            ArgumentNullException.ThrowIfNull(text2);
            if ((uint)gridpos > Right)
                throw new ArgumentOutOfRangeException(nameof(gridpos), gridpos, "not an mjtGridPos");
            // mjr_overlay's early return: an empty viewport draws nothing
            if (width <= 0 || height <= 0)
                return;
            BitmapFont font = NormalFont.Value;
            byte[] first = Encoding.UTF8.GetBytes(text1), second = Encoding.UTF8.GetBytes(text2);
            // ToUnorm8 is what the GPU stores glColor's 0.7 and 1 as (178 and 255)
            byte grey = GlContext.ToUnorm8(0.7f), white = GlContext.ToUnorm8(1f);
            if (second.Length > 0) {
                // the left-hand positions put the labels first; the right-hand ones the values first, so the labels stay outermost
                if (gridpos is TopLeft or BottomLeft) {
                    int skip = DrawColumn(target, font, left, bottom, width, height, 0, gridpos, grey, first);
                    DrawColumn(target, font, left, bottom, width, height, skip, gridpos, white, second);
                } else {
                    int skip = DrawColumn(target, font, left, bottom, width, height, 0, gridpos, white, second);
                    DrawColumn(target, font, left, bottom, width, height, skip, gridpos, grey, first);
                }
            } else {
                DrawColumn(target, font, left, bottom, width, height, 0, gridpos, white, first);
            }
        }

        /// <summary>
        ///     The size of the box <see cref="Draw"/> puts one column in (<c>draw_overlay</c>'s <c>W</c> and <c>H</c>): the
        ///     widest line plus 13 pixels, and each line's height plus 5 for every <c>'\n'</c> and once more.
        /// </summary>
        /// <param name="text">The column's text.</param>
        /// <returns>The width and height in pixels.</returns>
        /// <exception cref="ArgumentNullException">A null text.</exception>
        /// <exception cref="InvalidOperationException">The assembly lacks the embedded font.</exception>
        /// <exception cref="InvalidDataException">The embedded font is not MuJoCo's layout.</exception>
        internal static (int Width, int Height) Measure(string text) {
            ArgumentNullException.ThrowIfNull(text);
            return Measure(NormalFont.Value, Encoding.UTF8.GetBytes(text));
        }

        /// <summary>The box of a column's bytes (<c>draw_overlay</c>'s first loop).</summary>
        /// <param name="font">The font.</param>
        /// <param name="text">The bytes; only the first <see cref="MaxOverlay"/> count.</param>
        /// <returns>The width and height in pixels.</returns>
        private static (int Width, int Height) Measure(BitmapFont font, byte[] text) {
            int sz = Math.Min(MaxOverlay, text.Length);
            int nr = font.Height, nc = 0, ncthis = 0;
            for (int i = 0; i < sz; i++) {
                if (text[i] != '\n') {
                    ncthis += font.Width(text[i]);
                    nc = Math.Max(nc, ncthis);
                } else {
                    // a newline adds a line even when nothing follows it, so a text ending in '\n' gets an empty last line
                    nr += Pad + font.Height;
                    ncthis = 0;
                }
            }

            return (Pad + nc + 8, Pad + nr);
        }

        /// <summary>
        ///     One column (<c>draw_overlay</c>): the box at its grid position blended half with black, then each line's glyphs
        ///     from 3 pixels in, the first line at the top.
        /// </summary>
        /// <param name="target">The frame.</param>
        /// <param name="font">The font.</param>
        /// <param name="vl">The viewport's left edge.</param>
        /// <param name="vb">The viewport's bottom edge.</param>
        /// <param name="vw">The viewport's width.</param>
        /// <param name="vh">The viewport's height.</param>
        /// <param name="skip">The room the previous column took (its width less 2), 0 for the first.</param>
        /// <param name="gridpos">The grid position.</param>
        /// <param name="level">The text's colour level, the same on R, G and B.</param>
        /// <param name="text">The column's bytes.</param>
        /// <returns>The box's width less 2: where the next column starts (<c>skip</c>).</returns>
        /// <exception cref="InvalidOperationException">A depth-only target, which has no colour to draw into.</exception>
        private static int DrawColumn(GlTarget target, BitmapFont font, int vl, int vb, int vw, int vh, int skip, int gridpos, byte level, byte[] text) {
            (int w, int h) = Measure(font, text);
            // glViewport's corner per position; C's integer division truncates toward zero, as C#'s does
            (int x, int y) = gridpos switch {
                TopLeft => (vl + skip + Pad, vb + vh - 1 - Pad - h),
                TopRight => (vl + vw - skip - 1 - Pad - w, vb + vh - 1 - Pad - h),
                BottomLeft => (vl + skip + Pad, vb + Pad),
                BottomRight => (vl + vw - skip - 1 - Pad - w, vb + Pad),
                Top => (vl + (vw - skip - w) / 2 - 1 - Pad, vb + vh - 1 - Pad - h),
                Bottom => (vl + (vw - skip - w) / 2 - 1 - Pad, vb + Pad),
                Left => (vl + skip + Pad, vb + (vh - h) / 2 - 1 - Pad),
                _ => (vl + vw - skip - 1 - Pad - w, vb + (vh - h) / 2 - 1 - Pad),
            };

            // The box: the quad (0,0)-(W,H) under glOrtho(0, W-1, 0, H-1) reaches past the viewport and is clipped to it, so
            // it covers exactly the viewport's W x H pixels (every sample of them), black at alpha 0.5 over the frame. The
            // GPU stores the source first (alpha 0.5 is level 127), then blends the levels (measured, see BlendLevel).
            byte a8 = GlContext.ToUnorm8(0.5f);
            int sf = a8, df = 255 - a8;
            int ns = target.Samples;
            byte[] color = target.Color ?? throw new InvalidOperationException("The overlay needs a colour buffer.");
            for (int py = Math.Max(y, 0); py < Math.Min(y + h, target.Height); py++) {
                for (int px = Math.Max(x, 0); px < Math.Min(x + w, target.Width); px++) {
                    for (int s = 0; s < ns; s++) {
                        int o = 4 * ((py * target.Width + px) * ns + s);
                        color[o] = GlContext.BlendLevel(0, color[o], sf, df);
                        color[o + 1] = GlContext.BlendLevel(0, color[o + 1], sf, df);
                        color[o + 2] = GlContext.BlendLevel(0, color[o + 2], sf, df);
                        color[o + 3] = GlContext.BlendLevel(a8, color[o + 3], sf, df);
                    }
                }
            }

            // The lines: mjr_textActual's raster position (3, H - nr) maps to the viewport's (3 + 3/(W-1), Y + Y/(H-1)), whose
            // fractions stay below one pixel for every box (W >= 13, 0 < Y < H - 1), so glBitmap's floor puts the first
            // glyph's lower-left corner at (x + 3, y + Y); each glyph then advances the raster position by its width.
            int sz = Math.Min(MaxOverlay, text.Length);
            int nr = font.Height, start = 0;
            for (int i = 0; i < sz; i++) {
                // a line ends at a newline, or with the last byte drawn (which then belongs to it)
                if (text[i] != '\n' && i != sz - 1)
                    continue;
                int end = text[i] == '\n' ? i : i + 1;
                int penX = x + 3, penY = y + h - nr;
                for (int c = start; c < end; c++) {
                    DrawGlyph(target, font, text[c], penX, penY, level);
                    penX += font.Width(text[c]);
                }

                nr += Pad + font.Height;
                start = i + 1;
            }

            return w - 2;
        }

        /// <summary>
        ///     One glyph as <c>glBitmap</c> draws it: a fragment of the text's colour (alpha 1, no blending, no depth test) on
        ///     every sample of each pixel whose bit is set, the bitmap's first row at the bottom; pixels outside the frame are
        ///     dropped (the viewport does not clip bitmaps, the framebuffer does).
        /// </summary>
        /// <param name="target">The frame.</param>
        /// <param name="font">The font.</param>
        /// <param name="ch">The byte; one without a glyph draws nothing.</param>
        /// <param name="x0">The glyph's left column.</param>
        /// <param name="y0">The glyph's bottom row, from the frame's bottom.</param>
        /// <param name="level">The colour level.</param>
        private static void DrawGlyph(GlTarget target, BitmapFont font, byte ch, int x0, int y0, byte level) {
            int gw = font.Width(ch);
            if (gw == 0)
                return;
            int rowBytes = (gw - 1) / 8 + 1, offset = font.Offset(ch), ns = target.Samples;
            byte[] color = target.Color!;
            for (int j = 0; j < font.Height; j++) {
                int py = y0 + j;
                if ((uint)py >= (uint)target.Height)
                    continue;
                for (int i = 0; i < gw; i++) {
                    int px = x0 + i;
                    if ((uint)px >= (uint)target.Width)
                        continue;
                    // the rows are whole bytes (MuJoCo sets GL_UNPACK_ALIGNMENT to 1), the most significant bit leftmost
                    if ((font.Data[offset + j * rowBytes + (i >> 3)] & (0x80 >> (i & 7))) == 0)
                        continue;
                    for (int s = 0; s < ns; s++) {
                        int o = 4 * ((py * target.Width + px) * ns + s);
                        color[o] = color[o + 1] = color[o + 2] = level;
                        color[o + 3] = 255;
                    }
                }
            }
        }

        /// <summary>Reads the embedded <c>normal150</c> font.</summary>
        /// <returns>The font.</returns>
        /// <exception cref="InvalidOperationException">The assembly lacks the resource, or a field has the wrong JSON kind.</exception>
        /// <exception cref="InvalidDataException">The data is not MuJoCo's font layout.</exception>
        /// <exception cref="JsonException">The resource is not JSON.</exception>
        /// <exception cref="FormatException">The data is not base64.</exception>
        private static BitmapFont LoadEmbedded() {
            Assembly assembly = typeof(MujocoOverlay).Assembly;
            using Stream stream = assembly.GetManifestResourceStream(ResourceName)
                                  ?? throw new InvalidOperationException($"{assembly.GetName().Name} lacks its embedded resource '{ResourceName}', MuJoCo's overlay font.");
            using JsonDocument doc = JsonDocument.Parse(stream);
            string data = doc.RootElement.GetProperty("data").GetString() ?? throw new InvalidDataException("The font's data is null.");
            return BitmapFont.Parse(Convert.FromBase64String(data));
        }

        /// <summary>
        ///     A MuJoCo bitmap font as <c>makeFont</c> reads it: for each printable character (32 to 126) its code, width and
        ///     height, then its 1-bit rows from the bottom, each a whole number of bytes; the list ends with the byte 123.
        /// </summary>
        internal sealed class BitmapFont {
            /// <summary>The font's bytes, MuJoCo's array as compiled into the library.</summary>
            internal readonly byte[] Data;

            /// <summary>The characters' common height in pixels (<c>charHeight</c>).</summary>
            internal readonly int Height;

            /// <summary>Each byte's width in pixels (<c>charWidth</c>), 0 for bytes without a glyph.</summary>
            private readonly int[] _widths = new int[256];

            /// <summary>Each glyph's first row's offset in <see cref="Data"/>.</summary>
            private readonly int[] _offsets = new int[256];

            /// <summary>Reads the layout.</summary>
            /// <param name="data">The bytes.</param>
            /// <exception cref="InvalidDataException">A code out of order, a height that changes, or no terminating 123.</exception>
            private BitmapFont(byte[] data) {
                Data = data;
                int adr = 0;
                Height = -1;
                for (int code = 32; code <= 126; code++) {
                    // the same checks makeFont aborts on ("Invalid font data index", "Invalid font data termination")
                    if (adr + 2 >= data.Length || data[adr] != code)
                        throw new InvalidDataException($"MuJoCo's font: expected character {code} at byte {adr}.");
                    int width = data[adr + 1], height = data[adr + 2];
                    if (Height < 0)
                        Height = height;
                    else if (height != Height)
                        throw new InvalidDataException($"MuJoCo's font: character {code} is {height} pixels high, not {Height}.");
                    _widths[code] = width;
                    _offsets[code] = adr + 3;
                    adr += 3 + ((width - 1) / 8 + 1) * height;
                }

                if (adr >= data.Length || data[adr] != 123)
                    throw new InvalidDataException("MuJoCo's font does not end with its termination byte 123.");
            }

            /// <summary>Reads a font's bytes.</summary>
            /// <param name="data">The bytes.</param>
            /// <returns>The font.</returns>
            /// <exception cref="InvalidDataException">The bytes are not MuJoCo's font layout.</exception>
            internal static BitmapFont Parse(byte[] data) => new(data);

            /// <summary>A byte's advance in pixels: its glyph's width, 0 without a glyph.</summary>
            /// <param name="ch">The byte.</param>
            /// <returns>The width.</returns>
            internal int Width(byte ch) => _widths[ch];

            /// <summary>Where a byte's glyph rows start in <see cref="Data"/>.</summary>
            /// <param name="ch">The byte, one with a glyph.</param>
            /// <returns>The offset.</returns>
            internal int Offset(byte ch) => _offsets[ch];
        }
    }
}
