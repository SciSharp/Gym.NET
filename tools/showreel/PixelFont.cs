using System;

namespace Gym.Showreel;

/// <summary>
///     The showreel's only typeface: the classic 5x7 dot-matrix font, every font pixel drawn as a square block, so all text
///     in the video is pixel art and needs no font files. Covers ASCII 32-95 (space, digits, punctuation, capitals) plus
///     the middle dot <c>·</c>; lower case draws as upper case and anything else as a space.
/// </summary>
/// <remarks>
///     Glyphs are five column bytes, bit 0 the top row. At scale <c>s</c> a character cell is <c>6s</c> pixels wide (five
///     columns and one of spacing) and <c>7s</c> tall. The same table is ported to the docs site's gallery script
///     (<c>docs/website-src/templates/gymnet/public/main.js</c>) so the page and the video share one lettering.
/// </remarks>
internal static class PixelFont {
    /// <summary>Width of one character cell in font pixels (five columns and one of spacing).</summary>
    public const int CellWidth = 6;

    /// <summary>Height of one character in font pixels.</summary>
    public const int Height = 7;

    /// <summary>The glyph columns of ASCII 32-95, five bytes each.</summary>
    private static readonly byte[] Glyphs = {
        0x00, 0x00, 0x00, 0x00, 0x00, /* ' ' */ 0x00, 0x00, 0x5F, 0x00, 0x00, /* ! */
        0x00, 0x07, 0x00, 0x07, 0x00, /* " */ 0x14, 0x7F, 0x14, 0x7F, 0x14, /* # */
        0x24, 0x2A, 0x7F, 0x2A, 0x12, /* $ */ 0x23, 0x13, 0x08, 0x64, 0x62, /* % */
        0x36, 0x49, 0x55, 0x22, 0x50, /* & */ 0x00, 0x05, 0x03, 0x00, 0x00, /* ' */
        0x00, 0x1C, 0x22, 0x41, 0x00, /* ( */ 0x00, 0x41, 0x22, 0x1C, 0x00, /* ) */
        0x2A, 0x1C, 0x7F, 0x1C, 0x2A, /* * */ 0x08, 0x08, 0x3E, 0x08, 0x08, /* + */
        0x00, 0x50, 0x30, 0x00, 0x00, /* , */ 0x08, 0x08, 0x08, 0x08, 0x08, /* - */
        0x00, 0x60, 0x60, 0x00, 0x00, /* . */ 0x20, 0x10, 0x08, 0x04, 0x02, /* / */
        0x3E, 0x51, 0x49, 0x45, 0x3E, /* 0 */ 0x00, 0x42, 0x7F, 0x40, 0x00, /* 1 */
        0x42, 0x61, 0x51, 0x49, 0x46, /* 2 */ 0x21, 0x41, 0x45, 0x4B, 0x31, /* 3 */
        0x18, 0x14, 0x12, 0x7F, 0x10, /* 4 */ 0x27, 0x45, 0x45, 0x45, 0x39, /* 5 */
        0x3C, 0x4A, 0x49, 0x49, 0x30, /* 6 */ 0x01, 0x71, 0x09, 0x05, 0x03, /* 7 */
        0x36, 0x49, 0x49, 0x49, 0x36, /* 8 */ 0x06, 0x49, 0x49, 0x29, 0x1E, /* 9 */
        0x00, 0x36, 0x36, 0x00, 0x00, /* : */ 0x00, 0x56, 0x36, 0x00, 0x00, /* ; */
        0x08, 0x14, 0x22, 0x41, 0x00, /* < */ 0x14, 0x14, 0x14, 0x14, 0x14, /* = */
        0x00, 0x41, 0x22, 0x14, 0x08, /* > */ 0x02, 0x01, 0x51, 0x09, 0x06, /* ? */
        0x32, 0x49, 0x79, 0x41, 0x3E, /* @ */ 0x7E, 0x11, 0x11, 0x11, 0x7E, /* A */
        0x7F, 0x49, 0x49, 0x49, 0x36, /* B */ 0x3E, 0x41, 0x41, 0x41, 0x22, /* C */
        0x7F, 0x41, 0x41, 0x22, 0x1C, /* D */ 0x7F, 0x49, 0x49, 0x49, 0x41, /* E */
        0x7F, 0x09, 0x09, 0x09, 0x01, /* F */ 0x3E, 0x41, 0x49, 0x49, 0x7A, /* G */
        0x7F, 0x08, 0x08, 0x08, 0x7F, /* H */ 0x00, 0x41, 0x7F, 0x41, 0x00, /* I */
        0x20, 0x40, 0x41, 0x3F, 0x01, /* J */ 0x7F, 0x08, 0x14, 0x22, 0x41, /* K */
        0x7F, 0x40, 0x40, 0x40, 0x40, /* L */ 0x7F, 0x02, 0x0C, 0x02, 0x7F, /* M */
        0x7F, 0x04, 0x08, 0x10, 0x7F, /* N */ 0x3E, 0x41, 0x41, 0x41, 0x3E, /* O */
        0x7F, 0x09, 0x09, 0x09, 0x06, /* P */ 0x3E, 0x41, 0x51, 0x21, 0x5E, /* Q */
        0x7F, 0x09, 0x19, 0x29, 0x46, /* R */ 0x46, 0x49, 0x49, 0x49, 0x31, /* S */
        0x01, 0x01, 0x7F, 0x01, 0x01, /* T */ 0x3F, 0x40, 0x40, 0x40, 0x3F, /* U */
        0x1F, 0x20, 0x40, 0x20, 0x1F, /* V */ 0x3F, 0x40, 0x38, 0x40, 0x3F, /* W */
        0x63, 0x14, 0x08, 0x14, 0x63, /* X */ 0x07, 0x08, 0x70, 0x08, 0x07, /* Y */
        0x61, 0x51, 0x49, 0x45, 0x43, /* Z */ 0x00, 0x7F, 0x41, 0x41, 0x00, /* [ */
        0x02, 0x04, 0x08, 0x10, 0x20, /* \ */ 0x00, 0x41, 0x41, 0x7F, 0x00, /* ] */
        0x04, 0x02, 0x01, 0x02, 0x04, /* ^ */ 0x40, 0x40, 0x40, 0x40, 0x40, /* _ */
    };

    /// <summary>The middle dot <c>·</c>, the separator of captions (<c>SAME SEED · SAME ACTIONS</c>).</summary>
    private static readonly byte[] MiddleDot = { 0x00, 0x00, 0x08, 0x00, 0x00 };

    /// <summary>The five columns of a character's glyph.</summary>
    /// <param name="c">The character; lower case maps to upper case, unknown characters to a space.</param>
    /// <returns>A span over the five column bytes.</returns>
    private static ReadOnlySpan<byte> Glyph(char c) {
        if (c == '·')
            return MiddleDot;
        c = char.ToUpperInvariant(c);
        int index = c is >= ' ' and <= '_' ? c - ' ' : 0;
        return Glyphs.AsSpan(index * 5, 5);
    }

    /// <summary>Whether a font pixel of a character is lit.</summary>
    /// <param name="c">The character.</param>
    /// <param name="col">Column, 0-4.</param>
    /// <param name="row">Row, 0-6 (0 at the top).</param>
    /// <returns><see langword="true"/> when the dot is on.</returns>
    public static bool Lit(char c, int col, int row) => (Glyph(c)[col] & (1 << row)) != 0;

    /// <summary>The width in pixels of a text at a scale, without the trailing spacing column.</summary>
    /// <param name="text">The text.</param>
    /// <param name="scale">Pixels per font pixel.</param>
    /// <returns>The width; 0 for an empty text.</returns>
    public static float Measure(string text, float scale) => text.Length == 0 ? 0 : (text.Length * CellWidth - 1) * scale;

    /// <summary>
    ///     Draws a text with its top-left corner at (<paramref name="x"/>, <paramref name="y"/>). Fractional scales are
    ///     allowed (stamps pop in by scaling); each font pixel is a filled square.
    /// </summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The text.</param>
    /// <param name="x">Left edge.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Pixels per font pixel; at or below 0 nothing is drawn.</param>
    /// <param name="color">The colour.</param>
    /// <param name="alpha">Opacity (or strength when <paramref name="additive"/>).</param>
    /// <param name="additive">Adds light instead of painting, for glows.</param>
    public static void Draw(RgbImage img, string text, float x, float y, float scale, Rgb color, float alpha = 1, bool additive = false) {
        if (scale <= 0 || alpha <= 0)
            return;
        for (int i = 0; i < text.Length; i++) {
            float left = x + i * CellWidth * scale;
            for (int col = 0; col < 5; col++)
                for (int row = 0; row < Height; row++)
                    if (Lit(text[i], col, row))
                        Block(img, left + col * scale, y + row * scale, scale, color, alpha, additive);
        }
    }

    /// <summary>
    ///     Draws a text over a black outline one font pixel wide, so stamps and labels stay readable on any frame (white
    ///     ice, black space, green grass).
    /// </summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The text.</param>
    /// <param name="x">Left edge of the text (the outline extends one font pixel further).</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Pixels per font pixel.</param>
    /// <param name="color">The fill colour.</param>
    /// <param name="outline">The outline colour.</param>
    /// <param name="alpha">Opacity of both.</param>
    public static void DrawOutlined(RgbImage img, string text, float x, float y, float scale, Rgb color, Rgb outline, float alpha = 1) {
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                if (dx != 0 || dy != 0)
                    Draw(img, text, x + dx * scale, y + dy * scale, scale, outline, alpha);
        Draw(img, text, x, y, scale, color, alpha);
    }

    /// <summary>
    ///     Draws neon text: a wide faint halo, a tighter brighter one and a near-white core, all added as light, which the
    ///     bloom pass then softens further.
    /// </summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The text.</param>
    /// <param name="x">Left edge.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Pixels per font pixel.</param>
    /// <param name="color">The tube colour.</param>
    /// <param name="alpha">Overall strength, for fades.</param>
    public static void DrawNeon(RgbImage img, string text, float x, float y, float scale, Rgb color, float alpha = 1) {
        if (alpha <= 0)
            return;
        float r = MathF.Max(1, scale * 0.6f);
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++)
                Draw(img, text, x + dx * r, y + dy * r, scale, color, 0.18f * alpha, additive: true);
        Draw(img, text, x, y, scale, color, 0.9f * alpha);
        Draw(img, text, x, y, scale, Rgb.Lerp(color, Rgb.White, 0.55f), 0.55f * alpha);
    }

    /// <summary>
    ///     Draws the chrome logo lettering: an italic, extruded, gradient-filled version of the font with a hot-pink rim,
    ///     the 1980s chrome type the synthwave look is known for, built from font pixels so it stays pixel art.
    /// </summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The text, normally <c>GYM.NET</c>.</param>
    /// <param name="cx">Centre x of the text.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Pixels per font pixel (the title uses 15-18).</param>
    /// <param name="alpha">Opacity, for fades.</param>
    /// <remarks>
    ///     Each font row is shifted right by <c>0.22 * scale</c> per row above the baseline (the italic). The face's
    ///     colour depends only on the row: white to ice blue above the middle, a dark horizon band, then magenta to pink, so
    ///     every letter reflects the same sky. Eight one-pixel steps of dark purple below and to the right make the
    ///     extrusion.
    /// </remarks>
    public static void DrawChrome(RgbImage img, string text, float cx, float y, float scale, float alpha = 1) {
        if (alpha <= 0)
            return;
        float shear = 0.22f * scale;
        float x = cx - (Measure(text, scale) + shear * (Height - 1)) / 2;
        var rim = Rgb.Hex("#ff2fd0");
        var deep = Rgb.Hex("#2a0750");
        int depth = (int)MathF.Max(4, scale * 0.55f);
        // Back to front: extrusion, rim, face, so each layer only shows where the next one leaves room.
        for (int k = depth; k >= 1; k--)
            Letters(img, text, x + k, y + k, scale, shear, (row, sub) => Rgb.Lerp(deep, Rgb.Hex("#12002a"), (float)k / depth), alpha);
        float pad = MathF.Max(1, scale * 0.18f);
        Letters(img, text, x - pad, y - pad, scale, shear, (_, _) => rim, alpha, grow: 2 * pad);
        Letters(img, text, x, y, scale, shear, ChromeAt, alpha);
    }

    /// <summary>The chrome face colour at a position within the 7-row glyph height.</summary>
    /// <param name="row">The font row, 0-6.</param>
    /// <param name="sub">The position within that row's block, 0-1.</param>
    /// <returns>The face colour.</returns>
    private static Rgb ChromeAt(int row, float sub) {
        float t = (row + sub) / Height;
        if (t < 0.46f)
            return Rgb.Lerp(Rgb.Hex("#ffffff"), Rgb.Hex("#9fe9ff"), t / 0.46f);
        if (t < 0.54f)
            return Rgb.Lerp(Rgb.Hex("#3a1a7a"), Rgb.Hex("#1a0a3a"), (t - 0.46f) / 0.08f);
        return Rgb.Lerp(Rgb.Hex("#ff4fd8"), Rgb.Hex("#ffe6fb"), (t - 0.54f) / 0.46f);
    }

    /// <summary>Draws a text's font pixels as sheared blocks coloured per row, the shared core of the chrome layers.</summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The text.</param>
    /// <param name="x">Left edge of the baseline row.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Pixels per font pixel.</param>
    /// <param name="shear">Pixels each row moves right per row above the baseline.</param>
    /// <param name="color">The colour of a block's pixel by font row and the pixel's position within the block (0-1).</param>
    /// <param name="alpha">Opacity.</param>
    /// <param name="grow">Extra size of every block, for the rim drawn behind the face.</param>
    private static void Letters(RgbImage img, string text, float x, float y, float scale, float shear, Func<int, float, Rgb> color, float alpha,
        float grow = 0) {
        for (int i = 0; i < text.Length; i++) {
            float left = x + i * CellWidth * scale;
            for (int row = 0; row < Height; row++) {
                float rowShift = shear * (Height - 1 - row);
                for (int col = 0; col < 5; col++) {
                    if (!Lit(text[i], col, row))
                        continue;
                    float bx = left + col * scale + rowShift, by = y + row * scale, size = scale + grow;
                    int x0 = (int)MathF.Round(bx), y0 = (int)MathF.Round(by);
                    int x1 = (int)MathF.Round(bx + size), y1 = (int)MathF.Round(by + size);
                    for (int py = y0; py < y1; py++) {
                        Rgb c = color(row, (py - y0 + 0.5f) / Math.Max(1, y1 - y0));
                        for (int px = x0; px < x1; px++)
                            img.Blend(px, py, c, alpha);
                    }
                }
            }
        }
    }

    /// <summary>Fills one font pixel's square, snapped to whole pixels.</summary>
    /// <param name="img">The frame.</param>
    /// <param name="x">Left edge.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="size">Side length.</param>
    /// <param name="color">The colour.</param>
    /// <param name="alpha">Opacity or strength.</param>
    /// <param name="additive">Adds light instead of painting.</param>
    private static void Block(RgbImage img, float x, float y, float size, Rgb color, float alpha, bool additive) {
        int x0 = (int)MathF.Round(x), y0 = (int)MathF.Round(y);
        int x1 = Math.Max(x0 + 1, (int)MathF.Round(x + size)), y1 = Math.Max(y0 + 1, (int)MathF.Round(y + size));
        for (int py = y0; py < y1; py++)
            for (int px = x0; px < x1; px++)
                if (additive)
                    img.Add(px, py, color, alpha);
                else
                    img.Blend(px, py, color, alpha);
    }
}
