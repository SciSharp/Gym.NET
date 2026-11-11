using System;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Showreel;

/// <summary>
///     A colour with float channels on the 0-255 scale, so blends and glows can add and scale it without clamping at every
///     step; <see cref="RgbImage"/> clamps once, when it stores a pixel.
/// </summary>
/// <param name="R">Red, 0-255 (values above 255 are allowed in intermediate maths and clamp on store).</param>
/// <param name="G">Green, 0-255.</param>
/// <param name="B">Blue, 0-255.</param>
internal readonly record struct Rgb(float R, float G, float B) {
    /// <summary>Parses <c>#rrggbb</c>; the palette below is written in the hex the designs use.</summary>
    /// <param name="hex">Six hex digits, with or without a leading <c>#</c>.</param>
    /// <returns>The colour.</returns>
    /// <exception cref="FormatException">The text is not six hex digits.</exception>
    public static Rgb Hex(string hex) {
        string h = hex.TrimStart('#');
        if (h.Length != 6)
            throw new FormatException($"'{hex}' is not a #rrggbb colour");
        int v = Convert.ToInt32(h, 16);
        return new Rgb((v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF);
    }

    /// <summary>Linear interpolation between two colours.</summary>
    /// <param name="a">The colour at <paramref name="t"/> = 0.</param>
    /// <param name="b">The colour at <paramref name="t"/> = 1.</param>
    /// <param name="t">The position; not clamped, so values outside 0-1 extrapolate.</param>
    /// <returns>The blended colour.</returns>
    public static Rgb Lerp(Rgb a, Rgb b, float t) => new(a.R + (b.R - a.R) * t, a.G + (b.G - a.G) * t, a.B + (b.B - a.B) * t);

    /// <summary>Scales every channel, for dimming or for HDR-style overbright glows.</summary>
    /// <param name="c">The colour.</param>
    /// <param name="k">The factor.</param>
    /// <returns>The scaled colour (unclamped).</returns>
    public static Rgb operator *(Rgb c, float k) => new(c.R * k, c.G * k, c.B * k);

    /// <summary>White, the flash and highlight colour.</summary>
    public static Rgb White => new(255, 255, 255);

    /// <summary>Black, the outline colour of stamps and text.</summary>
    public static Rgb Black => new(0, 0, 0);
}

/// <summary>
///     An axis-aligned rectangle in float pixels: panels move with eased, sub-pixel motion, and only the drawing snaps to
///     whole pixels.
/// </summary>
/// <param name="X">Left edge.</param>
/// <param name="Y">Top edge.</param>
/// <param name="W">Width; zero or negative draws nothing.</param>
/// <param name="H">Height; zero or negative draws nothing.</param>
internal readonly record struct RectF(float X, float Y, float W, float H) {
    /// <summary>A rectangle from its centre and size, the way layouts are written.</summary>
    /// <param name="cx">Centre x.</param>
    /// <param name="cy">Centre y.</param>
    /// <param name="w">Width.</param>
    /// <param name="h">Height.</param>
    /// <returns>The rectangle.</returns>
    public static RectF Centered(float cx, float cy, float w, float h) => new(cx - w / 2, cy - h / 2, w, h);

    /// <summary>Centre x.</summary>
    public float Cx => X + W / 2;

    /// <summary>Centre y.</summary>
    public float Cy => Y + H / 2;

    /// <summary>Bottom edge.</summary>
    public float Bottom => Y + H;

    /// <summary>Right edge.</summary>
    public float Right => X + W;

    /// <summary>
    ///     The largest rectangle with the aspect ratio <paramref name="aspect"/> that fits inside this one, centred: how a
    ///     frame of any shape sits in a layout slot without distortion.
    /// </summary>
    /// <param name="aspect">Width divided by height of the content; must be positive.</param>
    /// <returns>The fitted rectangle.</returns>
    public RectF Fit(float aspect) {
        float w = W, h = W / aspect;
        if (h > H) {
            h = H;
            w = H * aspect;
        }
        return Centered(Cx, Cy, w, h);
    }

    /// <summary>Component-wise interpolation, for tweening a panel between two layout slots.</summary>
    /// <param name="a">The rectangle at <paramref name="t"/> = 0.</param>
    /// <param name="b">The rectangle at <paramref name="t"/> = 1.</param>
    /// <param name="t">The (already eased) position; values outside 0-1 overshoot, which the bouncy easings rely on.</param>
    /// <returns>The interpolated rectangle.</returns>
    public static RectF Lerp(RectF a, RectF b, float t) =>
        new(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t, a.W + (b.W - a.W) * t, a.H + (b.H - a.H) * t);
}

/// <summary>
///     An RGB24 raster, row-major with no padding (<c>3 * Width</c> bytes a row): the layout of a Gym.NET <c>rgb_array</c>
///     frame and of ffmpeg's <c>rgb24</c> input, so frames move between the envs, the compositor and the encoder without
///     conversion.
/// </summary>
/// <remarks>
///     Every drawing method clips to the raster, so callers may draw partly or wholly off-screen (panels fly in from outside
///     the frame). Not thread-safe; the compositor draws one frame at a time.
/// </remarks>
internal sealed class RgbImage {
    /// <summary>Creates a black raster.</summary>
    /// <param name="width">Width in pixels; must be positive.</param>
    /// <param name="height">Height in pixels; must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is not positive.</exception>
    public RgbImage(int width, int height) : this(width, height, new byte[checked(width * height * 3)]) { }

    /// <summary>Wraps existing RGB24 bytes without copying them; the raster then shares and mutates that array.</summary>
    /// <param name="width">Width in pixels; must be positive.</param>
    /// <param name="height">Height in pixels; must be positive.</param>
    /// <param name="pixels">Exactly <c>width * height * 3</c> bytes, row-major.</param>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is not positive.</exception>
    /// <exception cref="ArgumentException">The array has the wrong length.</exception>
    public RgbImage(int width, int height, byte[] pixels) {
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), width, "must be positive");
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), height, "must be positive");
        if (pixels.Length != width * height * 3)
            throw new ArgumentException($"expected {width * height * 3} bytes for {width}x{height}, got {pixels.Length}", nameof(pixels));
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>The RGB24 bytes; mutable and shared, never copied by the raster itself.</summary>
    public byte[] Pixels { get; }

    /// <summary>Width divided by height, the aspect a layout slot fits this raster to.</summary>
    public float Aspect => (float)Width / Height;

    /// <summary>Clamps a channel value to a byte, rounding to nearest.</summary>
    /// <param name="v">The value on the 0-255 scale, possibly out of range.</param>
    /// <returns>The stored byte.</returns>
    private static byte Clamp(float v) => v <= 0 ? (byte)0 : v >= 255 ? (byte)255 : (byte)(v + 0.5f);

    /// <summary>A deep copy, for frames that must survive the env rendering its next one.</summary>
    /// <returns>A raster with its own copy of the pixels.</returns>
    public RgbImage Clone() => new(Width, Height, (byte[])Pixels.Clone());

    /// <summary>Overwrites this raster with another of the same size (the cached background into the frame).</summary>
    /// <param name="source">The raster to copy.</param>
    /// <exception cref="ArgumentException">The sizes differ.</exception>
    public void CopyFrom(RgbImage source) {
        if (source.Width != Width || source.Height != Height)
            throw new ArgumentException($"size {source.Width}x{source.Height} differs from {Width}x{Height}", nameof(source));
        Buffer.BlockCopy(source.Pixels, 0, Pixels, 0, Pixels.Length);
    }

    /// <summary>Fills the whole raster with one colour.</summary>
    /// <param name="c">The colour.</param>
    public void Clear(Rgb c) {
        byte r = Clamp(c.R), g = Clamp(c.G), b = Clamp(c.B);
        for (int i = 0; i < Pixels.Length; i += 3) {
            Pixels[i] = r;
            Pixels[i + 1] = g;
            Pixels[i + 2] = b;
        }
    }

    /// <summary>Alpha-blends a colour over one pixel; pixels outside the raster are ignored.</summary>
    /// <param name="x">Column.</param>
    /// <param name="y">Row.</param>
    /// <param name="c">The colour.</param>
    /// <param name="a">Opacity, 0-1; values outside are clamped.</param>
    public void Blend(int x, int y, Rgb c, float a) {
        if ((uint)x >= (uint)Width || (uint)y >= (uint)Height || a <= 0)
            return;
        if (a > 1)
            a = 1;
        int i = (y * Width + x) * 3;
        Pixels[i] = Clamp(Pixels[i] + (c.R - Pixels[i]) * a);
        Pixels[i + 1] = Clamp(Pixels[i + 1] + (c.G - Pixels[i + 1]) * a);
        Pixels[i + 2] = Clamp(Pixels[i + 2] + (c.B - Pixels[i + 2]) * a);
    }

    /// <summary>
    ///     Adds a colour to one pixel (light, not paint): neon lines and glows brighten what is under them and saturate
    ///     toward white, the way an emissive tube reads on screen.
    /// </summary>
    /// <param name="x">Column.</param>
    /// <param name="y">Row.</param>
    /// <param name="c">The colour.</param>
    /// <param name="a">Strength; any non-negative value (the sum clamps at 255).</param>
    public void Add(int x, int y, Rgb c, float a) {
        if ((uint)x >= (uint)Width || (uint)y >= (uint)Height || a <= 0)
            return;
        int i = (y * Width + x) * 3;
        Pixels[i] = Clamp(Pixels[i] + c.R * a);
        Pixels[i + 1] = Clamp(Pixels[i + 1] + c.G * a);
        Pixels[i + 2] = Clamp(Pixels[i + 2] + c.B * a);
    }

    /// <summary>Alpha-blends a filled rectangle, snapped to whole pixels and clipped.</summary>
    /// <param name="r">The rectangle.</param>
    /// <param name="c">The colour.</param>
    /// <param name="a">Opacity, 0-1.</param>
    public void FillRect(RectF r, Rgb c, float a = 1) {
        int x0 = Math.Max(0, (int)MathF.Round(r.X)), y0 = Math.Max(0, (int)MathF.Round(r.Y));
        int x1 = Math.Min(Width, (int)MathF.Round(r.Right)), y1 = Math.Min(Height, (int)MathF.Round(r.Bottom));
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++)
                Blend(x, y, c, a);
    }

    /// <summary>Draws a rectangle's outline of a given thickness inside its edges.</summary>
    /// <param name="r">The rectangle.</param>
    /// <param name="c">The colour.</param>
    /// <param name="thickness">The line width in pixels.</param>
    /// <param name="a">Opacity, 0-1.</param>
    public void StrokeRect(RectF r, Rgb c, float thickness, float a = 1) {
        FillRect(new RectF(r.X, r.Y, r.W, thickness), c, a);
        FillRect(new RectF(r.X, r.Bottom - thickness, r.W, thickness), c, a);
        FillRect(new RectF(r.X, r.Y + thickness, thickness, r.H - 2 * thickness), c, a);
        FillRect(new RectF(r.Right - thickness, r.Y + thickness, thickness, r.H - 2 * thickness), c, a);
    }

    /// <summary>
    ///     Adds a soft neon halo around a rectangle: full strength on its edge, falling off quadratically to nothing at
    ///     <paramref name="radius"/> pixels outside, plus a faint inner rim so the border reads as a lit tube.
    /// </summary>
    /// <param name="r">The rectangle whose edge glows.</param>
    /// <param name="c">The glow colour.</param>
    /// <param name="radius">How far the halo reaches outside the edge, in pixels.</param>
    /// <param name="strength">The additive strength at the edge (1 adds the full colour).</param>
    /// <remarks>
    ///     Cost is proportional to the halo band's area (perimeter times radius), not to the rectangle's, so large panels
    ///     glow as cheaply as small ones.
    /// </remarks>
    public void Glow(RectF r, Rgb c, float radius, float strength) {
        int x0 = Math.Max(0, (int)(r.X - radius)), x1 = Math.Min(Width, (int)(r.Right + radius) + 1);
        int y0 = Math.Max(0, (int)(r.Y - radius)), y1 = Math.Min(Height, (int)(r.Bottom + radius) + 1);
        const float innerRim = 6;
        for (int y = y0; y < y1; y++) {
            float py = y + 0.5f;
            float dy = py < r.Y ? r.Y - py : py > r.Bottom ? py - r.Bottom : 0;
            // Rows well inside the rectangle only need their two side bands: skip the interior span in one jump.
            bool interiorRow = dy == 0 && py > r.Y + innerRim && py < r.Bottom - innerRim;
            for (int x = x0; x < x1; x++) {
                float px = x + 0.5f;
                if (interiorRow && px > r.X + innerRim && px < r.Right - innerRim) {
                    // Jump to the last interior column (its centre still left of the right rim). Computed from pixel
                    // centres and never below x: a jump that landed one column back would loop forever.
                    x = Math.Max(x, (int)MathF.Ceiling(r.Right - innerRim - 0.5f) - 1);
                    continue;
                }
                float dx = px < r.X ? r.X - px : px > r.Right ? px - r.Right : 0;
                float k;
                if (dx > 0 || dy > 0) {
                    float d = MathF.Sqrt(dx * dx + dy * dy);
                    if (d >= radius)
                        continue;
                    float u = 1 - d / radius;
                    k = u * u;
                } else {
                    // Inside: distance to the nearest edge, a faint rim that fades over innerRim pixels.
                    float d = MathF.Min(MathF.Min(px - r.X, r.Right - px), MathF.Min(py - r.Y, r.Bottom - py));
                    if (d >= innerRim)
                        continue;
                    float u = 1 - d / innerRim;
                    k = 0.35f * u * u;
                }
                Add(x, y, c, strength * k);
            }
        }
    }

    /// <summary>
    ///     Draws another raster scaled into a destination rectangle with bilinear filtering, optionally squeezed
    ///     horizontally about its centre (a card flip) and tinted toward a colour.
    /// </summary>
    /// <param name="src">The source raster, an env frame.</param>
    /// <param name="dst">Where it lands; any size, partly or wholly off-screen.</param>
    /// <param name="alpha">Opacity, 0-1.</param>
    /// <param name="squeeze">Horizontal scale about the centre, 0-1: 1 draws normally, 0 draws nothing.</param>
    /// <param name="tint">A colour to blend toward, or <see langword="null"/>.</param>
    /// <param name="tintAmount">How far toward <paramref name="tint"/>, 0-1.</param>
    /// <remarks>
    ///     Bilinear filtering is used for every scale. The video never shrinks an env frame below half size, where bilinear
    ///     starts to drop thin lines (CartPole's track is one pixel wide); the gallery's tiny loops are scaled by ffmpeg's
    ///     area filter instead.
    /// </remarks>
    public void DrawImage(RgbImage src, RectF dst, float alpha = 1, float squeeze = 1, Rgb? tint = null, float tintAmount = 0) {
        if (alpha <= 0 || squeeze <= 0 || dst.W <= 0 || dst.H <= 0)
            return;
        float w = dst.W * squeeze;
        var d = RectF.Centered(dst.Cx, dst.Cy, w, dst.H);
        int x0 = Math.Max(0, (int)MathF.Floor(d.X)), x1 = Math.Min(Width, (int)MathF.Ceiling(d.Right));
        int y0 = Math.Max(0, (int)MathF.Floor(d.Y)), y1 = Math.Min(Height, (int)MathF.Ceiling(d.Bottom));
        float sx = src.Width / d.W, sy = src.Height / d.H;
        byte[] s = src.Pixels;
        int sw = src.Width, sh = src.Height;
        bool opaque = alpha >= 1 && tint is null;
        for (int y = y0; y < y1; y++) {
            // Pixel centres map to source coordinates; -0.5 aligns centres so a 1:1 draw reproduces the source exactly.
            float fy = (y + 0.5f - d.Y) * sy - 0.5f;
            int iy = (int)MathF.Floor(fy);
            float ty = fy - iy;
            int ya = Math.Clamp(iy, 0, sh - 1), yb = Math.Clamp(iy + 1, 0, sh - 1);
            int rowA = ya * sw * 3, rowB = yb * sw * 3;
            for (int x = x0; x < x1; x++) {
                float fx = (x + 0.5f - d.X) * sx - 0.5f;
                int ix = (int)MathF.Floor(fx);
                float tx = fx - ix;
                int xa = Math.Clamp(ix, 0, sw - 1) * 3, xb = Math.Clamp(ix + 1, 0, sw - 1) * 3;
                int o = (y * Width + x) * 3;
                for (int ch = 0; ch < 3; ch++) {
                    float top = s[rowA + xa + ch] + (s[rowA + xb + ch] - s[rowA + xa + ch]) * tx;
                    float bottom = s[rowB + xa + ch] + (s[rowB + xb + ch] - s[rowB + xa + ch]) * tx;
                    float v = top + (bottom - top) * ty;
                    if (tint is Rgb t)
                        v += ((ch == 0 ? t.R : ch == 1 ? t.G : t.B) - v) * tintAmount;
                    Pixels[o + ch] = opaque ? Clamp(v) : Clamp(Pixels[o + ch] + (v - Pixels[o + ch]) * alpha);
                }
            }
        }
    }

    /// <summary>
    ///     Draws an anti-aliased line (Xiaolin Wu's algorithm) in additive light: the grid floor's neon lines, which must stay
    ///     smooth as they converge on the horizon.
    /// </summary>
    /// <param name="x0">Start x.</param>
    /// <param name="y0">Start y.</param>
    /// <param name="x1">End x.</param>
    /// <param name="y1">End y.</param>
    /// <param name="c">The colour.</param>
    /// <param name="a">Strength (additive).</param>
    public void AddLine(float x0, float y0, float x1, float y1, Rgb c, float a) {
        bool steep = MathF.Abs(y1 - y0) > MathF.Abs(x1 - x0);
        if (steep) {
            (x0, y0) = (y0, x0);
            (x1, y1) = (y1, x1);
        }
        if (x0 > x1) {
            (x0, x1) = (x1, x0);
            (y0, y1) = (y1, y0);
        }
        float dx = x1 - x0, gradient = dx == 0 ? 1 : (y1 - y0) / dx;
        // Clip the major axis to the raster first, so a line that starts far off-screen costs only its visible span.
        float limit = steep ? Height : Width;
        float start = MathF.Max(MathF.Floor(x0), -1), end = MathF.Min(MathF.Ceiling(x1), limit);
        for (float x = start; x <= end; x++) {
            float y = y0 + gradient * (x - x0);
            int iy = (int)MathF.Floor(y);
            float f = y - iy;
            if (steep) {
                Add(iy, (int)x, c, a * (1 - f));
                Add(iy + 1, (int)x, c, a * f);
            } else {
                Add((int)x, iy, c, a * (1 - f));
                Add((int)x, iy + 1, c, a * f);
            }
        }
    }

    /// <summary>Writes the raster as a PNG (stills, posters, the site's logo).</summary>
    /// <param name="path">The file; its folder must exist.</param>
    /// <exception cref="IOException">The file cannot be written.</exception>
    public void SavePng(string path) {
        using var image = Image.LoadPixelData<Rgb24>(Pixels, Width, Height);
        image.SaveAsPng(path);
    }
}
