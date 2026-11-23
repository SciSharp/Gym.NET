using System;

namespace Gym.Showreel;

/// <summary>A backdrop the compositor draws every frame on before the panels: a pure function of time.</summary>
internal interface IBackdrop {
    /// <summary>Draws the backdrop for a time into a frame of the backdrop's size.</summary>
    /// <param name="target">The frame; fully overwritten.</param>
    /// <param name="t">Seconds since the start of the video, which drives every animation.</param>
    /// <param name="sunRise">0 (the sun below the horizon) to 1 (risen), for the intro.</param>
    /// <exception cref="ArgumentException">The frame's size differs from the backdrop's.</exception>
    void Draw(RgbImage target, double t, float sunRise = 1);
}

/// <summary>The backdrops the showreel can draw, by the name <c>--backdrop</c> takes.</summary>
internal static class Backdrops {
    /// <summary>The backdrop the video and the README cut use unless <c>--backdrop</c> says otherwise (DR-108).</summary>
    public const string Default = "beach";

    /// <summary>Creates a backdrop by name.</summary>
    /// <param name="name"><c>beach</c> (the 16-bit dusk beach, DR-108) or <c>synthwave</c> (the sunset grid of DR-59).</param>
    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    /// <returns>The backdrop.</returns>
    /// <exception cref="ArgumentException">The name is neither.</exception>
    public static IBackdrop Create(string name, int width, int height) => name switch {
        "beach" => new DuskBeach(width, height),
        "synthwave" => new Synthwave(width, height),
        _ => throw new ArgumentException($"unknown backdrop '{name}': beach or synthwave"),
    };
}

/// <summary>
///     The 16-bit dusk beach (DR-108): pixel art drawn at 320x180 and scaled up four times without smoothing, so every
///     pixel is a crisp 4x4 block. An ordered-dithered sunset sky, the sun half-sunk in the sea with gaps rolling down
///     it, a far skyline whose lights blink on the horizon, sea bands with the sun's reflection shimmering on them, and
///     two palm silhouettes swaying at the edges.
/// </summary>
/// <remarks>
///     <para>
///     The maintainer chose it from eight candidates animated live under the video's panels, 2026-11-22; this class
///     draws the candidate's design (<c>bg_beach.html</c> of that question) in C#, with its random layout drawn from a
///     fixed seed so every run paints identical pixels.
///     </para>
///     <para>
///     The sky is static and computed once; the rest moves with <c>t</c> only, so any frame renders in any order. Its
///     pixel blocks are whole at 1280x720; the README cut's encoder scales the frame to 800 wide, which resamples the
///     blocks, as it resamples every panel.
///     </para>
/// </remarks>
internal sealed class DuskBeach : IBackdrop {
    /// <summary>The pixel-art canvas's width: the frame width over <see cref="Scale"/>.</summary>
    private const int LowWidth = 320;

    /// <summary>The pixel-art canvas's height.</summary>
    private const int LowHeight = 180;

    /// <summary>Frame pixels per art pixel, each way.</summary>
    private const int Scale = 4;

    /// <summary>The horizon's art row: the sea starts below it, the sun sits on it.</summary>
    private const int Horizon = 108;

    /// <summary>The sun's centre column and radius, in art pixels.</summary>
    private const int SunX = 160, SunRadius = 30;

    /// <summary>The 4x4 Bayer matrix the sky's bands are dithered with, as 16-bit consoles faked gradients.</summary>
    private static readonly int[,] Bayer = { { 0, 8, 2, 10 }, { 12, 4, 14, 6 }, { 3, 11, 1, 9 }, { 15, 7, 13, 5 } };

    /// <summary>The sky's bands, top to horizon: night purple to the sunset's orange.</summary>
    private static readonly Rgb[] SkyBands = {
        Rgb.Hex("#1c0b45"), Rgb.Hex("#3a1260"), Rgb.Hex("#6a1b72"), Rgb.Hex("#a02a72"), Rgb.Hex("#d8456a"), Rgb.Hex("#f8754e"), Rgb.Hex("#ffb15a"),
    };

    /// <summary>The sea's bands, from the horizon down, each 15 art rows.</summary>
    private static readonly Rgb[] SeaBands = { Rgb.Hex("#3a1a6a"), Rgb.Hex("#2a1660"), Rgb.Hex("#1e1254"), Rgb.Hex("#160e46"), Rgb.Hex("#100a38") };

    /// <summary>The palms' and the skyline's silhouette colours.</summary>
    private static readonly Rgb Palm = Rgb.Hex("#1a0726"), Skyline = Rgb.Hex("#2a0f45");

    /// <summary>The fronds of a palm: each one's angle (radians, 0 pointing right) and length in art pixels.</summary>
    private static readonly (float Angle, int Length)[] Fronds = { (-2.8f, 26), (-2.2f, 30), (-1.6f, 22), (-0.9f, 30), (-0.3f, 26), (0.4f, 20) };

    /// <summary>Frame width and height.</summary>
    private readonly int _w, _h;

    /// <summary>The dithered sky, computed once, RGB bytes at art resolution.</summary>
    private readonly byte[] _sky = new byte[LowWidth * Horizon * 3];

    /// <summary>The art canvas the moving parts are painted on each frame, RGB bytes.</summary>
    private readonly byte[] _low = new byte[LowWidth * LowHeight * 3];

    /// <summary>One frame row, the art row widened four times, copied into four frame rows.</summary>
    private readonly byte[] _row;

    /// <summary>The reflection's streaks: depth (0 near the horizon, 1 at the bottom), sideways offset, phase.</summary>
    private readonly (float Depth, float Offset, float Phase)[] _shimmer;

    /// <summary>The skyline's window lights: position and blink phase.</summary>
    private readonly (int X, int Y, float Phase)[] _lights;

    /// <summary>Creates the backdrop for a frame size of four times 320x180.</summary>
    /// <param name="width">Frame width; 1280.</param>
    /// <param name="height">Frame height; 720.</param>
    /// <exception cref="ArgumentException">The size is not four times the art's.</exception>
    public DuskBeach(int width, int height) {
        if (width != LowWidth * Scale || height != LowHeight * Scale)
            throw new ArgumentException($"the dusk beach draws {LowWidth * Scale}x{LowHeight * Scale} frames, not {width}x{height}");
        _w = width;
        _h = height;
        _row = new byte[width * 3];
        for (int y = 0; y < Horizon; y++) {
            float f = (float)y / Horizon * (SkyBands.Length - 1);
            int band = (int)f;
            float frac = f - band;
            for (int x = 0; x < LowWidth; x++) {
                // A pixel takes the next band where the band fraction beats its Bayer threshold: an even ordered dither.
                Rgb c = frac * 16 > Bayer[y % 4, x % 4] ? SkyBands[Math.Min(band + 1, SkyBands.Length - 1)] : SkyBands[band];
                Put(_sky, x, y, c);
            }
        }
        var rng = new Random(1991);
        _shimmer = new (float, float, float)[70];
        for (int i = 0; i < _shimmer.Length; i++)
            _shimmer[i] = ((float)rng.NextDouble(), (float)rng.NextDouble() - 0.5f, (float)(rng.NextDouble() * Math.PI * 2));
        _lights = new (int, int, float)[26];
        for (int i = 0; i < _lights.Length; i++)
            _lights[i] = (rng.Next(0, 110), Horizon - 2 - rng.Next(0, 9), (float)(rng.NextDouble() * Math.PI * 2));
    }

    /// <inheritdoc />
    public void Draw(RgbImage target, double t, float sunRise = 1) {
        if (target.Width != _w || target.Height != _h)
            throw new ArgumentException($"the frame is {target.Width}x{target.Height}, the backdrop {_w}x{_h}", nameof(target));
        Buffer.BlockCopy(_sky, 0, _low, 0, _sky.Length);
        DrawSun(t, sunRise);
        // The far skyline left of the sun, its lights blinking on and off.
        for (int x = 0; x < 110; x += 7) {
            int h = 4 + x * 37 % 11;
            Fill(x, Horizon - h, 6, h, Skyline);
        }
        var light = Rgb.Hex("#ffd27a");
        foreach (var (x, y, phase) in _lights)
            if (MathF.Sin((float)t * 1.5f + phase) > -0.3f)
                Fill(x, y, 1, 1, light);
        // The sea is drawn after the sun, so a sun below the horizon (the intro's rise) is hidden by it.
        for (int y = Horizon; y < LowHeight; y++)
            Fill(0, y, LowWidth, 1, SeaBands[Math.Min(SeaBands.Length - 1, (y - Horizon) / 15)]);
        DrawReflection(t, sunRise);
        DrawPalm(14, MathF.Sin((float)t * 1.2f), flip: false);
        DrawPalm(LowWidth - 17, MathF.Sin((float)t * 1.2f + 1), flip: true);
        Upscale(target);
    }

    /// <summary>
    ///     The sun: a half disc on the horizon in three bands, its lower rows cut by gaps that roll downward; the intro
    ///     raises it from below the sea.
    /// </summary>
    /// <param name="t">Seconds.</param>
    /// <param name="rise">0 (sunk) to 1 (on the horizon).</param>
    private void DrawSun(double t, float rise) {
        int sy = Horizon + (int)MathF.Round((1 - rise) * (SunRadius + 2));
        int roll = (int)Math.Floor(t * 4);
        Rgb top = Rgb.Hex("#fff0a0"), middle = Rgb.Hex("#ffd070"), bottom = Rgb.Hex("#ffa050");
        for (int y = -SunRadius; y < 0; y++) {
            int row = sy + y;
            if (row >= Horizon)
                continue;
            int half = (int)Math.Floor(Math.Sqrt(SunRadius * SunRadius - y * y));
            // One row in five of the lower half is a gap, the pattern moving down four rows a second.
            if (y > -16 && ((y + roll) % 5 + 5) % 5 == 0)
                continue;
            Fill(SunX - half, row, 2 * half, 1, y < -20 ? top : y < -10 ? middle : bottom);
        }
    }

    /// <summary>
    ///     The sun's reflection: short streaks under it that shimmer on and off and spread and lengthen toward the bottom,
    ///     fewer of them while the sun is still rising.
    /// </summary>
    /// <param name="t">Seconds.</param>
    /// <param name="rise">0 (no reflection) to 1 (all streaks).</param>
    private void DrawReflection(double t, float rise) {
        Rgb near = Rgb.Hex("#ffe09a"), far = Rgb.Hex("#ffa060");
        for (int i = 0; i < _shimmer.Length; i++) {
            if (i >= _shimmer.Length * rise)
                break;
            var (d, offset, phase) = _shimmer[i];
            if (MathF.Sin((float)t * 5 + phase) <= -0.4f)
                continue;
            int y = Horizon + 2 + (int)(d * d * (LowHeight - Horizon - 4));
            float spread = 6 + d * 38;
            int x = (int)MathF.Round(SunX + offset * spread * 2 + MathF.Sin((float)t * 3 + phase) * 2);
            int length = 2 + (int)((1 - d) * 2 + d * 7);
            Fill(x - length / 2, y, length, 1, d < 0.4f ? near : far);
        }
    }

    /// <summary>A palm silhouette: a curved trunk rising from the bottom edge and six fronds that sway with it.</summary>
    /// <param name="x">The trunk's foot, in art columns.</param>
    /// <param name="sway">The sway, -1 to 1.</param>
    /// <param name="flip">Whether it leans left (the right-hand palm) rather than right.</param>
    private void DrawPalm(int x, float sway, bool flip) {
        int f = flip ? -1 : 1;
        for (int i = 0; i < 70; i++) {
            float u = i / 70f;
            int px = x + f * (int)MathF.Round(u * u * 14 + MathF.Sin(u * 3) * 2);
            Fill(px, LowHeight - i, 3, 1, Palm);
        }
        int tx = x + f * 16, ty = LowHeight - 70;
        foreach (var (angle, length) in Fronds)
            for (int k = 0; k < length; k++) {
                float u = (float)k / length, a = angle + sway * 0.15f;
                int px = tx + f * (int)MathF.Round(MathF.Cos(a) * k), py = ty + (int)MathF.Round(MathF.Sin(a) * k + u * u * 10);
                Fill(px, py, 2, 1, Palm);
            }
    }

    /// <summary>Fills an art-pixel rectangle, clipped to the canvas.</summary>
    /// <param name="x">Left column.</param>
    /// <param name="y">Top row.</param>
    /// <param name="w">Width.</param>
    /// <param name="h">Height.</param>
    /// <param name="c">The colour.</param>
    private void Fill(int x, int y, int w, int h, Rgb c) {
        int x0 = Math.Max(0, x), x1 = Math.Min(LowWidth, x + w), y0 = Math.Max(0, y), y1 = Math.Min(LowHeight, y + h);
        for (int yy = y0; yy < y1; yy++)
            for (int xx = x0; xx < x1; xx++)
                Put(_low, xx, yy, c);
    }

    /// <summary>Stores one art pixel, clamping the colour to bytes.</summary>
    /// <param name="buffer">The art buffer, <see cref="LowWidth"/> pixels a row.</param>
    /// <param name="x">Column.</param>
    /// <param name="y">Row.</param>
    /// <param name="c">The colour.</param>
    private static void Put(byte[] buffer, int x, int y, Rgb c) {
        int o = (y * LowWidth + x) * 3;
        buffer[o] = (byte)Math.Clamp(c.R, 0, 255);
        buffer[o + 1] = (byte)Math.Clamp(c.G, 0, 255);
        buffer[o + 2] = (byte)Math.Clamp(c.B, 0, 255);
    }

    /// <summary>Copies the art canvas into the frame, each art pixel a 4x4 block (nearest neighbour, no smoothing).</summary>
    /// <param name="target">The frame.</param>
    private void Upscale(RgbImage target) {
        byte[] p = target.Pixels;
        for (int y = 0; y < LowHeight; y++) {
            // Widen the art row once, then copy it into the four frame rows it covers.
            for (int x = 0; x < LowWidth; x++) {
                int s = (y * LowWidth + x) * 3;
                for (int k = 0; k < Scale; k++) {
                    int d = ((x * Scale) + k) * 3;
                    _row[d] = _low[s];
                    _row[d + 1] = _low[s + 1];
                    _row[d + 2] = _low[s + 2];
                }
            }
            for (int k = 0; k < Scale; k++)
                Buffer.BlockCopy(_row, 0, p, (y * Scale + k) * _row.Length, _row.Length);
        }
    }
}
