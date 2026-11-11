using System;

namespace Gym.Showreel;

/// <summary>
///     The synthwave backdrop every frame is drawn on: a gradient sky with twinkling stars, a striped sun whose stripes slide
///     down, neon-rimmed mountains on the horizon and a perspective grid floor scrolling toward the viewer.
/// </summary>
/// <remarks>
///     <para>
///     The static parts (sky and floor gradients, horizon haze, the mountains) are computed once in the constructor; each
///     frame copies them and draws the moving parts. Everything is a pure function of time, so a frame can be rendered in
///     any order and the README loop can start and end on matching backgrounds: the grid repeats every
///     <see cref="GridPeriod"/> seconds and the sun's stripes every <see cref="StripePeriod"/>.
///     </para>
///     <para>
///     The palette and the procedural mountains come from fixed seeds and constants, so two runs draw identical pixels.
///     </para>
/// </remarks>
internal sealed class Synthwave {
    /// <summary>Seconds for the grid floor to scroll one line spacing: the floor's loop period.</summary>
    public const double GridPeriod = 0.75;

    /// <summary>Seconds for the sun's stripes to slide one stripe pitch.</summary>
    public const double StripePeriod = 1.5;

    /// <summary>Width of the frames this backdrop draws.</summary>
    private readonly int _w;

    /// <summary>Height of the frames this backdrop draws.</summary>
    private readonly int _h;

    /// <summary>The sky, floor and haze gradients, copied into every frame first.</summary>
    private readonly RgbImage _base;

    /// <summary>The mountains' colour, valid where <see cref="_mountainAlpha"/> is positive.</summary>
    private readonly RgbImage _mountainColor;

    /// <summary>The mountains' coverage per pixel (0 clear, 1 solid), drawn over the sun so it sets behind them.</summary>
    private readonly float[] _mountainAlpha;

    /// <summary>Star positions, sizes and twinkle phases, fixed by a seed.</summary>
    private readonly (float X, float Y, float Size, float Phase, float Speed)[] _stars;

    /// <summary>Creates the backdrop for one frame size.</summary>
    /// <param name="width">Frame width in pixels.</param>
    /// <param name="height">Frame height in pixels.</param>
    public Synthwave(int width, int height) {
        _w = width;
        _h = height;
        Horizon = MathF.Round(height * 0.6f);
        _base = new RgbImage(width, height);
        PaintGradients();
        _mountainColor = new RgbImage(width, height);
        _mountainAlpha = new float[width * height];
        PaintMountains();
        var rng = new Random(20261111);
        _stars = new (float, float, float, float, float)[150];
        for (int i = 0; i < _stars.Length; i++)
            _stars[i] = ((float)rng.NextDouble() * width, (float)rng.NextDouble() * (Horizon - 140), rng.NextDouble() < 0.15 ? 2 : 1,
                (float)(rng.NextDouble() * Math.PI * 2), 1.5f + (float)rng.NextDouble() * 3);
    }

    /// <summary>The horizon's row: the floor starts below it, the sun sets behind it.</summary>
    public float Horizon { get; }

    /// <summary>
    ///     Draws the backdrop for time <paramref name="t"/> into a frame of the constructor's size.
    /// </summary>
    /// <param name="target">The frame; fully overwritten.</param>
    /// <param name="t">Seconds since the start of the video, which drives every animation.</param>
    /// <param name="sunRise">0 (sun below the horizon) to 1 (risen), for the intro.</param>
    /// <exception cref="ArgumentException">The frame's size differs from the backdrop's.</exception>
    public void Draw(RgbImage target, double t, float sunRise = 1) {
        target.CopyFrom(_base);
        DrawStars(target, t);
        DrawSun(target, t, sunRise);
        // The mountains go over the sun (it sets behind them) and under the grid lines' glow.
        for (int i = 0; i < _mountainAlpha.Length; i++) {
            float a = _mountainAlpha[i];
            if (a <= 0)
                continue;
            int o = i * 3;
            byte[] m = _mountainColor.Pixels, p = target.Pixels;
            p[o] = (byte)(p[o] + (m[o] - p[o]) * a);
            p[o + 1] = (byte)(p[o + 1] + (m[o + 1] - p[o + 1]) * a);
            p[o + 2] = (byte)(p[o + 2] + (m[o + 2] - p[o + 2]) * a);
        }
        DrawGrid(target, t);
    }

    /// <summary>Paints the static sky and floor gradients and the pink haze along the horizon.</summary>
    private void PaintGradients() {
        Rgb[] sky = { Rgb.Hex("#05000f"), Rgb.Hex("#160634"), Rgb.Hex("#3b0d5c"), Rgb.Hex("#8a1b6e"), Rgb.Hex("#ff4f86") };
        float[] skyStops = { 0f, 0.35f, 0.65f, 0.88f, 1f };
        for (int y = 0; y < _h; y++) {
            Rgb c;
            if (y < Horizon) {
                c = Gradient(sky, skyStops, y / Horizon);
            } else {
                float f = (y - Horizon) / (_h - Horizon);
                c = f < 0.12f ? Rgb.Lerp(Rgb.Hex("#3a0c55"), Rgb.Hex("#14002c"), f / 0.12f) : Rgb.Lerp(Rgb.Hex("#14002c"), Rgb.Hex("#06000f"), (f - 0.12f) / 0.88f);
            }
            // A soft pink band straddling the horizon, the haze the sun lights.
            float d = (y - Horizon) / 20f;
            c = Rgb.Lerp(c, Rgb.Hex("#ff6a9a"), 0.4f * MathF.Exp(-d * d));
            for (int x = 0; x < _w; x++)
                _base.Blend(x, y, c, 1);
        }
    }

    /// <summary>
    ///     Paints two ridges of mountains from midpoint-displacement noise with a fixed seed: low in the middle so the sun
    ///     shows between them, with a neon rim along each ridge line.
    /// </summary>
    private void PaintMountains() {
        var rng = new Random(1980);
        (Rgb Body, Rgb Rim, float Height, float Rough)[] layers = {
            (Rgb.Hex("#2b0b4e"), Rgb.Hex("#ff4fd8"), 150, 0.55f),
            (Rgb.Hex("#14032a"), Rgb.Hex("#39f3ff"), 95, 0.6f),
        };
        foreach (var (body, rim, height, rough) in layers) {
            float[] ridge = Ridge(rng, _w, rough);
            for (int x = 0; x < _w; x++) {
                // Valley in the middle third: the profile scales down toward the centre.
                float centre = MathF.Abs(x - _w / 2f) / (_w / 2f);
                float valley = MathF.Min(1, MathF.Max(0.08f, (centre - 0.12f) / 0.45f));
                float top = Horizon - ridge[x] * height * valley;
                for (int y = (int)MathF.Floor(top); y < Horizon; y++) {
                    if (y < 0)
                        continue;
                    float cover = MathF.Min(1, y + 1 - top);
                    int i = y * _w + x;
                    // Rim: the first two pixels under the ridge carry the neon colour, then the body darkens downward.
                    float depth = y - top;
                    Rgb c = depth < 2 ? rim : Rgb.Lerp(body, Rgb.Hex("#090014"), MathF.Min(1, depth / (height * 1.2f)));
                    _mountainColor.Blend(x, y, c, 1);
                    _mountainAlpha[i] = MathF.Max(_mountainAlpha[i], cover);
                }
            }
        }
    }

    /// <summary>A 0-1 ridge profile across the width from midpoint displacement.</summary>
    /// <param name="rng">The seeded generator.</param>
    /// <param name="width">How many columns.</param>
    /// <param name="rough">How much each halving keeps of the displacement (higher is more jagged).</param>
    /// <returns>One height per column.</returns>
    private static float[] Ridge(Random rng, int width, float rough) {
        int n = 1;
        while (n < width)
            n <<= 1;
        var h = new float[n + 1];
        h[0] = (float)rng.NextDouble();
        h[n] = (float)rng.NextDouble();
        float amp = 1;
        for (int step = n; step > 1; step >>= 1) {
            for (int i = step / 2; i < n; i += step)
                h[i] = (h[i - step / 2] + h[i + step / 2]) / 2 + ((float)rng.NextDouble() - 0.5f) * amp;
            amp *= rough;
        }
        float min = float.MaxValue, max = float.MinValue;
        foreach (float v in h) {
            min = MathF.Min(min, v);
            max = MathF.Max(max, v);
        }
        var r = new float[width];
        for (int x = 0; x < width; x++)
            r[x] = (h[x] - min) / (max - min);
        return r;
    }

    /// <summary>Twinkling stars: each brightens and dims on its own sine.</summary>
    /// <param name="img">The frame.</param>
    /// <param name="t">Seconds.</param>
    private void DrawStars(RgbImage img, double t) {
        foreach (var (x, y, size, phase, speed) in _stars) {
            float b = 0.45f + 0.55f * (0.5f + 0.5f * MathF.Sin((float)(t * speed) + phase));
            img.FillRect(new RectF(x, y, size, size), new Rgb(255, 235, 255), b);
        }
    }

    /// <summary>
    ///     The sun: a yellow-to-pink disc behind a soft glow, its lower half cut by horizontal gaps that widen toward the
    ///     bottom and slide down over time.
    /// </summary>
    /// <param name="img">The frame.</param>
    /// <param name="t">Seconds.</param>
    /// <param name="rise">0 (sunk) to 1 (risen).</param>
    private void DrawSun(RgbImage img, double t, float rise) {
        float r = _h * 0.22f;
        float cx = _w / 2f, cy = Horizon - r * 0.35f + (1 - rise) * r * 1.4f;
        // Glow first: an additive halo, so the sky around the sun warms up.
        float gr = r * 1.7f;
        var glow = Rgb.Hex("#ff4f8b");
        for (int y = (int)(cy - gr); y < (int)MathF.Min(Horizon, cy + gr); y++)
            for (int x = (int)(cx - gr); x < (int)(cx + gr); x++) {
                float d = MathF.Sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
                if (d > r && d < gr) {
                    float u = 1 - (d - r) / (gr - r);
                    img.Add(x, y, glow, 0.32f * u * u);
                }
            }
        Rgb top = Rgb.Hex("#fff07a"), mid = Rgb.Hex("#ffa04a"), bottom = Rgb.Hex("#ff2f8e");
        const float pitch = 22;
        float slide = (float)(t / StripePeriod % 1.0) * pitch;
        for (int y = (int)(cy - r); y <= (int)(cy + r); y++) {
            if (y >= Horizon)
                break;
            float v = (y - (cy - r)) / (2 * r);
            Rgb c = v < 0.5f ? Rgb.Lerp(top, mid, v / 0.5f) : Rgb.Lerp(mid, bottom, (v - 0.5f) / 0.5f);
            // Stripes cut the lower 55% of the disc; their gaps grow from 1 to 11 pixels toward the bottom.
            if (v > 0.45f) {
                float k = (v - 0.45f) / 0.55f;
                float gap = 1 + 10 * k;
                float phase = ((y - cy) - slide) % pitch;
                if (phase < 0)
                    phase += pitch;
                if (phase < gap)
                    continue;
            }
            float half = MathF.Sqrt(MathF.Max(0, r * r - (y + 0.5f - cy) * (y + 0.5f - cy)));
            int x0 = (int)MathF.Floor(cx - half), x1 = (int)MathF.Ceiling(cx + half);
            for (int x = x0; x <= x1; x++) {
                // Anti-aliased edge: partial coverage on the first and last pixel of the row.
                float cover = MathF.Min(1, MathF.Min(x + 1 - (cx - half), cx + half - x));
                img.Blend(x, y, c, MathF.Max(0, cover));
            }
        }
    }

    /// <summary>
    ///     The floor: horizontal neon lines scrolling toward the viewer in perspective, and lines converging on the
    ///     vanishing point, both fading out near the horizon where they would crowd into noise.
    /// </summary>
    /// <param name="img">The frame.</param>
    /// <param name="t">Seconds.</param>
    private void DrawGrid(RgbImage img, double t) {
        var line = Rgb.Hex("#ff2fd0");
        float span = _h - Horizon;
        // Perspective: a floor line at depth z sits at Horizon + k / z; depths step by 1 and scroll by the phase.
        const float k = 90;
        float phase = (float)(t / GridPeriod % 1.0);
        for (int i = 0; i < 60; i++) {
            float z = (i + 1 - phase) * 0.55f;
            if (z <= 0)
                continue;
            float y = Horizon + k / z;
            if (y >= _h + 2)
                continue;
            float fade = MathF.Min(1, MathF.Pow((y - Horizon) / (span * 0.55f), 1.3f));
            if (fade < 0.02f)
                break;
            int iy = (int)MathF.Floor(y);
            float f = y - iy;
            for (int x = 0; x < _w; x++) {
                img.Add(x, iy, line, 0.95f * fade * (1 - f));
                img.Add(x, iy + 1, line, 0.95f * fade * f);
                img.Add(x, iy - 1, line, 0.22f * fade);
                img.Add(x, iy + 2, line, 0.22f * fade);
            }
        }
        float vx = _w / 2f;
        for (int i = -24; i <= 24; i++) {
            // Where the line meets the bottom edge: evenly spaced, so the lines fan out from the vanishing point.
            float bottomX = vx + i * 92f;
            float slope = (bottomX - vx) / span;
            for (int y = (int)Horizon + 1; y < _h; y++) {
                float x = vx + slope * (y - Horizon);
                if (x < -2 || x > _w + 2)
                    continue;
                float fade = MathF.Min(1, MathF.Pow((y - Horizon) / (span * 0.55f), 1.3f));
                int ix = (int)MathF.Floor(x);
                float f = x - ix;
                img.Add(ix, y, line, 0.9f * fade * (1 - f));
                img.Add(ix + 1, y, line, 0.9f * fade * f);
                img.Add(ix - 1, y, line, 0.2f * fade);
                img.Add(ix + 2, y, line, 0.2f * fade);
            }
        }
    }

    /// <summary>Samples a piecewise-linear gradient.</summary>
    /// <param name="colors">The colour at each stop.</param>
    /// <param name="stops">The stops' positions, increasing from 0 to 1.</param>
    /// <param name="t">The position, 0-1.</param>
    /// <returns>The colour at <paramref name="t"/>.</returns>
    private static Rgb Gradient(Rgb[] colors, float[] stops, float t) {
        for (int i = 1; i < stops.Length; i++)
            if (t <= stops[i])
                return Rgb.Lerp(colors[i - 1], colors[i], (t - stops[i - 1]) / (stops[i] - stops[i - 1]));
        return colors[^1];
    }
}
