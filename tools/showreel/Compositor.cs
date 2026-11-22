using System;
using System.Collections.Generic;
using System.Linq;

namespace Gym.Showreel;

/// <summary>
///     Renders a storyboard frame by frame: the backdrop (the dusk beach, or the synthwave grid), the panels with their
///     env frames, labels and stamps, the overlays, then the flashes, bloom, scanlines, vignette and fades.
/// </summary>
/// <remarks>
///     <para>
///     Frames must be requested in increasing time order: panel sources only move forward (an env cannot rewind). A
///     source is created when its content first shows and disposed when its track ends or flips to the next content.
///     </para>
///     <para>
///     <see cref="Render"/> returns the same buffer every call; the caller writes it out before asking for the next
///     frame.
///     </para>
/// </remarks>
internal sealed class Compositor : IDisposable {
    /// <summary>The timeline.</summary>
    private readonly Storyboard _board;

    /// <summary>The backdrop, drawn first in every frame.</summary>
    private readonly IBackdrop _backdrop;

    /// <summary>The output frame, reused.</summary>
    private readonly RgbImage _frame;

    /// <summary>Whether to darken every other row (the MP4 at full size; the scaled-down GIF would moiré).</summary>
    private readonly bool _scanlines;

    /// <summary>The live sources, by track and content index.</summary>
    private readonly Dictionary<(Track, int), IPanelSource> _sources = new();

    /// <summary>The vignette's per-pixel brightness factor, computed once.</summary>
    private readonly float[] _vignette;

    /// <summary>Scratch buffers of the bloom pass, at a quarter of the frame size.</summary>
    private readonly float[] _bloomA, _bloomB;

    /// <summary>Creates a compositor for a storyboard at the slots' 1280x720.</summary>
    /// <param name="board">The timeline.</param>
    /// <param name="scanlines">Whether to draw scanlines.</param>
    /// <param name="backdrop">
    ///     The backdrop's name for <see cref="Backdrops.Create"/>: <see cref="Backdrops.Default"/> (the dusk beach, DR-102)
    ///     unless <c>--backdrop</c> picks another.
    /// </param>
    /// <exception cref="ArgumentException">The backdrop's name is unknown.</exception>
    public Compositor(Storyboard board, bool scanlines, string backdrop = Backdrops.Default) {
        _board = board;
        _scanlines = scanlines;
        _backdrop = Backdrops.Create(backdrop, Slots.Width, Slots.Height);
        _frame = new RgbImage(Slots.Width, Slots.Height);
        _vignette = new float[Slots.Width * Slots.Height];
        for (int y = 0; y < Slots.Height; y++)
            for (int x = 0; x < Slots.Width; x++) {
                float dx = (x - Slots.Width / 2f) / (Slots.Width / 2f), dy = (y - Slots.Height / 2f) / (Slots.Height / 2f);
                _vignette[y * Slots.Width + x] = 1 - 0.32f * MathF.Pow((dx * dx + dy * dy) / 2, 1.4f);
            }
        _bloomA = new float[(Slots.Width / 4) * (Slots.Height / 4) * 3];
        _bloomB = new float[_bloomA.Length];
    }

    /// <summary>Renders the frame at a time.</summary>
    /// <param name="t">Seconds; must not be earlier than the previous call's.</param>
    /// <returns>The frame buffer (reused by the next call).</returns>
    public RgbImage Render(double t) {
        _backdrop.Draw(_frame, t, _board.SunRise(t));
        // Bloom softens the backdrop's light (sun, grid, rims) only: run after the panels, it would bleed the white
        // backgrounds of the classic-control frames over their thin lines and wash out the envs' own pixels.
        Bloom();
        foreach (Track track in _board.Tracks) {
            if (t < track.Start || t >= track.End) {
                Release(track, -1);
                continue;
            }
            DrawPanel(track, t);
        }
        foreach (Overlay o in _board.Overlays)
            if (t >= o.Start && t < o.End)
                o.Draw(_frame, t, t - o.Start);
        Flash(t);
        Finish(_board.Fade(t));
        return _frame;
    }

    /// <summary>Disposes every live source.</summary>
    public void Dispose() {
        foreach (IPanelSource s in _sources.Values)
            s.Dispose();
        _sources.Clear();
    }

    /// <summary>Disposes a track's sources other than the one in use.</summary>
    /// <param name="track">The track.</param>
    /// <param name="keep">The content index still in use, or -1 to release all.</param>
    private void Release(Track track, int keep) {
        foreach (var key in _sources.Keys.Where(k => k.Item1 == track && k.Item2 != keep).ToList()) {
            _sources[key].Dispose();
            _sources.Remove(key);
        }
    }

    /// <summary>
    ///     Draws one panel: a dark plate, the neon halo, the env frame (squeezed during a flip), the border, the id under
    ///     it, the status in its corner and the stamp over it.
    /// </summary>
    /// <param name="track">The panel's track.</param>
    /// <param name="t">Seconds.</param>
    private void DrawPanel(Track track, double t) {
        RectF slot = track.SlotAt(t);
        if (slot.W < 2 || slot.H < 2)
            return;
        int index = track.ContentAt(t);
        Release(track, index);
        if (!_sources.TryGetValue((track, index), out IPanelSource? source)) {
            source = track.Contents[index].Make();
            _sources[(track, index)] = source;
        }
        RgbImage frame = source.FrameAt(t);
        RectF r = slot.Fit(frame.Aspect);
        float squeeze = track.SqueezeAt(t);
        var visible = RectF.Centered(r.Cx, r.Cy, r.W * squeeze, r.H);
        bool big = r.W > 600;
        _frame.FillRect(new RectF(visible.X - 6, visible.Y - 6, visible.W + 12, visible.H + 12), Rgb.Hex("#0a0218"), 0.85f);
        _frame.Glow(new RectF(visible.X - 3, visible.Y - 3, visible.W + 6, visible.H + 6), track.Neon, big ? 26 : 18, 0.85f);
        // Mid-flip the face turns edge-on: brighten it toward the neon so the turn reads as light catching the panel.
        _frame.DrawImage(frame, r, 1, squeeze, track.Neon, 0.6f * (1 - squeeze));
        _frame.StrokeRect(new RectF(visible.X - 3, visible.Y - 3, visible.W + 6, visible.H + 6), track.Neon, 3);
        if (squeeze < 0.6f)
            return;
        float labelScale = big ? 3 : 2;
        string title = source.Title.ToUpperInvariant();
        // A dark plate under the label keeps neon letters readable where they cross the sun or the bright horizon.
        _frame.FillRect(new RectF(r.X - 4, r.Bottom + 6, PixelFont.Measure(title, labelScale) + 8, PixelFont.Height * labelScale + 8), Rgb.Hex("#0a0218"), 0.6f);
        PixelFont.DrawNeon(_frame, title, r.X, r.Bottom + 10, labelScale, track.Neon);
        string status = source.Status;
        float sw = PixelFont.Measure(status, 2);
        _frame.FillRect(new RectF(r.Right - sw - 14, r.Y + 6, sw + 10, 22), Rgb.Hex("#0a0218"), 0.7f);
        PixelFont.Draw(_frame, status, r.Right - sw - 9, r.Y + 10, 2, Rgb.White);
        if (source.Stamp is string stamp)
            DrawStamp(r, stamp, source.StampAge, t);
    }

    /// <summary>
    ///     The ending's stamp: pops in with an overshoot over 0.35 s, then bobs gently; red for failures, yellow for
    ///     successes, outlined in black so it reads over any frame.
    /// </summary>
    /// <param name="r">The panel's frame rectangle.</param>
    /// <param name="stamp">The text.</param>
    /// <param name="age">Seconds since it appeared.</param>
    /// <param name="t">Seconds, for the bob.</param>
    private void DrawStamp(RectF r, string stamp, double age, double t) {
        bool bad = stamp is "GAME OVER" or "CRASH" or "BUST" or "SPLASH" or "FELL";
        Rgb color = bad ? Rgb.Hex("#ff3b4f") : Storyboards.Yellow;
        float full = Math.Clamp(r.W * 0.7f / (stamp.Length * PixelFont.CellWidth), 3, 9);
        float scale = full * Ease.OutBack((float)Math.Clamp(age / 0.35, 0, 1));
        if (scale <= 0.5f)
            return;
        float w = PixelFont.Measure(stamp, scale);
        float bob = (float)Math.Sin(t * 5) * scale * 0.4f;
        PixelFont.DrawOutlined(_frame, stamp, r.Cx - w / 2, r.Cy - PixelFont.Height * scale / 2 + bob, scale, color, Rgb.Black);
    }

    /// <summary>
    ///     Bloom: the frame's bright parts, shrunk to a quarter, blurred and added back, so the neon lines, letters and
    ///     the sun glow softly.
    /// </summary>
    private void Bloom() {
        int w = Slots.Width / 4, h = Slots.Height / 4;
        byte[] p = _frame.Pixels;
        // Bright pass at quarter size: each cell averages its 4x4 pixels' brightness above a threshold.
        for (int cy = 0; cy < h; cy++)
            for (int cx = 0; cx < w; cx++) {
                float r = 0, g = 0, b = 0;
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++) {
                        int o = ((cy * 4 + dy) * Slots.Width + cx * 4 + dx) * 3;
                        float lum = 0.3f * p[o] + 0.59f * p[o + 1] + 0.11f * p[o + 2];
                        float k = MathF.Max(0, lum - 150) / 105;
                        r += p[o] * k;
                        g += p[o + 1] * k;
                        b += p[o + 2] * k;
                    }
                int q = (cy * w + cx) * 3;
                _bloomA[q] = r / 16;
                _bloomA[q + 1] = g / 16;
                _bloomA[q + 2] = b / 16;
            }
        // Two box blurs each way approximate a Gaussian.
        for (int pass = 0; pass < 2; pass++) {
            BoxBlur(_bloomA, _bloomB, w, h, 3, horizontal: true);
            BoxBlur(_bloomB, _bloomA, w, h, 3, horizontal: false);
        }
        const float strength = 0.75f;
        for (int y = 0; y < Slots.Height; y++) {
            float fy = (y + 0.5f) / 4 - 0.5f;
            int y0 = Math.Clamp((int)MathF.Floor(fy), 0, h - 1), y1 = Math.Min(y0 + 1, h - 1);
            float ty = Math.Clamp(fy - y0, 0, 1);
            for (int x = 0; x < Slots.Width; x++) {
                float fx = (x + 0.5f) / 4 - 0.5f;
                int x0 = Math.Clamp((int)MathF.Floor(fx), 0, w - 1), x1 = Math.Min(x0 + 1, w - 1);
                float tx = Math.Clamp(fx - x0, 0, 1);
                int o = (y * Slots.Width + x) * 3;
                for (int c = 0; c < 3; c++) {
                    float a = _bloomA[(y0 * w + x0) * 3 + c] + (_bloomA[(y0 * w + x1) * 3 + c] - _bloomA[(y0 * w + x0) * 3 + c]) * tx;
                    float bb = _bloomA[(y1 * w + x0) * 3 + c] + (_bloomA[(y1 * w + x1) * 3 + c] - _bloomA[(y1 * w + x0) * 3 + c]) * tx;
                    float v = p[o + c] + strength * (a + (bb - a) * ty);
                    p[o + c] = v >= 255 ? (byte)255 : (byte)v;
                }
            }
        }
    }

    /// <summary>A box blur along one axis, clamping at the edges.</summary>
    /// <param name="src">Source, RGB floats.</param>
    /// <param name="dst">Destination, same size.</param>
    /// <param name="w">Width.</param>
    /// <param name="h">Height.</param>
    /// <param name="radius">Half the box width.</param>
    /// <param name="horizontal">Blur along rows when true, columns when false.</param>
    private static void BoxBlur(float[] src, float[] dst, int w, int h, int radius, bool horizontal) {
        int n = 2 * radius + 1;
        int lines = horizontal ? h : w, length = horizontal ? w : h;
        for (int line = 0; line < lines; line++)
            for (int c = 0; c < 3; c++) {
                int Index(int i) => horizontal ? (line * w + Math.Clamp(i, 0, w - 1)) * 3 + c : (Math.Clamp(i, 0, h - 1) * w + line) * 3 + c;
                float sum = 0;
                for (int i = -radius; i <= radius; i++)
                    sum += src[Index(i)];
                for (int i = 0; i < length; i++) {
                    dst[Index(i)] = sum / n;
                    sum += src[Index(i + radius + 1)] - src[Index(i - radius)];
                }
            }
    }

    /// <summary>White flashes: each rises over 0.05 s to its time and decays over 0.35 s.</summary>
    /// <param name="t">Seconds.</param>
    private void Flash(double t) {
        float k = 0;
        foreach (double f in _board.Flashes) {
            double d = t - f;
            if (d > -0.05 && d < 0)
                k = MathF.Max(k, (float)(1 + d / 0.05));
            else if (d >= 0 && d < 0.35)
                k = MathF.Max(k, (float)(1 - d / 0.35));
        }
        if (k <= 0)
            return;
        byte[] p = _frame.Pixels;
        float a = 0.85f * k;
        for (int i = 0; i < p.Length; i++)
            p[i] = (byte)(p[i] + (255 - p[i]) * a);
    }

    /// <summary>The last pass: vignette, scanlines and the fade from or to black.</summary>
    /// <param name="fade">1 for the full picture, 0 for black.</param>
    private void Finish(float fade) {
        byte[] p = _frame.Pixels;
        for (int y = 0; y < Slots.Height; y++) {
            float line = _scanlines && (y & 1) == 1 ? 0.9f : 1;
            for (int x = 0; x < Slots.Width; x++) {
                int i = y * Slots.Width + x;
                float k = _vignette[i] * line * fade;
                int o = i * 3;
                p[o] = (byte)(p[o] * k);
                p[o + 1] = (byte)(p[o + 1] * k);
                p[o + 2] = (byte)(p[o + 2] * k);
            }
        }
    }
}
