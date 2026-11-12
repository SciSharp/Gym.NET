using System;
using System.Collections.Generic;
using System.Globalization;
using Gym.Rendering;
using Gym.MuJoCo;

namespace MujocoPoc.Play;

/// <summary>What the heads-up display shows besides the simulation itself.</summary>
/// <param name="Driver">Who drives: "AUTOPILOT" or "YOU".</param>
/// <param name="Paused">Whether the simulation is paused.</param>
/// <param name="SpeedFactor">The playback speed relative to real time.</param>
/// <param name="LastScore">The last finished episode's return, or NaN before the first.</param>
/// <param name="Banner">A centered message (an episode's score), or null.</param>
public readonly record struct Hud(string Driver, bool Paused, double SpeedFactor, double LastScore, string? Banner);

/// <summary>
///     Draws HalfCheetah side-on with Gym.Rendering: the camera follows the cheetah's center of mass along the x axis,
///     every capsule is drawn at the pose the core computed (<c>geom_xpos</c>, <c>geom_xmat</c>), each leg segment is
///     tinted by its motor's torque, and a heads-up display shows the score and the keys.
/// </summary>
/// <remarks>
///     This is not MuJoCo's renderer (not ported; DR-31): it is an orthographic view of the x-z plane the model moves in,
///     so it shows the simulated state exactly, though not Gymnasium's pixels. Rendering reuses one <see cref="Surface"/>,
///     so a view is not thread-safe.
/// </remarks>
public sealed class SideView {
    private static readonly Color SkyTop = new(96, 146, 206), SkyBottom = new(196, 224, 244);
    private static readonly Color FarHills = new(150, 184, 176), NearHills = new(122, 164, 134);
    private static readonly Color GroundLight = new(116, 160, 88), GroundDark = new(100, 142, 76), GroundEdge = new(66, 96, 52);
    private static readonly Color Shadow = new(82, 116, 62), Marker = new(236, 242, 226), MarkerShadow = new(52, 74, 42);
    private static readonly Color Body = new(204, 153, 102), Outline = new(104, 68, 38), JointColor = new(72, 46, 26);
    private static readonly Color Push = new(236, 92, 60), Pull = new(66, 124, 232);
    private static readonly Color Panel = new(22, 30, 46), PanelEdge = new(255, 206, 84), Text = new(240, 242, 246);
    private static readonly Color Accent = new(255, 206, 84), Dim = new(150, 162, 186), Good = new(120, 220, 140), Warn = new(255, 150, 90);
    private const int HudHeight = 92;

    private readonly Surface surface;
    private readonly double pixelsPerMeter;
    private readonly int groundY;

    /// <summary>The frame width in pixels.</summary>
    public int Width { get; }

    /// <summary>The frame height in pixels.</summary>
    public int Height { get; }

    /// <summary>A view of a given size.</summary>
    /// <param name="width">Frame width (at least 640 for the display to fit).</param>
    /// <param name="height">Frame height (at least 400).</param>
    /// <param name="pixelsPerMeter">The zoom.</param>
    /// <exception cref="ArgumentOutOfRangeException">The frame is smaller than 640x400 or the zoom is not positive.</exception>
    public SideView(int width = 960, int height = 480, double pixelsPerMeter = 200) {
        if (width < 640 || height < 400)
            throw new ArgumentOutOfRangeException(nameof(width), $"{width}x{height}: the display needs at least 640x400");
        if (pixelsPerMeter <= 0)
            throw new ArgumentOutOfRangeException(nameof(pixelsPerMeter), pixelsPerMeter, "the zoom must be positive");
        Width = width;
        Height = height;
        this.pixelsPerMeter = pixelsPerMeter;
        groundY = height - 110;
        surface = new Surface(width, height);
    }

    /// <summary>Draws the current state of a game into the view's frame.</summary>
    /// <param name="game">The game; its core's frames must be current (any reset or step leaves them so).</param>
    /// <param name="hud">The display's contents.</param>
    /// <returns>The frame, reused by the next call (copy it, e.g. with <see cref="Surface.ToRgbBytes"/>, to keep it).</returns>
    public Surface Render(CheetahGame game, Hud hud) {
        MjData core = game.Core;
        // the camera tracks the center of mass of the whole cheetah (body 1's subtree), as Gymnasium's "trackcom" camera does
        double cameraX = core.subtree_com[3];
        DrawSky(cameraX);
        DrawGround(cameraX);
        DrawShadow(cameraX, core.subtree_com[3]);
        DrawCheetah(game, cameraX);
        DrawHud(game, hud);
        return surface;
    }

    /// <summary>Screen x of a world x.</summary>
    /// <param name="x">World x in meters.</param>
    /// <param name="cameraX">The tracked x.</param>
    /// <returns>Pixels from the left; the cheetah sits a little left of center, leaving room to see ahead.</returns>
    private double ScreenX(double x, double cameraX) => Width * 0.42 + (x - cameraX) * pixelsPerMeter;

    /// <summary>Screen y of a world height.</summary>
    /// <param name="z">World z in meters.</param>
    /// <returns>Pixels from the top.</returns>
    private double ScreenY(double z) => groundY - z * pixelsPerMeter;

    /// <summary>The sky gradient and two layers of hills scrolling slower than the ground (parallax).</summary>
    /// <param name="cameraX">The tracked x.</param>
    private void DrawSky(double cameraX) {
        for (int y = 0; y < groundY; y++)
            GfxDraw.HLine(surface, 0, Width - 1, y, Lerp(SkyTop, SkyBottom, (double)y / groundY));
        Hills(cameraX * 0.25, FarHills, 70, 0.0);
        Hills(cameraX * 0.5, NearHills, 38, 2.0);
    }

    /// <summary>One layer of hills: a sum of sines over a shifted world coordinate.</summary>
    /// <param name="offset">How far the layer has scrolled, in meters.</param>
    /// <param name="color">The layer's color.</param>
    /// <param name="height">Its mean height in pixels.</param>
    /// <param name="phase">A phase that makes layers differ.</param>
    private void Hills(double offset, Color color, double height, double phase) {
        for (int x = 0; x < Width; x++) {
            double u = offset + (x - Width * 0.42) / pixelsPerMeter;
            double h = height + 0.45 * height * Math.Sin(0.8 * u + phase) + 0.25 * height * Math.Sin(2.1 * u + 1.7 * phase)
                       + 0.1 * height * Math.Sin(5.3 * u + phase);
            GfxDraw.VLine(surface, x, (long)Math.Round(groundY - h), groundY - 1, color);
        }
    }

    /// <summary>A checkered ground of half-meter tiles, a tick and label at every meter, and a start line at x = 0.</summary>
    /// <param name="cameraX">The tracked x.</param>
    private void DrawGround(double cameraX) {
        double left = cameraX - Width * 0.42 / pixelsPerMeter, right = left + Width / pixelsPerMeter;
        int rows = 3, rowHeight = (Height - groundY + rows - 1) / rows;
        for (long tile = (long)Math.Floor(left / 0.5); tile * 0.5 < right; tile++) {
            int x0 = Clamp(ScreenX(tile * 0.5, cameraX)), x1 = Clamp(ScreenX((tile + 1) * 0.5, cameraX));
            for (int row = 0; row < rows; row++)
                Draw.Rect(surface, ((tile + row) & 1) == 0 ? GroundLight : GroundDark, x0, groundY + row * rowHeight, x1 - x0, rowHeight);
        }
        Draw.Rect(surface, GroundEdge, 0, groundY, Width, 3);
        for (long meter = (long)Math.Ceiling(left); meter <= right; meter++) {
            int x = (int)Math.Round(ScreenX(meter, cameraX));
            Draw.Rect(surface, meter == 0 ? Warn : Marker, x - 1, groundY, 3, meter == 0 ? 26 : 12);
            string label = meter == 0 ? "START" : meter.ToString(CultureInfo.InvariantCulture) + "M";
            PixelFont.Draw(surface, label, x - PixelFont.Measure(label, 2) / 2, groundY + 16, 2, Marker, MarkerShadow);
        }
    }

    /// <summary>A soft elliptical shadow on the ground under the cheetah's center of mass.</summary>
    /// <param name="cameraX">The tracked x.</param>
    /// <param name="comX">The center of mass x.</param>
    private void DrawShadow(double cameraX, double comX) {
        var points = new List<(double X, double Y)>(32);
        double cx = ScreenX(comX, cameraX), rx = 0.6 * pixelsPerMeter, ry = 7;
        for (int i = 0; i < 32; i++) {
            double a = 2 * Math.PI * i / 32;
            points.Add((cx + rx * Math.Cos(a), groundY + 4 + ry * Math.Sin(a)));
        }
        GfxDraw.FilledPolygon(surface, points, Shadow);
    }

    /// <summary>
    ///     The cheetah: legs first, then torso and head on top, each capsule outlined; leg segments tinted toward red for a
    ///     positive motor torque and blue for a negative one; dots at the hinge joints.
    /// </summary>
    /// <param name="game">The game.</param>
    /// <param name="cameraX">The tracked x.</param>
    private void DrawCheetah(CheetahGame game, double cameraX) {
        MjData core = game.Core;
        MjModel m = game.Model;
        // geoms 3-8 are the leg segments, each moved by actuator g - 3 (back thigh, shin, foot, then the front three)
        foreach (int g in new[] { 3, 4, 5, 6, 7, 8, 1, 2 }) {
            if (g >= m.ngeom)
                continue;
            Color fill = Body;
            if (g >= 3 && g - 3 < CheetahGame.ActionSize) {
                float torque = game.Action[g - 3];
                fill = Lerp(Body, torque >= 0 ? Push : Pull, Math.Min(1.0, Math.Abs(torque)) * 0.75);
            }
            double cx = core.geom_xpos[3 * g], cz = core.geom_xpos[3 * g + 2];
            // a capsule's axis is its local z: the third column of its rotation, of which the view keeps x and z
            double ax = core.geom_xmat[9 * g + 2], az = core.geom_xmat[9 * g + 8];
            double half = m.geom_size[3 * g + 1], radius = m.geom_size[3 * g];
            Capsule(ScreenX(cx + half * ax, cameraX), ScreenY(cz + half * az), ScreenX(cx - half * ax, cameraX),
                    ScreenY(cz - half * az), radius * pixelsPerMeter, fill);
        }
        for (int j = 3; j < m.njnt; j++) {
            long x = (long)Math.Round(ScreenX(core.xanchor[3 * j], cameraX)), y = (long)Math.Round(ScreenY(core.xanchor[3 * j + 2]));
            GfxDraw.AACircle(surface, x, y, 4, JointColor);
            GfxDraw.FilledCircle(surface, x, y, 4, JointColor);
        }
    }

    /// <summary>
    ///     A filled, outlined capsule between two screen points: an anti-aliased outer shape in the outline color, then the
    ///     filled inner shape (a quad plus two end circles) in the fill color.
    /// </summary>
    /// <param name="x1">First end x.</param>
    /// <param name="y1">First end y.</param>
    /// <param name="x2">Second end x.</param>
    /// <param name="y2">Second end y.</param>
    /// <param name="radius">The capsule radius in pixels.</param>
    /// <param name="fill">The fill color.</param>
    private void Capsule(double x1, double y1, double x2, double y2, double radius, Color fill) {
        double dx = x1 - x2, dy = y1 - y2, length = Math.Sqrt(dx * dx + dy * dy);
        // a capsule seen end-on (never for a planar model) degenerates to a circle: any normal will do
        double nx = length > 1e-9 ? -dy / length : 0, ny = length > 1e-9 ? dx / length : 1;
        foreach (bool outer in new[] { true, false }) {
            double r = outer ? radius + 1.6 : radius;
            Color color = outer ? Outline : fill;
            var quad = new List<(double X, double Y)> {
                (x1 + nx * r, y1 + ny * r), (x2 + nx * r, y2 + ny * r), (x2 - nx * r, y2 - ny * r), (x1 - nx * r, y1 - ny * r),
            };
            long rr = (long)Math.Round(r);
            if (outer) {
                GfxDraw.AAPolygon(surface, quad, color);
                GfxDraw.AACircle(surface, (long)Math.Round(x1), (long)Math.Round(y1), rr, color);
                GfxDraw.AACircle(surface, (long)Math.Round(x2), (long)Math.Round(y2), rr, color);
            }
            GfxDraw.FilledPolygon(surface, quad, color);
            GfxDraw.FilledCircle(surface, (long)Math.Round(x1), (long)Math.Round(y1), rr, color);
            GfxDraw.FilledCircle(surface, (long)Math.Round(x2), (long)Math.Round(y2), rr, color);
        }
    }

    /// <summary>
    ///     The display: title, who drives, the episode's numbers, the six motor torques as bars, the keys, and the pause or
    ///     score banner.
    /// </summary>
    /// <param name="game">The game.</param>
    /// <param name="hud">The display's contents.</param>
    private void DrawHud(CheetahGame game, Hud hud) {
        Draw.Rect(surface, Panel, 0, 0, Width, HudHeight);
        Draw.Rect(surface, PanelEdge, 0, HudHeight, Width, 2);
        PixelFont.Draw(surface, "HALFCHEETAH-V5", 14, 10, 2, Accent);
        PixelFont.Draw(surface, "GYM.NET C# CORE, BIT-EXACT WITH MUJOCO 4.0.0", 14 + PixelFont.Measure("HALFCHEETAH-V5 ", 2), 10, 2, Text);
        // row 2: who drives and the last score; row 3: the episode's numbers. Both stay left of the torque bars.
        bool auto = hud.Driver == "AUTOPILOT";
        string driver = auto ? "AUTOPILOT (ARS LINEAR POLICY)" : "YOU ARE DRIVING";
        PixelFont.Draw(surface, driver, 14, 36, 2, auto ? Good : Warn);
        string last = double.IsNaN(hud.LastScore) ? "" : "  LAST " + hud.LastScore.ToString("F0", CultureInfo.InvariantCulture);
        PixelFont.Draw(surface, (auto ? "  TAB: TAKE OVER" : "  TAB: AUTOPILOT") + last, 14 + PixelFont.Measure(driver, 2), 36, 2, Dim);
        MjData core = game.Core;
        string speed = Math.Abs(hud.SpeedFactor - 1) < 1e-9 ? "" : $"   X{hud.SpeedFactor.ToString("0.###", CultureInfo.InvariantCulture)}";
        string stats = string.Create(CultureInfo.InvariantCulture,
            $"STEP {game.StepIndex}/1000   X {core.qpos[0]:F1}M   SPEED {game.LastVelocity:F1}M/S   RETURN {game.EpisodeReturn:F0}{speed}");
        PixelFont.Draw(surface, stats, 14, 62, 2, Text);
        DrawTorqueBars(game);
        const string back = "BACK LEG:  Q/A THIGH  W/S SHIN  E/D FOOT    R NEW EPISODE  TAB AUTOPILOT";
        const string front = "FRONT LEG: U/J THIGH  I/K SHIN  O/L FOOT    SPACE PAUSE  +/- SPEED  ESC QUIT";
        PixelFont.Draw(surface, back, 14, groundY + 48, 2, Marker, MarkerShadow);
        PixelFont.Draw(surface, front, 14, groundY + 72, 2, Marker, MarkerShadow);
        string? banner = hud.Paused ? "PAUSED" : hud.Banner;
        if (banner != null) {
            int scale = 4, width = PixelFont.Measure(banner, scale);
            Draw.Rect(surface, Panel, (Width - width) / 2 - 16, HudHeight + 40, width + 32, PixelFont.Height * scale + 24);
            PixelFont.Draw(surface, banner, (Width - width) / 2, HudHeight + 52, scale, Accent);
        }
    }

    /// <summary>Six bars at the display's right edge, one per motor, up for a positive torque and down for a negative one.</summary>
    /// <param name="game">The game, whose current action the bars show.</param>
    private void DrawTorqueBars(CheetahGame game) {
        string[] names = { "BT", "BS", "BF", "FT", "FS", "FF" };
        int left = Width - 6 * 28 - 6, mid = 38, half = 24;
        for (int i = 0; i < CheetahGame.ActionSize; i++) {
            int x = left + i * 28;
            Draw.Rect(surface, new Color(44, 56, 80), x, mid - half, 16, 2 * half);
            float a = Math.Clamp(game.Action[i], -1f, 1f);
            int h = (int)Math.Round(Math.Abs(a) * half);
            if (h > 0)
                Draw.Rect(surface, a >= 0 ? Push : Pull, x, a >= 0 ? mid - h : mid, 16, h);
            Draw.Rect(surface, Dim, x, mid, 16, 1);
            PixelFont.Draw(surface, names[i], x + 3, mid + half + 6, 1, Dim);
        }
    }

    /// <summary>Clamps a screen x to the frame (Rect wants non-negative sizes within it).</summary>
    /// <param name="x">The coordinate.</param>
    /// <returns>The rounded coordinate within [0, width].</returns>
    private int Clamp(double x) => (int)Math.Round(Math.Clamp(x, 0, Width));

    /// <summary>Linear interpolation between two colors.</summary>
    /// <param name="a">The color at 0.</param>
    /// <param name="b">The color at 1.</param>
    /// <param name="t">The position, clamped to [0, 1].</param>
    /// <returns>The blended color.</returns>
    private static Color Lerp(Color a, Color b, double t) {
        t = Math.Clamp(t, 0, 1);
        return new Color((byte)Math.Round(a.R + (b.R - a.R) * t), (byte)Math.Round(a.G + (b.G - a.G) * t),
                         (byte)Math.Round(a.B + (b.B - a.B) * t));
    }
}
