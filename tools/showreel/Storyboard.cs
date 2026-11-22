using System;
using System.Collections.Generic;
using System.Linq;

namespace Gym.Showreel;

/// <summary>Easing curves for panel motion: <c>t</c> in 0-1 to an eased 0-1 (the "back" and "bounce" curves overshoot).</summary>
internal static class Ease {
    /// <summary>Constant speed.</summary>
    /// <param name="t">Progress, 0-1.</param>
    /// <returns>The eased progress.</returns>
    public static float Linear(float t) => t;

    /// <summary>Slow-fast-slow, for panels gliding between layout slots.</summary>
    /// <param name="t">Progress, 0-1.</param>
    /// <returns>The eased progress.</returns>
    public static float InOutCubic(float t) => t < 0.5f ? 4 * t * t * t : 1 - MathF.Pow(-2 * t + 2, 3) / 2;

    /// <summary>Fast start, gentle stop.</summary>
    /// <param name="t">Progress, 0-1.</param>
    /// <returns>The eased progress.</returns>
    public static float OutCubic(float t) => 1 - MathF.Pow(1 - t, 3);

    /// <summary>Overshoots by about 10% and settles back: the playful arrival of a panel.</summary>
    /// <param name="t">Progress, 0-1.</param>
    /// <returns>The eased progress (above 1 near the end).</returns>
    public static float OutBack(float t) {
        const float c1 = 1.70158f, c3 = c1 + 1;
        return 1 + c3 * MathF.Pow(t - 1, 3) + c1 * MathF.Pow(t - 1, 2);
    }

    /// <summary>Pulls back slightly before leaving: the wind-up of a panel's exit.</summary>
    /// <param name="t">Progress, 0-1.</param>
    /// <returns>The eased progress (below 0 near the start).</returns>
    public static float InBack(float t) {
        const float c1 = 1.70158f, c3 = c1 + 1;
        return c3 * t * t * t - c1 * t * t;
    }

    /// <summary>A ball dropped on the floor: three bounces of decreasing height.</summary>
    /// <param name="t">Progress, 0-1.</param>
    /// <returns>The eased progress.</returns>
    public static float OutBounce(float t) {
        const float n1 = 7.5625f, d1 = 2.75f;
        if (t < 1 / d1)
            return n1 * t * t;
        if (t < 2 / d1)
            return n1 * (t -= 1.5f / d1) * t + 0.75f;
        if (t < 2.5f / d1)
            return n1 * (t -= 2.25f / d1) * t + 0.9375f;
        return n1 * (t -= 2.625f / d1) * t + 0.984375f;
    }
}

/// <summary>
///     The layout slots of a 1280x720 frame: one big panel, two side by side, or four in a grid, plus the off-screen and
///     zero-size positions panels fly in from and out to. A slot is the box a panel's frame is fitted into, so a square
///     env and a wide one both fill the same slot without distortion.
/// </summary>
internal static class Slots {
    /// <summary>Frame width the slots are laid out for.</summary>
    public const int Width = 1280;

    /// <summary>Frame height the slots are laid out for.</summary>
    public const int Height = 720;

    /// <summary>The single big panel.</summary>
    public static RectF One => RectF.Centered(640, 368, 800, 470);

    /// <summary>The left of two panels.</summary>
    public static RectF TwoLeft => RectF.Centered(340, 365, 560, 400);

    /// <summary>The right of two panels.</summary>
    public static RectF TwoRight => RectF.Centered(940, 365, 560, 400);

    /// <summary>Top-left of four.</summary>
    public static RectF FourTL => RectF.Centered(345, 222, 500, 235);

    /// <summary>Top-right of four.</summary>
    public static RectF FourTR => RectF.Centered(935, 222, 500, 235);

    /// <summary>Bottom-left of four.</summary>
    public static RectF FourBL => RectF.Centered(345, 505, 500, 235);

    /// <summary>Bottom-right of four.</summary>
    public static RectF FourBR => RectF.Centered(935, 505, 500, 235);

    /// <summary>The same slot moved past the left edge.</summary>
    /// <param name="s">The slot.</param>
    /// <returns>The off-screen slot.</returns>
    public static RectF OffLeft(RectF s) => s with { X = -s.W - 80 };

    /// <summary>The same slot moved past the right edge.</summary>
    /// <param name="s">The slot.</param>
    /// <returns>The off-screen slot.</returns>
    public static RectF OffRight(RectF s) => s with { X = Width + 80 };

    /// <summary>The same slot moved below the bottom edge.</summary>
    /// <param name="s">The slot.</param>
    /// <returns>The off-screen slot.</returns>
    public static RectF OffBottom(RectF s) => s with { Y = Height + 80 };

    /// <summary>The same slot moved above the top edge (far enough that its label goes too).</summary>
    /// <param name="s">The slot.</param>
    /// <returns>The off-screen slot.</returns>
    public static RectF OffTop(RectF s) => s with { Y = -s.H - 140 };

    /// <summary>A zero-size slot at the slot's centre, for zooming in and out.</summary>
    /// <param name="s">The slot.</param>
    /// <returns>The collapsed slot.</returns>
    public static RectF Point(RectF s) => RectF.Centered(s.Cx, s.Cy, 0, 0);
}

/// <summary>A keyframe of a panel's slot: the slot it reaches at <paramref name="T"/>, eased from the previous keyframe.</summary>
/// <param name="T">Seconds.</param>
/// <param name="Slot">The slot at that time.</param>
/// <param name="Curve">The easing of the motion that ends here (ignored on the first keyframe).</param>
internal readonly record struct Key(double T, RectF Slot, Func<float, float> Curve);

/// <summary>
///     What a panel shows from a given time: an episode played live, or frames replayed from a file (the Gymnasium side of
///     the parity segment).
/// </summary>
/// <param name="T">Seconds at which this content starts; the previous content flips over to it.</param>
/// <param name="Make">Creates the source when the content is first shown.</param>
internal sealed record Content(double T, Func<IPanelSource> Make);

/// <summary>
///     One panel's life in a storyboard: where it is over time (keyframes), what it shows (contents, flipping between
///     them), and its neon colour.
/// </summary>
/// <param name="Keys">Keyframes in time order; the panel exists from the first to the last.</param>
/// <param name="Contents">Contents in time order; the first starts no later than the first keyframe.</param>
/// <param name="Neon">The border and label colour.</param>
internal sealed record Track(IReadOnlyList<Key> Keys, IReadOnlyList<Content> Contents, Rgb Neon) {
    /// <summary>Seconds a content switch takes: the panel turns edge-on and back, showing the new content.</summary>
    public const double FlipSeconds = 0.36;

    /// <summary>When the panel appears.</summary>
    public double Start => Keys[0].T;

    /// <summary>When the panel is gone.</summary>
    public double End => Keys[^1].T;

    /// <summary>The panel's slot at a time: the keyframes' slots, eased between them.</summary>
    /// <param name="t">Seconds.</param>
    /// <returns>The slot (unfitted: the compositor fits the frame's aspect into it).</returns>
    public RectF SlotAt(double t) {
        if (t <= Keys[0].T)
            return Keys[0].Slot;
        for (int i = 1; i < Keys.Count; i++) {
            if (t > Keys[i].T)
                continue;
            Key a = Keys[i - 1], b = Keys[i];
            float u = b.T == a.T ? 1 : (float)((t - a.T) / (b.T - a.T));
            return RectF.Lerp(a.Slot, b.Slot, b.Curve(u));
        }
        return Keys[^1].Slot;
    }

    /// <summary>The index of the content shown at a time.</summary>
    /// <param name="t">Seconds.</param>
    /// <returns>The index into <see cref="Contents"/>.</returns>
    public int ContentAt(double t) {
        int index = 0;
        for (int i = 1; i < Contents.Count; i++)
            if (t >= Contents[i].T)
                index = i;
        return index;
    }

    /// <summary>
    ///     The horizontal squeeze of a flip at a time: 1 normally, falling to 0 at a content switch and back, over
    ///     <see cref="FlipSeconds"/> centred on the switch.
    /// </summary>
    /// <param name="t">Seconds.</param>
    /// <returns>The squeeze, 0-1.</returns>
    public float SqueezeAt(double t) {
        for (int i = 1; i < Contents.Count; i++) {
            double d = t - Contents[i].T;
            if (Math.Abs(d) < FlipSeconds / 2)
                return (float)Math.Abs(Math.Sin(Math.PI / 2 * (Math.Abs(d) / (FlipSeconds / 2))));
        }
        return 1;
    }
}

/// <summary>A caption or title drawn over the frame for a time window.</summary>
/// <param name="Start">Seconds at which it appears.</param>
/// <param name="End">Seconds at which it is gone.</param>
/// <param name="Draw">Draws it: frame, absolute time, time since <paramref name="Start"/>.</param>
internal sealed record Overlay(double Start, double End, Action<RgbImage, double, double> Draw);

/// <summary>
///     A whole timeline: its length and frame rate, the panels, the overlays, the white flashes, the sun's rise and the
///     fades, from which the compositor renders any frame.
/// </summary>
/// <param name="Seconds">Length.</param>
/// <param name="Fps">Frames per second.</param>
/// <param name="Tracks">The panels.</param>
/// <param name="Overlays">Captions and titles, drawn over the panels in list order.</param>
/// <param name="Flashes">Times of white flashes (each peaks at its time and decays over 0.35 s).</param>
/// <param name="SunRise">The sun's rise, 0-1, over time.</param>
/// <param name="Fade">The fade from or to black, 0 (black) to 1 (full picture), over time.</param>
internal sealed record Storyboard(
    double Seconds,
    int Fps,
    IReadOnlyList<Track> Tracks,
    IReadOnlyList<Overlay> Overlays,
    IReadOnlyList<double> Flashes,
    Func<double, float> SunRise,
    Func<double, float> Fade) {
    /// <summary>Frames in the timeline.</summary>
    public int Frames => (int)Math.Round(Seconds * Fps);
}

/// <summary>
///     One lettered scene of the video (DR-100): a self-contained stretch that starts and ends on an empty stage, so the
///     scenes play in any order a line of letters gives.
/// </summary>
/// <param name="Letter">The letter an order names it by, <c>A</c> to <c>N</c>.</param>
/// <param name="Name">What it shows, for messages and the README's table.</param>
/// <param name="Caption">
///     The caption over it, or <see langword="null"/> for none (the title and the outro). Scenes that play back to back
///     with the same caption share one, typed on once.
/// </param>
/// <param name="CaptionNeon">The caption's colour.</param>
/// <param name="Length">
///     Seconds from its start to the next scene's start. A little shorter than <paramref name="Tail"/> where the next
///     scene's panels fly in while this one's leave.
/// </param>
/// <param name="Tail">Seconds from its start until its last panel or overlay is gone; the video ends there after its last scene.</param>
/// <param name="Build">
///     Adds its panels, overlays and flashes to a stage, at an origin in seconds. It must add nothing that starts before
///     the origin or lasts past the origin plus <paramref name="Tail"/>: that is what lets the composer put it anywhere.
/// </param>
internal sealed record Scene(char Letter, string Name, string? Caption, Rgb CaptionNeon, double Length, double Tail, Action<SceneStage, double> Build);

/// <summary>What the scenes build the video on: the lists a <see cref="Storyboard"/> is made of, and the parity replays.</summary>
/// <param name="Tracks">The panels, in drawing order.</param>
/// <param name="Overlays">The overlays, in drawing order.</param>
/// <param name="Flashes">The white flashes' times.</param>
/// <param name="Parity">
///     The verified Gymnasium replays, or <see langword="null"/> with <c>--no-parity</c>; the composer never builds the
///     parity scene without them.
/// </param>
internal sealed record SceneStage(List<Track> Tracks, List<Overlay> Overlays, List<double> Flashes, ParityReplays? Parity);

/// <summary>
///     The showreel's timelines: the video, composed from 14 lettered scenes in the order the maintainer picks (DR-100),
///     and the README's 51-second cut of the video as it was before (DR-73).
/// </summary>
internal static class Storyboards {
    /// <summary>Neon cyan, for borders and captions.</summary>
    public static readonly Rgb Cyan = Rgb.Hex("#39f3ff");

    /// <summary>Neon pink.</summary>
    public static readonly Rgb Pink = Rgb.Hex("#ff4fd8");

    /// <summary>Neon yellow, for stamps and highlights.</summary>
    public static readonly Rgb Yellow = Rgb.Hex("#ffe14d");

    /// <summary>Neon green, for the parity segment's zero counters.</summary>
    public static readonly Rgb Green = Rgb.Hex("#5dff6a");

    /// <summary>Neon orange.</summary>
    public static readonly Rgb Orange = Rgb.Hex("#ff9e3d");

    /// <summary>The scene order the video plays when neither <c>--order</c> nor <c>video-order.txt</c> says otherwise.</summary>
    public const string DefaultOrder = "A B C D E F G H I J K L M N";

    /// <summary>
    ///     The video's 14 scenes, lettered A to N in the order they were made to play (DR-100): the title; classic control;
    ///     MuJoCo's control tasks; LunarLander; the walkers; CarRacing; HalfCheetah; Hopper and Walker2d; MuJoCo's four;
    ///     toy text's four; FrozenLake; the parity segment; the finale; the outro.
    /// </summary>
    /// <returns>The scenes, in letter order.</returns>
    /// <remarks>
    ///     <para>
    ///     Each scene starts and ends on an empty stage, so any order plays: its panels fly in after its start and are gone
    ///     by its <see cref="Scene.Tail"/>. The next scene starts at <see cref="Scene.Length"/>, a little before that, so
    ///     the incoming panels meet the outgoing ones as the sections of the 85 s video did. The timings inside each scene
    ///     are that video's, moved to start at 0.
    ///     </para>
    ///     <para>
    ///     Two hand-overs of the 85 s video crossed a section boundary and cannot survive reordering. CartPole now enters
    ///     scene C afresh, beside InvertedPendulum (the two cart-poles stay a pair, inside one scene), 0.3 s after it so
    ///     that it never meets scene B's CartPole on its way out. CarRacing no longer drives into HalfCheetah's scene: F
    ///     ends with it leaving, and G is HalfCheetah alone, joined at step 40 so it is already running.
    ///     </para>
    /// </remarks>
    public static IReadOnlyList<Scene> Scenes() => new[] {
        new Scene('A', "Title", null, Cyan, 4.0, 4.25, (stage, o) => {
            Title(stage.Overlays, o + 0.55, o + 4.25);
            // The flash where the bouncing logo first lands.
            stage.Flashes.Add(o + 1.62);
        }),
        new Scene('B', "Classic control", "CLASSIC CONTROL", Cyan, 11.3, 11.7, (stage, o) => {
            // One panel, then two, then four: CartPole grows the layout around itself.
            stage.Tracks.Add(Panel(Cyan, Play("CartPole-v1", o),
                K(o, Slots.OffBottom(Slots.One)), K(o + 0.6, Slots.One, Ease.OutBack), K(o + 4.0, Slots.One), K(o + 4.6, Slots.TwoLeft, Ease.InOutCubic),
                K(o + 7.4, Slots.TwoLeft), K(o + 8.0, Slots.FourTL, Ease.InOutCubic), K(o + 11.0, Slots.FourTL), K(o + 11.6, Slots.OffLeft(Slots.FourTL), Ease.InBack)));
            stage.Tracks.Add(Panel(Pink, Play("MountainCar-v0", o + 4.0),
                K(o + 4.0, Slots.OffRight(Slots.TwoRight)), K(o + 4.6, Slots.TwoRight, Ease.OutBack), K(o + 7.4, Slots.TwoRight), K(o + 8.0, Slots.FourTR, Ease.InOutCubic),
                K(o + 11.0, Slots.FourTR), K(o + 11.6, Slots.OffRight(Slots.FourTR), Ease.InBack)));
            stage.Tracks.Add(Panel(Yellow, Play("Pendulum-v1", o + 7.5),
                K(o + 7.5, Slots.OffBottom(Slots.FourBL)), K(o + 8.1, Slots.FourBL, Ease.OutBack), K(o + 11.05, Slots.FourBL), K(o + 11.65, Slots.OffBottom(Slots.FourBL), Ease.InBack)));
            stage.Tracks.Add(Panel(Green, Play("Acrobot-v1", o + 7.65, speed: 20.0 / 15.0),
                K(o + 7.65, Slots.OffBottom(Slots.FourBR)), K(o + 8.25, Slots.FourBR, Ease.OutBack), K(o + 11.1, Slots.FourBR), K(o + 11.7, Slots.OffBottom(Slots.FourBR), Ease.InBack)));
        }),
        new Scene('C', "MuJoCo control", "MUJOCO PHYSICS · BIT FOR BIT, IN C#", Orange, 8.7, 8.65, (stage, o) => {
            // The two cart-poles side by side (CartPole joined mid-move, at step 60), then MuJoCo's four control tasks.
            // CartPole comes 0.3 s after InvertedPendulum: after scene B, whose CartPole leaves the same side until then,
            // an earlier entry showed two CartPoles at once.
            stage.Tracks.Add(Panel(Cyan, new[] { new Content(o + 0.3, Mid("CartPole-v1", o + 0.3, 60)) },
                K(o + 0.3, Slots.OffLeft(Slots.TwoLeft)), K(o + 0.9, Slots.TwoLeft, Ease.OutBack), K(o + 3.3, Slots.TwoLeft), K(o + 3.9, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
            stage.Tracks.Add(Panel(Orange, Play("InvertedPendulum-v5", o),
                K(o, Slots.OffRight(Slots.TwoRight)), K(o + 0.6, Slots.TwoRight, Ease.OutBack), K(o + 3.3, Slots.TwoRight), K(o + 3.9, Slots.FourTL, Ease.InOutCubic),
                K(o + 7.9, Slots.FourTL), K(o + 8.5, Slots.OffLeft(Slots.FourTL), Ease.InBack)));
            var control = new[] { ("InvertedDoublePendulum-v5", Slots.FourTR, Pink), ("Reacher-v5", Slots.FourBL, Yellow), ("Pusher-v5", Slots.FourBR, Green) };
            for (int i = 0; i < control.Length; i++) {
                var (id, slot, neon) = control[i];
                double t0 = o + 3.4 + 0.1 * i;
                RectF entry = slot.Cy < 360 ? Slots.OffTop(slot) : Slots.OffBottom(slot);
                RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
                stage.Tracks.Add(Panel(neon, Play(id, t0), K(t0, entry), K(t0 + 0.7, slot, Ease.OutBack), K(o + 7.95 + 0.05 * i, slot),
                    K(o + 8.55 + 0.05 * i, exit, Ease.InBack)));
            }
        }),
        new Scene('D', "LunarLander", "BOX2D", Pink, 7.0, 7.3, (stage, o) =>
            stage.Tracks.Add(Panel(Cyan, Play("LunarLander-v3", o),
                K(o, Slots.Point(Slots.One)), K(o + 0.6, Slots.One, Ease.OutBack), K(o + 6.8, Slots.One), K(o + 7.3, Slots.OffTop(Slots.One), Ease.InBack)))),
        new Scene('E', "The walkers", "BOX2D", Pink, 5.5, 5.7, (stage, o) => {
            stage.Tracks.Add(Panel(Yellow, Play("BipedalWalker-v3", o),
                K(o, Slots.OffLeft(Slots.TwoLeft)), K(o + 0.6, Slots.TwoLeft, Ease.OutBack), K(o + 5.2, Slots.TwoLeft), K(o + 5.7, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
            // The hardcore walker trips over its first stump at step 157, 3.1 s in.
            stage.Tracks.Add(Panel(Orange, Play("BipedalWalkerHardcore-v3", o + 0.1),
                K(o + 0.1, Slots.OffRight(Slots.TwoRight)), K(o + 0.7, Slots.TwoRight, Ease.OutBack), K(o + 5.2, Slots.TwoRight), K(o + 5.7, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        }),
        new Scene('F', "CarRacing", "BOX2D", Pink, 5.6, 5.8, (stage, o) =>
            stage.Tracks.Add(Panel(Pink, Play("CarRacing-v3", o),
                K(o, Slots.OffBottom(Slots.One)), K(o + 0.6, Slots.One, Ease.OutBack), K(o + 5.3, Slots.One), K(o + 5.8, Slots.OffTop(Slots.One), Ease.InBack)))),
        new Scene('G', "HalfCheetah", "MUJOCO · LOCOMOTION", Orange, 4.2, 4.5, (stage, o) =>
            stage.Tracks.Add(Panel(Orange, new[] { new Content(o, Mid("HalfCheetah-v5", o, 40)) },
                K(o, Slots.Point(Slots.One)), K(o + 0.6, Slots.One, Ease.OutBack), K(o + 4.0, Slots.One), K(o + 4.5, Slots.OffTop(Slots.One), Ease.InBack)))),
        new Scene('H', "Hopper and Walker2d", "MUJOCO · LOCOMOTION", Orange, 4.3, 4.5, (stage, o) => {
            stage.Tracks.Add(Panel(Cyan, Play("Hopper-v5", o),
                K(o, Slots.OffLeft(Slots.TwoLeft)), K(o + 0.6, Slots.TwoLeft, Ease.OutBack), K(o + 4.0, Slots.TwoLeft), K(o + 4.5, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
            stage.Tracks.Add(Panel(Pink, Play("Walker2d-v5", o + 0.1),
                K(o + 0.1, Slots.OffRight(Slots.TwoRight)), K(o + 0.7, Slots.TwoRight, Ease.OutBack), K(o + 4.0, Slots.TwoRight), K(o + 4.5, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        }),
        new Scene('I', "MuJoCo's four", "MUJOCO · LOCOMOTION", Orange, 5.1, 5.2, (stage, o) => {
            var four = new[] { ("Ant-v5", Slots.FourTL, Yellow), ("Humanoid-v5", Slots.FourTR, Green), ("Swimmer-v5", Slots.FourBL, Cyan), ("HumanoidStandup-v5", Slots.FourBR, Pink) };
            for (int i = 0; i < four.Length; i++) {
                var (id, slot, neon) = four[i];
                double t0 = o + 0.1 * i;
                RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
                stage.Tracks.Add(Panel(neon, Play(id, t0 + 0.7), K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce), K(o + 4.7, slot), K(o + 5.2, exit, Ease.InBack)));
            }
        }),
        new Scene('J', "Toy text's four", "TOY TEXT", Yellow, 6.3, 6.5, (stage, o) => {
            var four = new[] { ("FrozenLake8x8-v1", Slots.FourTL, Cyan), ("CliffWalking-v1", Slots.FourTR, Pink), ("Taxi-v4", Slots.FourBL, Yellow), ("Blackjack-v1", Slots.FourBR, Green) };
            for (int i = 0; i < four.Length; i++) {
                var (id, slot, neon) = four[i];
                double t0 = o + 0.1 * i;
                RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
                stage.Tracks.Add(Panel(neon, Play(id, t0 + 0.7), K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce), K(o + 6.0, slot), K(o + 6.5, exit, Ease.InBack)));
            }
        }),
        new Scene('K', "FrozenLake", "TOY TEXT", Yellow, 5.4, 5.3, (stage, o) =>
            stage.Tracks.Add(Panel(Cyan, Play("FrozenLake-v1", o, speed: 10.0 / 8.0),
                K(o, Slots.Point(Slots.One)), K(o + 0.6, Slots.One, Ease.OutBack), K(o + 4.8, Slots.One), K(o + 5.3, Slots.OffBottom(Slots.One), Ease.InBack)))),
        new Scene('L', "Parity", "SAME SEED · SAME ACTIONS · SAME PIXELS", Green, 6.8, 6.9, (stage, o) => {
            // Gymnasium on the left, Gym.NET on the right, the same seeds and actions, a pixel count between them. The
            // composer leaves this scene out when there are no replays (--no-parity).
            ParityReplays parity = stage.Parity!;
            Columns(stage.Overlays, o + 0.3, o + 6.6);
            var rows = new[] { (parity.Lander, Slots.FourTL, Slots.FourTR), (parity.Racer, Slots.FourBL, Slots.FourBR) };
            for (int i = 0; i < rows.Length; i++) {
                var (replay, left, right) = rows[i];
                double t0 = o + 0.15 * i;
                stage.Tracks.Add(Panel(Cyan, new[] { new Content(t0, () => new ReplaySource(replay, t0, gymNet: false)) },
                    K(t0, Slots.Point(left)), K(t0 + 0.6, left, Ease.OutBack), K(o + 6.4, left), K(o + 6.9, Slots.OffLeft(left), Ease.InBack)));
                stage.Tracks.Add(Panel(Pink, new[] { new Content(t0, () => new ReplaySource(replay, t0, gymNet: true)) },
                    K(t0, Slots.Point(right)), K(t0 + 0.6, right, Ease.OutBack), K(o + 6.4, right), K(o + 6.9, Slots.OffRight(right), Ease.InBack)));
                DiffBadge(stage.Overlays, replay, t0, o + 6.6, (left.Cy + right.Cy) / 2);
            }
        }),
        new Scene('M', "Finale", "22 ENVS · 39 IDS · ONE API", Cyan, 4.2, 4.2, (stage, o) => {
            // Four panels flipping through all 24 ids, mid-episode, every 0.6 s. Each row of the table is one moment of the
            // four panels, so every moment mixes families; each column is one panel's run of six.
            string[,] order = {
                { "LunarLander-v3", "HalfCheetah-v5", "Taxi-v4", "Ant-v5" },
                { "Hopper-v5", "CartPole-v1", "Humanoid-v5", "CarRacing-v3" },
                { "FrozenLake8x8-v1", "Walker2d-v5", "Acrobot-v1", "Reacher-v5" },
                { "InvertedDoublePendulum-v5", "BipedalWalker-v3", "Swimmer-v5", "Blackjack-v1" },
                { "MountainCar-v0", "Pusher-v5", "CliffWalking-v1", "HumanoidStandup-v5" },
                { "InvertedPendulum-v5", "Pendulum-v1", "BipedalWalkerHardcore-v3", "FrozenLake-v1" },
            };
            RectF[] quad = { Slots.FourTL, Slots.FourTR, Slots.FourBL, Slots.FourBR };
            Rgb[] neons = { Cyan, Pink, Yellow, Green };
            for (int q = 0; q < 4; q++) {
                var contents = new List<Content>();
                for (int k = 0; k < order.GetLength(0); k++) {
                    double at = o + 0.05 * q + 0.6 * k;
                    contents.Add(new Content(at, Mid(order[k, q], at)));
                }
                RectF slot = quad[q];
                double t0 = o + 0.05 * q;
                stage.Tracks.Add(Panel(neons[q], contents, K(t0, Slots.Point(slot)), K(t0 + 0.5, slot, Ease.OutBack), K(o + 3.7, slot),
                    K(o + 4.2, Slots.Point(slot), Ease.InBack)));
            }
        }),
        new Scene('N', "Outro", null, Cyan, 4.4, 4.4, (stage, o) => {
            stage.Flashes.Add(o);
            Outro(stage.Overlays, o, o + 4.4);
        }),
    };

    /// <summary>
    ///     The video: the scenes an order names, played back to back (DR-100). <see cref="DefaultOrder"/> plays all 14 in
    ///     letter order, about 83 s.
    /// </summary>
    /// <param name="parity">The Gymnasium replays for the parity scene, or <see langword="null"/> to leave scene L out.</param>
    /// <param name="order">
    ///     Scene letters in the order they play, separated by spaces or commas (<c>"A D B I L M N"</c>); a letter may come
    ///     back. See <see cref="ParseOrder"/>.
    /// </param>
    /// <param name="letters">
    ///     Whether to print each scene's letter in the bottom-right corner while it plays: a preview for choosing an order,
    ///     never the published video.
    /// </param>
    /// <returns>The storyboard.</returns>
    /// <exception cref="ArgumentException">The order is empty or names a letter that is not a scene.</exception>
    /// <remarks>
    ///     <para>
    ///     The composer starts each scene at the previous one's <see cref="Scene.Length"/> and ends the video when the last
    ///     one's panels have left. A caption that consecutive scenes share (BOX2D over D, E and F) types on once and stays
    ///     until the next caption is due. The small header logo shows over every scene but the title and the outro: it
    ///     takes over from the big logo where the title hands it on, and leaves as the outro's logo zooms in.
    ///     </para>
    ///     <para>
    ///     The 85 s video before DR-100 ran these scenes in this order, with the two hand-overs that <see cref="Scenes"/>
    ///     describes; the README's 51 s cut (<see cref="Hero"/>) is still that video's own timeline and follows no order.
    ///     </para>
    /// </remarks>
    public static Storyboard Video(ParityReplays? parity, string order, bool letters = false) {
        var (placed, seconds) = Layout(order, withParity: parity is not null);
        List<Scene> chosen = placed.Select(p => p.Scene).ToList();
        double[] starts = placed.Select(p => p.Start).ToArray();
        var stage = new SceneStage(new List<Track>(), new List<Overlay>(), new List<double>(), parity);
        foreach (var (scene, start) in placed)
            scene.Build(stage, start);

        // Captions: one per run of scenes that share it, typed on at the run's start, gone just before the next scene.
        for (int i = 0; i < chosen.Count;) {
            int j = i;
            while (j + 1 < chosen.Count && chosen[j + 1].Caption == chosen[i].Caption)
                j++;
            if (chosen[i].Caption is { } text) {
                double end = j + 1 < chosen.Count ? starts[j + 1] - 0.1 : starts[j] + chosen[j].Tail;
                Caption(stage.Overlays, starts[i], end, text, chosen[i].CaptionNeon);
            }
            i = j + 1;
        }

        // The header logo, over each run of scenes that are neither the title nor the outro.
        static bool Framed(Scene s) => s.Letter is not ('A' or 'N');
        for (int i = 0; i < chosen.Count;) {
            if (!Framed(chosen[i])) {
                i++;
                continue;
            }
            int j = i;
            while (j + 1 < chosen.Count && Framed(chosen[j + 1]))
                j++;
            // After the title, the small logo fades in as the big one shrinks into its place (0.55 s before the title ends).
            double from = i > 0 && chosen[i - 1].Letter == 'A' ? starts[i - 1] + 3.7 : starts[i];
            // Before the outro, it stays until the outro's logo has begun to zoom in.
            double to = j + 1 < chosen.Count && chosen[j + 1].Letter == 'N' ? starts[j + 1] + 0.2 : starts[j] + chosen[j].Tail;
            HeaderLogo(stage.Overlays, from, to);
            i = j + 1;
        }

        if (letters)
            for (int i = 0; i < chosen.Count; i++)
                SceneLetter(stage.Overlays, chosen[i].Letter, starts[i], starts[i] + chosen[i].Length);

        return new Storyboard(seconds, 30, stage.Tracks, stage.Overlays, stage.Flashes,
            time => Ease.OutCubic((float)Math.Clamp(time / 2.2, 0, 1)),
            time => (float)Math.Min(Math.Clamp(time / 0.8, 0, 1), Math.Clamp((seconds - time) / 0.6, 0, 1)));
    }

    /// <summary>
    ///     Where each scene of an order starts and how long the video runs, without building a frame: what <c>scenes</c>
    ///     prints, so an order's length and <c>stills --at</c> times are known before minutes of rendering.
    /// </summary>
    /// <param name="order">The order's text (see <see cref="ParseOrder"/>).</param>
    /// <param name="withParity">
    ///     Whether the parity replays will be there; without them scene L is left out wherever the order names it, as
    ///     <see cref="Video"/> leaves it out.
    /// </param>
    /// <returns>The scenes with their start times, in playing order, and the video's length in seconds.</returns>
    /// <exception cref="ArgumentException">The order names a letter that is not a scene, or leaves no scene to play.</exception>
    public static (List<(Scene Scene, double Start)> Placed, double Seconds) Layout(string order, bool withParity) {
        List<Scene> chosen = ParseOrder(order, Scenes());
        // Without replays there is no parity scene to play (--no-parity), whatever the order says.
        if (!withParity)
            chosen.RemoveAll(s => s.Letter == 'L');
        if (chosen.Count == 0)
            throw new ArgumentException($"the order '{order.Trim()}' leaves no scene to play");
        var placed = new List<(Scene Scene, double Start)>(chosen.Count);
        double t = 0;
        foreach (Scene scene in chosen) {
            placed.Add((scene, t));
            t += scene.Length;
        }
        // The last scene has no successor to overlap: the video runs until its panels are gone (or its Length, if longer).
        var (last, lastStart) = placed[^1];
        return (placed, lastStart + Math.Max(last.Length, last.Tail));
    }

    /// <summary>
    ///     Reads an order: scene letters separated by spaces or commas, upper case; anything after a <c>#</c> on a line is
    ///     a comment, so <c>video-order.txt</c> can say what it is.
    /// </summary>
    /// <param name="order">The order's text.</param>
    /// <param name="scenes">The scenes it may name.</param>
    /// <returns>The scenes in that order (a scene named twice plays twice).</returns>
    /// <exception cref="ArgumentException">An entry is not exactly one of the scenes' letters.</exception>
    /// <remarks>
    ///     Lower case is refused rather than folded: it is kept free for a short form of a scene, should one be asked for.
    /// </remarks>
    public static List<Scene> ParseOrder(string order, IReadOnlyList<Scene> scenes) {
        var chosen = new List<Scene>();
        foreach (string line in order.Split('\n')) {
            string text = line.Split('#')[0];
            foreach (string entry in text.Split(new[] { ' ', ',', '\t', '\r' }, StringSplitOptions.RemoveEmptyEntries)) {
                Scene? scene = entry.Length == 1 ? scenes.FirstOrDefault(s => s.Letter == entry[0]) : null;
                chosen.Add(scene ?? throw new ArgumentException(
                    $"'{entry}' is not a scene: the scenes are {string.Join(" ", scenes.Select(s => s.Letter))} ({string.Join(", ", scenes.Select(s => $"{s.Letter} {s.Name}"))})"));
            }
        }
        return chosen;
    }

    /// <summary>A scene's letter in the bottom-right corner, on a dark plate, for the lettered preview.</summary>
    /// <param name="overlays">Where it is added.</param>
    /// <param name="letter">The scene's letter.</param>
    /// <param name="start">When the scene starts.</param>
    /// <param name="end">When the next scene starts.</param>
    private static void SceneLetter(List<Overlay> overlays, char letter, double start, double end) =>
        overlays.Add(new Overlay(start, end, (img, _, _) => {
            img.FillRect(new RectF(1280 - 96, 720 - 104, 80, 88), Rgb.Hex("#0a0218"), 0.8f);
            Centered(img, letter.ToString(), 1280 - 56, 720 - 92, 9, Yellow, 1);
        }));

    /// <summary>
    ///     The README's animation: the whole showreel cut to 51 seconds, the video's 85 minus two fifths. The sections are
    ///     those of the 85 s video as it was before DR-100 split it into scenes, in its order and with its hand-overs (the
    ///     maintainer kept this cut as it is: "The MP4 only"): intro; classic control (1, 2, then 4 panels); MuJoCo's
    ///     control tasks (CartPole beside InvertedPendulum, then 4); Box2D (1, 2, 1); MuJoCo's locomotion (CarRacing beside
    ///     HalfCheetah, then 1, 2 and 4); toy text (4, then 1); the parity segment; the finale through all 24 ids; the
    ///     outro, which fades into the intro where the loop starts over.
    /// </summary>
    /// <param name="parity">The Gymnasium replays for the parity segment, or <see langword="null"/> to leave it out (the cut is then 3.6 s shorter).</param>
    /// <returns>The storyboard (rendered at 1280x720 and 15 fps, scaled down by the encoder).</returns>
    /// <remarks>
    ///     <para>
    ///     The maintainer, 2026-11-12: "the gif is not great, redo the showreel but -2/5 of the time of the mp4" (DR-73).
    ///     The cut keeps every env and shortens how long each stays: a solo of 6 s in the video holds about 3.5 s here, and
    ///     the finale flips every 0.45 s instead of 0.6 s. The episodes play at their own speed, but for LunarLander.
    ///     </para>
    ///     <para>
    ///     Where a panel is now too short to reach the moment its episode is known for, it joins mid-episode
    ///     (<see cref="Mid"/>), timed so the moment lands on screen: MountainCar's summit, Acrobot crossing the line, the
    ///     toy-text goals. Each such tick is the episode's length minus what the panel can show, from the recipe's steps
    ///     per second (the toy texts' lengths are the README table's: goal in 63 and 41 steps, the edge path in 13, the
    ///     delivery in 16). A join only works where the story is in the episode's end: LunarLander's is its descent, which
    ///     ends at step 117 of 299, so it plays from its start at twice its speed instead.
    ///     </para>
    ///     <para>
    ///     The title runs 0.3–3.75 s, as long as its two lines need to type on at 32 letters a second and be read before it
    ///     leaves; the outro gets the last 3 s for the same reason. What remains is shared in the video's proportions.
    ///     </para>
    /// </remarks>
    public static Storyboard Hero(ParityReplays? parity) {
        var tracks = new List<Track>();
        var overlays = new List<Overlay>();
        var flashes = new List<double> { 1.37 };

        Title(overlays, 0.3, 3.75);

        // Classic control, 3.5-9.9 s: one panel, then two, then four; CartPole stays for the hand-over to MuJoCo.
        Caption(overlays, 3.5, 9.8, "CLASSIC CONTROL", Cyan);
        tracks.Add(Panel(Cyan, Play("CartPole-v1", 3.5),
            K(3.5, Slots.OffBottom(Slots.One)), K(4.0, Slots.One, Ease.OutBack), K(5.5, Slots.One), K(6.0, Slots.TwoLeft, Ease.InOutCubic),
            K(7.1, Slots.TwoLeft), K(7.6, Slots.FourTL, Ease.InOutCubic), K(9.5, Slots.FourTL), K(10.0, Slots.TwoLeft, Ease.InOutCubic),
            K(11.9, Slots.TwoLeft), K(12.4, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        // The summit comes at step 122 (30 steps a second): joined at step 30, the car tops the hill at 8.6 s.
        tracks.Add(Panel(Pink, new[] { new Content(5.5, Mid("MountainCar-v0", 5.5, 30)) },
            K(5.5, Slots.OffRight(Slots.TwoRight)), K(6.0, Slots.TwoRight, Ease.OutBack), K(7.1, Slots.TwoRight), K(7.6, Slots.FourTR, Ease.InOutCubic),
            K(9.5, Slots.FourTR), K(10.0, Slots.OffRight(Slots.FourTR), Ease.InBack)));
        tracks.Add(Panel(Yellow, Play("Pendulum-v1", 7.2),
            K(7.2, Slots.OffBottom(Slots.FourBL)), K(7.8, Slots.FourBL, Ease.OutBack), K(9.55, Slots.FourBL), K(10.05, Slots.OffBottom(Slots.FourBL), Ease.InBack)));
        // Over the line at step 65, at the video's 20 steps a second: joined at step 30, it crosses at 9.1 s.
        tracks.Add(Panel(Green, new[] { new Content(7.35, Mid("Acrobot-v1", 7.35, 30, speed: 20.0 / 15.0)) },
            K(7.35, Slots.OffBottom(Slots.FourBR)), K(7.95, Slots.FourBR, Ease.OutBack), K(9.6, Slots.FourBR), K(10.1, Slots.OffBottom(Slots.FourBR), Ease.InBack)));

        // MuJoCo's control tasks, 9.9-14.8 s: InvertedPendulum beside CartPole, then four panels.
        Caption(overlays, 9.9, 14.7, "MUJOCO PHYSICS · BIT FOR BIT, IN C#", Orange);
        tracks.Add(Panel(Orange, Play("InvertedPendulum-v5", 9.9),
            K(9.9, Slots.OffRight(Slots.TwoRight)), K(10.4, Slots.TwoRight, Ease.OutBack), K(11.9, Slots.TwoRight), K(12.4, Slots.FourTL, Ease.InOutCubic),
            K(14.3, Slots.FourTL), K(14.8, Slots.OffLeft(Slots.FourTL), Ease.InBack)));
        // Pusher joins at step 40 of 100, so its 2.4 s on screen show the push rather than the reach.
        var control = new[] { ("InvertedDoublePendulum-v5", Slots.FourTR, Pink, 0), ("Reacher-v5", Slots.FourBL, Yellow, 0), ("Pusher-v5", Slots.FourBR, Green, 40) };
        for (int i = 0; i < control.Length; i++) {
            var (id, slot, neon, tick) = control[i];
            double t0 = 11.9 + 0.1 * i;
            RectF entry = slot.Cy < 360 ? Slots.OffTop(slot) : Slots.OffBottom(slot);
            RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
            tracks.Add(Panel(neon, new[] { new Content(t0, Mid(id, t0, tick)) }, K(t0, entry), K(t0 + 0.6, slot, Ease.OutBack),
                K(14.35 + 0.05 * i, slot), K(14.85 + 0.05 * i, exit, Ease.InBack)));
        }

        // Box2D, 14.8-25.0 s: the lander alone, the two walkers side by side, then the race car, which stays for the hand-over.
        Caption(overlays, 14.8, 24.9, "BOX2D", Pink);
        // The lander touches down at step 117 and comes to rest, LANDED!, at step 299: from its start at twice its speed,
        // the descent plays out, it touches down at 16.0 s and the stamp holds the last second before the panel leaves.
        // Joining mid-episode instead (step 125, as the first cut did) skipped the whole descent.
        tracks.Add(Panel(Cyan, Play("LunarLander-v3", 14.8, speed: 2.0),
            K(14.8, Slots.Point(Slots.One)), K(15.3, Slots.One, Ease.OutBack), K(18.8, Slots.One), K(19.3, Slots.OffTop(Slots.One), Ease.InBack)));
        tracks.Add(Panel(Yellow, Play("BipedalWalker-v3", 19.0),
            K(19.0, Slots.OffLeft(Slots.TwoLeft)), K(19.6, Slots.TwoLeft, Ease.OutBack), K(22.3, Slots.TwoLeft), K(22.8, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        // The hardcore walker trips at step 157, 3.1 s after it starts: on screen, as in the video.
        tracks.Add(Panel(Orange, Play("BipedalWalkerHardcore-v3", 19.1),
            K(19.1, Slots.OffRight(Slots.TwoRight)), K(19.7, Slots.TwoRight, Ease.OutBack), K(22.3, Slots.TwoRight), K(22.8, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        tracks.Add(Panel(Pink, Play("CarRacing-v3", 22.6),
            K(22.6, Slots.OffBottom(Slots.One)), K(23.2, Slots.One, Ease.OutBack), K(24.9, Slots.One), K(25.4, Slots.TwoLeft, Ease.InOutCubic),
            K(27.0, Slots.TwoLeft), K(27.5, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));

        // MuJoCo's locomotion, 25.0-34.3 s: HalfCheetah beside CarRacing, then alone; the hopper and the walker; four more.
        Caption(overlays, 25.0, 34.2, "MUJOCO · LOCOMOTION", Orange);
        tracks.Add(Panel(Orange, Play("HalfCheetah-v5", 25.0),
            K(25.0, Slots.OffRight(Slots.TwoRight)), K(25.6, Slots.TwoRight, Ease.OutBack), K(27.0, Slots.TwoRight), K(27.5, Slots.One, Ease.InOutCubic),
            K(29.2, Slots.One), K(29.7, Slots.OffTop(Slots.One), Ease.InBack)));
        tracks.Add(Panel(Cyan, Play("Hopper-v5", 29.4),
            K(29.4, Slots.OffLeft(Slots.TwoLeft)), K(30.0, Slots.TwoLeft, Ease.OutBack), K(31.8, Slots.TwoLeft), K(32.3, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        tracks.Add(Panel(Pink, Play("Walker2d-v5", 29.5),
            K(29.5, Slots.OffRight(Slots.TwoRight)), K(30.1, Slots.TwoRight, Ease.OutBack), K(31.8, Slots.TwoRight), K(32.3, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        var locomotion = new[] { ("Ant-v5", Slots.FourTL, Yellow), ("Humanoid-v5", Slots.FourTR, Green), ("Swimmer-v5", Slots.FourBL, Cyan), ("HumanoidStandup-v5", Slots.FourBR, Pink) };
        for (int i = 0; i < locomotion.Length; i++) {
            var (id, slot, neon) = locomotion[i];
            double t0 = 32.1 + 0.1 * i;
            RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
            tracks.Add(Panel(neon, Play(id, t0 + 0.7), K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce), K(34.3, slot), K(34.8, exit, Ease.InBack)));
        }

        // Toy text, 34.3-41.25 s: four panels dropping in, each joined so its goal lands on screen, then the small lake alone.
        Caption(overlays, 34.3, 41.15, "TOY TEXT", Yellow);
        var toy = new[] {
            ("FrozenLake8x8-v1", Slots.FourTL, Cyan, 33), ("CliffWalking-v1", Slots.FourTR, Pink, 3), ("Taxi-v4", Slots.FourBL, Yellow, 7),
            ("Blackjack-v1", Slots.FourBR, Green, 0),
        };
        for (int i = 0; i < toy.Length; i++) {
            var (id, slot, neon, tick) = toy[i];
            double t0 = 34.2 + 0.1 * i;
            RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
            tracks.Add(Panel(neon, new[] { new Content(t0 + 0.7, Mid(id, t0 + 0.7, tick)) }, K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce),
                K(37.9, slot), K(38.4, exit, Ease.InBack)));
        }
        // The goal at step 41, at the video's 10 steps a second: joined at step 20, it arrives at 40.0 s.
        tracks.Add(Panel(Cyan, new[] { new Content(37.9, Mid("FrozenLake-v1", 37.9, 20, speed: 10.0 / 8.0)) },
            K(37.9, Slots.Point(Slots.One)), K(38.5, Slots.One, Ease.OutBack), K(40.75, Slots.One), K(41.25, Slots.OffBottom(Slots.One), Ease.InBack)));

        double finale = 41.25;
        if (parity is not null) {
            // Parity, 41.25-44.85 s: Gymnasium on the left, Gym.NET on the right, the same seeds and actions, a pixel count between them.
            Caption(overlays, 41.25, 44.75, "SAME SEED · SAME ACTIONS · SAME PIXELS", Green);
            Columns(overlays, 41.45, 44.65);
            var rows = new[] { (parity.Lander, Slots.FourTL, Slots.FourTR), (parity.Racer, Slots.FourBL, Slots.FourBR) };
            for (int i = 0; i < rows.Length; i++) {
                var (replay, left, right) = rows[i];
                double t0 = 41.25 + 0.15 * i;
                tracks.Add(Panel(Cyan, new[] { new Content(t0, () => new ReplaySource(replay, t0, gymNet: false)) },
                    K(t0, Slots.Point(left)), K(t0 + 0.6, left, Ease.OutBack), K(44.45, left), K(44.95, Slots.OffLeft(left), Ease.InBack)));
                tracks.Add(Panel(Pink, new[] { new Content(t0, () => new ReplaySource(replay, t0, gymNet: true)) },
                    K(t0, Slots.Point(right)), K(t0 + 0.6, right, Ease.OutBack), K(44.45, right), K(44.95, Slots.OffRight(right), Ease.InBack)));
                DiffBadge(overlays, replay, t0, 44.65, (left.Cy + right.Cy) / 2);
            }
            finale = 44.85;
        }

        // Finale: the video's four panels and order of all 24 ids, flipping every 0.45 s instead of 0.6 s.
        Caption(overlays, finale, finale + 3.0, "22 ENVS · 39 IDS · ONE API", Cyan);
        string[,] order = {
            { "LunarLander-v3", "HalfCheetah-v5", "Taxi-v4", "Ant-v5" },
            { "Hopper-v5", "CartPole-v1", "Humanoid-v5", "CarRacing-v3" },
            { "FrozenLake8x8-v1", "Walker2d-v5", "Acrobot-v1", "Reacher-v5" },
            { "InvertedDoublePendulum-v5", "BipedalWalker-v3", "Swimmer-v5", "Blackjack-v1" },
            { "MountainCar-v0", "Pusher-v5", "CliffWalking-v1", "HumanoidStandup-v5" },
            { "InvertedPendulum-v5", "Pendulum-v1", "BipedalWalkerHardcore-v3", "FrozenLake-v1" },
        };
        RectF[] quad = { Slots.FourTL, Slots.FourTR, Slots.FourBL, Slots.FourBR };
        Rgb[] neons = { Cyan, Pink, Yellow, Green };
        for (int q = 0; q < 4; q++) {
            var contents = new List<Content>();
            for (int k = 0; k < order.GetLength(0); k++) {
                double at = finale + 0.05 * q + 0.45 * k;
                contents.Add(new Content(at, Mid(order[k, q], at)));
            }
            RectF slot = quad[q];
            double t0 = finale + 0.05 * q;
            tracks.Add(Panel(neons[q], contents, K(t0, Slots.Point(slot)), K(t0 + 0.4, slot, Ease.OutBack), K(finale + 2.75, slot),
                K(finale + 3.15, Slots.Point(slot), Ease.InBack)));
        }

        double outro = finale + 3.15;
        flashes.Add(outro);
        HeaderLogo(overlays, 3.45, outro + 0.2);
        Outro(overlays, outro, outro + 3.0);
        double seconds = outro + 3.0;
        // The video's sun-rise and fades: the loop's seam is the fade from the outro to the next intro.
        return new Storyboard(seconds, 30, tracks, overlays, flashes,
            t => Ease.OutCubic((float)Math.Clamp(t / 2.2, 0, 1)),
            t => (float)Math.Min(Math.Clamp(t / 0.8, 0, 1), Math.Clamp((seconds - t) / 0.6, 0, 1)));
    }

    /// <summary>A keyframe, short for the storyboards' long lists.</summary>
    /// <param name="t">Seconds.</param>
    /// <param name="slot">The slot.</param>
    /// <param name="curve">The easing into it, linear by default.</param>
    /// <returns>The keyframe.</returns>
    private static Key K(double t, RectF slot, Func<float, float>? curve = null) => new(t, slot, curve ?? Ease.Linear);

    /// <summary>A panel with one content and its keyframes.</summary>
    /// <param name="neon">Border colour.</param>
    /// <param name="contents">What it shows.</param>
    /// <param name="keys">Where it is.</param>
    /// <returns>The track.</returns>
    private static Track Panel(Rgb neon, IReadOnlyList<Content> contents, params Key[] keys) => new(keys, contents, neon);

    /// <summary>A recipe played from its first step, starting at a time, at its own speed times a factor.</summary>
    /// <param name="id">The recipe's id.</param>
    /// <param name="at">Seconds at which the episode starts.</param>
    /// <param name="speed">Playback speed factor.</param>
    /// <returns>The content list.</returns>
    private static IReadOnlyList<Content> Play(string id, double at, double speed = 1) {
        EpisodeRecipe recipe = Episodes.Get(id);
        return new[] { new Content(at, () => new EpisodeSource(recipe, at, speed, 0)) };
    }

    /// <summary>A recipe joined mid-episode: the player is advanced to a tick before it is first shown.</summary>
    /// <param name="id">The recipe's id.</param>
    /// <param name="at">Seconds at which it appears.</param>
    /// <param name="tick">The tick it starts from; by default a fifth of the way into a typical episode.</param>
    /// <param name="speed">
    ///     Playback speed factor, as <see cref="Play"/>'s: the README cut joins Acrobot and FrozenLake mid-episode at the
    ///     speeds the video plays them at.
    /// </param>
    /// <returns>The source factory.</returns>
    private static Func<IPanelSource> Mid(string id, double at, int? tick = null, double speed = 1) {
        EpisodeRecipe recipe = Episodes.Get(id);
        int start = tick ?? (int)(recipe.StepsPerSecond * 1.5);
        return () => new EpisodeSource(recipe, at, speed, start);
    }

    /// <summary>The title: the chrome logo dropping in with a bounce, then the two lines of the pitch typing on.</summary>
    /// <param name="overlays">Where it is added.</param>
    /// <param name="start">When the logo starts falling.</param>
    /// <param name="end">When the title is gone (it shrinks away into the header logo before).</param>
    private static void Title(List<Overlay> overlays, double start, double end) {
        overlays.Add(new Overlay(start, end, (img, t, local) => {
            float drop = Ease.OutBounce((float)Math.Clamp(local / 1.0, 0, 1));
            float leave = (float)Math.Clamp((t - (end - 0.6)) / 0.6, 0, 1);
            float scale = 17 * (1 - 0.75f * Ease.InOutCubic(leave));
            float cx = 640 + (130 - 640) * Ease.InOutCubic(leave);
            float y = -200 + (150 + 200) * drop + (24 - 150) * Ease.InOutCubic(leave);
            PixelFont.DrawChrome(img, "GYM.NET", cx, y, scale, 1 - leave);
            float fadeLines = 1 - leave;
            // The pitch sits on the dark floor under the horizon, on plates, where it never crosses the bright sun.
            TypeOn(img, "REINFORCEMENT LEARNING ENVIRONMENTS FOR .NET", 640, 486, 3, Cyan, local - 1.25, fadeLines, plate: true);
            TypeOn(img, "BIT FOR BIT WITH GYMNASIUM", 640, 534, 3, Pink, local - 2.0, fadeLines, plate: true);
        }));
    }

    /// <summary>The small chrome logo in the top-left corner while the panels play.</summary>
    /// <param name="overlays">Where it is added.</param>
    /// <param name="start">When it appears (it fades in).</param>
    /// <param name="end">When it is gone.</param>
    private static void HeaderLogo(List<Overlay> overlays, double start, double end) =>
        overlays.Add(new Overlay(start, end, (img, t, local) => {
            float a = (float)Math.Min(Math.Clamp(local / 0.4, 0, 1), Math.Clamp((end - t) / 0.4, 0, 1));
            PixelFont.DrawChrome(img, "GYM.NET", 130, 22, 4.2f, a);
        }));

    /// <summary>A caption at the top centre, typed on letter by letter with a block cursor, faded out at the end.</summary>
    /// <param name="overlays">Where it is added.</param>
    /// <param name="start">When typing starts.</param>
    /// <param name="end">When it is gone.</param>
    /// <param name="text">The caption.</param>
    /// <param name="neon">Its colour.</param>
    /// <param name="typeOn">Whether to type it on (the README loop shows it whole, so the seam does not retype it).</param>
    private static void Caption(List<Overlay> overlays, double start, double end, string text, Rgb neon, bool typeOn = true) =>
        overlays.Add(new Overlay(start, end, (img, t, local) => {
            float a = (float)Math.Clamp((end - t) / 0.35, 0, 1);
            TypeOn(img, text, 690, 34, 3, neon, typeOn ? local : 100, a);
        }));

    /// <summary>The parity segment's column headers.</summary>
    /// <param name="overlays">Where they are added.</param>
    /// <param name="start">When they appear.</param>
    /// <param name="end">When they are gone.</param>
    private static void Columns(List<Overlay> overlays, double start, double end) =>
        overlays.Add(new Overlay(start, end, (img, t, local) => {
            float a = (float)Math.Min(Math.Clamp(local / 0.3, 0, 1), Math.Clamp((end - t) / 0.3, 0, 1));
            Centered(img, "GYMNASIUM 1.4.0 · PYTHON", Slots.FourTL.Cx, 80, 2, Cyan, a);
            Centered(img, "GYM.NET 1.0 · C#", Slots.FourTR.Cx, 80, 2, Pink, a);
        }));

    /// <summary>
    ///     The badge between a parity row's two panels: the number of pixels that differ in the frame on screen (always 0;
    ///     the replay is verified before rendering, and any difference stops the tool), and the step.
    /// </summary>
    /// <param name="overlays">Where it is added.</param>
    /// <param name="replay">The verified replay.</param>
    /// <param name="start">When the row appears.</param>
    /// <param name="end">When the badge is gone.</param>
    /// <param name="cy">The row's centre.</param>
    private static void DiffBadge(List<Overlay> overlays, ParityReplay replay, double start, double end, float cy) =>
        overlays.Add(new Overlay(start + 0.6, end, (img, t, local) => {
            float a = (float)Math.Min(Math.Clamp(local / 0.3, 0, 1), Math.Clamp((end - t) / 0.3, 0, 1));
            int step = ReplaySource.StepAt(replay, t - start);
            // The badge sits over the sun between the columns: a dark plate keeps the green zero readable on yellow.
            img.FillRect(new RectF(640 - 66, cy - 44, 132, 112), Rgb.Hex("#0a0218"), 0.78f * a);
            Centered(img, "PIXELS", 640, cy - 34, 2, Rgb.White, a);
            Centered(img, "DIFFERENT", 640, cy - 14, 2, Rgb.White, a);
            Centered(img, replay.DifferentPixels(step).ToString(System.Globalization.CultureInfo.InvariantCulture), 640, cy + 10, 4, Green, a);
            Centered(img, $"STEP {step}", 640, cy + 48, 2, Rgb.Hex("#b9a6d9"), a);
        }));

    /// <summary>The outro: the big logo zooming back in and the install line and address typing on.</summary>
    /// <param name="overlays">Where it is added.</param>
    /// <param name="start">When it starts.</param>
    /// <param name="end">When the video ends.</param>
    private static void Outro(List<Overlay> overlays, double start, double end) =>
        overlays.Add(new Overlay(start, end, (img, t, local) => {
            float zoom = Ease.OutBack((float)Math.Clamp(local / 0.7, 0, 1));
            PixelFont.DrawChrome(img, "GYM.NET", 640, 210 - 60 * zoom, 4 + 13 * zoom);
            TypeOn(img, "DOTNET ADD PACKAGE GYM.NET.ENVIRONMENTS", 640, 486, 3, Cyan, local - 0.8, 1, plate: true);
            TypeOn(img, "GITHUB.COM/SCISHARP/GYM.NET", 640, 534, 3, Pink, local - 1.9, 1, plate: true);
        }));

    /// <summary>Types a neon line on, centred, with a blinking block cursor while it types.</summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The full line.</param>
    /// <param name="cx">Centre x.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Font scale.</param>
    /// <param name="neon">Colour.</param>
    /// <param name="local">Seconds since typing started (negative: nothing yet).</param>
    /// <param name="alpha">Opacity.</param>
    /// <param name="plate">Whether to lay a dark plate the size of the whole line under it first.</param>
    private static void TypeOn(RgbImage img, string text, float cx, float y, float scale, Rgb neon, double local, float alpha, bool plate = false) {
        if (local < 0 || alpha <= 0)
            return;
        int shown = Math.Min(text.Length, (int)(local * 32));
        float x = cx - PixelFont.Measure(text, scale) / 2;
        if (plate)
            img.FillRect(new RectF(x - 14, y - 10, PixelFont.Measure(text, scale) + 28, PixelFont.Height * scale + 20), Rgb.Hex("#0a0218"), 0.72f * alpha);
        PixelFont.DrawNeon(img, text[..shown], x, y, scale, neon, alpha);
        bool typing = shown < text.Length;
        if (typing && (int)(local * 8) % 2 == 0)
            img.FillRect(new RectF(x + shown * PixelFont.CellWidth * scale, y, 5 * scale, 7 * scale), neon, 0.85f * alpha);
    }

    /// <summary>Draws a line of neon text centred on a point.</summary>
    /// <param name="img">The frame.</param>
    /// <param name="text">The text.</param>
    /// <param name="cx">Centre x.</param>
    /// <param name="y">Top edge.</param>
    /// <param name="scale">Font scale.</param>
    /// <param name="neon">Colour.</param>
    /// <param name="alpha">Opacity.</param>
    public static void Centered(RgbImage img, string text, float cx, float y, float scale, Rgb neon, float alpha) =>
        PixelFont.DrawNeon(img, text, cx - PixelFont.Measure(text, scale) / 2, y, scale, neon, alpha);
}

/// <summary>Helpers over a storyboard's tracks.</summary>
internal static class StoryboardExtensions {
    /// <summary>The tracks alive at a time.</summary>
    /// <param name="board">The storyboard.</param>
    /// <param name="t">Seconds.</param>
    /// <returns>The tracks whose first keyframe is at or before <paramref name="t"/> and last after it.</returns>
    public static IEnumerable<Track> LiveAt(this Storyboard board, double t) => board.Tracks.Where(k => k.Start <= t && t < k.End);
}
