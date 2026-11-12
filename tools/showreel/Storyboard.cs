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
///     The showreel's two timelines: the 60-second video (playful layouts of one, two or four envs, the maintainer's
///     direction) and the 12-second loop for the README.
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

    /// <summary>
    ///     The 85-second video: intro; classic control (1, 2, then 4 panels); MuJoCo's control tasks (CartPole stays on
    ///     beside its MuJoCo twin, then 4); Box2D (1, 2, 1); MuJoCo's locomotion (CarRacing stays on beside HalfCheetah,
    ///     then 1, 2 and 4); toy text (4, then 1); the parity segment (Gymnasium beside Gym.NET); a finale of four panels
    ///     flipping through all 24 ids; the outro.
    /// </summary>
    /// <param name="parity">The Gymnasium replays for the parity segment, or <see langword="null"/> to leave it out.</param>
    /// <returns>The storyboard.</returns>
    /// <remarks>
    ///     The MuJoCo sections were inserted between the existing ones (the maintainer, 2026-11-12: "insert between existing
    ///     envs the new envs"). Each section hands over to the next through a panel that stays on screen: CartPole meets
    ///     InvertedPendulum, MuJoCo's cart-pole, and CarRacing races beside HalfCheetah.
    /// </remarks>
    public static Storyboard Video(ParityReplays? parity) {
        var tracks = new List<Track>();
        var overlays = new List<Overlay>();
        var flashes = new List<double> { 1.62 };

        Title(overlays, 0.55, 4.25);

        // Classic control: one panel, then two, then four; CartPole stays for the hand-over to MuJoCo.
        Caption(overlays, 4.0, 15.2, "CLASSIC CONTROL", Cyan);
        tracks.Add(Panel(Cyan, new[] {
                new Content(4.0, () => new EpisodeSource(Episodes.Get("CartPole-v1"), 4.0, 1, 0)),
                // Its first episode ends at 500 steps (14 s); a fresh one, flipped in mid-move, keeps it balancing beside its twin.
                new Content(15.3, Mid("CartPole-v1", 15.3, 60)),
            },
            K(4.0, Slots.OffBottom(Slots.One)), K(4.6, Slots.One, Ease.OutBack), K(8.0, Slots.One), K(8.6, Slots.TwoLeft, Ease.InOutCubic),
            K(11.4, Slots.TwoLeft), K(12.0, Slots.FourTL, Ease.InOutCubic), K(15.0, Slots.FourTL), K(15.6, Slots.TwoLeft, Ease.InOutCubic),
            K(18.6, Slots.TwoLeft), K(19.2, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        tracks.Add(Panel(Pink, Play("MountainCar-v0", 8.0),
            K(8.0, Slots.OffRight(Slots.TwoRight)), K(8.6, Slots.TwoRight, Ease.OutBack), K(11.4, Slots.TwoRight), K(12.0, Slots.FourTR, Ease.InOutCubic),
            K(15.0, Slots.FourTR), K(15.6, Slots.OffRight(Slots.FourTR), Ease.InBack)));
        tracks.Add(Panel(Yellow, Play("Pendulum-v1", 11.5),
            K(11.5, Slots.OffBottom(Slots.FourBL)), K(12.1, Slots.FourBL, Ease.OutBack), K(15.05, Slots.FourBL), K(15.65, Slots.OffBottom(Slots.FourBL), Ease.InBack)));
        tracks.Add(Panel(Green, Play("Acrobot-v1", 11.65, speed: 20.0 / 15.0),
            K(11.65, Slots.OffBottom(Slots.FourBR)), K(12.25, Slots.FourBR, Ease.OutBack), K(15.1, Slots.FourBR), K(15.7, Slots.OffBottom(Slots.FourBR), Ease.InBack)));

        // MuJoCo's control tasks: InvertedPendulum beside CartPole, then four panels.
        Caption(overlays, 15.3, 23.9, "MUJOCO PHYSICS · BIT FOR BIT, IN C#", Orange);
        tracks.Add(Panel(Orange, Play("InvertedPendulum-v5", 15.3),
            K(15.3, Slots.OffRight(Slots.TwoRight)), K(15.9, Slots.TwoRight, Ease.OutBack), K(18.6, Slots.TwoRight), K(19.2, Slots.FourTL, Ease.InOutCubic),
            K(23.2, Slots.FourTL), K(23.8, Slots.OffLeft(Slots.FourTL), Ease.InBack)));
        var control = new[] { ("InvertedDoublePendulum-v5", Slots.FourTR, Pink), ("Reacher-v5", Slots.FourBL, Yellow), ("Pusher-v5", Slots.FourBR, Green) };
        for (int i = 0; i < control.Length; i++) {
            var (id, slot, neon) = control[i];
            double t0 = 18.7 + 0.1 * i;
            RectF entry = slot.Cy < 360 ? Slots.OffTop(slot) : Slots.OffBottom(slot);
            RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
            tracks.Add(Panel(neon, Play(id, t0), K(t0, entry), K(t0 + 0.7, slot, Ease.OutBack), K(23.25 + 0.05 * i, slot),
                K(23.85 + 0.05 * i, exit, Ease.InBack)));
        }

        // Box2D: the lander alone, the two walkers side by side, then the race car, which stays for the hand-over.
        Caption(overlays, 24.0, 41.8, "BOX2D", Pink);
        tracks.Add(Panel(Cyan, Play("LunarLander-v3", 24.0),
            K(24.0, Slots.Point(Slots.One)), K(24.6, Slots.One, Ease.OutBack), K(30.8, Slots.One), K(31.3, Slots.OffTop(Slots.One), Ease.InBack)));
        tracks.Add(Panel(Yellow, Play("BipedalWalker-v3", 31.0),
            K(31.0, Slots.OffLeft(Slots.TwoLeft)), K(31.6, Slots.TwoLeft, Ease.OutBack), K(36.2, Slots.TwoLeft), K(36.7, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        tracks.Add(Panel(Orange, Play("BipedalWalkerHardcore-v3", 31.1),
            K(31.1, Slots.OffRight(Slots.TwoRight)), K(31.7, Slots.TwoRight, Ease.OutBack), K(36.2, Slots.TwoRight), K(36.7, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        tracks.Add(Panel(Pink, Play("CarRacing-v3", 36.5),
            K(36.5, Slots.OffBottom(Slots.One)), K(37.1, Slots.One, Ease.OutBack), K(41.8, Slots.One), K(42.4, Slots.TwoLeft, Ease.InOutCubic),
            K(45.2, Slots.TwoLeft), K(45.7, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));

        // MuJoCo's locomotion: HalfCheetah beside CarRacing, then alone; the hopper and the walker; four more.
        Caption(overlays, 41.9, 57.6, "MUJOCO · LOCOMOTION", Orange);
        tracks.Add(Panel(Orange, Play("HalfCheetah-v5", 41.9),
            K(41.9, Slots.OffRight(Slots.TwoRight)), K(42.5, Slots.TwoRight, Ease.OutBack), K(45.2, Slots.TwoRight), K(45.8, Slots.One, Ease.InOutCubic),
            K(48.4, Slots.One), K(48.9, Slots.OffTop(Slots.One), Ease.InBack)));
        tracks.Add(Panel(Cyan, Play("Hopper-v5", 48.6),
            K(48.6, Slots.OffLeft(Slots.TwoLeft)), K(49.2, Slots.TwoLeft, Ease.OutBack), K(52.6, Slots.TwoLeft), K(53.1, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        tracks.Add(Panel(Pink, Play("Walker2d-v5", 48.7),
            K(48.7, Slots.OffRight(Slots.TwoRight)), K(49.3, Slots.TwoRight, Ease.OutBack), K(52.6, Slots.TwoRight), K(53.1, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        var locomotion = new[] { ("Ant-v5", Slots.FourTL, Yellow), ("Humanoid-v5", Slots.FourTR, Green), ("Swimmer-v5", Slots.FourBL, Cyan), ("HumanoidStandup-v5", Slots.FourBR, Pink) };
        for (int i = 0; i < locomotion.Length; i++) {
            var (id, slot, neon) = locomotion[i];
            double t0 = 52.9 + 0.1 * i;
            RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
            tracks.Add(Panel(neon, Play(id, t0 + 0.7), K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce), K(57.6, slot), K(58.1, exit, Ease.InBack)));
        }

        // Toy text: four panels dropping in with a bounce, then the small lake alone.
        Caption(overlays, 58.1, 69.5, "TOY TEXT", Yellow);
        var toy = new[] { ("FrozenLake8x8-v1", Slots.FourTL, Cyan), ("CliffWalking-v1", Slots.FourTR, Pink), ("Taxi-v4", Slots.FourBL, Yellow), ("Blackjack-v1", Slots.FourBR, Green) };
        for (int i = 0; i < toy.Length; i++) {
            var (id, slot, neon) = toy[i];
            double t0 = 58.0 + 0.1 * i;
            RectF exit = slot.Cx < 640 ? Slots.OffLeft(slot) : Slots.OffRight(slot);
            tracks.Add(Panel(neon, Play(id, t0 + 0.7), K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce), K(64.0, slot), K(64.5, exit, Ease.InBack)));
        }
        tracks.Add(Panel(Cyan, Play("FrozenLake-v1", 64.3, speed: 10.0 / 8.0),
            K(64.3, Slots.Point(Slots.One)), K(64.9, Slots.One, Ease.OutBack), K(69.1, Slots.One), K(69.6, Slots.OffBottom(Slots.One), Ease.InBack)));

        double finale = 69.7;
        if (parity is not null) {
            // Parity: Gymnasium on the left, Gym.NET on the right, the same seeds and actions, a pixel count between them.
            Caption(overlays, 69.7, 76.5, "SAME SEED · SAME ACTIONS · SAME PIXELS", Green);
            Columns(overlays, 70.0, 76.3);
            var rows = new[] { (parity.Lander, Slots.FourTL, Slots.FourTR), (parity.Racer, Slots.FourBL, Slots.FourBR) };
            for (int i = 0; i < rows.Length; i++) {
                var (replay, left, right) = rows[i];
                double t0 = 69.7 + 0.15 * i;
                tracks.Add(Panel(Cyan, new[] { new Content(t0, () => new ReplaySource(replay, t0, gymNet: false)) },
                    K(t0, Slots.Point(left)), K(t0 + 0.6, left, Ease.OutBack), K(76.1, left), K(76.6, Slots.OffLeft(left), Ease.InBack)));
                tracks.Add(Panel(Pink, new[] { new Content(t0, () => new ReplaySource(replay, t0, gymNet: true)) },
                    K(t0, Slots.Point(right)), K(t0 + 0.6, right, Ease.OutBack), K(76.1, right), K(76.6, Slots.OffRight(right), Ease.InBack)));
                DiffBadge(overlays, replay, t0, 76.3, (left.Cy + right.Cy) / 2);
            }
            finale = 76.5;
        }

        // Finale: four panels flipping through all 24 ids, mid-episode, every 0.6 s. Each row of the table is one moment of
        // the four panels, so every moment mixes families; each column is one panel's run of six.
        Caption(overlays, finale, finale + 4.0, "22 ENVS · 39 IDS · ONE API", Cyan);
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
                double at = finale + 0.05 * q + 0.6 * k;
                contents.Add(new Content(at, Mid(order[k, q], at)));
            }
            RectF slot = quad[q];
            double t0 = finale + 0.05 * q;
            tracks.Add(Panel(neons[q], contents, K(t0, Slots.Point(slot)), K(t0 + 0.5, slot, Ease.OutBack), K(finale + 3.7, slot),
                K(finale + 4.2, Slots.Point(slot), Ease.InBack)));
        }

        double outro = finale + 4.2;
        flashes.Add(outro);
        HeaderLogo(overlays, 3.7, outro + 0.2);
        Outro(overlays, outro, outro + 4.4);
        double seconds = outro + 4.4;
        return new Storyboard(seconds, 30, tracks, overlays, flashes,
            t => Ease.OutCubic((float)Math.Clamp(t / 2.2, 0, 1)),
            t => (float)Math.Min(Math.Clamp(t / 0.8, 0, 1), Math.Clamp((seconds - t) / 0.6, 0, 1)));
    }

    /// <summary>
    ///     The README's 12-second loop: the lander alone; HalfCheetah and the race car side by side; FrozenLake, Ant,
    ///     CartPole and Hopper; then a white flash that hides the seam where the GIF starts over.
    /// </summary>
    /// <returns>The storyboard (rendered at 1280x720, scaled down by the encoder).</returns>
    public static Storyboard Hero() {
        var tracks = new List<Track>();
        var overlays = new List<Overlay>();
        HeaderLogo(overlays, 0, 12);
        Caption(overlays, 0, 12, "RL ENVIRONMENTS FOR .NET · BIT FOR BIT", Cyan, typeOn: false);
        tracks.Add(Panel(Cyan, new[] { new Content(0, Mid("LunarLander-v3", 0, 165)) },
            K(0, Slots.One), K(3.0, Slots.One), K(3.5, Slots.OffTop(Slots.One), Ease.InBack)));
        tracks.Add(Panel(Orange, new[] { new Content(3.0, Mid("HalfCheetah-v5", 3.0, 40)) },
            K(3.0, Slots.OffLeft(Slots.TwoLeft)), K(3.6, Slots.TwoLeft, Ease.OutBack), K(6.6, Slots.TwoLeft), K(7.1, Slots.OffLeft(Slots.TwoLeft), Ease.InBack)));
        tracks.Add(Panel(Pink, new[] { new Content(3.1, Mid("CarRacing-v3", 3.1, 140)) },
            K(3.1, Slots.OffRight(Slots.TwoRight)), K(3.7, Slots.TwoRight, Ease.OutBack), K(6.6, Slots.TwoRight), K(7.1, Slots.OffRight(Slots.TwoRight), Ease.InBack)));
        var four = new[] { ("FrozenLake8x8-v1", Slots.FourTL, Cyan, 20), ("Ant-v5", Slots.FourTR, Yellow, 40), ("CartPole-v1", Slots.FourBL, Green, 100), ("Hopper-v5", Slots.FourBR, Pink, 150) };
        for (int i = 0; i < four.Length; i++) {
            var (id, slot, neon, tick) = four[i];
            double t0 = 6.9 + 0.1 * i;
            tracks.Add(Panel(neon, new[] { new Content(t0, Mid(id, t0, tick)) },
                K(t0, Slots.OffTop(slot)), K(t0 + 0.8, slot, Ease.OutBounce), K(11.2, slot), K(11.7, Slots.Point(slot), Ease.InBack)));
        }
        return new Storyboard(12, 30, tracks, overlays, new[] { 12.0, 0.0 }, _ => 1, _ => 1);
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
    /// <returns>The source factory.</returns>
    private static Func<IPanelSource> Mid(string id, double at, int? tick = null) {
        EpisodeRecipe recipe = Episodes.Get(id);
        int start = tick ?? (int)(recipe.StepsPerSecond * 1.5);
        return () => new EpisodeSource(recipe, at, 1, start);
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
