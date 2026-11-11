using System;

namespace Gym.Showreel;

/// <summary>
///     What a panel shows: a frame for any time at or after the previous call (sources only move forward), the env id for
///     the label, a status line and the stamp the ending earned.
/// </summary>
internal interface IPanelSource : IDisposable {
    /// <summary>The registered id, shown under the panel.</summary>
    string Title { get; }

    /// <summary>The status line drawn in the panel's corner, as of the last <see cref="FrameAt"/>.</summary>
    string Status { get; }

    /// <summary>The stamp the episode's ending earned, or <see langword="null"/> while it plays.</summary>
    string? Stamp { get; }

    /// <summary>Seconds since <see cref="Stamp"/> appeared, for its pop-in animation.</summary>
    double StampAge { get; }

    /// <summary>The frame at a time of the storyboard.</summary>
    /// <param name="t">Seconds; times before the source started show its first frame.</param>
    /// <returns>The frame; valid until the next call, which may overwrite it.</returns>
    RgbImage FrameAt(double t);
}

/// <summary>A panel source playing an <see cref="EpisodeRecipe"/> live, at its speed from a start time.</summary>
internal sealed class EpisodeSource : IPanelSource {
    /// <summary>The player.</summary>
    private readonly EpisodePlayer _player;

    /// <summary>Storyboard seconds at which the source shows <see cref="_startTick"/>.</summary>
    private readonly double _start;

    /// <summary>Ticks per storyboard second: the recipe's speed times the storyboard's factor.</summary>
    private readonly double _ticksPerSecond;

    /// <summary>The tick shown at <see cref="_start"/> (non-zero to join an episode mid-way).</summary>
    private readonly int _startTick;

    /// <summary>Creates the player and advances it to the start tick (stepping without rendering).</summary>
    /// <param name="recipe">The episode.</param>
    /// <param name="start">Storyboard seconds at which it starts playing.</param>
    /// <param name="speed">Factor on the recipe's own speed.</param>
    /// <param name="startTick">The tick to join at; 0 plays from the reset.</param>
    public EpisodeSource(EpisodeRecipe recipe, double start, double speed, int startTick) {
        _player = new EpisodePlayer(recipe);
        _start = start;
        _ticksPerSecond = recipe.StepsPerSecond * speed;
        _startTick = startTick;
        _player.AdvanceTo(startTick);
    }

    /// <inheritdoc />
    public string Title => _player.Recipe.Id;

    /// <inheritdoc />
    public string Status => _player.Status;

    /// <inheritdoc />
    public string? Stamp => _player.Stamp;

    /// <inheritdoc />
    public double StampAge => (_player.Tick - _player.StampTick) / _ticksPerSecond;

    /// <inheritdoc />
    public RgbImage FrameAt(double t) {
        int tick = _startTick + (int)Math.Floor(Math.Max(0, t - _start) * _ticksPerSecond);
        _player.AdvanceTo(tick);
        return _player.Frame;
    }

    /// <inheritdoc />
    public void Dispose() => _player.Dispose();
}

/// <summary>
///     A panel source replaying one side of the parity segment, Gymnasium's frames or Gym.NET's, at the recipe's speed,
///     from the verified replay: both columns show exactly the frames that were compared.
/// </summary>
internal sealed class ReplaySource : IPanelSource {
    /// <summary>The verified replay.</summary>
    private readonly ParityReplay _replay;

    /// <summary>Storyboard seconds of step 0.</summary>
    private readonly double _start;

    /// <summary>Whether this source shows Gym.NET's frames rather than Gymnasium's.</summary>
    private readonly bool _gymNet;

    /// <summary>The step last shown.</summary>
    private int _step;

    /// <summary>Wraps one side of a replay starting at a time.</summary>
    /// <param name="replay">The replay; owned by its <see cref="ParityReplays"/>, not by this source.</param>
    /// <param name="start">Storyboard seconds at which step 0 is shown.</param>
    /// <param name="gymNet">Show Gym.NET's frames (true) or Gymnasium's (false).</param>
    public ReplaySource(ParityReplay replay, double start, bool gymNet) {
        _replay = replay;
        _start = start;
        _gymNet = gymNet;
    }

    /// <summary>The replay step shown a number of seconds after the start, at the recipe's speed, held at the last step.</summary>
    /// <param name="replay">The replay.</param>
    /// <param name="local">Seconds since step 0 was shown.</param>
    /// <returns>The step, 0 to <c>Steps - 1</c>.</returns>
    public static int StepAt(ParityReplay replay, double local) =>
        Math.Clamp((int)Math.Floor(Math.Max(0, local) * replay.Recipe.StepsPerSecond), 0, replay.Steps - 1);

    /// <inheritdoc />
    public string Title => _replay.Recipe.Id;

    /// <inheritdoc />
    public string Status => $"STEP {_step}  RET {_replay.ReturnAt(_step).ToString("0.0", System.Globalization.CultureInfo.InvariantCulture)}";

    /// <inheritdoc />
    public string? Stamp => _step == _replay.Steps - 1 && _replay.Ended ? _replay.FinalStamp : null;

    /// <inheritdoc />
    public double StampAge => 0.35;

    /// <inheritdoc />
    public RgbImage FrameAt(double t) {
        _step = StepAt(_replay, t - _start);
        return _gymNet ? _replay.GymNetFrame(_step) : _replay.GymnasiumFrame(_step);
    }

    /// <summary>Nothing to release: the replay's file belongs to its owner.</summary>
    public void Dispose() { }
}
