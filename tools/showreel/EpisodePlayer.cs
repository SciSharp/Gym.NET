using System;
using System.Globalization;
using Gym.Envs;
using NumSharp;

namespace Gym.Showreel;

/// <summary>
///     Plays one <see cref="EpisodeRecipe"/> forward on demand: the compositor asks for the frame at a tick, and the player
///     steps the env up to it, rendering only the frame that is actually shown.
/// </summary>
/// <remarks>
///     <para>
///     A tick is one env step, or one hold between two Blackjack hands; after the last hand the player holds its final
///     frame. Players only move forward, so each panel appearance creates its own player: the policies and seeds are
///     deterministic, so two players of one recipe show the same episode.
///     </para>
///     <para>
///     Rendering is the costly part (CarRacing draws hundreds of polygons a frame), so stepping several ticks in one
///     <see cref="AdvanceTo"/> renders once, at the end, and only when <see cref="Frame"/> is read.
///     </para>
/// </remarks>
internal sealed class EpisodePlayer : IDisposable {
    /// <summary>The env as <c>make</c> builds it, with <c>render_mode="rgb_array"</c>.</summary>
    private readonly IEnv _env;

    /// <summary>The episode's policy (a fresh instance per player).</summary>
    private readonly IPolicy _policy;

    /// <summary>The last observation, fed to the policy at the next step.</summary>
    private object? _observation;

    /// <summary>The frame buffer, reused while the env keeps its frame size.</summary>
    private RgbImage? _frame;

    /// <summary>Whether the env moved since <see cref="_frame"/> was rendered.</summary>
    private bool _stale = true;

    /// <summary>Whether the current hand ended (terminated or truncated).</summary>
    private bool _handDone;

    /// <summary>Holds left before the next hand is dealt.</summary>
    private int _pauseLeft;

    /// <summary>Makes the env, resets it with the recipe's seed and creates the policy.</summary>
    /// <param name="recipe">The episode to play.</param>
    /// <exception cref="Gym.Exceptions.Error">The id is not registered (the built-in ids register before Main).</exception>
    public EpisodePlayer(EpisodeRecipe recipe) {
        Recipe = recipe;
        _env = Gymnasium.Make(recipe.Id, recipe.MakeArguments());
        _policy = recipe.Policy();
        _observation = _env.Reset(seed: recipe.Seed).Observation;
    }

    /// <summary>The recipe this player plays.</summary>
    public EpisodeRecipe Recipe { get; }

    /// <summary>Ticks consumed so far (steps plus holds).</summary>
    public int Tick { get; private set; }

    /// <summary>Steps taken in the current hand.</summary>
    public int Step { get; private set; }

    /// <summary>The current hand's return (sum of rewards, float64 as every Gym.NET reward).</summary>
    public double Return { get; private set; }

    /// <summary>The current hand, 0-based (always 0 for single-episode recipes).</summary>
    public int Hand { get; private set; }

    /// <summary>Hands won so far (positive return), for Blackjack's status line.</summary>
    public int Wins { get; private set; }

    /// <summary>Hands lost so far (negative return).</summary>
    public int Losses { get; private set; }

    /// <summary>The stamp the current hand's ending earned, or <see langword="null"/> while it plays.</summary>
    public string? Stamp { get; private set; }

    /// <summary>The tick at which <see cref="Stamp"/> appeared, so the panel can animate it popping in.</summary>
    public int StampTick { get; private set; }

    /// <summary>Whether every hand has ended; from then on the player holds its last frame.</summary>
    public bool Finished => _handDone && Hand + 1 >= Recipe.Hands;

    /// <summary>The frame at the current tick, rendered on first read after the env moved.</summary>
    /// <exception cref="InvalidOperationException">The env did not return an RGB frame.</exception>
    public RgbImage Frame {
        get {
            if (_stale || _frame is null) {
                using RenderResult r = _env.Render();
                if (r is not RenderResult.Rgb rgb)
                    throw new InvalidOperationException($"{Recipe.Id} rendered {r.GetType().Name}, not an RGB frame");
                int h = (int)rgb.Frame.shape[0], w = (int)rgb.Frame.shape[1];
                byte[] bytes = rgb.Frame.ToArray<byte>();
                if (_frame is null || _frame.Width != w || _frame.Height != h)
                    _frame = new RgbImage(w, h, bytes);
                else
                    Buffer.BlockCopy(bytes, 0, _frame.Pixels, 0, bytes.Length);
                _stale = false;
            }
            return _frame;
        }
    }

    /// <summary>
    ///     A short status for the panel's corner: step and return; for multi-episode recipes the episode's number, then
    ///     the won and lost count (Blackjack) or the episode's return (Reacher's targets).
    /// </summary>
    public string Status => Recipe.Hands <= 1
        ? $"STEP {Step}  RET {Return.ToString("0.0", CultureInfo.InvariantCulture)}"
        : Recipe.ScoresHands
            ? $"{Recipe.HandLabel} {Hand + 1}/{Recipe.Hands}  W{Wins} L{Losses}"
            : $"{Recipe.HandLabel} {Hand + 1}/{Recipe.Hands}  RET {Return.ToString("0.0", CultureInfo.InvariantCulture)}";

    /// <summary>
    ///     Steps the env (or holds between hands) until <see cref="Tick"/> reaches <paramref name="tick"/>; earlier ticks
    ///     are ignored, because a player never rewinds.
    /// </summary>
    /// <param name="tick">The tick to reach.</param>
    public void AdvanceTo(int tick) {
        while (Tick < tick) {
            if (!_handDone) {
                object action = _policy.Act(_env, _observation);
                var (next, reward, terminated, truncated, _) = _env.Step(action);
                if (action is NDArray used)
                    used.Dispose();
                (_observation as IDisposable)?.Dispose();
                _observation = next;
                Return += reward;
                Step++;
                _stale = true;
                if (terminated || truncated) {
                    _handDone = true;
                    _pauseLeft = Recipe.HandPauseTicks;
                    Stamp = Recipe.Stamp(terminated, truncated, Return);
                    StampTick = Tick + 1;
                    if (Return > 0)
                        Wins++;
                    else if (Return < 0)
                        Losses++;
                }
            } else if (Hand + 1 < Recipe.Hands) {
                if (_pauseLeft > 0) {
                    _pauseLeft--;
                } else {
                    // Next hand: the next seed, so every hand is a new deal but the whole run stays reproducible.
                    Hand++;
                    (_observation as IDisposable)?.Dispose();
                    _observation = _env.Reset(seed: Recipe.Seed + Hand).Observation;
                    Return = 0;
                    Step = 0;
                    Stamp = null;
                    _handDone = false;
                    _stale = true;
                }
            }
            Tick++;
        }
    }

    /// <summary>Closes the env, and the policy when it holds resources of its own (Pusher's scratch simulation).</summary>
    public void Dispose() {
        (_observation as IDisposable)?.Dispose();
        (_policy as IDisposable)?.Dispose();
        _env.Dispose();
    }
}
