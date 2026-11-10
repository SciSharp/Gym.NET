using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using Gym.Environments;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     What <see cref="Play.Run"/> calls after every step (Gymnasium's <c>play</c> callback), with the transition the
    ///     player caused. <see cref="PlayPlot.Callback"/> is one.
    /// </summary>
    /// <param name="obsT">
    ///     The observation before the action. After a reset it is the whole reset result, a <see cref="ResetResult{TObs}"/>
    ///     (Gymnasium passes the <c>(obs, info)</c> tuple there, and so does Gym.NET).
    /// </param>
    /// <param name="obsTp1">The observation after the action.</param>
    /// <param name="action">The action played: the mapped action of the held keys, or the no-op.</param>
    /// <param name="rew">The reward.</param>
    /// <param name="terminated">Whether the episode ended in a terminal state.</param>
    /// <param name="truncated">Whether the episode was cut short.</param>
    /// <param name="info">The step's info.</param>
    /// <remarks>
    ///     The arguments are borrowed: <c>play</c> releases each observation and info once it is done with them (an
    ///     observation after its second callback, as <paramref name="obsTp1"/> then <paramref name="obsT"/>). Copy what you
    ///     keep.
    /// </remarks>
    public delegate void PlayCallback(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info);

    /// <summary>
    ///     Gymnasium's <c>gymnasium.utils.play</c>: play an env from the keyboard (<see cref="Run"/>, <c>play</c>, or
    ///     <see cref="RunAsync"/> to await a game), and show a frame on a surface (<see cref="DisplayArr"/>,
    ///     <c>display_arr</c>). The window is a <see cref="PygameContext"/>'s display, <see cref="PygameContext.Default"/>'s
    ///     (the static <see cref="Display"/>) unless a context is given, made by the viewer factory (DR-20); the keys come
    ///     from it through the context's event queue, or from events a program posts there.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     To play for real, set a viewer that reports input (<c>Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory</c>,
    ///     or pass <c>viewerFactory</c>), then <c>Play.Run(Gymnasium.Make("CartPole-v1", new { render_mode = "rgb_array" }),
    ///     keysToAction: new Dictionary&lt;object, object?&gt; { ["a"] = 0, ["d"] = 1 })</c>. Headless, under
    ///     <see cref="NullEnvViewer"/>, nothing reports keys, so the loop runs on the no-op until events are posted (a callback
    ///     can post them, as Gymnasium's tests do).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py#L185">gymnasium/utils/play.py</see>.</para>
    /// </remarks>
    public static class Play {
        /// <summary>
        ///     Lets a person play <paramref name="env"/> with the keyboard (Gymnasium's <c>play</c>): resets with
        ///     <paramref name="seed"/>, opens the display, then each frame steps with the action of the keys held (or
        ///     <paramref name="noop"/>), calls <paramref name="callback"/>, shows the frame, reads the events and paces itself,
        ///     resetting with the same seed whenever an episode ends, until Escape, the window's close button or a
        ///     <see cref="EventType.Quit"/> event. Then it closes the display.
        /// </summary>
        /// <param name="env">The env, in render mode <c>rgb_array</c> or <c>rgb_array_list</c>; it is not closed.</param>
        /// <param name="transpose">Whether frames are transposed for display (<see langword="true"/>, the default, shows an <c>(H, W, 3)</c> frame upright); <see langword="null"/> is false.</param>
        /// <param name="fps">
        ///     The most steps per second; <see langword="null"/> reads <c>metadata["render_fps"]</c> (30 when absent), which must
        ///     then be a Python int.
        /// </param>
        /// <param name="zoom">Scales the window; must be positive.</param>
        /// <param name="callback">Called after every step, with borrowed arguments (see <see cref="PlayCallback"/>).</param>
        /// <param name="keysToAction">
        ///     A dictionary from keys to actions, or <see langword="null"/> for the env's own (<c>get_keys_to_action</c>,
        ///     through any wrapper). A key is a key code (<c>KeyCode.Space</c>, an int), a string whose characters are held
        ///     together (<c>"wa"</c>), a <see cref="KeyCombination"/>, or a tuple of codes and one-character strings held together
        ///     (<c>(KeyCode.Left, KeyCode.Right)</c>, <c>ValueTuple.Create("w")</c>); every action must be in the action space.
        ///     A given dictionary stays the caller's; the env's own is a new one <c>play</c> owns and releases (its actions
        ///     included) when it returns or throws (DR-45).
        /// </param>
        /// <param name="seed">The seed of every reset; <see langword="null"/> does not reseed.</param>
        /// <param name="noop">The action when no mapped combination is held; <see langword="null"/> is Gymnasium's default, the int 0.</param>
        /// <param name="waitOnPlayer">Step only while a mapped key is held (turn-based envs); otherwise the no-op plays on.</param>
        /// <param name="viewerFactory">
        ///     The window's factory; <see langword="null"/> takes the context's <see cref="PygameContext.ViewerFactory"/>, then
        ///     <c>Gymnasium.HumanViewerFactory</c>.
        /// </param>
        /// <param name="context">
        ///     The display and event queue the game runs in (DR-41); <see langword="null"/> is <see cref="PygameContext.Default"/>,
        ///     the static <see cref="Display"/> and <see cref="EventQueue"/>. Two games at once need two contexts.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="Error">Thrown for a negative seed, by the first reset.</exception>
        /// <exception cref="MissingKeysToAction">Thrown when no mapping is given and the env has none: <c>CartPole-v1 does not have explicit key to action mapping, please specify one manually</c>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown, with Gymnasium's messages, when no mapping is given and the env has none and no spec, when the env's
        ///     mapping is <c>None</c>, for an empty key tuple (MountainCar's own mapping has one), for an action outside the
        ///     action space, for the wrong render mode, and for a zoom that is not positive.
        /// </exception>
        /// <exception cref="TypeError">
        ///     Thrown, with Gymnasium's messages, when the mapping is no dictionary, a key is neither an int, a string nor a
        ///     tuple of them, a tuple element is a string of another length than one (<c>ord()</c>'s message), or
        ///     <c>render_fps</c> is not an int.
        /// </exception>
        /// <exception cref="AssertionError">Thrown, with an empty message, when a render is not a uint8 array.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown when no viewer factory is given or set.</exception>
        /// <remarks>
        ///     <para>
        ///     Gymnasium's own quirks are kept: the callback's first <c>obs_t</c> of an episode is the reset's
        ///     <c>(obs, info)</c>; every reset reuses <paramref name="seed"/>, so a seeded game repeats its starting state; and
        ///     an exception leaves the display open (call the context's <see cref="PygameDisplay.Quit"/>, <see cref="Display.Quit"/>
        ///     for the default one), as pygame stays initialized.
        ///     </para>
        ///     <para>
        ///     One game per context at a time; see <see cref="PygameEventQueue"/>. The game runs on the calling thread, which
        ///     the frame clock blocks between frames, as pygame's loop does; <see cref="RunAsync"/> runs the same loop without
        ///     blocking the caller. The env is stepped through its <c>StepAsync</c>, which every built-in env completes on the
        ///     calling thread; an env whose steps are genuinely asynchronous continues the game on a thread-pool thread after
        ///     its first such step, while this call waits.
        ///     </para>
        /// </remarks>
        [PyName("play")]
        public static void Run(IEnv env, bool? transpose = true, double? fps = null, double? zoom = null, PlayCallback? callback = null,
            object? keysToAction = null, BigInteger? seed = null, object? noop = null, bool waitOnPlayer = false,
            IEnvironmentViewerFactoryDelegate? viewerFactory = null, PygameContext? context = null) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            // The one loop body (DR-44) with blocking pacing: every await in it completes synchronously for an env whose steps
            // do, so the game runs on this thread, and GetResult rethrows a failure as it was thrown.
            RunLoopAsync(env, transpose, fps, zoom, callback, keysToAction, seed, noop, waitOnPlayer, viewerFactory, context ?? PygameContext.Default,
                awaitPacing: false, CancellationToken.None).GetAwaiter().GetResult();
        }

        /// <summary>
        ///     <see cref="Run"/> as a task that completes when the game ends (DR-44): the same loop, on a thread of its own when
        ///     the game shows a real window, on the thread pool when it is headless, so the caller (a UI, a test, an agent) can
        ///     await a game, or several in their own contexts, without blocking its thread.
        /// </summary>
        /// <param name="env">The env, in render mode <c>rgb_array</c> or <c>rgb_array_list</c>; it is not closed.</param>
        /// <param name="transpose">Whether frames are transposed for display, as in <see cref="Run"/>.</param>
        /// <param name="fps">The most steps per second, as in <see cref="Run"/>.</param>
        /// <param name="zoom">Scales the window; must be positive.</param>
        /// <param name="callback">Called after every step, with borrowed arguments, on the game's thread (not the caller's synchronization context).</param>
        /// <param name="keysToAction">A dictionary from keys to actions, or <see langword="null"/> for the env's own, as in <see cref="Run"/>.</param>
        /// <param name="seed">The seed of every reset; <see langword="null"/> does not reseed.</param>
        /// <param name="noop">The action when no mapped combination is held; <see langword="null"/> is the int 0.</param>
        /// <param name="waitOnPlayer">Step only while a mapped key is held (turn-based envs).</param>
        /// <param name="viewerFactory">The window's factory; <see langword="null"/> takes the context's, then <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <param name="context">The display and event queue the game runs in; <see langword="null"/> is <see cref="PygameContext.Default"/>.</param>
        /// <param name="cancellationToken">
        ///     Ends the game as Escape or the window's close button does: the current frame finishes (a frame wait or an env's
        ///     asynchronous step is cut short), the display closes and the task completes normally, not as canceled.
        /// </param>
        /// <returns>A task that completes when the game ends, faulted with what <see cref="Run"/> would have thrown.</returns>
        /// <exception cref="ArgumentNullException">Thrown here, not through the task, when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <remarks>
        ///     <para>
        ///     Where it runs follows DR-22's rule for thread-bound envs: a game whose env renders into a window of its own
        ///     (<c>render_mode="human"</c>, which <see cref="PlayableGame"/> then refuses) or whose display is a real window
        ///     (any viewer factory but <see cref="NullEnvViewer.Factory"/>) runs on a dedicated background thread, paced by
        ///     the blocking clock, as pygame's loop is; a headless game (<see cref="NullEnvViewer.Factory"/>) runs on the
        ///     thread pool with awaited pacing (<see cref="Clock.TickAsync"/>), holding no thread between frames. The dedicated
        ///     thread is a background thread: the process does not wait for a game to end.
        ///     </para>
        ///     <para>
        ///     Every error <see cref="Run"/> throws surfaces through the task instead, Gymnasium's messages unchanged. Games
        ///     that share a context steal each other's events, as in <see cref="Run"/>; give each its own
        ///     <see cref="PygameContext"/>.
        ///     </para>
        /// </remarks>
        public static Task RunAsync(IEnv env, bool? transpose = true, double? fps = null, double? zoom = null, PlayCallback? callback = null,
            object? keysToAction = null, BigInteger? seed = null, object? noop = null, bool waitOnPlayer = false,
            IEnvironmentViewerFactoryDelegate? viewerFactory = null, PygameContext? context = null, CancellationToken cancellationToken = default) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            PygameContext game = context ?? PygameContext.Default;
            // DR-22's auto-switch: a window binds the game to a thread; a headless game borrows the pool between frames.
            if (env.RenderMode is "human" || game.OpensRealWindow(viewerFactory)) {
                return RunOnOwnThread(() => RunLoopAsync(env, transpose, fps, zoom, callback, keysToAction, seed, noop, waitOnPlayer, viewerFactory, game,
                    awaitPacing: false, cancellationToken));
            }

            return Task.Run(() => RunLoopAsync(env, transpose, fps, zoom, callback, keysToAction, seed, noop, waitOnPlayer, viewerFactory, game,
                awaitPacing: true, cancellationToken));
        }

        /// <summary>
        ///     The one loop body of <see cref="Run"/> and <see cref="RunAsync"/> (DR-44): Gymnasium's <c>play</c>, stepping
        ///     through the env's <c>StepAsync</c> and pacing through the clock, blocking or awaited.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <param name="transpose">Whether frames are transposed for display.</param>
        /// <param name="fps">The most steps per second, or <see langword="null"/> for the env's <c>render_fps</c>.</param>
        /// <param name="zoom">The window's scale.</param>
        /// <param name="callback">The per-step callback.</param>
        /// <param name="keysToAction">The key mapping, or <see langword="null"/> for the env's own.</param>
        /// <param name="seed">The seed of every reset.</param>
        /// <param name="noop">The action when nothing mapped is held.</param>
        /// <param name="waitOnPlayer">Step only while a mapped key is held.</param>
        /// <param name="viewerFactory">The window's factory.</param>
        /// <param name="context">The display and event queue.</param>
        /// <param name="awaitPacing">Pace with <see cref="Clock.TickAsync"/> (no thread held) instead of <see cref="Clock.Tick"/>.</param>
        /// <param name="cancellationToken">Ends the game after the current frame, as Escape does.</param>
        /// <returns>A task that completes when the game ends.</returns>
        /// <exception cref="Exception">Every exception <see cref="Run"/> documents, through the task.</exception>
        private static async Task RunLoopAsync(IEnv env, bool? transpose, double? fps, double? zoom, PlayCallback? callback, object? keysToAction,
            BigInteger? seed, object? noop, bool waitOnPlayer, IEnvironmentViewerFactoryDelegate? viewerFactory, PygameContext context, bool awaitPacing,
            CancellationToken cancellationToken) {
            object noopAction = noop ?? 0;
            Release(env.Reset(seed));

            // A mapping taken from the env is a new one play owns (DR-45): its actions are released when play returns or throws,
            // a refused mapping (MountainCar's empty combination) included.
            object? envMapping = null;
            if (keysToAction is null) {
                if (env.HasWrapperAttr("get_keys_to_action")) {
                    keysToAction = envMapping = PlayKeys.Call(env.GetWrapperAttr("get_keys_to_action"));
                } else {
                    if (env.Spec is null)
                        throw new ValueError("The environment must have an `env.spec` to auto-detect missing keys to action mappings.");
                    throw new MissingKeysToAction($"{env.Spec.Id} does not have explicit key to action mapping, please specify one manually");
                }
            }

            try {
                if (keysToAction is null)
                    throw new ValueError("keys_to_action dictionary cannot be None");
                List<KeyValuePair<object?, object?>> entries = PlayKeys.DictEntries(keysToAction)
                                                               ?? throw new TypeError($"keys_to_action must be a dictionary, got {PyBuiltins.TypeRepr(keysToAction)}");

                // Validate every entry before converting any, key then action, in the mapping's order.
                foreach ((object? key, object? action) in entries) {
                    if (PlayKeys.TupleItems(key) is { } combination) {
                        if (combination.Count == 0)
                            throw new ValueError("Key combinations in keys_to_action tuples cannot be empty");
                        if (!combination.All(k => PlayKeys.IsStr(k) || PlayKeys.IsInt(k)))
                            throw new TypeError($"All keys in key combination tuple must be integers or strings, got {PyBuiltins.Str(key)}");
                    } else if (!PlayKeys.IsStr(key) && !PlayKeys.IsInt(key)) {
                        throw new TypeError($"Keys in keys_to_action must be integers or strings, got {PyBuiltins.TypeRepr(key)}");
                    }

                    if (!env.ActionSpace.Contains(action))
                        throw new ValueError($"Action {PyBuiltins.Str(action)} is not a valid action in the environment action space: {PyBuiltins.Str(env.ActionSpace)}");
                }

                // Every key becomes the sorted tuple of its codes (strings through ord), so a held set looks its action up directly.
                var keyCodeToAction = new Dictionary<object, object?>(PyKeyComparer.Instance);
                foreach ((object? keyCombination, object? action) in entries) {
                    IReadOnlyList<object?> keys = PlayKeys.IsInt(keyCombination) ? new[] { keyCombination } : PlayKeys.Iterate(keyCombination);
                    object?[] keyCode = PlayKeys.Sorted(keys.Select(k => k is string character ? PlayKeys.Ord(character) : k)).ToArray();
                    keyCodeToAction[keyCode] = action;
                }

                var game = new PlayableGame(env, keyCodeToAction, zoom, viewerFactory, context);

                double framerate;
                if (fps is { } given) {
                    framerate = given;
                } else {
                    object? metadataFps = env.Metadata.TryGetValue(Metadata.RenderFpsKey, out object? value) ? value : 30;
                    if (!PlayKeys.IsInt(metadataFps))
                        throw new TypeError($"Expected env.metadata['render_fps'] to be an integer, got {PyBuiltins.TypeRepr(metadataFps)}");
                    framerate = PyBuiltins.Float(metadataFps);
                }

                bool done = true;
                // Gymnasium's obs: the reset's (obs, info) after a reset, the step's observation after a step.
                object? obs = null;
                var clock = new Clock();
                while (game.Running) {
                    if (done) {
                        done = false;
                        // The episode's last observation was borrowed by its last callback only; nothing refers to it now.
                        Release(obs);
                        obs = env.Reset(seed);
                    } else if (!waitOnPlayer || game.PressedKeys.Count > 0) {
                        object?[] held = PlayKeys.Sorted(game.PressedKeys).ToArray();
                        object? action = keyCodeToAction.TryGetValue(held, out object? mapped) ? mapped : noopAction;
                        object? previous = obs;
                        StepResult<object?> step;
                        try {
                            // Every built-in env completes this on the calling thread, so the blocking loop stays where it is.
                            step = await env.StepAsync(action, cancellationToken).ConfigureAwait(false);
                        } catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) {
                            // Canceled before (or during) the step: the game ends as with Escape, and the step did not count.
                            break;
                        }

                        obs = step.Observation;
                        done = step.Terminated || step.Truncated;
                        callback?.Invoke(previous, obs, action, step.Reward, step.Terminated, step.Truncated, step.Info);
                        // The previous observation has had both its callbacks; an env that hands out the same array again keeps it.
                        ReleaseExcept(previous, obs);
                        ReleaseExcept(step.Info, obs);
                    }

                    if (obs is not null) {
                        using RenderResult rendered = env.Render();
                        DisplayArr(game.Screen, FrameOf(rendered), game.VideoSize, transpose);
                    }

                    foreach (Event @event in context.Events.Get())
                        game.ProcessEvent(@event);
                    context.Display.Flip();
                    if (awaitPacing) {
                        try {
                            await clock.TickAsync(framerate, cancellationToken).ConfigureAwait(false);
                        } catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) {
                            // Canceled during the wait: this frame was shown, and the check below ends the game.
                        }
                    } else {
                        clock.Tick(framerate);
                    }

                    // Cancellation acts like Escape or the close button: the game ends after the current frame.
                    if (cancellationToken.IsCancellationRequested)
                        game.Running = false;
                }

                context.Display.Quit();
                Release(obs);
            } finally {
                PlayKeys.ReleaseActions(envMapping);
            }
        }

        /// <summary>
        ///     Runs a game loop on a background thread of its own and reports its end through a task: the thread blocks in the
        ///     loop's frame clock, as pygame's does, and nothing else runs on it.
        /// </summary>
        /// <param name="loop">Starts the loop; called on the new thread.</param>
        /// <returns>A task that completes when the loop does, faulted with its exception.</returns>
        private static Task RunOnOwnThread(Func<Task> loop) {
            // Continuations run elsewhere, so an awaiting caller never resumes on the game's thread.
            var completion = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            var thread = new Thread(() => {
                try {
                    loop().GetAwaiter().GetResult();
                    completion.SetResult();
                } catch (Exception exception) {
                    completion.SetException(exception);
                }
            }) {
                IsBackground = true,
                Name = "Gym.Utils.Play",
            };
            thread.Start();
            return completion.Task;
        }

        /// <summary>
        ///     Shows a frame on a surface (Gymnasium's <c>display_arr</c>): the frame, transposed or not, scaled to
        ///     <paramref name="videoSize"/> by nearest neighbour, centred on a black surface (cropped where it is larger).
        /// </summary>
        /// <param name="screen">The surface to draw on, opaque (the display's).</param>
        /// <param name="arr">The frame, uint8, <c>(H, W, 3)</c> (or <c>(H, W)</c>, read through pygame's 8-bit palette, not as gray).</param>
        /// <param name="videoSize">The size the frame is scaled to, <c>(width, height)</c>.</param>
        /// <param name="transpose">Whether the frame's first two axes are swapped first, as an <c>(H, W, 3)</c> frame needs to be shown upright; <see langword="null"/> is false.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="screen"/> is <see langword="null"/>.</exception>
        /// <exception cref="AssertionError">Thrown, with an empty message, when <paramref name="arr"/> is <see langword="null"/> or not uint8.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for a frame of another shape (pygame's <c>must be a valid 2d or 3d array</c>, newline-terminated), and for a negative size
        ///     (<c>Cannot scale to negative size</c>).
        /// </exception>
        /// <exception cref="AxisError">Thrown when a frame of fewer than two dimensions is transposed (NumPy's <c>swapaxes</c> error).</exception>
        /// <exception cref="NotSupportedException">Thrown when <paramref name="screen"/> has per-pixel alpha (not a display surface).</exception>
        /// <remarks>
        ///     The offsets are <c>(screen size - video size) / 2</c> truncated toward zero, as pygame's <c>blit</c> truncates a
        ///     float position: a frame wider than the window is shifted left by half the excess, rounded toward zero.
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py#L160">gymnasium/utils/play.py</see>.
        /// </remarks>
        public static void DisplayArr(Surface screen, NDArray arr, (int Width, int Height) videoSize, bool? transpose) {
            if (screen is null)
                throw new ArgumentNullException(nameof(screen));
            // assert isinstance(arr, np.ndarray) and arr.dtype == np.uint8
            if (arr is null || arr.typecode != NPTypeCode.Byte)
                throw new AssertionError("");

            Surface image;
            if (transpose == true) {
                using NDArray swapped = arr.swapaxes(0, 1);
                image = SurfArray.MakeSurface(swapped);
            } else {
                image = SurfArray.MakeSurface(arr);
            }

            Surface scaled = Transform.Scale(image, videoSize.Width, videoSize.Height);
            // Python's (a - b) / 2 is a float that blit truncates toward zero; C#'s int division truncates the same way.
            int widthOffset = (screen.Width - videoSize.Width) / 2;
            int heightOffset = (screen.Height - videoSize.Height) / 2;
            screen.Fill(new Color(0, 0, 0));
            screen.Blit(scaled, widthOffset, heightOffset);
        }

        /// <summary>
        ///     The frame of a render as <c>play</c> reads it: the last of a frame list, which must then be an array
        ///     (<c>assert isinstance(rendered, np.ndarray)</c>).
        /// </summary>
        /// <param name="rendered">The render; still owned by the caller.</param>
        /// <returns>The frame, owned by <paramref name="rendered"/>.</returns>
        /// <exception cref="IndexError">Thrown for an empty frame list (<c>list index out of range</c>).</exception>
        /// <exception cref="AssertionError">Thrown, with an empty message, when the render is not an array.</exception>
        internal static NDArray FrameOf(RenderResult rendered) {
            RenderResult last = rendered;
            if (rendered is RenderResult.Frames frames) {
                if (frames.Items.Count == 0)
                    throw new IndexError("list index out of range");
                last = frames.Items[frames.Items.Count - 1];
            }

            return last switch {
                RenderResult.Rgb rgb => rgb.Frame,
                RenderResult.Depth depth => depth.Frame,
                _ => throw new AssertionError(""),
            };
        }

        /// <summary>Releases what <c>play</c> drops: a reset result's observation and info, an array, or the arrays in an info.</summary>
        /// <param name="value">The value; anything else is left alone.</param>
        private static void Release(object? value) => ReleaseExcept(value, null);

        /// <summary>
        ///     Releases a dropped value's arrays, except <paramref name="keep"/>: an env (or a wrapper) may hand out the same
        ///     array again, and the one still in use must survive.
        /// </summary>
        /// <param name="value">An array, a <see cref="ResetResult{TObs}"/>, an info, or a tuple of them.</param>
        /// <param name="keep">The value still in use (the current observation), or <see langword="null"/>.</param>
        private static void ReleaseExcept(object? value, object? keep) {
            switch (value) {
                case NDArray array when !ReferenceEquals(array, keep):
                    array.Dispose();
                    break;
                case ResetResult<object?> reset:
                    ReleaseExcept(reset.Observation, keep);
                    ReleaseExcept(reset.Info, keep);
                    break;
                case IReadOnlyDictionary<string, object?> info:
                    foreach (object? item in info.Values)
                        ReleaseExcept(item, keep);
                    break;
                case object?[] items when items.GetType() == typeof(object[]):
                    foreach (object? item in items)
                        ReleaseExcept(item, keep);
                    break;
            }
        }
    }
}
