using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Rendering;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     An env being played from the keyboard, as Gymnasium's <c>PlayableGame</c>: it opens its context's display at the
    ///     size of the env's frame (times the zoom), knows which keys matter, and tracks which of them are held as
    ///     <see cref="ProcessEvent"/> reads key events. <see cref="Play.Run"/> builds one and drives it; build one directly
    ///     to run a loop of your own.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Constructing it opens the display of its <see cref="Context"/> (pygame's <c>set_mode</c>), replacing any window
    ///     open there; call the context's <see cref="PygameDisplay.Quit"/> (<see cref="Display.Quit"/> for the default
    ///     context) when done, as Gymnasium's tests call <c>pygame.quit()</c>. Not thread-safe: process events from the
    ///     thread that plays.
    ///     </para>
    ///     <para>
    ///     The key sets hold keys with Python's equality and hashing (an event's <c>97.0</c> matches a relevant <c>97</c>);
    ///     <see cref="PressedKeys"/> holds the event's own key objects, as Python's set does.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py#L54">gymnasium/utils/play.py</see>.</para>
    /// </remarks>
    public class PlayableGame {
        /// <summary>
        ///     Opens the display for <paramref name="env"/>: checks the render mode and the zoom, collects the relevant keys,
        ///     renders once to size the window, then opens it.
        /// </summary>
        /// <param name="env">The env; its render mode must be <c>rgb_array</c> or <c>rgb_array_list</c>, and it must render now (after a reset).</param>
        /// <param name="keysToAction">
        ///     A dictionary whose keys are key combinations, tuples of key codes (<c>(KeyCode.A,)</c>, <c>(275, 276)</c>), or
        ///     <see langword="null"/> to ask the env (<c>get_keys_to_action</c>, found through any wrapper). Only the keys'
        ///     elements are read here; <see cref="Play.Run"/> converts strings and single codes into such tuples first.
        /// </param>
        /// <param name="zoom">Scales the window; <see langword="null"/> keeps the frame's size. Must be positive.</param>
        /// <param name="viewerFactory">
        ///     The window's factory; <see langword="null"/> takes the context's <see cref="PygameContext.ViewerFactory"/>, then
        ///     <c>Gymnasium.HumanViewerFactory</c>.
        /// </param>
        /// <param name="context">
        ///     The display and event queue the game uses (DR-41); <see langword="null"/> is <see cref="PygameContext.Default"/>,
        ///     the static <see cref="Display"/> and <see cref="EventQueue"/>.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for another render mode (<c>PlayableGame wrapper works only with rgb_array and rgb_array_list render
        ///     modes, but your environment render_mode = None.</c>) or a zoom that is not positive (<c>Zoom must be a positive
        ///     float, got 0.0</c>), with Gymnasium's messages; also for a NaN zoom, when the size is computed.
        /// </exception>
        /// <exception cref="MissingKeysToAction">Thrown when no mapping is given and the env has none.</exception>
        /// <exception cref="AssertionError">
        ///     Thrown, with an empty message as Python's bare <c>assert</c>, when the env has no mapping and no spec either, or
        ///     when its render is not an array.
        /// </exception>
        /// <exception cref="TypeError">Thrown when the mapping is not a dictionary, or a key is not iterable (an int).</exception>
        /// <exception cref="IndexError">Thrown for a one-dimensional render (<c>tuple index out of range</c>) or an empty frame list (<c>list index out of range</c>).</exception>
        /// <exception cref="OverflowError">Thrown, as Python's <c>OverflowError</c>, for an infinite zoom.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when the window's size comes out zero, which pygame would turn into a desktop-sized window (not modelled).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown when no viewer factory is given or set.</exception>
        public PlayableGame(IEnv env, object? keysToAction = null, double? zoom = null, EnvViewerFactory? viewerFactory = null,
            PygameContext? context = null) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            if (env.RenderMode is not ("rgb_array" or "rgb_array_list"))
                throw new ValueError($"PlayableGame wrapper works only with rgb_array and rgb_array_list render modes, but your environment render_mode = {PyBuiltins.Str(env.RenderMode)}.");
            // A NaN zoom passes this test, as in Python, and fails when the size is computed.
            if (zoom is { } factor && factor <= 0)
                throw new ValueError($"Zoom must be a positive float, got {PyBuiltins.Str(factor)}");

            Env = env;
            Context = context ?? PygameContext.Default;
            RelevantKeys = GetRelevantKeys(keysToAction);
            // Kept apart from VideoSize so that resizes always scale from the frame's size and never drift.
            OriginalVideoSize = GetVideoSize(zoom);
            VideoSize = OriginalVideoSize;
            Context.Display.SetMode(VideoSize.Width, VideoSize.Height, resizable: true, viewerFactory, env.Spec?.Id ?? env.Unwrapped.GetType().Name);
            PressedKeys = new HashSet<object?>(PyKeyComparer.Instance);
            Running = true;
        }

        /// <summary>The env being played (<c>env</c>).</summary>
        public IEnv Env { get; }

        /// <summary>
        ///     The display and event queue this game uses (DR-41): its window is the context's display, and
        ///     <see cref="Play.Run"/> reads the context's queue. Not one of Gymnasium's attributes.
        /// </summary>
        public PygameContext Context { get; }

        /// <summary>
        ///     Every key that appears in a combination of the mapping (<c>relevant_keys</c>): only these are tracked as
        ///     pressed. With Python's equality; modifying it changes which keys are tracked.
        /// </summary>
        public ISet<object?> RelevantKeys { get; }

        /// <summary>The frame's size times the zoom, <c>(width, height)</c> (<c>original_video_size</c>); every resize scales from it.</summary>
        public (int Width, int Height) OriginalVideoSize { get; }

        /// <summary>
        ///     The size frames are scaled to (<c>video_size</c>): <see cref="OriginalVideoSize"/> until the window is resized,
        ///     then the largest size of the same aspect that fits the window.
        /// </summary>
        public (int Width, int Height) VideoSize { get; set; }

        /// <summary>
        ///     The display surface (<c>screen</c>), read from the context's <see cref="PygameDisplay.GetSurface"/> each time,
        ///     because a resize of the window replaces it (pygame swaps the surface behind the same object).
        /// </summary>
        /// <exception cref="SdlError">Thrown after the context's display quit, with pygame's message: <c>display Surface quit</c>.</exception>
        public Surface Screen => Context.Display.GetSurface() ?? throw new SdlError("display Surface quit");

        /// <summary>The relevant keys held down now (<c>pressed_keys</c>), as the events gave them.</summary>
        public ISet<object?> PressedKeys { get; }

        /// <summary>Whether the game goes on (<c>running</c>); Escape (when not mapped), <see cref="EventType.Quit"/>, or you, set it false.</summary>
        public bool Running { get; set; }

        /// <summary>
        ///     Applies one event (<c>process_event</c>): a relevant key's press or release updates <see cref="PressedKeys"/>,
        ///     Escape (when it is not relevant) and <see cref="EventType.Quit"/> stop the game, and
        ///     <see cref="EventType.WindowResized"/> refits <see cref="VideoSize"/>; anything else is ignored.
        /// </summary>
        /// <param name="event">The event.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="event"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">Thrown when a key event has no <c>key</c>, or a resize no <c>x</c> or <c>y</c> (<c>'pygame.event.Event' object has no attribute 'key'</c>).</exception>
        /// <exception cref="TypeError">Thrown for a key Python cannot hash (<c>unhashable type: 'list'</c>), or a resize size that is no number.</exception>
        /// <exception cref="ZeroDivisionError">Thrown, as Python's <c>ZeroDivisionError</c>, for a resize of a game whose frame is zero pixels wide or high.</exception>
        /// <exception cref="ValueError">Thrown for a NaN resize size (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="OverflowError">Thrown, as Python's <c>OverflowError</c>, for an infinite resize size.</exception>
        /// <exception cref="OverflowException">Thrown for a resize size beyond an int (Gym.NET's limit; Python's ints are unbounded).</exception>
        /// <remarks>
        ///     A held key that repeats (several presses, one release) is released by its one release; a release of a key never
        ///     pressed does nothing. A resize keeps the frame's aspect: the scale is the smaller of the two axes' ratios, and
        ///     each side is rounded half to even, as Python's <c>round</c>.
        /// </remarks>
        public void ProcessEvent(Event @event) {
            if (@event is null)
                throw new ArgumentNullException(nameof(@event));
            switch (@event.Type) {
                case EventType.KeyDown: {
                    object? key = PyAttributes.GetAttr(@event, "key");
                    if (RelevantKeys.Contains(key))
                        PressedKeys.Add(key);
                    else if (PyEquality.AreEqual(key, KeyCode.Escape))
                        Running = false;
                    break;
                }
                case EventType.KeyUp: {
                    object? key = PyAttributes.GetAttr(@event, "key");
                    if (RelevantKeys.Contains(key))
                        PressedKeys.Remove(key);
                    break;
                }
                case EventType.Quit:
                    Running = false;
                    break;
                case EventType.WindowResized: {
                    // Python evaluates event.x / width before event.y / height, so a missing x is reported first.
                    double scaleWidth = TrueDivide(PyAttributes.GetAttr(@event, "x"), OriginalVideoSize.Width);
                    double scaleHeight = TrueDivide(PyAttributes.GetAttr(@event, "y"), OriginalVideoSize.Height);
                    // min(scale_height, scale_width) keeps its first argument unless the second is smaller (NaN included).
                    double scale = scaleWidth < scaleHeight ? scaleWidth : scaleHeight;
                    VideoSize = (RoundHalfEven(scale * OriginalVideoSize.Width), RoundHalfEven(scale * OriginalVideoSize.Height));
                    break;
                }
            }
        }

        /// <summary>
        ///     <c>_get_relevant_keys</c>: the mapping's (or the env's mapping's) key elements, as a set with Python's equality.
        /// </summary>
        /// <param name="keysToAction">The mapping, or <see langword="null"/> to ask the env.</param>
        /// <returns>The set.</returns>
        /// <exception cref="MissingKeysToAction">Thrown when no mapping is given and the env has none.</exception>
        /// <exception cref="AssertionError">Thrown, with an empty message, when the env has neither a mapping nor a spec.</exception>
        /// <exception cref="TypeError">Thrown when the mapping is no dictionary, or a key is not iterable.</exception>
        private ISet<object?> GetRelevantKeys(object? keysToAction) {
            // A mapping taken from the env is a new one this method owns (DR-45): its actions are released once the keys are
            // read, on a refusal too.
            object? envMapping = null;
            if (keysToAction is null) {
                if (Env.HasWrapperAttr("get_keys_to_action")) {
                    keysToAction = envMapping = PlayKeys.Call(Env.GetWrapperAttr("get_keys_to_action"));
                } else {
                    // assert self.env.spec is not None: a bare assertion, so its message is empty.
                    if (Env.Spec is null)
                        throw new AssertionError("");
                    throw new MissingKeysToAction($"{Env.Spec.Id} does not have explicit key to action mapping, please specify one manually, `play(env, keys_to_action=...)`");
                }
            }

            try {
                List<KeyValuePair<object?, object?>> entries = PlayKeys.DictEntries(keysToAction)
                                                               ?? throw new TypeError($"keys_to_action must be a dictionary, got {PyBuiltins.TypeRepr(keysToAction)}");
                // set(sum((list(k) for k in keys_to_action.keys()), [])): every element of every key.
                var keys = new HashSet<object?>(PyKeyComparer.Instance);
                foreach (KeyValuePair<object?, object?> entry in entries)
                    keys.UnionWith(PlayKeys.Iterate(entry.Key));
                return keys;
            } finally {
                PlayKeys.ReleaseActions(envMapping);
            }
        }

        /// <summary><c>_get_video_size</c>: the size of the env's frame now, <c>(width, height)</c>, times the zoom, truncated.</summary>
        /// <param name="zoom">The zoom, or <see langword="null"/>.</param>
        /// <returns>The size.</returns>
        /// <exception cref="AssertionError">Thrown, with an empty message, when the render is not an array.</exception>
        /// <exception cref="IndexError">Thrown for a one-dimensional frame or an empty frame list.</exception>
        /// <exception cref="ValueError">Thrown for a NaN zoom.</exception>
        /// <exception cref="OverflowError">Thrown for an infinite zoom, as Python's <c>int(inf)</c> raises.</exception>
        /// <exception cref="OverflowException">Thrown for a size beyond an int (Gym.NET's limit).</exception>
        private (int Width, int Height) GetVideoSize(double? zoom) {
            using RenderResult rendered = Env.Render();
            NDArray frame = Play.FrameOf(rendered);
            if (frame.ndim < 2)
                throw new IndexError("tuple index out of range");
            int width = checked((int)frame.shape[1]), height = checked((int)frame.shape[0]);
            if (zoom is not { } factor)
                return (width, height);
            return (checked((int)PyBuiltins.Int(width * factor)), checked((int)PyBuiltins.Int(height * factor)));
        }

        /// <summary>Python's <c>value / divisor</c> for a resize size and a frame side: true division, with Python's errors.</summary>
        /// <param name="value">The event's size, a number.</param>
        /// <param name="divisor">The frame's side.</param>
        /// <returns>The quotient.</returns>
        /// <exception cref="ZeroDivisionError">Thrown for a zero side: <c>division by zero</c> for an int size, <c>float division by zero</c> for a float.</exception>
        /// <exception cref="TypeError">Thrown when the size is no number: <c>unsupported operand type(s) for /: 'str' and 'int'</c>.</exception>
        private static double TrueDivide(object? value, int divisor) {
            if (value is NDArray { ndim: 0 } scalar && PlayKeys.Number(scalar) is { } numpyNumber)
                // A NumPy scalar divides as NumPy does: by zero it gives inf or NaN with a warning, not an error.
                return numpyNumber.AsDouble / divisor;
            if (PlayKeys.Number(value) is not { } number || value is NDArray)
                throw new TypeError($"unsupported operand type(s) for /: '{PyBuiltins.TypeName(value)}' and 'int'");
            if (divisor == 0)
                throw new ZeroDivisionError(PlayKeys.IsInt(value) ? "division by zero" : "float division by zero");
            // An int below 2^53 and a side are exact doubles, so one IEEE division rounds as Python's int / int does.
            return number.AsDouble / divisor;
        }

        /// <summary>Python's <c>round(x)</c> of a float: half to even, as an int.</summary>
        /// <param name="value">The float.</param>
        /// <returns>The rounded value.</returns>
        /// <exception cref="ValueError">Thrown for NaN.</exception>
        /// <exception cref="OverflowError">Thrown for an infinity, as Python's <c>int(inf)</c> raises.</exception>
        /// <exception cref="OverflowException">Thrown for a result beyond an int (Gym.NET's limit).</exception>
        private static int RoundHalfEven(double value) => checked((int)PyBuiltins.Int(Math.Round(value, MidpointRounding.ToEven)));
    }

    /// <summary>
    ///     Raised when <see cref="Play.Run"/> or <see cref="PlayableGame"/> is given no key mapping and the env has none
    ///     (Gymnasium's <c>MissingKeysToAction</c>): pass <c>keysToAction</c>, or give the env a <c>get_keys_to_action</c>
    ///     attribute.
    /// </summary>
    /// <remarks>
    ///     Derived from <see cref="Exception"/>, as Gymnasium's derives from Python's <c>Exception</c> and not from
    ///     <c>gymnasium.error.Error</c>.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py#L50">gymnasium/utils/play.py</see>.
    /// </remarks>
    public class MissingKeysToAction : Exception {
        /// <summary>The error with a message naming the env.</summary>
        /// <param name="message">What is missing.</param>
        public MissingKeysToAction(string message) : base(message) { }
    }
}
