using System;
using System.Collections.Generic;
using Gym.Environments;
using Gym.Envs;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Rendering {
    /// <summary>
    ///     pygame's display (<c>pygame.display</c>): the one window a program draws into through its surface, opened by
    ///     <see cref="SetMode"/>, shown by <see cref="Flip"/>, closed by <see cref="Quit"/>. <see cref="Gym.Utils.Play"/>
    ///     plays an env in it. The window is a viewer from the viewer factory (DR-20), so it is an Avalonia or WinForms window,
    ///     or nothing at all under <see cref="NullEnvViewer"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     One per process, as pygame's: <see cref="SetMode"/> while a window is open replaces it. The human-mode windows of
    ///     envs are separate viewers and are not affected.
    ///     </para>
    ///     <para>
    ///     The surface is opaque 32-bit (<c>XRGB8888</c>), black when the mode is set, as pygame's display surface. When the
    ///     window is an <see cref="IInteractiveEnvViewer"/>, its input reaches <see cref="EventQueue"/> as pygame's events
    ///     when the queue is pumped: <see cref="EventType.KeyDown"/> and <see cref="EventType.KeyUp"/> with <c>key</c>, the
    ///     close button as <see cref="EventType.WindowClose"/> then <see cref="EventType.Quit"/>, and a resize of a resizable
    ///     display as <see cref="EventType.WindowResized"/> with <c>x</c> and <c>y</c>, after the surface has been replaced by
    ///     a black one of the new size (read it again through <see cref="GetSurface"/>). The events carry only those
    ///     attributes, and pygame's companion events of a resize (<c>VIDEORESIZE</c>, <c>WINDOWSIZECHANGED</c>) are not
    ///     posted. A posted <see cref="EventType.WindowResized"/> changes nothing, as in pygame.
    ///     </para>
    ///     <para>
    ///     Thread-safe; draw on the surface from the thread that pumps the queue, which is the one that sees it change.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/display.c">src_c/display.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static class Display {
        /// <summary>Guards the window and the surface.</summary>
        private static readonly object Gate = new();

        /// <summary>The open window, or <see langword="null"/>.</summary>
        private static IEnvViewer? _window;

        /// <summary>The display surface, or <see langword="null"/> before <see cref="SetMode"/> and after <see cref="Quit"/>.</summary>
        private static Surface? _surface;

        /// <summary>Whether the mode was set resizable (<c>pygame.RESIZABLE</c>), so the window's resizes reach the surface.</summary>
        private static bool _resizable;

        /// <summary>
        ///     <c>pygame.display.set_mode((width, height), flags)</c>: opens a window of that client size and returns its black
        ///     surface. A window already open is closed first (pygame would reuse and resize it; the viewers cannot be
        ///     resized from code).
        /// </summary>
        /// <param name="width">The client width in pixels, positive.</param>
        /// <param name="height">The client height in pixels, positive.</param>
        /// <param name="resizable">Whether the user's resizes reach the surface (<c>pygame.RESIZABLE</c>); otherwise they are ignored.</param>
        /// <param name="viewerFactory">The window's factory; <see langword="null"/> reads <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <param name="title">The window's title, or <see langword="null"/> for the viewer's default (pygame's is <c>pygame window</c>).</param>
        /// <returns>The display surface, also <see cref="GetSurface"/>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">
        ///     Thrown for a size that is not positive: pygame refuses a negative one and opens a desktop-sized window for 0,
        ///     which is not modelled.
        /// </exception>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">Thrown when no factory is given or set, where pygame would be missing.</exception>
        /// <remarks>
        ///     An interactive window starts reporting input here (<see cref="IInteractiveEnvViewer.BeginInput"/>), so its
        ///     close button posts <see cref="EventType.Quit"/> instead of closing it until <see cref="Quit"/>.
        /// </remarks>
        public static Surface SetMode(int width, int height, bool resizable = false, IEnvironmentViewerFactoryDelegate? viewerFactory = null, string? title = null) {
            if (width <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), width, "The display's width must be positive.");
            if (height <= 0)
                throw new ArgumentOutOfRangeException(nameof(height), height, "The display's height must be positive.");
            // The new window opens before the old one closes, so a missing or failing factory leaves the display as it was.
            // The factory may start a UI thread and complete asynchronously; set_mode returns with the window up.
            IEnvironmentViewerFactoryDelegate factory = Gymnasium.ResolveHumanViewerFactory(viewerFactory);
            IEnvViewer window = factory(width, height, title).GetAwaiter().GetResult();
            if (window is IInteractiveEnvViewer interactive)
                interactive.BeginInput();
            lock (Gate) {
                CloseWindow();
                _window = window;
                _surface = new Surface(width, height);
                _resizable = resizable;
                return _surface;
            }
        }

        /// <summary><c>pygame.display.get_surface()</c>: the display surface, or <see langword="null"/> when no mode is set.</summary>
        /// <returns>The surface; after a resize, a new one of the new size.</returns>
        public static Surface? GetSurface() {
            lock (Gate)
                return _surface;
        }

        /// <summary><c>pygame.display.get_init()</c> read as "a mode is set": whether <see cref="Flip"/> has a window to show.</summary>
        /// <returns><see langword="true"/> between <see cref="SetMode"/> and <see cref="Quit"/>.</returns>
        public static bool GetInit() {
            lock (Gate)
                return _surface is not null;
        }

        /// <summary><c>pygame.display.flip()</c>: shows the surface in the window, as it is now.</summary>
        /// <exception cref="SdlError">Thrown when no mode is set, with pygame's message: <c>Display mode not set</c>.</exception>
        /// <remarks>The viewer copies the frame before this returns; drawing on the surface afterwards does not change what is shown.</remarks>
        public static void Flip() {
            lock (Gate) {
                if (_surface is null || _window is null)
                    throw new SdlError("Display mode not set");
                using Image<Rgb24> frame = Image.LoadPixelData<Rgb24>(_surface.ToRgbBytes(), _surface.Width, _surface.Height);
                _window.Render(frame);
            }
        }

        /// <summary>
        ///     <c>pygame.quit()</c> for the display: closes the window, forgets the surface and drops every pending event, as
        ///     SDL's queue goes with its video subsystem. Safe to call when nothing is open.
        /// </summary>
        public static void Quit() {
            lock (Gate) {
                CloseWindow();
                _surface = null;
                _resizable = false;
            }

            EventQueue.Reset();
        }

        /// <summary>
        ///     Moves the window's buffered input into <see cref="EventQueue"/> as pygame's events (SDL's event pump), replacing
        ///     the surface for each resize first, as pygame's resize watch does before the event is read.
        /// </summary>
        internal static void PumpInput() {
            List<Event> events;
            lock (Gate) {
                if (_window is not IInteractiveEnvViewer interactive)
                    return;
                IReadOnlyList<ViewerInput> inputs = interactive.TakeInput();
                if (inputs.Count == 0)
                    return;
                events = new List<Event>(inputs.Count + 1);
                foreach (ViewerInput input in inputs) {
                    switch (input.Kind) {
                        case ViewerInputKind.KeyDown:
                            events.Add(new Event(EventType.KeyDown, new Info { ["key"] = input.Key }));
                            break;
                        case ViewerInputKind.KeyUp:
                            events.Add(new Event(EventType.KeyUp, new Info { ["key"] = input.Key }));
                            break;
                        case ViewerInputKind.Close:
                            events.Add(new Event(EventType.WindowClose));
                            events.Add(new Event(EventType.Quit));
                            break;
                        case ViewerInputKind.Resized when _resizable && input.Width > 0 && input.Height > 0:
                            // SDL hands out a new window surface after a resize; its content is undefined, black here.
                            _surface = new Surface(input.Width, input.Height);
                            events.Add(new Event(EventType.WindowResized, new Info { ["x"] = input.Width, ["y"] = input.Height }));
                            break;
                    }
                }
            }

            // Posted outside the display's lock: the queue takes its own, and a poster never waits on the display.
            foreach (Event @event in events)
                EventQueue.Post(@event);
        }

        /// <summary>Closes the open window, if any; the caller holds <see cref="Gate"/>.</summary>
        private static void CloseWindow() {
            if (_window is null)
                return;
            IEnvViewer window = _window;
            _window = null;
            window.CloseEnvironment();
        }
    }
}
