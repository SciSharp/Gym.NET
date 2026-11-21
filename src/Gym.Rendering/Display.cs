using System;
using Gym.Environments;

namespace Gym.Rendering {
    /// <summary>
    ///     pygame's display (<c>pygame.display</c>) in pygame's spelling: the display of <see cref="PygameContext.Default"/>,
    ///     the process-wide context, so a line-for-line port keeps <c>pygame.display.set_mode(...)</c> as
    ///     <c>Display.SetMode(...)</c>. Every member forwards to <c>PygameContext.Default.Display</c>
    ///     (<see cref="PygameDisplay"/>, which documents the behaviour in full).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     One window per process through these statics, as pygame's: <see cref="SetMode"/> while a window is open replaces
    ///     it. A game that should run beside another takes a <see cref="PygameContext"/> of its own (DR-41); the human-mode
    ///     windows of envs are separate viewers and are not affected either way.
    ///     </para>
    ///     <para>Thread-safe; draw on the surface from the thread that pumps the queue, which is the one that sees it change.</para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/display.c">src_c/display.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static class Display {
        /// <summary>
        ///     <c>pygame.display.set_mode((width, height), flags)</c> on the default context: opens a window of that client size
        ///     and returns its black surface, closing a window already open (see <see cref="PygameDisplay.SetMode"/>).
        /// </summary>
        /// <param name="width">The client width in pixels, positive.</param>
        /// <param name="height">The client height in pixels, positive.</param>
        /// <param name="resizable">Whether the user's resizes reach the surface (<c>pygame.RESIZABLE</c>); otherwise they are ignored.</param>
        /// <param name="viewerFactory">The window's factory; <see langword="null"/> reads <c>Gymnasium.HumanViewerFactory</c> (the default context has no factory of its own).</param>
        /// <param name="title">The window's title, or <see langword="null"/> for the viewer's default (pygame's is <c>pygame window</c>).</param>
        /// <returns>The display surface, also <see cref="GetSurface"/>.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a size that is not positive (0 would be a desktop-sized window in pygame, not modelled).</exception>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">Thrown when no factory is given or set, where pygame would be missing.</exception>
        /// <remarks>An interactive window reports input from here on, so its close button posts <see cref="EventType.Quit"/> instead of closing it until <see cref="Quit"/>.</remarks>
        public static Surface SetMode(int width, int height, bool resizable = false, EnvViewerFactory? viewerFactory = null, string? title = null) =>
            PygameContext.Default.Display.SetMode(width, height, resizable, viewerFactory, title);

        /// <summary><c>pygame.display.get_surface()</c> on the default context: the display surface, or <see langword="null"/> when no mode is set.</summary>
        /// <returns>The surface; after a resize, a new one of the new size.</returns>
        public static Surface? GetSurface() => PygameContext.Default.Display.GetSurface();

        /// <summary><c>pygame.display.get_init()</c> on the default context, read as "a mode is set": whether <see cref="Flip"/> has a window to show.</summary>
        /// <returns><see langword="true"/> between <see cref="SetMode"/> and <see cref="Quit"/>.</returns>
        public static bool GetInit() => PygameContext.Default.Display.GetInit();

        /// <summary><c>pygame.display.flip()</c> on the default context: shows the surface in the window, as it is now (the viewer copies it).</summary>
        /// <exception cref="SdlError">Thrown when no mode is set, with pygame's message: <c>Display mode not set</c>.</exception>
        public static void Flip() => PygameContext.Default.Display.Flip();

        /// <summary>
        ///     <c>pygame.quit()</c> for the default context's display: closes the window, forgets the surface and drops every
        ///     pending event of <see cref="EventQueue"/>. Safe to call when nothing is open.
        /// </summary>
        public static void Quit() => PygameContext.Default.Display.Quit();
    }
}
