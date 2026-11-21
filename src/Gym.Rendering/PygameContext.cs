using System;
using Gym.Environments;
using Gym.Exceptions;

namespace Gym.Rendering {
    /// <summary>
    ///     One pygame "process" (DR-41): a display (<see cref="Display"/>) and the event queue its window feeds
    ///     (<see cref="Events"/>), so several games can run side by side, each in a window of its own and each reading only
    ///     its own events. <see cref="Default"/> is the process-wide context that pygame's spelling reaches: the static
    ///     <see cref="Rendering.Display"/> and <see cref="EventQueue"/> forward to it, so line-for-line ports keep
    ///     <c>pygame.display.set_mode(...)</c> as <c>Display.SetMode(...)</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Give <see cref="Gym.Utils.Play.Run"/>, <see cref="Gym.Utils.Play.RunAsync"/> or <see cref="Gym.Utils.PlayableGame"/>
    ///     a context of its own to play two games at once; with <see langword="null"/> they use <see cref="Default"/>, and two
    ///     games there steal each other's events and windows, as two loops in one pygame process would.
    ///     </para>
    ///     <para>
    ///     Disposing a context is <c>pygame.quit()</c> for it: the window closes and pending events are dropped. The context
    ///     stays usable afterwards (a new <see cref="PygameDisplay.SetMode"/> opens a window again), so disposing
    ///     <see cref="Default"/> is safe. Thread-safe, as its display and queue are.
    ///     </para>
    /// </remarks>
    public sealed class PygameContext : IDisposable {
        /// <summary>
        ///     A context whose display opens its windows with <paramref name="viewerFactory"/> (DR-20), or with
        ///     <c>Gymnasium.HumanViewerFactory</c> when that is <see langword="null"/>; no window opens until
        ///     <see cref="PygameDisplay.SetMode"/>.
        /// </summary>
        /// <param name="viewerFactory">
        ///     The factory of this context's windows (<c>AvaloniaEnvViewer.Factory</c>, <c>NullEnvViewer.Factory</c> headless),
        ///     or <see langword="null"/> to read <c>Gymnasium.HumanViewerFactory</c> each time a window opens. A factory passed
        ///     to <see cref="PygameDisplay.SetMode"/> itself still wins.
        /// </param>
        public PygameContext(EnvViewerFactory? viewerFactory = null) {
            ViewerFactory = viewerFactory;
            // The two halves refer to each other: a quit empties the queue, and a pump moves the window's input into it.
            Events = new PygameEventQueue(this);
            Display = new PygameDisplay(this);
        }

        /// <summary>
        ///     The process-wide context, the one pygame has: the static <see cref="Rendering.Display"/> and
        ///     <see cref="EventQueue"/> are its display and queue, and every <c>play</c> call without a context uses it.
        /// </summary>
        public static PygameContext Default { get; } = new();

        /// <summary>
        ///     The factory this context opens windows with, or <see langword="null"/> to read <c>Gymnasium.HumanViewerFactory</c>
        ///     at each <see cref="PygameDisplay.SetMode"/>.
        /// </summary>
        public EnvViewerFactory? ViewerFactory { get; }

        /// <summary>The context's display: its one window and surface (<c>pygame.display</c>).</summary>
        public PygameDisplay Display { get; }

        /// <summary>The context's event queue (<c>pygame.event</c>), fed by <see cref="Display"/>'s window and by posts.</summary>
        public PygameEventQueue Events { get; }

        /// <summary>
        ///     <c>pygame.quit()</c> for this context: closes the window, forgets the surface and drops every pending event.
        ///     Idempotent, and the context stays usable.
        /// </summary>
        public void Dispose() => Display.Quit();

        /// <summary>
        ///     The factory a window of this context opens with: the one given to the call, else this context's, else
        ///     <c>Gymnasium.HumanViewerFactory</c>.
        /// </summary>
        /// <param name="given">The factory passed to the call, or <see langword="null"/>.</param>
        /// <returns>The factory.</returns>
        /// <exception cref="DependencyNotInstalled">Thrown when none of the three is set, where pygame would be missing.</exception>
        internal EnvViewerFactory ResolveViewerFactory(EnvViewerFactory? given) =>
            Gymnasium.ResolveHumanViewerFactory(given ?? ViewerFactory);

        /// <summary>
        ///     Whether a window of this context would be a real one: the resolved factory is set and is not
        ///     <see cref="NullEnvViewer.Factory"/>. <see cref="Gym.Utils.Play.RunAsync"/> gives such a game a thread of its own.
        /// </summary>
        /// <param name="given">The factory passed to the call, or <see langword="null"/>.</param>
        /// <returns><see langword="true"/> for a viewer that shows a window; <see langword="false"/> headless or when no factory is set.</returns>
        internal bool OpensRealWindow(EnvViewerFactory? given) {
            EnvViewerFactory? factory = given ?? ViewerFactory ?? Gymnasium.HumanViewerFactory;
            // Reference equality: NullEnvViewer.Factory is one delegate instance, the one its docs tell callers to pass.
            return factory is not null && !ReferenceEquals(factory, NullEnvViewer.Factory);
        }
    }
}
