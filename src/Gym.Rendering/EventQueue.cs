using System;
using System.Collections.Generic;

namespace Gym.Rendering {
    /// <summary>
    ///     pygame's event queue (<c>pygame.event</c>) in pygame's spelling: the queue of <see cref="PygameContext.Default"/>,
    ///     the process-wide context, filled by <see cref="Display"/>'s window and by <see cref="Post"/>, drained by
    ///     <see cref="Get"/>. Every member forwards to <c>PygameContext.Default.Events</c> (<see cref="PygameEventQueue"/>,
    ///     which documents the behaviour in full).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Process-wide on purpose, as SDL's is: a callback deep inside <c>play</c> can post to it without a handle. The
    ///     footgun is the same as pygame's: two loops draining it in one process steal each other's events, and tests that
    ///     use it must not run in parallel with each other. A game that runs beside another takes a
    ///     <see cref="PygameContext"/> of its own (DR-41) and is posted to through its <see cref="PygameContext.Events"/>.
    ///     </para>
    ///     <para>
    ///     Thread-safe: any thread may post, and the window's input moves into the queue only when <see cref="Get"/> or
    ///     <see cref="Pump"/> runs, on the caller's thread. <see cref="Display.Quit"/> empties it. Posting works before any
    ///     display is set, and the queue has no capacity limit and no blocked types.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/event.c">src_c/event.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static class EventQueue {
        /// <summary><c>pygame.event.post(event)</c> on the default context: appends an event after the ones already pending.</summary>
        /// <param name="event">The event; queued as is (not copied), so changing it before it is taken changes what is read.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="event"/> is <see langword="null"/>.</exception>
        public static void Post(Event @event) => PygameContext.Default.Events.Post(@event);

        /// <summary><c>pygame.event.get()</c> on the default context: pumps the window's input, then takes every pending event, oldest first.</summary>
        /// <returns>The events, empty when there are none; the caller owns the list.</returns>
        public static IReadOnlyList<Event> Get() => PygameContext.Default.Events.Get();

        /// <summary>
        ///     <c>pygame.event.pump()</c> on the default context: moves the window's buffered input into the queue, resizing the
        ///     display surface for a resize, without taking anything out. A no-op without an interactive window.
        /// </summary>
        public static void Pump() => PygameContext.Default.Events.Pump();

        /// <summary><c>pygame.event.clear()</c> on the default context: drops every pending event, the window's buffered input included.</summary>
        public static void Clear() => PygameContext.Default.Events.Clear();

        /// <summary>The number of pending events of the default context, without pumping; for tests and diagnostics.</summary>
        public static int Count => PygameContext.Default.Events.Count;
    }
}
