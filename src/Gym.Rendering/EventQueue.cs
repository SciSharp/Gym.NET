using System;
using System.Collections.Generic;

namespace Gym.Rendering {
    /// <summary>
    ///     pygame's event queue (<c>pygame.event</c>): one per process, filled by the open <see cref="Display"/>'s window
    ///     (keys, its close button, resizes) and by <see cref="Post"/>, drained by <see cref="Get"/>. <see cref="Gym.Utils.Play"/>
    ///     drains it once per frame, so posting events is how a program or a test plays an env without a keyboard, as
    ///     Gymnasium's own tests do with <c>pygame.event.post</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Process-wide on purpose, as SDL's is: a callback deep inside <c>play</c> can post to it without a handle. The
    ///     footgun is the same as pygame's: two loops draining it in one process steal each other's events, and tests that
    ///     use it must not run in parallel with each other.
    ///     </para>
    ///     <para>
    ///     Thread-safe: any thread may post (a viewer's UI thread does), and the window's input moves into the queue only when
    ///     <see cref="Get"/> or <see cref="Pump"/> runs (SDL's <c>SDL_PumpEvents</c>), on the caller's thread, so the display
    ///     surface changes size only there. <see cref="Display.Quit"/> empties it, as <c>pygame.quit()</c> drops pending
    ///     events. Unlike pygame, posting works before any display is set (there is no video subsystem to initialize), and
    ///     the queue has no capacity limit and no blocked types.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/event.c">src_c/event.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static class EventQueue {
        /// <summary>Guards <see cref="Pending"/>.</summary>
        private static readonly object Gate = new();

        /// <summary>The events not yet taken, oldest first.</summary>
        private static readonly List<Event> Pending = new();

        /// <summary>
        ///     <c>pygame.event.post(event)</c>: appends an event; the next <see cref="Get"/> returns it after the ones already
        ///     pending.
        /// </summary>
        /// <param name="event">The event; queued as is (not copied), so changing it before it is taken changes what is read.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="event"/> is <see langword="null"/>.</exception>
        public static void Post(Event @event) {
            if (@event is null)
                throw new ArgumentNullException(nameof(@event));
            lock (Gate)
                Pending.Add(@event);
        }

        /// <summary>
        ///     <c>pygame.event.get()</c>: pumps the open window's input (<see cref="Pump"/>), then takes every pending event,
        ///     oldest first.
        /// </summary>
        /// <returns>The events, empty when there are none; the caller owns the list.</returns>
        public static IReadOnlyList<Event> Get() {
            Pump();
            lock (Gate) {
                var taken = new List<Event>(Pending);
                Pending.Clear();
                return taken;
            }
        }

        /// <summary>
        ///     <c>pygame.event.pump()</c>: moves the open window's buffered input into the queue as pygame's events, resizing
        ///     the display surface for a resize, without taking anything out. A no-op without an interactive window.
        /// </summary>
        public static void Pump() {
            // The display translates first, outside the queue's lock: it takes its own lock and posts through Post.
            Display.PumpInput();
        }

        /// <summary><c>pygame.event.clear()</c>: drops every pending event, the window's buffered input included.</summary>
        public static void Clear() {
            Pump();
            lock (Gate)
                Pending.Clear();
        }

        /// <summary>The number of pending events, without pumping; for tests and diagnostics.</summary>
        public static int Count {
            get {
                lock (Gate)
                    return Pending.Count;
            }
        }

        /// <summary>Drops every pending event without pumping the window: what <see cref="Display.Quit"/> does to the queue.</summary>
        internal static void Reset() {
            lock (Gate)
                Pending.Clear();
        }
    }
}
