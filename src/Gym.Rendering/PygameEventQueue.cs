using System;
using System.Collections.Generic;

namespace Gym.Rendering {
    /// <summary>
    ///     A <see cref="PygameContext"/>'s event queue (<c>pygame.event</c>): filled by the context's display window (keys,
    ///     its close button, resizes) and by <see cref="Post"/>, drained by <see cref="Get"/>. <see cref="Gym.Utils.Play"/>
    ///     drains its context's queue once per frame, so posting events is how a program or a test plays an env without a
    ///     keyboard, as Gymnasium's own tests do with <c>pygame.event.post</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     One per context; the static <see cref="EventQueue"/> is <see cref="PygameContext.Default"/>'s, process-wide as SDL's
    ///     is, so a callback deep inside <c>play</c> can post to it without a handle. The footgun is pygame's: two loops
    ///     draining one queue steal each other's events, so two games at once need two contexts, and tests that share the
    ///     default queue must not run in parallel with each other.
    ///     </para>
    ///     <para>
    ///     Thread-safe: any thread may post (a viewer's UI thread does), and the window's input moves into the queue only when
    ///     <see cref="Get"/> or <see cref="Pump"/> runs (SDL's <c>SDL_PumpEvents</c>), on the caller's thread, so the display
    ///     surface changes size only there. The display's <see cref="PygameDisplay.Quit"/> empties it, as <c>pygame.quit()</c>
    ///     drops pending events. Unlike pygame, posting works before any display is set (there is no video subsystem to
    ///     initialize), and the queue has no capacity limit and no blocked types.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/event.c">src_c/event.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public sealed class PygameEventQueue {
        /// <summary>Guards <see cref="_pending"/>.</summary>
        private readonly object _gate = new();

        /// <summary>The events not yet taken, oldest first.</summary>
        private readonly List<Event> _pending = new();

        /// <summary>The queue of <paramref name="context"/>; only <see cref="PygameContext"/> builds one.</summary>
        /// <param name="context">The context whose display's window this queue pumps.</param>
        internal PygameEventQueue(PygameContext context) {
            Context = context;
        }

        /// <summary>The context this queue belongs to, whose <see cref="PygameContext.Display"/> feeds it.</summary>
        public PygameContext Context { get; }

        /// <summary>
        ///     <c>pygame.event.post(event)</c>: appends an event; the next <see cref="Get"/> returns it after the ones already
        ///     pending.
        /// </summary>
        /// <param name="event">The event; queued as is (not copied), so changing it before it is taken changes what is read.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="event"/> is <see langword="null"/>.</exception>
        public void Post(Event @event) {
            if (@event is null)
                throw new ArgumentNullException(nameof(@event));
            lock (_gate)
                _pending.Add(@event);
        }

        /// <summary>
        ///     <c>pygame.event.get()</c>: pumps the context's window input (<see cref="Pump"/>), then takes every pending event,
        ///     oldest first.
        /// </summary>
        /// <returns>The events, empty when there are none; the caller owns the list.</returns>
        public IReadOnlyList<Event> Get() {
            Pump();
            lock (_gate) {
                var taken = new List<Event>(_pending);
                _pending.Clear();
                return taken;
            }
        }

        /// <summary>
        ///     <c>pygame.event.pump()</c>: moves the context's window input into the queue as pygame's events, resizing the
        ///     display surface for a resize, without taking anything out. A no-op without an interactive window.
        /// </summary>
        public void Pump() {
            // The display translates first, outside the queue's lock: it takes its own lock and posts through Post.
            Context.Display.PumpInput();
        }

        /// <summary><c>pygame.event.clear()</c>: drops every pending event, the window's buffered input included.</summary>
        public void Clear() {
            Pump();
            lock (_gate)
                _pending.Clear();
        }

        /// <summary>The number of pending events, without pumping; for tests and diagnostics.</summary>
        public int Count {
            get {
                lock (_gate)
                    return _pending.Count;
            }
        }

        /// <summary>Drops every pending event without pumping the window: what the display's <see cref="PygameDisplay.Quit"/> does to the queue.</summary>
        internal void Reset() {
            lock (_gate)
                _pending.Clear();
        }
    }
}
