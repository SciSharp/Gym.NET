using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace Gym.Rendering {
    /// <summary>
    ///     Paces a render loop the way <c>pygame.time.Clock</c> does. Human-mode rendering calls <see cref="Tick"/> with
    ///     <c>metadata["render_fps"]</c> once per frame, which sleeps just long enough to hold that frame rate, so a
    ///     human-mode episode plays at the env's intended speed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Time is wall-clock milliseconds since the clock was created (SDL's <c>SDL_GetTicks</c> counts from SDL's
    ///     initialization); only differences are observable, and timing takes no part in parity. The arithmetic is pygame's,
    ///     including its float32 frame-time computation and whole-millisecond sleeps, so the pacing matches.
    ///     </para>
    ///     <para>
    ///     Not thread-safe: tick a clock from the thread that renders, or, with <see cref="TickAsync"/>, from one loop that
    ///     awaits each tick before the next (its continuation may resume on another thread).
    ///     </para>
    ///     <para>
    ///     It lives in the Gym assembly, beside the viewers it paces (namespace <c>Gym.Rendering</c> unchanged), so the
    ///     <c>HumanRendering</c> wrapper paces its frames with it as the built-in envs do.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/time.c#L691">src_c/time.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public sealed class Clock {
        /// <summary>The monotonic time base, SDL's millisecond tick counter.</summary>
        private readonly Stopwatch _ticks = Stopwatch.StartNew();

        /// <summary>pygame's <c>last_tick</c>: the tick of the previous <see cref="Tick"/>.</summary>
        private long _lastTick;

        /// <summary>pygame's <c>rawpassed</c>: the time between ticks without the pacing delay.</summary>
        private long _rawPassed;

        /// <summary>pygame's <c>timepassed</c>: the time between the last two ticks, delay included.</summary>
        private long _timePassed;

        /// <summary>pygame's <c>fps_count</c>: ticks since the frame rate was last computed.</summary>
        private int _fpsCount;

        /// <summary>pygame's <c>fps_tick</c>: the tick the current frame-rate window started at, 0 before the first.</summary>
        private long _fpsTick;

        /// <summary>pygame's <c>fps</c>: the frame rate averaged over the last ten ticks.</summary>
        private float _fps;

        /// <summary>Creates a clock whose first <see cref="Tick"/> measures from now (pygame's <c>Clock()</c> records the current tick).</summary>
        public Clock() {
            _lastTick = Now();
        }

        /// <summary>
        ///     Marks a frame; with a positive <paramref name="framerate"/>, first sleeps until <c>1000 / framerate</c> ms have
        ///     passed since the previous tick, so the loop runs no faster than that rate.
        /// </summary>
        /// <param name="framerate">The frame rate to hold, frames per second; 0 (the default) does not sleep.</param>
        /// <returns>The milliseconds since the previous tick, the sleep included.</returns>
        /// <remarks>
        ///     pygame reads the rate as a C <c>float</c> and computes <c>(int)((1.0f / framerate) * 1000.0f)</c>, so the frame
        ///     time is whole milliseconds (20 ms at 50 fps). A negative or infinite rate gives a delay clamped at zero, as in
        ///     pygame, and nothing sleeps; so does NaN, for which pygame's C conversion is undefined.
        ///     Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/time.c#L535">src_c/time.c</see>.
        /// </remarks>
        public long Tick(double framerate = 0) {
            long delay = Delay(framerate, out float rate);
            if (delay > 0)
                Thread.Sleep(TimeSpan.FromMilliseconds(delay));
            return Mark(rate);
        }

        /// <summary>
        ///     <see cref="Tick"/> with an awaited wait instead of a blocking sleep (DR-44): the same arithmetic and the same
        ///     whole-millisecond delay, during which no thread is held, so a game loop on the thread pool paces itself as
        ///     pygame's clock does.
        /// </summary>
        /// <param name="framerate">The frame rate to hold, frames per second; 0 (the default) does not wait.</param>
        /// <param name="cancellationToken">Ends the wait early; the tick is then not marked.</param>
        /// <returns>A task of the milliseconds since the previous tick, the wait included.</returns>
        /// <exception cref="OperationCanceledException">Thrown (through the task) when <paramref name="cancellationToken"/> is canceled during the wait.</exception>
        /// <remarks>
        ///     A wait shorter than the system timer's period may last a whole period (about 15.6 ms on Windows unless the
        ///     process raised the timer resolution), where <see cref="Tick"/>'s sleep has the same limit: hold rates of 60 fps
        ///     and below for even pacing. One tick at a time: await each before the next, from any thread.
        /// </remarks>
        public async ValueTask<long> TickAsync(double framerate = 0, CancellationToken cancellationToken = default) {
            long delay = Delay(framerate, out float rate);
            if (delay > 0)
                await Task.Delay(TimeSpan.FromMilliseconds(delay), cancellationToken).ConfigureAwait(false);
            return Mark(rate);
        }

        /// <summary>
        ///     The first half of pygame's <c>accurate_delay</c> pacing: the milliseconds to wait before this tick, measuring
        ///     the raw time since the previous one on the way.
        /// </summary>
        /// <param name="framerate">The frame rate to hold.</param>
        /// <param name="rate">The rate as pygame's C <c>float</c>, for <see cref="Mark"/>.</param>
        /// <returns>The wait in milliseconds, 0 for a rate of 0 or a frame that already took long enough.</returns>
        private long Delay(double framerate, out float rate) {
            rate = (float)framerate;
            // C's `if (framerate)`: NaN is non-zero, so it takes this branch too.
            if (rate == 0f)
                return 0;
            int endtime = unchecked((int)((1.0f / rate) * 1000.0f));
            _rawPassed = Now() - _lastTick;
            // In long arithmetic, where pygame's int subtraction could overflow for an extreme endtime.
            long delay = endtime - _rawPassed;
            return Math.Max(delay, 0);
        }

        /// <summary>The second half of a tick, after the wait: records the tick's time and updates the frame-rate window.</summary>
        /// <param name="rate">The rate as pygame's C <c>float</c>, from <see cref="Delay"/>.</param>
        /// <returns>The milliseconds since the previous tick, the wait included.</returns>
        private long Mark(float rate) {
            long nowtime = Now();
            _timePassed = nowtime - _lastTick;
            _fpsCount += 1;
            _lastTick = nowtime;
            if (rate == 0f)
                _rawPassed = _timePassed;

            if (_fpsTick == 0) {
                _fpsCount = 0;
                _fpsTick = nowtime;
            } else if (_fpsCount >= 10) {
                _fps = _fpsCount / ((nowtime - _fpsTick) / 1000.0f);
                _fpsCount = 0;
                _fpsTick = nowtime;
            }

            return _timePassed;
        }

        /// <summary>The milliseconds between the last two ticks, the pacing sleep included (pygame's <c>Clock.get_time()</c>).</summary>
        /// <returns>The time in milliseconds.</returns>
        /// <remarks>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/time.c#L616">src_c/time.c</see>.</remarks>
        public long GetTime() => _timePassed;

        /// <summary>The milliseconds between the last two ticks, without the pacing sleep (pygame's <c>Clock.get_rawtime()</c>).</summary>
        /// <returns>The time in milliseconds.</returns>
        /// <remarks>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/time.c#L622">src_c/time.c</see>.</remarks>
        public long GetRawTime() => _rawPassed;

        /// <summary>The frame rate averaged over the last ten ticks, 0 until ten have run (pygame's <c>Clock.get_fps()</c>).</summary>
        /// <returns>Frames per second.</returns>
        /// <remarks>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/time.c#L610">src_c/time.c</see>.</remarks>
        public float GetFps() => _fps;

        /// <summary>
        ///     The current tick, milliseconds since the clock was created. Starts at 1 rather than 0 so that pygame's
        ///     "no frame-rate window yet" test (<c>fps_tick == 0</c>) cannot confuse a window that starts at the very first
        ///     millisecond with none.
        /// </summary>
        /// <returns>The tick.</returns>
        private long Now() => _ticks.ElapsedMilliseconds + 1;
    }
}
