using System;
using System.Diagnostics;
using System.Threading;

namespace Gym.Rendering {
    /// <summary>
    ///     Port of <c>pygame.time.Clock</c> (pygame-ce 2.5.8, <c>src_c/time.c</c>): paces a render loop. Human-mode rendering
    ///     calls <see cref="Tick"/> with <c>metadata["render_fps"]</c> once per frame, which sleeps just long enough to hold
    ///     that frame rate, so a human-mode episode plays at the env's intended speed.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Time is wall-clock milliseconds since the clock was created (SDL's <c>SDL_GetTicks</c> counts from SDL's
    ///     initialization); only differences are observable, and timing takes no part in parity. The arithmetic is pygame's,
    ///     including its float32 frame-time computation and whole-millisecond sleeps, so the pacing matches.
    ///     </para>
    ///     <para>Not thread-safe: tick a clock from the thread that renders.</para>
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
        ///     Port of <c>Clock.tick(framerate)</c>: marks a frame; with a positive <paramref name="framerate"/>, first sleeps
        ///     until <c>1000 / framerate</c> ms have passed since the previous tick.
        /// </summary>
        /// <param name="framerate">The frame rate to hold, frames per second; 0 (the default) does not sleep.</param>
        /// <returns>The milliseconds since the previous tick, the sleep included.</returns>
        /// <remarks>
        ///     pygame reads the rate as a C <c>float</c> and computes <c>(int)((1.0f / framerate) * 1000.0f)</c>, so the frame
        ///     time is whole milliseconds (20 ms at 50 fps). A negative or infinite rate gives a delay clamped at zero, as in
        ///     pygame, and nothing sleeps; so does NaN, for which pygame's C conversion is undefined.
        /// </remarks>
        public long Tick(double framerate = 0) {
            float rate = (float)framerate;
            // C's `if (framerate)`: NaN is non-zero, so it takes this branch too.
            if (rate != 0f) {
                int endtime = unchecked((int)((1.0f / rate) * 1000.0f));
                _rawPassed = Now() - _lastTick;
                // In long arithmetic, where pygame's int subtraction could overflow for an extreme endtime.
                long delay = endtime - _rawPassed;
                delay = Math.Max(delay, 0);
                if (delay > 0)
                    Thread.Sleep(TimeSpan.FromMilliseconds(delay));
            }

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

        /// <summary>Port of <c>Clock.get_time()</c>: the milliseconds between the last two ticks, the pacing sleep included.</summary>
        /// <returns>The time in milliseconds.</returns>
        public long GetTime() => _timePassed;

        /// <summary>Port of <c>Clock.get_rawtime()</c>: the milliseconds between the last two ticks, without the pacing sleep.</summary>
        /// <returns>The time in milliseconds.</returns>
        public long GetRawTime() => _rawPassed;

        /// <summary>Port of <c>Clock.get_fps()</c>: the frame rate averaged over the last ten ticks, 0 until ten have run.</summary>
        /// <returns>Frames per second.</returns>
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
