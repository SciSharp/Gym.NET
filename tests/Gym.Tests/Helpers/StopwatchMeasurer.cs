using System;
using System.Diagnostics;

namespace Gym.Tests.Helpers {
    /// <summary>
    ///     Simply computes time between construction to disposal.
    /// </summary>
    public class StopwatchMeasurer : IDisposable {
        public string Title { get; set; }
        public Func<TimeSpan, string> Formatter { get; set; }
        private readonly Stopwatch _timer;

        /// <inheritdoc />
        public StopwatchMeasurer(string title, Func<TimeSpan, string> formatter = null) {
            Title = title;
            Formatter = formatter ?? (span => span.TotalMilliseconds.ToString());
            _timer = new Stopwatch();
            _timer.Start();
        }

        /// <inheritdoc />
        public void Dispose() {
            _timer.Stop();
            Console.WriteLine($"{Title} {Formatter(_timer.Elapsed)}");
        }
    }

    /// <summary>
    ///     Simply computes time between construction to disposal.
    /// </summary>
    public class TimeMeasurer : IDisposable {
        public string Title { get; set; }
        public Func<TimeSpan, string> Formatter { get; set; }
        private readonly Stopwatch _timer;

        /// <inheritdoc />
        public TimeMeasurer(string title, Func<TimeSpan, string> formatter = null) {
            Title = title;
            Formatter = formatter ?? (span => span.TotalMilliseconds.ToString());
            _timer = new Stopwatch();
            _timer.Start();
        }

        /// <inheritdoc />
        public void Dispose() {
            _timer.Stop();
            Console.WriteLine($"{Title} {Formatter(_timer.Elapsed)}");
        }
    }
}