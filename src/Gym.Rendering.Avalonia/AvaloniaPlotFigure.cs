#nullable enable
using System;
using System.Collections.Generic;
using System.Threading;
using Avalonia.Controls;
using Avalonia.Threading;
using Gym.Utils;
using NumSharp;
using ScottPlot.Avalonia;

namespace Gym.Rendering.Avalonia {
    /// <summary>
    ///     <see cref="PlayPlot"/>'s figure drawn by ScottPlot in an Avalonia window (DR-39): one plot per row, stacked top to
    ///     bottom, with titles, ticks, axis labels and live updates, where the core's <see cref="PlotFigure"/> draws bare
    ///     markers. Pass <see cref="Subplots"/> to <see cref="PlayPlot"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The window lives on Avalonia's shared UI thread (the one <see cref="AvaloniaEnvViewer"/> windows use) and opens at
    ///     the first <see cref="Pause"/>, as matplotlib shows a figure there. <see cref="PlayPlot"/> edits the axes from the
    ///     thread it plays on; the edits lock each plot's <c>Sync</c> object, which ScottPlot's renderer locks too.
    ///     </para>
    ///     <para>
    ///     Footgun: closing the window ends the figure's display; later pauses still wait their interval and draw nothing,
    ///     as nothing reopens it. <see cref="Control"/> is ScottPlot's own control, for styling (axis labels, colors); change
    ///     it on the UI thread, or under a plot's <c>Sync</c> lock.
    ///     </para>
    /// </remarks>
    public sealed class AvaloniaPlotFigure : IPlotFigure {
        /// <summary>The window's width in pixels: matplotlib's 6.4 inches at 100 dots per inch, as <see cref="PlotFigure"/>.</summary>
        public const int Width = 640;

        /// <summary>The window's height in pixels: matplotlib's 4.8 inches at 100 dots per inch.</summary>
        public const int Height = 480;

        /// <summary>The window, created hidden on the UI thread.</summary>
        private readonly Window _window;

        /// <summary>The axes, top to bottom, each over one of the control's plots.</summary>
        private readonly ScottPlotAxes[] _axes;

        /// <summary>Whether the window was shown; read and written on the UI thread.</summary>
        private bool _shown;

        /// <summary>Whether the user closed the window; written on the UI thread.</summary>
        private volatile bool _closed;

        /// <summary>
        ///     A figure of <paramref name="rows"/> plots (matplotlib's <c>plt.subplots(rows)</c>), built on Avalonia's UI thread,
        ///     which this starts when no Avalonia window has yet; the window stays hidden until the first <see cref="Pause"/>.
        /// </summary>
        /// <param name="rows">The number of plots, at least 1.</param>
        /// <param name="title">The window's title, or <see langword="null"/> for <c>PlayPlot</c>.</param>
        /// <exception cref="ValueError">Thrown for fewer than one row, with matplotlib's message: <c>Number of rows must be a positive integer, not 0</c>.</exception>
        public AvaloniaPlotFigure(int rows, string? title = null) {
            if (rows < 1)
                throw new ValueError($"Number of rows must be a positive integer, not {rows}");
            // The control and the window must be made on the UI thread; this blocks until they are, as matplotlib's
            // subplots returns a figure ready to draw on. The rows are laid out there too, before the control first
            // renders, so the renderer never sees a half-built multiplot.
            (Window window, AvaPlot control, ScottPlotAxes[] axes) = StaticAvaloniaApp.InvokeAsync(() => {
                var plotControl = new AvaPlot();
                ScottPlotAxes[] plotAxes = ScottPlotAxes.StackRows(plotControl.Multiplot, rows);
                var plotWindow = new Window { Width = Width, Height = Height, Title = title ?? "PlayPlot", Content = plotControl };
                plotWindow.Closed += (_, _) => _closed = true;
                return (plotWindow, plotControl, plotAxes);
            }, title).GetAwaiter().GetResult();
            _window = window;
            Control = control;
            _axes = axes;
        }

        /// <summary>ScottPlot's control in the window, for styling; touch it on Avalonia's UI thread.</summary>
        public AvaPlot Control { get; }

        /// <summary>The axes, top to bottom.</summary>
        public IReadOnlyList<IPlotAxes> Axes => _axes;

        /// <summary>The maker <see cref="PlayPlot"/> takes: <c>new PlayPlot(callback, horizon, names, AvaloniaPlotFigure.Subplots())</c>.</summary>
        /// <param name="title">The windows' title, or <see langword="null"/> for <c>PlayPlot</c>.</param>
        /// <returns>The maker.</returns>
        public static PlayPlot.PlotSubplots Subplots(string? title = null) => rows => new AvaloniaPlotFigure(rows, title);

        /// <summary>
        ///     Shows the figure as it is now (<c>plt.pause</c>): opens the window the first time, redraws it, then waits
        ///     <paramref name="interval"/> seconds on the calling thread while the window keeps running on its own.
        /// </summary>
        /// <param name="interval">The pause in seconds; under a millisecond only yields the thread.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a negative or non-finite interval.</exception>
        public void Pause(double interval) {
            if (!(interval >= 0) || double.IsInfinity(interval))
                throw new ArgumentOutOfRangeException(nameof(interval), interval, "The pause must be a finite number of seconds, not negative.");
            Dispatcher.UIThread.Invoke(() => {
                // A window the user closed cannot be shown again; the figure simply stops drawing.
                if (_closed)
                    return;
                if (!_shown) {
                    _window.Show();
                    _shown = true;
                }

                Control.Refresh();
            }, DispatcherPriority.MaxValue);
            Thread.Sleep(TimeSpan.FromSeconds(interval));
        }
    }
}
