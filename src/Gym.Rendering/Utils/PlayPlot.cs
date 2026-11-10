using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Gym.Environments;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Rendering;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Color = Gym.Rendering.Color;

namespace Gym.Utils {
    /// <summary>
    ///     What <see cref="PlayPlot"/> computes its points with: a function of one transition (the arguments of
    ///     <see cref="PlayCallback"/>) returning one value per plot, in the order of <see cref="PlayPlot.PlotNames"/>.
    /// </summary>
    /// <param name="obsT">The observation before the action (the reset's <c>(obs, info)</c> after a reset, as <c>play</c> passes it).</param>
    /// <param name="obsTp1">The observation after the action.</param>
    /// <param name="action">The action.</param>
    /// <param name="rew">The reward.</param>
    /// <param name="terminated">Whether the episode ended in a terminal state.</param>
    /// <param name="truncated">Whether the episode was cut short.</param>
    /// <param name="info">The step's info.</param>
    /// <returns>One point per plot; more or fewer raises Python's <c>zip(strict=True)</c> error after the points it could pair.</returns>
    public delegate IEnumerable<double> PlayPlotMetrics(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info);

    /// <summary>
    ///     Live plots of metrics while playing (Gymnasium's <c>PlayPlot</c>): pass <see cref="Callback"/> to
    ///     <see cref="Play.Run"/>, and each step adds the points of <see cref="DataCallback"/> to one scatter plot per name,
    ///     showing the last <see cref="HorizonTimesteps"/> steps.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Gymnasium draws with matplotlib; Gym.NET draws through <see cref="IPlotFigure"/>, matplotlib's calls as
    ///     <c>PlayPlot</c> makes them (<c>subplots</c>, <c>set_title</c>, <c>scatter</c>, <c>remove</c>, <c>set_xlim</c>,
    ///     <c>pause</c>), so the plotted data is Gymnasium's call for call. The default figure (<see cref="PlotFigure"/>) is
    ///     a plain window from the viewer factory: matplotlib's layout and blue markers, without text (the window's title
    ///     names the plots). Pass <c>subplots</c> to draw elsewhere: the viewer packages ship ScottPlot figures with titles,
    ///     ticks, labels and live updates (DR-39), <c>AvaloniaPlotFigure.Subplots()</c> of Gym.NET.Rendering.Avalonia and
    ///     <c>WinFormPlotFigure.Subplots()</c> of Gym.NET.Rendering.WinForm; this package takes no chart dependency.
    ///     </para>
    ///     <para>Not thread-safe: call <see cref="Callback"/> from one thread, as <c>play</c> does.</para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py#L377">gymnasium/utils/play.py</see>.</para>
    /// </remarks>
    public class PlayPlot {
        /// <summary>Creates the figure, one axes per name with its title, and an empty series per plot.</summary>
        /// <param name="callback">Computes the points of a transition, one per plot.</param>
        /// <param name="horizonTimesteps">How many recent steps each plot shows (<c>deque(maxlen=...)</c>); 0 keeps none.</param>
        /// <param name="plotNames">The plots' titles, one plot each.</param>
        /// <param name="subplots">
        ///     The figure's maker, matplotlib's <c>plt.subplots(nrows)</c>; <see langword="null"/> uses
        ///     <see cref="PlotFigure.Subplots"/> with <c>Gymnasium.HumanViewerFactory</c>.
        /// </param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="callback"/> or <paramref name="plotNames"/> is <see langword="null"/>.</exception>
        /// <exception cref="DependencyNotInstalled">
        ///     Thrown by the default figure when no viewer factory is set, where Gymnasium raises it when matplotlib is missing.
        /// </exception>
        /// <exception cref="ValueError">
        ///     Thrown for no plot names by the default figure (matplotlib's <c>Number of rows must be a positive integer, not
        ///     0</c>), and for a negative horizon (<c>maxlen must be non-negative</c>).
        /// </exception>
        public PlayPlot(PlayPlotMetrics callback, int horizonTimesteps, IReadOnlyList<string> plotNames, PlotSubplots? subplots = null) {
            DataCallback = callback ?? throw new ArgumentNullException(nameof(callback));
            HorizonTimesteps = horizonTimesteps;
            PlotNames = plotNames ?? throw new ArgumentNullException(nameof(plotNames));

            int numPlots = PlotNames.Count;
            Fig = (subplots ?? PlotFigure.Subplots())(numPlots);
            Ax = Fig.Axes;
            // zip(self.ax, plot_names, strict=True): a figure with another number of axes is a broken subplots.
            if (Ax.Count != numPlots)
                throw new ValueError($"zip() argument 2 is {(Ax.Count < numPlots ? "longer" : "shorter")} than argument 1");
            for (int i = 0; i < numPlots; i++)
                Ax[i].SetTitle(PlotNames[i]);
            T = 0;
            _curPlot = new IPlotArtist?[numPlots];
            Data = Enumerable.Range(0, numPlots).Select(_ => new Deque<double>(horizonTimesteps)).ToList();
        }

        /// <summary>The points of each transition (<c>data_callback</c>).</summary>
        public PlayPlotMetrics DataCallback { get; }

        /// <summary>How many recent steps each plot shows (<c>horizon_timesteps</c>).</summary>
        public int HorizonTimesteps { get; }

        /// <summary>The plots' titles (<c>plot_names</c>).</summary>
        public IReadOnlyList<string> PlotNames { get; }

        /// <summary>The figure (<c>fig</c>).</summary>
        public IPlotFigure Fig { get; }

        /// <summary>One axes per plot, top to bottom (<c>ax</c>; always a list, where Gymnasium wraps a single axes in one).</summary>
        public IReadOnlyList<IPlotAxes> Ax { get; }

        /// <summary>The number of transitions plotted (<c>t</c>).</summary>
        public int T { get; private set; }

        /// <summary>The scatter shown on each axes, <see langword="null"/> before the first point (<c>cur_plot</c>).</summary>
        private readonly IPlotArtist?[] _curPlot;

        /// <summary>The scatter shown on each axes, <see langword="null"/> before the first point (<c>cur_plot</c>).</summary>
        public IReadOnlyList<IPlotArtist?> CurPlot => _curPlot;

        /// <summary>The last <see cref="HorizonTimesteps"/> points of each plot (<c>data</c>).</summary>
        public IReadOnlyList<Deque<double>> Data { get; }

        /// <summary>
        ///     Plots a transition (<c>callback</c>): appends <see cref="DataCallback"/>'s points, then redraws every plot over
        ///     the steps <c>[max(0, t - horizon), t)</c> and lets the figure show it (<c>plt.pause(0.000001)</c>). Pass it as
        ///     <see cref="Play.Run"/>'s <c>callback</c>.
        /// </summary>
        /// <param name="obsT">The observation before the action.</param>
        /// <param name="obsTp1">The observation after the action.</param>
        /// <param name="action">The action.</param>
        /// <param name="rew">The reward.</param>
        /// <param name="terminated">Whether the episode ended in a terminal state.</param>
        /// <param name="truncated">Whether the episode was cut short.</param>
        /// <param name="info">The step's info.</param>
        /// <exception cref="ValueError">
        ///     Thrown when the callback returns more or fewer points than there are plots (<c>zip() argument 2 is shorter than
        ///     argument 1</c>, or <c>longer</c>), after appending the points it could pair, as Python's <c>zip(strict=True)</c>
        ///     does; and by the default figure when the plots' lengths then disagree (<c>x and y must be the same size</c>).
        /// </exception>
        public void Callback(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info) {
            IEnumerable<double> points = DataCallback(obsT, obsTp1, action, rew, terminated, truncated, info);
            // for point, data_series in zip(points, self.data, strict=True): data_series.append(point)
            using (IEnumerator<double> point = points.GetEnumerator()) {
                for (int i = 0;; i++) {
                    if (!point.MoveNext()) {
                        if (i < Data.Count)
                            throw new ValueError("zip() argument 2 is longer than argument 1");
                        break;
                    }

                    if (i >= Data.Count)
                        throw new ValueError("zip() argument 2 is shorter than argument 1");
                    Data[i].Append(point.Current);
                }
            }

            T += 1;
            int xmin = Math.Max(0, T - HorizonTimesteps), xmax = T;
            double[] steps = Enumerable.Range(xmin, xmax - xmin).Select(x => (double)x).ToArray();
            for (int i = 0; i < _curPlot.Length; i++) {
                _curPlot[i]?.Remove();
                _curPlot[i] = Ax[i].Scatter(steps, Data[i].ToArray(), "blue");
                Ax[i].SetXLim(xmin, xmax);
            }

            Fig.Pause(0.000001);
        }

        /// <summary>The figure's maker: matplotlib's <c>plt.subplots(nrows)</c>, a figure with <paramref name="rows"/> axes stacked top to bottom.</summary>
        /// <param name="rows">The number of axes.</param>
        /// <returns>The figure.</returns>
        public delegate IPlotFigure PlotSubplots(int rows);
    }

    /// <summary>A figure as <see cref="PlayPlot"/> draws on it: matplotlib's <c>Figure</c> with <c>plt.pause</c>.</summary>
    public interface IPlotFigure {
        /// <summary>The axes, top to bottom.</summary>
        IReadOnlyList<IPlotAxes> Axes { get; }

        /// <summary>Shows the figure as it is now and lets its window run for <paramref name="interval"/> seconds (<c>plt.pause</c>).</summary>
        /// <param name="interval">The pause in seconds; <see cref="PlayPlot"/> passes 0.000001.</param>
        void Pause(double interval);
    }

    /// <summary>One plot of an <see cref="IPlotFigure"/>: matplotlib's <c>Axes</c> as <see cref="PlayPlot"/> uses it.</summary>
    public interface IPlotAxes {
        /// <summary>Sets the title (<c>set_title</c>).</summary>
        /// <param name="title">The title.</param>
        void SetTitle(string title);

        /// <summary>Adds a scatter of points (<c>scatter(x, y, c=color)</c>).</summary>
        /// <param name="x">The x values, the steps.</param>
        /// <param name="y">The y values, as many as <paramref name="x"/>.</param>
        /// <param name="color">The colour's name; <see cref="PlayPlot"/> passes <c>blue</c>.</param>
        /// <returns>The scatter, to remove before the next.</returns>
        IPlotArtist Scatter(IReadOnlyList<double> x, IReadOnlyList<double> y, string color);

        /// <summary>Sets the x range shown (<c>set_xlim(left, right)</c>).</summary>
        /// <param name="left">The left limit.</param>
        /// <param name="right">The right limit.</param>
        void SetXLim(double left, double right);
    }

    /// <summary>Something drawn on an <see cref="IPlotAxes"/>, which can be taken off it (matplotlib's <c>Artist.remove</c>).</summary>
    public interface IPlotArtist {
        /// <summary>Takes it off its axes.</summary>
        void Remove();
    }

    /// <summary>
    ///     <see cref="PlayPlot"/>'s default figure: matplotlib's default layout (a 640 by 480 white figure, axes at its
    ///     subplot parameters, blue round markers) drawn with <c>Gym.Rendering</c> and shown in a window from the viewer
    ///     factory at each <see cref="Pause"/>. No text is drawn (no titles, ticks or labels): the window's title names the
    ///     plots, top to bottom.
    /// </summary>
    /// <remarks>
    ///     Not matplotlib's rendering, and not compared with it: only the data <see cref="PlayPlot"/> hands it is Gymnasium's.
    ///     The y range follows the data with matplotlib's 5% margins; a range of one value is widened by 5% of it (0.05 around
    ///     zero), as matplotlib widens identical limits. The window opens at the first <see cref="Pause"/>, as matplotlib
    ///     shows a figure there.
    /// </remarks>
    public sealed class PlotFigure : IPlotFigure {
        /// <summary>The figure's width in pixels: matplotlib's 6.4 inches at 100 dots per inch.</summary>
        public const int Width = 640;

        /// <summary>The figure's height in pixels: matplotlib's 4.8 inches at 100 dots per inch.</summary>
        public const int Height = 480;

        /// <summary>The window's factory, resolved when the figure was made.</summary>
        private readonly IEnvironmentViewerFactoryDelegate _factory;

        /// <summary>The axes, top to bottom.</summary>
        private readonly PlotAxes[] _axes;

        /// <summary>The window, opened at the first <see cref="Pause"/>.</summary>
        private IEnvViewer? _window;

        /// <summary>A figure of <paramref name="rows"/> axes (matplotlib's <c>plt.subplots(rows)</c>).</summary>
        /// <param name="rows">The number of axes, at least 1.</param>
        /// <param name="viewerFactory">The window's factory; <see langword="null"/> reads <c>Gymnasium.HumanViewerFactory</c>.</param>
        /// <exception cref="ValueError">Thrown for fewer than one row, with matplotlib's message: <c>Number of rows must be a positive integer, not 0</c>.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown when no viewer factory is given or set.</exception>
        public PlotFigure(int rows, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            if (rows < 1)
                throw new ValueError($"Number of rows must be a positive integer, not {rows}");
            _factory = Gymnasium.ResolveHumanViewerFactory(viewerFactory);
            // matplotlib's subplot parameters: left 0.125, right 0.9, bottom 0.11, top 0.88, rows separated by 0.2 of a row.
            double cell = (0.88 - 0.11) / (rows + 0.2 * (rows - 1));
            _axes = new PlotAxes[rows];
            for (int i = 0; i < rows; i++) {
                double top = 0.88 - i * cell * 1.2;
                _axes[i] = new PlotAxes((int)Math.Round(0.125 * Width), (int)Math.Round((1 - top) * Height), (int)Math.Round(0.775 * Width), (int)Math.Round(cell * Height));
            }
        }

        /// <summary>The maker <see cref="PlayPlot"/> takes, with a viewer factory of the caller's.</summary>
        /// <param name="viewerFactory">The windows' factory; <see langword="null"/> reads <c>Gymnasium.HumanViewerFactory</c> when a figure is made.</param>
        /// <returns>The maker.</returns>
        public static PlayPlot.PlotSubplots Subplots(IEnvironmentViewerFactoryDelegate? viewerFactory = null) => rows => new PlotFigure(rows, viewerFactory);

        /// <inheritdoc />
        public IReadOnlyList<IPlotAxes> Axes => _axes;

        /// <summary>Draws the figure and shows it in its window, opening the window the first time; then sleeps <paramref name="interval"/> seconds.</summary>
        /// <param name="interval">The pause in seconds; under a millisecond only yields the thread.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a negative or non-finite interval.</exception>
        public void Pause(double interval) {
            if (!(interval >= 0) || double.IsInfinity(interval))
                throw new ArgumentOutOfRangeException(nameof(interval), interval, "The pause must be a finite number of seconds, not negative.");
            var figure = new Surface(Width, Height);
            figure.Fill(new Color(255, 255, 255));
            foreach (PlotAxes axes in _axes)
                axes.Draw(figure);

            // The factory may start a UI thread; the window is needed synchronously, as matplotlib's show is.
            _window ??= _factory(Width, Height, string.Join(" | ", _axes.Select(a => a.Title ?? ""))).GetAwaiter().GetResult();
            using (Image<Rgb24> frame = Image.LoadPixelData<Rgb24>(figure.ToRgbBytes(), Width, Height))
                _window.Render(frame);
            Thread.Sleep(TimeSpan.FromSeconds(interval));
        }

        /// <summary>One axes of the default figure: its pixel rectangle, title, x limits and current scatter.</summary>
        private sealed class PlotAxes : IPlotAxes {
            /// <summary>matplotlib's <c>blue</c>.</summary>
            private static readonly Color Blue = new(0, 0, 255);

            /// <summary>The rectangle's left column.</summary>
            private readonly int _left;

            /// <summary>The rectangle's top row.</summary>
            private readonly int _top;

            /// <summary>The rectangle's width.</summary>
            private readonly int _width;

            /// <summary>The rectangle's height.</summary>
            private readonly int _height;

            /// <summary>The x limits, (0, 1) until set, as matplotlib's empty axes.</summary>
            private (double Left, double Right) _xlim = (0, 1);

            /// <summary>The scatters on the axes, drawn in order.</summary>
            private readonly List<Scatter> _scatters = new();

            /// <summary>An axes at a pixel rectangle.</summary>
            /// <param name="left">The left column.</param>
            /// <param name="top">The top row.</param>
            /// <param name="width">The width.</param>
            /// <param name="height">The height.</param>
            public PlotAxes(int left, int top, int width, int height) {
                _left = left;
                _top = top;
                _width = width;
                _height = height;
            }

            /// <summary>The title, or <see langword="null"/> before <see cref="SetTitle"/>.</summary>
            public string? Title { get; private set; }

            /// <inheritdoc />
            public void SetTitle(string title) => Title = title;

            /// <summary>Adds a scatter.</summary>
            /// <param name="x">The x values.</param>
            /// <param name="y">The y values.</param>
            /// <param name="color">Ignored: the markers are matplotlib's <c>blue</c>, which is all <see cref="PlayPlot"/> asks for.</param>
            /// <returns>The scatter.</returns>
            /// <exception cref="ValueError">Thrown when the lengths differ, with matplotlib's message: <c>x and y must be the same size</c>.</exception>
            public IPlotArtist Scatter(IReadOnlyList<double> x, IReadOnlyList<double> y, string color) {
                if (x.Count != y.Count)
                    throw new ValueError("x and y must be the same size");
                var scatter = new Scatter(this, x.ToArray(), y.ToArray());
                _scatters.Add(scatter);
                return scatter;
            }

            /// <summary>Sets the x limits; identical limits widen by 5% of their value (0.05 around zero), as matplotlib does after its warning.</summary>
            /// <param name="left">The left limit.</param>
            /// <param name="right">The right limit.</param>
            public void SetXLim(double left, double right) => _xlim = NonSingular(left, right);

            /// <summary>Removes a scatter.</summary>
            /// <param name="scatter">The scatter.</param>
            public void Remove(Scatter scatter) => _scatters.Remove(scatter);

            /// <summary>Draws the axes' frame and the markers of its scatters inside it.</summary>
            /// <param name="figure">The figure's surface.</param>
            public void Draw(Surface figure) {
                List<double> ys = _scatters.SelectMany(s => s.Y).Where(double.IsFinite).ToList();
                (double bottom, double top) = ys.Count == 0 ? (0, 1) : NonSingular(ys.Min() - 0.05 * (ys.Max() - ys.Min()), ys.Max() + 0.05 * (ys.Max() - ys.Min()));
                foreach (Scatter scatter in _scatters) {
                    for (int i = 0; i < scatter.X.Length; i++) {
                        double px = _left + (scatter.X[i] - _xlim.Left) / (_xlim.Right - _xlim.Left) * _width;
                        double py = _top + _height - (scatter.Y[i] - bottom) / (top - bottom) * _height;
                        // A marker is drawn when its centre is inside the frame (matplotlib clips it at the frame instead).
                        if (double.IsFinite(px) && double.IsFinite(py) && px >= _left && px <= _left + _width && py >= _top && py <= _top + _height)
                            GfxDraw.FilledCircle(figure, (long)Math.Round(px), (long)Math.Round(py), 4, Blue);
                    }
                }

                Rendering.Draw.Rect(figure, new Color(0, 0, 0), _left, _top, _width, _height, 1);
            }

            /// <summary>matplotlib's widening of a range of one value: 5% of it to each side, or 0.05 around zero.</summary>
            /// <param name="low">The lower limit.</param>
            /// <param name="high">The upper limit.</param>
            /// <returns>The limits, widened when they are equal.</returns>
            private static (double, double) NonSingular(double low, double high) {
                if (low != high)
                    return (low, high);
                return low == 0 ? (-0.05, 0.05) : (low - 0.05 * Math.Abs(low), high + 0.05 * Math.Abs(high));
            }
        }

        /// <summary>A scatter on a default axes.</summary>
        private sealed class Scatter : IPlotArtist {
            /// <summary>The axes it is on.</summary>
            private readonly PlotAxes _axes;

            /// <summary>A scatter of points on an axes.</summary>
            /// <param name="axes">The axes.</param>
            /// <param name="x">The x values.</param>
            /// <param name="y">The y values.</param>
            public Scatter(PlotAxes axes, double[] x, double[] y) {
                _axes = axes;
                X = x;
                Y = y;
            }

            /// <summary>The x values.</summary>
            public double[] X { get; }

            /// <summary>The y values.</summary>
            public double[] Y { get; }

            /// <inheritdoc />
            public void Remove() => _axes.Remove(this);
        }
    }
}
