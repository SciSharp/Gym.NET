#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using ScottPlot;

namespace Gym.Utils {
    /// <summary>
    ///     One ScottPlot plot as <see cref="PlayPlot"/>'s axes (DR-39): titles, blue scatters of the recent points, the x range
    ///     PlayPlot sets and a y range that follows the data, with ScottPlot's own ticks and labels. Shared by the two viewer
    ///     packages' figures (this file is compiled into each), so both draw a figure the same way.
    /// </summary>
    /// <remarks>
    ///     Every change locks the plot's <c>Sync</c> object, which ScottPlot's renderer locks too, so <see cref="PlayPlot"/>
    ///     can edit the axes from the thread it plays on while the window's UI thread draws them. Only the data is
    ///     Gymnasium's; the look is ScottPlot's, not matplotlib's.
    /// </remarks>
    internal sealed class ScottPlotAxes : IPlotAxes {
        /// <summary>An adapter over <paramref name="plot"/>, which keeps being drawn by its owner.</summary>
        /// <param name="plot">The ScottPlot plot, one of a figure's subplots.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="plot"/> is <see langword="null"/>.</exception>
        public ScottPlotAxes(Plot plot) {
            Plot = plot ?? throw new ArgumentNullException(nameof(plot));
        }

        /// <summary>The ScottPlot plot these axes draw on.</summary>
        public Plot Plot { get; }

        /// <summary>
        ///     Lays <paramref name="multiplot"/> out as <paramref name="rows"/> plots stacked top to bottom (matplotlib's
        ///     <c>plt.subplots(rows)</c>) and adapts each one: a figure's axes, top to bottom. Call it on the thread that owns
        ///     the control, before the control first renders.
        /// </summary>
        /// <remarks>
        ///     Footgun: ScottPlot's <c>AddPlots(n)</c> sets the total number of plots, adding or removing until there are
        ///     <c>n</c>; it does not add <c>n</c> more. A control's multiplot starts with the control's own plot, which stays
        ///     the first row, so the total asked for is <paramref name="rows"/> itself; <c>rows - 1</c> leaves the figure one
        ///     plot short and its last row's lookup out of range.
        /// </remarks>
        /// <param name="multiplot">The control's multiplot.</param>
        /// <param name="rows">The number of rows, at least 1.</param>
        /// <returns>The axes, one per row, top to bottom; the first over the multiplot's first plot.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="multiplot"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="rows"/> is less than 1.</exception>
        public static ScottPlotAxes[] StackRows(IMultiplot multiplot, int rows) {
            if (multiplot is null)
                throw new ArgumentNullException(nameof(multiplot));
            if (rows < 1)
                throw new ArgumentOutOfRangeException(nameof(rows), rows, "A figure needs at least one row.");
            multiplot.Layout = new ScottPlot.MultiplotLayouts.Rows();
            multiplot.AddPlots(rows);
            return Enumerable.Range(0, rows).Select(i => new ScottPlotAxes(multiplot.GetPlot(i))).ToArray();
        }

        /// <summary>Sets the plot's title (<c>set_title</c>).</summary>
        /// <param name="title">The title.</param>
        public void SetTitle(string title) {
            lock (Plot.Sync)
                Plot.Title(title);
        }

        /// <summary>
        ///     Adds a scatter of markers without lines (<c>scatter(x, y, c=color)</c>) and fits the y range to the data, as
        ///     matplotlib autoscales a scatter's axes.
        /// </summary>
        /// <param name="x">The x values.</param>
        /// <param name="y">The y values.</param>
        /// <param name="color">A matplotlib color: a base name (<c>blue</c>, <c>b</c>, ... <c>black</c>, <c>k</c>) or <c>#rrggbb</c>; anything else is blue.</param>
        /// <returns>The scatter, which <see cref="IPlotArtist.Remove"/> takes off the plot.</returns>
        /// <exception cref="ValueError">Thrown when the lengths differ, with matplotlib's message: <c>x and y must be the same size</c>.</exception>
        public IPlotArtist Scatter(IReadOnlyList<double> x, IReadOnlyList<double> y, string color) {
            if (x.Count != y.Count)
                throw new ValueError("x and y must be the same size");
            ScottPlot.Plottables.Scatter scatter;
            lock (Plot.Sync) {
                scatter = Plot.Add.ScatterPoints(x.ToArray(), y.ToArray(), ParseColor(color));
                Plot.Axes.AutoScaleY();
            }

            return new Artist(Plot, scatter);
        }

        /// <summary>Sets the x range shown (<c>set_xlim(left, right)</c>), refitting the y range to the data.</summary>
        /// <param name="left">The left limit.</param>
        /// <param name="right">The right limit.</param>
        public void SetXLim(double left, double right) {
            lock (Plot.Sync) {
                Plot.Axes.SetLimitsX(left, right);
                Plot.Axes.AutoScaleY();
            }
        }

        /// <summary>matplotlib's base colors by name and letter, and <c>#rrggbb</c>; blue otherwise, PlayPlot's only color.</summary>
        /// <remarks>
        ///     The hex digits are checked here because ScottPlot's parser turns a malformed one (<c>#zzzzzz</c>) into a fully
        ///     transparent black, which would draw the scatter invisible rather than blue.
        /// </remarks>
        /// <param name="color">The color.</param>
        /// <returns>ScottPlot's color.</returns>
        private static Color ParseColor(string color) => color switch {
            "b" or "blue" => Colors.Blue,
            "g" or "green" => Colors.Green,
            "r" or "red" => Colors.Red,
            "c" or "cyan" => Colors.Cyan,
            "m" or "magenta" => Colors.Magenta,
            "y" or "yellow" => Colors.Yellow,
            "k" or "black" => Colors.Black,
            "w" or "white" => Colors.White,
            _ when color.Length == 7 && color[0] == '#' && color.Skip(1).All(Uri.IsHexDigit) => Color.FromHex(color),
            _ => Colors.Blue,
        };

        /// <summary>A scatter on a ScottPlot plot, which can be taken off it (matplotlib's <c>Artist.remove</c>).</summary>
        private sealed class Artist : IPlotArtist {
            /// <summary>The plot it is on.</summary>
            private readonly Plot _plot;

            /// <summary>The scatter.</summary>
            private readonly ScottPlot.Plottables.Scatter _scatter;

            /// <summary>A scatter on a plot.</summary>
            /// <param name="plot">The plot.</param>
            /// <param name="scatter">The scatter.</param>
            public Artist(Plot plot, ScottPlot.Plottables.Scatter scatter) {
                _plot = plot;
                _scatter = scatter;
            }

            /// <summary>Takes the scatter off its plot; removing it twice does nothing more.</summary>
            public void Remove() {
                lock (_plot.Sync)
                    _plot.Remove(_scatter);
            }
        }
    }
}
