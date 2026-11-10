#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using ScottPlot;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     The ScottPlot figures of the viewer packages (DR-39): the axes adapter both compile, checked headless on plain
    ///     ScottPlot plots (titles, blue marker-only scatters, removal, the x range, the figures' row layout, a PlayPlot run,
    ///     a rendered image), and
    ///     each package's window figure driven by <see cref="PlayPlot"/>, which shows windows: those cells end in
    ///     <c>AvaloniaEnv</c> and <c>WinFormEnv</c>, so the headless gate leaves them out and they run on a desktop.
    /// </summary>
    [TestClass]
    public class ScottPlotFigureTests {
        /// <summary>The adapter on a plain plot: the title, a blue scatter of markers only with the given points, its removal, the x range, and matplotlib's size refusal.</summary>
        [TestMethod]
        public void Axes_DrawTitlesScattersAndRanges_OnAPlot() {
            var plot = new ScottPlot.Plot();
            IPlotAxes axes = NewAxes(plot);
            axes.SetTitle("reward");
            Assert.AreEqual("reward", plot.Axes.Title.Label.Text);

            IPlotArtist artist = axes.Scatter(new[] { 1.0, 2.0, 3.0 }, new[] { 4.0, 5.0, 6.0 }, "blue");
            var scatter = (ScottPlot.Plottables.Scatter)plot.GetPlottables().Single();
            Assert.AreEqual("#0000FF", scatter.Color.ToHex());
            Assert.AreEqual(0f, scatter.LineWidth, "Markers only, as matplotlib's scatter.");
            CollectionAssert.AreEqual(new[] { (1.0, 4.0), (2.0, 5.0), (3.0, 6.0) }, scatter.Data.GetScatterPoints().Select(c => (c.X, c.Y)).ToArray());

            axes.SetXLim(3, 10);
            Assert.AreEqual(3, plot.Axes.GetLimits().Left);
            Assert.AreEqual(10, plot.Axes.GetLimits().Right);

            artist.Remove();
            artist.Remove();
            Assert.AreEqual(0, plot.GetPlottables().Count());
            Assert.AreEqual("x and y must be the same size", Assert.ThrowsExactly<ValueError>(() => axes.Scatter(new[] { 1.0 }, Array.Empty<double>(), "blue")).Message);

            axes.Scatter(new[] { 1.0 }, new[] { 2.0 }, "#ff0000");
            Assert.AreEqual("#FF0000", ((ScottPlot.Plottables.Scatter)plot.GetPlottables().Single()).Color.ToHex(), "A #rrggbb color is taken as given.");
            axes.Scatter(new[] { 1.0 }, new[] { 2.0 }, "#zzzzzz");
            Assert.AreEqual("#0000FF", ((ScottPlot.Plottables.Scatter)plot.GetPlottables().Last()).Color.ToHex(),
                "Malformed hex digits fall back to blue, not to ScottPlot's transparent black.");
        }

        /// <summary>
        ///     The figures' row layout on a plain multiplot, headless: ScottPlot's rows layout, one distinct plot per row with
        ///     the multiplot's own plot kept first, for one to four rows, and fewer than one row refused. ScottPlot's
        ///     <c>AddPlots</c> sets the total number of plots, so asking it for one fewer left a two-row figure short.
        /// </summary>
        [TestMethod]
        public void StackRows_GivesOnePlotPerRow_TheMultiplotsOwnFirst() {
            for (int rows = 1; rows <= 4; rows++) {
                var own = new ScottPlot.Plot();
                var multiplot = new ScottPlot.Multiplot(own);
                IReadOnlyList<IPlotAxes> axes = StackRows(multiplot, rows);
                Assert.HasCount(rows, axes);
                Assert.IsInstanceOfType<ScottPlot.MultiplotLayouts.Rows>(multiplot.Layout);
                ScottPlot.Plot[] plots = axes.Select(PlotOf).ToArray();
                CollectionAssert.AreEqual(multiplot.GetPlots(), plots, $"{rows} rows: the multiplot's plots, in order.");
                Assert.AreSame(own, plots[0], "The control's own plot stays the first row.");
                Assert.AreEqual(rows, plots.Distinct().Count(), "One plot per row.");
            }

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => StackRows(new ScottPlot.Multiplot(), 0));
        }

        /// <summary>
        ///     A <see cref="PlayPlot"/> run over two adapted plots, headless: after each callback every plot holds one scatter
        ///     of the last points within the horizon, the x range follows the steps, and the plots render as images.
        /// </summary>
        [TestMethod]
        public void PlayPlot_OverAdaptedPlots_KeepsTheHorizonsPoints() {
            var plots = new[] { new ScottPlot.Plot(), new ScottPlot.Plot() };
            var figure = new HeadlessFigure(plots.Select(NewAxes).ToArray());
            var playPlot = new PlayPlot((_, _, _, rew, _, _, _) => new[] { rew, -rew }, 3, new[] { "reward", "negated" }, _ => figure);
            for (int step = 1; step <= 5; step++)
                playPlot.Callback(null, null, null, step, false, false, new Info());

            Assert.AreEqual(5, figure.Pauses);
            CollectionAssert.AreEqual(new[] { 3.0, 4.0, 5.0 }, Points(plots[0]).Select(p => p.Y).ToArray());
            CollectionAssert.AreEqual(new[] { -3.0, -4.0, -5.0 }, Points(plots[1]).Select(p => p.Y).ToArray());
            Assert.AreEqual("negated", plots[1].Axes.Title.Label.Text);
            Assert.AreEqual(2, plots[0].Axes.GetLimits().Left);
            Assert.AreEqual(5, plots[0].Axes.GetLimits().Right);
            Assert.IsTrue(CountBluePixels(plots[0].GetImage(320, 240).GetImageBytes(ScottPlot.ImageFormat.Png)) > 0, "The markers are drawn.");
        }

        /// <summary>The Avalonia figure under PlayPlot: its window shows the live plot, and the control renders the markers. Opens a window.</summary>
        [TestMethod]
        public void Figure_ShowsPlayPlotLive_AvaloniaEnv() {
            var figure = (AvaloniaPlotFigure)AvaloniaPlotFigure.Subplots("ScottPlot under Avalonia")(2);
            AssertPlaysLive(figure, () => Avalonia.Threading.Dispatcher.UIThread.Invoke(() =>
                figure.Control.Multiplot.GetPlot(0).GetImage(320, 240).GetImageBytes(ScottPlot.ImageFormat.Png)));
        }

        /// <summary>The Windows Forms figure under PlayPlot: its form shows the live plot, and the control renders the markers. Opens a window.</summary>
        [TestMethod]
        public void Figure_ShowsPlayPlotLive_WinFormEnv() {
            var figure = (WinFormPlotFigure)WinFormPlotFigure.Subplots("ScottPlot under Windows Forms")(2);
            AssertPlaysLive(figure, () => (byte[])figure.Control.Invoke(() =>
                figure.Control.Multiplot.GetPlot(0).GetImage(320, 240).GetImageBytes(ScottPlot.ImageFormat.Png)));
        }

        /// <summary>Drives a window figure through 30 callbacks with short pauses, then checks a rendered plot has its markers.</summary>
        /// <param name="figure">The figure, two rows.</param>
        /// <param name="render">Renders the first plot to PNG bytes on the figure's UI thread.</param>
        /// <exception cref="AssertFailedException">Thrown when the figure has not two axes or the plot shows no marker.</exception>
        private static void AssertPlaysLive(IPlotFigure figure, Func<byte[]> render) {
            Assert.AreEqual(2, figure.Axes.Count);
            var playPlot = new PlayPlot((_, _, _, rew, _, _, _) => new[] { Math.Sin(rew / 3), Math.Cos(rew / 3) }, 20, new[] { "sin", "cos" }, _ => figure);
            for (int step = 1; step <= 30; step++) {
                playPlot.Callback(null, null, null, step, false, false, new Info());
                figure.Pause(0.02);
            }

            Assert.IsTrue(CountBluePixels(render()) > 0, "The live plot shows its markers.");
        }

        /// <summary>
        ///     The internal ScottPlot adapter type both viewer packages compile, taken from the Windows Forms package's copy (the
        ///     two copies share a full name, so neither package can expose it to the tests without an ambiguity).
        /// </summary>
        /// <returns>The adapter's type.</returns>
        /// <exception cref="TypeLoadException">Thrown when the package no longer compiles the adapter.</exception>
        private static Type AdapterType() => typeof(WinFormPlotFigure).Assembly.GetType("Gym.Utils.ScottPlotAxes", throwOnError: true)!;

        /// <summary>The internal ScottPlot adapter both viewer packages compile, built by reflection over a plot.</summary>
        /// <param name="plot">The plot to adapt.</param>
        /// <returns>The axes.</returns>
        private static IPlotAxes NewAxes(ScottPlot.Plot plot) => (IPlotAxes)Activator.CreateInstance(AdapterType(), plot)!;

        /// <summary>The figures' row layout (the adapter's <c>StackRows</c>), called by reflection.</summary>
        /// <param name="multiplot">The multiplot to lay out.</param>
        /// <param name="rows">The number of rows.</param>
        /// <returns>The axes, top to bottom.</returns>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for fewer than one row, unwrapped from the reflection call.</exception>
        private static IReadOnlyList<IPlotAxes> StackRows(ScottPlot.IMultiplot multiplot, int rows) {
            MethodInfo stackRows = AdapterType().GetMethod("StackRows", BindingFlags.Public | BindingFlags.Static)!;
            try {
                return (IPlotAxes[])stackRows.Invoke(null, new object[] { multiplot, rows })!;
            } catch (TargetInvocationException e) when (e.InnerException is not null) {
                // Rethrown as the adapter threw it, so the test asserts the exception a direct call would see.
                ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                throw;
            }
        }

        /// <summary>The ScottPlot plot an adapter draws on, read by reflection.</summary>
        /// <param name="axes">The adapter.</param>
        /// <returns>Its plot.</returns>
        private static ScottPlot.Plot PlotOf(IPlotAxes axes) => (ScottPlot.Plot)axes.GetType().GetProperty("Plot")!.GetValue(axes)!;

        /// <summary>The points of a plot's one scatter.</summary>
        /// <param name="plot">The plot.</param>
        /// <returns>Its points, in order.</returns>
        private static IReadOnlyList<ScottPlot.Coordinates> Points(ScottPlot.Plot plot) =>
            ((ScottPlot.Plottables.Scatter)plot.GetPlottables().Single()).Data.GetScatterPoints();

        /// <summary>Counts the pure-blue pixels of a PNG (ScottPlot draws the markers in matplotlib's blue, which nothing else on a plot is).</summary>
        /// <param name="png">The image.</param>
        /// <returns>The count.</returns>
        private static int CountBluePixels(byte[] png) {
            using SixLabors.ImageSharp.Image<Rgba32> image = SixLabors.ImageSharp.Image.Load<Rgba32>(png);
            int count = 0;
            for (int y = 0; y < image.Height; y++) {
                for (int x = 0; x < image.Width; x++) {
                    Rgba32 pixel = image[x, y];
                    if (pixel.R < 40 && pixel.G < 40 && pixel.B > 200)
                        count++;
                }
            }

            return count;
        }

        /// <summary>A figure that only holds axes and counts pauses: PlayPlot's figure without a window.</summary>
        private sealed class HeadlessFigure : IPlotFigure {
            /// <summary>A figure of the given axes.</summary>
            /// <param name="axes">The axes, top to bottom.</param>
            public HeadlessFigure(IReadOnlyList<IPlotAxes> axes) {
                Axes = axes;
            }

            /// <summary>The axes, top to bottom.</summary>
            public IReadOnlyList<IPlotAxes> Axes { get; }

            /// <summary>How many pauses were asked for.</summary>
            public int Pauses { get; private set; }

            /// <summary>Counts the pause and returns at once.</summary>
            /// <param name="interval">Ignored.</param>
            public void Pause(double interval) => Pauses++;
        }
    }
}
