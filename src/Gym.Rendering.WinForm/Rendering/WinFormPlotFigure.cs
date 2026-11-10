#nullable enable
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Gym.Utils;
using NumSharp;
using ScottPlot.WinForms;

namespace Gym.Rendering.WinForm {
    /// <summary>
    ///     <see cref="PlayPlot"/>'s figure drawn by ScottPlot in a Windows Forms window (DR-39): one plot per row, stacked top to
    ///     bottom, with titles, ticks, axis labels and live updates, where the core's <see cref="PlotFigure"/> draws bare
    ///     markers. Pass <see cref="Subplots"/> to <see cref="PlayPlot"/>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Each figure runs its form on a background STA thread of its own with its own message loop, as
    ///     <see cref="WinFormEnvViewer"/> does; the form is created hidden and shown at the first <see cref="Pause"/>, as
    ///     matplotlib shows a figure there (and never in a non-interactive session). <see cref="PlayPlot"/> edits the axes
    ///     from the thread it plays on; the edits lock each plot's <c>Sync</c> object, which ScottPlot's renderer locks too.
    ///     </para>
    ///     <para>
    ///     Footgun: closing the window ends the figure's display; later pauses still wait their interval and draw nothing.
    ///     <see cref="Control"/> is ScottPlot's own control, for styling (axis labels, colors); change it on its form's
    ///     thread (<c>Control.Invoke</c>), or under a plot's <c>Sync</c> lock.
    ///     </para>
    /// </remarks>
    public sealed class WinFormPlotFigure : IPlotFigure {
        /// <summary>The plot area's width in pixels: matplotlib's 6.4 inches at 100 dots per inch, as <see cref="PlotFigure"/>.</summary>
        public const int Width = 640;

        /// <summary>The plot area's height in pixels: matplotlib's 4.8 inches at 100 dots per inch.</summary>
        public const int Height = 480;

        /// <summary>The form, created hidden on its own thread.</summary>
        private readonly Form _form;

        /// <summary>The axes, top to bottom, each over one of the control's plots.</summary>
        private readonly ScottPlotAxes[] _axes;

        /// <summary>Whether the form was shown; read and written on the form's thread.</summary>
        private bool _shown;

        /// <summary>
        ///     A figure of <paramref name="rows"/> plots (matplotlib's <c>plt.subplots(rows)</c>): starts the form's thread and
        ///     waits until the form and its control exist; the form stays hidden until the first <see cref="Pause"/>.
        /// </summary>
        /// <param name="rows">The number of plots, at least 1.</param>
        /// <param name="title">The form's title, or <see langword="null"/> for <c>PlayPlot</c>.</param>
        /// <exception cref="ValueError">Thrown for fewer than one row, with matplotlib's message: <c>Number of rows must be a positive integer, not 0</c>.</exception>
        /// <exception cref="TimeoutException">Thrown when the form's thread does not build the form within 10 seconds.</exception>
        /// <exception cref="Exception">What building the form or the control threw, on its thread.</exception>
        public WinFormPlotFigure(int rows, string? title = null) {
            if (rows < 1)
                throw new ValueError($"Number of rows must be a positive integer, not {rows}");
            var built = new TaskCompletionSource<(Form, FormsPlot, ScottPlotAxes[])>(TaskCreationOptions.RunContinuationsAsynchronously);
            var thread = new Thread(() => {
                Form form;
                FormsPlot control;
                ScottPlotAxes[] axes;
                try {
                    control = new FormsPlot { Dock = DockStyle.Fill };
                    // Laid out on the form's thread, before the control first renders.
                    axes = ScottPlotAxes.StackRows(control.Multiplot, rows);
                    form = new Form { ClientSize = new System.Drawing.Size(Width, Height), Text = title ?? "PlayPlot" };
                    form.Controls.Add(control);
                    // The handle must exist before another thread can Invoke on the form; creating it does not show it.
                    _ = form.Handle;
                } catch (Exception e) {
                    built.SetException(e);
                    return;
                }

                built.SetResult((form, control, axes));
                // The message loop of this thread, running until the process ends (a background thread never holds it).
                Application.Run(new ApplicationContext());
            }) {
                IsBackground = true,
                Name = $"PlotFigure{(string.IsNullOrEmpty(title) ? "" : $"-{title}")}",
            };
            thread.SetApartmentState(ApartmentState.STA);
            thread.Start();
            if (!built.Task.Wait(TimeSpan.FromSeconds(10)))
                throw new TimeoutException("The plot figure's form did not start within 10 seconds.");
            (_form, Control, _axes) = built.Task.GetAwaiter().GetResult();
        }

        /// <summary>ScottPlot's control in the form, for styling; touch it on the form's thread.</summary>
        public FormsPlot Control { get; }

        /// <summary>The axes, top to bottom.</summary>
        public IReadOnlyList<IPlotAxes> Axes => _axes;

        /// <summary>The maker <see cref="PlayPlot"/> takes: <c>new PlayPlot(callback, horizon, names, WinFormPlotFigure.Subplots())</c>.</summary>
        /// <param name="title">The forms' title, or <see langword="null"/> for <c>PlayPlot</c>.</param>
        /// <returns>The maker.</returns>
        public static PlayPlot.PlotSubplots Subplots(string? title = null) => rows => new WinFormPlotFigure(rows, title);

        /// <summary>
        ///     Shows the figure as it is now (<c>plt.pause</c>): shows the form the first time (in an interactive session),
        ///     redraws it, then waits <paramref name="interval"/> seconds on the calling thread while the form keeps running.
        /// </summary>
        /// <param name="interval">The pause in seconds; under a millisecond only yields the thread.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a negative or non-finite interval.</exception>
        public void Pause(double interval) {
            if (!(interval >= 0) || double.IsInfinity(interval))
                throw new ArgumentOutOfRangeException(nameof(interval), interval, "The pause must be a finite number of seconds, not negative.");
            // A form the user closed is disposed: the figure simply stops drawing.
            if (!_form.IsDisposed) {
                try {
                    _form.Invoke(() => {
                        if (!_shown && SystemInformation.UserInteractive) {
                            _form.Show();
                            _shown = true;
                        }

                        Control.Refresh();
                    });
                } catch (ObjectDisposedException) {
                    // Closed between the check and the call.
                } catch (InvalidOperationException) when (_form.IsDisposed) {
                    // Its handle went with the close.
                }
            }

            Thread.Sleep(TimeSpan.FromSeconds(interval));
        }
    }
}
