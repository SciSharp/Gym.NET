using System;
using System.ComponentModel;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using Gym.Environments;
using Gym.Threading;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;

namespace Gym.Rendering.WinForm {
    /// <summary>
    ///     A form with PictureBox that accepts <see cref="IImageCanvas"/> and renders it on it. Start <see cref="Viewer"/> by calling <see cref="Run"/>
    /// </summary>
    public partial class WinFormEnvViewer : Form, IEnvViewer {
        private int _lastSize = 0;
        private readonly ManualResetEventSlim _ready = new ManualResetEventSlim();

        /// <summary>
        ///     A delegate that creates a <see cref="WinFormEnvViewer"/> based on given parameters.
        /// </summary>
        public static IEnvironmentViewerFactoryDelegate Factory => Run;

        /// <summary>
        ///     Starts a <see cref="WinFormEnvViewer"/> in seperate thread.
        /// </summary>
        /// <param name="height">The height of the form</param>
        /// <param name="width">The width of the form</param>
        /// <param name="title">The title of the form, also mentioned in the thread name.</param>
        public static async Task<IEnvViewer> Run(int width, int height, string title = null) {
            var taskResult = new TaskCompletionSource<IEnvViewer>();
            var thread = new Thread(() => {
                var v = new WinFormEnvViewer(width + 12, height + 12, title);
                taskResult.SetResult(v);
                v.ShowDialog();
            });
            thread.Start();
            thread.Name = $"Viewer{(string.IsNullOrEmpty(title) ? "" : $"-{title}")}";

            var timeout = Task.Delay(10_000);
            if (await Task.WhenAny(taskResult.Task, timeout) == timeout)
                throw new Exception("Starting viewer timed out.");

            Debug.Assert(taskResult.Task.Result != null, "At this point viewer shouldn't be null.");

            return taskResult.Task.Result;
        }

        public WinFormEnvViewer(int width, int height, string title = null) {
            InitializeComponent();
            Height = height;
            Width = width;
            if (title != null)
                Text = title;
        }

        /// <summary>
        ///     Renders this canvas onto <see cref="PictureFrame"/>.
        /// </summary>
        /// <param name="canvas">Canvas painted from <see cref="NGraphics"/></param>
        public void Render(Image canvas) {
            if (InvokeRequired) {
                Invoke(new Action(() => Render(canvas)));
                return;
            }

            using (var ms = new MemoryStream(_lastSize)) {
                canvas.SaveAsBmp(ms);
                _lastSize = (int) ms.Length;
                Clear();
                PictureFrame.Image = new Bitmap(ms);
            }
        }

        /// <summary>
        ///     Renders this canvas onto <see cref="PictureFrame"/>.
        /// </summary>
        /// <param name="canvas">Canvas painted from <see cref="NGraphics"/></param>
        public void RenderAsync(Image canvas) {
            if (InvokeRequired) {
                BeginInvoke(new Action(() => RenderAsync(canvas)));
                return;
            }

            using (var ms = new MemoryStream(_lastSize)) {
                canvas.SaveAsBmp(ms);
                _lastSize = (int) ms.Length;
                Clear();
                PictureFrame.Image = new Bitmap(ms);
            }
        }

        /// <summary>Raises the <see cref="E:System.Windows.Forms.Form.Shown" /> event.</summary>
        /// <param name="e">A <see cref="T:System.EventArgs" /> that contains the event data. </param>
        protected override void OnShown(EventArgs e) {
            base.OnShown(e);
            _ready.Set();
        }

        public void Clear() {
            if (PictureFrame.Image != null) {
                var img = PictureFrame.Image;
                PictureFrame.Image = null;
                img.Dispose();
            }
        }

        public void CloseEnvironment() {
            if (InvokeRequired) {
                Invoke(new Action(CloseEnvironment));
                return;
            }

            Close();
            Dispose();
        }

        protected override void OnClosing(CancelEventArgs e) {
            base.OnClosing(e);
            PictureFrame.Image.TryDispose();
            _ready.TryDispose();
        }
    }
}