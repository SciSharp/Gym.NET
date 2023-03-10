using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Gym.Environments;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Image = SixLabors.ImageSharp.Image;
using AVImage = Avalonia.Controls.Image;
using Size = Avalonia.Size;

namespace Gym.Rendering.Avalonia {
    public class AvaloniaEnvViewer : Window, IEnvViewer {
        private static int _width;
        private static int _height;
        private static string _title;
        private static AvaloniaEnvViewer _viewer;
        private static readonly ManualResetEventSlim ReadyResetEvent = new ManualResetEventSlim();
        private static readonly ManualResetEventSlim RenderResetEvent = new ManualResetEventSlim();

        public AvaloniaEnvViewer() {
            InitializeComponent();
        }

        private void InitializeComponent() {
            AvaloniaXamlLoader.Load(this);
        }

        public void Dispose() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(Dispose, DispatcherPriority.MaxValue);
                return;
            }

            Close();
        }

        /// <summary>
        ///     A delegate that creates a <see cref="AvaloniaEnvViewer"/> based on given parameters.
        /// </summary>
        public static IEnvironmentViewerFactoryDelegate Factory => Run;

        public static IEnvViewer Run(int width, int height, string title = null) {
            _width = width;
            _height = height;
            _title = title;

            var thread = new Thread(() => { BuildAvaloniaApp().Start(BuildViewer, Array.Empty<string>()); });
            thread.Start();
            thread.Name = $"{nameof(AvaloniaEnvViewer)} {(string.IsNullOrEmpty(title) ? "" : $"-{title}")}";

            if (!ReadyResetEvent.Wait(10_000))
                throw new Exception("Starting viewer timed out.");

            Debug.Assert(_viewer != null, "At this point viewer shouldn't be null.");

            return _viewer;
        }

        public void Render(Image img) {
            using (var ms = new MemoryStream(img.Height * img.Width * img.PixelType.BitsPerPixel / 8)) {
                img.SaveAsBmp(ms);
                ms.Seek(0, SeekOrigin.Begin);
                var bitmap = new Bitmap(ms);
                if (Dispatcher.UIThread.CheckAccess()) {
                    Content = new AVImage {
                        Source = bitmap
                    };
                } else {
                    RenderResetEvent.Reset();
                    Dispatcher.UIThread.InvokeAsync(() => {
                        Content = new AVImage {
                            Source = bitmap
                        };
                        RenderResetEvent.Set();
                    }, DispatcherPriority.MaxValue);
                    if (!RenderResetEvent.Wait(4_000))
                        throw new Exception("Rendering timed out.");
                }
            }

        private static void BuildViewer(Application app, string[] args) {
            _viewer = new AvaloniaEnvViewer {
                Title = _title,
                ClientSize = new Size(_width, _height)
            };
            app.Run(_viewer);
        }

        protected override void OnOpened(EventArgs e) {
            base.OnOpened(e);
            ReadyResetEvent.Set();
        }

        public void Close() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(Close, DispatcherPriority.MaxValue);
                return;
            }

            base.Close();
        }

        [Obsolete]
        protected static AppBuilder BuildAvaloniaApp() =>
            AppBuilder.Configure<App>()
                      .UsePlatformDetect()
                      .LogToDebug();
    }
}