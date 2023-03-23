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
  
        private readonly ManualResetEventSlim ReadyResetEvent = new ManualResetEventSlim();
        private readonly ManualResetEventSlim RenderResetEvent = new ManualResetEventSlim();

        public AvaloniaEnvViewer() {
            InitializeComponent();
        }

        public AvaloniaEnvViewer(int width, int height, string title) : this() {
            Width = width;
            Height = height;
            Title = title;
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

            CloseEnvironment();
        }

        /// <summary>
        ///     A delegate that creates a <see cref="AvaloniaEnvViewer"/> based on given parameters.
        /// </summary>
        public static IEnvironmentViewerFactoryDelegate Factory => StaticAvaloniaApp.Run;

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
        }

        protected override void OnOpened(EventArgs e) {
            base.OnOpened(e);
            ReadyResetEvent.Set();
        }

        public void CloseEnvironment() {
            //invoke if required
            if (!Dispatcher.UIThread.CheckAccess()) {
                Dispatcher.UIThread.InvokeAsync(CloseEnvironment, DispatcherPriority.MaxValue);
                return;
            }

            Hide();
            //Dispose();
        }
    }
}