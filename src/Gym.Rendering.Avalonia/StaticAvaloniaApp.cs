using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Threading;
using Gym.Environments;
using Image = SixLabors.ImageSharp.Image;
using AVImage = Avalonia.Controls.Image;
using Size = Avalonia.Size;

namespace Gym.Rendering.Avalonia;

public static class StaticAvaloniaApp {
    private static readonly SemaphoreSlim _syncRoot = new SemaphoreSlim(1, 1);
    private static Thread _thread;
    private static Application _app;
    private static ClassicDesktopStyleApplicationLifetime _lifetime;
    private static bool _initialized = false;

    public static AppBuilder BuildAvaloniaApp() {
        return AppBuilder.Configure<App>()
                         .UsePlatformDetect();
    }

    public static async Task<IEnvViewer> Run(int width, int height, string title = null) {
        await _syncRoot.WaitAsync();

        try {
            var resultCallback = new TaskCompletionSource<AvaloniaEnvViewer>();

            if (!_initialized) {
                var app = BuildAvaloniaApp();
                var appStartedEvent = new ManualResetEventSlim(false);
                _thread = new Thread(() => {
                    _lifetime = new ClassicDesktopStyleApplicationLifetime()
                    {
                        Args = Array.Empty<string>(),
                        ShutdownMode = ShutdownMode.OnExplicitShutdown
                    };

                    app.SetupWithLifetime(_lifetime);
                    _app = app.Instance;
                    _initialized = true;
                    appStartedEvent.Set();
                    _lifetime.Start(Array.Empty<string>());
                });
                _thread.IsBackground = true;
                _thread.Name = $"{nameof(AvaloniaEnvViewer)} {(string.IsNullOrEmpty(title) ? "" : $"-{title}")}";
                _thread.Start();

                appStartedEvent.Wait();
            }

            Dispatcher.UIThread.Post(() => {
                var viewer = new AvaloniaEnvViewer(width, height, title);
                viewer.Show();
                resultCallback.TrySetResult(viewer);
            }, DispatcherPriority.MaxValue);

            return await resultCallback.Task;
        } finally {
            _syncRoot.Release();
        }
    }

    public static void Shutdown() {
        // Avalonia lifetime is shared for the lifecycle of the host process across tests.
    }
}
