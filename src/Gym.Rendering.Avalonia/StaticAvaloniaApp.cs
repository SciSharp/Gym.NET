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

    public static AppBuilder BuildAvaloniaApp() {
        return AppBuilder.Configure<App>()
                         .UsePlatformDetect();
        //.LogToTrace();
    }

    public static async Task<IEnvViewer> Run(int width, int height, string title = null) {
        // ReSharper disable once MethodHasAsyncOverload
        if (!_syncRoot.Wait(0)) //async lock
            await _syncRoot.WaitAsync();

        try {
            var resultCallback = new TaskCompletionSource<AvaloniaEnvViewer>();
            if (_app != null) {
                _ = Dispatcher.UIThread.InvokeAsync(() => {
                    var viewer = new AvaloniaEnvViewer(width, height, title);
                    resultCallback.SetResult(viewer);
                    _app.Run(viewer);
                }, DispatcherPriority.MaxValue);

                return await resultCallback.Task;
            }

            var app = BuildAvaloniaApp();
            _thread = new Thread(() => {
                _lifetime = new ClassicDesktopStyleApplicationLifetime()
                {
                    Args = Array.Empty<string>(),
                    ShutdownMode = ShutdownMode.OnExplicitShutdown
                };
                
                app.SetupWithLifetime(_lifetime);
                var viewer = new AvaloniaEnvViewer(width, height, title);
                resultCallback.TrySetResult(viewer);
                _lifetime.Start(Array.Empty<string>());
                _app = app.Instance;
                _app.Run(viewer);
            });
            _thread.Start();
            _thread.Name = $"{nameof(AvaloniaEnvViewer)} {(string.IsNullOrEmpty(title) ? "" : $"-{title}")}";
            return await resultCallback.Task;
        } finally {
            _syncRoot.Release();
        }
    }

    public static void Shutdown() {
        if (_lifetime != null && _lifetime.TryShutdown()) {
            _lifetime = null;
            _app = null;
            _thread = null;
        }
    }
}