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

/// <summary>
///     The one Avalonia application every Gym.NET Avalonia window of the process lives on (<see cref="AvaloniaEnvViewer"/>,
///     <see cref="AvaloniaPlotFigure"/>): a UI thread of its own, started by the first window and kept until the process
///     ends, so windows open and close without restarting Avalonia.
/// </summary>
/// <remarks>
///     Footgun: Avalonia sets up one application per process. A program that already runs its own Avalonia application
///     cannot also open these windows (starting this one fails), and once started this one cannot be shut down and
///     started again, which is why <see cref="Shutdown"/> does nothing.
/// </remarks>
public static class StaticAvaloniaApp {
    /// <summary>Serializes starting the application and opening windows, so two first windows cannot start two applications.</summary>
    private static readonly SemaphoreSlim _syncRoot = new SemaphoreSlim(1, 1);

    /// <summary>The UI thread, a background thread: it never keeps the process alive.</summary>
    private static Thread _thread;

    /// <summary>The running application, once started.</summary>
    private static Application _app;

    /// <summary>The desktop lifetime, with explicit shutdown: closing the last window does not end the UI thread.</summary>
    private static ClassicDesktopStyleApplicationLifetime _lifetime;

    /// <summary>Whether the application runs; set on the UI thread before the starting caller is released.</summary>
    private static bool _initialized = false;

    /// <summary>A new builder of Gym.NET's <see cref="App"/> on the platform Avalonia detects (Win32, X11 or macOS).</summary>
    /// <remarks>Each call builds a new builder; setting one up after the shared application started fails, as Avalonia sets up one application per process.</remarks>
    /// <returns>The builder, not yet set up.</returns>
    public static AppBuilder BuildAvaloniaApp() {
        return AppBuilder.Configure<App>()
                         .UsePlatformDetect();
    }

    /// <summary>
    ///     Opens an <see cref="AvaloniaEnvViewer"/> window on the shared UI thread, starting the application first if no
    ///     window has yet, and completes once the window is shown.
    /// </summary>
    /// <remarks>
    ///     Footgun: the viewer is built and shown on the UI thread, so an exception there is raised on that thread, not
    ///     through the returned task, which then never completes.
    /// </remarks>
    /// <param name="width">The window's width in pixels.</param>
    /// <param name="height">The window's height in pixels.</param>
    /// <param name="title">The window's title, which also names the UI thread when this call starts it; <see langword="null"/> for none.</param>
    /// <returns>A task of the shown viewer.</returns>
    public static async Task<IEnvViewer> Run(int width, int height, string title = null) {
        await _syncRoot.WaitAsync();

        try {
            var resultCallback = new TaskCompletionSource<AvaloniaEnvViewer>();

            EnsureStarted(title);

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

    /// <summary>
    ///     Runs <paramref name="create"/> on Avalonia's UI thread, starting the shared application first if no viewer has
    ///     yet: how windows other than <see cref="AvaloniaEnvViewer"/> (a <see cref="AvaloniaPlotFigure"/>) join the one UI
    ///     thread every Avalonia window of the process lives on.
    /// </summary>
    /// <typeparam name="T">What <paramref name="create"/> returns.</typeparam>
    /// <param name="create">The work, typically building a window and its controls; it runs at the dispatcher's highest priority.</param>
    /// <param name="threadName">The UI thread's name suffix, when this call starts it.</param>
    /// <returns>A task of what <paramref name="create"/> returned, faulted with what it threw.</returns>
    internal static async Task<T> InvokeAsync<T>(Func<T> create, string threadName = null) {
        await _syncRoot.WaitAsync();
        try {
            EnsureStarted(threadName);
            return await Dispatcher.UIThread.InvokeAsync(create, DispatcherPriority.MaxValue);
        } finally {
            _syncRoot.Release();
        }
    }

    /// <summary>
    ///     Starts the shared Avalonia application on a background thread of its own, once per process, and waits until its
    ///     dispatcher runs; later calls return at once. The caller holds <see cref="_syncRoot"/>.
    /// </summary>
    /// <param name="title">The first window's title, which names the UI thread.</param>
    private static void EnsureStarted(string title) {
        if (_initialized)
            return;
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

    /// <summary>
    ///     Does nothing: the shared application lives as long as the process (see the class remarks), so closing it here
    ///     would leave later windows without a UI thread. Kept for callers that pair it with <see cref="Run"/>.
    /// </summary>
    public static void Shutdown() {
        // Avalonia lifetime is shared for the lifecycle of the host process across tests.
    }
}
