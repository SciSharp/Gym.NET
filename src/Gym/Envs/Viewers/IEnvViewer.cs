using System;
using System.Threading.Tasks;
using SixLabors.ImageSharp;

// The viewer contract lives in the core assembly so that Gymnasium.HumanViewerFactory (DR-20) can name it; it keeps the
// Gym.Environments namespace its implementations (the Avalonia and WinForms viewers) and users already import.
namespace Gym.Environments {
    /// <summary>
    ///     Creates the window an env renders into in <c>render_mode="human"</c> (Gymnasium's pygame display). Envs call it
    ///     once, on their first human render, with the frame size and a title; set one app-wide through
    ///     <c>Gymnasium.HumanViewerFactory</c>, or per env through its <c>viewer_factory</c> argument (DR-20).
    /// </summary>
    /// <param name="width">The window's client width, the frame's width in pixels.</param>
    /// <param name="height">The window's client height, the frame's height in pixels.</param>
    /// <param name="title">The window's title, or <see langword="null"/> for the viewer's default.</param>
    /// <returns>
    ///     The viewer, once its window is up. The env waits for it synchronously, so a factory must not need the calling
    ///     thread to complete (a UI framework runs its own thread, as <c>AvaloniaEnvViewer.Factory</c> does).
    /// </returns>
    /// <remarks>Usually a static <c>Factory</c> member of the viewer class: <c>AvaloniaEnvViewer.Factory</c>, <c>WinFormEnvViewer.Factory</c>, <c>NullEnvViewer.Factory</c>.</remarks>
    // TODO(async viewer path, DR-53): rendering is synchronous, so every caller blocks on this Task with
    // .GetAwaiter().GetResult() (CartPoleEnv, HumanWindow, HumanRendering and its vector twin, PygameDisplay, PlayPlot), which
    // deadlocks if a factory resumes on the calling thread's synchronization context. Support an async path that awaits the
    // factory instead: an asynchronous render entry through which human mode opens its window without blocking, the blocking
    // calls kept for the synchronous Render (R8 of docs/plans/CSHARP_CODE_REVIEW.md; the maintainer kept this contract).
    public delegate Task<IEnvViewer> IEnvironmentViewerFactoryDelegate(int width, int height, string? title = null);

    /// <summary>
    ///     A window that shows an env's human-mode frames. The env owns the viewer its factory made and closes it when the env
    ///     closes.
    /// </summary>
    public interface IEnvViewer : IDisposable {
        /// <summary>Shows <paramref name="img"/>, replacing the previous frame; the image stays the caller's.</summary>
        /// <param name="img">The frame.</param>
        void Render(Image img);

        /// <summary>Closes the window; rendering afterwards is up to the viewer (the built-in ones ignore it).</summary>
        void CloseEnvironment();
    }
}
