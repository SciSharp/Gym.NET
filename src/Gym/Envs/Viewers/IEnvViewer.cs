#nullable enable
using System;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;

// The viewer contract lives in the core assembly so that Gymnasium.HumanViewerFactory (DR-20) can name it; it keeps the
// Gym.Environments namespace its implementations (the Avalonia and WinForms viewers) and users already import.
namespace Gym.Environments {
    /// <summary>
    ///     Creates the window an env renders into in <c>render_mode="human"</c> (Gymnasium's pygame display), synchronously
    ///     (DR-85): envs call it once, on their first human render, with the frame size and a title, and use the viewer it
    ///     returns at once. Set one app-wide through <c>Gymnasium.HumanViewerFactory</c>, or per env through its
    ///     <c>viewer_factory</c> argument (DR-20).
    /// </summary>
    /// <param name="width">The window's client width, the frame's width in pixels.</param>
    /// <param name="height">The window's client height, the frame's height in pixels.</param>
    /// <param name="title">The window's title, or <see langword="null"/> for the viewer's default.</param>
    /// <returns>
    ///     The viewer, its window up: an <see cref="IEnvViewer"/>, or an <see cref="IAsyncEnvViewer"/> when the viewer also
    ///     offers the asynchronous path (one factory creates either).
    /// </returns>
    /// <remarks>
    ///     <para>
    ///     Usually a static <c>Factory</c> member of the viewer class: <c>AvaloniaEnvViewer.Factory</c>,
    ///     <c>WinFormEnvViewer.Factory</c>, <c>NullEnvViewer.Factory</c>.
    ///     </para>
    ///     <para>
    ///     Footgun: the factory runs on the env's thread and returns only once the window exists. A UI framework's factory
    ///     must therefore hand the creation to its own UI thread and wait for it there (the built-in ones do, with the
    ///     framework's synchronous invoke), never block that UI thread on the env's thread, or the two deadlock.
    ///     </para>
    /// </remarks>
    public delegate IEnvViewer EnvViewerFactory(int width, int height, string? title = null);

    /// <summary>
    ///     A window that shows an env's human-mode frames, synchronously (DR-85): the path the envs' human mode uses inside
    ///     <c>Step</c> and <c>Reset</c>, which every viewer has. The env owns the viewer its factory made and closes it when
    ///     the env closes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A frame is an <c>(H, W, 3)</c> uint8 array, rows top to bottom, what an env's <c>rgb_array</c> mode returns: the
    ///     contract carries no imaging library (ImageSharp left it with DR-85). A viewer that can also render without blocking
    ///     implements <see cref="IAsyncEnvViewer"/>, which an inherently asynchronous env awaits.
    ///     </para>
    ///     <para>
    ///     Footgun for implementers deriving from a UI framework's window: <c>Close()</c> usually exists on the window
    ///     already (WinForms' <c>Form.Close</c>, Avalonia's <c>Window.Close</c>) and would implement this interface's
    ///     <see cref="Close"/> silently; implement it explicitly (<c>void IEnvViewer.Close()</c>) to keep control of what
    ///     closing means.
    ///     </para>
    /// </remarks>
    public interface IEnvViewer : IDisposable {
        /// <summary>
        ///     Shows <paramref name="frame"/>, replacing the previous frame, and returns once the viewer has its own copy: the
        ///     caller may dispose or reuse the frame right after.
        /// </summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c> uint8; it stays the caller's.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/> (the built-in viewers).</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c> (the built-in viewers).</exception>
        void Render(NDArray frame);

        /// <summary>Closes the window; what rendering afterwards does is the viewer's (the built-in ones ignore it or reopen nothing).</summary>
        void Close();

        /// <summary>0.2's name for <see cref="Close"/>, kept so 0.2 code compiles until 2.0 (DR-58, DR-85).</summary>
        [Obsolete("0.2's name: write Close(). Removed in 2.0.")]
        void CloseEnvironment() => Close();
    }

    /// <summary>
    ///     A viewer that also renders asynchronously (DR-85): <see cref="RenderAsync"/> and <see cref="CloseAsync"/> complete
    ///     once the UI has taken the frame or closed the window, without blocking the caller's thread. The same factory
    ///     creates it as any viewer; an inherently asynchronous env checks for this interface and awaits it, and falls back to
    ///     the synchronous <see cref="IEnvViewer.Render"/> otherwise (<see cref="EnvViewerExtensions.RenderAsync"/> does both).
    /// </summary>
    public interface IAsyncEnvViewer : IEnvViewer {
        /// <summary>Shows <paramref name="frame"/> without blocking; completes once the viewer has its own copy of it.</summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c> uint8; it stays the caller's, and must not change before the task completes.</param>
        /// <param name="cancellationToken">Cancels the wait for the UI; a cancelled render may still show the frame.</param>
        /// <returns>The pending render.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/> (the built-in viewers).</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c> (the built-in viewers).</exception>
        /// <exception cref="OperationCanceledException">Thrown through the task when <paramref name="cancellationToken"/> is cancelled first.</exception>
        ValueTask RenderAsync(NDArray frame, CancellationToken cancellationToken = default);

        /// <summary>Closes the window without blocking; completes once it is closed.</summary>
        /// <returns>The pending close.</returns>
        ValueTask CloseAsync();
    }

    /// <summary>
    ///     The asynchronous path for any <see cref="IEnvViewer"/> (DR-85): awaits an <see cref="IAsyncEnvViewer"/>'s own
    ///     methods, and runs a synchronous viewer's on the caller's thread, so an asynchronous env writes one call for both.
    /// </summary>
    public static class EnvViewerExtensions {
        /// <summary>
        ///     Shows <paramref name="frame"/>: <see cref="IAsyncEnvViewer.RenderAsync"/> when the viewer has it, else
        ///     <see cref="IEnvViewer.Render"/> on the calling thread, which completes synchronously (no thread-pool hop).
        /// </summary>
        /// <param name="viewer">The viewer.</param>
        /// <param name="frame">The frame, <c>(H, W, 3)</c> uint8; it stays the caller's.</param>
        /// <param name="cancellationToken">Cancels before a synchronous render, or the wait of an asynchronous one.</param>
        /// <returns>The render: pending for an asynchronous viewer, completed for a synchronous one.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="viewer"/> is <see langword="null"/>.</exception>
        /// <exception cref="OperationCanceledException">Thrown, or through the task, when <paramref name="cancellationToken"/> is cancelled first.</exception>
        /// <exception cref="ArgumentException">Thrown, or through the task, by the viewer for a frame it cannot show.</exception>
        public static ValueTask RenderAsync(this IEnvViewer viewer, NDArray frame, CancellationToken cancellationToken = default) {
            if (viewer is null)
                throw new ArgumentNullException(nameof(viewer));
            if (viewer is IAsyncEnvViewer asynchronous)
                return asynchronous.RenderAsync(frame, cancellationToken);
            // A synchronous viewer renders here, on the caller's thread, as StepAsync steps there (never through Task.Run).
            cancellationToken.ThrowIfCancellationRequested();
            viewer.Render(frame);
            return default;
        }

        /// <summary>Closes the window: <see cref="IAsyncEnvViewer.CloseAsync"/> when the viewer has it, else <see cref="IEnvViewer.Close"/> on the calling thread.</summary>
        /// <param name="viewer">The viewer.</param>
        /// <returns>The close: pending for an asynchronous viewer, completed for a synchronous one.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="viewer"/> is <see langword="null"/>.</exception>
        public static ValueTask CloseAsync(this IEnvViewer viewer) {
            if (viewer is null)
                throw new ArgumentNullException(nameof(viewer));
            if (viewer is IAsyncEnvViewer asynchronous)
                return asynchronous.CloseAsync();
            viewer.Close();
            return default;
        }
    }
}
