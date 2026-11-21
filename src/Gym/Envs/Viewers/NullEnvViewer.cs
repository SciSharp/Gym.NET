#nullable enable
using System;
using System.Threading;
using System.Threading.Tasks;
using NumSharp;

namespace Gym.Environments {
    /// <summary>
    ///     A viewer that shows nothing: human mode without a window, for headless tests and servers. Since DR-20 nothing falls
    ///     back to it silently; pass <see cref="Factory"/> (as <c>viewer_factory</c>, or as <c>Gymnasium.HumanViewerFactory</c>)
    ///     to run human mode headless, where Gymnasium would need a display.
    /// </summary>
    /// <remarks>
    ///     Every env shares one instance, which holds no state: disposing or closing it does nothing. The env still draws
    ///     each human frame before handing it over, and still paces itself at <c>render_fps</c>. It implements
    ///     <see cref="IAsyncEnvViewer"/> too (DR-85), its asynchronous calls completing at once, so headless runs exercise an
    ///     asynchronous env's path as well.
    /// </remarks>
    public sealed class NullEnvViewer : IAsyncEnvViewer {
        /// <summary>The one instance; stateless, so sharing it between envs and threads is safe.</summary>
        private static readonly NullEnvViewer _singleton = new NullEnvViewer();

        /// <summary>
        ///     The factory to hand to envs for headless human mode. One delegate instance for the process: code that must know
        ///     whether a real window would open compares a factory with it by reference.
        /// </summary>
        public static readonly EnvViewerFactory Factory = Open;

        /// <summary>The factory's body: returns the shared instance.</summary>
        /// <param name="width">Ignored.</param>
        /// <param name="height">Ignored.</param>
        /// <param name="title">Ignored.</param>
        /// <returns>The shared viewer.</returns>
        public static IEnvViewer Open(int width, int height, string? title = null) => _singleton;

        /// <summary>The factory's body as 0.2's asynchronous factories were shaped: a completed task of the shared instance.</summary>
        /// <param name="width">Ignored.</param>
        /// <param name="height">Ignored.</param>
        /// <param name="title">Ignored.</param>
        /// <returns>A completed task holding the shared viewer.</returns>
        public static Task<IEnvViewer> Run(int width, int height, string? title = null) => Task.FromResult<IEnvViewer>(_singleton);

        /// <summary>Discards the frame; it is checked as the windowed viewers check it, so a headless run fails where a windowed one would.</summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; not read further.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        public void Render(NDArray frame) => Check(frame);

        /// <summary>Discards the frame at once, as <see cref="Render"/> does.</summary>
        /// <param name="frame">The frame, <c>(H, W, 3)</c>; not read further.</param>
        /// <param name="cancellationToken">Checked once, before anything else.</param>
        /// <returns>A completed task.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        /// <exception cref="OperationCanceledException">Thrown when <paramref name="cancellationToken"/> is already cancelled.</exception>
        public ValueTask RenderAsync(NDArray frame, CancellationToken cancellationToken = default) {
            cancellationToken.ThrowIfCancellationRequested();
            Check(frame);
            return default;
        }

        /// <summary>Does nothing: there is no window.</summary>
        public void Close() {
            // No window, nothing to close: the shared instance stays usable by every env.
        }

        /// <summary>Does nothing: there is no window.</summary>
        /// <returns>A completed task.</returns>
        public ValueTask CloseAsync() => default;

        /// <summary>Does nothing: the shared instance holds no resources.</summary>
        public void Dispose() {
            // The singleton is shared by every env; disposing one env's viewer must not affect the others.
        }

        /// <summary>The frame check every viewer makes: present and three-dimensional with three channels.</summary>
        /// <param name="frame">The frame.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="frame"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">Thrown for a frame that is not <c>(H, W, 3)</c>.</exception>
        private static void Check(NDArray frame) {
            if (frame is null)
                throw new ArgumentNullException(nameof(frame));
            if (frame.ndim != 3 || frame.shape[2] != 3)
                throw new ArgumentException($"A viewer shows (H, W, 3) frames, got shape ({string.Join(", ", frame.shape)}).", nameof(frame));
        }
    }
}
