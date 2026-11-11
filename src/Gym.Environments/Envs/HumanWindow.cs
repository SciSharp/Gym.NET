#nullable enable
using System;
using Gym.Rendering;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments.Envs {
    /// <summary>
    ///     The human-mode window of an env, what <c>pygame.display.set_mode</c> opens: created on the first human render
    ///     from the env's <c>viewer_factory</c> or else <c>Gymnasium.HumanViewerFactory</c> (DR-20), shown a frame per
    ///     render, and shut down by the env's <c>close</c>.
    /// </summary>
    /// <remarks>
    ///     The window's size is the one passed when it opens, as <c>set_mode((w, h))</c> takes the env's screen size at its
    ///     first human render, not at construction. Creation takes a lock, since a viewer factory may start a UI thread and
    ///     two renders must not open two windows. A window that <see cref="Close"/> shut is reopened by the next
    ///     <see cref="Show"/>; pygame would raise <c>display Surface quit</c> there instead. CartPole keeps the same logic
    ///     inline.
    /// </remarks>
    internal sealed class HumanWindow {
        /// <summary>Guards <see cref="_viewer"/>'s creation.</summary>
        private readonly object _gate = new();

        /// <summary>The window's title.</summary>
        private readonly string _title;

        /// <summary>The env's own factory, or <see langword="null"/> to read <c>Gymnasium.HumanViewerFactory</c> when the window opens.</summary>
        private IEnvironmentViewerFactoryDelegate? _factory;

        /// <summary>The open window, or <see langword="null"/>.</summary>
        private IEnvViewer? _viewer;

        /// <summary>A window with a title, not opened yet.</summary>
        /// <param name="title">The title.</param>
        /// <param name="factory">The env's <c>viewer_factory</c>, or <see langword="null"/>.</param>
        public HumanWindow(string title, IEnvironmentViewerFactoryDelegate? factory) {
            _title = title;
            _factory = factory;
        }

        /// <summary>Opens the window at a size when it is not open (Gymnasium's <c>pygame.display.set_mode((width, height))</c>).</summary>
        /// <param name="width">The width in pixels.</param>
        /// <param name="height">The height in pixels.</param>
        /// <exception cref="Exceptions.DependencyNotInstalled">Thrown when no factory is set, where Gymnasium raises it for a missing pygame.</exception>
        public void Ensure(int width, int height) {
            if (_viewer != null)
                return;
            lock (_gate) {
                if (_viewer != null)
                    return;
                _factory = global::Gym.Gymnasium.ResolveHumanViewerFactory(_factory);
                // The factory may start a UI thread and complete asynchronously; render calls are synchronous, as pygame's are.
                // TODO(async viewer path, DR-53): blocks on the factory; see IEnvironmentViewerFactoryDelegate.
                _viewer = _factory(width, height, _title).GetAwaiter().GetResult();
            }
        }

        /// <summary><c>pygame.display.flip()</c>: shows the surface's pixels, opening the window at the surface's size first if needed.</summary>
        /// <param name="screen">The frame; the viewer copies it synchronously.</param>
        /// <exception cref="Exceptions.DependencyNotInstalled">Thrown when the window must open and no factory is set.</exception>
        public void Show(Surface screen) {
            Ensure(screen.Width, screen.Height);
            // The viewer copies the frame before Render returns, so the image is released right after.
            using Image<Rgb24> frame = Image.LoadPixelData<Rgb24>(screen.ToRgbBytes(), screen.Width, screen.Height);
            _viewer!.Render(frame);
        }

        /// <summary><c>pygame.display.quit()</c>: shuts the window down if it is open. Safe to call twice.</summary>
        public void Close() {
            lock (_gate) {
                if (_viewer is null)
                    return;
                _viewer.CloseEnvironment();
                _viewer = null;
            }
        }
    }
}
