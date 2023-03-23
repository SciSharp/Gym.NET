using System;
using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments {
    /// <summary>
    ///     A delegate that creates a <see cref="IEnvViewer"/> from given parameters.
    /// </summary>
    /// <param name="width">The width of the rendering window.</param>
    /// <param name="height">The height of the rendering window.</param>
    /// <param name="title">The title of the rendering window.</param>
    /// <returns>A new environment viewer created based on parameters passed.</returns>
    /// <remarks>Usually provided as a static method in the rendering class, e.g. <see cref="WinFormsViewer.Run"/><br></br>The method should run the environment in a separate background worker / thread.</remarks>
    public delegate Task<IEnvViewer> IEnvironmentViewerFactoryDelegate(int width, int height, string title = null);

    /// <summary>
    ///     Represents a graphics engine that is able to <see cref="Render"/> images.
    /// </summary>
    public interface IEnvViewer : IDisposable {
        /// <summary>
        ///     Render the given <paramref name="img"/>.
        /// </summary>
        /// <param name="img">The image to render.</param>
        void Render(Image img);

        /// <summary>
        ///     Close the rendering window.
        /// </summary>
        void CloseEnvironment();
    }
}