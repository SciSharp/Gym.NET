using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments {
    public interface IEnvViewer : IDisposable {
        /// <summary>
        ///     Render the given <paramref name="img"/>.
        /// </summary>
        /// <param name="img">The image to render.</param>
        void Render(Image<Rgba32> img);

        /// <summary>
        ///     Close the rendering window.
        /// </summary>
        void Close();
    }
}