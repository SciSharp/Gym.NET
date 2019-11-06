using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments {
    public sealed class NullEnvViewer : IEnvViewer {
        private static readonly NullEnvViewer _singleton = new NullEnvViewer();

        public static readonly IEnvironmentViewerFactoryDelegate Factory = Run;
        public static IEnvViewer Run(int width, int height, string title = null) {
            return _singleton;
        }

        /// <summary>
        ///     Render the given <paramref name="img"/>.
        /// </summary>
        /// <param name="img">The image to render.</param>
        public void Render(Image img) {
            //null
        }

        /// <summary>
        ///     Close the rendering window.
        /// </summary>
        public void Close() {
            //null
        }

        public void Dispose() {
            //null
        }
    }
}