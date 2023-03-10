using System.Threading.Tasks;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments {
    public sealed class NullEnvViewer : IEnvViewer {
        private static readonly NullEnvViewer _singleton = new NullEnvViewer();

        public static readonly IEnvironmentViewerFactoryDelegate Factory = Run;
        public static Task<IEnvViewer> Run(int width, int height, string title = null) {
            return Task.FromResult((IEnvViewer) _singleton);
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
        public void CloseEnvironment() {
            //null
        }

        public void Dispose() {
            //null
        }
    }
}