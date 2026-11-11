using System.Threading.Tasks;
using SixLabors.ImageSharp;

namespace Gym.Environments {
    /// <summary>
    ///     A viewer that shows nothing: human mode without a window, for headless tests and servers. Since DR-20 nothing falls
    ///     back to it silently; pass <see cref="Factory"/> (as <c>viewer_factory</c>, or as <c>Gymnasium.HumanViewerFactory</c>)
    ///     to run human mode headless, where Gymnasium would need a display.
    /// </summary>
    /// <remarks>
    ///     Every env shares one instance, which holds no state: disposing or closing it does nothing. The env still draws
    ///     each human frame before handing it over, and still paces itself at <c>render_fps</c>.
    /// </remarks>
    public sealed class NullEnvViewer : IEnvViewer {
        /// <summary>The one instance; stateless, so sharing it between envs and threads is safe.</summary>
        private static readonly NullEnvViewer _singleton = new NullEnvViewer();

        /// <summary>The factory to hand to envs for headless human mode.</summary>
        public static readonly IEnvironmentViewerFactoryDelegate Factory = Run;

        /// <summary>The factory's body: returns the shared instance, already complete.</summary>
        /// <param name="width">Ignored.</param>
        /// <param name="height">Ignored.</param>
        /// <param name="title">Ignored.</param>
        /// <returns>A completed task holding the shared viewer.</returns>
        public static Task<IEnvViewer> Run(int width, int height, string? title = null) {
            return Task.FromResult((IEnvViewer) _singleton);
        }

        /// <summary>Discards the frame.</summary>
        /// <param name="img">The frame; not read.</param>
        public void Render(Image img) {
            //null
        }

        /// <summary>Does nothing: there is no window.</summary>
        public void CloseEnvironment() {
            //null
        }

        /// <summary>Does nothing: the shared instance holds no resources.</summary>
        public void Dispose() {
            //null
        }
    }
}
