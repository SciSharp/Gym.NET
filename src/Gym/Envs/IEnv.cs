using System.Threading.Tasks;
using Gym.Collections;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// ReSharper disable once CheckNamespace
namespace Gym.Envs {
    public interface IEnv {
        Dict Metadata { get; set; }
        (float From, float To) RewardRange { get; set; }
        Space ActionSpace { get; set; }
        Space ObservationSpace { get; set; }
        NDArray Reset();
        Step Step(object action);
        Task<Step> StepAsync(object action);
        Image Render(string mode = "human");
        void CloseEnvironment();

        /// <summary>
        ///     Reseeds the env's generator, so the episodes that follow are reproducible. Which other generators this also
        ///     reseeds (for example an action space created with the same generator) is up to the env.
        /// </summary>
        /// <param name="seed">
        ///     A NumPy seed in <c>[0, 2**32 - 1]</c>; <see langword="long"/> because seeds above <see cref="int.MaxValue"/>
        ///     are valid.
        /// </param>
        /// <exception cref="NumSharp.ValueError">Thrown when <paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c>, with NumPy's message.</exception>
        void Seed(long seed);
    }
}