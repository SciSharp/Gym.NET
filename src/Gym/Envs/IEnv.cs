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
        ///     Any non-negative seed, as in Gymnasium (a Python int with no upper bound); <see langword="long"/> covers every
        ///     seed up to 2^63 - 1.
        /// </param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        void Seed(long seed);
    }
}