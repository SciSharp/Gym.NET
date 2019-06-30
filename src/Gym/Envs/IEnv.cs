using System.Threading.Tasks;
using Gym.Collections;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

// ReSharper disable once CheckNamespace
namespace Gym.Envs {
    public interface IEnv {
        Dict Metadata { get; set; }
        (float From, float To) RewardRange { get; set; }
        Space ActionSpace { get; set; }
        Space ObservationSpace { get; set; }
        NDArray Reset();
        Step Step(int action);
        Task<Step> StepAsync(int action);
        byte[] Render(string mode = "human");
        void Close();
        void Seed(int seed);
    }
}