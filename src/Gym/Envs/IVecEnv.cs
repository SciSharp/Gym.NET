using System.Collections.Generic;
using Gym.Collections;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Envs {
    public interface IVecEnv {
        IList<IEnv> Environments { get; }
        Dict Metadata { get; set; }
        (float From, float To) RewardRange { get; set; }
        Space ActionSpace { get; set; }
        Space ObservationSpace { get; set; }
        NDArray[] Reset();
        Step[] Step(int action);
        void Close();
        void Seed(int[] seed);
        void Seed(int seed);
    }
}