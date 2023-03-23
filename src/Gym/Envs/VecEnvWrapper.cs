using System.Collections.Generic;
using System.Linq;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Envs {
    public class VecEnvWrapper : VecEnv {
        public VecEnvWrapper(Space observationSpace, Space actionSpace, params IEnv[] envs) : base(envs.Length, observationSpace, actionSpace) {
            Environments = new List<IEnv>(envs);
        }

        public VecEnvWrapper(Space observationSpace, Space actionSpace, ICollection<IEnv> envs) : base(envs.Count, observationSpace, actionSpace) {
            Environments = new List<IEnv>(envs);
        }


        public override NDArray[] Reset() {
            return Environments.Select(e => e.Reset()).ToArray();
        }

        public override Step[] Step(int action) {
            return Environments.Select(e => e.Step(action)).ToArray();
        }

        public override void Close() {
            foreach (var env in Environments) {
                env.CloseEnvironment();
            }
        }
    }
}