using System;
using System.Threading.Tasks;
using Gym.Collections;
using Gym.Observations;
using Gym.Spaces;
using Gym.Threading;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

// ReSharper disable once CheckNamespace
namespace Gym.Envs {
    public abstract class Env : IDisposable, IEnv {
        public Dict Metadata { get; set; }
        public (float From, float To) RewardRange { get; set; }

        public Space ActionSpace { get; set; }
        public Space ObservationSpace { get; set; }

        public abstract NDArray Reset();
        public abstract Step Step(object action);

        public virtual Task<Step> StepAsync(object action) {
            return DistributedScheduler.Default.Run(() => Step(action));
        }

        public abstract Image Render(string mode = "human");
        public abstract void CloseEnvironment();
        public abstract void Seed(int seed);

        /// <summary>
        ///     Returns unwrapped version of current environment.
        /// </summary>
        protected virtual Env Unwrapped() {
            return this;
        }

        public void Dispose() {
            CloseEnvironment();
        }
    }

    public abstract class Env<TAction> : Env where TAction : Enum {
        public override Step Step(object action) {
            return Step((TAction) (object) action);
        }

        public override Task<Step> StepAsync(object action) {
            return DistributedScheduler.Default.Run(() => Step((TAction) (object) action));
        }

        public abstract Step Step(TAction action);
    }

    public abstract class GoalEnv : Env {
        public override NDArray Reset() {
            //todo var result = base.Reset();
            return null;
        }

        public abstract object ComputeReward(object achievedGoal, object DesiredGoal, Dict info);
    }
}