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

        /// <summary>
        ///     Reseeds the env's generator, so the episodes that follow are reproducible; see <see cref="IEnv.Seed"/>.
        /// </summary>
        /// <param name="seed">A NumPy seed in <c>[0, 2**32 - 1]</c>.</param>
        /// <exception cref="ValueError">Thrown when <paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c>, with NumPy's message.</exception>
        /// <remarks>This took an <see cref="int"/> before the migration to NumSharp master; overrides must now take <see langword="long"/>.</remarks>
        public abstract void Seed(long seed);

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
        /// <summary>
        ///     Converts <paramref name="action"/> to <typeparamref name="TAction"/> and steps with it. Any integer width, a
        ///     <typeparamref name="TAction"/> member, and the 0-d int64 array that <see cref="Discrete.Sample"/> returns are
        ///     accepted.
        /// </summary>
        /// <param name="action">A boxed integer, a <typeparamref name="TAction"/> member, or a 0-d integer <see cref="NDArray"/>.</param>
        /// <returns>The result of <see cref="Step(TAction)"/>.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="action"/> does not denote an integer.</exception>
        /// <exception cref="OverflowException">Thrown when an unsigned 64-bit action exceeds <see cref="long.MaxValue"/>.</exception>
        public override Step Step(object action) {
            return Step(ToAction(action));
        }

        /// <summary>
        ///     Steps on <see cref="DistributedScheduler.Default"/>. The conversion runs inside the scheduled work, so a bad
        ///     action faults the returned task instead of throwing here.
        /// </summary>
        /// <param name="action">A boxed integer, a <typeparamref name="TAction"/> member, or a 0-d integer <see cref="NDArray"/>.</param>
        /// <returns>A task that completes with the result of <see cref="Step(TAction)"/>.</returns>
        public override Task<Step> StepAsync(object action) {
            return DistributedScheduler.Default.Run(() => Step(ToAction(action)));
        }

        /// <summary>
        ///     Maps an untyped action to the enum member of the same underlying value, whatever the enum's underlying type.
        /// </summary>
        /// <param name="action">A boxed integer, a <typeparamref name="TAction"/> member, or a 0-d integer <see cref="NDArray"/>.</param>
        /// <returns>The member of <typeparamref name="TAction"/> with that value; it need not be a named member.</returns>
        /// <exception cref="ArgumentException">Thrown when <paramref name="action"/> does not denote an integer.</exception>
        /// <exception cref="OverflowException">Thrown when an unsigned 64-bit action exceeds <see cref="long.MaxValue"/>.</exception>
        /// <remarks>
        ///     Replaces <c>(TAction)(object)action</c>, which unboxes exactly (trap 1 of docs/plans/NUMSHARP_MIGRATION_PLAN.md
        ///     §7): it threw <see cref="InvalidCastException"/> for a <see langword="long"/>, for a sampled action, and for any
        ///     integer whose width differs from the enum's underlying type.
        /// </remarks>
        private static TAction ToAction(object action) {
            return (TAction) Enum.ToObject(typeof(TAction), Discrete.ToInt64(action));
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