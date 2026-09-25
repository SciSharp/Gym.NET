using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Gym.Spaces;
using NumSharp;

namespace Gym.Legacy {
    /// <summary>A <see cref="VecEnv"/> over a fixed set of legacy envs, stepped one after another on the calling thread.</summary>
    /// <remarks>Moved from <c>Gym.Envs</c> by the core-API work (P1); the sub-envs are <see cref="ILegacyEnv"/>s.</remarks>
    [Obsolete("The baselines-style VecEnv is legacy (B-33: one action for every sub-env); gymnasium.vector replaces it (P6). Gym.Legacy is removed in 3.0.")]
    public class VecEnvWrapper : VecEnv {
        /// <summary>Wraps the given envs, which share the spaces given.</summary>
        /// <param name="observationSpace">The observation space of one sub-env.</param>
        /// <param name="actionSpace">The action space of one sub-env.</param>
        /// <param name="envs">The sub-envs, in stepping order.</param>
        public VecEnvWrapper(Space observationSpace, Space actionSpace, params ILegacyEnv[] envs) : base(envs.Length, observationSpace, actionSpace) {
            Environments = new List<ILegacyEnv>(envs);
        }

        /// <summary>Wraps the given envs, which share the spaces given.</summary>
        /// <param name="observationSpace">The observation space of one sub-env.</param>
        /// <param name="actionSpace">The action space of one sub-env.</param>
        /// <param name="envs">The sub-envs, in stepping order.</param>
        public VecEnvWrapper(Space observationSpace, Space actionSpace, ICollection<ILegacyEnv> envs) : base(envs.Count, observationSpace, actionSpace) {
            Environments = new List<ILegacyEnv>(envs);
        }


        /// <summary>Resets every sub-env, one after another.</summary>
        /// <returns>One first observation per sub-env.</returns>
        public override NDArray[] Reset() {
            return Environments.Select(e => e.Reset()).ToArray();
        }

        /// <summary>
        ///     Steps every sub-env, one after another, with the same <paramref name="action"/> (B-33). The action reaches each
        ///     env boxed as a <see langword="long"/>, so an env that still casts <c>(int)action</c> throws
        ///     <see cref="System.InvalidCastException"/>; envs convert with <see cref="Discrete.ToInt64"/>.
        /// </summary>
        /// <param name="action">The discrete action.</param>
        /// <returns>One step result per sub-env, in <see cref="VecEnv.Environments"/> order.</returns>
        public override Step[] Step(long action) {
            return Environments.Select(e => e.Step(action)).ToArray();
        }

        /// <summary>
        ///     Steps every sub-env through its own <see cref="ILegacyEnv.StepAsync"/>, one after another in
        ///     <see cref="VecEnv.Environments"/> order (B-33: the same action for all), so the results equal
        ///     <see cref="Step(long)"/>'s and a sub-env that is genuinely asynchronous stays so.
        /// </summary>
        /// <param name="action">The discrete action.</param>
        /// <returns>A value task that completes with one step result per sub-env, or is faulted with the first step's error.</returns>
        public override async ValueTask<Step[]> StepAsync(long action) {
            var steps = new Step[Environments.Count];
            for (int i = 0; i < steps.Length; i++)
                steps[i] = await Environments[i].StepAsync(action).ConfigureAwait(false);
            return steps;
        }

        /// <summary>Closes every sub-env.</summary>
        public override void Close() {
            foreach (var env in Environments) {
                env.CloseEnvironment();
            }
        }
    }
}
