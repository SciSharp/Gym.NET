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

        /// <summary>
        ///     Steps every sub-env with the same <paramref name="action"/> (B-33: a vector env should take one action per
        ///     sub-env; <c>gymnasium.vector</c> replaces this type in P6).
        /// </summary>
        /// <param name="action">The discrete action, <see langword="long"/> like Gymnasium's <c>np.int64</c>.</param>
        /// <returns>One step result per sub-env, in <see cref="Environments"/> order.</returns>
        Step[] Step(long action);
        void Close();

        /// <summary>Seeds each sub-env with its own seed, in <see cref="Environments"/> order.</summary>
        /// <param name="seed">One non-negative seed per sub-env.</param>
        /// <exception cref="System.ArgumentException">Thrown when the number of seeds differs from the number of sub-envs.</exception>
        /// <exception cref="Gym.Exceptions.Error">Thrown when a seed is negative, with Gymnasium's message.</exception>
        void Seed(long[] seed);

        /// <summary>Seeds every sub-env with the same <paramref name="seed"/>, so they all replay the same stream.</summary>
        /// <param name="seed">A non-negative seed.</param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        void Seed(long seed);
    }
}