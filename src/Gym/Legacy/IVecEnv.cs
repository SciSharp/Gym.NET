using System;
using System.Collections.Generic;
using Gym.Collections;
using Gym.Spaces;
using NumSharp;

namespace Gym.Legacy {
    /// <summary>
    ///     The baselines-style vector env contract Gym.NET had before the core-API port: several legacy envs stepped together.
    ///     <c>gymnasium.vector</c> (<c>VectorEnv</c>, <c>SyncVectorEnv</c>, <c>AsyncVectorEnv</c>) replaces it in P6.
    /// </summary>
    /// <remarks>
    ///     Moved from <c>Gym.Envs</c> by the core-API port (P1). Its sub-envs are <see cref="ILegacyEnv"/>s: wrap ported envs
    ///     in <see cref="LegacyEnvAdapter"/>.
    /// </remarks>
    [Obsolete("The baselines-style VecEnv is legacy (B-33: one action for every sub-env); gymnasium.vector replaces it (P6). Gym.Legacy is removed in 3.0.")]
    public interface IVecEnv {
        /// <summary>The sub-envs, stepped in this order.</summary>
        IList<ILegacyEnv> Environments { get; }

        /// <summary>The vector env's metadata.</summary>
        Dict Metadata { get; set; }

        /// <summary><c>gym</c>'s <c>reward_range</c>.</summary>
        (float From, float To) RewardRange { get; set; }

        /// <summary>The action space of one sub-env.</summary>
        Space ActionSpace { get; set; }

        /// <summary>The observation space of one sub-env.</summary>
        Space ObservationSpace { get; set; }

        /// <summary>Resets every sub-env.</summary>
        /// <returns>One first observation per sub-env, in <see cref="Environments"/> order.</returns>
        NDArray[] Reset();

        /// <summary>
        ///     Steps every sub-env with the same <paramref name="action"/> (B-33: a vector env should take one action per
        ///     sub-env; <c>gymnasium.vector</c> replaces this type in P6).
        /// </summary>
        /// <param name="action">The discrete action, <see langword="long"/> like Gymnasium's <c>np.int64</c>.</param>
        /// <returns>One step result per sub-env, in <see cref="Environments"/> order.</returns>
        Step[] Step(long action);

        /// <summary>Closes every sub-env.</summary>
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
