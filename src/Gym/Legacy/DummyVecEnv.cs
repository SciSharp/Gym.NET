using System;

namespace Gym.Legacy {
    /// <summary>A <see cref="VecEnvWrapper"/> of a single legacy env, with that env's spaces.</summary>
    /// <remarks>Moved from <c>Gym.Envs</c> by the core-API port (P1).</remarks>
    [Obsolete("The baselines-style VecEnv is legacy (B-33: one action for every sub-env); gymnasium.vector replaces it (P6). Gym.Legacy is removed in 3.0.")]
    public class DummyVecEnv : VecEnvWrapper {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The only sub-env.</param>
        public DummyVecEnv(ILegacyEnv env) : base(env.ObservationSpace, env.ActionSpace, env) { }
    }
}
