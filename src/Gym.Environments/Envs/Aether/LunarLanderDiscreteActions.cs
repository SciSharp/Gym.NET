#nullable enable
using System;

// 0.2's namespace for its Aether-physics LunarLander, which 2.0 removed (P8.5). It holds only this enum, so 0.2's
// `using Gym.Environments.Envs.Aether;` still compiles until 3.0 (DR-58); the env is Gym.Environments.Envs.Box2D's.
namespace Gym.Environments.Envs.Aether {
    /// <summary>
    ///     0.2's names for LunarLander's four discrete actions in 0.2's namespace, kept so a 0.2 file importing
    ///     <c>Gym.Environments.Envs.Aether</c> compiles until 3.0 (DR-58). The values are Gymnasium's action indices, the
    ///     same as <see cref="Box2D.LunarLanderDiscreteActions"/>'s.
    /// </summary>
    /// <remarks>
    ///     The Aether engine and its <c>LunarLanderEnv</c> are gone: use <see cref="Box2D.LunarLanderEnv"/>. Footgun: a file
    ///     that imports this namespace and <c>Gym.Environments.Envs.Box2D</c> sees two enums of this name (CS0104); drop this
    ///     namespace's import.
    /// </remarks>
    [Obsolete("0.2's namespace, gone with the Aether engine: replace using Gym.Environments.Envs.Aether with using Gym.Environments.Envs.Box2D, whose LunarLanderEnv is Gymnasium's. Removed in 3.0 with Gym.Legacy.")]
    public enum LunarLanderDiscreteActions {
        /// <summary>0: do nothing.</summary>
        Nothing = 0,

        /// <summary>1: fire the left orientation engine.</summary>
        FireLeftThruster = 1,

        /// <summary>2: fire the main engine.</summary>
        FireMainEngine = 2,

        /// <summary>3: fire the right orientation engine.</summary>
        FireRightThruster = 3
    }
}
