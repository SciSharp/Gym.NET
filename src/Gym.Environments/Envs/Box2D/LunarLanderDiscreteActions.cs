#nullable enable
using System;

namespace Gym.Environments.Envs.Box2D {
    /// <summary>
    ///     0.2's names for LunarLander's four discrete actions, kept so 0.2 code compiles until 2.0 (DR-58). The values are
    ///     Gymnasium's action indices, so <c>env.Step((int)LunarLanderDiscreteActions.FireMainEngine)</c> fires the main
    ///     engine of <see cref="LunarLanderEnv"/> as 0.2's did.
    /// </summary>
    /// <remarks>
    ///     Gymnasium names the actions only in its docs (0 do nothing, 1 fire left orientation engine, 2 fire main engine,
    ///     3 fire right orientation engine). Cast to <see cref="int"/> for the typed <c>Step(NDArray)</c>; the untyped
    ///     <see cref="Gym.Envs.IEnv.Step(object)"/> takes the enum itself. Footgun: 0.2's namespace,
    ///     <c>Gym.Environments.Envs.Aether</c>, has a twin of this enum, so a file importing both namespaces must drop the
    ///     Aether one (CS0104 names it).
    /// </remarks>
    [Obsolete("0.2's action names: Gymnasium's LunarLander takes the action indices 0 (nothing), 1 (left engine), 2 (main engine), 3 (right engine). Removed in 2.0 with Gym.Legacy.")]
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
