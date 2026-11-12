using System;
using Gym.Envs;
using Gym.Exceptions;
using Gym.MuJoCo;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Gymnasium's MuJoCo utilities (<c>gymnasium.envs.mujoco.utils</c>), mainly for tests: a MuJoCo env's whole state as
    ///     one vector, restoring it, and the check that a seeded reset is deterministic. They reach the simulation through
    ///     the env's unwrapped <see cref="MujocoEnv"/>, so any wrapper stack works.
    /// </summary>
    /// <remarks>
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/utils.py">gymnasium/envs/mujoco/utils.py</see>.
    /// </remarks>
    [PyName("utils")]
    public static class MujocoUtils {
        /// <summary>
        ///     The env's state as one float64 vector (<c>get_state</c>): the elements <paramref name="stateType"/> selects, in
        ///     MuJoCo's order (<see cref="Mujoco.MjGetState"/>). A new array the caller owns.
        /// </summary>
        /// <param name="env">A MuJoCo env, wrapped or not.</param>
        /// <param name="stateType">The elements: <see cref="MjtState.FullPhysics"/> (time, positions, velocities) by default,
        ///     <see cref="MjtState.Integration"/> for everything a step reads.</param>
        /// <returns>The state vector.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="env"/> is null.</exception>
        /// <exception cref="AttributeError">The unwrapped env is not a MuJoCo env (it has no <c>model</c>).</exception>
        [PyName("get_state")]
        public static NDArray GetState(IEnv env, MjtState stateType = MjtState.FullPhysics) {
            MujocoEnv mujoco = Unwrap(env);
            NDArray state = np.zeros(new Shape(Mujoco.MjStateSize(mujoco.Model, stateType)), NPTypeCode.Double);
            Mujoco.MjGetState(mujoco.Model, mujoco.Data, state, stateType);
            return state;
        }

        /// <summary>
        ///     Restores the env's state from a vector <see cref="GetState"/> made (<c>set_state</c>); nothing is recomputed
        ///     until the next step or forward pass, as in MuJoCo.
        /// </summary>
        /// <param name="env">A MuJoCo env, wrapped or not.</param>
        /// <param name="state">The state vector, of the same <paramref name="stateType"/>.</param>
        /// <param name="stateType">The elements the vector holds.</param>
        /// <returns><paramref name="state"/> itself, as Gymnasium returns it.</returns>
        /// <exception cref="ArgumentNullException">A null env or state.</exception>
        /// <exception cref="AttributeError">The unwrapped env is not a MuJoCo env.</exception>
        /// <exception cref="TypeError">The vector is not float64 or not of the signature's size.</exception>
        /// <exception cref="NotSupportedException">The vector holds nonzero applied forces, which the engine does not take.</exception>
        [PyName("set_state")]
        public static NDArray SetState(IEnv env, NDArray state, MjtState stateType = MjtState.FullPhysics) {
            MujocoEnv mujoco = Unwrap(env);
            ArgumentNullException.ThrowIfNull(state);
            Mujoco.MjSetState(mujoco.Model, mujoco.Data, state, stateType);
            return state;
        }

        /// <summary>
        ///     Asserts that a seeded reset restores the same state whatever ran before it (<c>check_mujoco_reset_state</c>):
        ///     reset with <paramref name="seed"/>, record the state, take one sampled step, reset with the same seed, and
        ///     compare.
        /// </summary>
        /// <param name="env">A MuJoCo env, wrapped or not; it is reset and stepped.</param>
        /// <param name="seed">The reset and action-space seed.</param>
        /// <param name="stateType">The elements compared; <see cref="MjtState.Integration"/> by default.</param>
        /// <exception cref="ArgumentNullException"><paramref name="env"/> is null.</exception>
        /// <exception cref="AttributeError">The unwrapped env is not a MuJoCo env.</exception>
        /// <exception cref="AssertionError">The two states differ (<c>reset is not deterministic</c>).</exception>
        [PyName("check_mujoco_reset_state")]
        public static void CheckMujocoResetState(IEnv env, int seed = 1234, MjtState stateType = MjtState.Integration) {
            ArgumentNullException.ThrowIfNull(env);
            env.ActionSpace.Seed(seed);
            object? action = env.ActionSpace.Sample();
            try {
                // the results are released at once: only the states are compared
                env.Reset(seed).Dispose();
                using NDArray first = GetState(env, stateType);
                env.Step(action).Dispose();
                env.Reset(seed).Dispose();
                using NDArray second = GetState(env, stateType);
                // np.all(a == b): any NaN makes the states unequal, as there
                for (long i = 0; i < first.size; i++)
                    if (!(first.GetDouble(i) == second.GetDouble(i)))
                        throw new AssertionError("reset is not deterministic");
            } finally {
                (action as IDisposable)?.Dispose();
            }
        }

        /// <summary>The unwrapped MuJoCo env of a stack.</summary>
        /// <param name="env">The env.</param>
        /// <returns>The <see cref="MujocoEnv"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="env"/> is null.</exception>
        /// <exception cref="AttributeError">The unwrapped env is not a MuJoCo env, with Python's message for the missing <c>model</c>.</exception>
        private static MujocoEnv Unwrap(IEnv env) {
            ArgumentNullException.ThrowIfNull(env);
            return env.Unwrapped as MujocoEnv
                   ?? throw new AttributeError($"'{PyAttributes.PyTypeName(env.Unwrapped.GetType())}' object has no attribute 'model'");
        }
    }
}
