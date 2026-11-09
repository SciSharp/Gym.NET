#nullable enable
using System;
using Gym.Envs;

namespace Gym.Tests.Gymnasium.Wrappers {
    /// <summary>
    ///     The helpers of Gymnasium's <c>tests/wrappers/utils.py</c> that the ported tests use; the rest arrive with the
    ///     wrapper tests that need them (P6).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py">tests/wrappers/utils.py</see>.</remarks>
    public static class WrapperUtils {
        /// <summary>
        ///     Whether a wrapper class is anywhere in an env's wrapper stack (Gymnasium's <c>has_wrapper</c>), walking
        ///     <c>wrapped_env.env</c> down to the unwrapped env.
        /// </summary>
        /// <param name="wrappedEnv">The outermost env.</param>
        /// <param name="wrapperType">The wrapper class, open (<c>typeof(TimeLimit&lt;,&gt;)</c>, any type arguments) or closed.</param>
        /// <returns><see langword="true"/> when a layer is of that class.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/wrappers/utils.py#L87">tests/wrappers/utils.py (has_wrapper)</see>.</remarks>
        public static bool HasWrapper(IEnv wrappedEnv, Type wrapperType) {
            IEnv? env = wrappedEnv;
            while (env is IWrapper wrapper) {
                if (IsOfClass(env.GetType(), wrapperType))
                    return true;
                env = wrapper.Env;
            }

            return false;
        }

        /// <summary>Python's <c>isinstance</c> for an open generic class: the type or one of its bases is a closed form of it.</summary>
        /// <param name="type">The env's class.</param>
        /// <param name="wrapperType">The wrapper class, open or closed.</param>
        /// <returns><see langword="true"/> when the env is one.</returns>
        private static bool IsOfClass(Type type, Type wrapperType) {
            if (!wrapperType.IsGenericTypeDefinition)
                return wrapperType.IsAssignableFrom(type);
            for (Type? level = type; level != null; level = level.BaseType) {
                if (level.IsGenericType && level.GetGenericTypeDefinition() == wrapperType)
                    return true;
            }

            return false;
        }
    }
}
