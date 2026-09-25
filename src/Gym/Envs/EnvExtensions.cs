#nullable enable
using System;
using System.Threading.Tasks;
using Gym.Observations;

namespace Gym.Envs {
    /// <summary>
    ///     Gym.NET additions over Gymnasium's env contract (port plan §15): asynchronous stepping as a thin layer over
    ///     <see cref="Task.Run(Func{Task})"/>. It replaces the former <c>DistributedScheduler</c> family, which relied on
    ///     <c>Thread.Abort</c> and threw on .NET 5+ (B-34).
    /// </summary>
    public static class EnvExtensions {
        /// <summary>
        ///     Steps on the thread pool, so the caller can await a slow env (physics, rendering) without blocking. The env is
        ///     not made thread-safe by this: await each step before issuing the next one on the same env.
        /// </summary>
        /// <typeparam name="TObs">The env's observation type.</typeparam>
        /// <typeparam name="TAct">The env's action type.</typeparam>
        /// <param name="env">The env.</param>
        /// <param name="action">The action.</param>
        /// <returns>A task that completes with the transition, or faults with what <see cref="Env{TObs, TAct}.Step"/> threw.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public static Task<StepResult<TObs>> StepAsync<TObs, TAct>(this Env<TObs, TAct> env, TAct action) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            return Task.Run(() => env.Step(action));
        }

        /// <summary>
        ///     Steps an env of unknown types on the thread pool (see <see cref="StepAsync{TObs, TAct}"/>); the action is
        ///     converted inside the task, so a bad action faults the task instead of throwing here.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <param name="action">The action.</param>
        /// <returns>A task that completes with the transition, its observation boxed.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public static Task<StepResult<object?>> StepAsync(this IEnv env, object? action) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            return Task.Run(() => env.Step(action));
        }
    }
}
