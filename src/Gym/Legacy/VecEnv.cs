using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Gym.Collections;
using Gym.Spaces;
using NumSharp;

namespace Gym.Legacy {
    /// <summary>
    ///     The base of the baselines-style vector envs Gym.NET had before the core-API work (see <see cref="IVecEnv"/>):
    ///     holds the sub-envs, their shared spaces, and the seeding and attribute helpers.
    /// </summary>
    /// <remarks>
    ///     Moved from <c>Gym.Envs</c> by the core-API work (P1). <see cref="StepAsync"/> returns a <see cref="ValueTask{TResult}"/>
    ///     that completes on the calling thread; the <c>DistributedScheduler</c> it ran on relied on <c>Thread.Abort</c> and
    ///     threw on .NET 5+ (B-34).
    /// </remarks>
    [Obsolete("The baselines-style VecEnv is legacy (B-33: one action for every sub-env); gymnasium.vector replaces it (P6). Gym.Legacy is removed in 3.0.")]
    public abstract class VecEnv : IVecEnv {
        /// <summary>Records the number of sub-envs and the spaces they share; subclasses fill <see cref="Environments"/>.</summary>
        /// <param name="num_envs">The number of sub-envs.</param>
        /// <param name="observationSpace">The observation space of one sub-env.</param>
        /// <param name="actionSpace">The action space of one sub-env.</param>
        public VecEnv(int num_envs, Space observationSpace, Space actionSpace) {
            NumberOfEnvironments = num_envs;
            ObservationSpace = observationSpace;
            ActionSpace = actionSpace;
            Environments = new List<ILegacyEnv>();
        }

        /// <summary>The vector env's metadata.</summary>
        public Dict Metadata { get; set; }

        /// <summary><c>gym</c>'s <c>reward_range</c>, in float64 like the rewards.</summary>
        public (double From, double To) RewardRange { get; set; }

        /// <summary>The action space of one sub-env.</summary>
        public Space ActionSpace { get; set; }

        /// <summary>The number of sub-envs given to the constructor.</summary>
        public int NumberOfEnvironments { get; }

        /// <summary>The sub-envs, stepped in this order.</summary>
        public IList<ILegacyEnv> Environments { get; set; }

        /// <summary>The observation space of one sub-env.</summary>
        public Space ObservationSpace { get; set; }

        //
        //         Reset all the environments and return an array of
        //         observations, or a tuple of observation arrays.
        //
        //         If step_async is still doing work, that work will
        //         be cancelled and step_wait() should not be called
        //         until step_async() is invoked again.
        //
        //         :return: ([int] or [float]) observation
        //
        /// <summary>Resets every sub-env.</summary>
        /// <returns>One first observation per sub-env, in <see cref="Environments"/> order.</returns>
        public abstract NDArray[] Reset();

        /// <summary>Steps every sub-env with the same <paramref name="action"/>; see <see cref="IVecEnv.Step"/> (B-33).</summary>
        /// <param name="action">The discrete action, <see langword="long"/> like Gymnasium's <c>np.int64</c>.</param>
        /// <returns>One step result per sub-env, in <see cref="Environments"/> order.</returns>
        public abstract Step[] Step(long action);

        /// <summary>Closes every sub-env.</summary>
        public abstract void Close();

        /// <summary>Seeds every sub-env with the same <paramref name="seed"/>, so they all replay the same stream.</summary>
        /// <param name="seed">A non-negative seed.</param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        public void Seed(long seed) {
            Seed(Enumerable.Repeat(seed, Environments.Count).ToArray());
        }

        /// <summary>Seeds each sub-env with its own seed, in <see cref="Environments"/> order.</summary>
        /// <param name="seed">One non-negative seed per sub-env.</param>
        /// <exception cref="ArgumentException">Thrown when the number of seeds differs from the number of sub-envs.</exception>
        /// <exception cref="Gym.Exceptions.Error">
        ///     Thrown when a seed is negative, with Gymnasium's message. The sub-envs before it have already been reseeded.
        /// </exception>
        public void Seed(long[] seed) {
            if (seed.Length != Environments.Count) throw new ArgumentException("Number of seeds passed should be equals to number of environments", nameof(seed));
            for (int i = 0; i < Environments.Count; i++) {
                Environments[i].Seed(seed[i]);
            }
        }

        //
        //         Tell all the environments to start taking a step
        //         with the given actions.
        //         Call step_wait() to get the results of the step.
        //
        //         You should not call this if a step_async run is
        //         already pending.
        //
        /// <summary>
        ///     Steps every sub-env asynchronously. The default completes with <see cref="Step(long)"/>, run on the calling thread;
        ///     <see cref="VecEnvWrapper"/> awaits each sub-env's own <see cref="ILegacyEnv.StepAsync"/> instead.
        /// </summary>
        /// <param name="action">The discrete action sent to every sub-env.</param>
        /// <returns>A value task that completes with one step result per sub-env, or is faulted with what a step threw.</returns>
        /// <remarks>Exceptions surface when the task is awaited, never from this call. Await it before the next call.</remarks>
        public virtual ValueTask<Step[]> StepAsync(long action) {
            try {
                return new ValueTask<Step[]>(Step(action));
            } catch (Exception exception) {
                // An asynchronous method reports failure through its task, so the caller's await, not this call, throws.
                return ValueTask.FromException<Step[]>(exception);
            }
        }

        //
        //         Return attribute from vectorized environment.
        //
        //         :param attr_name: (str) The name of the attribute whose value to return
        //         :param indices: (list,int) Indices of envs to get attribute from
        //         :return: (list) List of values of 'attr_name' in all environments
        //
        /// <summary>Reads a value from every sub-env.</summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="selector">Reads the value from one sub-env.</param>
        /// <returns>One value per sub-env, in <see cref="Environments"/> order.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="selector"/> is <see langword="null"/>.</exception>
        public virtual T[] get_attr<T>(Func<ILegacyEnv, T> selector) {
            if (selector == null) throw new ArgumentNullException(nameof(selector));
            return Environments.Select(selector).ToArray();
        }

        //
        //         Set attribute inside vectorized environments.
        //
        //         :param attr_name: (str) The name of attribute to assign new value
        //         :param value: (obj) Value to assign to `attr_name`
        //         :param indices: (list,int) Indices of envs to assign value
        //         :return: (NoneType)
        //
        /// <summary>Assigns a value on every sub-env.</summary>
        /// <typeparam name="T">The value's type.</typeparam>
        /// <param name="selector">Assigns the value on one sub-env.</param>
        /// <param name="object">The value.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="object"/> is <see langword="null"/>.</exception>
        public void set_attr<T>(Action<ILegacyEnv, T> selector, T @object) {
            if (@object == null) throw new ArgumentNullException(nameof(@object));
            foreach (var env in Environments) {
                selector(env, @object);
            }
        }
    }
}
