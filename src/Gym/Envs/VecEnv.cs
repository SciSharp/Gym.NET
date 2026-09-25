using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Gym.Collections;
using Gym.Observations;
using Gym.Spaces;
using Gym.Threading;
using NumSharp;

namespace Gym.Envs {
    public abstract class VecEnv : IVecEnv {
        public VecEnv(int num_envs, Space observationSpace, Space actionSpace) {
            NumberOfEnvironments = num_envs;
            ObservationSpace = observationSpace;
            ActionSpace = actionSpace;
            Environments = new List<IEnv>();
        }

        public Dict Metadata { get; set; }
        public (float From, float To) RewardRange { get; set; }
        public Space ActionSpace { get; set; }
        public int NumberOfEnvironments { get; }

        public IList<IEnv> Environments { get; set; }

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
        public abstract NDArray[] Reset();

        /// <summary>Steps every sub-env with the same <paramref name="action"/>; see <see cref="IVecEnv.Step"/> (B-33).</summary>
        /// <param name="action">The discrete action, <see langword="long"/> like Gymnasium's <c>np.int64</c>.</param>
        /// <returns>One step result per sub-env, in <see cref="Environments"/> order.</returns>
        public abstract Step[] Step(long action);

        public abstract void Close();

        /// <summary>Seeds every sub-env with the same <paramref name="seed"/>, so they all replay the same stream.</summary>
        /// <param name="seed">A NumPy seed in <c>[0, 2**32 - 1]</c>.</param>
        /// <exception cref="ValueError">Thrown when <paramref name="seed"/> is outside <c>[0, 2**32 - 1]</c>.</exception>
        public void Seed(long seed) {
            Seed(Enumerable.Repeat(seed, Environments.Count).ToArray());
        }

        /// <summary>Seeds each sub-env with its own seed, in <see cref="Environments"/> order.</summary>
        /// <param name="seed">One NumPy seed in <c>[0, 2**32 - 1]</c> per sub-env.</param>
        /// <exception cref="ArgumentException">Thrown when the number of seeds differs from the number of sub-envs.</exception>
        /// <exception cref="ValueError">
        ///     Thrown when a seed is outside <c>[0, 2**32 - 1]</c>. The sub-envs before it have already been reseeded.
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
        /// <summary>Runs <see cref="Step(long)"/> on <see cref="DistributedScheduler.Default"/>.</summary>
        /// <param name="action">The discrete action sent to every sub-env.</param>
        /// <returns>A task that completes with one step result per sub-env.</returns>
        public Task<Step[]> StepAsync(long action) {
            return DistributedScheduler.Default.Run(() => Step(action));
        }

        // 
        //         Return attribute from vectorized environment.
        // 
        //         :param attr_name: (str) The name of the attribute whose value to return
        //         :param indices: (list,int) Indices of envs to get attribute from
        //         :return: (list) List of values of 'attr_name' in all environments
        //         
        public virtual T[] get_attr<T>(Func<IEnv, T> selector) {
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
        public void set_attr<T>(Action<IEnv, T> selector, T @object) {
            if (@object == null) throw new ArgumentNullException(nameof(@object));
            foreach (var env in Environments) {
                selector(env, @object);
            }
        }
    }
}