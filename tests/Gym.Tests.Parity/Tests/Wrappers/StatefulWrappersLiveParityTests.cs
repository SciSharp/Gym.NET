using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Wrappers;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Wrappers {
    /// <summary>
    ///     The wrappers of P6 that <c>make</c>'s spec re-creation needs (<c>NormalizeObservation</c>, <c>NormalizeReward</c>,
    ///     <c>TimeAwareObservation</c>, <c>FlattenObservation</c>) against Gymnasium's, live: whole episodes through each over
    ///     envs of float32, integer and tuple observations, the running statistics afterwards, a stack of three, and the
    ///     argument refusals.
    /// </summary>
    /// <remarks>
    ///     The normalizers' arithmetic is where NumPy 2's dtype rules show: float32 statistics updated by float32
    ///     observations, promoted to float64 by integer ones, and weak Python counts cast to float32; every observation and
    ///     reward is compared bit for bit. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py">gymnasium/wrappers/stateful_observation.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_reward.py">stateful_reward.py</see>,
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py">transform_observation.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class StatefulWrappersLiveParityTests : ParityTestBase {
        /// <summary>The running statistics of a normalizer, <c>(mean, var, count)</c>.</summary>
        private const string ObsRmsStatements = "result = (env.obs_rms.mean, env.obs_rms.var, env.obs_rms.count)\n";

        /// <summary>The return statistics of a reward normalizer, <c>(mean, var, count, discounted_reward)</c>.</summary>
        private const string ReturnRmsStatements = "result = (env.return_rms.mean, env.return_rms.var, env.return_rms.count, env.discounted_reward)\n";

        /// <summary>
        ///     <c>NormalizeObservation(make(id))</c> through 300 sampled steps (several episodes), and the statistics afterwards:
        ///     CartPole's and Pendulum's float32 observations keep float32 statistics, FrozenLake's integer ones promote them to
        ///     float64.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("CartPole-v1")]
        [DataRow("Pendulum-v1")]
        [DataRow("FrozenLake-v1")]
        public void NormalizeObservation_Episodes_MatchGymnasium(string id) {
            using var pair = new GymPair(PyGymEnv.Wrap($"NormalizeObservation(make('{id}'))", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.NormalizeObservation(made);
            }), new GymNetEnv(Wrap(id, env => (IEnv)Activator.CreateInstance(typeof(NormalizeObservation<,>).MakeGenericType(EnvTypes(env)), env, 1e-8)!)));
            RunEpisodes(pair, 300);
            AssertStatisticsSame(pair, ObsRmsStatements, env => {
                RunningMeanStd rms = (RunningMeanStd)PyGetAttr(env, "obs_rms");
                return new object?[] { rms.Mean.copy(), rms.Var.copy(), rms.Count };
            });
        }

        /// <summary>
        ///     <c>NormalizeReward(make(id), gamma)</c> through 300 sampled steps: float rewards (CartPole, Pendulum) and int ones
        ///     (FrozenLake), the discounted return carried across episode ends, and the statistics afterwards.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="gamma">The discount.</param>
        [TestMethod]
        [DataRow("CartPole-v1", 0.99)]
        [DataRow("Pendulum-v1", 0.8)]
        [DataRow("FrozenLake-v1", 0.99)]
        public void NormalizeReward_Episodes_MatchGymnasium(string id, double gamma) {
            using var pair = new GymPair(PyGymEnv.Wrap($"NormalizeReward(make('{id}'), gamma={gamma})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.NormalizeReward(made, gamma);
            }), new GymNetEnv(Wrap(id, env => (IEnv)Activator.CreateInstance(typeof(NormalizeReward<,>).MakeGenericType(EnvTypes(env)), env, gamma, 1e-8)!)));
            RunEpisodes(pair, 300);
            AssertStatisticsSame(pair, ReturnRmsStatements, env => {
                RunningMeanStd rms = (RunningMeanStd)PyGetAttr(env, "return_rms");
                return new object?[] { rms.Mean.copy(), rms.Var.copy(), rms.Count, ((NDArray)PyGetAttr(env, "discounted_reward")).copy() };
            });
        }

        /// <summary>
        ///     <c>TimeAwareObservation(make(id), flatten, normalize_time)</c> through truncation: the step count as int32 or its
        ///     float32 fraction of the limit, flattened into one array or kept as <c>{"obs": ..., "time": ...}</c>.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="flatten">Whether to flatten.</param>
        /// <param name="normalizeTime">Whether to normalize the time.</param>
        [TestMethod]
        [DataRow("CartPole-v1", true, false)]
        [DataRow("CartPole-v1", false, false)]
        [DataRow("CartPole-v1", true, true)]
        [DataRow("FrozenLake-v1", true, true)]
        [DataRow("FrozenLake-v1", false, true)]
        public void TimeAwareObservation_Episodes_MatchGymnasium(string id, bool flatten, bool normalizeTime) {
            using var pair = new GymPair(PyGymEnv.Wrap($"TimeAwareObservation(make('{id}'), flatten={flatten}, normalize_time={normalizeTime})", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.TimeAwareObservation(made, flatten, normalizeTime);
            }), new GymNetEnv(Wrap(id, env => (IEnv)Activator.CreateInstance(typeof(TimeAwareObservation<,>).MakeGenericType(EnvTypes(env)), env, flatten, normalizeTime, "time")!)));
            RunEpisodes(pair, 300);
        }

        /// <summary>
        ///     <c>FlattenObservation(make(id))</c>: a <c>Box</c> flattened as it is, FrozenLake's <c>Discrete</c> one-hot, and
        ///     Blackjack's <c>Tuple</c> of <c>Discrete</c>s concatenated one-hots.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("CartPole-v1")]
        [DataRow("FrozenLake-v1")]
        [DataRow("Blackjack-v1")]
        public void FlattenObservation_Episodes_MatchGymnasium(string id) {
            using var pair = new GymPair(PyGymEnv.Wrap($"FlattenObservation(make('{id}'))", () => {
                using PyObject made = Python.gymnasium.make(id);
                return Python.gymnasium.wrappers.FlattenObservation(made);
            }), new GymNetEnv(Wrap(id, env => (IEnv)Activator.CreateInstance(typeof(FlattenObservation<,>).MakeGenericType(EnvTypes(env)), env)!)));
            RunEpisodes(pair, 200);
        }

        /// <summary>The stack of Gymnasium's make tests, <c>NormalizeReward(TimeAwareObservation(FlattenObservation(make("CartPole-v1"))), gamma=0.8)</c>.</summary>
        [TestMethod]
        public void Stack_MatchesGymnasium() {
            using var pair = new GymPair(PyGymEnv.Wrap("NormalizeReward(TimeAwareObservation(FlattenObservation(make('CartPole-v1'))), gamma=0.8)", () => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                using PyObject flat = Python.gymnasium.wrappers.FlattenObservation(made);
                using PyObject timed = Python.gymnasium.wrappers.TimeAwareObservation(flat);
                return Python.gymnasium.wrappers.NormalizeReward(timed, 0.8);
            }), new GymNetEnv(new NormalizeReward<object?, NDArray>(new TimeAwareObservation<NDArray, NDArray>(
                new FlattenObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1"))), gamma: 0.8)));
            RunEpisodes(pair, 400);
        }

        /// <summary>The refusals: a gamma outside [0, 1], a non-positive epsilon, and <c>TimeAwareObservation</c> without a step limit.</summary>
        [TestMethod]
        public void Refusals_MatchGymnasium() {
            ByteContract.AssertSame("NormalizeReward(gamma=1.5)", Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.NormalizeReward(made, 1.5);
            }), Outcome.Capture(() => new NormalizeReward<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1"), gamma: 1.5)));
            ByteContract.AssertSame("NormalizeReward(epsilon=0.0)", Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.NormalizeReward(made, 0.99, 0.0);
            }), Outcome.Capture(() => new NormalizeReward<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1"), epsilon: 0.0)));
            ByteContract.AssertSame("NormalizeObservation(epsilon=-1.0)", Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("CartPole-v1");
                return Python.gymnasium.wrappers.NormalizeObservation(made, -1.0);
            }), Outcome.Capture(() => new NormalizeObservation<NDArray, NDArray>((Env<NDArray, NDArray>)Gymnasium.Make("CartPole-v1"), epsilon: -1.0)));
            ByteContract.AssertSame("TimeAwareObservation(make('Blackjack-v1'))", Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("Blackjack-v1");
                return Python.gymnasium.wrappers.TimeAwareObservation(made);
            }), Outcome.Capture(() => new TimeAwareObservation<object?[], NDArray>((Env<object?[], NDArray>)Gymnasium.Make("Blackjack-v1"))));
        }

        /// <summary>Runs a pair through episodes of seeded sampled actions, resetting after each end, for <paramref name="steps"/> steps in all.</summary>
        /// <param name="pair">The pair.</param>
        /// <param name="steps">The steps across all episodes.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private void RunEpisodes(GymPair pair, int steps) {
            int taken = 0;
            for (long seed = 0; taken < steps; seed++) {
                PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: steps - taken);
                taken += Math.Max(run.Steps, 1);
                TestContext.WriteLine(run.ToString());
            }
        }

        /// <summary>Compares a wrapper's running statistics after a run, read by a snippet on Gymnasium's stack and a function on Gym.NET's.</summary>
        /// <param name="pair">The pair, after its run.</param>
        /// <param name="statements">The snippet reading the statistics as <c>result</c> from <c>env</c>.</param>
        /// <param name="dotnet">Reads the same values from Gym.NET's outermost wrapper.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private static void AssertStatisticsSame(GymPair pair, string statements, Func<IEnv, object?[]> dotnet) {
            Outcome python = Outcome.CapturePython(() => Python.evalStmts(statements, "result", ("env", pair.Python.Stepped)));
            ByteContract.AssertSame("statistics after the run", python, Outcome.Capture(() => dotnet(((GymNetEnv)pair.DotNet).Env)));
        }

        /// <summary>An attribute of a Gym.NET wrapper by its Python name.</summary>
        /// <param name="env">The wrapper.</param>
        /// <param name="name">The Python name.</param>
        /// <returns>The value.</returns>
        private static object PyGetAttr(IEnv env, string name) => Gym.Utils.PyAttributes.GetAttr(env, name)!;

        /// <summary>Gym.NET's <c>make(id)</c> wrapped by a wrapper built for its types.</summary>
        /// <param name="id">The id.</param>
        /// <param name="wrap">Builds the wrapper over the made env.</param>
        /// <returns>The wrapper; the pair owns it.</returns>
        private static IEnv Wrap(string id, Func<IEnv, IEnv> wrap) => wrap(Gymnasium.Make(id));

        /// <summary>An env's <c>Env&lt;TObs, TAct&gt;</c> type arguments, to close a wrapper's generic definition over it.</summary>
        /// <param name="env">The env.</param>
        /// <returns>The observation and action types.</returns>
        /// <exception cref="InvalidOperationException">Thrown for an env that is no <c>Env&lt;TObs, TAct&gt;</c>.</exception>
        private static Type[] EnvTypes(IEnv env) {
            for (Type? type = env.GetType(); type is not null; type = type.BaseType) {
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Env<,>))
                    return type.GetGenericArguments();
            }

            throw new InvalidOperationException($"{env.GetType().Name} is not an Env<TObs, TAct>.");
        }
    }
}
