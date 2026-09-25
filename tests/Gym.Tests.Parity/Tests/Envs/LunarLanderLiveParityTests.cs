using System.Collections.Generic;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     LunarLander-v3 against Gymnasium's <c>LunarLander</c> (<c>gymnasium/envs/box2d/lunar_lander.py</c> @ 9e04324), live:
    ///     the spaces, and whole episodes of Gymnasium's own heuristic in lockstep.
    /// </summary>
    /// <remarks>
    ///     Gym.NET's lander is a different simulation today, not a divergent copy: Aether.Physics2D instead of Box2D 2.3.10
    ///     (F6), v2 geometry and observation bounds, its own draw order (B-17, B-18). Its episodes are open bugs until P8; the
    ///     cells are here so the port has its gate waiting.
    /// </remarks>
    [TestClass]
    public sealed class LunarLanderLiveParityTests : ParityTestBase {
        /// <summary>
        ///     The discrete action space, <c>Discrete(4)</c>: <c>n</c>, <c>start</c>, dtype, and seeded samples with the generator
        ///     state behind them, match Gymnasium's.
        /// </summary>
        [TestMethod]
        public void GymNet_DiscreteActionSpace_MatchesGymnasium() {
            using PyRef python = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace("LunarLander-v3", Python.gymnasium.Env.action_space));
            var space = (Discrete)new LunarLanderEnv(NullEnvViewer.Factory).ActionSpace;
            ByteContract.AssertSame("LunarLander action_space.n", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.n(python.Object)), Outcome.Returned(ValueFacts.Of(space.N)));
            ByteContract.AssertSame("LunarLander action_space.start", Outcome.CapturePython(() => Python.gymnasium.spaces.discrete.Discrete.start(python.Object)), Outcome.Returned(ValueFacts.Of(space.Start)));
            ByteContract.AssertSame("LunarLander action_space.dtype", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.dtype(python.Object)), Outcome.Returned(ValueFacts.Of(space.DType)));

            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(python.Object, 11)).Expect("action_space.seed(11)");
            space.Seed(11);
            for (int i = 1; i <= 20; i++)
                ByteContract.AssertSame($"LunarLander action_space.sample() #{i} after seed(11)", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(python.Object)), Outcome.Capture(() => space.Sample()));
        }

        /// <summary>
        ///     The continuous action space is <c>Box(-1, 1, (2,), float32)</c> in Gymnasium: two throttles. Gym.NET's is 0-d
        ///     (B-20), so its samples cannot drive the lander.
        /// </summary>
        [TestMethod]
        [OpenBugs("B-20")]
        public void GymNet_ContinuousActionSpace_MatchesGymnasium() {
            var kwargs = new Dictionary<string, object?> { ["continuous"] = true };
            using PyRef python = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace("LunarLander-v3", Python.gymnasium.Env.action_space, kwargs));
            var space = (Box)new LunarLanderEnv(NullEnvViewer.Factory, continuous: true).ActionSpace;
            ByteContract.AssertSame("LunarLanderContinuous action_space.shape", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.shape(python.Object)), Outcome.Returned(Spaces.DiscreteLiveParityTests.ShapeFacts(space.Shape)));
            ByteContract.AssertSame("LunarLanderContinuous action_space.low", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.low(python.Object)), Outcome.Returned(ValueFacts.Of(space.Low)));
            ByteContract.AssertSame("LunarLanderContinuous action_space.high", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.high(python.Object)), Outcome.Returned(ValueFacts.Of(space.High)));
        }

        /// <summary>
        ///     The observation space: Gymnasium's v3 bounds (±2.5, ±2.5, ±10, ±10, ±2π, ±10, [-0, 1], [-0, 1]) against Gym.NET's
        ///     v2 ones (B-19), under which 6.5–16.3% of random-play observations fall outside the env's own space.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        [TestMethod]
        [OpenBugs("B-19")]
        [DataRow(false)]
        [DataRow(true)]
        public void GymNet_ObservationSpace_MatchesGymnasium(bool continuous) {
            var kwargs = new Dictionary<string, object?> { ["continuous"] = continuous };
            using PyRef python = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace("LunarLander-v3", Python.gymnasium.Env.observation_space, kwargs));
            var space = (Box)new LunarLanderEnv(NullEnvViewer.Factory, continuous: continuous).ObservationSpace;
            ByteContract.AssertSame("LunarLander observation_space.low", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.low(python.Object)), Outcome.Returned(ValueFacts.Of(space.Low)));
            ByteContract.AssertSame("LunarLander observation_space.high", Outcome.CapturePython(() => Python.gymnasium.spaces.box.Box.high(python.Object)), Outcome.Returned(ValueFacts.Of(space.High)));
        }

        /// <summary>
        ///     A whole episode of Gymnasium's heuristic (<c>lunar_lander.heuristic</c>, evaluated on Gymnasium's observations and
        ///     fed to both sides) in lockstep. Gym.NET's Aether simulation leaves Box2D's trajectory at once (F6), from a
        ///     different initial state (B-17), so this differs at the reset.
        /// </summary>
        /// <param name="continuous">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [OpenBugs("F6", "P8", "B-17", "B-19")]
        [DataRow(false, 0L)]
        [DataRow(false, 42L)]
        [DataRow(true, 0L)]
        [DataRow(true, 42L)]
        public void GymNet_HeuristicEpisode_MatchesGymnasium(bool continuous, long seed) {
            var kwargs = new Dictionary<string, object?> { ["continuous"] = continuous };
            using var pair = new GymPair(PyGymEnv.Make("LunarLander-v3", kwargs), new GymNetEnv(new LunarLanderEnv(NullEnvViewer.Factory, continuous: continuous)));
            PairRun run = pair.Run(seed, ActionSources.Policy("lunar_lander_heuristic"), maxSteps: 1000);
            TestContext.WriteLine(run.ToString());
        }
    }
}
