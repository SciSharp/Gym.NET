using Gym.Environments;
using Gym.Environments.Envs.Classic;
using Gym.Spaces;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Reference;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     CartPole-v1 against Gymnasium's <c>CartPoleEnv</c> (<c>gymnasium/envs/classic_control/cartpole.py</c> @ 9e04324),
    ///     live and in lockstep: first the float64 reference transcription (what the P4 port must produce, proven reachable),
    ///     then Gym.NET's own <see cref="CartPoleEnv"/>: reset, dynamics, frames and spaces.
    /// </summary>
    /// <remarks>
    ///     Every lockstep here compares the float64 <c>state</c> after each call too (Gymnasium's <c>env.unwrapped.state</c>,
    ///     Gym.NET's <c>state</c> field), so dynamics are compared at full precision even where the float32 observation would
    ///     round a difference away. Dynamics cells are pinned to the reference platform: they rest on the platform's
    ///     <c>cos</c>/<c>sin</c>.
    /// </remarks>
    [TestClass]
    public sealed class CartPoleLiveParityTests : ParityTestBase {
        /// <summary>The CartPole states of the port plan's pixel differential (A.9): centered, tilted, and left of center.</summary>
        public static readonly double[][] FrameStates = {
            new[] { 0.0, 0.0, 0.0, 0.0 },
            new[] { 0.5, 0.0, 0.15, 0.0 },
            new[] { -1.2, 0.0, -0.1, 0.0 },
        };

        /// <summary>
        ///     The reference transcription runs a whole episode in lockstep with Gymnasium on each of A.7's seeds: 500 steps of
        ///     the balancing policy, every observation, reward, flag, info, float64 state and PCG64 state identical. This is
        ///     the result the P4 port has to reproduce.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void Reference_MatchesGymnasium_ForAWholeEpisode(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.Faithful(), stateAttribute: "state");
            PairRun run = pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 500);
            TestContext.WriteLine(run.ToString());
            Assert.AreEqual(500, run.Steps);
        }

        /// <summary>
        ///     Termination and what follows it: pushing right every step ends the episode, with reward 1.0 on the terminating
        ///     step, and every further step returns reward 0.0 and <c>terminated</c> (Gymnasium's
        ///     <c>steps_beyond_terminated</c> branch), all in lockstep.
        /// </summary>
        [TestMethod]
        public void Reference_MatchesGymnasium_ThroughAndPastTermination() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), CartPoleReference.Faithful(), stateAttribute: "state");
            PairRun run = pair.Run(3, ActionSources.Constant(1), maxSteps: 500);
            Assert.IsTrue(run.Ended, "Pushing right every step must topple the pole.");
            for (int i = 0; i < 5; i++) {
                using var action = ParityAction.Discrete(1);
                pair.Step(action);
            }

            TestContext.WriteLine($"{run}; then 5 more steps past termination, identical");
        }

        /// <summary>
        ///     Gym.NET's reset, <c>Seed(seed)</c> then <c>Reset()</c>, against <c>reset(seed=seed)</c>. The float64 state and the
        ///     generator already match (F2); the observation is the float64 state itself instead of its float32 copy (B-13).
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [OpenBugs("B-13")]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void GymNet_Reset_MatchesGymnasium(long seed) {
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv(NullEnvViewer.Factory)), stateAttribute: "state");
            pair.Reset(seed);
        }

        /// <summary>
        ///     Gym.NET's CartPole in lockstep with Gymnasium's for a whole episode of the balancing policy. Today it differs at
        ///     the reset (the float64 observation, B-13), and its float32 constants move the state from step 1 on (B-11).
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [OpenBugs("B-11", "B-13")]
        [DataRow(0L)]
        [DataRow(1L)]
        [DataRow(42L)]
        [DataRow(1234L)]
        public void GymNet_Episode_MatchesGymnasium(long seed) {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv(NullEnvViewer.Factory)), stateAttribute: "state");
            pair.Run(seed, ActionSources.Policy("cartpole_balance"), maxSteps: 500);
        }

        /// <summary>
        ///     One step of Gym.NET's dynamics from a state written into both envs (state injection), isolated from reset and the
        ///     generator: the float64 state after the step differs from Gymnasium's in its last bits (B-11), next to the
        ///     observation's dtype (B-13).
        /// </summary>
        [TestMethod]
        [OpenBugs("B-11", "B-13")]
        public void GymNet_Dynamics_MatchGymnasium_FromAnInjectedState() {
            SkipOffReferencePlatform("CartPole's cos/sin");
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1"), new GymNetEnv(new CartPoleEnv(NullEnvViewer.Factory)), stateAttribute: "state");
            pair.InjectState("state", 0.01, -0.02, 0.03, 0.04);
            using var action = ParityAction.Discrete(1);
            pair.Step(action);
        }

        /// <summary>
        ///     Gym.NET's <c>rgb_array</c> frames against pygame's at A.9's states. Today about 1.1% of the pixels differ (pole
        ///     colour, axle colour, ImageSharp's anti-aliasing against SDL_gfx; B-16); the artifacts show where.
        /// </summary>
        /// <param name="state">The index into <see cref="FrameStates"/>.</param>
        [TestMethod]
        [OpenBugs("B-16")]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        public void GymNet_Frame_MatchesGymnasium(int state) {
            using var pair = new GymPair(PyGymEnv.Make("CartPole-v1", renderMode: "rgb_array"), new GymNetEnv(new CartPoleEnv(NullEnvViewer.Factory)), stateAttribute: "state");
            pair.InjectState("state", FrameStates[state]);
            pair.CompareFrame($"state {state}");
        }

        /// <summary>
        ///     The observation space: Gymnasium's bounds are <c>±[4.8, inf, 0.4189, inf]</c> as float32; Gym.NET uses
        ///     <c>float.MaxValue</c> where Gymnasium has infinity (B-12), which also flips <c>bounded_above</c>/<c>bounded_below</c>.
        /// </summary>
        [TestMethod]
        [OpenBugs("B-12")]
        public void GymNet_ObservationSpace_MatchesGymnasium() {
            using PyHandle python = PyGym.Evaluate("gymnasium.make('CartPole-v1').unwrapped.observation_space");
            var space = (Box)new CartPoleEnv(NullEnvViewer.Factory).ObservationSpace;
            ByteContract.AssertSame("CartPole observation_space.high", python.Get("high"), Outcome.Returned(ValueFacts.Of(space.High)));
            ByteContract.AssertSame("CartPole observation_space.low", python.Get("low"), Outcome.Returned(ValueFacts.Of(space.Low)));
            ByteContract.AssertSame("CartPole observation_space.bounded_above", python.Get("bounded_above"), Outcome.Returned(ValueFacts.Of(space.BoundedHigh)));
        }

        /// <summary>
        ///     The action space, <c>Discrete(2)</c>: its <c>n</c>, <c>start</c> and dtype, and seeded samples with the generator
        ///     state behind them, are Gymnasium's (the shape is covered by <see cref="Spaces.DiscreteLiveParityTests.Shape_MatchesGymnasium"/>).
        /// </summary>
        [TestMethod]
        public void GymNet_ActionSpace_MatchesGymnasium() {
            using PyHandle python = PyGym.Evaluate("gymnasium.make('CartPole-v1').unwrapped.action_space");
            var space = (Discrete)new CartPoleEnv(NullEnvViewer.Factory).ActionSpace;
            ByteContract.AssertSame("CartPole action_space.n", python.Get("n"), Outcome.Returned(ValueFacts.Of(space.N)));
            ByteContract.AssertSame("CartPole action_space.start", python.Get("start"), Outcome.Returned(ValueFacts.Of(space.Start)));
            ByteContract.AssertSame("CartPole action_space.dtype.str", python.Evaluate("x.dtype.str"), Outcome.Returned(ValueFacts.FromText(space.DType.str)));

            python.Invoke("seed", 5L).Expect("action_space.seed(5)");
            space.Seed(5);
            for (int i = 1; i <= 20; i++)
                ByteContract.AssertSame($"CartPole action_space.sample() #{i} after seed(5)", python.Invoke("sample"), Outcome.Capture(() => space.Sample()));
            ByteContract.AssertSame("CartPole action_space generator state", python.Evaluate("x.np_random.bit_generator.state"), Outcome.Returned(GeneratorProbe.StateOf(space)));
        }
    }
}
