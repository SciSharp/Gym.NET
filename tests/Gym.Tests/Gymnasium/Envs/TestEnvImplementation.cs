#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Environments.Envs.Box2D;
using Gym.Environments.Envs.Classic;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using Gym.Vector;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Gym.Tests.Helpers;

namespace Gym.Tests.Gymnasium.Envs {
    /// <summary>
    ///     Gymnasium's env implementation tests (<c>tests/envs/test_env_implementation.py</c>): LunarLander's heuristic and
    ///     wind seeding, BipedalWalker's hardcore terrain, CliffWalking's transition tables, FrozenLake's map generation, the
    ///     classic envs' customizable resets, CartPole's vector equivalence, and the discrete action check.
    /// </summary>
    /// <remarks>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py">tests/envs/test_env_implementation.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestEnvImplementation {
        /// <summary>The heuristic lands LunarLander-v3 from seed 1 with a total reward above 100.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L12">test_lunar_lander_heuristics</see>.</remarks>
        [TestMethod]
        public void TestLunarLanderHeuristics() {
            using IEnv lunarLander = Make("LunarLander-v3", disableEnvChecker: true);
            double totalReward = LunarLanderEnv.DemoHeuristicLander(lunarLander, seed: 1);
            Assert.IsTrue(totalReward > 100, $"total reward {totalReward}");
        }

        /// <summary>The wind and torque indices follow the reset seed: the same seed draws the same ones, other seeds others.</summary>
        /// <param name="seed">The seed.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L20">test_lunar_lander_random_wind_seed</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(10)]
        [DataRow(20)]
        [DataRow(30)]
        [DataRow(40)]
        public void TestLunarLanderRandomWindSeed(int seed) {
            using IEnv made = Make("LunarLander-v3", new { enable_wind = true }, disableEnvChecker: true);
            var lunarLander = (LunarLanderEnv)made.Unwrapped;
            lunarLander.Reset(seed).Observation.Dispose();

            // Test that same seed gives same wind
            (long w1, long t1) = (lunarLander.WindIdx, lunarLander.TorqueIdx);
            lunarLander.Reset(seed).Observation.Dispose();
            (long w2, long t2) = (lunarLander.WindIdx, lunarLander.TorqueIdx);
            Assert.IsTrue(w1 == w2 && t1 == t2, "Setting same seed caused different initial wind or torque index");

            // Test that different seed gives different wind. There is a small chance that different seeds causes same number
            // so test 10 times (with different seeds) to make this chance incredibly tiny.
            for (int i = 1; i < 11; i++) {
                lunarLander.Reset(seed + i).Observation.Dispose();
                (long w3, long t3) = (lunarLander.WindIdx, lunarLander.TorqueIdx);
                if (w2 != w3 && t1 != t3)
                    return; // Found different initial values
            }

            Assert.Fail("Setting different seed caused same initial wind or torque index");
        }

        /// <summary>
        ///     CarRacing's domain randomization: new colours at every reset, the same ones with <c>options={"randomize": False}</c>.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L50">test_carracing_domain_randomize</see>.</remarks>
        [TestMethod]
        public void TestCarracingDomainRandomize() {
            using IEnv made = Make("CarRacing-v3", new { domain_randomize = true });
            IEnv env = made.Unwrapped;
            NDArray Color(string name) => (NDArray)env.GetWrapperAttr(name)!;

            using NDArray roadColor = Color("road_color").copy(), bgColor = Color("bg_color").copy(), grassColor = Color("grass_color").copy();
            Release(env.Reset(options: new Dictionary<string, object?> { ["randomize"] = false }));
            Assert.IsTrue(np.array_equal(roadColor, Color("road_color")), "Have different road color after reset with randomize turned off.");
            Assert.IsTrue(np.array_equal(bgColor, Color("bg_color")), "Have different bg color after reset with randomize turned off.");
            Assert.IsTrue(np.array_equal(grassColor, Color("grass_color")), "Have different grass color after reset with randomize turned off.");

            Release(env.Reset());
            Assert.IsTrue(np.all(np.not_equal(roadColor, Color("road_color"))), "Have same road color after reset.");
            Assert.IsTrue(np.all(np.not_equal(bgColor, Color("bg_color"))), "Have same bg color after reset.");
            Assert.IsTrue(np.all(np.not_equal(grassColor, Color("grass_color"))), "Have same grass color after reset.");
        }

        /// <summary>Slippery CliffWalking: every action of every state has three transitions of probability 1/3.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L87">test_slippery_cliffwalking</see>.</remarks>
        [TestMethod]
        public void TestSlipperyCliffwalking() {
            using var envs = new CliffWalkingEnv(isSlippery: true);
            foreach (IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>> actionsDict in envs.P.Values) {
                foreach (IReadOnlyList<ToyTextTransition> transitions in actionsDict.Values) {
                    Assert.AreEqual(3, transitions.Count);
                    Assert.IsTrue(transitions.All(r => r.Probability == 1.0 / 3));
                }
            }
        }

        /// <summary>Dry CliffWalking: every action of every state has one certain transition.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L99">test_cliffwalking</see>.</remarks>
        [TestMethod]
        public void TestCliffwalking() {
            using var env = new CliffWalkingEnv(isSlippery: false);
            foreach (IReadOnlyDictionary<long, IReadOnlyList<ToyTextTransition>> actionsDict in env.P.Values) {
                foreach (IReadOnlyList<ToyTextTransition> transitions in actionsDict.Values) {
                    Assert.AreEqual(1, transitions.Count);
                    Assert.IsTrue(transitions.All(r => r.Probability == 1.0));
                }
            }
        }

        /// <summary>
        ///     BipedalWalker with <c>hardcore=True</c> has ladders, stumps and pitfalls, which a convenient way identifies: their
        ///     white and grey terrain colours, absent from the normal variant's terrain and present in the hardcore one's.
        /// </summary>
        /// <param name="seed">The environment seed.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L107">test_bipedal_walker_hardcore_creation</see>.</remarks>
        [TestMethod]
        [DataRow(0)]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(3)]
        [DataRow(4)]
        public void TestBipedalWalkerHardcoreCreation(int seed) {
            var hcTerrainsColor1 = new Gym.Rendering.Color(255, 255, 255);
            var hcTerrainsColor2 = new Gym.Rendering.Color(153, 153, 153);

            using IEnv envMade = Make("BipedalWalker-v3", disableEnvChecker: true);
            using IEnv hcEnvMade = Make("BipedalWalkerHardcore-v3", disableEnvChecker: true);
            var env = envMade.Unwrapped as BipedalWalkerEnv;
            var hcEnv = hcEnvMade.Unwrapped as BipedalWalkerEnv;
            Assert.IsTrue(env != null && hcEnv != null);
            Assert.IsTrue(env.Hardcore == false && hcEnv.Hardcore == true);

            env.Reset(seed).Observation.Dispose();
            hcEnv.Reset(seed).Observation.Dispose();

            foreach ((Gym.Rendering.Color color1, Gym.Rendering.Color color2) in env.TerrainColors) {
                Assert.AreNotEqual(hcTerrainsColor1, color1);
                Assert.AreNotEqual(hcTerrainsColor2, color2);
            }

            int hcTerrainsColor1Count = 0;
            int hcTerrainsColor2Count = 0;
            foreach ((Gym.Rendering.Color color1, Gym.Rendering.Color color2) in hcEnv.TerrainColors) {
                if (color1 == hcTerrainsColor1)
                    hcTerrainsColor1Count += 1;
                if (color2 == hcTerrainsColor2)
                    hcTerrainsColor2Count += 1;
            }

            Assert.IsTrue(hcTerrainsColor1Count > 0);
            Assert.IsTrue(hcTerrainsColor2Count > 0);
        }

        /// <summary>FrozenLake's random maps of sizes 5, 10 and 16 always have a path from the start to the goal.</summary>
        /// <param name="mapSize">The size.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L147">test_frozenlake_dfs_map_generation</see>.</remarks>
        [TestMethod]
        [DataRow(5)]
        [DataRow(10)]
        [DataRow(16)]
        public void TestFrozenlakeDfsMapGeneration(int mapSize) {
            IReadOnlyList<string> newFrozenlake = FrozenLakeEnv.GenerateRandomMap(mapSize);
            Assert.AreEqual(mapSize, newFrozenlake.Count);
            Assert.AreEqual(mapSize, newFrozenlake[0].Length);

            // Runs a depth first search through the map to find the path.
            (int Row, int Col)[] directions = { (1, 0), (0, 1), (-1, 0), (0, -1) };
            var frontier = new Stack<(int Row, int Col)>();
            var discovered = new HashSet<(int Row, int Col)>();
            frontier.Push((0, 0));
            while (frontier.Count > 0) {
                (int row, int col) = frontier.Pop();
                if (!discovered.Add((row, col)))
                    continue;
                foreach ((int rowDirection, int colDirection) in directions) {
                    int newRow = row + rowDirection, newCol = col + colDirection;
                    if (0 <= newRow && newRow < mapSize && 0 <= newCol && newCol < mapSize) {
                        if (newFrozenlake[newRow][newCol] == 'G')
                            return; // Successful, a route through the map was found
                        if (!"#H".Contains(newFrozenlake[newRow][newCol]))
                            frontier.Push((newRow, newCol));
                    }
                }
            }

            Assert.Fail("No path through the frozenlake was found.");
        }

        /// <summary>A seeded map is the same for the same seed and differs for the next seed.</summary>
        /// <param name="mapSize">The size.</param>
        /// <param name="seed">The seed.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L178">test_frozenlake_map_generation_with_seed</see>.</remarks>
        [TestMethod]
        [DataRow(5, 123)]
        [DataRow(10, 42)]
        [DataRow(16, 987)]
        public void TestFrozenlakeMapGenerationWithSeed(int mapSize, int seed) {
            CollectionAssert.AreEqual(FrozenLakeEnv.GenerateRandomMap(size: mapSize, seed: seed).ToList(), FrozenLakeEnv.GenerateRandomMap(size: mapSize, seed: seed).ToList());
            CollectionAssert.AreNotEqual(FrozenLakeEnv.GenerateRandomMap(size: mapSize, seed: seed).ToList(), FrozenLakeEnv.GenerateRandomMap(size: mapSize, seed: seed + 1).ToList());
        }

        /// <summary>The bounds of <see cref="TestCustomizableResets"/>: none, Python floats, and 0-d NumPy arrays.</summary>
        public static IEnumerable<object?[]> CustomizableResetCases =>
            from envName in new[] { "Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0" }
            from lowHigh in new[] { "none", "floats", "arrays" }
            select new object?[] { envName, lowHigh };

        /// <summary>The classic envs reset within custom <c>low</c>/<c>high</c> options and then step.</summary>
        /// <param name="envName">The id.</param>
        /// <param name="lowHigh">Which bounds: <c>none</c>, <c>floats</c> (-0.4, 0.4) or <c>arrays</c> (the same as 0-d arrays).</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L194">test_customizable_resets</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(CustomizableResetCases))]
        public void TestCustomizableResets(string envName, string lowHigh) {
            using IEnv env = Make(envName);
            env.ActionSpace.Seed(0);
            // First ensure we can do a reset.
            if (lowHigh == "none") {
                Release(env.Reset());
            } else {
                using NDArray? lowArray = lowHigh == "arrays" ? NDArray.Scalar<double>(-0.4) : null;
                using NDArray? highArray = lowHigh == "arrays" ? NDArray.Scalar<double>(0.4) : null;
                object low = lowArray ?? (object)(-0.4), high = highArray ?? (object)0.4;
                Release(env.Reset(options: new Dictionary<string, object?> { ["low"] = low, ["high"] = high }));
                var state = (double[])PyAttributes.GetAttr(env.Unwrapped, "state")!;
                Assert.IsTrue(state.All(v => v >= -0.4 && v <= 0.4), $"state [{string.Join(", ", state)}] outside [-0.4, 0.4]");
            }

            // Make sure we can take a step.
            object? action = env.ActionSpace.Sample();
            StepResult<object?> step = env.Step(action);
            Release(action, step.Observation);
        }

        /// <summary>Pendulum resets with custom <c>x_init</c>/<c>y_init</c> options and then steps.</summary>
        /// <param name="lowHigh">Which bounds: <c>none</c>, <c>floats</c> (1.2, 1.0) or <c>arrays</c> (the same as 0-d arrays).</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L217">test_customizable_pendulum_resets</see>.</remarks>
        [TestMethod]
        [DataRow("none")]
        [DataRow("floats")]
        [DataRow("arrays")]
        public void TestCustomizablePendulumResets(string lowHigh) {
            using IEnv env = Make("Pendulum-v1");
            env.ActionSpace.Seed(0);
            // First ensure we can do a reset and the values are within expected ranges.
            if (lowHigh == "none") {
                Release(env.Reset());
            } else {
                using NDArray? xArray = lowHigh == "arrays" ? NDArray.Scalar<double>(1.2) : null;
                using NDArray? yArray = lowHigh == "arrays" ? NDArray.Scalar<double>(1.0) : null;
                object x = xArray ?? (object)1.2, y = yArray ?? (object)1.0;
                // Pendulum is initialized a little differently than the other environments, where we specify the x and y values
                // for the upper limit (and lower limit is just the negative of it).
                Release(env.Reset(options: new Dictionary<string, object?> { ["x_init"] = x, ["y_init"] = y }));
            }

            // Make sure we can take a step.
            object? action = env.ActionSpace.Sample();
            StepResult<object?> step = env.Step(action);
            Release(action, step.Observation);
        }

        /// <summary>The refused bounds of <see cref="TestInvalidCustomizableResets"/>: strings, a lower bound above the upper, lists, arrays.</summary>
        public static IEnumerable<object?[]> InvalidResetCases =>
            from envName in new[] { "Acrobot-v1", "CartPole-v1", "MountainCar-v0", "MountainCarContinuous-v0" }
            from lowHigh in new[] { "strings", "inverted", "lists", "arrays" }
            select new object?[] { envName, lowHigh };

        /// <summary>The classic envs refuse bounds that are not numbers, or a lower bound above the upper, with <see cref="ValueError"/>.</summary>
        /// <param name="envName">The id.</param>
        /// <param name="lowHigh">Which bounds: <c>strings</c>, <c>inverted</c> (10.0, 8.0), <c>lists</c> or <c>arrays</c> of two values.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L246">test_invalid_customizable_resets</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(InvalidResetCases))]
        public void TestInvalidCustomizableResets(string envName, string lowHigh) {
            using IEnv env = Make(envName);
            using NDArray lowArray = np.array(new[] { -1.0, -1.0 }), highArray = np.array(new[] { 1.0, 1.0 });
            (object low, object high) = lowHigh switch {
                "strings" => ("x", "y"),
                "inverted" => (10.0, 8.0),
                "lists" => (new List<object?> { -1.0, -1.0 }, new List<object?> { 1.0, 1.0 }),
                _ => ((object)lowArray, (object)highArray),
            };
            Assert.ThrowsExactly<ValueError>(() => Release(env.Reset(options: new Dictionary<string, object?> { ["low"] = low, ["high"] = high })));
        }

        /// <summary>
        ///     CartPole-v1 and its one-env vector entry point step alike: the same spaces, observations, rewards, flags, infos and
        ///     states for up to 100 steps from a random seed, and after an episode end the vector env's autoreset step equals a
        ///     reset of the single env.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L255">test_cartpole_vector_equiv</see>.</remarks>
        [TestMethod]
        public void TestCartpoleVectorEquiv() {
            using var env = (Env<NDArray, NDArray>)Make("CartPole-v1");
            using IVectorEnv envs = MakeVec("CartPole-v1", numEnvs: 1);
            var vector = (CartPoleVectorEnv)envs.Unwrapped;

            Assert.AreEqual(env.ActionSpace, envs.SingleActionSpace);
            Assert.AreEqual(env.ObservationSpace, envs.SingleObservationSpace);

            long seed = new Random().Next(0, 1000);

            // reset
            (NDArray obs, Info info) = env.Reset(seed);
            VectorResetResult<object?> vecReset = envs.Reset(seed);
            var vecObs = (NDArray)vecReset.Observations!;
            env.ActionSpace.Seed(seed);

            Assert.IsTrue(env.ObservationSpace.Contains(obs));
            Assert.IsTrue(envs.ObservationSpace.Contains(vecObs));
            using (NDArray first = vecObs[0])
                Assert.IsTrue(np.array_equal(obs, first));
            PyAssert.AreEqual(info, vecReset.Infos);
            AssertStatesEqual(env, vector);

            // step
            bool term = false, trunc = false;
            for (int i = 0; i < 100; i++) {
                NDArray action = env.ActionSpace.Sample();
                using NDArray batched = np.array(new[] { action.GetAtIndex<long>(0) });
                Assert.IsTrue(envs.ActionSpace.Contains(batched));

                StepResult<NDArray> step = env.Step(action);
                VectorStepResult<object?> vecStep = envs.Step(batched);
                (obs, term, trunc) = (step.Observation, step.Terminated, step.Truncated);
                vecObs = (NDArray)vecStep.Observations!;

                Assert.IsTrue(env.ObservationSpace.Contains(obs));
                Assert.IsTrue(envs.ObservationSpace.Contains(vecObs));
                using (NDArray first = vecObs[0])
                    Assert.IsTrue(np.array_equal(obs, first));
                Assert.AreEqual(step.Reward, vecStep.Rewards.GetAtIndex<double>(0));
                Assert.AreEqual(term, vecStep.Terminations.GetAtIndex<bool>(0));
                Assert.AreEqual(trunc, vecStep.Truncations.GetAtIndex<bool>(0));
                PyAssert.AreEqual(step.Info, vecStep.Infos);
                AssertStatesEqual(env, vector);
                if (term || trunc)
                    break;
            }

            // if the sub-environment episode ended
            if (term || trunc) {
                (obs, info) = env.Reset();
                // the vector action shouldn't matter as autoreset
                Assert.IsTrue(vector.PrevDone.GetAtIndex<bool>(0));
                object? sampled = envs.ActionSpace.Sample();
                VectorStepResult<object?> vecStep = envs.Step(sampled);
                vecObs = (NDArray)vecStep.Observations!;

                Assert.IsTrue(env.ObservationSpace.Contains(obs));
                Assert.IsTrue(envs.ObservationSpace.Contains(vecObs));
                using (NDArray first = vecObs[0])
                    Assert.IsTrue(np.array_equal(obs, first));
                Assert.AreEqual(0.0, vecStep.Rewards.GetAtIndex<double>(0));
                Assert.IsFalse(vecStep.Terminations.GetAtIndex<bool>(0));
                Assert.IsFalse(vecStep.Truncations.GetAtIndex<bool>(0));
                PyAssert.AreEqual(info, vecStep.Infos);
                AssertStatesEqual(env, vector);
            }
        }

        /// <summary>
        ///     The discrete LunarLander-v3 refuses a continuous action (Gymnasium's assertion) and takes a sampled discrete one.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L324">test_discrete_action_validation</see>.</remarks>
        [TestMethod]
        public void TestDiscreteActionValidation_LunarLander() => AssertDiscreteActionValidation("LunarLander-v3");

        /// <summary>The same for CarRacing-v3, whose refusal is <c>InvalidAction</c>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/test_env_implementation.py#L324">test_discrete_action_validation</see>.</remarks>
        [TestMethod]
        public void TestDiscreteActionValidation_CarRacing() => AssertDiscreteActionValidation("CarRacing-v3");

        /// <summary>
        ///     <c>test_discrete_action_validation</c>'s body: a continuous env's sample stepped into the discrete env raises
        ///     <c>InvalidAction</c> or <c>AssertionError</c>, and a discrete sample steps.
        /// </summary>
        /// <param name="envId">The id.</param>
        private static void AssertDiscreteActionValidation(string envId) {
            // get continuous action
            object? continuousAction;
            using (IEnv continuousEnv = Make(envId, new { continuous = true }))
                continuousAction = continuousEnv.ActionSpace.Sample();

            // create discrete env
            using IEnv discreteEnv = Make(envId, new { continuous = false });
            Release(discreteEnv.Reset());

            // expect InvalidAction (caused by CarRacing) or AssertionError (caused by LunarLander)
            Exception? raised = null;
            try {
                discreteEnv.Step(continuousAction);
            } catch (Exception ex) when (ex is InvalidAction or AssertionError) {
                raised = ex;
            }

            Assert.IsNotNull(raised, "The discrete env must refuse a continuous action.");
            Release(continuousAction);

            // expect no error
            object? discreteAction = discreteEnv.ActionSpace.Sample();
            StepResult<object?> step = discreteEnv.Step(discreteAction);
            Release(discreteAction, step.Observation);
        }

        /// <summary>The single env's state equals the vector env's only column (<c>envs.unwrapped.state[:, 0]</c>).</summary>
        /// <param name="env">The single env.</param>
        /// <param name="vector">The one-env vector env.</param>
        private static void AssertStatesEqual(IEnv env, CartPoleVectorEnv vector) {
            var single = (double[])PyAttributes.GetAttr(env.Unwrapped, "state")!;
            NDArray state = vector.State!;
            for (int i = 0; i < single.Length; i++)
                Assert.AreEqual(single[i], state.GetDouble(i, 0), $"state[{i}]");
        }

        /// <summary>Releases the arrays a reset returned.</summary>
        /// <param name="result">The reset's result.</param>
        private static void Release(ResetResult<object?> result) => Release(result.Observation);

        /// <summary>Releases values that may be NDArrays the test owns.</summary>
        /// <param name="values">The values.</param>
        private static void Release(params object?[] values) {
            foreach (object? value in values)
                (value as IDisposable)?.Dispose();
        }
    }
}
