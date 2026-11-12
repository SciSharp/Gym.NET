#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.MuJoCo;
using Gym.Observations;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.MuJoCo {
    /// <summary>
    ///     Gymnasium's MuJoCo v4/v5 tests (<c>tests/envs/mujoco/test_mujoco_v5.py</c>) on Gym.NET's engine: the info entries
    ///     against the simulation, the survival rewards, every v5 env under five frame skips through <c>check_env</c>, the
    ///     reward as the sum of its info terms, v4 against v5, the models' object counts and arrays, <c>dt</c>, the reset's
    ///     noise and determinism.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Pusher-v4 is skipped where Gymnasium skips it on MuJoCo 3 and later: Gym.NET's engine follows MuJoCo 4.0.0
    ///         (DR-64), and the id refuses at construction as Gymnasium's does.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py">tests/envs/mujoco/test_mujoco_v5.py</see>.</para>
    /// </remarks>
    [TestClass]
    public class TestMujocoV5 {
        /// <summary>Gymnasium's <c>ALL_MUJOCO_ENVS</c>: the 11 env names, each with a v4 and a v5.</summary>
        private static readonly string[] AllMujocoEnvs = {
            "Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup", "InvertedDoublePendulum", "InvertedPendulum", "Pusher",
            "Reacher", "Swimmer", "Walker2d",
        };

        /// <summary>
        ///     <c>check_env</c>'s warnings a MuJoCo env may raise (Gymnasium's <c>CHECK_ENV_IGNORE_WARNINGS</c>): the unbounded
        ///     observation space and a control range other than [-1, 1].
        /// </summary>
        private static readonly string[] CheckEnvIgnoreWarnings = new[] {
            "A Box observation space minimum value is -infinity. This is probably too low.",
            "A Box observation space maximum value is infinity. This is probably too high.",
            "For Box action spaces, we recommend using a symmetric and normalized space (range=[-1, 1] or [0, 1]). See https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html for more information.",
        }.Select(message => $"\u001b[33mWARN: {message}\u001b[0m").ToArray();

        /// <summary>The ids of the x-position test, each made with the root's x observed.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object[]> XPositionIds() => new[] {
            "Ant-v5", "HalfCheetah-v5", "HalfCheetah-v4", "Hopper-v5", "Hopper-v4", "Humanoid-v5", "HumanoidStandup-v5", "Swimmer-v5",
            "Swimmer-v4", "Walker2d-v5", "Walker2d-v4",
        }.Select(id => new object[] { id });

        /// <summary>The first observation is the root's x, which the info reports as <c>x_position</c>.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L47">test_verify_info_x_position</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(XPositionIds))]
        public void TestVerifyInfoXPosition(string envId) {
            using IEnv env = Make(envId, new { exclude_current_positions_from_observation = false });
            Release(env.Reset());
            using var step = env.Step(env.ActionSpace.Sample());
            var obs = (NDArray)step.Observation!;
            Assert.AreEqual(obs.GetDouble(0), Num(step.Info["x_position"]));
        }

        /// <summary>The second observation is the root's y, which the info reports as <c>y_position</c>.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L70">test_verify_info_y_position</see>.</remarks>
        [TestMethod]
        [DataRow("Ant-v5")]
        [DataRow("Humanoid-v5")]
        [DataRow("HumanoidStandup-v5")]
        [DataRow("Swimmer-v5")]
        [DataRow("Swimmer-v4")]
        public void TestVerifyInfoYPosition(string envId) {
            using IEnv env = Make(envId, new { exclude_current_positions_from_observation = false });
            Release(env.Reset());
            using var step = env.Step(env.ActionSpace.Sample());
            var obs = (NDArray)step.Observation!;
            Assert.AreEqual(obs.GetDouble(1), Num(step.Info["y_position"]));
        }

        /// <summary><c>x_velocity</c> is the root's x displacement over <c>dt</c>.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L83">test_verify_info_x_velocity</see>.</remarks>
        [TestMethod]
        [DataRow("HalfCheetah", "v5")]
        [DataRow("HalfCheetah", "v4")]
        [DataRow("Hopper", "v5")]
        [DataRow("Hopper", "v4")]
        [DataRow("Swimmer", "v5")]
        [DataRow("Swimmer", "v4")]
        [DataRow("Walker2d", "v5")]
        [DataRow("Walker2d", "v4")]
        public void TestVerifyInfoXVelocity(string envName, string version) {
            using IEnv made = Make($"{envName}-{version}");
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            double oldX = env.Data.qpos[0];
            using var step = env.Step(SampleOf(env));
            double newX = env.Data.qpos[0];
            double dx = newX - oldX;
            double velX = dx / env.Dt;
            Assert.AreEqual(velX, Num(step.Info["x_velocity"]));
        }

        /// <summary><c>y_velocity</c> is the root's y displacement over <c>dt</c>.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L100">test_verify_info_y_velocity</see>.</remarks>
        [TestMethod]
        [DataRow("Swimmer-v5")]
        [DataRow("Swimmer-v4")]
        public void TestVerifyInfoYVelocity(string envId) {
            using IEnv made = Make(envId);
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            double oldY = env.Data.qpos[1];
            using var step = env.Step(SampleOf(env));
            double newY = env.Data.qpos[1];
            double dy = newY - oldY;
            double velY = dy / env.Dt;
            Assert.AreEqual(velY, Num(step.Info["y_velocity"]));
        }

        /// <summary>Ant's velocities are the torso frame's displacement over <c>dt</c>.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L117">test_verify_info_xy_velocity_xpos</see>.</remarks>
        [TestMethod]
        [DataRow("Ant-v5")]
        [DataRow("Ant-v4")]
        public void TestVerifyInfoXyVelocityXpos(string envId) {
            using IEnv made = Make(envId);
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            (double oldX, double oldY) = TorsoXy(env);
            using var step = env.Step(SampleOf(env));
            (double newX, double newY) = TorsoXy(env);
            double velX = (newX - oldX) / env.Dt, velY = (newY - oldY) / env.Dt;
            Assert.AreEqual(velX, Num(step.Info["x_velocity"]));
            Assert.AreEqual(velY, Num(step.Info["y_velocity"]));
        }

        /// <summary>The humanoid's velocities are its center of mass's displacement over <c>dt</c>, the test's own center of mass.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L135">test_verify_info_xy_velocity_com</see>.</remarks>
        [TestMethod]
        [DataRow("Humanoid-v5")]
        [DataRow("Humanoid-v4")]
        public void TestVerifyInfoXyVelocityCom(string envId) {
            // np.sum(mass * xpos, axis=0) adds row by row from 0; np.sum(mass) is NumPy's pairwise sum of the masses
            (double X, double Y) MassCenter(MujocoEnv e) {
                double x = 0, y = 0;
                for (int b = 0; b < e.Model.nbody; b++) {
                    x += e.Model.body_mass[b] * e.Data.xipos[3 * b];
                    y += e.Model.body_mass[b] * e.Data.xipos[3 * b + 1];
                }
                double mass = MujocoMath.Sum(e.Model.body_mass);
                return (x / mass, y / mass);
            }

            using IEnv made = Make(envId);
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            (double oldX, double oldY) = MassCenter(env);
            using var step = env.Step(SampleOf(env));
            (double newX, double newY) = MassCenter(env);
            Assert.AreEqual((newX - oldX) / env.Dt, Num(step.Info["x_velocity"]));
            Assert.AreEqual((newY - oldY) / env.Dt, Num(step.Info["y_velocity"]));
        }

        /// <summary>The v5 envs' rows of the survival-reward test.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object[]> SurviveEnvs() =>
            new[] { "Ant", "Hopper", "Humanoid", "InvertedDoublePendulum", "InvertedPendulum", "Walker2d" }.Select(name => new object[] { name, "v5" });

        /// <summary><c>reward_survive</c> is 0 on the terminal step and nonzero before it.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L176">test_verify_reward_survive</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(SurviveEnvs))]
        public void TestVerifyRewardSurvive(string envName, string version) {
            using IEnv made = Make($"{envName}-{version}", new { reset_noise_scale = 0 });
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset(0));
            env.ActionSpace.Seed(2);

            bool terminal = false;
            for (int i = 0; i < 80; i++) {
                using var step = env.Step(SampleOf(env));
                terminal = step.Terminated;
                if (terminal) {
                    Assert.AreEqual(0.0, Num(step.Info["reward_survive"]));
                    break;
                }
                Assert.AreNotEqual(0.0, Num(step.Info["reward_survive"]));
            }

            Assert.IsTrue(terminal, "The environment, should have terminated, if not the test is not valid.");
        }

        /// <summary>The frame skips of <see cref="TestFrameSkip"/>: every v5 env with 1 to 5.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object[]> FrameSkipRows() =>
            from name in AllMujocoEnvs from frameSkip in Enumerable.Range(1, 5) select new object[] { name, "v5", frameSkip };

        /// <summary>Every v5 env passes <c>check_env</c> under frame skips 1 to 5, with only the expected warnings.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <param name="frameSkip">The frame skip.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L203">test_frame_skip</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(FrameSkipRows))]
        public void TestFrameSkip(string envName, string version, int frameSkip) {
            string envId = $"{envName}-{version}";
            using IEnv env = Make(envId, new { frame_skip = frameSkip });

            // Test if env adheres to Gym API
            using WarningCatcher w = Warnings.Catch();
            EnvChecker.CheckEnv(env.Unwrapped, skipRenderCheck: true);
            env.Close();
            foreach (WarningMessage warning in w.Records) {
                if (!CheckEnvIgnoreWarnings.Contains(warning.Message))
                    Assert.Fail($"Unexpected warning: {warning.Message}");
            }
        }

        /// <summary>Every v5 reward is the sum of its info terms, in Gymnasium's order and arithmetic, over 100 steps.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L221">test_reward_sum</see>.</remarks>
        [TestMethod]
        [DataRow("v5")]
        public void TestRewardSum(string version) {
            const int numSteps = 100;
            // every sum below meets a float64 operand, so NumPy computes it in float64 after widening the float32 terms exactly
            CheckRewardSum($"Ant-{version}", numSteps, i => (V(i, "reward_survive") + V(i, "reward_forward")) - (-V(i, "reward_ctrl") + -V(i, "reward_contact")));
            CheckRewardSum($"HalfCheetah-{version}", numSteps, i => V(i, "reward_forward") + V(i, "reward_ctrl"));
            CheckRewardSum($"Hopper-{version}", numSteps, i => V(i, "reward_forward") + V(i, "reward_survive") + V(i, "reward_ctrl"));
            CheckRewardSum($"Humanoid-{version}", numSteps, i => (V(i, "reward_forward") + V(i, "reward_survive")) + (V(i, "reward_ctrl") + V(i, "reward_contact")));
            CheckRewardSum($"HumanoidStandup-{version}", numSteps, i => V(i, "reward_linup") + V(i, "reward_quadctrl") + V(i, "reward_impact") + 1);
            CheckRewardSum($"InvertedDoublePendulum-{version}", numSteps, i => V(i, "reward_survive") + V(i, "distance_penalty") + V(i, "velocity_penalty"));
            CheckRewardSum($"InvertedPendulum-{version}", numSteps, i => V(i, "reward_survive"));
            CheckRewardSum($"Pusher-{version}", numSteps, i => V(i, "reward_dist") + V(i, "reward_ctrl") + V(i, "reward_near"));
            CheckRewardSum($"Reacher-{version}", numSteps, i => V(i, "reward_dist") + V(i, "reward_ctrl"));
            CheckRewardSum($"Swimmer-{version}", numSteps, i => V(i, "reward_forward") + V(i, "reward_ctrl"));
            CheckRewardSum($"Walker2d-{version}", numSteps, i => V(i, "reward_forward") + V(i, "reward_survive") + V(i, "reward_ctrl"));
        }

        /// <summary>The v4 to v5 transitions of <see cref="TestIdenticalBehaviourV45"/>: what each pair may differ in.</summary>
        /// <returns>The data rows: name, skip observations, skip rewards, skip terminations, info comparison.</returns>
        public static IEnumerable<object[]> V45Rows() => new[] {
            new object[] { "Ant", true, true, false, "skip" },
            new object[] { "HalfCheetah", false, false, false, "skip" },
            new object[] { "Hopper", false, true, false, "superset" },
            // skipping humanoid, everything has changed
            new object[] { "HumanoidStandup", true, false, false, "superset" },
            new object[] { "InvertedDoublePendulum", true, true, false, "superset" },
            new object[] { "InvertedPendulum", false, true, false, "superset" },
            new object[] { "Pusher", true, true, false, "keys-superset" },
            new object[] { "Reacher", true, true, false, "keys-equivalence" },
            new object[] { "Swimmer", false, false, false, "skip" },
            new object[] { "Walker2d", true, true, true, "keys-superset" },
        };

        /// <summary>v4 and v5 match over 100 steps where Gymnasium's transition says they should.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="obs">Whether the observations may differ.</param>
        /// <param name="rew">Whether the rewards may differ.</param>
        /// <param name="term">Whether the terminations may differ.</param>
        /// <param name="info">How the infos are compared.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L313">test_identical_behaviour_v45</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(V45Rows))]
        public void TestIdenticalBehaviourV45(string envName, bool obs, bool rew, bool term, string info) {
            const int numSteps = 100;
            if (envName == "Pusher")
                Assert.Inconclusive("Pusher-v4 is not compatible with mujoco >= 3");

            using IEnv envV4 = Make($"{envName}-v4");
            using IEnv envV5 = Make($"{envName}-v5");
            EnvMatch.CheckEnvironmentsMatch(envV4, envV5, numSteps, skipObs: obs, skipRew: rew, skipTerminal: term, infoComparison: info);
        }

        /// <summary>The ant's root position is its torso frame's, before a step and after one followed by the kinematics.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L341">test_ant_com</see>.</remarks>
        [TestMethod]
        [DataRow("v5")]
        [DataRow("v4")]
        public void TestAntCom(string version) {
            // `env` contains `data : MjData` and `model : MjModel`
            using IEnv made = Make($"Ant-{version}");
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset()); // randomly initializes the `data.qpos` and `data.qvel`, calls mujoco.mj_forward(env.model, env.data)

            double xPositionBefore = env.Data.qpos[0];
            double xPositionBeforeCom = TorsoXy(env).X;
            Assert.AreEqual(xPositionBefore, xPositionBeforeCom, "before failed"); // This succeeds

            // This calls mujoco.mj_step(env.model, env.data, nstep=env.frame_skip)
            using (env.Step(SampleOf(env))) { }
            Mujoco.MjKinematics(env.Model, env.Data);

            double xPositionAfter = env.Data.qpos[0];
            double xPositionAfterCom = TorsoXy(env).X;
            Assert.AreEqual(xPositionAfter, xPositionAfterCom, "after failed"); // This succeeds
        }

        /// <summary><c>set_state</c> writes the positions and velocities given.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L364">test_set_state</see>.</remarks>
        [TestMethod]
        [DataRow("v5")]
        [DataRow("v4")]
        public void TestSetState(string version) {
            using IEnv made = Make($"Hopper-{version}");
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            double[] newQpos = { 0.00136962, 1.24769787, -0.00459026, -0.00483472, 0.0031327, 0.00412756 };
            double[] newQvel = { 0.00106636, 0.00229497, 0.00043625, 0.00435072, 0.00315854, -0.00497261 };
            using NDArray qpos = np.array(newQpos), qvel = np.array(newQvel);
            env.SetState(qpos, qvel);
            CollectionAssert.AreEqual(newQpos, env.Data.qpos);
            CollectionAssert.AreEqual(newQvel, env.Data.qvel);
        }

        /// <summary><c>distance_from_origin</c> is the root's planar distance from its reference position.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L387">test_distance_from_origin_info</see>.</remarks>
        [TestMethod]
        [DataRow("Ant-v5")]
        [DataRow("Humanoid-v5")]
        [DataRow("Swimmer-v5")]
        [DataRow("Swimmer-v4")]
        public void TestDistanceFromOriginInfo(string envId) {
            using IEnv made = Make(envId);
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            using var step = env.Step(SampleOf(env));
            double expected = MujocoMath.Norm(env.Data.qpos[0] - env.InitQpos.GetDouble(0), env.Data.qpos[1] - env.InitQpos.GetDouble(1));
            Assert.AreEqual(expected, Num(step.Info["distance_from_origin"]));
        }

        /// <summary><c>z_distance_from_origin</c> is the height coordinate's offset from its reference.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L401">test_z_distance_from_origin_info</see>.</remarks>
        [TestMethod]
        [DataRow("Hopper", "v5")]
        [DataRow("HumanoidStandup", "v5")]
        [DataRow("Walker2d", "v5")]
        public void TestZDistanceFromOriginInfo(string envName, string version) {
            using IEnv made = Make($"{envName}-{version}");
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            using var step = env.Step(SampleOf(env));
            Mujoco.MjKinematics(env.Model, env.Data);
            var structure = (Info)env.GetWrapperAttr("observation_structure")!;
            int zIndex = (int)(long)structure["skipped_qpos"]!;
            Assert.AreEqual(env.Data.qpos[zIndex] - env.InitQpos.GetDouble(zIndex), Num(step.Info["z_distance_from_origin"]));
        }

        /// <summary>The v5 rows of <see cref="TestObservationStructure"/>.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object[]> V5Rows() => AllMujocoEnvs.Select(name => new object[] { name, "v5" });

        /// <summary><c>observation_structure</c>'s counts agree with the model's sizes, where an env has one.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L418">test_observation_structure</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(V5Rows))]
        public void TestObservationStructure(string envName, string version) {
            using IEnv made = Make($"{envName}-{version}");
            var env = (MujocoEnv)made.Unwrapped;
            if (!env.HasWrapperAttr("observation_structure"))
                Assert.Inconclusive("Environment doesn't have an `observation_structure` attribute");

            var obsStruct = (Info)env.GetWrapperAttr("observation_structure")!;
            long Get(string key, long otherwise = 0) => obsStruct.TryGetValue(key, out object? value) ? (long)value! : otherwise;

            Assert.AreEqual(env.Model.Nq, Get("skipped_qpos") + (long)obsStruct["qpos"]!);
            Assert.AreEqual(env.Model.Nv, (long)obsStruct["qvel"]!);
            if (Get("cinert") != 0)
                Assert.AreEqual((env.Model.Nbody - 1) * 10, Get("cinert"));
            if (Get("cvel") != 0)
                Assert.AreEqual((env.Model.Nbody - 1) * 6, Get("cvel"));
            if (Get("qfrc_actuator") != 0)
                Assert.AreEqual(env.Model.Nv - 6, Get("qfrc_actuator"));
            if (Get("cfrc_ext") != 0)
                Assert.AreEqual((env.Model.Nbody - 1) * 6, Get("cfrc_ext"));
            // Gymnasium checks "ten_lenght", a key no env has, so that line never runs there either
            if (Get("ten_lenght") != 0)
                Assert.AreEqual(env.Model.Ntendon, Get("ten_lenght"));
            if (Get("ten_velocity") != 0)
                Assert.AreEqual(env.Model.Ntendon, Get("ten_velocity"));
        }

        /// <summary>The v5 envs with a reset info return a nonempty one.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L444">test_reset_info</see>.</remarks>
        [TestMethod]
        [DataRow("Ant", "v5")]
        [DataRow("HalfCheetah", "v5")]
        [DataRow("Hopper", "v5")]
        [DataRow("Humanoid", "v5")]
        [DataRow("HumanoidStandup", "v5")]
        [DataRow("Swimmer", "v5")]
        [DataRow("Walker2d", "v5")]
        public void TestResetInfo(string envName, string version) {
            using IEnv env = Make($"{envName}-{version}");
            ResetResult<object?> reset = env.Reset();
            try {
                Assert.IsTrue(reset.Info.Count > 0);
            } finally {
                Release(reset);
            }
        }

        /// <summary>InvertedDoublePendulum-v5's tip starts at 1.2 m without reset noise.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L469">test_inverted_double_pendulum_max_height</see>.</remarks>
        [TestMethod]
        [DataRow("v5")]
        public void TestInvertedDoublePendulumMaxHeight(string version) {
            using IEnv made = Make($"InvertedDoublePendulum-{version}", new { reset_noise_scale = 0 });
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            double y = env.Data.site_xpos[2];
            Assert.AreEqual(1.2, y);
        }

        /// <summary>InvertedDoublePendulum-v4's tip is at 1.2 m in the reference state.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L480">test_inverted_double_pendulum_max_height_old</see>.</remarks>
        [TestMethod]
        [DataRow("v4")]
        public void TestInvertedDoublePendulumMaxHeightOld(string version) {
            using IEnv made = Make($"InvertedDoublePendulum-{version}");
            var env = (MujocoEnv)made.Unwrapped;
            env.SetState(env.InitQpos, env.InitQvel);
            double y = env.Data.site_xpos[2];
            Assert.AreEqual(1.2, y);
        }

        /// <summary>Every model loads with the object counts MuJoCo 3.1.4 and later give.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L493">test_model_object_count</see>.</remarks>
        [TestMethod]
        [DataRow("v5")]
        [DataRow("v4")]
        public void TestModelObjectCount(string version) {
            CheckCounts($"Ant-{version}", nq: 15, nv: 14, nu: 8, nbody: 14, nbvh: 14, njnt: 9, ngeom: 14, ntendon: 0);
            CheckCounts($"HalfCheetah-{version}", nq: 9, nv: 9, nu: 6, nbody: 8, nbvh: 10, njnt: 9, ngeom: 9, ntendon: 0);
            CheckCounts($"Hopper-{version}", nq: 6, nv: 6, nu: 3, nbody: 5, nbvh: 5, njnt: 6, ngeom: 5, ntendon: 0);
            CheckCounts($"Humanoid-{version}", nq: 24, nv: 23, nu: 17, nbody: 14, nbvh: 22, njnt: 18, ngeom: 18, ntendon: 2);
            CheckCounts($"HumanoidStandup-{version}", nq: 24, nv: 23, nu: 17, nbody: 14, nbvh: 22, njnt: 18, ngeom: 18, ntendon: 2);
            CheckCounts($"InvertedDoublePendulum-{version}", nq: 3, nv: 3, nu: 1, nbody: 4, nbvh: 6, njnt: 3, ngeom: 5, ntendon: 0);
            CheckCounts($"InvertedPendulum-{version}", nq: 2, nv: 2, nu: 1, nbody: 3, nbvh: 3, njnt: 2, ngeom: 3, ntendon: 0);
            // Pusher-v4 is skipped on MuJoCo 3 and later, as there; v5's counts are MuJoCo 3.1.4's and later
            if (version != "v4")
                CheckCounts($"Pusher-{version}", nq: 11, nv: 11, nu: 7, nbody: 13, nbvh: 7, njnt: 11, ngeom: 20, ntendon: 0);
            CheckCounts($"Reacher-{version}", nq: 4, nv: 4, nu: 2, nbody: 5, nbvh: 3, njnt: 4, ngeom: 10, ntendon: 0);
            CheckCounts($"Swimmer-{version}", nq: 5, nv: 5, nu: 2, nbody: 4, nbvh: 0, njnt: 5, ngeom: 4, ntendon: 0);
            CheckCounts($"Walker2d-{version}", nq: 9, nv: 9, nu: 6, nbody: 8, nbvh: 8, njnt: 9, ngeom: 8, ntendon: 0);
        }

        /// <summary>The arrays the observations read have the shapes MuJoCo gives them.</summary>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L604">test_model_sensors</see>.</remarks>
        [TestMethod]
        [DataRow("v5")]
        [DataRow("v4")]
        public void TestModelSensors(string version) {
            using (IEnv made = Make($"Ant-{version}"))
                AssertShape(((MujocoEnv)made.Unwrapped).Data.CfrcExt, 14, 6);

            foreach (string name in new[] { "Humanoid", "HumanoidStandup" }) {
                using IEnv made = Make($"{name}-{version}");
                MjData data = ((MujocoEnv)made.Unwrapped).Data;
                AssertShape(data.Cinert, 14, 10);
                AssertShape(data.Cvel, 14, 6);
                AssertShape(data.QfrcActuator, 23);
                AssertShape(data.CfrcExt, 14, 6);
            }
        }

        /// <summary><c>dt</c> is the time step times the frame skip, after a time step assigned to the model.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L627">test_dt</see>.</remarks>
        [TestMethod]
        public void TestDt() {
            using IEnv madeA = Make("Ant-v5", new { include_cfrc_ext_in_observation = false });
            using IEnv madeB = Make("Ant-v5", new { include_cfrc_ext_in_observation = false, frame_skip = 1 });
            var envA = (MujocoEnv)madeA.Unwrapped;
            var envB = (MujocoEnv)madeB.Unwrapped;
            envB.Model.Opt.Timestep = 0.05;

            Assert.AreEqual(envA.Dt, envB.Dt);
            // check_environments_match(env_a, env_b, num_steps=100)   # This Fails as expected
        }

        /// <summary>The ids of the reset-noise test.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object[]> ResetNoiseIds() => new[] {
            "Ant-v5", "Ant-v4", "HalfCheetah-v5", "HalfCheetah-v4", "Hopper-v5", "Hopper-v4", "Humanoid-v5", "Humanoid-v4",
            "HumanoidStandup-v5", "InvertedDoublePendulum-v5", "InvertedPendulum-v5", "Swimmer-v5", "Swimmer-v4", "Walker2d-v5",
            "Walker2d-v4",
        }.Select(id => new object[] { id });

        /// <summary>With <c>reset_noise_scale=0</c> the reset is the reference state exactly.</summary>
        /// <param name="envId">The id.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L641">test_reset_noise_scale</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ResetNoiseIds))]
        public void TestResetNoiseScale(string envId) {
            using IEnv made = Make(envId, new { reset_noise_scale = 0 });
            var env = (MujocoEnv)made.Unwrapped;
            Release(env.Reset());
            CollectionAssert.AreEqual(env.InitQpos.ToArray<double>(), env.Data.qpos);
            CollectionAssert.AreEqual(env.InitQvel.ToArray<double>(), env.Data.qvel);
        }

        /// <summary>The rows of <see cref="TestResetState"/>: every env, v5 and v4.</summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object[]> ResetStateRows() =>
            from name in AllMujocoEnvs from version in new[] { "v5", "v4" } select new object[] { name, version };

        /// <summary>A seeded reset restores the same state whatever ran before it.</summary>
        /// <param name="envName">The env's name.</param>
        /// <param name="version">The version.</param>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/mujoco/test_mujoco_v5.py#L668">test_reset_state</see>.</remarks>
        [TestMethod]
        [DynamicData(nameof(ResetStateRows))]
        public void TestResetState(string envName, string version) {
            if (envName == "Pusher" && version == "v4")
                Assert.Inconclusive("Skipping Pusher-v4 as not compatible with mujoco >= 3.0");

            using IEnv env = Make($"{envName}-{version}");
            MujocoUtils.CheckMujocoResetState(env);
        }

        /// <summary>Steps an env with sampled actions and checks each reward against <paramref name="sum"/> of its info.</summary>
        /// <param name="envId">The id.</param>
        /// <param name="numSteps">The steps.</param>
        /// <param name="sum">The info terms' sum in Gymnasium's order.</param>
        private static void CheckRewardSum(string envId, int numSteps, Func<Info, double> sum) {
            using IEnv env = Make(envId);
            Release(env.Reset());
            for (int i = 0; i < numSteps; i++) {
                using var step = env.Step(env.ActionSpace.Sample());
                Assert.AreEqual(sum(step.Info), step.Reward, $"{envId} step {i}");
            }
        }

        /// <summary>Asserts one model's object counts.</summary>
        /// <param name="envId">The id.</param>
        /// <param name="nq">Positions.</param>
        /// <param name="nv">Velocities.</param>
        /// <param name="nu">Actuators.</param>
        /// <param name="nbody">Bodies.</param>
        /// <param name="nbvh">Bounding volumes.</param>
        /// <param name="njnt">Joints.</param>
        /// <param name="ngeom">Geoms.</param>
        /// <param name="ntendon">Tendons.</param>
        private static void CheckCounts(string envId, int nq, int nv, int nu, int nbody, int nbvh, int njnt, int ngeom, int ntendon) {
            using IEnv made = Make(envId);
            MjModel model = ((MujocoEnv)made.Unwrapped).Model;
            Assert.AreEqual(nq, model.Nq, $"{envId} nq");
            Assert.AreEqual(nv, model.Nv, $"{envId} nv");
            Assert.AreEqual(nu, model.Nu, $"{envId} nu");
            Assert.AreEqual(nbody, model.Nbody, $"{envId} nbody");
            Assert.AreEqual(nbvh, model.Nbvh, $"{envId} nbvh");
            Assert.AreEqual(njnt, model.Njnt, $"{envId} njnt");
            Assert.AreEqual(ngeom, model.Ngeom, $"{envId} ngeom");
            Assert.AreEqual(ntendon, model.Ntendon, $"{envId} ntendon");
        }

        /// <summary>Asserts a view's shape, then releases it.</summary>
        /// <param name="view">The view, which this releases.</param>
        /// <param name="shape">The expected shape.</param>
        private static void AssertShape(NDArray view, params long[] shape) {
            using (view)
                CollectionAssert.AreEqual(shape, view.shape);
        }

        /// <summary>The torso frame's x and y: <c>env.data.body("torso").xpos[:2]</c>.</summary>
        /// <param name="env">The env.</param>
        /// <returns>The coordinates.</returns>
        private static (double X, double Y) TorsoXy(MujocoEnv env) {
            int torso = env.Data.Body("torso").Id;
            return (env.Data.xpos[3 * torso], env.Data.xpos[3 * torso + 1]);
        }

        /// <summary>A sample of an env's action space, which the caller owns.</summary>
        /// <param name="env">The env.</param>
        /// <returns>The action.</returns>
        private static NDArray SampleOf(MujocoEnv env) => env.ActionSpace.Sample();

        /// <summary>An info value as a double: a 0-d array's element, a double, a long.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The number.</returns>
        /// <exception cref="AssertFailedException">The value is not a number, which fails the test.</exception>
        private static double Num(object? value) => value switch {
            NDArray array => array.GetDouble(),
            double d => d,
            long l => l,
            _ => throw new AssertFailedException($"not a number: {value}"),
        };

        /// <summary>An info entry as a double (<see cref="Num"/>).</summary>
        /// <param name="info">The info.</param>
        /// <param name="key">The key.</param>
        /// <returns>The number.</returns>
        /// <exception cref="AssertFailedException">The entry is not a number, which fails the test.</exception>
        private static double V(Info info, string key) => Num(info[key]);

        /// <summary>Releases the arrays a reset returned.</summary>
        /// <param name="result">The reset's result.</param>
        private static void Release(ResetResult<object?> result) => result.Dispose();

        /// <summary>Releases the arrays a typed reset returned.</summary>
        /// <param name="result">The reset's result.</param>
        private static void Release(ResetResult<NDArray> result) => result.Dispose();
    }
}
