#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.MuJoCo;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.MuJoCo {
    /// <summary>
    ///     The MuJoCo envs as Gym.NET registers and builds them (DR-61): the 22 ids with Gymnasium's limits and thresholds,
    ///     their spaces and frame rates, Pusher-v4's and the custom models' refusals, Ant-v5's <c>main_body</c> forms, the
    ///     NumPy types of the info values (DR-08), the control cost's dtype paths and HumanoidStandup-v4's integer cap.
    /// </summary>
    /// <remarks>
    ///     Bit-exactness against Gymnasium is the parity suite's (and the trace checker's) to prove; these tests pin the
    ///     shapes, types and refusals a user meets.
    /// </remarks>
    [TestClass]
    public class MujocoEnvTests {
        /// <summary>
        ///     Every id: its class, Gymnasium's step limit and reward threshold, the observation length, the action bounds, the
        ///     frame rate and the reset info's keys.
        /// </summary>
        /// <returns>The data rows.</returns>
        public static IEnumerable<object?[]> Ids() => new object?[][] {
            new object?[] { "Reacher-v4", typeof(ReacherEnvV4), 50L, -3.75, 11, 2, 1.0, 50L, "" },
            new object?[] { "Reacher-v5", typeof(ReacherEnvV5), 50L, -3.75, 10, 2, 1.0, 50L, "" },
            new object?[] { "Pusher-v5", typeof(PusherEnvV5), 100L, 0.0, 23, 7, 2.0, 20L, "" },
            new object?[] { "InvertedPendulum-v4", typeof(InvertedPendulumEnvV4), 1000L, 950.0, 4, 1, 3.0, 25L, "" },
            new object?[] { "InvertedPendulum-v5", typeof(InvertedPendulumEnvV5), 1000L, 950.0, 4, 1, 3.0, 25L, "" },
            new object?[] { "InvertedDoublePendulum-v4", typeof(InvertedDoublePendulumEnvV4), 1000L, 9100.0, 11, 1, 1.0, 20L, "" },
            new object?[] { "InvertedDoublePendulum-v5", typeof(InvertedDoublePendulumEnvV5), 1000L, 9100.0, 9, 1, 1.0, 20L, "" },
            new object?[] { "HalfCheetah-v4", typeof(HalfCheetahEnvV4), 1000L, 4800.0, 17, 6, 1.0, 20L, "" },
            new object?[] { "HalfCheetah-v5", typeof(HalfCheetahEnvV5), 1000L, 4800.0, 17, 6, 1.0, 20L, "x_position" },
            new object?[] { "Hopper-v4", typeof(HopperEnvV4), 1000L, 3800.0, 11, 3, 1.0, 125L, "" },
            new object?[] { "Hopper-v5", typeof(HopperEnvV5), 1000L, 3800.0, 11, 3, 1.0, 125L, "x_position z_distance_from_origin" },
            new object?[] { "Swimmer-v4", typeof(SwimmerEnvV4), 1000L, 360.0, 8, 2, 1.0, 25L, "" },
            new object?[] { "Swimmer-v5", typeof(SwimmerEnvV5), 1000L, 360.0, 8, 2, 1.0, 25L, "x_position y_position distance_from_origin" },
            new object?[] { "Walker2d-v4", typeof(Walker2dEnvV4), 1000L, null, 17, 6, 1.0, 125L, "" },
            new object?[] { "Walker2d-v5", typeof(Walker2dEnvV5), 1000L, null, 17, 6, 1.0, 125L, "x_position z_distance_from_origin" },
            new object?[] { "Ant-v4", typeof(AntEnvV4), 1000L, 6000.0, 27, 8, 1.0, 20L, "" },
            new object?[] { "Ant-v5", typeof(AntEnvV5), 1000L, 6000.0, 105, 8, 1.0, 20L, "x_position y_position distance_from_origin" },
            new object?[] { "Humanoid-v4", typeof(HumanoidEnvV4), 1000L, null, 376, 17, 0.4, 67L, "" },
            new object?[] {
                "Humanoid-v5", typeof(HumanoidEnvV5), 1000L, null, 348, 17, 0.4, 67L,
                "x_position y_position tendon_length tendon_velocity distance_from_origin",
            },
            new object?[] { "HumanoidStandup-v4", typeof(HumanoidStandupEnvV4), 1000L, null, 376, 17, 0.4, 67L, "" },
            new object?[] {
                "HumanoidStandup-v5", typeof(HumanoidStandupEnvV5), 1000L, null, 348, 17, 0.4, 67L,
                "x_position y_position z_distance_from_origin tendon_length tendon_velocity",
            },
        };

        /// <summary>
        ///     Each id registers with Gymnasium's limit and threshold and no kwargs, builds its class under <c>make</c>'s
        ///     stack, and has the observation length, the symmetric float32 action bounds, the frame rate and the reset info
        ///     Gymnasium's has.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="type">The env class.</param>
        /// <param name="steps">The step limit.</param>
        /// <param name="threshold">The reward threshold, or <see langword="null"/>.</param>
        /// <param name="obsSize">The observation length.</param>
        /// <param name="actions">The number of actuators.</param>
        /// <param name="bound">The control range's upper bound (the lower is its negation).</param>
        /// <param name="fps">The frame rate.</param>
        /// <param name="resetKeys">The reset info's keys, space-separated, in order.</param>
        [TestMethod]
        [DynamicData(nameof(Ids))]
        public void Id_RegistersAndBuildsAsGymnasiums(string id, Type type, long steps, double? threshold, int obsSize, int actions, double bound,
                                                      long fps, string resetKeys) {
            GymEnvironments.Register();
            EnvSpec spec = Spec(id);
            Assert.AreEqual(type, spec.EntryPoint, $"{id} entry point");
            Assert.AreEqual(steps, spec.MaxEpisodeSteps, $"{id} max_episode_steps");
            Assert.AreEqual(threshold, spec.RewardThreshold, $"{id} reward_threshold");
            Assert.AreEqual(0, spec.Kwargs.Count, $"{id} kwargs");

            using IEnv env = Make(id);
            Assert.IsInstanceOfType(env.Unwrapped, type);
            var box = (Box)env.ObservationSpace;
            CollectionAssert.AreEqual(new long[] { obsSize }, box.Shape.dimensions, $"{id} observation shape");
            Assert.AreEqual(NPTypeCode.Double, box.DType.typecode, $"{id} observation dtype");
            var action = (Box)env.ActionSpace;
            Assert.AreEqual(NPTypeCode.Single, action.DType.typecode, $"{id} action dtype");
            using (NDArray high = action.High.copy(), low = action.Low.copy()) {
                Assert.AreEqual(actions, (int)high.size, $"{id} actuators");
                for (int i = 0; i < actions; i++) {
                    Assert.AreEqual((float)bound, high.GetSingle(i), $"{id} high[{i}]");
                    Assert.AreEqual(-(float)bound, low.GetSingle(i), $"{id} low[{i}]");
                }
            }
            Assert.AreEqual(fps, Convert.ToInt64(env.Unwrapped.Metadata[Metadata.RenderFpsKey]), $"{id} render_fps");

            using ResetResult<object?> reset = env.Reset(0);
            Assert.AreEqual(resetKeys, string.Join(" ", reset.Info.Keys), $"{id} reset info keys");
            Assert.AreEqual(obsSize, (int)((NDArray)reset.Observation!).size, $"{id} first observation");
        }

        /// <summary>
        ///     Pusher-v4 is registered like Gymnasium's and refuses at construction with Gymnasium's <c>ImportError</c>, through
        ///     <c>make</c> and directly.
        /// </summary>
        [TestMethod]
        public void PusherV4_RefusesAsGymnasiumDoesOnMuJoCo3() {
            GymEnvironments.Register();
            EnvSpec spec = Spec("Pusher-v4");
            Assert.AreEqual(typeof(PusherEnvV4), spec.EntryPoint);
            Assert.AreEqual(100L, spec.MaxEpisodeSteps);
            Assert.AreEqual(0.0, spec.RewardThreshold);
            const string message = "`Pusher-v4` is only supported on `mujoco<3`, for more information https://github.com/Farama-Foundation/Gymnasium/issues/950";
            using (Warnings.Catch())
                Assert.AreEqual(message, Assert.ThrowsExactly<ImportError>(() => Make("Pusher-v4")).Message);
            Assert.AreEqual(message, Assert.ThrowsExactly<ImportError>(() => new PusherEnvV4()).Message);
        }

        /// <summary>A model path that is not one of Gymnasium's 14 bundled names is refused with the list (DR-63).</summary>
        [TestMethod]
        public void XmlFile_OutsideTheBundledModels_IsRefused() {
            var e = Assert.ThrowsExactly<NotSupportedException>(() => Make("Ant-v5", new { xml_file = "./tests/envs/mujoco/assets/walker2d_v5_uneven_feet.xml" }));
            StringAssert.Contains(e.Message, "'./tests/envs/mujoco/assets/walker2d_v5_uneven_feet.xml' is not one of ant.xml");
            StringAssert.Contains(e.Message, "walker2d_v5.xml");
            // a bundled model under another env is taken: Walker2d-v5 on v4's model
            using IEnv walker = Make("Walker2d-v5", new { xml_file = "walker2d.xml" });
            Assert.AreEqual("walker2d.xml", ((MujocoEnv)walker.Unwrapped).Fullpath);
        }

        /// <summary>
        ///     Ant-v5's <c>main_body</c> takes the forms MuJoCo's <c>data.body</c> takes (an index, Python's <c>bool</c>, a
        ///     NumPy integer, a name), all giving the default's velocities; bad ones fail at the first step with MuJoCo's errors.
        /// </summary>
        [TestMethod]
        public void AntV5_MainBody_TakesMujocosForms() {
            double[] Velocities(object? mainBody) {
                using IEnv env = Make("Ant-v5", mainBody is null ? null : new Info { ["main_body"] = mainBody });
                ((Env<NDArray, NDArray>)env.Unwrapped).ActionSpace.Seed(3);
                env.Reset(5).Dispose();
                using var step = env.Step(env.ActionSpace.Sample());
                return new[] { ((NDArray)step.Info["x_velocity"]!).GetDouble(), ((NDArray)step.Info["y_velocity"]!).GetDouble() };
            }

            double[] expected = Velocities(null);
            CollectionAssert.AreEqual(expected, Velocities(1L));
            CollectionAssert.AreEqual(expected, Velocities(1));
            CollectionAssert.AreEqual(expected, Velocities(true));
            CollectionAssert.AreEqual(expected, Velocities("torso"));
            using (NDArray npInt = NDArray.Scalar(1L))
                CollectionAssert.AreEqual(expected, Velocities(npInt));
            // body 2 (front_left_leg) sits at the torso's origin and moves with it; body 3 (aux_1) is offset and hinged
            CollectionAssert.AreEqual(expected, Velocities(2L), "front_left_leg's frame is the torso's");
            CollectionAssert.AreNotEqual(expected, Velocities(3L), "aux_1 moves otherwise");

            StringAssert.StartsWith(Assert.ThrowsExactly<KeyError>(() => Velocities("nope")).Message, "\"Invalid name 'nope'. Valid names: ['aux_1'");
            Assert.AreEqual("Invalid index 14. Valid indices from 0 to 13", Assert.ThrowsExactly<IndexError>(() => Velocities(14L)).Message);
            StringAssert.StartsWith(Assert.ThrowsExactly<TypeError>(() => Velocities(1.5)).Message,
                "body(): incompatible function arguments. The following argument types are supported:");
        }

        /// <summary>
        ///     The info values carry NumPy's types (DR-08): an <c>np.float64</c> is a 0-d float64 array, the float32 control cost
        ///     a 0-d float32 one, a Python <c>float</c> a <see langword="double"/>, a Python <c>int</c> a <see langword="long"/>,
        ///     and the tendon arrays float64 arrays.
        /// </summary>
        [TestMethod]
        public void InfoValues_CarryNumpysTypes() {
            Info StepInfo(string id) {
                using IEnv env = Make(id);
                env.Reset(0).Dispose();
                StepResult<object?> step = env.Step(env.ActionSpace.Sample());
                (step.Observation as IDisposable)?.Dispose();
                return step.Info;
            }

            Info cheetah = StepInfo("HalfCheetah-v5");
            Assert.AreEqual(NPTypeCode.Double, ((NDArray)cheetah["x_position"]!).typecode);
            Assert.AreEqual(0, ((NDArray)cheetah["x_position"]!).ndim);
            Assert.AreEqual(NPTypeCode.Single, ((NDArray)cheetah["reward_ctrl"]!).typecode);
            Assert.IsInstanceOfType<double>(StepInfo("Hopper-v5")["reward_survive"], "Hopper-v5: Python bool * float");
            Assert.IsInstanceOfType<NDArray>(StepInfo("Walker2d-v5")["reward_survive"], "Walker2d-v5: np.bool_ * float");
            Assert.IsInstanceOfType<long>(StepInfo("InvertedPendulum-v5")["reward_survive"], "InvertedPendulum-v5: int(not terminated)");
            Assert.IsInstanceOfType<double>(StepInfo("InvertedDoublePendulum-v5")["reward_survive"], "a Python float");
            Info humanoid = StepInfo("Humanoid-v5");
            var tendons = (NDArray)humanoid["tendon_length"]!;
            CollectionAssert.AreEqual(new long[] { 2 }, tendons.shape);
            Assert.AreEqual(NPTypeCode.Double, tendons.typecode);
            Assert.AreEqual(NPTypeCode.Double, ((NDArray)humanoid["reward_ctrl"]!).typecode, "Humanoid's cost reads data.ctrl, float64");
        }

        /// <summary>
        ///     The control cost follows the action's dtype as NumPy's does: float32 and float64 actions price in their own
        ///     precision, integers in their own width (int8's squares wrap), booleans count; Reacher-v5's default int weight
        ///     keeps an integer cost an <c>np.int64</c>, a given float weight makes it an <c>np.float64</c>.
        /// </summary>
        [TestMethod]
        public void ControlCost_FollowsTheActionsDtype() {
            using var cheetah = new HalfCheetahEnvV5();
            using (NDArray f32 = np.array(new[] { 0.5f, 0.25f, 0, 0, 0, 0 }))
                Assert.AreEqual(0.1f * (0.25f + 0.0625f), (float)cheetah.ControlCost(f32));
            using (NDArray f64 = np.array(new[] { 0.5, 0.25, 0, 0, 0, 0 }))
                Assert.AreEqual(0.1 * (0.25 + 0.0625), cheetah.ControlCost(f64));
            // int8 100**2 wraps to 16 in int8, as np.square keeps the dtype
            using (NDArray i8 = np.array(new sbyte[] { 100, 1, 0, 0, 0, 0 }))
                Assert.AreEqual(0.1 * 17, cheetah.ControlCost(i8));
            using (NDArray flags = np.array(new[] { true, true, false, false, false, true }))
                Assert.AreEqual(0.1 * 3, cheetah.ControlCost(flags));
            using (NDArray half = np.array(new[] { (Half)1, (Half)0, (Half)0, (Half)0, (Half)0, (Half)0 }))
                Assert.ThrowsExactly<NotSupportedException>(() => cheetah.ControlCost(half));

            object? ReacherCtrl(object? weight) {
                using IEnv env = Make("Reacher-v5", weight is null ? null : new { reward_control_weight = weight });
                env.Reset(0).Dispose();
                using NDArray ints = np.array(new long[] { 1, -2 });
                StepResult<object?> step = env.Step(ints);
                (step.Observation as IDisposable)?.Dispose();
                return step.Info["reward_ctrl"];
            }

            var intCost = (NDArray)ReacherCtrl(null)!;
            Assert.AreEqual(NPTypeCode.Int64, intCost.typecode, "the default weight is Python's int 1");
            Assert.AreEqual(-5L, intCost.GetInt64());
            var floatCost = (NDArray)ReacherCtrl(1.0)!;
            Assert.AreEqual(NPTypeCode.Double, floatCost.typecode, "a given weight is a Python float");
            Assert.AreEqual(-5.0, floatCost.GetDouble());
        }

        /// <summary>
        ///     A float16 action is refused before the physics runs, so the simulation does not advance (NumPy would keep its
        ///     cost in float16, which Gym.NET does not model).
        /// </summary>
        [TestMethod]
        public void Float16Action_IsRefusedBeforeThePhysics() {
            using var env = new HopperEnvV5();
            env.Reset(0).Dispose();
            double time = env.Data.time;
            using NDArray half = np.array(new[] { (Half)0.5, (Half)0, (Half)0 });
            Assert.ThrowsExactly<NotSupportedException>(() => env.Step(half));
            Assert.AreEqual(time, env.Data.time);
        }

        /// <summary>
        ///     HumanoidStandup-v4 caps the impact cost with Python's <c>min(cost, 10)</c>: past the cap the info holds the int
        ///     -10 and the reward subtracts 10.
        /// </summary>
        [TestMethod]
        public void HumanoidStandupV4_CapsTheImpactWithPythonsMin() {
            using var env = new HumanoidStandupEnvV4();
            env.Reset(0).Dispose();
            // a violent start: every velocity at 50, so the contacts push hard enough for the cost to pass 10
            using NDArray qpos = env.InitQpos.copy();
            using NDArray qvel = np.full(new Shape(env.Model.Nv), 50.0);
            env.SetState(qpos, qvel);
            bool capped = false;
            for (int i = 0; i < 20 && !capped; i++) {
                using NDArray zeros = np.zeros(new Shape(env.Model.Nu), NPTypeCode.Single);
                using var step = env.Step(zeros);
                if (step.Info["reward_impact"] is long impact) {
                    Assert.AreEqual(-10L, impact);
                    double linup = ((NDArray)step.Info["reward_linup"]!).GetDouble();
                    double quadCtrl = ((NDArray)step.Info["reward_quadctrl"]!).GetDouble();
                    Assert.AreEqual(linup - -quadCtrl - 10 + 1, step.Reward);
                    capped = true;
                } else {
                    Assert.IsTrue(((NDArray)step.Info["reward_impact"]!).GetDouble() >= -10);
                }
            }
            Assert.IsTrue(capped, "the impact cost should have passed the cap");
        }

        /// <summary>
        ///     Pusher-v5's reset writes the cylinder's start and the goal into <c>init_qpos</c> itself, as Gymnasium's
        ///     <c>qpos = self.init_qpos</c> does, and exposes them as <c>cylinder_pos</c> and the int64 <c>goal_pos</c>.
        /// </summary>
        [TestMethod]
        public void PusherV5_ResetWritesIntoInitQpos() {
            using var env = new PusherEnvV5();
            Assert.IsNull(env.CylinderPos);
            env.Reset(4).Dispose();
            NDArray cylinder = env.CylinderPos!;
            Assert.AreEqual(NPTypeCode.Int64, env.GoalPos!.typecode);
            int nq = env.Model.Nq;
            Assert.AreEqual(cylinder.GetDouble(0), env.InitQpos.GetDouble(nq - 4));
            Assert.AreEqual(cylinder.GetDouble(1), env.InitQpos.GetDouble(nq - 3));
            Assert.AreEqual(0.0, env.InitQpos.GetDouble(nq - 2));
            Assert.IsTrue(MujocoMath.Norm(cylinder.GetDouble(0), cylinder.GetDouble(1)) > 0.17, "drawn until away from the goal");
            Assert.AreEqual(cylinder.GetDouble(0), env.Data.qpos[nq - 4]);
        }

        /// <summary>
        ///     The classes carry Gymnasium's Python names (DR-66), v5's class metadata has no frame rate (each instance derives
        ///     one from its time step) and v4's has Gymnasium's fixed one; the generated options records build the envs.
        /// </summary>
        [TestMethod]
        public void Classes_CarryGymnasiumsNamesAndMetadata() {
            Assert.AreEqual("AntEnv", PyAttributes.PyTypeName(typeof(AntEnvV5)));
            Assert.AreEqual("AntEnv", PyAttributes.PyTypeName(typeof(AntEnvV4)));
            Assert.AreEqual("MujocoEnv", PyAttributes.PyTypeName(typeof(MujocoEnv)));
            Assert.IsFalse(AntEnvV5.ClassMetadata.ContainsKey(Metadata.RenderFpsKey));
            Assert.AreEqual(20L, AntEnvV4.ClassMetadata[Metadata.RenderFpsKey]);
            CollectionAssert.AreEqual(new[] { "human", "rgb_array", "depth_array", "rgbd_tuple" },
                ((IEnumerable<string>)AntEnvV5.ClassMetadata[Metadata.RenderModesKey]!).ToArray());

            using var hopper = new HopperEnvV5(new HopperEnvV5Options { FrameSkip = 2, HealthyZRange = (0.9, 1.5) });
            Assert.AreEqual(2, hopper.FrameSkip);
            Assert.AreEqual((0.9, 1.5), hopper.HealthyZRange);
            Assert.AreEqual(250L, hopper.Metadata[Metadata.RenderFpsKey], "1 / (0.002 * 2)");
            using IEnv ant = Make("Ant-v5", new AntEnvV5Options { MainBody = "torso", IncludeCfrcExtInObservation = false });
            Assert.AreEqual("torso", ((AntEnvV5)ant.Unwrapped).MainBody);
            Assert.AreEqual(27L, ((Box)ant.ObservationSpace).Shape.size);
        }

        /// <summary>
        ///     A zero frame skip fails the v5 frame rate as Python's <c>int(np.round(1 / 0.0))</c> fails, with
        ///     <c>OverflowError</c>'s message.
        /// </summary>
        [TestMethod]
        public void ZeroFrameSkip_FailsTheFrameRateAsPython() {
            Assert.AreEqual("cannot convert float infinity to integer",
                Assert.ThrowsExactly<OverflowException>(() => new SwimmerEnvV5(frameSkip: 0)).Message);
        }
    }
}
