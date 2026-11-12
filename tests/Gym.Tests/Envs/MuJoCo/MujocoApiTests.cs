#nullable enable
using System;
using System.Linq;
using Gym.Environments.Envs.Classic;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.Exceptions;
using Gym.MuJoCo;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.MuJoCo {
    /// <summary>
    ///     <c>Gym.MuJoCo</c>, the reach into MuJoCo that Gymnasium's envs and tests use, under .NET names (DR-68): the bundled
    ///     models (DR-63), the model's and data's writable views, the body lookups with MuJoCo's errors, the <c>mj_*</c>
    ///     functions, the state vectors and their signatures, and <c>gymnasium.envs.mujoco.utils</c>.
    /// </summary>
    [TestClass]
    public class MujocoApiTests {
        /// <summary>
        ///     Each of Gymnasium's 14 bundled models loads by its file name, as a model of its own; anything else is refused
        ///     with the list of names.
        /// </summary>
        [TestMethod]
        public void FromXmlPath_LoadsTheBundledModels_AndRefusesTheRest() {
            Assert.AreEqual(14, MjModel.BundledNames.Length);
            foreach (string name in MjModel.BundledNames) {
                using MjModel model = MjModel.FromXmlPath(name);
                Assert.IsTrue(model.Nq > 0 && model.Nbody > 1, name);
            }

            using (MjModel ant = MjModel.FromXmlPath("ant.xml"))
            using (MjModel other = MjModel.FromXmlPath("ant.xml")) {
                Assert.AreEqual(15, ant.Nq);
                // two loads are two models: a colour written into one is not in the other
                using (NDArray rgba = ant.GeomRgba)
                    rgba.SetSingle(0.25f, 0, 0);
                using NDArray untouched = other.GeomRgba;
                Assert.AreNotEqual(0.25f, untouched.GetSingle(0, 0));
            }

            var e = Assert.ThrowsExactly<NotSupportedException>(() => MjModel.FromXmlPath("./ant.xml"));
            StringAssert.Contains(e.Message, "'./ant.xml'");
            StringAssert.Contains(e.Message, string.Join(", ", MjModel.BundledNames));
        }

        /// <summary>
        ///     The model's options, statistics and visual globals read MuJoCo's values; the time step and the offscreen size are
        ///     writable, as in Python.
        /// </summary>
        [TestMethod]
        public void ModelScalars_ReadAndWrite() {
            using MjModel model = MjModel.FromXmlPath("half_cheetah.xml");
            Assert.AreEqual(0.01, model.Opt.Timestep);
            model.Opt.Timestep = 0.02;
            Assert.AreEqual(0.02, model.Opt.Timestep);
            Assert.IsTrue(model.Stat.Extent > 0);
            MjVisualGlobal global = model.Vis.Global;
            global.Offwidth = 1234;
            Assert.AreEqual(1234, model.Vis.Global.Offwidth);
            using NDArray mass = model.BodyMass, range = model.ActuatorCtrlrange;
            CollectionAssert.AreEqual(new long[] { model.Nbody }, mass.shape);
            CollectionAssert.AreEqual(new long[] { model.Nu, 2 }, range.shape);
        }

        /// <summary>
        ///     The data's arrays are writable views of the simulation's own arrays, shaped as MuJoCo shapes them: a write through
        ///     a view is in the simulation.
        /// </summary>
        [TestMethod]
        public void DataViews_AreWritableAndShared() {
            using MjModel model = MjModel.FromXmlPath("humanoid.xml");
            using var data = new MjData(model);
            using (NDArray qpos = data.Qpos)
                qpos.SetDouble(1.5, 0);
            Assert.AreEqual(1.5, data.qpos[0]);
            using (NDArray cinert = data.Cinert)
                CollectionAssert.AreEqual(new long[] { 14, 10 }, cinert.shape);
            using (NDArray tendons = data.TenLength)
                CollectionAssert.AreEqual(new long[] { 2 }, tendons.shape);
            using (NDArray act = data.Act)
                Assert.AreEqual(0, (int)act.size);
            data.Time = 3.0;
            Assert.AreEqual(3.0, data.time);
        }

        /// <summary>
        ///     <c>data.body(name)</c> and <c>data.body(id)</c> find a body and its frame; an unknown name and an index out of range
        ///     fail with MuJoCo's messages.
        /// </summary>
        [TestMethod]
        public void Body_FindsBodies_WithMujocosErrors() {
            using MjModel model = MjModel.FromXmlPath("ant.xml");
            using var data = new MjData(model);
            Mujoco.MjForward(model, data);
            MjDataBody torso = data.Body("torso");
            Assert.AreEqual(1, torso.Id);
            Assert.AreEqual("torso", torso.Name);
            using (NDArray xpos = torso.Xpos)
                Assert.AreEqual(data.xpos[3], xpos.GetDouble(0));
            Assert.AreEqual(1, data.Body(1).Id);

            string keyError = Assert.ThrowsExactly<KeyError>(() => data.Body("nope")).Message;
            // MuJoCo 4.0.0's text: the named bodies only (Ant's four feet are unnamed), sorted
            Assert.AreEqual("\"Invalid name 'nope'. Valid names: ['aux_1', 'aux_2', 'aux_3', 'aux_4', 'back_leg', 'front_left_leg', "
                            + "'front_right_leg', 'right_back_leg', 'torso', 'world']\"",
                keyError);
            Assert.AreEqual("Invalid index 14. Valid indices from 0 to 13", Assert.ThrowsExactly<IndexError>(() => data.Body(14)).Message);
            Assert.AreEqual(1, Mujoco.MjName2Id(model, MjtObj.Body, "torso"));
            Assert.AreEqual(-1, Mujoco.MjName2Id(model, MjtObj.Body, "nope"));
        }

        /// <summary>
        ///     <c>mj_step</c> advances <c>nstep</c> time steps (none for 0 or fewer, as MuJoCo's binding), <c>mj_resetData</c>
        ///     returns to the reference state at time 0.
        /// </summary>
        [TestMethod]
        public void StepAndReset_MoveTheClock() {
            using MjModel model = MjModel.FromXmlPath("hopper.xml");
            using var data = new MjData(model);
            Mujoco.MjStep(model, data, 0);
            Mujoco.MjStep(model, data, -1);
            Assert.AreEqual(0.0, data.time);
            Mujoco.MjStep(model, data, 3);
            Assert.AreEqual(3 * model.Opt.Timestep, data.time, 1e-15);
            Assert.AreNotEqual(model.qpos0[1], data.qpos[1]);
            Mujoco.MjResetData(model, data);
            Assert.AreEqual(0.0, data.time);
            CollectionAssert.AreEqual(model.qpos0, data.qpos);
        }

        /// <summary>
        ///     The state vectors (<c>mj_stateSize</c>, <c>mj_getState</c>, <c>mj_setState</c>) follow MuJoCo 4.0's signature
        ///     bits; a vector of the wrong size or dtype is refused with MuJoCo's message.
        /// </summary>
        [TestMethod]
        public void StateVectors_RoundTrip() {
            Assert.AreEqual(1, (int)MjtState.Time);
            Assert.AreEqual(16414, (int)MjtState.Physics);
            Assert.AreEqual(24607, (int)MjtState.FullPhysics);
            Assert.AreEqual(32767, (int)MjtState.Integration);

            using MjModel model = MjModel.FromXmlPath("walker2d_v5.xml");
            using var data = new MjData(model);
            Assert.AreEqual(1 + model.Nq + model.Nv, Mujoco.MjStateSize(model, MjtState.FullPhysics));
            Mujoco.MjStep(model, data, 5);
            using NDArray state = np.zeros(new Shape(Mujoco.MjStateSize(model, MjtState.FullPhysics)), NPTypeCode.Double);
            Mujoco.MjGetState(model, data, state, MjtState.FullPhysics);
            double[] qpos = (double[])data.qpos.Clone();
            Mujoco.MjStep(model, data, 5);
            Mujoco.MjSetState(model, data, state, MjtState.FullPhysics);
            CollectionAssert.AreEqual(qpos, data.qpos);
            Assert.AreEqual(state.GetDouble(0), data.time);

            using NDArray wrong = np.zeros(new Shape(3), NPTypeCode.Double);
            Assert.AreEqual("state size should equal mj_stateSize(m, sig)",
                Assert.ThrowsExactly<TypeError>(() => Mujoco.MjGetState(model, data, wrong, MjtState.FullPhysics)).Message);
        }

        /// <summary>
        ///     <c>gymnasium.envs.mujoco.utils</c> through a wrapped env: the state round trip, the reset determinism check, and
        ///     the <c>AttributeError</c> for an env that is not a MuJoCo one.
        /// </summary>
        [TestMethod]
        public void Utils_WorkThroughWrappers() {
            using IEnv env = Make("Hopper-v5");
            env.Reset(1).Dispose();
            using NDArray state = MujocoUtils.GetState(env);
            using (var step = env.Step(env.ActionSpace.Sample()))
                Assert.IsNotNull(step.Observation);
            using NDArray returned = MujocoUtils.SetState(env, state);
            using NDArray again = MujocoUtils.GetState(env);
            CollectionAssert.AreEqual(state.ToArray<double>(), again.ToArray<double>());
            MujocoUtils.CheckMujocoResetState(env);

            using var cartPole = new CartPoleEnv();
            Assert.AreEqual("'CartPoleEnv' object has no attribute 'model'",
                Assert.ThrowsExactly<AttributeError>(() => MujocoUtils.GetState(cartPole)).Message);
        }

        /// <summary>The model's bundled names are the 14 XML files Gymnasium ships, in MuJoCo's spelling.</summary>
        [TestMethod]
        public void BundledNames_AreGymnasiumsFiles() {
            CollectionAssert.AreEquivalent(new[] {
                "ant.xml", "half_cheetah.xml", "hopper.xml", "humanoid.xml", "humanoidstandup.xml", "inverted_double_pendulum.xml",
                "inverted_pendulum.xml", "point.xml", "pusher.xml", "pusher_v5.xml", "reacher.xml", "swimmer.xml", "walker2d.xml",
                "walker2d_v5.xml",
            }, MjModel.BundledNames.ToArray());
        }
    }
}
