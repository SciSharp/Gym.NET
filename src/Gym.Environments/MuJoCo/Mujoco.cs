using System;
using Gym.Utils;
using NumSharp;

namespace Gym.MuJoCo {
    /// <summary>
    ///     The <c>mujoco</c> module's functions that Gymnasium's MuJoCo envs, its <c>utils</c> and its tests call (DR-68),
    ///     under .NET names, each <c>[PyName]</c>d to Python's: stepping, the forward dynamics, resetting, the post-step
    ///     forces, kinematics alone, name lookup and the state vector. Each runs Gym.NET's own engine, bit for bit with
    ///     MuJoCo 4.0.0 on win-x64 for Gymnasium's bundled models (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     Every function takes the model and the data as MuJoCo's do, and refuses a data made for another model
    ///     (<see cref="ArgumentException"/>), where MuJoCo's C would read past its arrays.
    /// </remarks>
    [PyName("mujoco")]
    public static class Mujoco {
        /// <summary>MuJoCo's <c>mjNSTATE</c>: the number of state elements, so a valid signature is below 2^15.</summary>
        private const int NState = 15;

        /// <summary>
        ///     Advances the simulation <paramref name="nstep"/> time steps (<c>mj_step</c>): each the forward dynamics at the
        ///     current state, then the model's integrator, which updates the positions, velocities and time.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <param name="nstep">
        ///     The number of steps (Gymnasium passes its frame skip); MuJoCo's Python default is 1, and 0 or a negative
        ///     count steps nothing, as there.
        /// </param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        /// <exception cref="NotSupportedException">The engine met a path it does not reproduce (the message names it).</exception>
        /// <remarks>
        ///     A NaN or a value beyond 1e10 in the positions, velocities or accelerations resets the data first, as MuJoCo
        ///     does (its warning is not recorded). Footgun: the external contact forces (<see cref="MjData.CfrcExt"/>) are not
        ///     updated by a step; call <see cref="MjRnePostConstraint"/> after it, as Gymnasium's envs do.
        /// </remarks>
        [PyName("mj_step")]
        public static void MjStep(MjModel model, MjData data, int nstep = 1) {
            Check(model, data);
            for (int i = 0; i < nstep; i++)
                data.Step();
        }

        /// <summary>
        ///     The forward dynamics at the current state, without integrating (<c>mj_forward</c>): positions, velocities,
        ///     forces, the unconstrained and the constrained accelerations. Gymnasium's <c>set_state</c> calls it after
        ///     writing the positions and velocities.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        /// <exception cref="NotSupportedException">The engine met a path it does not reproduce (the message names it).</exception>
        [PyName("mj_forward")]
        public static void MjForward(MjModel model, MjData data) {
            Check(model, data);
            data.Forward();
        }

        /// <summary>
        ///     Resets the data to the model's defaults (<c>mj_resetData</c>): every array cleared, the positions at the
        ///     reference configuration, at rest, time 0. Nothing is computed from that state until the next
        ///     <see cref="MjForward"/> or <see cref="MjStep"/>.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        [PyName("mj_resetData")]
        public static void MjResetData(MjModel model, MjData data) {
            Check(model, data);
            data.ResetData();
        }

        /// <summary>
        ///     The forces the forward dynamics leaves out (<c>mj_rnePostConstraint</c>): the bodies' external contact forces
        ///     (<see cref="MjData.CfrcExt"/>) from the last solve. Gymnasium's envs call it after each step's physics.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        [PyName("mj_rnePostConstraint")]
        public static void MjRnePostConstraint(MjModel model, MjData data) {
            Check(model, data);
            data.RnePostConstraint();
        }

        /// <summary>
        ///     The frames alone from the positions (<c>mj_kinematics</c>): bodies, centers of mass, joints, geoms and sites,
        ///     without the dynamics. Gymnasium's tests call it to read positions without stepping.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        [PyName("mj_kinematics")]
        public static void MjKinematics(MjModel model, MjData data) {
            Check(model, data);
            data.Kinematics();
        }

        /// <summary>
        ///     The index of a named element (<c>mj_name2id</c>): a body (also as <see cref="MjtObj.XBody"/>), joint, geom,
        ///     site, camera, light, actuator, tendon, material or texture of the model.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="type">The element kind.</param>
        /// <param name="name">The name; an empty name matches nothing.</param>
        /// <returns>The index, or -1 when no element of that kind has the name (or the model has no such kind).</returns>
        /// <exception cref="ArgumentNullException">A null model or name.</exception>
        [PyName("mj_name2id")]
        public static int MjName2Id(MjModel model, MjtObj type, string name) {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(name);
            string? kind = type switch {
                MjtObj.Body or MjtObj.XBody => "body",
                MjtObj.Joint => "jnt",
                MjtObj.Geom => "geom",
                MjtObj.Site => "site",
                MjtObj.Camera => "cam",
                MjtObj.Light => "light",
                MjtObj.Actuator => "actuator",
                MjtObj.Tendon => "tendon",
                MjtObj.Material => "mat",
                MjtObj.Texture => "tex",
                // every other kind has no elements in the bundled models
                _ => null,
            };
            return kind == null ? -1 : model.NameToId(kind, name);
        }

        /// <summary>The length of the state vector a signature selects (<c>mj_stateSize</c>), its elements' sizes summed.</summary>
        /// <param name="model">The model.</param>
        /// <param name="spec">The signature: one element or a union (<see cref="MjtState.FullPhysics"/>, <see cref="MjtState.Integration"/>).</param>
        /// <returns>The number of values.</returns>
        /// <exception cref="ArgumentNullException">A null model.</exception>
        /// <exception cref="ArgumentOutOfRangeException">A signature below 0 or from 2^15 on, with MuJoCo's message.</exception>
        [PyName("mj_stateSize")]
        public static int MjStateSize(MjModel model, MjtState spec) {
            ArgumentNullException.ThrowIfNull(model);
            CheckSignature(spec);
            int size = 0;
            for (int i = 0; i < NState; i++) {
                var element = (MjtState)(1 << i);
                if ((spec & element) != 0)
                    size += ElementSize(model, element);
            }
            return size;
        }

        /// <summary>
        ///     Copies the state elements a signature selects into <paramref name="state"/> (<c>mj_getState</c>), in MuJoCo's
        ///     element order: time, positions, velocities, activations, history, warm start, controls, applied forces, the
        ///     rest (empty for the bundled models).
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <param name="state">A float64 array of <see cref="MjStateSize"/> values, written in place.</param>
        /// <param name="spec">The signature.</param>
        /// <exception cref="ArgumentNullException">A null argument.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        /// <exception cref="TypeError">The array is not float64, or not of the signature's size (MuJoCo's message).</exception>
        /// <exception cref="ArgumentOutOfRangeException">A signature below 0 or from 2^15 on.</exception>
        [PyName("mj_getState")]
        public static void MjGetState(MjModel model, MjData data, NDArray state, MjtState spec) {
            Check(model, data);
            ArgumentNullException.ThrowIfNull(state);
            CheckSignature(spec);
            CheckStateArray(state, MjStateSize(model, spec));
            long adr = 0;
            for (int i = 0; i < NState; i++) {
                var element = (MjtState)(1 << i);
                if ((spec & element) == 0)
                    continue;
                int size = ElementSize(model, element);
                for (int k = 0; k < size; k++)
                    state.SetDouble(Element(data, element, k), adr + k);
                adr += size;
            }
        }

        /// <summary>
        ///     Writes the state elements a signature selects from <paramref name="state"/> (<c>mj_setState</c>), in
        ///     <see cref="MjGetState"/>'s order. Nothing is recomputed until the next <see cref="MjForward"/> or step.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data, made for <paramref name="model"/>.</param>
        /// <param name="state">The float64 values, <see cref="MjStateSize"/> of them.</param>
        /// <param name="spec">The signature.</param>
        /// <exception cref="ArgumentNullException">A null argument.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        /// <exception cref="TypeError">The array is not float64, or not of the signature's size (MuJoCo's message).</exception>
        /// <exception cref="ArgumentOutOfRangeException">A signature below 0 or from 2^15 on.</exception>
        /// <exception cref="NotSupportedException">
        ///     A nonzero applied force (<see cref="MjtState.QfrcApplied"/>, <see cref="MjtState.XfrcApplied"/>): the engine
        ///     steps without applied forces, so it refuses them rather than ignore them.
        /// </exception>
        [PyName("mj_setState")]
        public static void MjSetState(MjModel model, MjData data, NDArray state, MjtState spec) {
            Check(model, data);
            ArgumentNullException.ThrowIfNull(state);
            CheckSignature(spec);
            CheckStateArray(state, MjStateSize(model, spec));
            // the applied forces are checked before anything is written, so a refused state leaves the data as it was
            long adr = 0;
            for (int i = 0; i < NState; i++) {
                var element = (MjtState)(1 << i);
                if ((spec & element) == 0)
                    continue;
                int size = ElementSize(model, element);
                if (element is MjtState.QfrcApplied or MjtState.XfrcApplied)
                    for (int k = 0; k < size; k++)
                        if (state.GetDouble(adr + k) != 0)
                            throw new NotSupportedException("the core: applied forces (qfrc_applied, xfrc_applied) must be zero");
                adr += size;
            }
            adr = 0;
            for (int i = 0; i < NState; i++) {
                var element = (MjtState)(1 << i);
                if ((spec & element) == 0)
                    continue;
                int size = ElementSize(model, element);
                for (int k = 0; k < size; k++)
                    SetElement(data, element, k, state.GetDouble(adr + k));
                adr += size;
            }
        }

        /// <summary>One state element's size (<c>mj_stateElemSize</c>) for the bundled models, whose optional parts are empty.</summary>
        /// <param name="model">The model.</param>
        /// <param name="element">One element (a single bit).</param>
        /// <returns>The number of values.</returns>
        private static int ElementSize(MjModel model, MjtState element) => element switch {
            MjtState.Time => 1,
            MjtState.Qpos => model.nq,
            MjtState.Qvel => model.nv,
            MjtState.Warmstart => model.nv,
            MjtState.Ctrl => model.nu,
            MjtState.QfrcApplied => model.nv,
            MjtState.XfrcApplied => 6 * model.nbody,
            // activations, history, equality flags, mocap, user data, plugins and plasticity: none in the bundled models
            // (the loader refuses models that have them)
            _ => 0,
        };

        /// <summary>Reads value <paramref name="k"/> of one state element.</summary>
        /// <param name="data">The data.</param>
        /// <param name="element">The element.</param>
        /// <param name="k">The index within it.</param>
        /// <returns>The value; the applied forces, which the engine does not have, are 0.</returns>
        private static double Element(MjData data, MjtState element, int k) => element switch {
            MjtState.Time => data.time,
            MjtState.Qpos => data.qpos[k],
            MjtState.Qvel => data.qvel[k],
            MjtState.Warmstart => data.qacc_warmstart[k],
            MjtState.Ctrl => data.ctrl[k],
            _ => 0,
        };

        /// <summary>Writes value <paramref name="k"/> of one state element; applied forces were checked to be 0 and are dropped.</summary>
        /// <param name="data">The data.</param>
        /// <param name="element">The element.</param>
        /// <param name="k">The index within it.</param>
        /// <param name="value">The value.</param>
        private static void SetElement(MjData data, MjtState element, int k, double value) {
            switch (element) {
                case MjtState.Time:
                    data.time = value;
                    break;
                case MjtState.Qpos:
                    data.qpos[k] = value;
                    break;
                case MjtState.Qvel:
                    data.qvel[k] = value;
                    break;
                case MjtState.Warmstart:
                    data.qacc_warmstart[k] = value;
                    break;
                case MjtState.Ctrl:
                    data.ctrl[k] = value;
                    break;
            }
        }

        /// <summary>Requires a model and a data made for it.</summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The data.</param>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <exception cref="ArgumentException">The data was made for another model.</exception>
        private static void Check(MjModel model, MjData data) {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(data);
            if (!ReferenceEquals(data.Model, model))
                throw new ArgumentException("the data was made for another MjModel", nameof(data));
        }

        /// <summary>Requires a signature MuJoCo accepts.</summary>
        /// <param name="spec">The signature.</param>
        /// <exception cref="ArgumentOutOfRangeException">Below 0 or from 2^15 on, with MuJoCo's message.</exception>
        private static void CheckSignature(MjtState spec) {
            if ((int)spec < 0)
                throw new ArgumentOutOfRangeException(nameof(spec), $"invalid state signature {(int)spec} < 0");
            if ((int)spec >= 1 << NState)
                throw new ArgumentOutOfRangeException(nameof(spec), $"invalid state signature {(int)spec} >= 2^mjNSTATE");
        }

        /// <summary>Requires a float64 array of the state's size, as MuJoCo's binding does.</summary>
        /// <param name="state">The array.</param>
        /// <param name="size">The size the signature needs.</param>
        /// <exception cref="TypeError">Another size, with the binding's message, or another dtype.</exception>
        private static void CheckStateArray(NDArray state, int size) {
            if (state.typecode != NPTypeCode.Double)
                throw new TypeError($"the state array must be float64, not {state.typecode}");
            if (state.size != size)
                throw new TypeError("state size should equal mj_stateSize(m, sig)");
        }
    }

    /// <summary>
    ///     MuJoCo's state elements (<c>mujoco.mjtState</c>), bit flags of a state signature, and its usual unions:
    ///     <see cref="FullPhysics"/> (Gymnasium's <c>get_state</c> default) and <see cref="Integration"/> (its reset check's).
    /// </summary>
    [Flags]
    [PyName("mjtState")]
    public enum MjtState {
        /// <summary>The simulation time (<c>mjSTATE_TIME</c>).</summary>
        [PyName("mjSTATE_TIME")]
        Time = 1,

        /// <summary>The positions (<c>mjSTATE_QPOS</c>).</summary>
        [PyName("mjSTATE_QPOS")]
        Qpos = 2,

        /// <summary>The velocities (<c>mjSTATE_QVEL</c>).</summary>
        [PyName("mjSTATE_QVEL")]
        Qvel = 4,

        /// <summary>The actuator activations (<c>mjSTATE_ACT</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_ACT")]
        Act = 8,

        /// <summary>The delay buffers (<c>mjSTATE_HISTORY</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_HISTORY")]
        History = 16,

        /// <summary>The solver's warm start (<c>mjSTATE_WARMSTART</c>), the last step's acceleration.</summary>
        [PyName("mjSTATE_WARMSTART")]
        Warmstart = 32,

        /// <summary>The controls (<c>mjSTATE_CTRL</c>).</summary>
        [PyName("mjSTATE_CTRL")]
        Ctrl = 64,

        /// <summary>The applied joint forces (<c>mjSTATE_QFRC_APPLIED</c>), always zero here.</summary>
        [PyName("mjSTATE_QFRC_APPLIED")]
        QfrcApplied = 128,

        /// <summary>The applied body forces (<c>mjSTATE_XFRC_APPLIED</c>), always zero here.</summary>
        [PyName("mjSTATE_XFRC_APPLIED")]
        XfrcApplied = 256,

        /// <summary>The equality constraints' flags (<c>mjSTATE_EQ_ACTIVE</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_EQ_ACTIVE")]
        EqActive = 512,

        /// <summary>The mocap bodies' positions (<c>mjSTATE_MOCAP_POS</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_MOCAP_POS")]
        MocapPos = 1024,

        /// <summary>The mocap bodies' orientations (<c>mjSTATE_MOCAP_QUAT</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_MOCAP_QUAT")]
        MocapQuat = 2048,

        /// <summary>The user data (<c>mjSTATE_USERDATA</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_USERDATA")]
        Userdata = 4096,

        /// <summary>The plugins' state (<c>mjSTATE_PLUGIN</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_PLUGIN")]
        Plugin = 8192,

        /// <summary>The plastic deformations (<c>mjSTATE_PLASTIC</c>), none in the bundled models.</summary>
        [PyName("mjSTATE_PLASTIC")]
        Plastic = 16384,

        /// <summary>The user inputs (<c>mjSTATE_USER</c>): controls, applied forces, equality flags, mocap, user data.</summary>
        [PyName("mjSTATE_USER")]
        User = Ctrl | QfrcApplied | XfrcApplied | EqActive | MocapPos | MocapQuat | Userdata,

        /// <summary>The physical state (<c>mjSTATE_PHYSICS</c>): positions, velocities, activations, history, plasticity.</summary>
        [PyName("mjSTATE_PHYSICS")]
        Physics = Qpos | Qvel | Act | History | Plastic,

        /// <summary>The full physical state (<c>mjSTATE_FULLPHYSICS</c>): time, <see cref="Physics"/>, plugins.</summary>
        [PyName("mjSTATE_FULLPHYSICS")]
        FullPhysics = Time | Physics | Plugin,

        /// <summary>Everything a step reads (<c>mjSTATE_INTEGRATION</c>): <see cref="FullPhysics"/>, <see cref="User"/>, the warm start.</summary>
        [PyName("mjSTATE_INTEGRATION")]
        Integration = FullPhysics | User | Warmstart,
    }

    /// <summary>MuJoCo's element kinds (<c>mujoco.mjtObj</c>), for <see cref="Mujoco.MjName2Id"/>.</summary>
    [PyName("mjtObj")]
    public enum MjtObj {
        /// <summary>No kind (<c>mjOBJ_UNKNOWN</c>).</summary>
        [PyName("mjOBJ_UNKNOWN")]
        Unknown = 0,

        /// <summary>A body (<c>mjOBJ_BODY</c>).</summary>
        [PyName("mjOBJ_BODY")]
        Body = 1,

        /// <summary>A body's frame (<c>mjOBJ_XBODY</c>), named as the body.</summary>
        [PyName("mjOBJ_XBODY")]
        XBody = 2,

        /// <summary>A joint (<c>mjOBJ_JOINT</c>).</summary>
        [PyName("mjOBJ_JOINT")]
        Joint = 3,

        /// <summary>A degree of freedom (<c>mjOBJ_DOF</c>), unnamed.</summary>
        [PyName("mjOBJ_DOF")]
        Dof = 4,

        /// <summary>A geom (<c>mjOBJ_GEOM</c>).</summary>
        [PyName("mjOBJ_GEOM")]
        Geom = 5,

        /// <summary>A site (<c>mjOBJ_SITE</c>).</summary>
        [PyName("mjOBJ_SITE")]
        Site = 6,

        /// <summary>A camera (<c>mjOBJ_CAMERA</c>).</summary>
        [PyName("mjOBJ_CAMERA")]
        Camera = 7,

        /// <summary>A light (<c>mjOBJ_LIGHT</c>).</summary>
        [PyName("mjOBJ_LIGHT")]
        Light = 8,

        /// <summary>A flex (<c>mjOBJ_FLEX</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_FLEX")]
        Flex = 9,

        /// <summary>A mesh (<c>mjOBJ_MESH</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_MESH")]
        Mesh = 10,

        /// <summary>A skin (<c>mjOBJ_SKIN</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_SKIN")]
        Skin = 11,

        /// <summary>A height field (<c>mjOBJ_HFIELD</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_HFIELD")]
        HField = 12,

        /// <summary>A texture (<c>mjOBJ_TEXTURE</c>).</summary>
        [PyName("mjOBJ_TEXTURE")]
        Texture = 13,

        /// <summary>A material (<c>mjOBJ_MATERIAL</c>).</summary>
        [PyName("mjOBJ_MATERIAL")]
        Material = 14,

        /// <summary>A geom pair (<c>mjOBJ_PAIR</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_PAIR")]
        Pair = 15,

        /// <summary>An excluded body pair (<c>mjOBJ_EXCLUDE</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_EXCLUDE")]
        Exclude = 16,

        /// <summary>An equality constraint (<c>mjOBJ_EQUALITY</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_EQUALITY")]
        Equality = 17,

        /// <summary>A tendon (<c>mjOBJ_TENDON</c>).</summary>
        [PyName("mjOBJ_TENDON")]
        Tendon = 18,

        /// <summary>An actuator (<c>mjOBJ_ACTUATOR</c>).</summary>
        [PyName("mjOBJ_ACTUATOR")]
        Actuator = 19,

        /// <summary>A sensor (<c>mjOBJ_SENSOR</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_SENSOR")]
        Sensor = 20,

        /// <summary>A numeric field (<c>mjOBJ_NUMERIC</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_NUMERIC")]
        Numeric = 21,

        /// <summary>A text field (<c>mjOBJ_TEXT</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_TEXT")]
        Text = 22,

        /// <summary>A tuple (<c>mjOBJ_TUPLE</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_TUPLE")]
        Tuple = 23,

        /// <summary>A keyframe (<c>mjOBJ_KEY</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_KEY")]
        Key = 24,

        /// <summary>A plugin instance (<c>mjOBJ_PLUGIN</c>), none in the bundled models.</summary>
        [PyName("mjOBJ_PLUGIN")]
        Plugin = 25,

        /// <summary>A frame of the model's specification (<c>mjOBJ_FRAME</c>), not in compiled models.</summary>
        [PyName("mjOBJ_FRAME")]
        Frame = 100,

        /// <summary>A defaults class of the specification (<c>mjOBJ_DEFAULT</c>), not in compiled models.</summary>
        [PyName("mjOBJ_DEFAULT")]
        Default = 101,

        /// <summary>The model itself (<c>mjOBJ_MODEL</c>).</summary>
        [PyName("mjOBJ_MODEL")]
        Model = 102,
    }
}
