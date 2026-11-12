using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.MuJoCo;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     The base of Gymnasium's MuJoCo envs (<c>MujocoEnv</c>): one embedded model (DR-63) and its simulation, stepped by
    ///     Gym.NET's own engine bit for bit as MuJoCo 4.0.0 steps it on win-x64 (DR-61, DR-64); the action space from the
    ///     actuators' control ranges; <see cref="SetState"/>, <see cref="DoSimulation"/>, <see cref="StateVector"/> and
    ///     <see cref="GetBodyCom"/> for the envs; the reset that clears the data and lets the env draw its start.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Python attributes keep their names through Gym.NET's snake_case mapping: <see cref="Model"/> is <c>model</c>,
    ///         <see cref="Data"/> <c>data</c>, <see cref="InitQpos"/> <c>init_qpos</c>, <see cref="FrameSkip"/>
    ///         <c>frame_skip</c>, <see cref="MujocoRenderer"/> <c>mujoco_renderer</c>. The model and data expose what
    ///         Gymnasium's code reads of MuJoCo's, under .NET names (DR-68).
    ///     </para>
    ///     <para>
    ///         Footgun: only Gymnasium's 14 bundled models exist (<see cref="MjModel.FromXmlPath"/>), so an
    ///         <c>xml_file</c> that is a path is refused rather than compiled. Frames are Gym.NET's own drawing of the scene,
    ///         compared with Gymnasium's and measured, not required equal (DR-62, DR-67).
    ///     </para>
    ///     <para>
    ///         Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_env.py">gymnasium/envs/mujoco/mujoco_env.py</see>.
    ///     </para>
    /// </remarks>
    [PyName("MujocoEnv")]
    public abstract class MujocoEnv : Env<NDArray, NDArray> {
        /// <summary>Gymnasium's <c>DEFAULT_SIZE</c>: the default width and height of a rendered frame, 480.</summary>
        public const int DefaultSize = 480;

        /// <summary>The render modes every MuJoCo env lists (<c>metadata["render_modes"]</c>).</summary>
        internal static readonly string[] RenderModeNames = ["human", "rgb_array", "depth_array", "rgbd_tuple"];

        /// <summary>Whether <see cref="Dispose(bool)"/> released the model and data.</summary>
        private bool released;

        /// <summary>
        ///     Loads the model, makes its data and the action space, and the renderer that draws them (Gymnasium's
        ///     <c>MujocoEnv.__init__</c>).
        /// </summary>
        /// <param name="modelPath">
        ///     Gymnasium's <c>model_path</c>: the bundled model's file name (<c>"ant.xml"</c>); see <see cref="ExpandModelPath"/>.
        /// </param>
        /// <param name="frameSkip">Gymnasium's <c>frame_skip</c>: physics steps per env step.</param>
        /// <param name="observationSpace">The observation space, or <see langword="null"/> for an env that sets it after.</param>
        /// <param name="classMetadata">
        ///     The class's metadata as Python's class attribute holds it before <c>__init__</c>: a <c>render_fps</c> in it
        ///     must equal <c>round(1 / dt)</c>, as Gymnasium asserts (its v4 envs carry one, its v5 envs set theirs after).
        /// </param>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>.</param>
        /// <param name="width">The rendered frame's width in pixels (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height in pixels (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>); -1 is the free camera.</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>), exclusive with <paramref name="cameraId"/>.</param>
        /// <param name="defaultCameraConfig">The free camera's settings (<c>default_camera_config</c>): distance, elevation, azimuth, lookat, trackbodyid, type.</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>), by <c>mjtVisFlag</c> value.</param>
        /// <exception cref="NotSupportedException"><paramref name="modelPath"/> is not one of the bundled models.</exception>
        /// <exception cref="AssertionError">The class metadata's <c>render_fps</c> does not match the time step.</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name, with Gymnasium's message.</exception>
        protected MujocoEnv(string modelPath, int frameSkip, Space<NDArray>? observationSpace, Metadata classMetadata,
                            string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                            string? cameraName = null, IReadOnlyDictionary<string, object?>? defaultCameraConfig = null,
                            int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null) {
            Metadata = classMetadata;
            Fullpath = ExpandModelPath(modelPath);
            Width = width;
            Height = height;
            (Model, Data) = InitializeSimulation();
            using (NDArray qpos = Data.Qpos)
                InitQpos = qpos.copy();
            using (NDArray qvel = Data.Qvel)
                InitQvel = qvel.copy();
            FrameSkip = frameSkip;
            if (Metadata.ContainsKey(Metadata.RenderFpsKey)) {
                long expected = (long)Math.Round(1.0 / Dt, MidpointRounding.ToEven);
                long actual = Convert.ToInt64(Metadata[Metadata.RenderFpsKey]);
                if (expected != actual)
                    throw new AssertionError($"Expected value: {expected}, Actual value: {actual}");
            }
            if (observationSpace != null)
                ObservationSpace = observationSpace;
            SetActionSpace();
            RenderMode = renderMode;
            CameraName = cameraName;
            CameraId = cameraId;
            MujocoRenderer = new MujocoRenderer(Model, Data, defaultCameraConfig, width, height, maxGeom, cameraId, cameraName, visualOptions);
        }

        /// <summary>
        ///     The model (<c>model</c>): the bundled model this env loaded, its own copy. Writing it (colours, the offscreen
        ///     size) changes this env only.
        /// </summary>
        public MjModel Model { get; }

        /// <summary>The simulation (<c>data</c>): the state the env steps, with everything computed from it.</summary>
        public MjData Data { get; }

        /// <summary>
        ///     The reference positions (<c>init_qpos</c>), a float64 copy of the model's <c>qpos0</c> made at construction; the
        ///     resets add their noise to it. The env owns it.
        /// </summary>
        public NDArray InitQpos { get; }

        /// <summary>The reference velocities (<c>init_qvel</c>), zeros, copied at construction; the env owns it.</summary>
        public NDArray InitQvel { get; }

        /// <summary>Physics steps per env step (<c>frame_skip</c>).</summary>
        public int FrameSkip { get; }

        /// <summary>
        ///     The model's file (<c>fullpath</c>): the bundled file name the model was loaded by, since Gym.NET embeds the
        ///     compiled models instead of the XML files (Gymnasium's is a path into its package).
        /// </summary>
        public string Fullpath { get; }

        /// <summary>The rendered frame's width in pixels (<c>width</c>).</summary>
        public int Width { get; }

        /// <summary>The rendered frame's height in pixels (<c>height</c>).</summary>
        public int Height { get; }

        /// <summary>The camera name given to the constructor (<c>camera_name</c>), or <see langword="null"/>.</summary>
        public string? CameraName { get; }

        /// <summary>The camera id given to the constructor (<c>camera_id</c>), or <see langword="null"/>; the renderer resolves the one it uses.</summary>
        public int? CameraId { get; }

        /// <summary>The renderer that draws the frames (<c>mujoco_renderer</c>).</summary>
        public MujocoRenderer MujocoRenderer { get; }

        /// <summary>The env step's duration in seconds (<c>dt</c>): the model's time step times <see cref="FrameSkip"/>.</summary>
        public double Dt => Model.Opt.Timestep * FrameSkip;

        /// <summary>
        ///     Resolves a model path (<c>expand_model_path</c>) to the bundled model it names. Gym.NET has the 14 models
        ///     Gymnasium ships, by file name (DR-63), and no MJCF compiler, so any path is refused.
        /// </summary>
        /// <param name="modelPath">A bundled file name, <c>"ant.xml"</c> to <c>"walker2d_v5.xml"</c>.</param>
        /// <returns>The file name, the key <see cref="MjModel.FromXmlPath"/> loads by.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="modelPath"/> is null.</exception>
        /// <exception cref="NotSupportedException">Not a bundled model's name; the message lists them.</exception>
        [PyName("expand_model_path")]
        public static string ExpandModelPath(string modelPath) {
            ArgumentNullException.ThrowIfNull(modelPath);
            if (Array.IndexOf(MjModel.BundledNames, modelPath) < 0)
                throw new NotSupportedException(
                    $"Gym.NET embeds Gymnasium's {MjModel.BundledNames.Length} compiled MuJoCo models and has no MJCF compiler (DR-63): "
                    + $"'{modelPath}' is not one of {string.Join(", ", MjModel.BundledNames)}.");
            return modelPath;
        }

        /// <summary>
        ///     Loads the model and makes its data (<c>_initialize_simulation</c>), growing the model's offscreen buffer to the
        ///     render size and never shrinking it below the model's own.
        /// </summary>
        /// <returns>The model and its data.</returns>
        /// <exception cref="NotSupportedException">The model file is not bundled.</exception>
        protected virtual (MjModel Model, MjData Data) InitializeSimulation() {
            MjModel model = MjModel.FromXmlPath(Fullpath);
            MjVisualGlobal global = model.Vis.Global;
            global.Offwidth = Math.Max(global.Offwidth, Width);
            global.Offheight = Math.Max(global.Offheight, Height);
            return (model, new MjData(model));
        }

        /// <summary>
        ///     Sets the action space from the actuators' control ranges cast to float32 (<c>_set_action_space</c>): a float32
        ///     <see cref="Box"/>, one dimension per actuator, which the env owns (read it back from
        ///     <see cref="Env{TObs,TAct}.ActionSpace"/>; Gymnasium's also returns it).
        /// </summary>
        protected void SetActionSpace() {
            using NDArray bounds = Model.ActuatorCtrlrange;
            int nu = Model.Nu;
            var low = new float[nu];
            var high = new float[nu];
            for (int i = 0; i < nu; i++) {
                low[i] = (float)bounds.GetDouble(i, 0);
                high[i] = (float)bounds.GetDouble(i, 1);
            }
            using NDArray lowArray = np.array(low);
            using NDArray highArray = np.array(high);
            ActionSpace = new Box(lowArray, highArray, dtype: np.float32);
        }

        /// <summary>
        ///     Sets the joint positions and velocities, then runs the forward dynamics (<c>set_state</c>), so everything
        ///     computed from them is current. The arrays are copied.
        /// </summary>
        /// <param name="qpos">The positions, shape <c>(nq,)</c>.</param>
        /// <param name="qvel">The velocities, shape <c>(nv,)</c>.</param>
        /// <exception cref="ArgumentNullException">A null array.</exception>
        /// <exception cref="AssertionError">A wrong shape (Gymnasium's bare <c>assert</c>).</exception>
        /// <remarks>
        ///     Footgun, as in Gymnasium: the positions and velocities are not the whole state; the warm start and the
        ///     controls carry over, so <see cref="MujocoUtils"/>' state functions are the way to restore a step exactly.
        /// </remarks>
        public void SetState(NDArray qpos, NDArray qvel) {
            ArgumentNullException.ThrowIfNull(qpos);
            ArgumentNullException.ThrowIfNull(qvel);
            if (!SameShape(qpos, Model.Nq) || !SameShape(qvel, Model.Nv))
                throw new AssertionError();
            CopyInto(Data.qpos, qpos);
            CopyInto(Data.qvel, qvel);
            // data.act[:] = None for a model without activations: there is nothing to clear
            Mujoco.MjForward(Model, Data);
        }

        /// <summary>
        ///     The physics of one env step (<c>_step_mujoco_simulation</c>): the controls set, <paramref name="nFrames"/>
        ///     physics steps, then the external forces Gymnasium's envs read (<c>mj_rnePostConstraint</c>).
        /// </summary>
        /// <param name="ctrl">The controls, one per actuator, any float dtype (cast to float64 as NumPy assigns).</param>
        /// <param name="nFrames">The physics steps.</param>
        protected void StepMujocoSimulation(NDArray ctrl, int nFrames) {
            CopyInto(Data.ctrl, ctrl);
            Mujoco.MjStep(Model, Data, nFrames);
            Mujoco.MjRnePostConstraint(Model, Data);
        }

        /// <summary>
        ///     Steps the simulation <paramref name="nFrames"/> times holding the controls (<c>do_simulation</c>), after
        ///     checking their shape.
        /// </summary>
        /// <param name="ctrl">The controls, shape <c>(nu,)</c>.</param>
        /// <param name="nFrames">The physics steps.</param>
        /// <exception cref="ArgumentNullException"><paramref name="ctrl"/> is null.</exception>
        /// <exception cref="ValueError">Another shape, with Gymnasium's message.</exception>
        public void DoSimulation(NDArray ctrl, int nFrames) {
            ArgumentNullException.ThrowIfNull(ctrl);
            if (!SameShape(ctrl, Model.Nu))
                throw new ValueError($"Action dimension mismatch. Expected ({Model.Nu},), found {ShapeRepr(ctrl)}");
            StepMujocoSimulation(ctrl, nFrames);
        }

        /// <summary>
        ///     The positions and velocities in one float64 array (<c>state_vector</c>), a new array the caller owns. Not the
        ///     whole physics state (see <see cref="SetState"/>).
        /// </summary>
        /// <returns>The <c>(nq + nv,)</c> array.</returns>
        public NDArray StateVector() {
            var state = new double[Model.Nq + Model.Nv];
            Array.Copy(Data.qpos, state, Model.Nq);
            Array.Copy(Data.qvel, 0, state, Model.Nq, Model.Nv);
            return np.array(state);
        }

        /// <summary>
        ///     A body's position in the world (<c>get_body_com</c>): its frame's origin, <c>data.body(name).xpos</c>, despite
        ///     the name (not its center of mass). A writable view of the simulation's array, which the caller owns.
        /// </summary>
        /// <param name="bodyName">The body's name.</param>
        /// <returns>The <c>(3,)</c> view.</returns>
        /// <exception cref="KeyError">No body has that name (MuJoCo's message).</exception>
        public NDArray GetBodyCom(string bodyName) => Data.Body(bodyName).Xpos;

        /// <summary>
        ///     Starts an episode (<c>reset</c>): the generator reseeded from <paramref name="seed"/>, the data cleared
        ///     (<c>mj_resetData</c>), then the env's own start (<see cref="ResetModel"/>) and reset info; a human-mode env
        ///     draws the first frame.
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/> to continue the generator's stream.</param>
        /// <param name="options">Ignored, as in Gymnasium's MuJoCo envs.</param>
        /// <returns>The first observation and the reset info.</returns>
        /// <exception cref="Error">A negative seed, as Gymnasium's seeding refuses it.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            Mujoco.MjResetData(Model, Data);
            NDArray ob = ResetModel();
            Info info = GetResetInfo();
            if (RenderMode == "human")
                Render();
            return new ResetResult<NDArray>(ob, info);
        }

        /// <summary>
        ///     The env's start (<c>reset_model</c>): its positions and velocities drawn from <see cref="Env{TObs,TAct}.NpRandom"/>
        ///     in Gymnasium's order and set with <see cref="SetState"/>, and the first observation.
        /// </summary>
        /// <returns>The first observation, a new array.</returns>
        protected abstract NDArray ResetModel();

        /// <summary>The reset's info (<c>_get_reset_info</c>): empty unless the env reports positions at reset.</summary>
        /// <returns>The info.</returns>
        [PyName("_get_reset_info")]
        protected virtual Info GetResetInfo() => new();

        /// <summary>
        ///     Draws a frame in the render mode given at construction (<c>render</c>), through <see cref="MujocoRenderer"/>.
        /// </summary>
        /// <returns><c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c>: the frame; <c>human</c>: nothing.</returns>
        /// <exception cref="AttributeError">No render mode was given (Gymnasium's renderer refuses <c>None</c>).</exception>
        public override RenderResult Render() => MujocoRenderer.Render(RenderMode);

        /// <summary>Closes the renderer's windows and buffers (<c>close</c>); the simulation stays usable, as in Gymnasium.</summary>
        public override void Close() => MujocoRenderer.Close();

        /// <summary>Closes, then releases the model's and data's views and the reference arrays.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs,TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing || released)
                return;
            released = true;
            InitQpos.Dispose();
            InitQvel.Dispose();
            Data.Dispose();
            Model.Dispose();
        }

        /// <summary>The render modes and frame rate a MuJoCo env's metadata holds.</summary>
        /// <param name="renderFps">The frame rate, or <see langword="null"/> for a v5 class before its instance sets one.</param>
        /// <returns>A new metadata dict.</returns>
        internal static Metadata MetadataFor(long? renderFps) {
            var metadata = new Metadata { [Metadata.RenderModesKey] = RenderModeNames.ToList() };
            if (renderFps is { } fps)
                metadata[Metadata.RenderFpsKey] = fps;
            return metadata;
        }

        /// <summary>The v5 envs' instance metadata: the render modes and <c>int(np.round(1.0 / self.dt))</c>.</summary>
        /// <returns>A new metadata dict.</returns>
        protected Metadata MetadataFromDt() => MetadataFor((long)Math.Round(1.0 / Dt, MidpointRounding.ToEven));

        /// <summary>Whether an array has the one-dimensional shape <c>(n,)</c>.</summary>
        /// <param name="array">The array.</param>
        /// <param name="n">The length.</param>
        /// <returns>The answer.</returns>
        private static bool SameShape(NDArray array, int n) => array.ndim == 1 && array.shape[0] == n;

        /// <summary>Python's repr of an array's shape: <c>(7,)</c>, <c>()</c>, <c>(2, 3)</c>.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The tuple's text.</returns>
        private static string ShapeRepr(NDArray array) =>
            array.ndim == 1 ? $"({array.shape[0]},)" : "(" + string.Join(", ", array.shape) + ")";

        /// <summary>Copies an array's values into an engine array, cast to float64 as NumPy's slice assignment casts.</summary>
        /// <param name="target">The engine array, as long as the source.</param>
        /// <param name="source">A one-dimensional array of any real dtype.</param>
        private static void CopyInto(double[] target, NDArray source) {
            for (int i = 0; i < target.Length; i++)
                target[i] = source.GetDouble(i);
        }
    }
}
