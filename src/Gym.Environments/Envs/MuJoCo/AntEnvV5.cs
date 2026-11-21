using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Ant-v5: a four-legged ant (a torso and four two-link legs on a free joint) driven by 8 hip and ankle torques in
    ///     [-1, 1]. The reward pays the main body's forward velocity and staying healthy, and charges the squared torques and
    ///     the clipped contact forces; a non-finite state or a torso height outside [0.2, 1.0] ends the episode. Byte for byte
    ///     what Gymnasium's Ant-v5 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the positions without the root's x and y (unless asked for), the velocities, then the
    ///         contact forces of every body but the world clipped to <c>contact_force_range</c> (unless
    ///         <c>include_cfrc_ext_in_observation=False</c>), 105 values by default. Reward:
    ///         <c>x_velocity * forward_reward_weight + healthy_reward * is_healthy - ctrl_cost_weight * sum(action**2) -
    ///         contact_cost_weight * sum(clip(cfrc_ext)**2)</c>, the velocity measured on <c>main_body</c>'s frame. Info:
    ///         <c>x_position</c>, <c>y_position</c>, <c>distance_from_origin</c> (of the root joint), <c>x_velocity</c>,
    ///         <c>y_velocity</c>, <c>reward_forward</c>, <c>reward_ctrl</c>, <c>reward_contact</c>, <c>reward_survive</c>; the
    ///         reset's info holds the first three.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/ant_v5.py#L14">gymnasium/envs/mujoco/ant_v5.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("AntEnv")]
    public partial class AntEnvV5 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>AntEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default,
        ///     <c>{"distance": 4.0}</c>, and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>; Gymnasium's default is the int 1).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="contactCostWeight">The weight of the squared clipped contact forces (<c>contact_cost_weight</c>).</param>
        /// <param name="healthyReward">The reward for each healthy step (<c>healthy_reward</c>).</param>
        /// <param name="mainBody">
        ///     The body whose motion is the forward velocity (<c>main_body</c>): its index (an integer, a <see langword="bool"/>
        ///     or a 0-d integer array) or its name; <see langword="null"/> is Gymnasium's default, body 1 (the torso). Resolved
        ///     at each step, as Gymnasium resolves it, so a bad one fails at the first step with MuJoCo's error.
        /// </param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The inclusive range of a healthy torso height (<c>healthy_z_range</c>); <see langword="null"/> is (0.2, 1.0).</param>
        /// <param name="contactForceRange">The range the contact forces are clipped to (<c>contact_force_range</c>); <see langword="null"/> is (-1.0, 1.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="includeCfrcExtInObservation">Whether the clipped contact forces join the observation (<c>include_cfrc_ext_in_observation</c>).</param>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="NotSupportedException"><paramref name="xmlFile"/> is not a bundled model.</exception>
        /// <exception cref="OverflowError">A frame skip of 0 (Python's <c>OverflowError</c> deriving the frame rate).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public AntEnvV5(string xmlFile = "ant.xml", int frameSkip = 5, IReadOnlyDictionary<string, object?>? defaultCameraConfig = null,
                        double forwardRewardWeight = 1, double ctrlCostWeight = 0.5, double contactCostWeight = 5e-4, double healthyReward = 1.0,
                        object? mainBody = null, bool terminateWhenUnhealthy = true,
                        (double Low, double High)? healthyZRange = null, (double Low, double High)? contactForceRange = null,
                        double resetNoiseScale = 0.1, bool excludeCurrentPositionsFromObservation = true,
                        bool includeCfrcExtInObservation = true, string? renderMode = null, int width = DefaultSize, int height = DefaultSize,
                        int? cameraId = null, string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                        IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base(xmlFile, frameSkip, null, ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            ContactCostWeight = contactCostWeight;
            HealthyRewardWeight = healthyReward;
            TerminateWhenUnhealthy = terminateWhenUnhealthy;
            HealthyZRange = healthyZRange ?? (0.2, 1.0);
            ContactForceRange = contactForceRange ?? (-1.0, 1.0);
            MainBody = mainBody ?? 1L;
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
            IncludeCfrcExtInObservation = includeCfrcExtInObservation;
            Metadata = MetadataFromDt();
            int skipped = excludeCurrentPositionsFromObservation ? 2 : 0;
            int contacts = includeCfrcExtInObservation ? (Model.Nbody - 1) * 6 : 0;
            ObservationSpace = UnboundedBox(Model.Nq + Model.Nv - skipped + contacts);
            ObservationStructure = new Info {
                ["skipped_qpos"] = (long)skipped,
                ["qpos"] = (long)(Model.Nq - skipped),
                ["qvel"] = (long)Model.Nv,
                ["cfrc_ext"] = (long)contacts,
            };
        }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, <c>{"distance": 4.0}</c>: a new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> { ["distance"] = 4.0 };

        /// <summary>The weight of the forward velocity (<c>_forward_reward_weight</c>), read every step.</summary>
        [PyName("_forward_reward_weight")]
        public double ForwardRewardWeight { get; set; }

        /// <summary>The weight of the squared torques (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The weight of the squared clipped contact forces (<c>_contact_cost_weight</c>), read every step.</summary>
        [PyName("_contact_cost_weight")]
        public double ContactCostWeight { get; set; }

        /// <summary>The reward for each healthy step (<c>_healthy_reward</c>), read every step.</summary>
        [PyName("_healthy_reward")]
        public double HealthyRewardWeight { get; set; }

        /// <summary>Whether an unhealthy state ends the episode (<c>_terminate_when_unhealthy</c>), read every step.</summary>
        [PyName("_terminate_when_unhealthy")]
        public bool TerminateWhenUnhealthy { get; set; }

        /// <summary>The inclusive range of a healthy torso height (<c>_healthy_z_range</c>).</summary>
        [PyName("_healthy_z_range")]
        public (double Low, double High) HealthyZRange { get; set; }

        /// <summary>The range the contact forces are clipped to (<c>_contact_force_range</c>), for the observation and the cost.</summary>
        [PyName("_contact_force_range")]
        public (double Low, double High) ContactForceRange { get; set; }

        /// <summary>The body whose motion is the forward velocity (<c>_main_body</c>): an index or a name, resolved at each step.</summary>
        [PyName("_main_body")]
        public object? MainBody { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>
        ///     Whether the root's x and y leave the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
        ///     changing it after construction changes the observation's length but not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>Whether the clipped contact forces join the observation (<c>_include_cfrc_ext_in_observation</c>).</summary>
        [PyName("_include_cfrc_ext_in_observation")]
        public bool IncludeCfrcExtInObservation { get; set; }

        /// <summary>
        ///     How the observation is laid out (<c>observation_structure</c>): <c>skipped_qpos</c>, <c>qpos</c>, <c>qvel</c> and
        ///     <c>cfrc_ext</c> counts.
        /// </summary>
        public Info ObservationStructure { get; }

        /// <summary>
        ///     Whether the ant is healthy now (<c>is_healthy</c>): every position and velocity finite and the torso's height
        ///     inside the inclusive z range.
        /// </summary>
        [PyName("is_healthy")]
        public bool IsHealthy {
            get {
                for (int i = 0; i < Model.Nq; i++)
                    if (!double.IsFinite(Data.qpos[i]))
                        return false;
                for (int i = 0; i < Model.Nv; i++)
                    if (!double.IsFinite(Data.qvel[i]))
                        return false;
                double z = Data.qpos[2];
                return HealthyZRange.Low <= z && z <= HealthyZRange.High;
            }
        }

        /// <summary>The healthy reward now (<c>healthy_reward</c>): <c>is_healthy * _healthy_reward</c>, an <c>np.float64</c> there.</summary>
        [PyName("healthy_reward")]
        public double HealthyReward => HealthyRewardScalar().Value;

        /// <summary>
        ///     The contact forces clipped to <see cref="ContactForceRange"/> (<c>contact_forces</c>), one row of 6 per body
        ///     (the world's included). A new <c>(nbody, 6)</c> array on every read, which the caller owns.
        /// </summary>
        [PyName("contact_forces")]
        public NDArray ContactForces {
            get {
                var clipped = new double[Data.cfrc_ext.Length];
                for (int i = 0; i < clipped.Length; i++)
                    clipped[i] = PyMath.Clip(Data.cfrc_ext[i], ContactForceRange.Low, ContactForceRange.High);
                using NDArray flat = np.array(clipped);
                return flat.reshape(Model.Nbody, 6);
            }
        }

        /// <summary>The contact cost now (<c>contact_cost</c>): <c>contact_cost_weight * sum(contact_forces**2)</c>.</summary>
        [PyName("contact_cost")]
        public double ContactCost => ContactCostWeight * MujocoMath.SumOfClippedSquares(Data.cfrc_ext, ContactForceRange.Low, ContactForceRange.High);

        /// <summary>
        ///     The control cost of <paramref name="action"/> (<c>control_cost</c>): <c>ctrl_cost_weight * sum(action**2)</c>,
        ///     computed in the action's dtype and returned widened exactly.
        /// </summary>
        /// <param name="action">The action.</param>
        /// <returns>The cost.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        [PyName("control_cost")]
        public double ControlCost(NDArray action) => MujocoMath.SumOfSquares(action).Times(CtrlCostWeight).Value;

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 8 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, whether an unhealthy state ended the episode, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(8,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        /// <exception cref="KeyError"><see cref="MainBody"/> names no body (MuJoCo's message).</exception>
        /// <exception cref="IndexError"><see cref="MainBody"/> is an index outside the model's bodies (MuJoCo's message).</exception>
        /// <exception cref="TypeError"><see cref="MainBody"/> is neither an integer nor a string (pybind11's message).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            int body = MainBodyId();
            double xBefore = Data.xpos[3 * body], yBefore = Data.xpos[3 * body + 1];
            DoSimulation(action, FrameSkip);
            body = MainBodyId();
            double xAfter = Data.xpos[3 * body], yAfter = Data.xpos[3 * body + 1];
            double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;

            NDArray observation = GetObs();
            NpScalar forwardReward = NpScalar.Float64(xVelocity) * NpScalar.PyFloat(ForwardRewardWeight);
            NpScalar healthyReward = HealthyRewardScalar();
            NpScalar rewards = forwardReward + healthyReward;
            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NpScalar contactCost = NpScalar.PyFloat(ContactCostWeight)
                                   * NpScalar.Float64(MujocoMath.SumOfClippedSquares(Data.cfrc_ext, ContactForceRange.Low, ContactForceRange.High));
            NpScalar costs = ctrlCost + contactCost;
            NpScalar reward = rewards - costs;
            bool terminated = !IsHealthy && TerminateWhenUnhealthy;
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
                ["y_position"] = MujocoMath.Float64(Data.qpos[1]),
                ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(Data.qpos[0], Data.qpos[1])),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["y_velocity"] = MujocoMath.Float64(yVelocity),
                ["reward_forward"] = MujocoMath.InfoValue(forwardReward),
                ["reward_ctrl"] = MujocoMath.InfoValue(-ctrlCost),
                ["reward_contact"] = MujocoMath.InfoValue(-contactCost),
                ["reward_survive"] = MujocoMath.InfoValue(healthyReward),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, terminated, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-scale, scale, nq)</c>, then
        ///     velocities <c>init_qvel + scale * standard_normal(nv)</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            double noiseLow = -ResetNoiseScale, noiseHigh = ResetNoiseScale;
            double[] qpos = AddNoise(InitQpos, UniformNoise(noiseLow, noiseHigh, Model.Nq));
            double[] normal = StandardNormalNoise(Model.Nv);
            for (int i = 0; i < normal.Length; i++)
                normal[i] = ResetNoiseScale * normal[i];
            double[] qvel = AddNoise(InitQvel, normal);
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>
        ///     The reset's info (<c>_get_reset_info</c>): <c>x_position</c>, <c>y_position</c> and
        ///     <c>distance_from_origin</c> of the root joint, <c>np.float64</c>s.
        /// </summary>
        /// <returns>The info.</returns>
        protected override Info GetResetInfo() => new() {
            ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
            ["y_position"] = MujocoMath.Float64(Data.qpos[1]),
            ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(Data.qpos[0], Data.qpos[1])),
        };

        /// <summary><c>is_healthy * _healthy_reward</c>: NumPy's <c>np.bool_</c> times a Python <c>float</c>, so an <c>np.float64</c>.</summary>
        /// <returns>The healthy reward and its kind.</returns>
        private NpScalar HealthyRewardScalar() => NpScalar.Float64((IsHealthy ? 1.0 : 0.0) * HealthyRewardWeight);

        /// <summary>
        ///     <c>self.data.body(self._main_body)</c>'s index: a name looked up, an integer (Python's <c>bool</c> and NumPy's
        ///     integer scalars included) checked against the model's bodies, as MuJoCo's binding accepts them.
        /// </summary>
        /// <returns>The body's index.</returns>
        /// <exception cref="KeyError">An unknown name.</exception>
        /// <exception cref="IndexError">An index outside <c>[0, nbody)</c>.</exception>
        /// <exception cref="TypeError">Anything else, or an integer beyond 32 bits, as pybind11 refuses them.</exception>
        private int MainBodyId() {
            switch (MainBody) {
                case string name:
                    return Data.Body(name).Id;
                case bool flag:
                    return Data.Body(flag ? 1 : 0).Id;
                case sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger:
                    return BodyByInteger(MainBody is BigInteger big ? big : new BigInteger(Convert.ToDecimal(MainBody)));
                case NDArray { ndim: 0 } scalar when scalar.typecode is not (NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Half
                                                                               or NPTypeCode.Decimal or NPTypeCode.Complex or NPTypeCode.String
                                                                               or NPTypeCode.Char):
                    // NumPy's integer scalars (np.int64(1)) and np.bool_ index like ints
                    return BodyByInteger(scalar.typecode == NPTypeCode.UInt64 ? new BigInteger(scalar.GetUInt64()) : new BigInteger(scalar.GetInt64()));
                default:
                    throw new TypeError(
                        "body(): incompatible function arguments. The following argument types are supported:\n"
                        + "    1. (self: mujoco._structs.MjData, arg0: typing.SupportsInt | typing.SupportsIndex) -> mujoco::python::MjDataBodyViews\n"
                        + "    2. (self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataBodyViews\n\n"
                        + $"Invoked with: <mujoco._structs.MjData object>, {PyBuiltins.Repr(MainBody)}");
            }
        }

        /// <summary>A body by an integer index, refused as pybind11 refuses one beyond a C <c>int</c>.</summary>
        /// <param name="index">The index.</param>
        /// <returns>The body's index.</returns>
        /// <exception cref="IndexError">An index outside <c>[0, nbody)</c>.</exception>
        /// <exception cref="TypeError">An index beyond 32 bits.</exception>
        private int BodyByInteger(BigInteger index) {
            if (index < int.MinValue || index > int.MaxValue) {
                throw new TypeError(
                    "body(): incompatible function arguments. The following argument types are supported:\n"
                    + "    1. (self: mujoco._structs.MjData, arg0: typing.SupportsInt | typing.SupportsIndex) -> mujoco::python::MjDataBodyViews\n"
                    + "    2. (self: mujoco._structs.MjData, name: str = '') -> mujoco::python::MjDataBodyViews\n\n"
                    + $"Invoked with: <mujoco._structs.MjData object>, {index}");
            }
            return Data.Body((int)index).Id;
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the positions (the root's x and y left out unless asked for), the velocities,
        ///     then the clipped contact forces of every body but the world (unless left out).
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 2 : 0;
            int positions = Model.Nq - skipped;
            int contacts = IncludeCfrcExtInObservation ? (Model.Nbody - 1) * 6 : 0;
            var values = new double[positions + Model.Nv + contacts];
            Array.Copy(Data.qpos, skipped, values, 0, positions);
            Array.Copy(Data.qvel, 0, values, positions, Model.Nv);
            // contact_forces[1:]: the world body's row (the first 6 values) left out
            for (int i = 0; i < contacts; i++)
                values[positions + Model.Nv + i] = PyMath.Clip(Data.cfrc_ext[6 + i], ContactForceRange.Low, ContactForceRange.High);
            return np.array(values);
        }
    }
}
