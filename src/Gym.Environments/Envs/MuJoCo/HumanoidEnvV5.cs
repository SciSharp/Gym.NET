using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.MuJoCo;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Humanoid-v5: a 3D humanoid (torso, abdomen, legs and arms on a free joint) driven by 17 joint torques in
    ///     [-0.4, 0.4]. The reward pays the center of mass's forward velocity and staying healthy, and charges the squared
    ///     controls and the external contact forces; a torso height outside (1, 2) ends the episode. Byte for byte what
    ///     Gymnasium's Humanoid-v5 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the positions without the root's x and y (unless asked for), the velocities, then (each
    ///         unless switched off) the bodies' inertias <c>cinert</c> and velocities <c>cvel</c> and the external forces
    ///         <c>cfrc_ext</c>, all without the world body's row, and the actuator forces <c>qfrc_actuator</c> without the
    ///         root's 6 dofs: 348 values by default. Reward: <c>forward_reward_weight * x_velocity + healthy_reward *
    ///         is_healthy - ctrl_cost_weight * sum(ctrl**2) - clip(contact_cost_weight * sum(cfrc_ext**2), *contact_cost_range)</c>,
    ///         all float64 (the control cost reads <c>data.ctrl</c>, so it does not depend on the action's dtype). Info:
    ///         <c>x_position</c>, <c>y_position</c>, <c>tendon_length</c>, <c>tendon_velocity</c>,
    ///         <c>distance_from_origin</c>, <c>x_velocity</c>, <c>y_velocity</c>, <c>reward_survive</c>,
    ///         <c>reward_forward</c>, <c>reward_ctrl</c>, <c>reward_contact</c>; the reset's info holds the first five.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/humanoid_v5.py#L24">gymnasium/envs/mujoco/humanoid_v5.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("HumanoidEnv")]
    public partial class HumanoidEnvV5 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>HumanoidEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default (tracking
        ///     body 1 from 4 m, looking at (0, 0, 2), 20° from above), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared controls (<c>ctrl_cost_weight</c>).</param>
        /// <param name="contactCostWeight">The weight of the squared external forces (<c>contact_cost_weight</c>).</param>
        /// <param name="contactCostRange">The range the contact cost is clipped to (<c>contact_cost_range</c>); <see langword="null"/> is (-inf, 10.0).</param>
        /// <param name="healthyReward">The reward for each healthy step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The exclusive range of a healthy torso height (<c>healthy_z_range</c>); <see langword="null"/> is (1.0, 2.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="includeCinertInObservation">Whether the bodies' inertias join the observation (<c>include_cinert_in_observation</c>).</param>
        /// <param name="includeCvelInObservation">Whether the bodies' velocities join the observation (<c>include_cvel_in_observation</c>).</param>
        /// <param name="includeQfrcActuatorInObservation">Whether the actuator forces join the observation (<c>include_qfrc_actuator_in_observation</c>).</param>
        /// <param name="includeCfrcExtInObservation">Whether the external forces join the observation (<c>include_cfrc_ext_in_observation</c>).</param>
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
        public HumanoidEnvV5(string xmlFile = "humanoid.xml", int frameSkip = 5, IReadOnlyDictionary<string, object?>? defaultCameraConfig = null,
                             double forwardRewardWeight = 1.25, double ctrlCostWeight = 0.1, double contactCostWeight = 5e-7,
                             (double Low, double High)? contactCostRange = null, double healthyReward = 5.0, bool terminateWhenUnhealthy = true,
                             (double Low, double High)? healthyZRange = null, double resetNoiseScale = 1e-2,
                             bool excludeCurrentPositionsFromObservation = true, bool includeCinertInObservation = true,
                             bool includeCvelInObservation = true, bool includeQfrcActuatorInObservation = true,
                             bool includeCfrcExtInObservation = true, string? renderMode = null, int width = DefaultSize, int height = DefaultSize,
                             int? cameraId = null, string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                             IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base(xmlFile, frameSkip, null, ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            ContactCostWeight = contactCostWeight;
            ContactCostRange = contactCostRange ?? (double.NegativeInfinity, 10.0);
            HealthyRewardWeight = healthyReward;
            TerminateWhenUnhealthy = terminateWhenUnhealthy;
            HealthyZRange = healthyZRange ?? (1.0, 2.0);
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
            IncludeCinertInObservation = includeCinertInObservation;
            IncludeCvelInObservation = includeCvelInObservation;
            IncludeQfrcActuatorInObservation = includeQfrcActuatorInObservation;
            IncludeCfrcExtInObservation = includeCfrcExtInObservation;
            Metadata = MetadataFromDt();
            int skipped = excludeCurrentPositionsFromObservation ? 2 : 0;
            int bodies = Model.Nbody - 1;
            int cinert = includeCinertInObservation ? bodies * 10 : 0;
            int cvel = includeCvelInObservation ? bodies * 6 : 0;
            int actuator = includeQfrcActuatorInObservation ? Model.Nv - 6 : 0;
            int external = includeCfrcExtInObservation ? bodies * 6 : 0;
            ObservationSpace = UnboundedBox(Model.Nq + Model.Nv - skipped + cinert + cvel + actuator + external);
            ObservationStructure = new Info {
                ["skipped_qpos"] = (long)skipped,
                ["qpos"] = (long)(Model.Nq - skipped),
                ["qvel"] = (long)Model.Nv,
                ["cinert"] = (long)cinert,
                ["cvel"] = (long)cvel,
                ["qfrc_actuator"] = (long)actuator,
                ["cfrc_ext"] = (long)external,
                ["ten_length"] = 0L,
                ["ten_velocity"] = 0L,
            };
        }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>
        ///     Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> 1, <c>distance</c> 4.0, <c>lookat</c> (0, 0, 2.0),
        ///     <c>elevation</c> -20.0. A new dictionary on every read.
        /// </summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 1L, ["distance"] = 4.0, ["lookat"] = new[] { 0.0, 0.0, 2.0 }, ["elevation"] = -20.0,
        };

        /// <summary>The weight of the forward velocity (<c>_forward_reward_weight</c>), read every step.</summary>
        [PyName("_forward_reward_weight")]
        public double ForwardRewardWeight { get; set; }

        /// <summary>The weight of the squared controls (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The weight of the squared external forces (<c>_contact_cost_weight</c>), read every step.</summary>
        [PyName("_contact_cost_weight")]
        public double ContactCostWeight { get; set; }

        /// <summary>The range the contact cost is clipped to (<c>_contact_cost_range</c>).</summary>
        [PyName("_contact_cost_range")]
        public (double Low, double High) ContactCostRange { get; set; }

        /// <summary>The reward for each healthy step (<c>_healthy_reward</c>), read every step.</summary>
        [PyName("_healthy_reward")]
        public double HealthyRewardWeight { get; set; }

        /// <summary>Whether an unhealthy state ends the episode (<c>_terminate_when_unhealthy</c>), read every step.</summary>
        [PyName("_terminate_when_unhealthy")]
        public bool TerminateWhenUnhealthy { get; set; }

        /// <summary>The exclusive range of a healthy torso height (<c>_healthy_z_range</c>).</summary>
        [PyName("_healthy_z_range")]
        public (double Low, double High) HealthyZRange { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>
        ///     Whether the root's x and y leave the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
        ///     this and the four <c>Include...</c> flags change the observation's length when changed after construction, but
        ///     not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>Whether the bodies' inertias join the observation (<c>_include_cinert_in_observation</c>).</summary>
        [PyName("_include_cinert_in_observation")]
        public bool IncludeCinertInObservation { get; set; }

        /// <summary>Whether the bodies' velocities join the observation (<c>_include_cvel_in_observation</c>).</summary>
        [PyName("_include_cvel_in_observation")]
        public bool IncludeCvelInObservation { get; set; }

        /// <summary>Whether the actuator forces join the observation (<c>_include_qfrc_actuator_in_observation</c>).</summary>
        [PyName("_include_qfrc_actuator_in_observation")]
        public bool IncludeQfrcActuatorInObservation { get; set; }

        /// <summary>Whether the external forces join the observation (<c>_include_cfrc_ext_in_observation</c>).</summary>
        [PyName("_include_cfrc_ext_in_observation")]
        public bool IncludeCfrcExtInObservation { get; set; }

        /// <summary>
        ///     How the observation is laid out (<c>observation_structure</c>): <c>skipped_qpos</c>, <c>qpos</c>, <c>qvel</c>,
        ///     <c>cinert</c>, <c>cvel</c>, <c>qfrc_actuator</c>, <c>cfrc_ext</c>, and 0 for <c>ten_length</c> and
        ///     <c>ten_velocity</c> (reported in the info, not observed).
        /// </summary>
        public Info ObservationStructure { get; }

        /// <summary>Whether the humanoid is healthy now (<c>is_healthy</c>): the torso's height inside the exclusive z range.</summary>
        [PyName("is_healthy")]
        public bool IsHealthy => HealthyZRange.Low < Data.qpos[2] && Data.qpos[2] < HealthyZRange.High;

        /// <summary>The healthy reward now (<c>healthy_reward</c>): <c>is_healthy * _healthy_reward</c>, an <c>np.float64</c> there.</summary>
        [PyName("healthy_reward")]
        public double HealthyReward => HealthyRewardScalar().Value;

        /// <summary>
        ///     The contact cost now (<c>contact_cost</c>): <c>contact_cost_weight * sum(cfrc_ext**2)</c> clipped to
        ///     <see cref="ContactCostRange"/>.
        /// </summary>
        [PyName("contact_cost")]
        public double ContactCost => PyMath.Clip(ContactCostWeight * MujocoMath.SumOfSquares(Data.cfrc_ext), ContactCostRange.Low, ContactCostRange.High);

        /// <summary>
        ///     The center of mass's x and y (<c>mass_center</c>): <c>einsum("b,bj->j", body_mass, xipos) / body_mass.sum()</c>,
        ///     from the bodies' inertial frames of the last forward pass. A new <c>(2,)</c> float64 array the caller owns.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The simulation.</param>
        /// <returns>The <c>(2,)</c> array.</returns>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <remarks>
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/humanoid_v5.py#L17">gymnasium/envs/mujoco/humanoid_v5.py</see>.
        /// </remarks>
        [PyName("mass_center")]
        public static NDArray MassCenter(MjModel model, MjData data) {
            ArgumentNullException.ThrowIfNull(model);
            ArgumentNullException.ThrowIfNull(data);
            (double x, double y) = MujocoMath.MassCenter(model, data);
            return np.array(new[] { x, y });
        }

        /// <summary>
        ///     The control cost (<c>control_cost</c>): <c>ctrl_cost_weight * sum(ctrl**2)</c> of the simulation's controls,
        ///     float64; <paramref name="action"/> is taken but not read, as in Gymnasium.
        /// </summary>
        /// <param name="action">The action; ignored.</param>
        /// <returns>The cost.</returns>
        [PyName("control_cost")]
        public double ControlCost(NDArray? action) => CtrlCostWeight * MujocoMath.SumOfSquares(Data.ctrl);

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 17 controls, any real dtype (stored as float64).</param>
        /// <returns>The observation, the float64 reward, whether an unhealthy state ended the episode, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(17,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            (double xBefore, double yBefore) = MujocoMath.MassCenter(Model, Data);
            DoSimulation(action, FrameSkip);
            (double xAfter, double yAfter) = MujocoMath.MassCenter(Model, Data);
            double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;

            NDArray observation = GetObs();
            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NpScalar healthyReward = HealthyRewardScalar();
            NpScalar rewards = forwardReward + healthyReward;
            // float64 costs: data.ctrl holds the action widened to float64
            NpScalar ctrlCost = NpScalar.PyFloat(CtrlCostWeight) * NpScalar.Float64(MujocoMath.SumOfSquares(Data.ctrl));
            NpScalar contactCost = NpScalar.Float64(PyMath.Clip(
                ContactCostWeight * MujocoMath.SumOfSquares(Data.cfrc_ext), ContactCostRange.Low, ContactCostRange.High));
            NpScalar costs = ctrlCost + contactCost;
            NpScalar reward = rewards - costs;
            bool terminated = !IsHealthy && TerminateWhenUnhealthy;
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
                ["y_position"] = MujocoMath.Float64(Data.qpos[1]),
                ["tendon_length"] = MujocoMath.Copy(Data.ten_length),
                ["tendon_velocity"] = MujocoMath.Copy(Data.ten_velocity),
                ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(Data.qpos[0], Data.qpos[1])),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["y_velocity"] = MujocoMath.Float64(yVelocity),
                ["reward_survive"] = MujocoMath.InfoValue(healthyReward),
                ["reward_forward"] = MujocoMath.InfoValue(forwardReward),
                ["reward_ctrl"] = MujocoMath.InfoValue(-ctrlCost),
                ["reward_contact"] = MujocoMath.InfoValue(-contactCost),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, terminated, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-scale, scale, nq)</c>, then
        ///     velocities <c>init_qvel + uniform(-scale, scale, nv)</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            double noiseLow = -ResetNoiseScale, noiseHigh = ResetNoiseScale;
            double[] qpos = AddNoise(InitQpos, UniformNoise(noiseLow, noiseHigh, Model.Nq));
            double[] qvel = AddNoise(InitQvel, UniformNoise(noiseLow, noiseHigh, Model.Nv));
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>
        ///     The reset's info (<c>_get_reset_info</c>): <c>x_position</c>, <c>y_position</c>, copies of
        ///     <c>tendon_length</c> and <c>tendon_velocity</c>, and <c>distance_from_origin</c>.
        /// </summary>
        /// <returns>The info.</returns>
        protected override Info GetResetInfo() => new() {
            ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
            ["y_position"] = MujocoMath.Float64(Data.qpos[1]),
            ["tendon_length"] = MujocoMath.Copy(Data.ten_length),
            ["tendon_velocity"] = MujocoMath.Copy(Data.ten_velocity),
            ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(Data.qpos[0], Data.qpos[1])),
        };

        /// <summary><c>is_healthy * _healthy_reward</c>: NumPy's <c>np.bool_</c> times a Python <c>float</c>, so an <c>np.float64</c>.</summary>
        /// <returns>The healthy reward and its kind.</returns>
        private NpScalar HealthyRewardScalar() => NpScalar.Float64((IsHealthy ? 1.0 : 0.0) * HealthyRewardWeight);

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the positions (the root's x and y left out unless asked for), the velocities,
        ///     then each included block: <c>cinert[1:]</c>, <c>cvel[1:]</c>, <c>qfrc_actuator[6:]</c>, <c>cfrc_ext[1:]</c>.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 2 : 0;
            var values = new List<double>(Model.Nq + Model.Nv + Model.Nbody * 22 + Model.Nv);
            values.AddRange(new ArraySegment<double>(Data.qpos, skipped, Model.Nq - skipped));
            values.AddRange(Data.qvel);
            // the world body's rows (10 inertia, 6 velocity and 6 force values) and the free root's 6 dofs are left out
            if (IncludeCinertInObservation)
                values.AddRange(new ArraySegment<double>(Data.cinert, 10, Data.cinert.Length - 10));
            if (IncludeCvelInObservation)
                values.AddRange(new ArraySegment<double>(Data.cvel, 6, Data.cvel.Length - 6));
            if (IncludeQfrcActuatorInObservation)
                values.AddRange(new ArraySegment<double>(Data.qfrc_actuator, 6, Data.qfrc_actuator.Length - 6));
            if (IncludeCfrcExtInObservation)
                values.AddRange(new ArraySegment<double>(Data.cfrc_ext, 6, Data.cfrc_ext.Length - 6));
            return np.array(values.ToArray());
        }
    }
}
