using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.MuJoCo;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Humanoid-v4: the humanoid of <see cref="HumanoidEnvV5"/> with v4's fixed model and frame skip, every body's rows
    ///     observed (the world's included), no contact cost, v4's healthy reward (paid on every step while
    ///     <c>terminate_when_unhealthy</c> is set) and the center of mass in the info. Byte for byte what Gymnasium's
    ///     Humanoid-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the positions without the root's x and y (unless asked for), the velocities, and the whole
    ///         <c>cinert</c>, <c>cvel</c>, <c>qfrc_actuator</c> and <c>cfrc_ext</c> arrays; the space declares 376 values
    ///         (378 with the root's x and y). Reward: <c>forward_reward_weight * x_velocity + float(is_healthy or
    ///         terminate_when_unhealthy) * healthy_reward - ctrl_cost_weight * sum(ctrl**2)</c>. Info: <c>reward_linvel</c>,
    ///         <c>reward_quadctrl</c>, <c>reward_alive</c>, then the center of mass's <c>x_position</c>, <c>y_position</c> and
    ///         <c>distance_from_origin</c>, <c>x_velocity</c>, <c>y_velocity</c> and <c>forward_reward</c>. The reset's info
    ///         is empty; <c>make</c> warns that the id is out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/humanoid_v4.py#L22">gymnasium/envs/mujoco/humanoid_v4.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>HumanoidInfoV4 info = step.Info;</c> (<see cref="HumanoidInfoV4"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("HumanoidEnv")]
    public partial class HumanoidEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>HumanoidEnv.__init__</c>): <c>humanoid.xml</c> with a frame skip of 5.</summary>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared controls (<c>ctrl_cost_weight</c>).</param>
        /// <param name="healthyReward">The healthy reward per step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The exclusive range of a healthy torso height (<c>healthy_z_range</c>); <see langword="null"/> is (1.0, 2.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public HumanoidEnvV4(double forwardRewardWeight = 1.25, double ctrlCostWeight = 0.1, double healthyReward = 5.0,
                             bool terminateWhenUnhealthy = true, (double Low, double High)? healthyZRange = null, double resetNoiseScale = 1e-2,
                             bool excludeCurrentPositionsFromObservation = true, string? renderMode = null, int width = DefaultSize,
                             int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                             IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : base("humanoid.xml", 5, UnboundedBox(excludeCurrentPositionsFromObservation ? 376 : 378), ClassMetadata, renderMode, width,
                   height, cameraId, cameraName, DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            HealthyRewardWeight = healthyReward;
            TerminateWhenUnhealthy = terminateWhenUnhealthy;
            HealthyZRange = healthyZRange ?? (1.0, 2.0);
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="HumanoidEnvV4(double, double, double, bool, Nullable{ValueTuple{double,double}}, double, bool, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new HumanoidEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new HumanoidEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new HumanoidEnvV4()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared controls (<c>ctrl_cost_weight</c>).</param>
        /// <param name="healthyReward">The healthy reward per step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The exclusive range of a healthy torso height (<c>healthy_z_range</c>); <see langword="null"/> is (1.0, 2.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="Gym.Envs.RenderMode"/> does not declare (a cast integer such as <c>(RenderMode)99</c>).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public HumanoidEnvV4(RenderMode renderMode, double forwardRewardWeight = 1.25, double ctrlCostWeight = 0.1, double healthyReward = 5.0,
                             bool terminateWhenUnhealthy = true, (double Low, double High)? healthyZRange = null, double resetNoiseScale = 1e-2,
                             bool excludeCurrentPositionsFromObservation = true, int width = DefaultSize, int height = DefaultSize,
                             int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                             IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : this(forwardRewardWeight, ctrlCostWeight, healthyReward, terminateWhenUnhealthy, healthyZRange, resetNoiseScale,
                   excludeCurrentPositionsFromObservation, RenderModes.Name(renderMode), width, height, cameraId, cameraName, maxGeom, visualOptions,
                   viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 67 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(67);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, as <see cref="HumanoidEnvV5"/>'s. A new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 1L, ["distance"] = 4.0, ["lookat"] = new[] { 0.0, 0.0, 2.0 }, ["elevation"] = -20.0,
        };

        /// <summary>The weight of the forward velocity (<c>_forward_reward_weight</c>), read every step.</summary>
        [PyName("_forward_reward_weight")]
        public double ForwardRewardWeight { get; set; }

        /// <summary>The weight of the squared controls (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The healthy reward per step (<c>_healthy_reward</c>), read every step.</summary>
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
        ///     changing it after construction changes the observation's length but not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>Whether the humanoid is healthy now (<c>is_healthy</c>): the torso's height inside the exclusive z range.</summary>
        [PyName("is_healthy")]
        public bool IsHealthy => HealthyZRange.Low < Data.qpos[2] && Data.qpos[2] < HealthyZRange.High;

        /// <summary>
        ///     The healthy reward now (<c>healthy_reward</c>): <c>float(is_healthy or terminate_when_unhealthy) *
        ///     _healthy_reward</c>, so paid on every step while unhealthy states end the episode.
        /// </summary>
        [PyName("healthy_reward")]
        public double HealthyReward => ((IsHealthy || TerminateWhenUnhealthy) ? 1.0 : 0.0) * HealthyRewardWeight;

        /// <summary>Whether the episode ended (<c>terminated</c>): unhealthy while <c>terminate_when_unhealthy</c> is set.</summary>
        [PyName("terminated")]
        public bool Terminated => TerminateWhenUnhealthy && !IsHealthy;

        /// <summary>
        ///     The center of mass's x and y (<c>mass_center</c>), as <see cref="HumanoidEnvV5.MassCenter"/> computes it. A new
        ///     <c>(2,)</c> float64 array the caller owns.
        /// </summary>
        /// <param name="model">The model.</param>
        /// <param name="data">The simulation.</param>
        /// <returns>The <c>(2,)</c> array.</returns>
        /// <exception cref="ArgumentNullException">A null model or data.</exception>
        /// <remarks>
        ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/humanoid_v4.py#L15">gymnasium/envs/mujoco/humanoid_v4.py</see>.
        /// </remarks>
        [PyName("mass_center")]
        public static NDArray MassCenter(MjModel model, MjData data) => HumanoidEnvV5.MassCenter(model, data);

        /// <summary>
        ///     The control cost (<c>control_cost</c>): <c>ctrl_cost_weight * sum(ctrl**2)</c> of the simulation's controls,
        ///     float64; <paramref name="action"/> is taken but not read, as in Gymnasium.
        /// </summary>
        /// <param name="action">The action; ignored.</param>
        /// <returns>The cost.</returns>
        [PyName("control_cost")]
        public double ControlCost(NDArray? action) => CtrlCostWeight * MujocoMath.SumOfSquares(Data.ctrl);

        /// <summary>Applies <paramref name="action"/> for 5 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 17 controls, any real dtype (stored as float64).</param>
        /// <returns>The observation, the float64 reward, <see cref="Terminated"/>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(17,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            (double xBefore, double yBefore) = MujocoMath.MassCenter(Model, Data);
            DoSimulation(action, FrameSkip);
            (double xAfter, double yAfter) = MujocoMath.MassCenter(Model, Data);
            double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;

            NpScalar ctrlCost = NpScalar.PyFloat(CtrlCostWeight) * NpScalar.Float64(MujocoMath.SumOfSquares(Data.ctrl));
            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NpScalar healthyReward = NpScalar.PyFloat(HealthyReward);
            NpScalar rewards = forwardReward + healthyReward;
            NDArray observation = GetObs();
            NpScalar reward = rewards - ctrlCost;
            bool terminated = Terminated;
            var info = new Info {
                ["reward_linvel"] = MujocoMath.InfoValue(forwardReward),
                ["reward_quadctrl"] = MujocoMath.InfoValue(-ctrlCost),
                ["reward_alive"] = MujocoMath.InfoValue(healthyReward),
                ["x_position"] = MujocoMath.Float64(xAfter),
                ["y_position"] = MujocoMath.Float64(yAfter),
                ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(xAfter, yAfter)),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["y_velocity"] = MujocoMath.Float64(yVelocity),
                ["forward_reward"] = MujocoMath.InfoValue(forwardReward),
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
        ///     The observation (<c>_get_obs</c>): the positions (the root's x and y left out unless asked for), the velocities,
        ///     then the whole <c>cinert</c>, <c>cvel</c>, <c>qfrc_actuator</c> and <c>cfrc_ext</c> arrays.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 2 : 0;
            var values = new List<double>(Model.Nq + 2 * Model.Nv + Data.cinert.Length + Data.cvel.Length + Data.cfrc_ext.Length);
            values.AddRange(new ArraySegment<double>(Data.qpos, skipped, Model.Nq - skipped));
            values.AddRange(Data.qvel);
            values.AddRange(Data.cinert);
            values.AddRange(Data.cvel);
            values.AddRange(Data.qfrc_actuator);
            values.AddRange(Data.cfrc_ext);
            return np.array(values.ToArray());
        }
    }
}
