using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Hopper-v5: a planar one-legged hopper (torso, thigh, leg, foot) driven by 3 joint torques in [-1, 1]. The reward
    ///     pays the forward velocity and staying healthy, and charges the squared torques; an unhealthy state (too low,
    ///     too tilted, or any coordinate beyond ±100) ends the episode. Byte for byte what Gymnasium's Hopper-v5 computes on
    ///     MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the positions without the root's x (unless asked for), then the velocities clipped to
    ///         ±10, 11 values by default. Reward: <c>forward_reward_weight * x_velocity + healthy_reward * is_healthy -
    ///         ctrl_cost_weight * sum(action**2)</c>; v5 pays the healthy reward only while healthy (a Python <c>float</c>
    ///         in the info, as Gymnasium's <c>bool * float</c> gives). Info: <c>x_position</c>, <c>z_distance_from_origin</c>,
    ///         <c>x_velocity</c>, <c>reward_forward</c>, <c>reward_ctrl</c>, <c>reward_survive</c>; the reset's info holds the
    ///         first two.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/hopper_v5.py#L17">gymnasium/envs/mujoco/hopper_v5.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("HopperEnv")]
    public partial class HopperEnvV5 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>HopperEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default (tracking
        ///     body 2 from 3 m, looking at (0, 0, 1.15), 20° from above), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="healthyReward">The reward for each healthy step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyStateRange">
        ///     The exclusive range every coordinate but the root's x and z must stay in (<c>healthy_state_range</c>);
        ///     <see langword="null"/> is Gymnasium's (-100, 100).
        /// </param>
        /// <param name="healthyZRange">The exclusive range of a healthy height (<c>healthy_z_range</c>); <see langword="null"/> is (0.7, inf).</param>
        /// <param name="healthyAngleRange">The exclusive range of a healthy torso angle (<c>healthy_angle_range</c>); <see langword="null"/> is (-0.2, 0.2).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x leaves the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="NotSupportedException"><paramref name="xmlFile"/> is not a bundled model.</exception>
        /// <exception cref="OverflowException">A frame skip of 0 (Python's <c>OverflowError</c> deriving the frame rate).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        /// <remarks>
        ///     The ranges are tuples in C#, so Python's <c>None</c> (which Gymnasium fails on when it unpacks it) has no
        ///     counterpart: <see langword="null"/> stands for the default.
        /// </remarks>
        public HopperEnvV5(string xmlFile = "hopper.xml", int frameSkip = 4, IReadOnlyDictionary<string, object?>? defaultCameraConfig = null,
                           double forwardRewardWeight = 1.0, double ctrlCostWeight = 1e-3, double healthyReward = 1.0,
                           bool terminateWhenUnhealthy = true, (double Low, double High)? healthyStateRange = null,
                           (double Low, double High)? healthyZRange = null, (double Low, double High)? healthyAngleRange = null,
                           double resetNoiseScale = 5e-3, bool excludeCurrentPositionsFromObservation = true, string? renderMode = null,
                           int width = DefaultSize, int height = DefaultSize, int? cameraId = null, string? cameraName = null,
                           int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                           IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base(xmlFile, frameSkip, null, ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            HealthyRewardWeight = healthyReward;
            TerminateWhenUnhealthy = terminateWhenUnhealthy;
            HealthyStateRange = healthyStateRange ?? (-100.0, 100.0);
            HealthyZRange = healthyZRange ?? (0.7, double.PositiveInfinity);
            HealthyAngleRange = healthyAngleRange ?? (-0.2, 0.2);
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
            Metadata = MetadataFromDt();
            int skipped = excludeCurrentPositionsFromObservation ? 1 : 0;
            ObservationSpace = UnboundedBox(Model.Nq + Model.Nv - skipped);
            ObservationStructure = new Info {
                ["skipped_qpos"] = (long)skipped,
                ["qpos"] = (long)(Model.Nq - skipped),
                ["qvel"] = (long)Model.Nv,
            };
        }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>
        ///     Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> 2, <c>distance</c> 3.0, <c>lookat</c> (0, 0, 1.15),
        ///     <c>elevation</c> -20.0. A new dictionary on every read.
        /// </summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 2L, ["distance"] = 3.0, ["lookat"] = new[] { 0.0, 0.0, 1.15 }, ["elevation"] = -20.0,
        };

        /// <summary>The weight of the forward velocity (<c>_forward_reward_weight</c>), read every step.</summary>
        [PyName("_forward_reward_weight")]
        public double ForwardRewardWeight { get; set; }

        /// <summary>The weight of the squared torques (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The reward for each healthy step (<c>_healthy_reward</c>), read every step.</summary>
        [PyName("_healthy_reward")]
        public double HealthyRewardWeight { get; set; }

        /// <summary>Whether an unhealthy state ends the episode (<c>_terminate_when_unhealthy</c>), read every step.</summary>
        [PyName("_terminate_when_unhealthy")]
        public bool TerminateWhenUnhealthy { get; set; }

        /// <summary>The exclusive range of every coordinate but the root's x and z (<c>_healthy_state_range</c>).</summary>
        [PyName("_healthy_state_range")]
        public (double Low, double High) HealthyStateRange { get; set; }

        /// <summary>The exclusive range of a healthy height (<c>_healthy_z_range</c>).</summary>
        [PyName("_healthy_z_range")]
        public (double Low, double High) HealthyZRange { get; set; }

        /// <summary>The exclusive range of a healthy torso angle (<c>_healthy_angle_range</c>).</summary>
        [PyName("_healthy_angle_range")]
        public (double Low, double High) HealthyAngleRange { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>
        ///     Whether the root's x leaves the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
        ///     changing it after construction changes the observation's length but not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>How the observation is laid out (<c>observation_structure</c>): <c>skipped_qpos</c>, <c>qpos</c> and <c>qvel</c> counts.</summary>
        public Info ObservationStructure { get; }

        /// <summary>
        ///     Whether the hopper is healthy now (<c>is_healthy</c>): every coordinate but the root's x and z inside the state
        ///     range, the height inside the z range and the torso angle inside the angle range, all exclusive (a NaN is unhealthy).
        /// </summary>
        [PyName("is_healthy")]
        public bool IsHealthy {
            get {
                (double minState, double maxState) = HealthyStateRange;
                // state_vector()[2:]: the positions from the torso angle on, then every velocity
                bool healthyState = true;
                for (int i = 2; i < Model.Nq && healthyState; i++)
                    healthyState = minState < Data.qpos[i] && Data.qpos[i] < maxState;
                for (int i = 0; i < Model.Nv && healthyState; i++)
                    healthyState = minState < Data.qvel[i] && Data.qvel[i] < maxState;
                double z = Data.qpos[1], angle = Data.qpos[2];
                bool healthyZ = HealthyZRange.Low < z && z < HealthyZRange.High;
                bool healthyAngle = HealthyAngleRange.Low < angle && angle < HealthyAngleRange.High;
                return healthyState && healthyZ && healthyAngle;
            }
        }

        /// <summary>The healthy reward now (<c>healthy_reward</c>): <c>is_healthy * _healthy_reward</c>, a Python float.</summary>
        [PyName("healthy_reward")]
        public double HealthyReward => HealthyRewardScalar().Value;

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
        /// <param name="action">The 3 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, whether an unhealthy state ended the episode, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(3,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            double xPositionBefore = Data.qpos[0];
            DoSimulation(action, FrameSkip);
            double xPositionAfter = Data.qpos[0];
            double xVelocity = (xPositionAfter - xPositionBefore) / Dt;

            NDArray observation = GetObs();
            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NpScalar healthyReward = HealthyRewardScalar();
            NpScalar rewards = forwardReward + healthyReward;
            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NpScalar reward = rewards - ctrlCost;
            bool terminated = !IsHealthy && TerminateWhenUnhealthy;
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(xPositionAfter),
                ["z_distance_from_origin"] = MujocoMath.Float64(Data.qpos[1] - InitQpos.GetDouble(1)),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["reward_forward"] = MujocoMath.InfoValue(forwardReward),
                ["reward_ctrl"] = MujocoMath.InfoValue(-ctrlCost),
                ["reward_survive"] = MujocoMath.InfoValue(healthyReward),
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

        /// <summary>The reset's info (<c>_get_reset_info</c>): <c>x_position</c> and <c>z_distance_from_origin</c>, <c>np.float64</c>s.</summary>
        /// <returns>The info.</returns>
        protected override Info GetResetInfo() => new() {
            ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
            ["z_distance_from_origin"] = MujocoMath.Float64(Data.qpos[1] - InitQpos.GetDouble(1)),
        };

        /// <summary><c>is_healthy * _healthy_reward</c>: a Python <c>bool</c> times a Python <c>float</c>, so a Python float.</summary>
        /// <returns>The healthy reward and its kind.</returns>
        private NpScalar HealthyRewardScalar() => NpScalar.PyFloat((IsHealthy ? 1.0 : 0.0) * HealthyRewardWeight);

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the positions (the root's x left out unless asked for), then the velocities
        ///     clipped to ±10.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 1 : 0;
            int positions = Model.Nq - skipped;
            var values = new double[positions + Model.Nv];
            Array.Copy(Data.qpos, skipped, values, 0, positions);
            for (int i = 0; i < Model.Nv; i++)
                values[positions + i] = PyMath.Clip(Data.qvel[i], -10, 10);
            return np.array(values);
        }
    }
}
