using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Walker2d-v4: the two-legged walker of <see cref="Walker2dEnvV5"/> on v4's model (<c>walker2d.xml</c>, whose feet
    ///     differ in friction) with a frame skip of 4, an info of <c>x_position</c> and <c>x_velocity</c> only, and v4's
    ///     healthy reward, paid on every step while <c>terminate_when_unhealthy</c> is set. Byte for byte what Gymnasium's
    ///     Walker2d-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Reward: <c>forward_reward_weight * x_velocity + float(is_healthy or terminate_when_unhealthy) * healthy_reward -
    ///         ctrl_cost_weight * sum(action**2)</c>. The observation space is 17 (18 with the root's x) float64 values. The
    ///         reset's info is empty; <c>make</c> warns that the id is out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/walker2d_v4.py#L15">gymnasium/envs/mujoco/walker2d_v4.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>Walker2dInfoV4 info = step.Info;</c> (<see cref="Walker2dInfoV4"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("Walker2dEnv")]
    public partial class Walker2dEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>Walker2dEnv.__init__</c>): <c>walker2d.xml</c> with a frame skip of 4.</summary>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="healthyReward">The healthy reward per step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The exclusive range of a healthy height (<c>healthy_z_range</c>); <see langword="null"/> is (0.8, 2.0).</param>
        /// <param name="healthyAngleRange">The exclusive range of a healthy torso angle (<c>healthy_angle_range</c>); <see langword="null"/> is (-1.0, 1.0).</param>
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
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public Walker2dEnvV4(double forwardRewardWeight = 1.0, double ctrlCostWeight = 1e-3, double healthyReward = 1.0,
                             bool terminateWhenUnhealthy = true, (double Low, double High)? healthyZRange = null,
                             (double Low, double High)? healthyAngleRange = null, double resetNoiseScale = 5e-3,
                             bool excludeCurrentPositionsFromObservation = true, string? renderMode = null, int width = DefaultSize,
                             int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                             IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : base("walker2d.xml", 4, UnboundedBox(excludeCurrentPositionsFromObservation ? 17 : 18), ClassMetadata, renderMode, width,
                   height, cameraId, cameraName, DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            HealthyRewardWeight = healthyReward;
            TerminateWhenUnhealthy = terminateWhenUnhealthy;
            HealthyZRange = healthyZRange ?? (0.8, 2.0);
            HealthyAngleRange = healthyAngleRange ?? (-1.0, 1.0);
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="Walker2dEnvV4(double, double, double, bool, Nullable{ValueTuple{double,double}}, Nullable{ValueTuple{double,double}}, double, bool, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new Walker2dEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new Walker2dEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new Walker2dEnvV4()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="healthyReward">The healthy reward per step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The exclusive range of a healthy height (<c>healthy_z_range</c>); <see langword="null"/> is (0.8, 2.0).</param>
        /// <param name="healthyAngleRange">The exclusive range of a healthy torso angle (<c>healthy_angle_range</c>); <see langword="null"/> is (-1.0, 1.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x leaves the observation (<c>exclude_current_positions_from_observation</c>).
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
        public Walker2dEnvV4(RenderMode renderMode, double forwardRewardWeight = 1.0, double ctrlCostWeight = 1e-3, double healthyReward = 1.0,
                             bool terminateWhenUnhealthy = true, (double Low, double High)? healthyZRange = null,
                             (double Low, double High)? healthyAngleRange = null, double resetNoiseScale = 5e-3,
                             bool excludeCurrentPositionsFromObservation = true, int width = DefaultSize, int height = DefaultSize,
                             int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                             IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : this(forwardRewardWeight, ctrlCostWeight, healthyReward, terminateWhenUnhealthy, healthyZRange, healthyAngleRange, resetNoiseScale,
                   excludeCurrentPositionsFromObservation, RenderModes.Name(renderMode), width, height, cameraId, cameraName, maxGeom, visualOptions,
                   viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 125 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(125);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, as <see cref="Walker2dEnvV5"/>'s. A new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 2L, ["distance"] = 4.0, ["lookat"] = new[] { 0.0, 0.0, 1.15 }, ["elevation"] = -20.0,
        };

        /// <summary>The weight of the forward velocity (<c>_forward_reward_weight</c>), read every step.</summary>
        [PyName("_forward_reward_weight")]
        public double ForwardRewardWeight { get; set; }

        /// <summary>The weight of the squared torques (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The healthy reward per step (<c>_healthy_reward</c>), read every step.</summary>
        [PyName("_healthy_reward")]
        public double HealthyRewardWeight { get; set; }

        /// <summary>Whether an unhealthy state ends the episode (<c>_terminate_when_unhealthy</c>), read every step.</summary>
        [PyName("_terminate_when_unhealthy")]
        public bool TerminateWhenUnhealthy { get; set; }

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

        /// <summary>Whether the walker is healthy now (<c>is_healthy</c>): the torso's height and angle inside their exclusive ranges.</summary>
        [PyName("is_healthy")]
        public bool IsHealthy {
            get {
                double z = Data.qpos[1], angle = Data.qpos[2];
                bool healthyZ = HealthyZRange.Low < z && z < HealthyZRange.High;
                bool healthyAngle = HealthyAngleRange.Low < angle && angle < HealthyAngleRange.High;
                return healthyZ && healthyAngle;
            }
        }

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
        ///     The control cost of <paramref name="action"/> (<c>control_cost</c>): <c>ctrl_cost_weight * sum(action**2)</c>,
        ///     computed in the action's dtype and returned widened exactly.
        /// </summary>
        /// <param name="action">The action.</param>
        /// <returns>The cost.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        [PyName("control_cost")]
        public double ControlCost(NDArray action) => MujocoMath.SumOfSquares(action).Times(CtrlCostWeight).Value;

        /// <summary>Applies <paramref name="action"/> for 4 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 6 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <see cref="Terminated"/>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(6,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            double xPositionBefore = Data.qpos[0];
            DoSimulation(action, FrameSkip);
            double xPositionAfter = Data.qpos[0];
            double xVelocity = (xPositionAfter - xPositionBefore) / Dt;

            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NpScalar rewards = forwardReward + NpScalar.PyFloat(HealthyReward);
            NDArray observation = GetObs();
            NpScalar reward = rewards - ctrlCost;
            bool terminated = Terminated;
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(xPositionAfter),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
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
