using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     HalfCheetah-v5: a planar cheetah (a torso and two legs of thigh, shin and foot) driven by 6 joint torques in
    ///     [-1, 1]. The reward pays the forward velocity and charges the squared torques; the episode never ends on its own
    ///     (<c>make</c>'s 1000-step limit truncates it). Byte for byte what Gymnasium's HalfCheetah-v5 computes on MuJoCo
    ///     4.0.0 (DR-61, DR-64): observations, rewards, infos and the generator's draws.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the joint positions without the root's x (unless
    ///         <c>exclude_current_positions_from_observation=False</c>), then all joint velocities, 17 values by default.
    ///         Reward: <c>forward_reward_weight * x_velocity - ctrl_cost_weight * sum(action**2)</c>, the control cost in the
    ///         action's dtype (float32 for the action space's samples, so the info's <c>reward_ctrl</c> is an <c>np.float32</c>
    ///         0-d array there). Info: <c>x_position</c>, <c>x_velocity</c>, <c>reward_forward</c>, <c>reward_ctrl</c>;
    ///         the reset's info holds <c>x_position</c>.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/half_cheetah_v5.py#L14">gymnasium/envs/mujoco/half_cheetah_v5.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("HalfCheetahEnv")]
    public partial class HalfCheetahEnvV5 : MujocoEnv {
        /// <summary>
        ///     Creates the env (Gymnasium's <c>HalfCheetahEnv.__init__</c>): the model, its spaces and the frame rate the time
        ///     step gives.
        /// </summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default,
        ///     <c>{"distance": 4.0}</c>, and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
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
        /// <exception cref="OverflowError">A frame skip of 0 (Python's <c>OverflowError</c> deriving the frame rate).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public HalfCheetahEnvV5(string xmlFile = "half_cheetah.xml", int frameSkip = 5,
                                IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double forwardRewardWeight = 1.0,
                                double ctrlCostWeight = 0.1, double resetNoiseScale = 0.1, bool excludeCurrentPositionsFromObservation = true,
                                string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base(xmlFile, frameSkip, null, ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
            // the v5 envs derive their frame rate from the time step after the base constructor
            Metadata = MetadataFromDt();
            int skipped = excludeCurrentPositionsFromObservation ? 1 : 0;
            ObservationSpace = UnboundedBox(Model.Nq + Model.Nv - skipped);
            ObservationStructure = new Info {
                ["skipped_qpos"] = (long)skipped,
                ["qpos"] = (long)(Model.Nq - skipped),
                ["qvel"] = (long)Model.Nv,
            };
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: the four render modes, without a frame rate (each instance derives
        ///     its own from the time step). A new dict on every read.
        /// </summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, <c>{"distance": 4.0}</c>: a new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> { ["distance"] = 4.0 };

        /// <summary>The weight of the forward velocity (<c>_forward_reward_weight</c>), read every step.</summary>
        [PyName("_forward_reward_weight")]
        public double ForwardRewardWeight { get; set; }

        /// <summary>The weight of the squared torques (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>
        ///     Whether the root's x leaves the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
        ///     changing it after construction changes the observation's length but not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>
        ///     How the observation is laid out (<c>observation_structure</c>): <c>skipped_qpos</c>, <c>qpos</c> and <c>qvel</c>
        ///     counts, as Python ints.
        /// </summary>
        public Info ObservationStructure { get; }

        /// <summary>
        ///     The control cost of <paramref name="action"/> (<c>control_cost</c>): <c>ctrl_cost_weight * sum(action**2)</c>,
        ///     computed in the action's dtype and returned widened exactly (Gymnasium returns an <c>np.float32</c> for a
        ///     float32 action).
        /// </summary>
        /// <param name="action">The action.</param>
        /// <returns>The cost.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        [PyName("control_cost")]
        public double ControlCost(NDArray action) => MujocoMath.SumOfSquares(action).Times(CtrlCostWeight).Value;

        /// <summary>
        ///     Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition
        ///     (<c>step</c>).
        /// </summary>
        /// <param name="action">The 6 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
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

            NDArray observation = GetObs();
            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NpScalar reward = forwardReward - ctrlCost;
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(xPositionAfter),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["reward_forward"] = MujocoMath.InfoValue(forwardReward),
                ["reward_ctrl"] = MujocoMath.InfoValue(-ctrlCost),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, false, false, info);
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

        /// <summary>The reset's info (<c>_get_reset_info</c>): <c>x_position</c>, an <c>np.float64</c>.</summary>
        /// <returns>The info.</returns>
        protected override Info GetResetInfo() => new() { ["x_position"] = MujocoMath.Float64(Data.qpos[0]) };

        /// <summary>The observation (<c>_get_obs</c>): the positions (the root's x left out unless asked for), then the velocities.</summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 1 : 0;
            int positions = Model.Nq - skipped;
            var values = new double[positions + Model.Nv];
            Array.Copy(Data.qpos, skipped, values, 0, positions);
            Array.Copy(Data.qvel, 0, values, positions, Model.Nv);
            return np.array(values);
        }
    }
}
