using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     HalfCheetah-v4: the planar cheetah of <see cref="HalfCheetahEnvV5"/> with v4's fixed model and frame skip, the same
    ///     dynamics and reward, an info of <c>x_position</c>, <c>x_velocity</c>, <c>reward_run</c> and <c>reward_ctrl</c>,
    ///     and an empty reset info. Byte for byte what Gymnasium's HalfCheetah-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The observation space is 17 (18 with the root's x) float64 values whatever the model, as v4 declares it.
    ///         <c>make</c> warns that the id is out of date (v5 exists), as Gymnasium does.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/half_cheetah_v4.py#L14">gymnasium/envs/mujoco/half_cheetah_v4.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>HalfCheetahInfoV4 info = step.Info;</c> (<see cref="HalfCheetahInfoV4"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("HalfCheetahEnv")]
    public partial class HalfCheetahEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>HalfCheetahEnv.__init__</c>): <c>half_cheetah.xml</c> with a frame skip of 5.</summary>
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
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public HalfCheetahEnvV4(double forwardRewardWeight = 1.0, double ctrlCostWeight = 0.1, double resetNoiseScale = 0.1,
                                bool excludeCurrentPositionsFromObservation = true, string? renderMode = null, int width = DefaultSize,
                                int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                                IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : base("half_cheetah.xml", 5, UnboundedBox(excludeCurrentPositionsFromObservation ? 17 : 18), ClassMetadata, renderMode,
                   width, height, cameraId, cameraName, DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="HalfCheetahEnvV4(double, double, double, bool, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new HalfCheetahEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new HalfCheetahEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new HalfCheetahEnvV4()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
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
        public HalfCheetahEnvV4(RenderMode renderMode, double forwardRewardWeight = 1.0, double ctrlCostWeight = 0.1, double resetNoiseScale = 0.1,
                                bool excludeCurrentPositionsFromObservation = true, int width = DefaultSize, int height = DefaultSize,
                                int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                                IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : this(forwardRewardWeight, ctrlCostWeight, resetNoiseScale, excludeCurrentPositionsFromObservation, RenderModes.Name(renderMode), width,
                   height, cameraId, cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 20 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(20);

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
        ///     The control cost of <paramref name="action"/> (<c>control_cost</c>): <c>ctrl_cost_weight * sum(action**2)</c>,
        ///     computed in the action's dtype and returned widened exactly.
        /// </summary>
        /// <param name="action">The action.</param>
        /// <returns>The cost.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        [PyName("control_cost")]
        public double ControlCost(NDArray action) => MujocoMath.SumOfSquares(action).Times(CtrlCostWeight).Value;

        /// <summary>
        ///     Applies <paramref name="action"/> for 5 physics steps and reports the transition (<c>step</c>).
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

            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NDArray observation = GetObs();
            NpScalar reward = forwardReward - ctrlCost;
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(xPositionAfter),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["reward_run"] = MujocoMath.InfoValue(forwardReward),
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
