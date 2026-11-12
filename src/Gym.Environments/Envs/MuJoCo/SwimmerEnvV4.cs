using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Swimmer-v4: the three-link swimmer of <see cref="SwimmerEnvV5"/> with v4's fixed model and frame skip, the same
    ///     dynamics and reward, v4's info (<c>reward_fwd</c>, <c>reward_ctrl</c>, <c>x_position</c>, <c>y_position</c>,
    ///     <c>distance_from_origin</c>, <c>x_velocity</c>, <c>y_velocity</c>, <c>forward_reward</c>) and an empty reset info.
    ///     Byte for byte what Gymnasium's Swimmer-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         The observation space is 8 (10 with the root's x and y) float64 values. Unlike the other v4 envs it takes a
    ///         <c>default_camera_config</c>, as Gymnasium's passes its keyword arguments on. <c>make</c> warns that the id is
    ///         out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/swimmer_v4.py#L10">gymnasium/envs/mujoco/swimmer_v4.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("SwimmerEnv")]
    public partial class SwimmerEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>SwimmerEnv.__init__</c>): <c>swimmer.xml</c> with a frame skip of 4.</summary>
        /// <param name="forwardRewardWeight">The weight of the forward velocity (<c>forward_reward_weight</c>).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="defaultCameraConfig">The free camera's settings (<c>default_camera_config</c>); none by default.</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public SwimmerEnvV4(double forwardRewardWeight = 1.0, double ctrlCostWeight = 1e-4, double resetNoiseScale = 0.1,
                            bool excludeCurrentPositionsFromObservation = true, string? renderMode = null, int width = DefaultSize,
                            int height = DefaultSize, int? cameraId = null, string? cameraName = null,
                            IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, int maxGeom = 1000,
                            IReadOnlyDictionary<int, bool>? visualOptions = null, IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base("swimmer.xml", 4, UnboundedBox(excludeCurrentPositionsFromObservation ? 8 : 10), ClassMetadata, renderMode, width, height,
                   cameraId, cameraName, defaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ForwardRewardWeight = forwardRewardWeight;
            CtrlCostWeight = ctrlCostWeight;
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
        }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 25 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(25);

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
        ///     Whether the root's x and y leave the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
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

        /// <summary>Applies <paramref name="action"/> for 4 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 2 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(2,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            double xBefore = Data.qpos[0], yBefore = Data.qpos[1];
            DoSimulation(action, FrameSkip);
            double xAfter = Data.qpos[0], yAfter = Data.qpos[1];
            double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;

            NpScalar forwardReward = NpScalar.PyFloat(ForwardRewardWeight) * NpScalar.Float64(xVelocity);
            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NDArray observation = GetObs();
            NpScalar reward = forwardReward - ctrlCost;
            var info = new Info {
                ["reward_fwd"] = MujocoMath.InfoValue(forwardReward),
                ["reward_ctrl"] = MujocoMath.InfoValue(-ctrlCost),
                ["x_position"] = MujocoMath.Float64(xAfter),
                ["y_position"] = MujocoMath.Float64(yAfter),
                ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(xAfter, yAfter)),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["y_velocity"] = MujocoMath.Float64(yVelocity),
                ["forward_reward"] = MujocoMath.InfoValue(forwardReward),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, false, false, info);
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

        /// <summary>The observation (<c>_get_obs</c>): the positions (the root's x and y left out unless asked for), then the velocities.</summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 2 : 0;
            int positions = Model.Nq - skipped;
            var values = new double[positions + Model.Nv];
            Array.Copy(Data.qpos, skipped, values, 0, positions);
            Array.Copy(Data.qvel, 0, values, positions, Model.Nv);
            return np.array(values);
        }
    }
}
