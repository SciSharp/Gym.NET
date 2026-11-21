using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     InvertedDoublePendulum-v5: two poles chained on a cart, kept upright by a force on the cart in [-1, 1]. Each step
    ///     pays the healthy reward while the tip stays above 1 m and charges its distance from the top position and the
    ///     poles' angular velocities; the tip falling to 1 m ends the episode. Byte for byte what Gymnasium's
    ///     InvertedDoublePendulum-v5 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64, 9 values): the cart's position, the sines and cosines of the two angles, the velocities
    ///         clipped to ±10, and the cart's constraint force clipped to ±10. Reward: <c>healthy_reward * int(not terminated) -
    ///         (0.01 * x**2 + (y - 2)**2) - (1e-3 * v1**2 + 5e-3 * v2**2)</c> with the tip at (x, y) and the angular velocities
    ///         v1, v2, each <c>**2</c> NumPy's (the C runtime's <c>pow</c>, not a product). Info: <c>reward_survive</c> (a
    ///         Python float), <c>distance_penalty</c>, <c>velocity_penalty</c>; the reset's info is empty. Its default camera
    ///         configuration is none (Gymnasium's v5 passes <c>None</c>).
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/inverted_double_pendulum_v5.py#L16">gymnasium/envs/mujoco/inverted_double_pendulum_v5.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>InvertedDoublePendulumInfoV5 info = step.Info;</c> (<see cref="InvertedDoublePendulumInfoV5"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("InvertedDoublePendulumEnv")]
    public partial class InvertedDoublePendulumEnvV5 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>InvertedDoublePendulumEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">The free camera's settings (<c>default_camera_config</c>); none by default.</param>
        /// <param name="healthyReward">The reward for each step the tip stays up (<c>healthy_reward</c>).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
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
        public InvertedDoublePendulumEnvV5(string xmlFile = "inverted_double_pendulum.xml", int frameSkip = 5,
                                           IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double healthyReward = 10.0,
                                           double resetNoiseScale = 0.1, string? renderMode = null, int width = DefaultSize,
                                           int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                                           IReadOnlyDictionary<int, bool>? visualOptions = null,
                                           EnvViewerFactory? viewerFactory = null)
            : base(xmlFile, frameSkip, UnboundedBox(9), ClassMetadata, renderMode, width, height, cameraId, cameraName, defaultCameraConfig,
                   maxGeom, visualOptions, viewerFactory) {
            HealthyReward = healthyReward;
            ResetNoiseScale = resetNoiseScale;
            Metadata = MetadataFromDt();
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="InvertedDoublePendulumEnvV5(string, int, IReadOnlyDictionary{string,object}, double, double, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new InvertedDoublePendulumEnvV5(RenderMode.RgbArray)</c> builds what
        ///     <c>new InvertedDoublePendulumEnvV5(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new InvertedDoublePendulumEnvV5()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">The free camera's settings (<c>default_camera_config</c>); none by default.</param>
        /// <param name="healthyReward">The reward for each step the tip stays up (<c>healthy_reward</c>).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="Gym.Envs.RenderMode"/> does not declare (a cast integer such as <c>(RenderMode)99</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="xmlFile"/> is not a bundled model.</exception>
        /// <exception cref="OverflowError">A frame skip of 0 (Python's <c>OverflowError</c> deriving the frame rate).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public InvertedDoublePendulumEnvV5(RenderMode renderMode, string xmlFile = "inverted_double_pendulum.xml", int frameSkip = 5,
                                           IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double healthyReward = 10.0,
                                           double resetNoiseScale = 0.1, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                           string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                           EnvViewerFactory? viewerFactory = null)
            : this(xmlFile, frameSkip, defaultCameraConfig, healthyReward, resetNoiseScale, RenderModes.Name(renderMode), width, height, cameraId,
                   cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>The reward for each step the tip stays up (<c>_healthy_reward</c>), read every step.</summary>
        [PyName("_healthy_reward")]
        public double HealthyReward { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The force on the cart, shape <c>(1,)</c>, any real dtype (stored as float64).</param>
        /// <returns>The observation, the float64 reward, whether the tip fell to 1 m, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(1,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            DoSimulation(action, FrameSkip);
            // x, _, y = self.data.site_xpos[0]: the tip site's x and height
            double x = Data.site_xpos[0], y = Data.site_xpos[2];
            NDArray observation = GetObs();
            bool terminated = y <= 1;

            NpScalar v1 = NpScalar.Float64(Data.qvel[1]), v2 = NpScalar.Float64(Data.qvel[2]);
            NpScalar distPenalty = NpScalar.PyFloat(0.01) * NpScalar.Float64(x).NpPow(2) + (NpScalar.Float64(y) - NpScalar.PyInt(2)).NpPow(2);
            NpScalar velPenalty = NpScalar.PyFloat(1e-3) * v1.NpPow(2) + NpScalar.PyFloat(5e-3) * v2.NpPow(2);
            NpScalar aliveBonus = NpScalar.PyFloat(HealthyReward) * NpScalar.PyInt(terminated ? 0 : 1);
            NpScalar reward = aliveBonus - distPenalty - velPenalty;
            var info = new Info {
                ["reward_survive"] = MujocoMath.InfoValue(aliveBonus),
                ["distance_penalty"] = MujocoMath.InfoValue(-distPenalty),
                ["velocity_penalty"] = MujocoMath.InfoValue(-velPenalty),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, terminated, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-scale, scale, nq)</c>, then
        ///     velocities <c>init_qvel + standard_normal(nv) * scale</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            double noiseLow = -ResetNoiseScale, noiseHigh = ResetNoiseScale;
            double[] qpos = AddNoise(InitQpos, UniformNoise(noiseLow, noiseHigh, Model.Nq));
            double[] normal = StandardNormalNoise(Model.Nv);
            for (int i = 0; i < normal.Length; i++)
                normal[i] = normal[i] * ResetNoiseScale;
            double[] qvel = AddNoise(InitQvel, normal);
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the cart's position, the sines then the cosines of the joint angles, the
        ///     velocities clipped to ±10, and the first constraint force clipped to ±10.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int nq = Model.Nq, nv = Model.Nv;
            var values = new double[1 + 2 * (nq - 1) + nv + 1];
            values[0] = Data.qpos[0];
            for (int i = 1; i < nq; i++) {
                values[i] = Math.Sin(Data.qpos[i]);
                values[nq - 1 + i] = Math.Cos(Data.qpos[i]);
            }
            for (int i = 0; i < nv; i++)
                values[2 * nq - 1 + i] = PyMath.Clip(Data.qvel[i], -10, 10);
            // np.clip(qfrc_constraint, -10, 10)[:1]: only the cart's
            values[2 * nq - 1 + nv] = PyMath.Clip(Data.qfrc_constraint[0], -10, 10);
            return np.array(values);
        }
    }
}
