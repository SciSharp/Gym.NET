using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     InvertedPendulum-v5: a pole hinged on a cart that slides on a rail, kept upright by a force on the cart in
    ///     [-3, 3]. Each step pays 1 while the pole stays within 0.2 rad of upright and 0 on the step it falls, which ends
    ///     the episode (as does a non-finite state). Byte for byte what Gymnasium's InvertedPendulum-v5 computes on MuJoCo
    ///     4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the cart's position, the pole's angle, then their velocities. The reward is Gymnasium's
    ///         Python <c>int</c> (1 or 0), so the info's <c>reward_survive</c> is a <see langword="long"/>; the step's
    ///         reward is its float64 value, as every Gym.NET reward. The reset's info is empty.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/inverted_pendulum_v5.py#L15">gymnasium/envs/mujoco/inverted_pendulum_v5.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("InvertedPendulumEnv")]
    public partial class InvertedPendulumEnvV5 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>InvertedPendulumEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default
        ///     (tracking body 0 from 2.04 m), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
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
        /// <exception cref="OverflowException">A frame skip of 0 (Python's <c>OverflowError</c> deriving the frame rate).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public InvertedPendulumEnvV5(string xmlFile = "inverted_pendulum.xml", int frameSkip = 2,
                                     IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double resetNoiseScale = 0.01,
                                     string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                     string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                     IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base(xmlFile, frameSkip, UnboundedBox(4), ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            ResetNoiseScale = resetNoiseScale;
            Metadata = MetadataFromDt();
            ObservationStructure = new Info { ["qpos"] = (long)Model.Nq, ["qvel"] = (long)Model.Nv };
        }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> 0, <c>distance</c> 2.04. A new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 0L, ["distance"] = 2.04,
        };

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>How the observation is laid out (<c>observation_structure</c>): the <c>qpos</c> and <c>qvel</c> counts.</summary>
        public Info ObservationStructure { get; }

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The force on the cart, shape <c>(1,)</c>, any real dtype (stored as float64).</param>
        /// <returns>The observation, the reward 1 or 0, whether the pole fell, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(1,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            DoSimulation(action, FrameSkip);
            double[] values = ObservationValues();
            bool terminated = Fell(values);
            // int(not terminated): a Python int, kept as one in the info
            long reward = terminated ? 0 : 1;
            var info = new Info { ["reward_survive"] = reward };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(np.array(values), reward, terminated, false, info);
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
            return np.array(ObservationValues());
        }

        /// <summary>
        ///     Whether the pole fell: a non-finite observation value, or <c>abs(observation[1]) &gt; 0.2</c>.
        /// </summary>
        /// <param name="observation">The observation's values.</param>
        /// <returns>The termination.</returns>
        private static bool Fell(double[] observation) {
            foreach (double value in observation)
                if (!double.IsFinite(value))
                    return true;
            return Math.Abs(observation[1]) > 0.2;
        }

        /// <summary>The observation's values (<c>_get_obs</c>): the positions, then the velocities.</summary>
        /// <returns>A new array of <c>nq + nv</c> values.</returns>
        private double[] ObservationValues() {
            var values = new double[Model.Nq + Model.Nv];
            Array.Copy(Data.qpos, values, Model.Nq);
            Array.Copy(Data.qvel, 0, values, Model.Nq, Model.Nv);
            return values;
        }
    }
}
