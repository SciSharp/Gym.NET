using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     InvertedDoublePendulum-v4: the double pendulum of <see cref="InvertedDoublePendulumEnvV5"/> with v4's fixed model,
    ///     frame skip and noise, an alive bonus of 10 on every step (the falling one included), every constraint force in
    ///     the observation and an empty info. Byte for byte what Gymnasium's InvertedDoublePendulum-v4 computes on MuJoCo
    ///     4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64, 11 values): the cart's position, the sines and cosines of the two angles, the velocities
    ///         and the constraint forces, both clipped to ±10. Reward: <c>10 - (0.01 * x**2 + (y - 2)**2) - (1e-3 * v1**2 +
    ///         5e-3 * v2**2)</c>. <c>make</c> warns that the id is out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/inverted_double_pendulum_v4.py#L14">gymnasium/envs/mujoco/inverted_double_pendulum_v4.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("InvertedDoublePendulumEnv")]
    public partial class InvertedDoublePendulumEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>InvertedDoublePendulumEnv.__init__</c>): <c>inverted_double_pendulum.xml</c> with a frame skip of 5.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public InvertedDoublePendulumEnvV4(string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                           string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                           IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base("inverted_double_pendulum.xml", 5, UnboundedBox(11), ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 20 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(20);

        /// <summary>
        ///     Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> 0, <c>distance</c> 4.1225, <c>lookat</c>
        ///     (0, 0, 0.12250000000000005). A new dictionary on every read.
        /// </summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 0L, ["distance"] = 4.1225, ["lookat"] = new[] { 0.0, 0.0, 0.12250000000000005 },
        };

        /// <summary>Applies <paramref name="action"/> for 5 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The force on the cart, shape <c>(1,)</c>, any real dtype (stored as float64).</param>
        /// <returns>The observation, the float64 reward, whether the tip fell to 1 m, <c>truncated=False</c> and an empty info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(1,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            DoSimulation(action, FrameSkip);
            NDArray ob = GetObs();
            double x = Data.site_xpos[0], y = Data.site_xpos[2];
            NpScalar distPenalty = NpScalar.PyFloat(0.01) * NpScalar.Float64(x).NpPow(2) + (NpScalar.Float64(y) - NpScalar.PyInt(2)).NpPow(2);
            NpScalar v1 = NpScalar.Float64(Data.qvel[1]), v2 = NpScalar.Float64(Data.qvel[2]);
            NpScalar velPenalty = NpScalar.PyFloat(1e-3) * v1.NpPow(2) + NpScalar.PyFloat(5e-3) * v2.NpPow(2);
            NpScalar aliveBonus = NpScalar.PyInt(10);
            NpScalar r = aliveBonus - distPenalty - velPenalty;
            bool terminated = y <= 1;
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(ob, r.Value, terminated, false, new Info());
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-0.1, 0.1, nq)</c>, then velocities
        ///     <c>init_qvel + standard_normal(nv) * 0.1</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            double[] qpos = AddNoise(InitQpos, UniformNoise(-0.1, 0.1, Model.Nq));
            double[] normal = StandardNormalNoise(Model.Nv);
            for (int i = 0; i < normal.Length; i++)
                normal[i] = normal[i] * 0.1;
            double[] qvel = AddNoise(InitQvel, normal);
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the cart's position, the sines then the cosines of the joint angles, the
        ///     velocities and every constraint force, both clipped to ±10.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int nq = Model.Nq, nv = Model.Nv;
            var values = new double[1 + 2 * (nq - 1) + 2 * nv];
            values[0] = Data.qpos[0];
            for (int i = 1; i < nq; i++) {
                values[i] = Math.Sin(Data.qpos[i]);
                values[nq - 1 + i] = Math.Cos(Data.qpos[i]);
            }
            for (int i = 0; i < nv; i++) {
                values[2 * nq - 1 + i] = PyMath.Clip(Data.qvel[i], -10, 10);
                values[2 * nq - 1 + nv + i] = PyMath.Clip(Data.qfrc_constraint[i], -10, 10);
            }
            return np.array(values);
        }
    }
}
