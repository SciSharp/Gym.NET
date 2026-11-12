using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     HumanoidStandup-v4: the lying humanoid of <see cref="HumanoidStandupEnvV5"/> with v4's fixed model, frame skip and
    ///     weights, every body's rows observed (the world's included) and v4's three-entry info. Byte for byte what
    ///     Gymnasium's HumanoidStandup-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): <c>qpos[2:]</c>, the velocities, and the whole <c>cinert</c>, <c>cvel</c>,
    ///         <c>qfrc_actuator</c> and <c>cfrc_ext</c> arrays, 376 values. Reward: <c>z / timestep - 0.1 * sum(ctrl**2) -
    ///         min(0.5e-6 * sum(cfrc_ext**2), 10) + 1</c>; Python's <c>min</c> returns the int 10 when the impact cost exceeds
    ///         it, so the info's <c>reward_impact</c> is then the Python int -10 (a <see langword="long"/>) instead of an
    ///         <c>np.float64</c>, as in Gymnasium. Info: <c>reward_linup</c>, <c>reward_quadctrl</c>, <c>reward_impact</c>;
    ///         the reset's info is empty; <c>make</c> warns that the id is out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/humanoidstandup_v4.py#L15">gymnasium/envs/mujoco/humanoidstandup_v4.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("HumanoidStandupEnv")]
    public partial class HumanoidStandupEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>HumanoidStandupEnv.__init__</c>): <c>humanoidstandup.xml</c> with a frame skip of 5.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public HumanoidStandupEnvV4(string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                    string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                    IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base("humanoidstandup.xml", 5, UnboundedBox(376), ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 67 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(67);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, as <see cref="HumanoidStandupEnvV5"/>'s. A new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 1L, ["distance"] = 4.0, ["lookat"] = new[] { 0.0, 0.0, 0.8925 }, ["elevation"] = -20.0,
        };

        /// <summary>Applies <paramref name="action"/> (Gymnasium's <c>a</c>) for 5 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 17 controls, any real dtype (stored as float64).</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(17,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            DoSimulation(action, FrameSkip);
            double posAfter = Data.qpos[2];
            NpScalar uphCost = NpScalar.Float64((posAfter - 0) / Model.Opt.Timestep);

            NpScalar quadCtrlCost = NpScalar.PyFloat(0.1) * NpScalar.Float64(MujocoMath.SumOfSquares(Data.ctrl));
            NpScalar quadImpactCost = NpScalar.PyFloat(0.5e-6) * NpScalar.Float64(MujocoMath.SumOfSquares(Data.cfrc_ext));
            // Python's min(cost, 10): the cost unless 10 is smaller, in which case the int 10 itself (NaN stays the cost)
            quadImpactCost = NpScalar.PyMin(quadImpactCost, NpScalar.PyInt(10));
            NpScalar reward = uphCost - quadCtrlCost - quadImpactCost + NpScalar.PyInt(1);
            RenderIfHuman();
            var info = new Info {
                ["reward_linup"] = MujocoMath.InfoValue(uphCost),
                ["reward_quadctrl"] = MujocoMath.InfoValue(-quadCtrlCost),
                ["reward_impact"] = MujocoMath.InfoValue(-quadImpactCost),
            };
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(GetObs(), reward.Value, false, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-0.01, 0.01, nq)</c>, then velocities
        ///     <c>init_qvel + uniform(-0.01, 0.01, nv)</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            const double c = 0.01;
            double[] qpos = AddNoise(InitQpos, UniformNoise(-c, c, Model.Nq));
            double[] qvel = AddNoise(InitQvel, UniformNoise(-c, c, Model.Nv));
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): <c>qpos[2:]</c>, the velocities, then the whole <c>cinert</c>, <c>cvel</c>,
        ///     <c>qfrc_actuator</c> and <c>cfrc_ext</c> arrays.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            var values = new List<double>(Model.Nq + 2 * Model.Nv + Data.cinert.Length + Data.cvel.Length + Data.cfrc_ext.Length);
            values.AddRange(new ArraySegment<double>(Data.qpos, 2, Model.Nq - 2));
            values.AddRange(Data.qvel);
            values.AddRange(Data.cinert);
            values.AddRange(Data.cvel);
            values.AddRange(Data.qfrc_actuator);
            values.AddRange(Data.cfrc_ext);
            return np.array(values.ToArray());
        }
    }
}
