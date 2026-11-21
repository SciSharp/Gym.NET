using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Pusher-v5: a 7-joint arm that pushes a cylinder to a goal on a table, driven by 7 joint torques in [-2, 2]. The
    ///     reward charges the cylinder's distance to the goal, half the fingertip's distance to the cylinder and the squared
    ///     torques; the episode never ends on its own (<c>make</c>'s 100-step limit truncates it). Byte for byte what
    ///     Gymnasium's Pusher-v5 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64, 23 values): the arm's 7 joint angles and velocities, then the positions of the
    ///         fingertip, the cylinder and the goal. Reward: <c>-|object - goal| * reward_dist_weight - sum(action**2) *
    ///         reward_control_weight - |object - tips_arm| * reward_near_weight</c>, the control term in the action's dtype.
    ///         Info: <c>reward_dist</c>, <c>reward_ctrl</c>, <c>reward_near</c>; the reset's info is empty.
    ///     </para>
    ///     <para>
    ///         Footgun, as in Gymnasium: the reset writes the cylinder's and the goal's start into <see cref="MujocoEnv.InitQpos"/>
    ///         itself (Gymnasium's <c>qpos = self.init_qpos</c> is no copy), so <c>init_qpos</c>'s last four values change on
    ///         every reset. The dynamics are unaffected: those four are overwritten before they are used.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/pusher_v5.py#L15">gymnasium/envs/mujoco/pusher_v5.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>PusherInfoV5 info = step.Info;</c> (<see cref="PusherInfoV5"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("PusherEnv")]
    public partial class PusherEnvV5 : MujocoEnv {
        /// <summary>The cylinder's start drawn by the last reset (<c>cylinder_pos</c>), or <see langword="null"/> before it.</summary>
        private NDArray? cylinderPos;

        /// <summary>The goal of the last reset, <c>np.asarray([0, 0])</c> (<c>goal_pos</c>), or <see langword="null"/> before it.</summary>
        private NDArray? goalPos;

        /// <summary>Creates the env (Gymnasium's <c>PusherEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63); <c>"pusher.xml"</c> is v4's.</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default
        ///     (<c>trackbodyid</c> -1, 4 m away), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="rewardNearWeight">The weight of the fingertip's distance to the cylinder (<c>reward_near_weight</c>).</param>
        /// <param name="rewardDistWeight">The weight of the cylinder's distance to the goal (<c>reward_dist_weight</c>; Gymnasium's default is the int 1).</param>
        /// <param name="rewardControlWeight">The weight of the squared torques (<c>reward_control_weight</c>).</param>
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
        public PusherEnvV5(string xmlFile = "pusher_v5.xml", int frameSkip = 5, IReadOnlyDictionary<string, object?>? defaultCameraConfig = null,
                           double rewardNearWeight = 0.5, double rewardDistWeight = 1, double rewardControlWeight = 0.1, string? renderMode = null,
                           int width = DefaultSize, int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                           IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : base(xmlFile, frameSkip, UnboundedBox(23), ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            RewardNearWeight = rewardNearWeight;
            RewardDistWeight = rewardDistWeight;
            RewardControlWeight = rewardControlWeight;
            Metadata = MetadataFromDt();
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="PusherEnvV5(string, int, IReadOnlyDictionary{string,object}, double, double, double, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new PusherEnvV5(RenderMode.RgbArray)</c> builds what
        ///     <c>new PusherEnvV5(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new PusherEnvV5()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63); <c>"pusher.xml"</c> is v4's.</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default
        ///     (<c>trackbodyid</c> -1, 4 m away), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="rewardNearWeight">The weight of the fingertip's distance to the cylinder (<c>reward_near_weight</c>).</param>
        /// <param name="rewardDistWeight">The weight of the cylinder's distance to the goal (<c>reward_dist_weight</c>; Gymnasium's default is the int 1).</param>
        /// <param name="rewardControlWeight">The weight of the squared torques (<c>reward_control_weight</c>).</param>
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
        public PusherEnvV5(RenderMode renderMode, string xmlFile = "pusher_v5.xml", int frameSkip = 5,
                           IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double rewardNearWeight = 0.5,
                           double rewardDistWeight = 1, double rewardControlWeight = 0.1, int width = DefaultSize, int height = DefaultSize,
                           int? cameraId = null, string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                           EnvViewerFactory? viewerFactory = null)
            : this(xmlFile, frameSkip, defaultCameraConfig, rewardNearWeight, rewardDistWeight, rewardControlWeight, RenderModes.Name(renderMode),
                   width, height, cameraId, cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> -1, <c>distance</c> 4.0. A new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = -1L, ["distance"] = 4.0,
        };

        /// <summary>The weight of the fingertip's distance to the cylinder (<c>_reward_near_weight</c>), read every step.</summary>
        [PyName("_reward_near_weight")]
        public double RewardNearWeight { get; set; }

        /// <summary>The weight of the cylinder's distance to the goal (<c>_reward_dist_weight</c>), read every step.</summary>
        [PyName("_reward_dist_weight")]
        public double RewardDistWeight { get; set; }

        /// <summary>The weight of the squared torques (<c>_reward_control_weight</c>), read every step.</summary>
        [PyName("_reward_control_weight")]
        public double RewardControlWeight { get; set; }

        /// <summary>
        ///     The cylinder's start the last reset drew (<c>cylinder_pos</c>), float64 <c>(2,)</c>, or <see langword="null"/>
        ///     before the first reset. Owned by the env, replaced by each reset: do not dispose it.
        /// </summary>
        [NDBorrowed]
        public NDArray? CylinderPos => cylinderPos;

        /// <summary>
        ///     The goal the last reset set (<c>goal_pos</c>), the int64 <c>[0, 0]</c>, or <see langword="null"/> before the first
        ///     reset. Owned by the env, replaced by each reset: do not dispose it.
        /// </summary>
        [NDBorrowed]
        public NDArray? GoalPos => goalPos;

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 7 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(7,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        /// <exception cref="KeyError">The model lacks one of the bodies <c>object</c>, <c>tips_arm</c>, <c>goal</c> (MuJoCo's message).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            DoSimulation(action, FrameSkip);
            NDArray observation = GetObs();

            int obj = Data.Body("object").Id, tips = Data.Body("tips_arm").Id, goal = Data.Body("goal").Id;
            NpScalar rewardNear = NpScalar.Float64(-Distance(obj, tips)) * NpScalar.PyFloat(RewardNearWeight);
            NpScalar rewardDist = NpScalar.Float64(-Distance(obj, goal)) * NpScalar.PyFloat(RewardDistWeight);
            NpScalar rewardCtrl = squares.Negate().Times(RewardControlWeight);
            NpScalar reward = rewardDist + rewardCtrl + rewardNear;
            var info = new Info {
                ["reward_dist"] = MujocoMath.InfoValue(rewardDist),
                ["reward_ctrl"] = MujocoMath.InfoValue(rewardCtrl),
                ["reward_near"] = MujocoMath.InfoValue(rewardNear),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, false, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): the arm at <c>init_qpos</c>; the cylinder drawn as
        ///     <c>[uniform(-0.3, 0), uniform(-0.2, 0.2)]</c> until it is farther than 0.17 from the goal at the origin; both
        ///     written into <c>init_qpos</c>'s last four values (see the class remarks); velocities
        ///     <c>init_qvel + uniform(-0.005, 0.005, nv)</c> with the cylinder's and the goal's zeroed.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            int nq = Model.Nq, nv = Model.Nv;
            ReplaceGoalPos(np.array(new long[] { 0, 0 }));
            double c0, c1;
            while (true) {
                c0 = UniformNoise(-0.3, 0, 1)[0];
                c1 = UniformNoise(-0.2, 0.2, 1)[0];
                // np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17, the goal being the int origin
                if (MujocoMath.Norm(c0 - 0, c1 - 0) > 0.17)
                    break;
            }
            ReplaceCylinderPos(np.array(new[] { c0, c1 }));
            // qpos is init_qpos itself: the writes land in the env's reference positions, as Gymnasium's do
            InitQpos.SetDouble(c0, nq - 4);
            InitQpos.SetDouble(c1, nq - 3);
            InitQpos.SetDouble(0, nq - 2);
            InitQpos.SetDouble(0, nq - 1);
            double[] qpos = InitQpos.ToArray<double>();
            double[] qvel = AddNoise(InitQvel, UniformNoise(-0.005, 0.005, nv));
            for (int i = nv - 4; i < nv; i++)
                qvel[i] = 0;
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>Closes, then releases the cylinder's and the goal's last arrays and everything <see cref="MujocoEnv"/> holds.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs,TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            ReplaceCylinderPos(null);
            ReplaceGoalPos(null);
        }

        /// <summary>The distance between two bodies' frames, <c>np.linalg.norm(get_body_com(a) - get_body_com(b))</c>.</summary>
        /// <param name="a">The first body.</param>
        /// <param name="b">The second body.</param>
        /// <returns>The Euclidean distance.</returns>
        private double Distance(int a, int b) =>
            MujocoMath.Norm(Data.xpos[3 * a] - Data.xpos[3 * b], Data.xpos[3 * a + 1] - Data.xpos[3 * b + 1], Data.xpos[3 * a + 2] - Data.xpos[3 * b + 2]);

        /// <summary>Replaces <see cref="CylinderPos"/>, releasing the array it held.</summary>
        /// <param name="value">The new array, which the env now owns, or <see langword="null"/>.</param>
        private void ReplaceCylinderPos(NDArray? value) {
            cylinderPos?.Dispose();
            cylinderPos = value;
        }

        /// <summary>Replaces <see cref="GoalPos"/>, releasing the array it held.</summary>
        /// <param name="value">The new array, which the env now owns, or <see langword="null"/>.</param>
        private void ReplaceGoalPos(NDArray? value) {
            goalPos?.Dispose();
            goalPos = value;
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the first 7 positions and velocities, then the frames of <c>tips_arm</c>,
        ///     <c>object</c> and <c>goal</c>.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        /// <exception cref="KeyError">The model lacks one of the three bodies.</exception>
        private NDArray GetObs() {
            var values = new double[23];
            Array.Copy(Data.qpos, values, 7);
            Array.Copy(Data.qvel, 0, values, 7, 7);
            int tips = Data.Body("tips_arm").Id, obj = Data.Body("object").Id, goal = Data.Body("goal").Id;
            Array.Copy(Data.xpos, 3 * tips, values, 14, 3);
            Array.Copy(Data.xpos, 3 * obj, values, 17, 3);
            Array.Copy(Data.xpos, 3 * goal, values, 20, 3);
            return np.array(values);
        }
    }
}
