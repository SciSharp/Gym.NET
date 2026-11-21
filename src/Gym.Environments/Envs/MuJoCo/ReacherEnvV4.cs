using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Reacher-v4: the two-joint arm of <see cref="ReacherEnvV5"/> with v4's fixed model and frame skip, a reward measured
    ///     before the step (the distance and the action's cost as they stand when the action arrives), unweighted, and the
    ///     fingertip's full 3D offset in the observation. Byte for byte what Gymnasium's Reacher-v4 computes on MuJoCo 4.0.0
    ///     (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64, 11 values): the cosines and sines of the arm angles, the target's coordinates, the arm's
    ///         angular velocities and the fingertip's offset from the target in x, y and z. Reward: <c>-|fingertip - target| -
    ///         sum(action**2)</c> before the physics runs, the control term in the action's own dtype (an integer action's an
    ///         <c>np.int64</c>, as there). Info: <c>reward_dist</c>, <c>reward_ctrl</c>; the reset's info is empty;
    ///         <c>make</c> warns that the id is out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/reacher_v4.py#L10">gymnasium/envs/mujoco/reacher_v4.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>ReacherInfoV4 info = step.Info;</c> (<see cref="ReacherInfoV4"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("ReacherEnv")]
    public partial class ReacherEnvV4 : MujocoEnv {
        /// <summary>The target the last reset drew (<c>goal</c>), or <see langword="null"/> before it.</summary>
        private NDArray? goal;

        /// <summary>Creates the env (Gymnasium's <c>ReacherEnv.__init__</c>): <c>reacher.xml</c> with a frame skip of 2.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public ReacherEnvV4(string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                            string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                            EnvViewerFactory? viewerFactory = null)
            : base("reacher.xml", 2, UnboundedBox(11), ClassMetadata, renderMode, width, height, cameraId, cameraName, DefaultCameraConfig,
                   maxGeom, visualOptions, viewerFactory) { }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="ReacherEnvV4(string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new ReacherEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new ReacherEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new ReacherEnvV4()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
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
        public ReacherEnvV4(RenderMode renderMode, int width = DefaultSize, int height = DefaultSize, int? cameraId = null, string? cameraName = null,
                            int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                            EnvViewerFactory? viewerFactory = null)
            : this(RenderModes.Name(renderMode), width, height, cameraId, cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 50 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(50);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, <c>{"trackbodyid": 0}</c>: a new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> { ["trackbodyid"] = 0L };

        /// <summary>
        ///     The target the last reset drew (<c>goal</c>), float64 <c>(2,)</c>, or <see langword="null"/> before the first
        ///     reset. Owned by the env, replaced by each reset: do not dispose it.
        /// </summary>
        [NDBorrowed]
        public NDArray? Goal => goal;

        /// <summary>
        ///     Prices <paramref name="action"/> in the current state, then applies it for 2 physics steps and reports the
        ///     transition (<c>step</c>; Gymnasium's <c>a</c>).
        /// </summary>
        /// <param name="action">The 2 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(2,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        /// <exception cref="KeyError">The model lacks the body <c>fingertip</c> or <c>target</c> (MuJoCo's message).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            (double dx, double dy, double dz) = FingertipOffset();
            NpScalar rewardDist = NpScalar.Float64(-MujocoMath.Norm(dx, dy, dz));
            SquaredSum rewardCtrl = MujocoMath.SumOfSquares(action).Negate();
            NpScalar reward = rewardDist + rewardCtrl.AsOperand();

            DoSimulation(action, FrameSkip);
            RenderIfHuman();
            NDArray ob = GetObs();
            var info = new Info {
                ["reward_dist"] = MujocoMath.InfoValue(rewardDist),
                ["reward_ctrl"] = rewardCtrl.InfoValue(),
            };
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(ob, reward.Value, false, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>uniform(-0.1, 0.1, nq) + init_qpos</c>; the target drawn
        ///     as <c>uniform(-0.2, 0.2, 2)</c> until its norm is below 0.2, into the last two positions; velocities
        ///     <c>init_qvel + uniform(-0.005, 0.005, nv)</c> with the target's zeroed.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            int nq = Model.Nq, nv = Model.Nv;
            double[] qpos = AddNoise(InitQpos, UniformNoise(-0.1, 0.1, nq));
            double[] drawn;
            while (true) {
                drawn = UniformNoise(-0.2, 0.2, 2);
                if (MujocoMath.Norm(drawn[0], drawn[1]) < 0.2)
                    break;
            }
            goal?.Dispose();
            goal = np.array(drawn);
            qpos[nq - 2] = drawn[0];
            qpos[nq - 1] = drawn[1];
            double[] qvel = AddNoise(InitQvel, UniformNoise(-0.005, 0.005, nv));
            qvel[nv - 2] = 0;
            qvel[nv - 1] = 0;
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>Closes, then releases the last target array and everything <see cref="MujocoEnv"/> holds.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs,TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            goal?.Dispose();
            goal = null;
        }

        /// <summary><c>get_body_com("fingertip") - get_body_com("target")</c>: the fingertip's offset from the target.</summary>
        /// <returns>The three components.</returns>
        /// <exception cref="KeyError">The model lacks one of the two bodies.</exception>
        private (double X, double Y, double Z) FingertipOffset() {
            int fingertip = Data.Body("fingertip").Id, target = Data.Body("target").Id;
            return (Data.xpos[3 * fingertip] - Data.xpos[3 * target], Data.xpos[3 * fingertip + 1] - Data.xpos[3 * target + 1],
                    Data.xpos[3 * fingertip + 2] - Data.xpos[3 * target + 2]);
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): <c>cos</c> and <c>sin</c> of the first two positions, the rest of the
        ///     positions, the first two velocities, and the fingertip's full offset from the target.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        /// <exception cref="KeyError">The model lacks the body <c>fingertip</c> or <c>target</c>.</exception>
        private NDArray GetObs() {
            int nq = Model.Nq;
            var values = new double[4 + (nq - 2) + 2 + 3];
            values[0] = Math.Cos(Data.qpos[0]);
            values[1] = Math.Cos(Data.qpos[1]);
            values[2] = Math.Sin(Data.qpos[0]);
            values[3] = Math.Sin(Data.qpos[1]);
            Array.Copy(Data.qpos, 2, values, 4, nq - 2);
            values[nq + 2] = Data.qvel[0];
            values[nq + 3] = Data.qvel[1];
            (double dx, double dy, double dz) = FingertipOffset();
            values[nq + 4] = dx;
            values[nq + 5] = dy;
            values[nq + 6] = dz;
            return np.array(values);
        }
    }
}
