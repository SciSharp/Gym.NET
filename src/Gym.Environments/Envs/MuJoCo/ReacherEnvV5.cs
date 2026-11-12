using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Reacher-v5: a two-joint planar arm that reaches for a random target, driven by 2 joint torques in [-1, 1]. The
    ///     reward charges the fingertip's distance to the target and the squared torques; the episode never ends on its own
    ///     (<c>make</c>'s 50-step limit truncates it). Byte for byte what Gymnasium's Reacher-v5 computes on MuJoCo 4.0.0
    ///     (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64, 10 values): the cosines and sines of the two arm angles, the target's two coordinates,
    ///         the arm's two angular velocities, and the fingertip's offset from the target in x and y. Reward:
    ///         <c>-|fingertip - target| * reward_dist_weight - sum(action**2) * reward_control_weight</c>, measured after the
    ///         step, the control term in the action's dtype. Info: <c>reward_dist</c>, <c>reward_ctrl</c>; the reset's info
    ///         is empty. The reset draws the target uniformly in the disc of radius 0.2 by rejection (<see cref="Goal"/>).
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/reacher_v5.py#L14">gymnasium/envs/mujoco/reacher_v5.py</see>.
    ///     </para>
    /// </remarks>
    [GymOptions]
    [PyName("ReacherEnv")]
    public partial class ReacherEnvV5 : MujocoEnv {
        /// <summary>The target the last reset drew (<c>goal</c>), or <see langword="null"/> before it.</summary>
        private NDArray? goal;

        /// <summary>The weight of the squared torques, <see cref="RewardControlWeight"/>'s value.</summary>
        private double rewardControlWeight;

        /// <summary>
        ///     Whether the control weight is Gymnasium's default, the Python int 1: an integer action's cost then stays an
        ///     integer (<c>-np.square(action).sum() * 1</c> is an <c>np.int64</c>).
        /// </summary>
        private bool controlWeightIsInt;

        /// <summary>Creates the env (Gymnasium's <c>ReacherEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default
        ///     (<c>trackbodyid</c> 0), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="rewardDistWeight">The weight of the fingertip's distance to the target (<c>reward_dist_weight</c>; Gymnasium's default is the int 1).</param>
        /// <param name="rewardControlWeight">
        ///     The weight of the squared torques (<c>reward_control_weight</c>); <see langword="null"/> is Gymnasium's default,
        ///     the Python int 1, which keeps an integer action's cost an <c>np.int64</c> (a given weight is a Python float).
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
        /// <exception cref="OverflowException">A frame skip of 0 (Python's <c>OverflowError</c> deriving the frame rate).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public ReacherEnvV5(string xmlFile = "reacher.xml", int frameSkip = 2, IReadOnlyDictionary<string, object?>? defaultCameraConfig = null,
                            double rewardDistWeight = 1, double? rewardControlWeight = null, string? renderMode = null, int width = DefaultSize,
                            int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                            IReadOnlyDictionary<int, bool>? visualOptions = null, IEnvironmentViewerFactoryDelegate? viewerFactory = null)
            : base(xmlFile, frameSkip, UnboundedBox(10), ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            RewardDistWeight = rewardDistWeight;
            RewardControlWeight = rewardControlWeight ?? 1;
            // the default is Python's int 1, so an integer action's cost stays an integer; set after the property, which
            // clears the flag for a float weight
            controlWeightIsInt = rewardControlWeight is null;
            Metadata = MetadataFromDt();
        }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, <c>{"trackbodyid": 0}</c>: a new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> { ["trackbodyid"] = 0L };

        /// <summary>The weight of the fingertip's distance to the target (<c>_reward_dist_weight</c>), read every step.</summary>
        [PyName("_reward_dist_weight")]
        public double RewardDistWeight { get; set; }

        /// <summary>
        ///     The weight of the squared torques (<c>_reward_control_weight</c>), read every step. Assigning one makes it a
        ///     Python float, as assigning <c>1.0</c> in Python would: an integer action's cost then becomes an <c>np.float64</c>.
        /// </summary>
        [PyName("_reward_control_weight")]
        public double RewardControlWeight {
            get => rewardControlWeight;
            set {
                rewardControlWeight = value;
                controlWeightIsInt = false;
            }
        }

        /// <summary>
        ///     The target the last reset drew (<c>goal</c>), float64 <c>(2,)</c>, or <see langword="null"/> before the first
        ///     reset. Owned by the env, replaced by each reset: do not dispose it.
        /// </summary>
        [NDBorrowed]
        public NDArray? Goal => goal;

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 2 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(2,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        /// <exception cref="KeyError">The model lacks the body <c>fingertip</c> or <c>target</c> (MuJoCo's message).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            DoSimulation(action, FrameSkip);
            NDArray observation = GetObs();

            (double dx, double dy, double dz) = FingertipOffset();
            NpScalar rewardDist = NpScalar.Float64(-MujocoMath.Norm(dx, dy, dz)) * NpScalar.PyFloat(RewardDistWeight);
            SquaredSum negated = squares.Negate();
            // an integer cost times Python's int 1 is the same integer; any other pairing follows NEP 50 (Times)
            bool integral = controlWeightIsInt && negated.IsInteger;
            NpScalar rewardCtrl = integral ? negated.AsOperand() : negated.Times(RewardControlWeight);
            NpScalar reward = rewardDist + rewardCtrl;
            var info = new Info {
                ["reward_dist"] = MujocoMath.InfoValue(rewardDist),
                ["reward_ctrl"] = integral ? negated.InfoValue() : MujocoMath.InfoValue(rewardCtrl),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, false, false, info);
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
        ///     positions, the first two velocities, and the fingertip's offset in x and y.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        /// <exception cref="KeyError">The model lacks the body <c>fingertip</c> or <c>target</c>.</exception>
        private NDArray GetObs() {
            int nq = Model.Nq;
            var values = new double[4 + (nq - 2) + 2 + 2];
            values[0] = Math.Cos(Data.qpos[0]);
            values[1] = Math.Cos(Data.qpos[1]);
            values[2] = Math.Sin(Data.qpos[0]);
            values[3] = Math.Sin(Data.qpos[1]);
            Array.Copy(Data.qpos, 2, values, 4, nq - 2);
            values[nq + 2] = Data.qvel[0];
            values[nq + 3] = Data.qvel[1];
            (double dx, double dy, _) = FingertipOffset();
            values[nq + 4] = dx;
            values[nq + 5] = dy;
            return np.array(values);
        }
    }
}
