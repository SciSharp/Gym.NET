using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     InvertedPendulum-v4: the cart and pole of <see cref="InvertedPendulumEnvV5"/> with v4's fixed model, frame skip and
    ///     noise, a reward of 1.0 on every step (the step the pole falls included) and an empty info. Byte for byte what
    ///     Gymnasium's InvertedPendulum-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the cart's position, the pole's angle, then their velocities. A non-finite state or an
    ///         angle beyond 0.2 rad ends the episode. <c>make</c> warns that the id is out of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/inverted_pendulum_v4.py#L13">gymnasium/envs/mujoco/inverted_pendulum_v4.py</see>.
    ///     </para>
    ///     <para>Its info is always <c>{}</c>, as Gymnasium's, so it has no typed info record (DR-95).</para>
    /// </remarks>
    [GymOptions]
    [PyName("InvertedPendulumEnv")]
    public partial class InvertedPendulumEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>InvertedPendulumEnv.__init__</c>): <c>inverted_pendulum.xml</c> with a frame skip of 2.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <c>human</c>, <c>rgb_array</c>, <c>depth_array</c>, <c>rgbd_tuple</c> or <see langword="null"/>.</param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public InvertedPendulumEnvV4(string? renderMode = null, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                     string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                     EnvViewerFactory? viewerFactory = null)
            : base("inverted_pendulum.xml", 2, UnboundedBox(4), ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 25 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(25);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> 0, <c>distance</c> 2.04. A new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 0L, ["distance"] = 2.04,
        };

        /// <summary>Applies <paramref name="action"/> (Gymnasium's <c>a</c>) for 2 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The force on the cart, shape <c>(1,)</c>, any real dtype (stored as float64).</param>
        /// <returns>The observation, the reward 1.0, whether the pole fell, <c>truncated=False</c> and an empty info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(1,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            const double reward = 1.0;
            DoSimulation(action, FrameSkip);
            double[] ob = ObservationValues();
            bool terminated = false;
            foreach (double value in ob)
                terminated |= !double.IsFinite(value);
            terminated = terminated || Math.Abs(ob[1]) > 0.2;
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(np.array(ob), reward, terminated, false, new Info());
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="InvertedPendulumEnvV4(string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new InvertedPendulumEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new InvertedPendulumEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new InvertedPendulumEnvV4()</c> keeps meaning the other constructor, without a
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
        public InvertedPendulumEnvV4(RenderMode renderMode, int width = DefaultSize, int height = DefaultSize, int? cameraId = null,
                                     string? cameraName = null, int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                     EnvViewerFactory? viewerFactory = null)
            : this(RenderModes.Name(renderMode), width, height, cameraId, cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-0.01, 0.01, nq)</c>, then velocities
        ///     <c>init_qvel + uniform(-0.01, 0.01, nv)</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            double[] qpos = AddNoise(InitQpos, UniformNoise(-0.01, 0.01, Model.Nq));
            double[] qvel = AddNoise(InitQvel, UniformNoise(-0.01, 0.01, Model.Nv));
            SetStateValues(qpos, qvel);
            return np.array(ObservationValues());
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
