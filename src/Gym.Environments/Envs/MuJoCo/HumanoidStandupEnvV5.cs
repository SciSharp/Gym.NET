using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     HumanoidStandup-v5: the humanoid lying on the ground, rewarded for raising its torso. The reward pays the torso's
    ///     height over the time step plus 1, and charges the squared controls and the external contact forces; the episode
    ///     never ends on its own (<c>make</c>'s 1000-step limit truncates it). Byte for byte what Gymnasium's
    ///     HumanoidStandup-v5 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation: as <see cref="HumanoidEnvV5"/>'s, 348 values by default. Reward: <c>z / timestep -
    ///         ctrl_cost_weight * sum(ctrl**2) - clip(impact_cost_weight * sum(cfrc_ext**2), *impact_cost_range) + 1</c>, all
    ///         float64. <c>uph_cost_weight</c> is taken and kept but never used, as in Gymnasium. Info: <c>x_position</c>,
    ///         <c>y_position</c>, <c>z_distance_from_origin</c>, <c>tendon_length</c>, <c>tendon_velocity</c>,
    ///         <c>reward_linup</c>, <c>reward_quadctrl</c>, <c>reward_impact</c>; the reset's info holds the first five.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/humanoidstandup_v5.py#L17">gymnasium/envs/mujoco/humanoidstandup_v5.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>HumanoidStandupInfoV5 info = step.Info;</c> (<see cref="HumanoidStandupInfoV5"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("HumanoidStandupEnv")]
    public partial class HumanoidStandupEnvV5 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>HumanoidStandupEnv.__init__</c>): the model, its spaces and the frame rate the time step gives.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default (tracking
        ///     body 1 from 4 m, looking at (0, 0, 0.8925), 20° from above), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="uphCostWeight">The weight Gymnasium documents for the height term (<c>uph_cost_weight</c>); kept, never used.</param>
        /// <param name="ctrlCostWeight">The weight of the squared controls (<c>ctrl_cost_weight</c>).</param>
        /// <param name="impactCostWeight">The weight of the squared external forces (<c>impact_cost_weight</c>).</param>
        /// <param name="impactCostRange">The range the impact cost is clipped to (<c>impact_cost_range</c>); <see langword="null"/> is (-inf, 10.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="includeCinertInObservation">Whether the bodies' inertias join the observation (<c>include_cinert_in_observation</c>).</param>
        /// <param name="includeCvelInObservation">Whether the bodies' velocities join the observation (<c>include_cvel_in_observation</c>).</param>
        /// <param name="includeQfrcActuatorInObservation">Whether the actuator forces join the observation (<c>include_qfrc_actuator_in_observation</c>).</param>
        /// <param name="includeCfrcExtInObservation">Whether the external forces join the observation (<c>include_cfrc_ext_in_observation</c>).</param>
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
        public HumanoidStandupEnvV5(string xmlFile = "humanoidstandup.xml", int frameSkip = 5,
                                    IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double uphCostWeight = 1,
                                    double ctrlCostWeight = 0.1, double impactCostWeight = 0.5e-6, (double Low, double High)? impactCostRange = null,
                                    double resetNoiseScale = 1e-2, bool excludeCurrentPositionsFromObservation = true,
                                    bool includeCinertInObservation = true, bool includeCvelInObservation = true,
                                    bool includeQfrcActuatorInObservation = true, bool includeCfrcExtInObservation = true, string? renderMode = null,
                                    int width = DefaultSize, int height = DefaultSize, int? cameraId = null, string? cameraName = null,
                                    int maxGeom = 1000, IReadOnlyDictionary<int, bool>? visualOptions = null,
                                    EnvViewerFactory? viewerFactory = null)
            : base(xmlFile, frameSkip, null, ClassMetadata, renderMode, width, height, cameraId, cameraName,
                   defaultCameraConfig ?? DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            UphCostWeight = uphCostWeight;
            CtrlCostWeight = ctrlCostWeight;
            ImpactCostWeight = impactCostWeight;
            ImpactCostRange = impactCostRange ?? (double.NegativeInfinity, 10.0);
            ResetNoiseScale = resetNoiseScale;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
            IncludeCinertInObservation = includeCinertInObservation;
            IncludeCvelInObservation = includeCvelInObservation;
            IncludeQfrcActuatorInObservation = includeQfrcActuatorInObservation;
            IncludeCfrcExtInObservation = includeCfrcExtInObservation;
            Metadata = MetadataFromDt();
            int skipped = excludeCurrentPositionsFromObservation ? 2 : 0;
            int bodies = Model.Nbody - 1;
            int cinert = includeCinertInObservation ? bodies * 10 : 0;
            int cvel = includeCvelInObservation ? bodies * 6 : 0;
            int actuator = includeQfrcActuatorInObservation ? Model.Nv - 6 : 0;
            int external = includeCfrcExtInObservation ? bodies * 6 : 0;
            ObservationSpace = UnboundedBox(Model.Nq + Model.Nv - skipped + cinert + cvel + actuator + external);
            ObservationStructure = new Info {
                ["skipped_qpos"] = (long)skipped,
                ["qpos"] = (long)(Model.Nq - skipped),
                ["qvel"] = (long)Model.Nv,
                ["cinert"] = (long)cinert,
                ["cvel"] = (long)cvel,
                ["qfrc_actuator"] = (long)actuator,
                ["cfrc_ext"] = (long)external,
                ["ten_length"] = 0L,
                ["ten_velocity"] = 0L,
            };
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="HumanoidStandupEnvV5(string, int, IReadOnlyDictionary{string,object}, double, double, double, Nullable{ValueTuple{double,double}}, double, bool, bool, bool, bool, bool, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new HumanoidStandupEnvV5(RenderMode.RgbArray)</c> builds what
        ///     <c>new HumanoidStandupEnvV5(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new HumanoidStandupEnvV5()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="frameSkip">Physics steps per env step (<c>frame_skip</c>).</param>
        /// <param name="defaultCameraConfig">
        ///     The free camera's settings (<c>default_camera_config</c>); <see langword="null"/> is Gymnasium's default (tracking
        ///     body 1 from 4 m, looking at (0, 0, 0.8925), 20° from above), and an empty dictionary leaves MuJoCo's camera as it is.
        /// </param>
        /// <param name="uphCostWeight">The weight Gymnasium documents for the height term (<c>uph_cost_weight</c>); kept, never used.</param>
        /// <param name="ctrlCostWeight">The weight of the squared controls (<c>ctrl_cost_weight</c>).</param>
        /// <param name="impactCostWeight">The weight of the squared external forces (<c>impact_cost_weight</c>).</param>
        /// <param name="impactCostRange">The range the impact cost is clipped to (<c>impact_cost_range</c>); <see langword="null"/> is (-inf, 10.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="includeCinertInObservation">Whether the bodies' inertias join the observation (<c>include_cinert_in_observation</c>).</param>
        /// <param name="includeCvelInObservation">Whether the bodies' velocities join the observation (<c>include_cvel_in_observation</c>).</param>
        /// <param name="includeQfrcActuatorInObservation">Whether the actuator forces join the observation (<c>include_qfrc_actuator_in_observation</c>).</param>
        /// <param name="includeCfrcExtInObservation">Whether the external forces join the observation (<c>include_cfrc_ext_in_observation</c>).</param>
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
        public HumanoidStandupEnvV5(RenderMode renderMode, string xmlFile = "humanoidstandup.xml", int frameSkip = 5,
                                    IReadOnlyDictionary<string, object?>? defaultCameraConfig = null, double uphCostWeight = 1,
                                    double ctrlCostWeight = 0.1, double impactCostWeight = 0.5e-6, (double Low, double High)? impactCostRange = null,
                                    double resetNoiseScale = 1e-2, bool excludeCurrentPositionsFromObservation = true,
                                    bool includeCinertInObservation = true, bool includeCvelInObservation = true,
                                    bool includeQfrcActuatorInObservation = true, bool includeCfrcExtInObservation = true, int width = DefaultSize,
                                    int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                                    IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : this(xmlFile, frameSkip, defaultCameraConfig, uphCostWeight, ctrlCostWeight, impactCostWeight, impactCostRange, resetNoiseScale,
                   excludeCurrentPositionsFromObservation, includeCinertInObservation, includeCvelInObservation, includeQfrcActuatorInObservation,
                   includeCfrcExtInObservation, RenderModes.Name(renderMode), width, height, cameraId, cameraName, maxGeom, visualOptions,
                   viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes, the frame rate set per instance. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(null);

        /// <summary>
        ///     Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>: <c>trackbodyid</c> 1, <c>distance</c> 4.0, <c>lookat</c>
        ///     (0, 0, 0.8925), <c>elevation</c> -20.0. A new dictionary on every read.
        /// </summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> {
            ["trackbodyid"] = 1L, ["distance"] = 4.0, ["lookat"] = new[] { 0.0, 0.0, 0.8925 }, ["elevation"] = -20.0,
        };

        /// <summary>The documented weight of the height term (<c>_uph_cost_weight</c>): stored, never read, as in Gymnasium.</summary>
        [PyName("_uph_cost_weight")]
        public double UphCostWeight { get; set; }

        /// <summary>The weight of the squared controls (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The weight of the squared external forces (<c>_impact_cost_weight</c>), read every step.</summary>
        [PyName("_impact_cost_weight")]
        public double ImpactCostWeight { get; set; }

        /// <summary>The range the impact cost is clipped to (<c>_impact_cost_range</c>).</summary>
        [PyName("_impact_cost_range")]
        public (double Low, double High) ImpactCostRange { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>
        ///     Whether the root's x and y leave the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
        ///     this and the four <c>Include...</c> flags change the observation's length when changed after construction, but
        ///     not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>Whether the bodies' inertias join the observation (<c>_include_cinert_in_observation</c>).</summary>
        [PyName("_include_cinert_in_observation")]
        public bool IncludeCinertInObservation { get; set; }

        /// <summary>Whether the bodies' velocities join the observation (<c>_include_cvel_in_observation</c>).</summary>
        [PyName("_include_cvel_in_observation")]
        public bool IncludeCvelInObservation { get; set; }

        /// <summary>Whether the actuator forces join the observation (<c>_include_qfrc_actuator_in_observation</c>).</summary>
        [PyName("_include_qfrc_actuator_in_observation")]
        public bool IncludeQfrcActuatorInObservation { get; set; }

        /// <summary>Whether the external forces join the observation (<c>_include_cfrc_ext_in_observation</c>).</summary>
        [PyName("_include_cfrc_ext_in_observation")]
        public bool IncludeCfrcExtInObservation { get; set; }

        /// <summary>
        ///     How the observation is laid out (<c>observation_structure</c>): <c>skipped_qpos</c>, <c>qpos</c>, <c>qvel</c>,
        ///     <c>cinert</c>, <c>cvel</c>, <c>qfrc_actuator</c>, <c>cfrc_ext</c>, and 0 for <c>ten_length</c> and
        ///     <c>ten_velocity</c>.
        /// </summary>
        public Info ObservationStructure { get; }

        /// <summary>Applies <paramref name="action"/> for <see cref="MujocoEnv.FrameSkip"/> physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 17 controls, any real dtype (stored as float64).</param>
        /// <returns>The observation, the float64 reward, <c>terminated=False</c>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(17,)</c>, with Gymnasium's message.</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            DoSimulation(action, FrameSkip);
            double posAfter = Data.qpos[2];

            NpScalar uphCost = NpScalar.Float64((posAfter - 0) / Model.Opt.Timestep);
            NpScalar quadCtrlCost = NpScalar.PyFloat(CtrlCostWeight) * NpScalar.Float64(MujocoMath.SumOfSquares(Data.ctrl));
            double impact = ImpactCostWeight * MujocoMath.SumOfSquares(Data.cfrc_ext);
            NpScalar quadImpactCost = NpScalar.Float64(PyMath.Clip(impact, ImpactCostRange.Low, ImpactCostRange.High));
            NpScalar reward = uphCost - quadCtrlCost - quadImpactCost + NpScalar.PyInt(1);
            var info = new Info {
                ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
                ["y_position"] = MujocoMath.Float64(Data.qpos[1]),
                ["z_distance_from_origin"] = MujocoMath.Float64(Data.qpos[2] - InitQpos.GetDouble(2)),
                ["tendon_length"] = MujocoMath.Copy(Data.ten_length),
                ["tendon_velocity"] = MujocoMath.Copy(Data.ten_velocity),
                ["reward_linup"] = MujocoMath.InfoValue(uphCost),
                ["reward_quadctrl"] = MujocoMath.InfoValue(-quadCtrlCost),
                ["reward_impact"] = MujocoMath.InfoValue(-quadImpactCost),
            };
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(GetObs(), reward.Value, false, false, info);
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

        /// <summary>
        ///     The reset's info (<c>_get_reset_info</c>): <c>x_position</c>, <c>y_position</c>, <c>z_distance_from_origin</c>,
        ///     and copies of <c>tendon_length</c> and <c>tendon_velocity</c>.
        /// </summary>
        /// <returns>The info.</returns>
        protected override Info GetResetInfo() => new() {
            ["x_position"] = MujocoMath.Float64(Data.qpos[0]),
            ["y_position"] = MujocoMath.Float64(Data.qpos[1]),
            ["z_distance_from_origin"] = MujocoMath.Float64(Data.qpos[2] - InitQpos.GetDouble(2)),
            ["tendon_length"] = MujocoMath.Copy(Data.ten_length),
            ["tendon_velocity"] = MujocoMath.Copy(Data.ten_velocity),
        };

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the positions (the root's x and y left out unless asked for), the velocities,
        ///     then each included block: <c>cinert[1:]</c>, <c>cvel[1:]</c>, <c>qfrc_actuator[6:]</c>, <c>cfrc_ext[1:]</c>.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 2 : 0;
            var values = new List<double>(Model.Nq + Model.Nv + Model.Nbody * 22 + Model.Nv);
            values.AddRange(new ArraySegment<double>(Data.qpos, skipped, Model.Nq - skipped));
            values.AddRange(Data.qvel);
            // the world body's rows (10 inertia, 6 velocity and 6 force values) and the free root's 6 dofs are left out
            if (IncludeCinertInObservation)
                values.AddRange(new ArraySegment<double>(Data.cinert, 10, Data.cinert.Length - 10));
            if (IncludeCvelInObservation)
                values.AddRange(new ArraySegment<double>(Data.cvel, 6, Data.cvel.Length - 6));
            if (IncludeQfrcActuatorInObservation)
                values.AddRange(new ArraySegment<double>(Data.qfrc_actuator, 6, Data.qfrc_actuator.Length - 6));
            if (IncludeCfrcExtInObservation)
                values.AddRange(new ArraySegment<double>(Data.cfrc_ext, 6, Data.cfrc_ext.Length - 6));
            return np.array(values.ToArray());
        }
    }
}
