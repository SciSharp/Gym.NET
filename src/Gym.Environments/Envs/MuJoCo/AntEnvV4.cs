using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.MuJoCo {
    /// <summary>
    ///     Ant-v4: the four-legged ant of <see cref="AntEnvV5"/> with v4's frame skip, the torso's velocity as the forward
    ///     reward (unweighted), v4's healthy reward (paid on every step while <c>terminate_when_unhealthy</c> is set), no
    ///     contact forces unless <c>use_contact_forces=True</c>, and an empty reset info. Byte for byte what Gymnasium's
    ///     Ant-v4 computes on MuJoCo 4.0.0 (DR-61, DR-64).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Observation (float64): the positions without the root's x and y (unless asked for), the velocities, and with
    ///         <c>use_contact_forces</c> the clipped contact forces of every body, the world's included; the space declares
    ///         27 values (+2 with the root's x and y, +84 with contact forces) whatever the model, as v4 does. Reward:
    ///         <c>x_velocity + float(is_healthy or terminate_when_unhealthy) * healthy_reward - ctrl_cost_weight *
    ///         sum(action**2)</c>, minus the contact cost with <c>use_contact_forces</c>, which then also replaces the info's
    ///         <c>reward_ctrl</c> with <c>-contact_cost</c> (Gymnasium's own quirk, kept). <c>make</c> warns that the id is out
    ///         of date.
    ///     </para>
    ///     <para>
    ///         Not thread-safe. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/ant_v4.py#L12">gymnasium/envs/mujoco/ant_v4.py</see>.
    ///     </para>
    ///     <para>Its info reads typed (DR-91): <c>AntInfoV4 info = step.Info;</c> (<see cref="AntInfoV4"/>), with the dict's own values still behind the indexer.</para>
    /// </remarks>
    [GymOptions]
    [PyName("AntEnv")]
    public partial class AntEnvV4 : MujocoEnv {
        /// <summary>Creates the env (Gymnasium's <c>AntEnv.__init__</c>): the model with a frame skip of 5.</summary>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="useContactForces">Whether the clipped contact forces join the observation and the cost (<c>use_contact_forces</c>).</param>
        /// <param name="contactCostWeight">The weight of the squared clipped contact forces (<c>contact_cost_weight</c>).</param>
        /// <param name="healthyReward">The healthy reward per step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The inclusive range of a healthy torso height (<c>healthy_z_range</c>); <see langword="null"/> is (0.2, 1.0).</param>
        /// <param name="contactForceRange">The range the contact forces are clipped to (<c>contact_force_range</c>); <see langword="null"/> is (-1.0, 1.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
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
        /// <exception cref="AssertionError">A model whose time step does not give v4's 20 frames per second (Gymnasium's assertion).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public AntEnvV4(string xmlFile = "ant.xml", double ctrlCostWeight = 0.5, bool useContactForces = false, double contactCostWeight = 5e-4,
                        double healthyReward = 1.0, bool terminateWhenUnhealthy = true, (double Low, double High)? healthyZRange = null,
                        (double Low, double High)? contactForceRange = null, double resetNoiseScale = 0.1,
                        bool excludeCurrentPositionsFromObservation = true, string? renderMode = null, int width = DefaultSize,
                        int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                        IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : base(xmlFile, 5, UnboundedBox(27 + (excludeCurrentPositionsFromObservation ? 0 : 2) + (useContactForces ? 84 : 0)), ClassMetadata,
                   renderMode, width, height, cameraId, cameraName, DefaultCameraConfig, maxGeom, visualOptions, viewerFactory) {
            CtrlCostWeight = ctrlCostWeight;
            ContactCostWeight = contactCostWeight;
            HealthyRewardWeight = healthyReward;
            TerminateWhenUnhealthy = terminateWhenUnhealthy;
            HealthyZRange = healthyZRange ?? (0.2, 1.0);
            ContactForceRange = contactForceRange ?? (-1.0, 1.0);
            ResetNoiseScale = resetNoiseScale;
            UseContactForces = useContactForces;
            ExcludeCurrentPositionsFromObservation = excludeCurrentPositionsFromObservation;
        }

        /// <summary>
        ///     Builds the env with the render mode given as the <see cref="Gym.Envs.RenderMode"/> enum (DR-84, DR-93), as
        ///     <see cref="AntEnvV4(string, double, bool, double, double, bool, Nullable{ValueTuple{double,double}}, Nullable{ValueTuple{double,double}}, double, bool, string, int, int, int?, string, int, IReadOnlyDictionary{int,bool}, EnvViewerFactory)"/>
        ///     builds it from the mode's string: <c>new AntEnvV4(RenderMode.RgbArray)</c> builds what
        ///     <c>new AntEnvV4(renderMode: "rgb_array")</c> builds, the member reaching the env as Gymnasium's string
        ///     (<see cref="RenderModes.Name"/>), which its <c>RenderMode</c> property then holds.
        /// </summary>
        /// <remarks>
        ///     The mode is required and comes first, so <c>new AntEnvV4()</c> keeps meaning the other constructor, without a
        ///     mode; leave the mode out there for no rendering. Every other parameter means what it means there.
        /// </remarks>
        /// <param name="renderMode">
        ///     The render mode, which the env must list in its <c>metadata["render_modes"]</c> as for the string; it behaves
        ///     exactly as its Gymnasium string does in the other constructor.
        /// </param>
        /// <param name="xmlFile">The model: a bundled file name (<c>xml_file</c>, DR-63).</param>
        /// <param name="ctrlCostWeight">The weight of the squared torques (<c>ctrl_cost_weight</c>).</param>
        /// <param name="useContactForces">Whether the clipped contact forces join the observation and the cost (<c>use_contact_forces</c>).</param>
        /// <param name="contactCostWeight">The weight of the squared clipped contact forces (<c>contact_cost_weight</c>).</param>
        /// <param name="healthyReward">The healthy reward per step (<c>healthy_reward</c>).</param>
        /// <param name="terminateWhenUnhealthy">Whether an unhealthy state ends the episode (<c>terminate_when_unhealthy</c>).</param>
        /// <param name="healthyZRange">The inclusive range of a healthy torso height (<c>healthy_z_range</c>); <see langword="null"/> is (0.2, 1.0).</param>
        /// <param name="contactForceRange">The range the contact forces are clipped to (<c>contact_force_range</c>); <see langword="null"/> is (-1.0, 1.0).</param>
        /// <param name="resetNoiseScale">The scale of the reset's noise (<c>reset_noise_scale</c>).</param>
        /// <param name="excludeCurrentPositionsFromObservation">
        ///     Whether the root's x and y leave the observation (<c>exclude_current_positions_from_observation</c>).
        /// </param>
        /// <param name="width">The rendered frame's width (<c>width</c>).</param>
        /// <param name="height">The rendered frame's height (<c>height</c>).</param>
        /// <param name="cameraId">The camera to render from (<c>camera_id</c>).</param>
        /// <param name="cameraName">The camera to render from by name (<c>camera_name</c>).</param>
        /// <param name="maxGeom">The most geoms a frame draws (<c>max_geom</c>).</param>
        /// <param name="visualOptions">MuJoCo visual flags to switch (<c>visual_options</c>).</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for <c>human</c> mode.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for a value <see cref="Gym.Envs.RenderMode"/> does not declare (a cast integer such as <c>(RenderMode)99</c>).</exception>
        /// <exception cref="NotSupportedException"><paramref name="xmlFile"/> is not a bundled model.</exception>
        /// <exception cref="AssertionError">A model whose time step does not give v4's 20 frames per second (Gymnasium's assertion).</exception>
        /// <exception cref="ValueError">Both a camera id and a camera name.</exception>
        public AntEnvV4(RenderMode renderMode, string xmlFile = "ant.xml", double ctrlCostWeight = 0.5, bool useContactForces = false,
                        double contactCostWeight = 5e-4, double healthyReward = 1.0, bool terminateWhenUnhealthy = true,
                        (double Low, double High)? healthyZRange = null, (double Low, double High)? contactForceRange = null,
                        double resetNoiseScale = 0.1, bool excludeCurrentPositionsFromObservation = true, int width = DefaultSize,
                        int height = DefaultSize, int? cameraId = null, string? cameraName = null, int maxGeom = 1000,
                        IReadOnlyDictionary<int, bool>? visualOptions = null, EnvViewerFactory? viewerFactory = null)
            : this(xmlFile, ctrlCostWeight, useContactForces, contactCostWeight, healthyReward, terminateWhenUnhealthy, healthyZRange,
                   contactForceRange, resetNoiseScale, excludeCurrentPositionsFromObservation, RenderModes.Name(renderMode), width, height, cameraId,
                   cameraName, maxGeom, visualOptions, viewerFactory) { }

        /// <summary>Gymnasium's class attribute <c>metadata</c>: the four render modes at 20 frames per second. A new dict on every read.</summary>
        public static new Metadata ClassMetadata => MetadataFor(20);

        /// <summary>Gymnasium's <c>DEFAULT_CAMERA_CONFIG</c>, <c>{"distance": 4.0}</c>: a new dictionary on every read.</summary>
        private static IReadOnlyDictionary<string, object?> DefaultCameraConfig => new Dictionary<string, object?> { ["distance"] = 4.0 };

        /// <summary>The weight of the squared torques (<c>_ctrl_cost_weight</c>), read every step.</summary>
        [PyName("_ctrl_cost_weight")]
        public double CtrlCostWeight { get; set; }

        /// <summary>The weight of the squared clipped contact forces (<c>_contact_cost_weight</c>), read every step.</summary>
        [PyName("_contact_cost_weight")]
        public double ContactCostWeight { get; set; }

        /// <summary>The healthy reward per step (<c>_healthy_reward</c>), read every step.</summary>
        [PyName("_healthy_reward")]
        public double HealthyRewardWeight { get; set; }

        /// <summary>Whether an unhealthy state ends the episode (<c>_terminate_when_unhealthy</c>), read every step.</summary>
        [PyName("_terminate_when_unhealthy")]
        public bool TerminateWhenUnhealthy { get; set; }

        /// <summary>The inclusive range of a healthy torso height (<c>_healthy_z_range</c>).</summary>
        [PyName("_healthy_z_range")]
        public (double Low, double High) HealthyZRange { get; set; }

        /// <summary>The range the contact forces are clipped to (<c>_contact_force_range</c>).</summary>
        [PyName("_contact_force_range")]
        public (double Low, double High) ContactForceRange { get; set; }

        /// <summary>The scale of the reset's noise (<c>_reset_noise_scale</c>), read every reset.</summary>
        [PyName("_reset_noise_scale")]
        public double ResetNoiseScale { get; set; }

        /// <summary>
        ///     Whether the clipped contact forces join the observation and the cost (<c>_use_contact_forces</c>). Footgun:
        ///     changing it after construction changes the observation's length but not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_use_contact_forces")]
        public bool UseContactForces { get; set; }

        /// <summary>
        ///     Whether the root's x and y leave the observation (<c>_exclude_current_positions_from_observation</c>). Footgun:
        ///     changing it after construction changes the observation's length but not the observation space, as in Gymnasium.
        /// </summary>
        [PyName("_exclude_current_positions_from_observation")]
        public bool ExcludeCurrentPositionsFromObservation { get; set; }

        /// <summary>
        ///     Whether the ant is healthy now (<c>is_healthy</c>): every position and velocity finite and the torso's height
        ///     inside the inclusive z range.
        /// </summary>
        [PyName("is_healthy")]
        public bool IsHealthy {
            get {
                for (int i = 0; i < Model.Nq; i++)
                    if (!double.IsFinite(Data.qpos[i]))
                        return false;
                for (int i = 0; i < Model.Nv; i++)
                    if (!double.IsFinite(Data.qvel[i]))
                        return false;
                double z = Data.qpos[2];
                return HealthyZRange.Low <= z && z <= HealthyZRange.High;
            }
        }

        /// <summary>
        ///     The healthy reward now (<c>healthy_reward</c>): <c>float(is_healthy or terminate_when_unhealthy) *
        ///     _healthy_reward</c>, so paid on every step while unhealthy states end the episode.
        /// </summary>
        [PyName("healthy_reward")]
        public double HealthyReward => ((IsHealthy || TerminateWhenUnhealthy) ? 1.0 : 0.0) * HealthyRewardWeight;

        /// <summary>Whether the episode ended (<c>terminated</c>): unhealthy while <c>terminate_when_unhealthy</c> is set.</summary>
        [PyName("terminated")]
        public bool Terminated => TerminateWhenUnhealthy && !IsHealthy;

        /// <summary>
        ///     The contact forces clipped to <see cref="ContactForceRange"/> (<c>contact_forces</c>), one row of 6 per body
        ///     (the world's included). A new <c>(nbody, 6)</c> array on every read, which the caller owns.
        /// </summary>
        [PyName("contact_forces")]
        public NDArray ContactForces {
            get {
                var clipped = new double[Data.cfrc_ext.Length];
                for (int i = 0; i < clipped.Length; i++)
                    clipped[i] = PyMath.Clip(Data.cfrc_ext[i], ContactForceRange.Low, ContactForceRange.High);
                using NDArray flat = np.array(clipped);
                return flat.reshape(Model.Nbody, 6);
            }
        }

        /// <summary>The contact cost now (<c>contact_cost</c>): <c>contact_cost_weight * sum(contact_forces**2)</c>.</summary>
        [PyName("contact_cost")]
        public double ContactCost => ContactCostWeight * MujocoMath.SumOfClippedSquares(Data.cfrc_ext, ContactForceRange.Low, ContactForceRange.High);

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

        /// <summary>Applies <paramref name="action"/> for 5 physics steps and reports the transition (<c>step</c>).</summary>
        /// <param name="action">The 8 torques, float32 (the action space's samples) or any real dtype.</param>
        /// <returns>The observation, the float64 reward, <see cref="Terminated"/>, <c>truncated=False</c> and the info.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="action"/> is null.</exception>
        /// <exception cref="ValueError">The action is not of shape <c>(8,)</c>, with Gymnasium's message.</exception>
        /// <exception cref="NotSupportedException">An action of a dtype whose cost NumPy keeps in that dtype (float16, complex).</exception>
        /// <exception cref="KeyError">The model has no body named <c>torso</c> (MuJoCo's message).</exception>
        public override StepResult<NDArray> Step(NDArray action) {
            // the cost depends only on the action: computed first, an unsupported dtype is refused before the physics runs
            SquaredSum squares = MujocoMath.SumOfSquares(action);
            int torso = Data.Body("torso").Id;
            double xBefore = Data.xpos[3 * torso], yBefore = Data.xpos[3 * torso + 1];
            DoSimulation(action, FrameSkip);
            torso = Data.Body("torso").Id;
            double xAfter = Data.xpos[3 * torso], yAfter = Data.xpos[3 * torso + 1];
            double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;

            NpScalar forwardReward = NpScalar.Float64(xVelocity);
            NpScalar healthyReward = NpScalar.PyFloat(HealthyReward);
            NpScalar rewards = forwardReward + healthyReward;
            NpScalar ctrlCost = squares.Times(CtrlCostWeight);
            NpScalar costs = ctrlCost;
            bool terminated = Terminated;
            NDArray observation = GetObs();
            var info = new Info {
                ["reward_forward"] = MujocoMath.InfoValue(forwardReward),
                ["reward_ctrl"] = MujocoMath.InfoValue(-ctrlCost),
                ["reward_survive"] = MujocoMath.InfoValue(healthyReward),
                ["x_position"] = MujocoMath.Float64(xAfter),
                ["y_position"] = MujocoMath.Float64(yAfter),
                ["distance_from_origin"] = MujocoMath.Float64(MujocoMath.Norm(xAfter, yAfter)),
                ["x_velocity"] = MujocoMath.Float64(xVelocity),
                ["y_velocity"] = MujocoMath.Float64(yVelocity),
                ["forward_reward"] = MujocoMath.InfoValue(forwardReward),
            };
            if (UseContactForces) {
                NpScalar contactCost = NpScalar.PyFloat(ContactCostWeight)
                                       * NpScalar.Float64(MujocoMath.SumOfClippedSquares(Data.cfrc_ext, ContactForceRange.Low, ContactForceRange.High));
                costs += contactCost;
                // Gymnasium's v4 overwrites the control term with the contact one here; the replaced 0-d array is released
                (info["reward_ctrl"] as IDisposable)?.Dispose();
                info["reward_ctrl"] = MujocoMath.InfoValue(-contactCost);
            }
            NpScalar reward = rewards - costs;
            RenderIfHuman();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
            return new StepResult<NDArray>(observation, reward.Value, terminated, false, info);
        }

        /// <summary>
        ///     The episode's start (<c>reset_model</c>): positions <c>init_qpos + uniform(-scale, scale, nq)</c>, then
        ///     velocities <c>init_qvel + scale * standard_normal(nv)</c>, drawn in that order.
        /// </summary>
        /// <returns>The first observation.</returns>
        protected override NDArray ResetModel() {
            double noiseLow = -ResetNoiseScale, noiseHigh = ResetNoiseScale;
            double[] qpos = AddNoise(InitQpos, UniformNoise(noiseLow, noiseHigh, Model.Nq));
            double[] normal = StandardNormalNoise(Model.Nv);
            for (int i = 0; i < normal.Length; i++)
                normal[i] = ResetNoiseScale * normal[i];
            double[] qvel = AddNoise(InitQvel, normal);
            SetStateValues(qpos, qvel);
            return GetObs();
        }

        /// <summary>
        ///     The observation (<c>_get_obs</c>): the positions (the root's x and y left out unless asked for), the velocities,
        ///     then with <see cref="UseContactForces"/> the clipped contact forces of every body, the world's included.
        /// </summary>
        /// <returns>A new float64 array.</returns>
        private NDArray GetObs() {
            int skipped = ExcludeCurrentPositionsFromObservation ? 2 : 0;
            int positions = Model.Nq - skipped;
            int contacts = UseContactForces ? Data.cfrc_ext.Length : 0;
            var values = new double[positions + Model.Nv + contacts];
            Array.Copy(Data.qpos, skipped, values, 0, positions);
            Array.Copy(Data.qvel, 0, values, positions, Model.Nv);
            for (int i = 0; i < contacts; i++)
                values[positions + Model.Nv + i] = PyMath.Clip(Data.cfrc_ext[i], ContactForceRange.Low, ContactForceRange.High);
            return np.array(values);
        }
    }
}
