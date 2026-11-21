using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.MuJoCo;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     Gymnasium's 22 MuJoCo ids against Gym.NET's (DR-61), live and in lockstep, on MuJoCo 4.0.0 (DR-64): sampled episodes
    ///     of every id, Gymnasium's keyword variants, every form of action NumPy takes or refuses, <c>make</c>'s stacks
    ///     through their time limits and the warnings <c>make</c> issues, the state vectors of
    ///     <c>gymnasium.envs.mujoco.utils</c>, spaces, metadata, and Pusher-v4's refusal.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares, after each call, the float64 observation, the float64 value of the reward, the flags,
    ///     the info with its NumPy types (a 0-d array of Gym.NET's stands for NumPy's scalar of its dtype, DR-08) and the
    ///     generator's state. Gym.NET's engine reproduces MuJoCo's arithmetic on win-x64, the C runtime's
    ///     <c>sin</c>/<c>cos</c>/<c>pow</c> and MuJoCo's AVX kernels included, so every cell that steps is pinned to the
    ///     reference platform. Frames are not compared: Gym.NET's MuJoCo frames are measured, not gated (DR-67), and not
    ///     drawn yet (DR-62). Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/mujoco/mujoco_env.py">gymnasium/envs/mujoco</see>.
    /// </remarks>
    [TestClass]
    public sealed class MujocoLiveParityTests : ParityTestBase {
        /// <summary>The platform qualification of every cell that steps.</summary>
        private const string Platform = "MuJoCo 4.0.0's win-x64 arithmetic (the C runtime's sin/cos/pow, its AVX kernels)";

        /// <summary>
        ///     The 21 ids that run on MuJoCo 3 and later, in Gymnasium's registration order (Pusher-v4 refuses): the one list
        ///     the other classes' MuJoCo rows read (the vector, bridge and registration cells), so an id added here reaches
        ///     every layer. Read-only by convention; never write into it.
        /// </summary>
        internal static readonly string[] Ids = {
            "Reacher-v4", "Reacher-v5", "Pusher-v5", "InvertedPendulum-v4", "InvertedPendulum-v5", "InvertedDoublePendulum-v4",
            "InvertedDoublePendulum-v5", "HalfCheetah-v4", "HalfCheetah-v5", "Hopper-v4", "Hopper-v5", "Swimmer-v4", "Swimmer-v5",
            "Walker2d-v4", "Walker2d-v5", "Ant-v4", "Ant-v5", "Humanoid-v4", "Humanoid-v5", "HumanoidStandup-v4", "HumanoidStandup-v5",
        };

        /// <summary>Every id with two seeds.</summary>
        public static IEnumerable<object[]> IdsAndSeeds => from id in Ids from seed in new[] { 0L, 7L } select new object[] { id, seed };

        /// <summary>Every id.</summary>
        public static IEnumerable<object[]> EveryId => Ids.Select(id => new object[] { id });

        /// <summary>All 22 registered ids, Pusher-v4 in its registration place, for the cells that only call <c>make</c>.</summary>
        public static IEnumerable<object[]> MakeIds => Ids.Take(2).Append("Pusher-v4").Concat(Ids.Skip(2)).Select(id => new object[] { id });

        /// <summary>
        ///     Gymnasium's keyword arguments that change a code path (observation layout, cost, health, model, frame skip,
        ///     body), as JSON objects: numbers with a point are floats, without one ints, two-number lists tuples.
        /// </summary>
        public static IEnumerable<object[]> KeywordVariants => new (string Id, string Kwargs)[] {
            ("HalfCheetah-v5", """{"exclude_current_positions_from_observation": false, "ctrl_cost_weight": 0.3}"""),
            ("HalfCheetah-v5", """{"frame_skip": 3, "forward_reward_weight": 2.0, "reset_noise_scale": 0.0}"""),
            ("HalfCheetah-v4", """{"exclude_current_positions_from_observation": false, "forward_reward_weight": 0.5}"""),
            ("Hopper-v5", """{"terminate_when_unhealthy": false, "healthy_reward": 2.5}"""),
            ("Hopper-v5", """{"healthy_z_range": [0.9, 1.5], "healthy_angle_range": [-0.1, 0.1], "healthy_state_range": [-50.0, 50.0], "frame_skip": 2}"""),
            ("Hopper-v5", """{"exclude_current_positions_from_observation": false, "xml_file": "hopper.xml"}"""),
            ("Hopper-v4", """{"terminate_when_unhealthy": false}"""),
            ("Hopper-v4", """{"healthy_z_range": [0.9, 1.5], "ctrl_cost_weight": 0.01}"""),
            ("Walker2d-v5", """{"xml_file": "walker2d.xml"}"""),
            ("Walker2d-v5", """{"terminate_when_unhealthy": false, "healthy_reward": 0.5}"""),
            ("Walker2d-v5", """{"healthy_z_range": [1.0, 1.8], "exclude_current_positions_from_observation": false}"""),
            ("Walker2d-v4", """{"terminate_when_unhealthy": false}"""),
            ("Swimmer-v5", """{"exclude_current_positions_from_observation": false, "ctrl_cost_weight": 0.01}"""),
            ("Swimmer-v5", """{"frame_skip": 5}"""),
            ("Swimmer-v4", """{"exclude_current_positions_from_observation": false}"""),
            ("Ant-v5", """{"include_cfrc_ext_in_observation": false}"""),
            ("Ant-v5", """{"main_body": "torso"}"""),
            ("Ant-v5", """{"main_body": 3}"""),
            ("Ant-v5", """{"contact_force_range": [-0.5, 0.5], "contact_cost_weight": 0.01, "healthy_z_range": [0.3, 0.9]}"""),
            ("Ant-v5", """{"terminate_when_unhealthy": false, "exclude_current_positions_from_observation": false}"""),
            ("Ant-v5", """{"frame_skip": 1}"""),
            ("Ant-v4", """{"use_contact_forces": true}"""),
            ("Ant-v4", """{"terminate_when_unhealthy": false, "healthy_z_range": [0.3, 0.9]}"""),
            ("Ant-v4", """{"exclude_current_positions_from_observation": false}"""),
            ("Humanoid-v5", """{"include_cinert_in_observation": false, "include_cvel_in_observation": false}"""),
            ("Humanoid-v5", """{"include_qfrc_actuator_in_observation": false, "include_cfrc_ext_in_observation": false}"""),
            ("Humanoid-v5", """{"contact_cost_range": [-1.0, 0.0001], "contact_cost_weight": 0.001}"""),
            ("Humanoid-v5", """{"terminate_when_unhealthy": false, "exclude_current_positions_from_observation": false}"""),
            ("Humanoid-v4", """{"terminate_when_unhealthy": false}"""),
            ("Humanoid-v4", """{"exclude_current_positions_from_observation": false}"""),
            ("HumanoidStandup-v5", """{"impact_cost_range": [-1.0, 0.0], "impact_cost_weight": 0.01}"""),
            ("HumanoidStandup-v5", """{"include_cinert_in_observation": false, "exclude_current_positions_from_observation": false}"""),
            ("InvertedPendulum-v5", """{"reset_noise_scale": 0.1}"""),
            ("InvertedPendulum-v5", """{"frame_skip": 5}"""),
            ("InvertedDoublePendulum-v5", """{"healthy_reward": 3.0, "reset_noise_scale": 0.3}"""),
            ("Pusher-v5", """{"xml_file": "pusher.xml"}"""),
            ("Pusher-v5", """{"reward_near_weight": 1.5, "reward_dist_weight": 0.5, "reward_control_weight": 0.01}"""),
            ("Reacher-v5", """{"reward_dist_weight": 2.0, "reward_control_weight": 0.5}"""),
            ("Reacher-v5", """{"frame_skip": 4}"""),
        }.Select(row => new object[] { row.Id, row.Kwargs });

        /// <summary>The ids whose control costs read the action itself, for the action-form cells.</summary>
        public static IEnumerable<object[]> ActionFormIds => new[] {
            "HalfCheetah-v5", "Hopper-v4", "Ant-v5", "Reacher-v4", "Reacher-v5", "Pusher-v5", "Swimmer-v4", "Walker2d-v5", "Humanoid-v5",
            "InvertedPendulum-v5", "HumanoidStandup-v4",
        }.Select(id => new object[] { id });

        /// <summary>
        ///     Three sampled episodes of an id, bare (Gymnasium's <c>.unwrapped</c> against Gym.NET's class): the first from
        ///     <paramref name="seed"/>, then two more in the same env, through terminations, each reset compared with its
        ///     generator.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="seed">The first reset's seed.</param>
        [TestMethod]
        [DynamicData(nameof(IdsAndSeeds))]
        public void GymNet_SampledEpisodes_MatchGymnasium(string id, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(id, new Dictionary<string, object?>());
            PairRun first = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 300);
            PairRun second = pair.Run(seed + 100, ActionSources.Sampled(seed + 100), maxSteps: 200);
            PairRun third = pair.Run(seed + 200, ActionSources.Sampled(seed + 200), maxSteps: 100);
            TestContext.WriteLine(first.ToString());
            TestContext.WriteLine(second.ToString());
            TestContext.WriteLine(third.ToString());
        }

        /// <summary>
        ///     An id under one of Gymnasium's keyword variants: two sampled episodes, bare, both sides built with the same
        ///     arguments.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="kwargs">The keyword arguments as a JSON object.</param>
        [TestMethod]
        [DynamicData(nameof(KeywordVariants))]
        public void GymNet_KeywordVariants_MatchGymnasium(string id, string kwargs) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(id, Kwargs(kwargs));
            TestContext.WriteLine(pair.Run(3, ActionSources.Sampled(3), maxSteps: 300).ToString());
            TestContext.WriteLine(pair.Run(4, ActionSources.Sampled(4), maxSteps: 150).ToString());
        }

        /// <summary>
        ///     Every form of action NumPy takes, each followed by a float32 step: float32, float64, int64, int32, int8 (whose
        ///     squares wrap in the control cost), uint8, bool, a NaN (MuJoCo zeroes the controls and warns), and the refused
        ///     ones with Gymnasium's <c>ValueError</c>: a 0-d action, one actuator too many, a row.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DynamicData(nameof(ActionFormIds))]
        public void GymNet_ActionForms_MatchGymnasium(string id) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(id, new Dictionary<string, object?>());
            pair.Reset(21);
            int nu = (int)((Box)((GymNetEnv)pair.DotNet).Env.ActionSpace).Shape.size;
            double[] values = Enumerable.Range(0, nu).Select(i => ((i * 3) % 5 - 2) / 2.5).ToArray();
            var actions = new (string What, Func<NDArray> Make)[] {
                ("float32", () => np.array(values.Select(v => (float)v).ToArray())),
                ("float64", () => np.array(values.Select(v => v * 0.7).ToArray())),
                ("int64", () => np.array(values.Select(v => (long)Math.Round(v * 2.5)).ToArray())),
                ("int32", () => np.array(values.Select(v => (int)Math.Round(v * 2.5)).ToArray())),
                ("int8, squares wrapping", () => np.array(values.Select(v => (sbyte)Math.Round(v * 50)).ToArray())),
                ("uint8", () => np.array(values.Select(v => (byte)Math.Round(v * 2.5 + 2)).ToArray())),
                ("bool", () => np.array(values.Select(v => v > 0).ToArray())),
                ("float64 with a NaN", () => np.array(values.Select((v, i) => i == 0 ? double.NaN : v).ToArray())),
                ("0-d float64", () => NDArray.Scalar<double>(0.5)),
                ("one too many", () => np.array(Enumerable.Repeat(0.25f, nu + 1).ToArray())),
                ("a row", () => np.full(new Shape(1, nu), 0.25f, np.float32)),
            };
            foreach ((string what, Func<NDArray> make) in actions) {
                using (ParityAction action = ParityAction.Continuous(make()))
                    pair.Step(action);
                using ParityAction follow = ParityAction.Continuous(np.array(values.Select(v => (float)(v * -0.5)).ToArray()));
                pair.Step(follow);
                TestContext.WriteLine($"{id} {what}: compared, and the step after it");
            }
        }

        /// <summary>
        ///     <c>make</c>'s whole stack of an id (<c>TimeLimit</c>, <c>OrderEnforcing</c>, <c>PassiveEnvChecker</c>) against
        ///     <c>Gymnasium.Make</c>: a sampled episode to its end, the time limit's truncation (1,000 steps, Reacher's 50,
        ///     Pusher's 100) or the env's termination. The warnings <c>make</c> issues are
        ///     <see cref="GymNet_MakeWarnings_MatchGymnasium"/>'s: the pair compares calls, not warnings.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DynamicData(nameof(EveryId))]
        public void GymNet_MakeStack_MatchesGymnasium(string id) {
            SkipOffReferencePlatform(Platform);
            using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true), new GymNetEnv(Gymnasium.Make(id)));
            PairRun run = pair.Run(31, ActionSources.Sampled(31), maxSteps: 1005);
            Assert.IsTrue(run.Ended, $"The episode must end: {run}");
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     The state vectors of <c>gymnasium.envs.mujoco.utils.get_state</c> after a seeded reset and 25 sampled steps, for
        ///     <c>mjSTATE_FULLPHYSICS</c> (time, positions, velocities) and <c>mjSTATE_INTEGRATION</c> (everything a step
        ///     reads: the warm start and the controls too).
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DynamicData(nameof(EveryId))]
        public void GymNet_StateVectors_MatchGymnasium(string id) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(id, new Dictionary<string, object?>());
            pair.Run(5, ActionSources.Sampled(5), maxSteps: 25);
            IEnv dotnet = ((GymNetEnv)pair.DotNet).Env;
            foreach (MjtState state in new[] { MjtState.FullPhysics, MjtState.Integration }) {
                Outcome python = Outcome.CapturePython(() => Python.gymnasium.envs.mujoco.utils.get_state(pair.Python.Stepped, (int)state));
                ByteContract.AssertSame($"{id} get_state({state})", python, Outcome.Capture(() => MujocoUtils.GetState(dotnet, state)));
            }
        }

        /// <summary>The observation space byte for byte, and the action space with 20 samples after <c>seed(5)</c>.</summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DynamicData(nameof(EveryId))]
        public void GymNet_Spaces_MatchGymnasium(string id) {
            using IEnv made = Gymnasium.Make(id);
            ClassicSpaces.AssertBoxSame($"{id} observation_space", id, Python.gymnasium.Env.observation_space, (Box)made.Unwrapped.ObservationSpace);
            ClassicSpaces.AssertActionSpaceSame(id, made.Unwrapped.ActionSpace);
        }

        /// <summary>The instance's <c>metadata</c>: the four render modes and the frame rate (v5's derived from the time step).</summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DynamicData(nameof(EveryId))]
        public void GymNet_Metadata_MatchesGymnasium(string id) {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            using IEnv made = Gymnasium.Make(id);
            ByteContract.AssertSame($"{id} metadata", python, Outcome.Returned(ValueFacts.Of(made.Unwrapped.Metadata)));
        }

        /// <summary>
        ///     The warnings <c>make</c> issues for an id, each with its category and text, every warning recorded
        ///     (<c>simplefilter('always')</c> on Gymnasium's side, <see cref="Warnings.Catch"/> on Gym.NET's): the v4 ids'
        ///     out-of-date notice (<c>gymnasium.logger.deprecation</c>, colourised, a <c>DeprecationWarning</c>), none for
        ///     the v5 ids, and Pusher-v4's notice, which comes before its refusal (the version check runs before the env is
        ///     built, and the refusal itself is <see cref="GymNet_PusherV4_RefusesLikeGymnasium"/>'s).
        /// </summary>
        /// <param name="id">The id.</param>
        /// <exception cref="AssertFailedException">Thrown when the categories, texts, order or count differ.</exception>
        [TestMethod]
        [DynamicData(nameof(MakeIds))]
        public void GymNet_MakeWarnings_MatchGymnasium(string id) {
            // Pusher-v4's ImportError is swallowed on both sides: what is compared here is what was warned before it.
            string statements =
                "import warnings\n" +
                "import gymnasium\n" +
                "with warnings.catch_warnings(record=True) as caught:\n" +
                "    warnings.simplefilter('always')\n" +
                "    try:\n" +
                $"        gymnasium.make('{id}').close()\n" +
                "    except ImportError:\n" +
                "        pass\n" +
                "recorded = [(w.category.__name__, str(w.message)) for w in caught]\n";
            ValueFacts pythonWarnings = Outcome.CapturePython(() => Python.evalStmts(statements, "recorded")).Expect($"warnings of make('{id}')");
            IReadOnlyList<WarningMessage> records;
            using (WarningCatcher catcher = Warnings.Catch()) {
                try {
                    Gymnasium.Make(id).Dispose();
                } catch (NumSharp.ImportError) {
                    // Pusher-v4 refuses in its constructor, after make's version check has warned.
                }

                records = catcher.Records;
            }

            ValueFacts dotnetWarnings = ValueFacts.FromSequence(records.Select(w => ValueFacts.FromSequence(new[] { ValueFacts.FromText(w.Category.ToString()), ValueFacts.FromText(w.Message) }, "tuple")), "list");
            ByteContract.AssertSame($"warnings of make('{id}')", pythonWarnings, dotnetWarnings);
            TestContext.WriteLine($"make('{id}'): {records.Count} warning(s)");
        }

        /// <summary>
        ///     <c>make("Pusher-v4")</c>: Gymnasium's <c>ImportError</c> on MuJoCo 3 and later, with its message, from both
        ///     registries.
        /// </summary>
        [TestMethod]
        public void GymNet_PusherV4_RefusesLikeGymnasium() {
            Outcome python = Outcome.CapturePython(() => Python.gymnasium.make("Pusher-v4"));
            ByteContract.AssertSame("make('Pusher-v4')", python, Outcome.Capture(() => Gymnasium.Make("Pusher-v4")));
        }

        /// <summary>A pair of the bare envs: Gymnasium's <c>.unwrapped</c> and Gym.NET's registered class, both with <paramref name="kwargs"/>.</summary>
        /// <param name="id">The id.</param>
        /// <param name="kwargs">The keyword arguments in Python spelling.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(string id, Dictionary<string, object?> kwargs) =>
            new(PyGymEnv.Make(id, kwargs), new GymNetEnv((IEnv)PyKwargs.Construct((Type)Gymnasium.Spec(id).EntryPoint!, kwargs)));

        /// <summary>A JSON object as keyword arguments: integers as <see langword="long"/>, decimals as <see langword="double"/>, two-number lists as tuples.</summary>
        /// <param name="json">The object.</param>
        /// <returns>The arguments by Python name.</returns>
        /// <exception cref="NotSupportedException">A value of another JSON kind.</exception>
        private static Dictionary<string, object?> Kwargs(string json) {
            using JsonDocument document = JsonDocument.Parse(json);
            var kwargs = new Dictionary<string, object?>();
            foreach (JsonProperty p in document.RootElement.EnumerateObject()) {
                kwargs[p.Name] = p.Value.ValueKind switch {
                    JsonValueKind.Number when p.Value.GetRawText().IndexOfAny(['.', 'e', 'E']) >= 0 => p.Value.GetDouble(),
                    JsonValueKind.Number => p.Value.GetInt64(),
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    JsonValueKind.String => p.Value.GetString(),
                    JsonValueKind.Array => (p.Value[0].GetDouble(), p.Value[1].GetDouble()),
                    _ => throw new NotSupportedException($"kwarg {p.Name}: {p.Value.ValueKind}"),
                };
            }
            return kwargs;
        }
    }
}
