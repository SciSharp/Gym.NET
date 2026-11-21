using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Spaces;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;
using Color = Gym.Rendering.Color;

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     BipedalWalker-v3 and BipedalWalkerHardcore-v3 against Gymnasium's <c>BipedalWalker</c>, live and in lockstep with
    ///     Gym.NET's <see cref="BipedalWalkerEnv"/> on its managed Box2D: resets and the terrain they generate (grass, stumps,
    ///     stairs, pits, clouds), whole episodes of Gymnasium's gait and of sampled actions, every form of action NumPy takes
    ///     or refuses, <c>rgb_array</c> frames of the scrolling world with its lidar sweep, <c>BipedalWalkerHeuristics</c>
    ///     itself, spaces, metadata and <c>make</c>'s stacks.
    /// </summary>
    /// <remarks>
    ///     Every lockstep compares, after each call, the float32 observation (lidar fractions included), the float64 value of
    ///     the reward (float32 for the float32 actions sampled, float64 for the gait's, as NumPy 2 computes it), the flags,
    ///     the info and the generator's state. The env keeps one Box2D world for its life, so episodes after the first run in
    ///     a world whose broad phase reuses freed nodes; the cells that reset several times pin that too. Cells that step or
    ///     draw are pinned to the reference platform: Box2D's rotations call the C runtime's <c>sinf</c>/<c>cosf</c>, the
    ///     clouds and the lidar <c>sin</c>/<c>cos</c>. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L103">gymnasium/envs/box2d/bipedal_walker.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class BipedalWalkerLiveParityTests : ParityTestBase {
        /// <summary>The id; the hardcore variant is the same class with <c>hardcore=True</c>.</summary>
        private const string Id = "BipedalWalker-v3";

        /// <summary>The platform qualification of every cell that steps or draws.</summary>
        private const string Platform = "Box2D's sinf/cosf and the env's sin/cos";

        /// <summary>
        ///     Gymnasium's side of <see cref="GymNet_Terrain_MatchesGymnasium"/>: the terrain, clouds and bodies a reset builds,
        ///     every float as a Python float (the ground polygons' zeros are ints in Python, and some heights NumPy floats).
        /// </summary>
        private const string TerrainStatements = """
            from gymnasium.envs.box2d.bipedal_walker import BipedalWalker
            env = BipedalWalker(hardcore=hardcore)
            env.reset(seed=seed)
            def body(b):
                return (float(b.position[0]), float(b.position[1]), float(b.angle))
            result = [
                [float(x) for x in env.terrain_x],
                [float(y) for y in env.terrain_y],
                [([(float(p[0]), float(p[1])) for p in poly], color) for poly, color in env.terrain_poly],
                [([(float(p[0]), float(p[1])) for p in poly], float(x1), float(x2)) for poly, x1, x2 in env.cloud_poly],
                [(t.color1, t.color2, [[(float(v[0]), float(v[1])) for v in f.shape.vertices] for f in t.fixtures]) for t in env.terrain],
                [body(leg) for leg in env.legs] + [body(env.hull)],
            ]
            env.close()
            """;

        /// <summary>
        ///     Gymnasium's side of <see cref="GymNet_HeuristicGait_MatchesGymnasium"/>: its env walked by its own
        ///     <c>BipedalWalkerHeuristics</c>, recording each action (a copy: the next call overwrites the array in place) and
        ///     the gait's state after each call.
        /// </summary>
        private const string GaitStatements = """
            from gymnasium.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHeuristics
            env = BipedalWalker(hardcore=hardcore)
            s, _ = env.reset(seed=seed)
            h = BipedalWalkerHeuristics()
            actions = []
            states = []
            for _ in range(steps):
                a = h.step_heuristic(s)
                actions.append(a.copy())
                states.append((h.state, h.moving_leg, h.supporting_leg, float(h.supporting_knee_angle)))
                s, r, terminated, truncated, info = env.step(a)
                if terminated:
                    break
            env.close()
            result = [actions, states]
            """;

        /// <summary>
        ///     Gymnasium's side of <see cref="GymNet_HeuristicOnObservations_MatchesGymnasium"/>: one gait fed a list of
        ///     observations in order, recording each action and the state after each call.
        /// </summary>
        private const string ObservationGaitStatements = """
            from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics
            h = BipedalWalkerHeuristics()
            actions = []
            states = []
            for s in observations:
                actions.append(h.step_heuristic(s).copy())
                states.append((h.state, h.moving_leg, h.supporting_leg, float(h.supporting_knee_angle)))
            result = [actions, states]
            """;

        /// <summary>
        ///     <c>reset(seed=seed)</c> of both variants: the zero-action step's observation, the info and the generator after the
        ///     terrain's and clouds' draws (the hardcore ones depend on the obstacles drawn).
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 1L)]
        [DataRow(false, 42L)]
        [DataRow(true, 0L)]
        [DataRow(true, 3L)]
        [DataRow(true, 1234L)]
        public void GymNet_Reset_MatchesGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(hardcore);
            pair.Reset(seed);
        }

        /// <summary>
        ///     The world a reset builds, value for value: the 200 terrain heights, the ground polygons, the ten clouds, every
        ///     terrain body's colours and fixture vertices (Box2D's hull order of the obstacles' polygons), and the hull's and
        ///     legs' poses after the reset's step; five hardcore seeds cover stumps, stairs up and down, and pits.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 7L)]
        [DataRow(true, 0L)]
        [DataRow(true, 1L)]
        [DataRow(true, 2L)]
        [DataRow(true, 3L)]
        [DataRow(true, 4L)]
        public void GymNet_Terrain_MatchesGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            ValueFacts python = Outcome.CapturePython(() => {
                using PyObject flag = hardcore.ToPython();
                using var seedValue = new PyInt(seed);
                return Python.evalStmts(TerrainStatements, "result", ("hardcore", flag), ("seed", seedValue));
            }).Expect($"BipedalWalker(hardcore={hardcore}).reset(seed={seed})'s world");

            ValueFacts dotnet = Outcome.Capture(() => {
                using var env = new BipedalWalkerEnv(hardcore: hardcore);
                env.Reset(seed).Observation.Dispose();
                static (double, double, double) Body(Gym.Box2D.b2Body b) => (b.Position.x, b.Position.y, b.Angle);
                static (long, long, long) Rgb(Color c) => (c.R, c.G, c.B);
                IReadOnlyList<(Color Color1, Color Color2)> colors = env.TerrainColors;
                IReadOnlyList<Gym.Box2D.b2Body> terrain = env.Terrain;
                return new object?[] {
                    env.TerrainX.ToList(),
                    env.TerrainY.ToList(),
                    env.TerrainPoly.Select(p => (object)(p.Poly.ToList(), Rgb(p.Color))).ToList(),
                    env.CloudPoly.Select(c => (object)(c.Poly.ToList(), c.X1, c.X2)).ToList(),
                    terrain.Select((t, i) => (object)(Rgb(colors[i].Color1), Rgb(colors[i].Color2), t.Fixtures.Select(f => Vertices(f.Shape)).ToList())).ToList(),
                    env.Legs.Select(Body).Append(Body(env.Hull!)).ToList(),
                };
            }).Expect($"BipedalWalkerEnv(hardcore: {hardcore}).Reset({seed})'s world");

            ByteContract.AssertSame($"BipedalWalker(hardcore={hardcore}) world after reset(seed={seed})", python, dotnet);
        }

        /// <summary>
        ///     Whole episodes of Gymnasium's own gait (<c>BipedalWalkerHeuristics</c> on Gymnasium's observations, one per
        ///     episode, fed to both sides): the normal variant walks for 1,600 steps, the hardcore one until it falls at an
        ///     obstacle (-100 as an int).
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 42L)]
        [DataRow(true, 0L)]
        [DataRow(true, 5L)]
        public void GymNet_HeuristicEpisodes_MatchGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(hardcore);
            PairRun run = pair.Run(seed, ActionSources.Policy("bipedal_walker_heuristic"), maxSteps: hardcore ? 2000 : 1600);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Episodes of seeded sampled actions (both action spaces sampled and compared): float32 commands, so float32
        ///     rewards, until the walker falls or 500 steps; then a second episode in the same world after a reset.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 3L)]
        [DataRow(true, 0L)]
        [DataRow(true, 9L)]
        public void GymNet_SampledEpisodes_MatchGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(hardcore);
            PairRun first = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 500);
            PairRun second = pair.Run(seed + 100, ActionSources.Sampled(seed + 100), maxSteps: 300);
            TestContext.WriteLine($"{first}; then in the same world: {second}");
        }

        /// <summary>
        ///     Episodes after episodes reset with the same seed, as Gymnasium's <c>check_environments_match</c> resets after
        ///     each end: the same terrain rebuilt in the world the last episode's bodies left, whose broad phase hands the
        ///     freed tree nodes out again, so the contacts' creation order (and with it the order of the contact listener's
        ///     events) depends on the destruction order.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed of every episode, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 123L)]
        [DataRow(true, 0L)]
        [DataRow(true, 123L)]
        public void GymNet_SameSeedEpisodes_MatchGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(hardcore);
            var runs = new List<string>();
            for (int episode = 0; episode < 4; episode++)
                runs.Add(pair.Run(seed, ActionSources.Sampled(seed + episode), maxSteps: 250).ToString());
            TestContext.WriteLine(string.Join("; then ", runs));
        }

        /// <summary>
        ///     The loop of Gymnasium's <c>check_environments_match</c> in lockstep, twice on the same envs as the bridge cell
        ///     runs it: <c>make</c>'s stacks in <c>rgb_array</c> mode, the action space seeded once per call and sampled across
        ///     episodes, a frame after every call, and a reset with the same seed after every end.
        /// </summary>
        /// <param name="id">The id.</param>
        /// <param name="seed">The seed of the calls, as the tool takes it.</param>
        [TestMethod]
        [DataRow("BipedalWalker-v3", 0L)]
        [DataRow("BipedalWalker-v3", 123L)]
        [DataRow("BipedalWalkerHardcore-v3", 0L)]
        [DataRow("BipedalWalkerHardcore-v3", 123L)]
        public void GymNet_EnvironmentsMatchLoop_MatchesGymnasium(string id, long seed) {
            SkipOffReferencePlatform(Platform);
            using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true, renderMode: "rgb_array"), new GymNetEnv(Gymnasium.Make(id, new { render_mode = "rgb_array" })));
            int resets = 0;
            for (int call = 0; call < 2; call++) {
                IActionSource actions = ActionSources.Sampled(seed);
                actions.Start(pair);
                pair.Reset(seed);
                pair.CompareFrame($"call {call} reset");
                for (int step = 0; step < 200; step++) {
                    CallResult result;
                    using (ParityAction action = actions.Next(pair))
                        result = pair.Step(action);
                    pair.CompareFrame($"call {call} step {step}");
                    if (result.Transition is { } t && (t.Terminated?.Boolean == true || t.Truncated?.Boolean == true)) {
                        pair.Reset(seed);
                        resets++;
                    }
                }
            }

            TestContext.WriteLine($"2 calls of 200 steps, {resets} resets after episode ends, identical");
        }

        /// <summary>
        ///     Frames along the gait's episodes, every seventh step: the sky, the clouds scrolling at half speed, the ground
        ///     polygons, the lidar ray of the frame (one per frame, sweeping out and back), the terrain lines and obstacles, the
        ///     legs and hull, and the start flag, on a surface widening with the scroll and cropped to its right 600 columns.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(true, 3L)]
        public void GymNet_FramesAlongEpisodes_MatchGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(hardcore, "rgb_array");
            PairRun run = pair.Run(seed, ActionSources.Policy("bipedal_walker_heuristic"), maxSteps: 700, frameEvery: 7);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Frames along sampled episodes, every other step: the lidar sweep's every ray, a walker that falls backwards
        ///     (the scroll then goes negative and the frame stops widening), and a crash.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void GymNet_FramesOfSampledEpisodes_MatchGymnasium(bool hardcore) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(hardcore, "rgb_array");
            PairRun run = pair.Run(13, ActionSources.Sampled(13), maxSteps: 240, frameEvery: 2);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Actions of every form, each followed by an ordinary step so their effect is compared too: float32 and float64
        ///     commands inside and beyond ±1 (the energy cost in each precision), int64, int8 with its minimum (which
        ///     <c>np.abs</c> leaves negative, so no torque), uint8, NaN (it propagates into the physics), five elements (the
        ///     fifth costs energy only); and the refused ones with NumPy's messages: a 0-d array, two elements (after the first
        ///     two joints took their settings), a row, booleans.
        /// </summary>
        [TestMethod]
        public void GymNet_ActionForms_MatchGymnasium() {
            SkipOffReferencePlatform(Platform);
            var actions = new (string What, Func<NDArray> Make)[] {
                ("float32 [0.7, -0.8, 0.2, -0.05]", () => np.array(new[] { 0.7f, -0.8f, 0.2f, -0.05f })),
                ("float64 [2.5, -3.0, 0.3, 0.9]", () => np.array(new[] { 2.5, -3.0, 0.3, 0.9 })),
                ("int64 [1, -1, 0, 2]", () => np.array(new[] { 1L, -1L, 0L, 2L })),
                ("int8 [-128, 1, -1, 0]", () => np.array(new sbyte[] { -128, 1, -1, 0 })),
                ("uint8 [255, 0, 1, 2]", () => np.array(new byte[] { 255, 0, 1, 2 })),
                ("float32 [0.3, 0.9, -0.4, 0.1, 7.0]", () => np.array(new[] { 0.3f, 0.9f, -0.4f, 0.1f, 7.0f })),
                ("0-d float64 0.5", () => NDArray.Scalar<double>(0.5)),
                ("float32 [0.5, -0.5]", () => np.array(new[] { 0.5f, -0.5f })),
                ("float32 [[0.1, 0.2, 0.3, 0.4]]", () => np.array(new[,] { { 0.1f, 0.2f, 0.3f, 0.4f } })),
                ("bool [True, False, True, False]", () => np.array(new[] { true, false, true, false })),
                ("float64 [nan, 0.5, -0.5, 0.25]", () => np.array(new[] { double.NaN, 0.5, -0.5, 0.25 })),
            };
            using GymPair pair = Pair(false);
            pair.Reset(21);
            foreach ((string what, Func<NDArray> make) in actions) {
                using (ParityAction action = ParityAction.Continuous(make()))
                    pair.Step(action);
                using ParityAction follow = ParityAction.Continuous(np.array(new[] { 0.4f, -0.6f, 0.2f, 0.1f }));
                pair.Step(follow);
                TestContext.WriteLine($"{what}: compared, and the step after it");
            }
        }

        /// <summary>
        ///     A step before any reset: Gymnasium's bare <c>assert self.hull is not None</c>, whose empty message NumSharp's
        ///     <c>AssertionError()</c> does not give yet (N-1).
        /// </summary>
        [TestMethod]
        public void GymNet_StepBeforeReset_RaisesLikeGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            IEnv dotnet = new BipedalWalkerEnv();
            using NDArray zeros = np.zeros(4);
            Outcome pythonStep = Outcome.CapturePython(() => {
                using PyObject action = zeros.ToNumpyCopy();
                return Python.gymnasium.Env.step(python.Object, action);
            });
            ByteContract.AssertSame("step(zeros(4)) before reset", pythonStep, Outcome.Capture(() => dotnet.Step(zeros)));
        }

        /// <summary><c>render()</c> of an <c>rgb_array</c> env before any reset: the <c>AttributeError</c> for the missing <c>scroll</c>, both variants.</summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_RaisesLikeGymnasium() {
            foreach (bool hardcore in new[] { false, true }) {
                using PyGymEnv python = PyGymEnv.Make(Id, Kwargs(hardcore), renderMode: "rgb_array");
                var dotnet = new BipedalWalkerEnv(renderMode: "rgb_array", hardcore: hardcore);
                RenderOutcome.AssertSame($"render() before reset (hardcore={hardcore})", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>
        ///     <see cref="BipedalWalkerHeuristics"/> walking Gym.NET's env against Gymnasium's gait walking Gymnasium's, each on
        ///     its own side: every action (float64, from float32 arithmetic on the observations) and the gait's state after
        ///     every call, through hundreds of phase changes and the knee target's float/float32 switches.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(false, 0L)]
        [DataRow(false, 11L)]
        [DataRow(true, 2L)]
        public void GymNet_HeuristicGait_MatchesGymnasium(bool hardcore, long seed) {
            SkipOffReferencePlatform(Platform);
            const int steps = 1200;
            ValueFacts python = Outcome.CapturePython(() => {
                using PyObject flag = hardcore.ToPython();
                using var seedValue = new PyInt(seed);
                using var stepsValue = new PyInt(steps);
                return Python.evalStmts(GaitStatements, "result", ("hardcore", flag), ("seed", seedValue), ("steps", stepsValue));
            }).Expect($"Gymnasium's gait on BipedalWalker(hardcore={hardcore}), seed {seed}");

            ValueFacts dotnet = Outcome.Capture(() => {
                using var env = new BipedalWalkerEnv(hardcore: hardcore);
                var heuristics = new BipedalWalkerHeuristics();
                var actions = new List<object?>();
                var states = new List<object?>();
                NDArray s = env.Reset(seed).Observation;
                try {
                    for (int i = 0; i < steps; i++) {
                        NDArray a = heuristics.StepHeuristic(s);
                        actions.Add(a);
                        states.Add(((long)heuristics.State, (long)heuristics.MovingLeg, (long)heuristics.SupportingLeg, heuristics.SupportingKneeAngle));
                        var result = env.Step(a);
                        s.Dispose();
                        s = result.Observation;
                        if (result.Terminated)
                            break;
                    }
                } finally {
                    s.Dispose();
                }

                return new object?[] { actions, states };
            }).Expect($"Gym.NET's gait on BipedalWalkerEnv(hardcore: {hardcore}), seed {seed}");

            ByteContract.AssertSame($"the gait's actions and states (hardcore={hardcore}, seed {seed})", python, dotnet);
            TestContext.WriteLine($"{python.Items[0].Items.Count} actions, identical");
        }

        /// <summary>
        ///     <see cref="BipedalWalkerHeuristics"/> against Gymnasium's on 1,500 random float32 observations in a row: angles
        ///     and speeds around every threshold of the state machine, contacts on and off, and NaNs (truthy in Python), so the
        ///     phases change often and the knee target alternates between a Python float and a float32.
        /// </summary>
        [TestMethod]
        public void GymNet_HeuristicOnObservations_MatchesGymnasium() {
            var rng = new Random(20261111);
            var observations = new List<NDArray>();
            try {
                for (int i = 0; i < 1500; i++) {
                    float[] values = new float[24];
                    for (int k = 0; k < values.Length; k++)
                        values[k] = (float)((rng.NextDouble() * 2 - 1) * 1.5);
                    // Speeds near SPEED (0.29) and 1.2 * SPEED, the supporting hip near 0.10, the knees near 0.88 - 1.
                    values[2] = (float)(rng.NextDouble() * 0.5);
                    values[8] = rng.Next(3) == 0 ? 1f : 0f;
                    values[13] = rng.Next(3) == 0 ? 1f : 0f;
                    if (i % 101 == 7)
                        values[rng.Next(14)] = float.NaN;
                    observations.Add(np.array(values));
                }

                ValueFacts python = Outcome.CapturePython(() => {
                    using var list = new PyList();
                    foreach (NDArray observation in observations) {
                        using PyObject item = observation.ToNumpyCopy();
                        list.Append(item);
                    }

                    return Python.evalStmts(ObservationGaitStatements, "result", ("observations", list));
                }).Expect("Gymnasium's gait on random observations");

                ValueFacts dotnet = Outcome.Capture(() => {
                    var heuristics = new BipedalWalkerHeuristics();
                    var actions = new List<object?>();
                    var states = new List<object?>();
                    foreach (NDArray observation in observations) {
                        actions.Add(heuristics.StepHeuristic(observation));
                        states.Add(((long)heuristics.State, (long)heuristics.MovingLeg, (long)heuristics.SupportingLeg, heuristics.SupportingKneeAngle));
                    }

                    return new object?[] { actions, states };
                }).Expect("Gym.NET's gait on random observations");

                ByteContract.AssertSame("the gait on random observations", python, dotnet);
            } finally {
                foreach (NDArray observation in observations)
                    observation.Dispose();
            }
        }

        /// <summary>
        ///     The spaces: the 24 observation bounds (±π angles, ±5 speeds, contacts in [-0, 5], lidar in ±1, float32) and
        ///     <c>Box(-1, 1, (4,), float32)</c> with seeded samples, both variants.
        /// </summary>
        /// <param name="hardcore">Which variant.</param>
        [TestMethod]
        [DataRow(false)]
        [DataRow(true)]
        public void GymNet_Spaces_MatchGymnasium(bool hardcore) {
            var env = new BipedalWalkerEnv(hardcore: hardcore);
            Dictionary<string, object?> kwargs = Kwargs(hardcore);
            ClassicSpaces.AssertBoxSame($"{Id}(hardcore={hardcore}) observation_space", Id, Python.gymnasium.Env.observation_space, (Box)env.ObservationSpace, kwargs);
            ClassicSpaces.AssertActionSpaceSame(Id, env.ActionSpace, kwargs);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 50}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(Id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame($"{Id} metadata", python, Outcome.Returned(ValueFacts.Of(new BipedalWalkerEnv().Metadata)));
        }

        /// <summary>
        ///     <c>make</c>'s whole stack of both ids (<c>TimeLimit</c> at 1,600 and 2,000 steps, <c>OrderEnforcing</c>,
        ///     <c>PassiveEnvChecker</c>) against <c>Gymnasium.Make</c>: the gait's episodes to the time limit or the fall.
        /// </summary>
        /// <param name="id">The id.</param>
        [TestMethod]
        [DataRow("BipedalWalker-v3")]
        [DataRow("BipedalWalkerHardcore-v3")]
        public void GymNet_MakeStack_MatchesGymnasium(string id) {
            SkipOffReferencePlatform(Platform);
            using var pair = new GymPair(PyGymEnv.Make(id, viaMake: true), new GymNetEnv(Gymnasium.Make(id)));
            PairRun run = pair.Run(31, ActionSources.Policy("bipedal_walker_heuristic"), maxSteps: 2100);
            Assert.IsTrue(run.Ended, $"The episode must end: {run}");
        }

        /// <summary><c>BipedalWalkerHardcore()</c>: the class only refuses, with Gymnasium's <c>gymnasium.error.Error</c> and its message.</summary>
        [TestMethod]
        public void GymNet_BipedalWalkerHardcoreClass_RefusesLikeGymnasium() {
            Outcome python = Outcome.CapturePython(() => Python.evalStmts(
                "from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHardcore\n" +
                "BipedalWalkerHardcore()\n" +
                "result = None\n", "result"));
            ByteContract.AssertSame("BipedalWalkerHardcore()", python, Outcome.Capture(() => {
                _ = new BipedalWalkerHardcoreEnv();
                return null;
            }));
        }

        /// <summary>A fixture shape's vertices as pybox2d's <c>shape.vertices</c> lists them: a polygon's in hull order, an edge's two.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The vertices as float pairs.</returns>
        /// <exception cref="InvalidOperationException">Thrown for another kind of shape, which BipedalWalker's terrain never has.</exception>
        private static List<(double, double)> Vertices(Gym.Box2D.b2Shape shape) => shape switch {
            Gym.Box2D.b2EdgeShape edge => new List<(double, double)> { (edge.Vertex1.x, edge.Vertex1.y), (edge.Vertex2.x, edge.Vertex2.y) },
            Gym.Box2D.b2PolygonShape polygon => polygon.Vertices.Select(v => ((double)v.x, (double)v.y)).ToList(),
            _ => throw new InvalidOperationException($"BipedalWalker's terrain has no {shape.GetType().Name}."),
        };

        /// <summary>The kwargs of a variant, as <c>make</c> takes them.</summary>
        /// <param name="hardcore">Which variant.</param>
        /// <returns>The kwargs.</returns>
        private static Dictionary<string, object?> Kwargs(bool hardcore) => new() { ["hardcore"] = hardcore };

        /// <summary>A lockstep pair of Gymnasium's unwrapped env and Gym.NET's, built with the same arguments.</summary>
        /// <param name="hardcore">Which variant.</param>
        /// <param name="renderMode">The render mode of both, or <see langword="null"/>.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(bool hardcore, string? renderMode = null) =>
            new(PyGymEnv.Make(Id, Kwargs(hardcore), renderMode: renderMode), new GymNetEnv(new BipedalWalkerEnv(renderMode: renderMode, hardcore: hardcore)));

        /// <summary>Gym.NET's <c>render()</c> of an <c>rgb_array</c> env as a frame, or <see langword="null"/> when it returned <c>None</c>.</summary>
        /// <param name="env">The env, built with <c>render_mode="rgb_array"</c>.</param>
        /// <returns>The frame, or <see langword="null"/>.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the rendering is neither a frame nor <c>None</c>.</exception>
        private static Frame? RenderGymNet(IEnv env) {
            switch (env.Render()) {
                case RenderResult.NoneResult:
                    return null;
                case RenderResult.Rgb rgb: {
                    using NDArray frame = rgb.Frame;
                    return new Frame((int)frame.shape[0], (int)frame.shape[1], frame.ToArray<byte>());
                }
                case var other:
                    throw new InvalidOperationException($"{env} in rgb_array mode rendered {other}.");
            }
        }
    }
}
