using System;
using System.Collections.Generic;
using System.IO;
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

namespace Gym.Tests.Parity.Tests.Envs {
    /// <summary>
    ///     CarRacing-v3 against Gymnasium's <c>CarRacing</c>, live and in lockstep with Gym.NET's <see cref="CarRacingEnv"/> on
    ///     its managed Box2D: the score's embedded glyph data against a live recording, its text against pygame's own renders
    ///     and its box for every kind of value in both frame kinds, resets and the tracks they generate, episodes of sampled
    ///     actions and of a track-following driver that visits tiles and completes laps, domain randomization, the refused
    ///     actions, <c>rgb_array</c> frames with skid marks, leaving the playfield, spaces, metadata and <c>make</c>'s stack.
    /// </summary>
    /// <remarks>
    ///     Every observation is a 96×96 frame, so every lockstep step compares a rendering (the road and grass through SDL_gfx,
    ///     the car through <c>draw.polygon</c>, the indicators, the score) besides the reward, the flags, the info and the
    ///     generator. The score's glyphs are data embedded in Gym.NET.Environments (DR-48), recorded from pygame by
    ///     <c>tools/parity/record_score_font.py</c>. Cells that step or draw are pinned to the reference platform (Box2D's
    ///     <c>sinf</c>/<c>cosf</c>, the track's <c>atan2</c>/<c>sin</c>/<c>cos</c>). Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L106">gymnasium/envs/box2d/car_racing.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class CarRacingLiveParityTests : ParityTestBase {
        /// <summary>The id.</summary>
        private const string Id = "CarRacing-v3";

        /// <summary>The platform qualification of every cell that steps or draws.</summary>
        private const string Platform = "Box2D's sinf/cosf and the track's atan2/sin/cos";

        /// <summary>
        ///     Records the score font from the running pygame (the recorder in <c>tools/parity</c>) and lists where the embedded
        ///     data differs from it: each differing top-level field, and each differing glyph or kerned pair by name.
        /// </summary>
        private const string CompareFontStatements = """
            import sys, json
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            import record_score_font
            live = record_score_font.record()
            shipped = json.loads(embedded)
            result = [key for key in sorted(set(live) | set(shipped)) if live.get(key) != shipped.get(key)]
            for name in ("glyphs", "kerning"):
                if name in result:
                    a, b = live.get(name) or {}, shipped.get(name) or {}
                    result += [f"{name}[{key!r}]" for key in sorted(set(a) | set(b)) if a.get(key) != b.get(key)]
            """;

        /// <summary>
        ///     Every kind of score <c>f"{self.reward:04.0f}"</c> can show (DR-48, the maintainer's "all cases of what might be
        ///     plotted in that box"): NaN of both signs (<c>0nan</c>), both infinities, both zeros and the halves around them
        ///     (rounded to even, <c>-000</c>), the four-digit scores and their carries (<c>10000</c>, <c>-1000</c>), an episode's
        ///     usual values, scores far beyond any episode (2^53's neighbour, 1e20, 1e23's exact digits) and the extreme doubles
        ///     (309 digits, a box seven times the frame's width, and the subnormal ±5e-324).
        /// </summary>
        private static readonly double[] ScoreValues = {
            double.NaN, BitConverter.Int64BitsToDouble(0x7FF8000000000000), double.PositiveInfinity, double.NegativeInfinity,
            0.0, -0.0, 0.5, -0.5, 1.5, 2.5, -0.4, -0.6, 12.3, -12.3, 99.5, 999.5, 9999.5, -999.5, -100.1, 932.9, 12345.678, -12345.678,
            1e6, -1e6, 9007199254740993.0, 1e20, -1e20, 1e23, double.MaxValue, double.MinValue, double.Epsilon, -double.Epsilon,
        };

        /// <summary>pygame's shaded renders of the given strings: each one's width, height and palette indices, concatenated.</summary>
        private const string RenderTextStatements = """
            import numpy as np, pygame
            pygame.font.init()
            font = pygame.font.Font(pygame.font.get_default_font(), 42)
            sizes = []
            chunks = []
            for text in texts:
                surf = font.render(text, True, (255, 255, 255), (0, 0, 0))
                levels = np.ascontiguousarray(np.array(pygame.surfarray.pixels2d(surf)).T, dtype=np.uint8)
                sizes.append((surf.get_width(), surf.get_height()))
                chunks.append(levels.tobytes())
            result = (sizes, b"".join(chunks))
            """;

        /// <summary>
        ///     Gymnasium's side of <see cref="GymNet_DomainRandomize_MatchesGymnasium"/>: one randomized env through four resets,
        ///     each giving the three colours (as NumPy arrays) and the observation.
        /// </summary>
        private const string DomainRandomizeStatements = """
            from gymnasium.envs.box2d.car_racing import CarRacing
            env = CarRacing(domain_randomize=True)
            result = []
            for seed, randomize in [(3, None), (3, False), (3, True), (None, None)]:
                options = None if randomize is None else {"randomize": randomize}
                obs, info = env.reset(seed=seed, options=options)
                result.append([env.road_color, env.bg_color, env.grass_color, obs])
            env.close()
            """;

        /// <summary>
        ///     Gymnasium's side of <see cref="GymNet_Track_MatchesGymnasium"/>: the world a reset builds, every float as a Python
        ///     float and every colour as a list of floats.
        /// </summary>
        private const string TrackStatements = """
            from gymnasium.envs.box2d.car_racing import CarRacing
            env = CarRacing(domain_randomize=domain_randomize)
            env.reset(seed=seed)
            result = [
                float(env.start_alpha),
                [tuple(float(v) for v in step) for step in env.track],
                [([(float(p[0]), float(p[1])) for p in poly], [float(c) for c in color]) for poly, color in env.road_poly],
                [float(c) for c in env.road_color], [float(c) for c in env.bg_color], [float(c) for c in env.grass_color],
                [(float(t.position[0]), float(t.position[1]), [[(float(v[0]), float(v[1])) for v in f.shape.vertices] for f in t.fixtures]) for t in env.road],
                [(float(b.position[0]), float(b.position[1]), float(b.angle)) for b in env.car.drawlist],
            ]
            env.close()
            """;

        /// <summary>
        ///     The embedded glyph data (DR-48) is what <c>tools/parity/record_score_font.py</c> records from the running pygame
        ///     now: the same font file (name and SHA-256), metrics, glyphs, kerning and ligatures. A pygame-ce re-pin that
        ///     changes the font fails here, with the recorder's command to refresh the asset.
        /// </summary>
        /// <exception cref="AssertFailedException">Thrown when the resource or the repository layout is missing, or the data differs.</exception>
        [TestMethod]
        public void GymNet_ScoreFont_EmbeddedDataIsTheLiveRecording() {
            string embedded;
            using (Stream stream = typeof(CarRacingEnv).Assembly.GetManifestResourceStream(ScoreFont.ResourceName)
                                   ?? throw new AssertFailedException($"Gym.Environments lacks the resource '{ScoreFont.ResourceName}'."))
            using (var reader = new StreamReader(stream))
                embedded = reader.ReadToEnd();

            List<string> differences = PythonRuntime.Run(() => {
                string tools = GymPythonSession.Repository?.ParityTools ?? throw new AssertFailedException("The repository layout was not found.");
                using var toolsDir = new PyString(tools);
                using var embeddedText = new PyString(embedded);
                using PyObject result = Python.evalStmts(CompareFontStatements, "result", ("tools_dir", toolsDir), ("embedded", embeddedText));
                var listed = new List<string>();
                long count = result.Length();
                for (long i = 0; i < count; i++) {
                    using PyObject item = result[(int)i];
                    listed.Add(item.As<string>());
                }

                return listed;
            });

            Assert.AreEqual(0, differences.Count,
                            "The embedded score font differs from pygame's in: " + string.Join(", ", differences)
                            + ". Re-record it: python tools/parity/record_score_font.py src/Gym.Environments/Assets/fonts/freesansbold-42.json");
        }

        /// <summary>
        ///     The score's text, the embedded <see cref="ScoreFont"/> against pygame-ce 2.5.8's own shaded renders, level for
        ///     level, over the 15 characters a score can show (the digits, <c>-</c> and the letters of <c>nan</c> and
        ///     <c>inf</c>): every string of one and two characters, the scores -300 to 1,100 (and their halves, rounded to even),
        ///     the text of every kind of score (<see cref="ScoreValues"/>, up to 309 characters), and 600 random strings of up
        ///     to 12 characters, laid out with SDL_ttf 2.24's kerning and floored 26.6 positions. Strings with the ligature
        ///     sequence <c>fi</c>, which no score contains and <see cref="ScoreFont"/> refuses, are left out.
        /// </summary>
        [TestMethod]
        public void GymNet_ScoreFont_RendersAsPygame() {
            ScoreFont font = ScoreFont.Embedded;
            var texts = new List<string>();
            const string chars = "0123456789-afin";
            foreach (char a in chars) {
                texts.Add(a.ToString());
                foreach (char b in chars)
                    texts.Add($"{a}{b}");
            }

            for (int r = -300; r <= 1100; r++) {
                texts.Add(CarRacingEnv.FormatScore(r));
                texts.Add(CarRacingEnv.FormatScore(r + 0.5));
            }

            texts.AddRange(ScoreValues.Select(CarRacingEnv.FormatScore));
            var rng = new Random(20261111);
            for (int i = 0; i < 600; i++)
                texts.Add(new string(Enumerable.Range(0, rng.Next(1, 13)).Select(_ => chars[rng.Next(chars.Length)]).ToArray()));
            // pygame shapes "fi" into one ligature glyph; no score contains it, and the font refuses it (CarRacingTests).
            texts.RemoveAll(text => text.Contains("fi", StringComparison.Ordinal));

            (List<(int W, int H)> sizes, byte[] levels) = PythonRuntime.Run(() => {
                using var list = new PyList(texts.Select(t => (PyObject)new PyString(t)).ToArray());
                using PyObject result = Python.evalStmts(RenderTextStatements, "result", ("texts", list));
                using PyObject pySizes = result[0];
                using PyObject pyBytes = result[1];
                var parsed = new List<(int W, int H)>();
                long count = pySizes.Length();
                for (long i = 0; i < count; i++) {
                    using PyObject item = pySizes[(int)i];
                    using PyObject w = item[0];
                    using PyObject h = item[1];
                    parsed.Add((w.As<int>(), h.As<int>()));
                }

                return (parsed, pyBytes.As<byte[]>());
            });

            int offset = 0;
            for (int i = 0; i < texts.Count; i++) {
                (int w, int h) = sizes[i];
                (byte[] ours, int width, int height) = font.RenderShaded(texts[i]);
                Assert.AreEqual((w, h), (width, height), $"Surface size of '{texts[i]}'.");
                for (int k = 0; k < ours.Length; k++) {
                    if (ours[k] != levels[offset + k])
                        Assert.Fail($"'{texts[i]}': level at ({k % width}, {k / width}) is {ours[k]}, pygame's is {levels[offset + k]}.");
                }

                offset += w * h;
            }

            TestContext.WriteLine($"{texts.Count} strings, every level identical");
        }

        /// <summary>
        ///     The score's box for every kind of value (<see cref="ScoreValues"/>), set as <c>reward</c> on both envs after a
        ///     reset and rendered as the 600×400 <c>rgb_array</c> frame and as the 96×96 observation (<c>state_pixels</c>), pixel
        ///     for pixel: the text's surface, its centring by truncated halves, and the blit's clipping when the box is wider
        ///     than the frame.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown when the pair's .NET side is not a CarRacing env.</exception>
        /// <exception cref="ParityException">Thrown at the first frame that differs, with its artifacts.</exception>
        [TestMethod]
        public void GymNet_ScoreBox_EveryKindOfValue_MatchesGymnasium() {
            SkipOffReferencePlatform(Platform);
            foreach (string mode in new[] { "rgb_array", "state_pixels" }) {
                using GymPair pair = Pair(true, mode);
                pair.Reset(23);
                IEnv unwrapped = ((GymNetEnv)pair.DotNet).Env.Unwrapped;
                CarRacingEnv dotnet = unwrapped as CarRacingEnv ?? throw new InvalidOperationException("The pair's .NET side is not CarRacing.");
                foreach (double score in ScoreValues) {
                    pair.Python.WithUnwrapped(env => {
                        using var value = new PyFloat(score);
                        using PyObject none = Python.builtins.setattr(env, "reward", value);
                    });
                    dotnet.Reward = score;
                    string text = CarRacingEnv.FormatScore(score);
                    pair.CompareFrame($"{mode}, reward {score:R} ('{(text.Length > 24 ? text[..12] + $"... {text.Length} characters" : text)}')");
                }

                TestContext.WriteLine($"{mode}: {ScoreValues.Length} scores, every frame identical");
            }
        }

        /// <summary><c>reset(seed=seed)</c> of both action modes: the track's draws, the car, and the first observation's frame.</summary>
        /// <param name="continuous">Which action mode.</param>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(true, 0L)]
        [DataRow(true, 1L)]
        [DataRow(true, 42L)]
        [DataRow(false, 0L)]
        [DataRow(false, 7L)]
        public void GymNet_Reset_MatchesGymnasium(bool continuous, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous);
            pair.Reset(seed);
        }

        /// <summary>
        ///     The world a reset builds, value for value: the start angle, every track step, every road and kerb polygon with its
        ///     colour, the three colours, every tile's body and hulled vertices, and the car's bodies; with and without domain
        ///     randomization.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        /// <param name="domainRandomize">Whether the colours are random.</param>
        [TestMethod]
        [DataRow(0L, false)]
        [DataRow(3L, false)]
        [DataRow(11L, false)]
        [DataRow(0L, true)]
        [DataRow(5L, true)]
        public void GymNet_Track_MatchesGymnasium(long seed, bool domainRandomize) {
            SkipOffReferencePlatform(Platform);
            ValueFacts python = Outcome.CapturePython(() => {
                using PyObject flag = domainRandomize.ToPython();
                using var seedValue = new PyInt(seed);
                return Python.evalStmts(TrackStatements, "result", ("domain_randomize", flag), ("seed", seedValue));
            }).Expect($"CarRacing(domain_randomize={domainRandomize}).reset(seed={seed})'s world");

            ValueFacts dotnet = Outcome.Capture(() => {
                using var env = new CarRacingEnv(domainRandomize: domainRandomize);
                env.Reset(seed).Observation.Dispose();
                static List<double> Channels(NDArray color) {
                    using (color) {
                        using NDArray widened = color.astype(np.float64);
                        return widened.ToArray<double>().ToList();
                    }
                }

                static List<(double, double)> Vertices(Gym.Box2D.b2Fixture f) => ((Gym.Box2D.b2PolygonShape)f.Shape).Vertices.Select(v => ((double)v.x, (double)v.y)).ToList();
                return new object?[] {
                    env.StartAlpha,
                    env.Track.Select(s => (object)(s.Alpha, s.Beta, s.X, s.Y)).ToList(),
                    env.RoadPoly.Select(p => (object)(p.Poly.ToList(), p.Color.ToList())).ToList(),
                    Channels(env.RoadColor), Channels(env.BgColor), Channels(env.GrassColor),
                    env.Road!.Select(t => (object)((double)t.Body.Position.x, (double)t.Body.Position.y, t.Body.Fixtures.Select(Vertices).ToList())).ToList(),
                    env.Car!.Drawlist.Select(b => (object)((double)b.Position.x, (double)b.Position.y, (double)b.Angle)).ToList(),
                };
            }).Expect($"CarRacingEnv(domainRandomize: {domainRandomize}).Reset({seed})'s world");

            ByteContract.AssertSame($"CarRacing world after reset(seed={seed}), domain_randomize={domainRandomize}", python, dotnet);
        }

        /// <summary>Episodes of seeded sampled actions of both modes, 250 steps: skids on grass, tiles visited, every frame compared.</summary>
        /// <param name="continuous">Which action mode.</param>
        /// <param name="seed">The reset seed, also the action spaces' seed.</param>
        [TestMethod]
        [DataRow(true, 0L)]
        [DataRow(true, 9L)]
        [DataRow(false, 0L)]
        [DataRow(false, 4L)]
        public void GymNet_SampledEpisodes_MatchGymnasium(bool continuous, long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(continuous);
            PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 250);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     A track-following driver for a whole lap: hundreds of tiles visited (each paying 1000/N and recolouring), the
        ///     score's text through every value, and the lap's completion (<c>lap_finished</c>), unless the car spins out first.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        [TestMethod]
        [DataRow(0L)]
        [DataRow(2L)]
        public void GymNet_FollowerEpisodes_MatchGymnasium(long seed) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(true);
            PairRun run = pair.Run(seed, new FollowerSource(), maxSteps: 3000);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     <c>rgb_array</c> frames along the driver's and sampled episodes, every fifth step: the 600×400 view of the road, the
        ///     car, the indicators, the score, and the skid marks (2-pixel polylines), which the observation leaves out.
        /// </summary>
        /// <param name="follower">Whether the driver (else sampled actions) drives.</param>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void GymNet_Frames_MatchGymnasium(bool follower) {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(true, "rgb_array");
            PairRun run = pair.Run(5, follower ? new FollowerSource() : ActionSources.Sampled(5), maxSteps: 400, frameEvery: 5);
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     Domain randomization through a sequence of resets on one env: seeded (new colours, drawn before the track), seeded
        ///     with <c>{"randomize": False}</c> (the colours kept, so the same seed builds another track), seeded with
        ///     <c>{"randomize": True}</c>, and unseeded (the generator continues); after each, the colours and the observation.
        /// </summary>
        [TestMethod]
        public void GymNet_DomainRandomize_MatchesGymnasium() {
            SkipOffReferencePlatform(Platform);
            ValueFacts python = Outcome.CapturePython(() => Python.evalStmts(DomainRandomizeStatements, "result")).Expect("Gymnasium's randomized resets");
            ValueFacts dotnet = Outcome.Capture(() => {
                using var env = new CarRacingEnv(domainRandomize: true);
                var results = new List<object?>();
                foreach ((long? seed, object? randomize) in new (long?, object?)[] { (3, null), (3, false), (3, true), (null, null) }) {
                    Dictionary<string, object?>? options = randomize is null ? null : new Dictionary<string, object?> { ["randomize"] = randomize };
                    NDArray observation = env.Reset(seed, options).Observation;
                    results.Add(new object?[] { env.RoadColor, env.BgColor, env.GrassColor, observation });
                }

                return results;
            }).Expect("Gym.NET's randomized resets");
            ByteContract.AssertSame("CarRacing(domain_randomize=True) resets", python, dotnet);
        }

        /// <summary>
        ///     Actions of every form: continuous float32 and float64 controls inside and beyond their bounds, int64, NaN steering,
        ///     four elements, and the refused 0-d and two-element arrays (NumPy's <c>IndexError</c> after the steering and gas were
        ///     applied); discrete actions outside <c>Discrete(5)</c> with Gymnasium's <c>InvalidAction</c> message.
        /// </summary>
        [TestMethod]
        public void GymNet_ActionForms_MatchGymnasium() {
            SkipOffReferencePlatform(Platform);
            var continuous = new (string What, Func<NDArray> Make)[] {
                ("float32 [0.5, 0.8, 0.0]", () => np.array(new[] { 0.5f, 0.8f, 0.0f })),
                ("float64 [-3.0, 2.0, 0.95]", () => np.array(new[] { -3.0, 2.0, 0.95 })),
                ("int64 [1, 1, 0]", () => np.array(new[] { 1L, 1L, 0L })),
                ("float64 [nan, 0.5, 0.0]", () => np.array(new[] { double.NaN, 0.5, 0.0 })),
                ("float32 [0.1, 0.3, 0.2, 9.0]", () => np.array(new[] { 0.1f, 0.3f, 0.2f, 9.0f })),
                ("0-d float64 0.5", () => NDArray.Scalar<double>(0.5)),
                ("float32 [0.2, 0.9]", () => np.array(new[] { 0.2f, 0.9f })),
            };
            using (GymPair pair = Pair(true)) {
                pair.Reset(13);
                foreach ((string what, Func<NDArray> make) in continuous) {
                    using (ParityAction action = ParityAction.Continuous(make()))
                        pair.Step(action);
                    using ParityAction follow = ParityAction.Continuous(np.array(new[] { -0.2f, 0.5f, 0.0f }));
                    pair.Step(follow);
                    TestContext.WriteLine($"{what}: compared, and the step after it");
                }
            }

            using (GymPair pair = Pair(false)) {
                pair.Reset(13);
                foreach (long refused in new long[] { 5, -1, 100 }) {
                    using (ParityAction action = ParityAction.Discrete(refused))
                        pair.Step(action);
                    using ParityAction follow = ParityAction.Discrete(3);
                    pair.Step(follow);
                }
            }
        }

        /// <summary>
        ///     Gas 0.3 straight ahead until the car leaves the playfield (at step 310 for seed 17, near 97 m/s): the -100 and
        ///     <c>lap_finished=False</c>. Full gas would not do: the rear wheels spin and the car circles at about 11 m/s.
        /// </summary>
        [TestMethod]
        public void GymNet_LeavingThePlayfield_MatchesGymnasium() {
            SkipOffReferencePlatform(Platform);
            using GymPair pair = Pair(true);
            PairRun run = pair.Run(17, new ConstantSource(new[] { 0.0, 0.3, 0.0 }), maxSteps: 600);
            Assert.IsTrue(run.Ended, $"The car must leave the playfield: {run}");
            TestContext.WriteLine(run.ToString());
        }

        /// <summary>
        ///     A step before any reset: Gymnasium's bare <c>assert self.car is not None</c>, whose empty message NumSharp's
        ///     <c>AssertionError()</c> does not give yet (N-1).
        /// </summary>
        [TestMethod]
        [OpenBugs("N-1")]
        public void GymNet_StepBeforeReset_RaisesLikeGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make(Id);
                return Python.gymnasium.Env.unwrapped(made);
            });
            IEnv dotnet = new CarRacingEnv();
            using NDArray zeros = np.zeros(3);
            Outcome pythonStep = Outcome.CapturePython(() => {
                using PyObject action = zeros.ToNumpyCopy();
                return Python.gymnasium.Env.step(python.Object, action);
            });
            ByteContract.AssertSame("step(zeros(3)) before reset", pythonStep, Outcome.Capture(() => dotnet.Step(zeros)));
        }

        /// <summary><c>render()</c> before any reset in <c>rgb_array</c> and <c>state_pixels</c> mode: <c>None</c> without an error, as Gymnasium's early return gives.</summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_ReturnsNoneLikeGymnasium() {
            foreach (string mode in new[] { "rgb_array", "state_pixels" }) {
                using PyGymEnv python = PyGymEnv.Make(Id, renderMode: mode);
                var dotnet = new CarRacingEnv(renderMode: mode);
                RenderOutcome.AssertSame($"render() before reset ({mode})", python.TryRender(), RenderOutcome.Capture(() => RenderGymNet(dotnet)));
            }
        }

        /// <summary>The spaces: <c>Box(0, 255, (96, 96, 3), uint8)</c>, <c>Box([-1, 0, 0], [1, 1, 1], float32)</c> and <c>Discrete(5)</c>, with seeded samples.</summary>
        /// <param name="continuous">Which action mode.</param>
        [TestMethod]
        [DataRow(true)]
        [DataRow(false)]
        public void GymNet_Spaces_MatchGymnasium(bool continuous) {
            var env = new CarRacingEnv(continuous: continuous);
            var kwargs = new Dictionary<string, object?> { ["continuous"] = continuous };
            ClassicSpaces.AssertBoxSame($"{Id}(continuous={continuous}) observation_space", Id, Python.gymnasium.Env.observation_space, (Box)env.ObservationSpace, kwargs);
            ClassicSpaces.AssertActionSpaceSame(Id, env.ActionSpace, kwargs);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array', 'state_pixels'], 'render_fps': 50}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make(Id);
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame($"{Id} metadata", python, Outcome.Returned(ValueFacts.Of(new CarRacingEnv().Metadata)));
        }

        /// <summary><c>make</c>'s stack (<c>TimeLimit</c> at 1,000 steps) against <c>Gymnasium.Make</c>: the driver until the limit or the lap.</summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasium() {
            SkipOffReferencePlatform(Platform);
            using var pair = new GymPair(PyGymEnv.Make(Id, viaMake: true), new GymNetEnv(Gymnasium.Make(Id)));
            PairRun run = pair.Run(31, new FollowerSource(), maxSteps: 1100);
            Assert.IsTrue(run.Ended, $"The episode must end: {run}");
        }

        /// <summary>A lockstep pair of Gymnasium's unwrapped env and Gym.NET's, built with the same arguments.</summary>
        /// <param name="continuous">Which action mode.</param>
        /// <param name="renderMode">The render mode of both, or <see langword="null"/>.</param>
        /// <returns>The pair; the caller disposes it.</returns>
        private static GymPair Pair(bool continuous, string? renderMode = null) =>
            new(PyGymEnv.Make(Id, new Dictionary<string, object?> { ["continuous"] = continuous }, renderMode: renderMode),
                new GymNetEnv(new CarRacingEnv(renderMode: renderMode, continuous: continuous)));

        /// <summary>Gym.NET's <c>render()</c> of an <c>rgb_array</c> env as a frame, or <see langword="null"/> when it returned <c>None</c>.</summary>
        /// <param name="env">The env.</param>
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
                    throw new InvalidOperationException($"{env} rendered {other}.");
            }
        }

        /// <summary>The same continuous float64 action every step.</summary>
        /// <param name="Controls">Steer, gas, brake.</param>
        private sealed record ConstantSource(double[] Controls) : IActionSource {
            /// <inheritdoc />
            public string Description => $"always [{string.Join(", ", Controls)}]";

            /// <inheritdoc />
            public void Start(GymPair pair) { }

            /// <inheritdoc />
            public ParityAction Next(GymPair pair) => ParityAction.Continuous(np.array((double[])Controls.Clone()));
        }

        /// <summary>
        ///     A track-following driver reading Gym.NET's env (identical to Gymnasium's while the pair agrees): it steers toward the
        ///     track point six steps ahead of the nearest one and holds about 30 m/s, as float64 controls both sides receive.
        /// </summary>
        private sealed class FollowerSource : IActionSource {
            /// <inheritdoc />
            public string Description => "track follower";

            /// <inheritdoc />
            public void Start(GymPair pair) { }

            /// <inheritdoc />
            public ParityAction Next(GymPair pair) {
                IEnv unwrapped = ((GymNetEnv)pair.DotNet).Env.Unwrapped;
                CarRacingEnv env = unwrapped as CarRacingEnv ?? throw new InvalidOperationException("The follower drives CarRacing only.");
                Car car = env.Car!;
                Gym.Box2D.b2Vec2 position = car.Hull!.Position;
                IReadOnlyList<(double Alpha, double Beta, double X, double Y)> track = env.Track;
                int nearest = 0;
                double best = double.MaxValue;
                for (int i = 0; i < track.Count; i++) {
                    double dx = track[i].X - position.x, dy = track[i].Y - position.y;
                    double d = dx * dx + dy * dy;
                    if (d < best) {
                        best = d;
                        nearest = i;
                    }
                }

                (double _, double _, double tx, double ty) = track[(nearest + 6) % track.Count];
                double desired = Math.Atan2(ty - position.y, tx - position.x);
                double heading = car.Hull.Angle + Math.PI / 2; // the car's forward is its local +y
                double error = Math.IEEERemainder(desired - heading, 2 * Math.PI);
                double steer = Math.Clamp(-2.0 * error, -1.0, 1.0);
                Gym.Box2D.b2Vec2 velocity = car.Hull.LinearVelocity;
                double speed = Math.Sqrt((double)velocity.x * velocity.x + (double)velocity.y * velocity.y);
                double gas = speed < 30 ? 0.4 : 0.0;
                double brake = speed > 40 ? 0.3 : 0.0;
                return ParityAction.Continuous(np.array(new[] { steer, gas, brake }));
            }
        }
    }
}
