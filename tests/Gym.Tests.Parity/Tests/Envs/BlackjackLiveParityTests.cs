using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.ToyText;
using Gym.Envs;
using Gym.Observations;
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
    ///     Blackjack-v1 against Gymnasium's <c>BlackjackEnv</c>, live and in lockstep with Gym.NET's <see cref="BlackjackEnv"/>:
    ///     games under every rule set (<c>sab</c> and <c>natural</c> on and off), the cards each draw takes from the generator, the
    ///     module's hand functions, every reward rule on injected hands, the actions the space assertion admits, the
    ///     <c>rgb_array</c> table, spaces, metadata and <c>make</c>'s stack.
    /// </summary>
    /// <remarks>
    ///     Gymnasium's observation is a tuple of three Python ints; Gym.NET's is <see langword="object"/>?[] of three 0-d int64
    ///     arrays, which the comparison equates item by item (DR-08). The hands, the face-up card's suit and face are
    ///     compared after every game, so a card drawn out of order shows even when the observation agrees. Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L124">gymnasium/envs/toy_text/blackjack.py</see>.
    /// </remarks>
    [TestClass]
    public sealed class BlackjackLiveParityTests : ParityTestBase {
        /// <summary>
        ///     The rule sets compared, each in Python spelling and as Gym.NET's constructor call. Both flags are always given:
        ///     <c>make</c> merges them into the registered <c>sab=True, natural=False</c>, so a flag left out would be the
        ///     registration's on Gymnasium's side and the constructor's default on Gym.NET's.
        /// </summary>
        private static readonly (string Python, Dictionary<string, object?> Kwargs, Func<string?, BlackjackEnv> DotNet)[] Variants = {
            ("sab=False, natural=False", new Dictionary<string, object?> { ["sab"] = false, ["natural"] = false }, mode => new BlackjackEnv(renderMode: mode)),
            ("sab=False, natural=True", new Dictionary<string, object?> { ["sab"] = false, ["natural"] = true }, mode => new BlackjackEnv(renderMode: mode, natural: true)),
            ("sab=True, natural=False", new Dictionary<string, object?> { ["sab"] = true, ["natural"] = false }, mode => new BlackjackEnv(renderMode: mode, sab: true)),
            ("sab=True, natural=True", new Dictionary<string, object?> { ["sab"] = true, ["natural"] = true }, mode => new BlackjackEnv(renderMode: mode, natural: true, sab: true)),
        };

        /// <summary>
        ///     Games of seeded sampled actions under each rule set, 150 seeds each (every game to its end), with the hands, the
        ///     face-up card's suit and face, and the generator compared after each game.
        /// </summary>
        [TestMethod]
        public void GymNet_SampledGames_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, BlackjackEnv> create) in Variants) {
                BlackjackEnv dotnet = create(null);
                using var pair = new GymPair(PyGymEnv.Make("Blackjack-v1", kwargs), new GymNetEnv(dotnet));
                int steps = 0;
                for (long seed = 0; seed < 150; seed++) {
                    PairRun run = pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 20);
                    steps += run.Steps;
                    AssertHandsSame($"{text}, seed {seed}", pair, dotnet);
                }

                TestContext.WriteLine($"{text}: 150 games, {steps} steps");
            }
        }

        /// <summary>
        ///     Games played by a fixed strategy (hit below 17, then stick), so the dealer's drawing loop and every scoring path
        ///     run often: 150 seeds per rule set.
        /// </summary>
        [TestMethod]
        public void GymNet_StrategyGames_MatchGymnasium() {
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, BlackjackEnv> create) in Variants) {
                BlackjackEnv dotnet = create(null);
                using var pair = new GymPair(PyGymEnv.Make("Blackjack-v1", kwargs), new GymNetEnv(dotnet));
                for (long seed = 1000; seed < 1150; seed++) {
                    pair.Reset(seed);
                    for (int i = 0; i < 10 && !pair.Ended; i++) {
                        using var action = ParityAction.Discrete(BlackjackEnv.SumHand(dotnet.Player!) < 17 ? 1 : 0);
                        pair.Step(action);
                    }

                    AssertHandsSame($"{text}, seed {seed}", pair, dotnet);
                }
            }
        }

        /// <summary>
        ///     Every reward rule on injected hands, with stick and hit: wins, losses and draws, the dealer drawing to 17 or
        ///     busting, naturals on either side under each rule set (<c>sab</c>'s automatic win, <c>natural</c>'s 1.5, both
        ///     naturals), and a hit that busts or does not.
        /// </summary>
        [TestMethod]
        public void GymNet_RewardRules_MatchGymnasium() {
            var hands = new (long[] Player, long[] Dealer)[] {
                (new long[] { 1, 10 }, new long[] { 10, 7 }), (new long[] { 1, 10 }, new long[] { 1, 10 }), (new long[] { 10, 1 }, new long[] { 9, 9 }),
                (new long[] { 10, 10 }, new long[] { 10, 6 }), (new long[] { 10, 10 }, new long[] { 2, 3 }), (new long[] { 9, 8 }, new long[] { 10, 8 }),
                (new long[] { 5, 5, 1 }, new long[] { 10, 7 }), (new long[] { 1, 1 }, new long[] { 1, 5 }), (new long[] { 10, 9, 2 }, new long[] { 10, 1 }),
                (new long[] { 4, 2 }, new long[] { 6, 1 }), (new long[] { 10, 5, 7 }, new long[] { 3, 3 }), (new long[] { 1, 10, 10 }, new long[] { 7, 10 }),
            };
            foreach ((string text, Dictionary<string, object?> kwargs, Func<string?, BlackjackEnv> create) in Variants) {
                BlackjackEnv dotnet = create(null);
                using var pair = new GymPair(PyGymEnv.Make("Blackjack-v1", kwargs), new GymNetEnv(dotnet));
                pair.Reset(77);
                foreach ((long[] player, long[] dealer) in hands) {
                    foreach (long action in new long[] { 0, 1 }) {
                        InjectHands(pair, dotnet, player, dealer);
                        using var parity = ParityAction.Discrete(action);
                        pair.Step(parity);
                        AssertHandsSame($"{text}: [{string.Join(", ", player)}] vs [{string.Join(", ", dealer)}], step({action})", pair, dotnet);
                    }
                }
            }
        }

        /// <summary>
        ///     <c>draw_card</c> and <c>draw_hand</c> on generators seeded on both sides: 200 cards and 50 hands per seed, value for
        ///     value, and the generator's state after them.
        /// </summary>
        [TestMethod]
        public void GymNet_Draws_MatchGymnasium() {
            foreach (long seed in new long[] { 0, 1, 2024, long.MaxValue }) {
                using PyRef python = PyRef.Of(() => {
                    using PyObject seedValue = PyConvert.ToPython(seed);
                    using PyObject seeded = Python.gymnasium.utils.seeding.np_random(seedValue);
                    return seeded[0];
                });
                (Generator dotnet, _) = Seeding.NpRandom(seed);
                for (int i = 0; i < 200; i++)
                    ByteContract.AssertSame($"draw_card #{i + 1} (seed {seed})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.blackjack.draw_card(python.Object)),
                        Outcome.Returned(ValueFacts.Of(BlackjackEnv.DrawCard(dotnet))));
                for (int i = 0; i < 50; i++)
                    ByteContract.AssertSame($"draw_hand #{i + 1} (seed {seed})", Outcome.CapturePython(() => Python.gymnasium.envs.toy_text.blackjack.draw_hand(python.Object)),
                        Outcome.Returned(ValueFacts.Of(BlackjackEnv.DrawHand(dotnet))));
                ByteContract.AssertSame($"generator after the draws (seed {seed})", RngFacts.OfPython(python.Object), Outcome.Returned(RngFacts.Of(dotnet)));
            }
        }

        /// <summary>
        ///     The module's hand functions (<c>usable_ace</c>, <c>sum_hand</c>, <c>is_bust</c>, <c>score</c>, <c>is_natural</c>) over
        ///     every hand of up to three cards and an empty one, and <c>cmp</c> over pairs of scores and NaN.
        /// </summary>
        [TestMethod]
        public void GymNet_HandFunctions_MatchGymnasium() {
            var hands = new List<long[]> { Array.Empty<long>() };
            for (long a = 1; a <= 10; a++) {
                hands.Add(new[] { a });
                for (long b = 1; b <= 10; b++) {
                    hands.Add(new[] { a, b });
                    for (long c = 1; c <= 10; c++)
                        hands.Add(new[] { a, b, c });
                }
            }

            var functions = new (string Name, Func<IReadOnlyList<long>, object> DotNet)[] {
                ("usable_ace", hand => BlackjackEnv.UsableAce(hand)), ("sum_hand", hand => BlackjackEnv.SumHand(hand)), ("is_bust", hand => BlackjackEnv.IsBust(hand)),
                ("score", hand => BlackjackEnv.Score(hand)), ("is_natural", hand => BlackjackEnv.IsNatural(hand)),
            };
            foreach ((string name, Func<IReadOnlyList<long>, object> dotnet) in functions) {
                ValueFacts python = Outcome.CapturePython(() => {
                    using PyObject function = Python.builtins.getattr(PythonRuntime.Import("gymnasium.envs.toy_text.blackjack"), name);
                    using PyObject allHands = PyConvert.ToPython(hands.Select(h => (object?)h.Cast<object?>().ToList()).ToList());
                    return Python.evalStmts("results = [f(h) for h in hands]\n", "results", ("f", function), ("hands", allHands));
                }).Expect(name);
                ByteContract.AssertSame($"{name} over {hands.Count} hands", Outcome.Returned(python), Outcome.Returned(ValueFacts.Of(hands.Select(h => dotnet(h)).ToList())));
            }

            double[] scores = { 0, 1, 17, 21, 22, -1, double.NaN };
            foreach (double a in scores) {
                foreach (double b in scores) {
                    ByteContract.AssertSame($"cmp({a}, {b})", Outcome.CapturePython(() => {
                        using PyObject left = PyConvert.ToPython(a);
                        using PyObject right = PyConvert.ToPython(b);
                        return Python.gymnasium.envs.toy_text.blackjack.cmp(left, right);
                    }), Outcome.Returned(ValueFacts.Of(BlackjackEnv.Cmp(a, b))));
                }
            }
        }

        /// <summary>
        ///     Actions the space assertion admits, one step each on a fresh deal: 0, 1, <c>True</c>, <c>False</c>, and NumPy integer
        ///     scalars; and stepping before a reset, which fails at the hand Python reaches first (<c>player</c> for a hit,
        ///     <c>dealer</c> for a stick).
        /// </summary>
        [TestMethod]
        public void GymNet_Actions_MatchGymnasium() {
            var accepted = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("0", () => PyConvert.ToPython(0L), () => 0L),
                ("1", () => PyConvert.ToPython(1L), () => 1L),
                ("True", () => PyConvert.ToPython(true), () => true),
                ("False", () => PyConvert.ToPython(false), () => false),
                ("np.int64(1)", () => Python.numpy.int64(1), () => NDArray.Scalar<long>(1L)),
                ("np.uint8(0)", () => Python.numpy.uint8(0), () => NDArray.Scalar<byte>(0)),
            };

            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Blackjack-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new BlackjackEnv(sab: true);
            IEnv untyped = dotnet;
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in accepted.Take(2)) {
                Outcome before = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                ByteContract.AssertSame($"step({text}) before reset", before, Outcome.Capture(() => untyped.Step(value())));
            }

            long seed = 30;
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in accepted) {
                long s = seed++;
                ByteContract.AssertSame($"reset(seed={s})", Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, s)), Outcome.Capture(() => {
                    ResetResult<object?[]> reset = dotnet.Reset(s);
                    return new object?[] { reset.Observation, reset.Info };
                }));
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                FrozenLakeLiveParityTests.AssertStepSame($"step({text})", pythonStep, Outcome.Capture(() => FrozenLakeLiveParityTests.StepTuple(untyped.Step(value()))));
            }
        }

        /// <summary>
        ///     Actions the space assertion refuses (2, -1, 0.5, <c>np.True_</c>, a one-element array, <c>None</c>): Gymnasium raises
        ///     a bare <c>AssertionError</c>, whose empty message NumSharp's parameterless <c>AssertionError</c> does not carry yet
        ///     (N-1).
        /// </summary>
        [TestMethod]
        public void GymNet_RefusedActions_RaiseTheBareAssertion() {
            var refused = new (string Python, Func<PyObject> Gymnasium, Func<object?> DotNet)[] {
                ("2", () => PyConvert.ToPython(2L), () => 2L),
                ("-1", () => PyConvert.ToPython(-1L), () => -1L),
                ("0.5", () => PyConvert.ToPython(0.5), () => 0.5),
                ("None", () => PyConvert.ToPython(null), () => null),
            };
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Blackjack-v1");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new BlackjackEnv(sab: true);
            IEnv untyped = dotnet;
            Outcome.CapturePython(() => Python.gymnasium.Env.reset(python.Object, 3)).Expect("reset(seed=3)");
            DisposeObservation(dotnet.Reset(3L).Observation);
            foreach ((string text, Func<PyObject> gymnasium, Func<object?> value) in refused) {
                Outcome pythonStep = Outcome.CapturePython(() => {
                    using PyObject action = gymnasium();
                    return Python.gymnasium.Env.step(python.Object, action);
                });
                ByteContract.AssertSame($"step({text})", pythonStep, Outcome.Capture(() => untyped.Step(value())));
            }
        }

        /// <summary>
        ///     <c>rgb_array</c> frames of the table after the reset and every step of 60 seeded games (every face-up card's face and
        ///     suit, one- and two-digit totals, the usable-ace label), and of injected hands at the totals' extremes.
        /// </summary>
        [TestMethod]
        public void GymNet_Frames_MatchGymnasium() {
            var kwargs = new Dictionary<string, object?> { ["sab"] = true };
            var dotnet = new BlackjackEnv(renderMode: "rgb_array", sab: true);
            using var pair = new GymPair(PyGymEnv.Make("Blackjack-v1", kwargs, renderMode: "rgb_array"), new GymNetEnv(dotnet));
            for (long seed = 0; seed < 60; seed++)
                pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 10, frameEvery: 1);

            var hands = new (long[] Player, long[] Dealer)[] {
                (new long[] { 2, 2 }, new long[] { 1, 9 }), (new long[] { 1, 1 }, new long[] { 10, 9 }), (new long[] { 10, 10, 10, 1 }, new long[] { 5, 5 }),
                (new long[] { 1, 9 }, new long[] { 9, 1 }), (new long[] { 3 }, new long[] { 2 }),
            };
            pair.Reset(5);
            foreach ((long[] player, long[] dealer) in hands) {
                InjectHands(pair, dotnet, player, dealer);
                pair.CompareFrame($"[{string.Join(", ", player)}] vs [{string.Join(", ", dealer)}]");
            }
        }

        /// <summary>Rendering before the first reset raises Python's <c>AttributeError</c> for <c>player</c>.</summary>
        [TestMethod]
        public void GymNet_RenderBeforeReset_RaisesAsGymnasium() {
            using PyRef python = PyRef.Of(() => {
                using PyObject made = Python.gymnasium.make("Blackjack-v1", render_mode: "rgb_array");
                return Python.gymnasium.Env.unwrapped(made);
            });
            var dotnet = new BlackjackEnv(renderMode: "rgb_array");
            ByteContract.AssertSame("rgb_array render() before reset", Outcome.CapturePython(() => Python.gymnasium.Env.render(python.Object)),
                Outcome.Capture(() => ((RenderResult.Rgb)dotnet.Render()).Frame));
        }

        /// <summary>
        ///     The spaces: <c>Tuple(Discrete(32), Discrete(11), Discrete(2))</c> observations (<c>repr</c>, seeded samples, the
        ///     generator) and <c>Discrete(2)</c> actions with seeded samples.
        /// </summary>
        [TestMethod]
        public void GymNet_Spaces_MatchGymnasium() {
            const string id = "Blackjack-v1";
            IEnv env = Gymnasium.Make(id).Unwrapped;
            using PyRef observation = PyRef.Of(() => CartPoleLiveParityTests.MadeSpace(id, Python.gymnasium.Env.observation_space));
            Space space = env.ObservationSpace;
            ByteContract.AssertSame($"{id} repr(observation_space)", Outcome.CapturePython(() => Python.builtins.repr(observation.Object)),
                Outcome.Returned(ValueFacts.FromText(space.ToString()!)));
            Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.seed(observation.Object, 5)).Expect("observation_space.seed(5)");
            space.Seed(5);
            for (int i = 1; i <= 20; i++)
                ByteContract.AssertSame($"{id} observation_space.sample() #{i} after seed(5)", Outcome.CapturePython(() => Python.gymnasium.spaces.space.Space.sample(observation.Object)),
                    Outcome.Capture(() => space.Sample()));
            ClassicSpaces.AssertActionSpaceSame(id, env.ActionSpace);
        }

        /// <summary><c>metadata</c>: <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 4}</c>.</summary>
        [TestMethod]
        public void GymNet_Metadata_MatchesGymnasium() {
            Outcome python = Outcome.CapturePython(() => {
                using PyObject made = Python.gymnasium.make("Blackjack-v1");
                using PyObject unwrapped = Python.gymnasium.Env.unwrapped(made);
                return Python.gymnasium.Env.metadata(unwrapped);
            });
            ByteContract.AssertSame("Blackjack metadata", python, Outcome.Returned(ValueFacts.Of(new BlackjackEnv().Metadata)));
        }

        /// <summary>
        ///     <c>make("Blackjack-v1")</c>'s whole stack (<c>sab=True</c>, no time limit) against <c>Gymnasium.Make</c>, its
        ///     passive checker included: 100 seeded games of sampled actions.
        /// </summary>
        [TestMethod]
        public void GymNet_MakeStack_MatchesGymnasium() {
            using var pair = new GymPair(PyGymEnv.Make("Blackjack-v1", viaMake: true), new GymNetEnv(Gymnasium.Make("Blackjack-v1")));
            for (long seed = 500; seed < 600; seed++)
                pair.Run(seed, ActionSources.Sampled(seed), maxSteps: 20);
        }

        /// <summary>Releases an observation the test does not compare: its three 0-d arrays.</summary>
        /// <param name="observation">The observation.</param>
        private static void DisposeObservation(object?[] observation) {
            foreach (object? item in observation)
                (item as IDisposable)?.Dispose();
        }

        /// <summary>Writes the same hands into both envs: fresh lists on each side, which the env then appends to.</summary>
        /// <param name="pair">The pair; Gymnasium's env is reached through it.</param>
        /// <param name="dotnet">Gym.NET's env in the pair.</param>
        /// <param name="player">The player's cards.</param>
        /// <param name="dealer">The dealer's cards.</param>
        /// <exception cref="PythonException">Thrown when an assignment fails.</exception>
        private static void InjectHands(GymPair pair, BlackjackEnv dotnet, long[] player, long[] dealer) {
            pair.Python.WithUnwrapped(env => {
                using PyObject playerList = PyConvert.ToPython(player.Cast<object?>().ToList());
                using PyObject dealerList = PyConvert.ToPython(dealer.Cast<object?>().ToList());
                Python.gymnasium.envs.toy_text.blackjack.BlackjackEnv.player(env, playerList);
                Python.gymnasium.envs.toy_text.blackjack.BlackjackEnv.dealer(env, dealerList);
            });
            dotnet.Player = player.ToList();
            dotnet.Dealer = dealer.ToList();
        }

        /// <summary>Compares the hands and the face-up card's face and suit of both sides.</summary>
        /// <param name="what">Where in the run, for the report.</param>
        /// <param name="pair">The pair.</param>
        /// <param name="dotnet">Gym.NET's env in the pair.</param>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        private static void AssertHandsSame(string what, GymPair pair, BlackjackEnv dotnet) {
            ByteContract.AssertSame($"{what}: player", pair.Python.GetAttribute("player"), Outcome.Returned(ValueFacts.Of(dotnet.Player)));
            ByteContract.AssertSame($"{what}: dealer", pair.Python.GetAttribute("dealer"), Outcome.Returned(ValueFacts.Of(dotnet.Dealer)));
            ByteContract.AssertSame($"{what}: dealer_top_card_value_str", pair.Python.GetAttribute("dealer_top_card_value_str"), Outcome.Returned(ValueFacts.Of(dotnet.DealerTopCardValueStr)));
            ByteContract.AssertSame($"{what}: dealer_top_card_suit", pair.Python.GetAttribute("dealer_top_card_suit"), Outcome.Returned(ValueFacts.Of(dotnet.DealerTopCardSuit)));
        }
    }
}
