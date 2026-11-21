#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments;
using Gym.Envs;
using Gym.Rendering;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Utils {
    /// <summary>Gymnasium's <c>play</c> tests (<c>tests/utils/test_play.py</c>), run against <see cref="Play"/> and <see cref="PlayableGame"/>.</summary>
    /// <remarks>
    ///     <para>
    ///     The window is <see cref="NullEnvViewer"/>'s (DR-20), so nothing opens on screen; events are posted to
    ///     <see cref="EventQueue"/> as Gymnasium's posts them with <c>pygame.event.post</c>. Gymnasium's fixture
    ///     <c>close_pygame</c> (<c>pygame.quit()</c> after each test) is <see cref="Cleanup"/>'s <see cref="Display.Quit"/>.
    ///     The display, the queue and the viewer factory are process-wide, so the class never runs in parallel.
    ///     </para>
    ///     <para>
    ///     <c>PlayStatus</c> copies the observation it keeps: <c>play</c> lends its callback's arguments and releases them
    ///     afterwards (D19), where Python keeps the array alive by reference.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py">tests/utils/test_play.py</see>.</para>
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class TestPlay {
        /// <summary><c>RELEVANT_KEY_1 = ord("a")</c>.</summary>
        private const int RelevantKey1 = 'a';

        /// <summary><c>RELEVANT_KEY_2 = ord("d")</c>.</summary>
        private const int RelevantKey2 = 'd';

        /// <summary><c>IRRELEVANT_KEY = 1</c>.</summary>
        private const int IrrelevantKey = 1;

        /// <summary>The viewer factory before the test, restored by <see cref="Cleanup"/>.</summary>
        private EnvViewerFactory? _previousFactory;

        /// <summary>Points the process-wide viewer factory at <see cref="NullEnvViewer"/>, so no window opens.</summary>
        [TestInitialize]
        public void Initialize() {
            _previousFactory = HumanViewerFactory;
            HumanViewerFactory = NullEnvViewer.Factory;
            EventQueue.Clear();
        }

        /// <summary>Gymnasium's <c>close_pygame</c> fixture: quits the display (dropping pending events), then restores the factory.</summary>
        [TestCleanup]
        public void Cleanup() {
            Display.Quit();
            HumanViewerFactory = _previousFactory;
        }

        /// <summary>A <c>PlayableGame</c> of a tuple mapping has the tuples' keys as its relevant keys.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L69">tests/utils/test_play.py (test_play_relevant_keys)</see>.</remarks>
        [TestMethod]
        public void TestPlayRelevantKeys() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            AssertKeys(game.RelevantKeys, RelevantKey1, RelevantKey2);
        }

        /// <summary>Without a mapping, from the caller or the env, <c>PlayableGame</c> raises <see cref="MissingKeysToAction"/>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L75">tests/utils/test_play.py (test_play_relevant_keys_no_mapping)</see>.</remarks>
        [TestMethod]
        public void TestPlayRelevantKeysNoMapping() {
            using GenericTestEnv env = PlayableEnv();
            Assert.ThrowsExactly<MissingKeysToAction>(() => new PlayableGame(env));
        }

        /// <summary>An env attribute <c>get_keys_to_action</c> (a run-time attribute here) supplies the mapping.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L82">tests/utils/test_play.py (test_play_relevant_keys_with_env_attribute)</see>.</remarks>
        [TestMethod]
        public void TestPlayRelevantKeysWithEnvAttribute() {
            using GenericTestEnv env = PlayableEnv();
            PyAttributes.SetAttr(env, "get_keys_to_action", (Func<object>)DummyKeysToAction);
            var game = new PlayableGame(env);
            AssertKeys(game.RelevantKeys, RelevantKey1, RelevantKey2);
        }

        /// <summary>Without a zoom the video size is the frame's.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L90">tests/utils/test_play.py (test_video_size_no_zoom)</see>.</remarks>
        [TestMethod]
        public void TestVideoSizeNoZoom() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            Assert.AreEqual(FrameSize(env), game.VideoSize);
        }

        /// <summary>A zoom of 2.2 scales each side and truncates: <c>int(10 * 2.2)</c> is 22.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L96">tests/utils/test_play.py (test_video_size_zoom)</see>.</remarks>
        [TestMethod]
        public void TestVideoSizeZoom() {
            using GenericTestEnv env = PlayableEnv();
            const double zoom = 2.2;
            var game = new PlayableGame(env, DummyKeysToAction(), zoom);
            (int width, int height) = FrameSize(env);
            Assert.AreEqual(((int)(width * zoom), (int)(height * zoom)), game.VideoSize);
        }

        /// <summary>Escape, when it is not a mapped key, stops the game.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L103">tests/utils/test_play.py (test_keyboard_quit_event)</see>.</remarks>
        [TestMethod]
        public void TestKeyboardQuitEvent() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            Assert.IsTrue(game.Running);
            game.ProcessEvent(new Event(EventType.KeyDown, new { key = KeyCode.Escape }));
            Assert.IsFalse(game.Running);
        }

        /// <summary>A <c>QUIT</c> event stops the game.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L112">tests/utils/test_play.py (test_pygame_quit_event)</see>.</remarks>
        [TestMethod]
        public void TestPygameQuitEvent() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            Assert.IsTrue(game.Running);
            game.ProcessEvent(new Event(EventType.Quit));
            Assert.IsFalse(game.Running);
        }

        /// <summary>A relevant key's press is tracked.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L121">tests/utils/test_play.py (test_keyboard_relevant_keydown_event)</see>.</remarks>
        [TestMethod]
        public void TestKeyboardRelevantKeydownEvent() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            game.ProcessEvent(new Event(EventType.KeyDown, new { key = RelevantKey1 }));
            AssertKeys(game.PressedKeys, RelevantKey1);
        }

        /// <summary>An irrelevant key's press is not tracked.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L129">tests/utils/test_play.py (test_keyboard_irrelevant_keydown_event)</see>.</remarks>
        [TestMethod]
        public void TestKeyboardIrrelevantKeydownEvent() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            game.ProcessEvent(new Event(EventType.KeyDown, new { key = IrrelevantKey }));
            AssertKeys(game.PressedKeys);
        }

        /// <summary>A release untracks the key its press tracked.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L137">tests/utils/test_play.py (test_keyboard_keyup_event)</see>.</remarks>
        [TestMethod]
        public void TestKeyboardKeyupEvent() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            game.ProcessEvent(new Event(EventType.KeyDown, new { key = RelevantKey1 }));
            game.ProcessEvent(new Event(EventType.KeyUp, new { key = RelevantKey1 }));
            AssertKeys(game.PressedKeys);
        }

        /// <summary>
        ///     <c>play</c> over CartPole, driven by events its callback posts, takes the same actions as stepping the env by
        ///     hand: the no-op first, then each pressed key's action. For each key form (string, int, tuple) with the mapping
        ///     passed in, and through a wrapper's <c>get_keys_to_action</c>; it ends at the posted <c>QUIT</c>.
        /// </summary>
        /// <exception cref="AssertionError">Thrown for an unknown key type, as Gymnasium's <c>raise AssertionError()</c> (never: the list is fixed).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L147">tests/utils/test_play.py (test_play_loop_real_env)</see>.</remarks>
        [TestMethod]
        public void TestPlayLoopRealEnv() {
            const int seed = 42;
            const string envId = "CartPole-v1";

            foreach (bool applyWrapper in new[] { false, true }) {
                foreach (string keyType in new[] { "str", "int", "tuple" }) {
                    var callbackEvents = new List<Event> {
                        new(EventType.KeyDown, new { key = RelevantKey1 }), new(EventType.KeyUp, new { key = RelevantKey1 }),
                        new(EventType.KeyDown, new { key = RelevantKey2 }), new(EventType.KeyUp, new { key = RelevantKey2 }),
                        new(EventType.KeyDown, new { key = RelevantKey1 }), new(EventType.KeyUp, new { key = RelevantKey1 }),
                        new(EventType.KeyDown, new { key = RelevantKey1 }), new(EventType.KeyUp, new { key = RelevantKey1 }),
                        new(EventType.KeyDown, new { key = RelevantKey2 }), new(EventType.KeyUp, new { key = RelevantKey2 }),
                        new(EventType.Quit),
                    };
                    List<Event> keydownEvents = callbackEvents.Where(e => e.Type == EventType.KeyDown).ToList();

                    // Posts the next event after each step; after a release, keeps posting until a press (or the quit).
                    object?[] Callback(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info) {
                        Event pygameEvent = callbackEvents[0];
                        callbackEvents.RemoveAt(0);
                        EventQueue.Post(pygameEvent);
                        while (pygameEvent.Type == EventType.KeyUp) {
                            pygameEvent = callbackEvents[0];
                            callbackEvents.RemoveAt(0);
                            EventQueue.Post(pygameEvent);
                        }

                        return new[] { obsT, obsTp1, action, rew, terminated, truncated, info };
                    }

                    using var env = (Env<NDArray, NDArray>)Make(envId, new { render_mode = "rgb_array" }, disableEnvChecker: true);
                    env.Reset(seed).Observation.Dispose();

                    Dictionary<object, object?> keysToAction = keyType switch {
                        "tuple" => DummyKeysToAction(),
                        "str" => DummyKeysToActionStr(),
                        "int" => DummyKeysToActionInt(),
                        _ => throw new AssertionError(""),
                    };

                    // The first action is the no-op: the loop's first frame comes before any event can be posted. Python's int
                    // actions go through the untyped step, as play's do.
                    var untyped = (IEnv)env;
                    var obs = (NDArray)untyped.Step(0).Observation!;
                    foreach (Event e in keydownEvents) {
                        int key = (int)PyAttributes.GetAttr(e, "key")!;
                        object? action = keyType switch {
                            "tuple" => keysToAction[ValueTuple.Create(key)],
                            "str" => keysToAction[((char)key).ToString()],
                            _ => keysToAction[key],
                        };
                        obs.Dispose();
                        obs = (NDArray)untyped.Step(action).Observation!;
                    }

                    using var envPlay = (Env<NDArray, NDArray>)Make(envId, new { render_mode = "rgb_array" }, disableEnvChecker: true);
                    IEnv played = envPlay;
                    if (applyWrapper) {
                        // As Gymnasium's test, the wrapper goes around the reference env, which play resets with the same seed.
                        played = new KeysToActionWrapper(env, keysToAction);
                        Assert.IsTrue(PyAttributes.HasAttr(played, "get_keys_to_action"));
                    }

                    var status = new PlayStatus(Callback);
                    Play.Run(played, callback: status.Callback, keysToAction: applyWrapper ? null : keysToAction, seed: seed);

                    Assert.IsNotNull(status.LastObservation);
                    Assert.IsTrue(np.array_equal(status.LastObservation!, obs), $"apply_wrapper={applyWrapper}, key_type={keyType}");
                    status.LastObservation!.Dispose();
                    obs.Dispose();
                }
            }
        }

        /// <summary><c>play</c> without a mapping on an env without one raises <see cref="MissingKeysToAction"/>.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L233">tests/utils/test_play.py (test_play_no_keys)</see>.</remarks>
        [TestMethod]
        public void TestPlayNoKeys() {
            using IEnv env = Make("CartPole-v1", new { render_mode = "rgb_array" });
            Assert.ThrowsExactly<MissingKeysToAction>(() => Play.Run(env));
        }

        /// <summary><c>play</c> refuses an env without an array render mode, with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L238">tests/utils/test_play.py (test_wrong_render_mode)</see>.</remarks>
        [TestMethod]
        public void TestWrongRenderMode() {
            using IEnv env = Make("CartPole-v1");
            var error = Assert.ThrowsExactly<ValueError>(() => Play.Run(env, keysToAction: new Dictionary<object, object?>()));
            StringAssert.StartsWith(error.Message, "PlayableGame wrapper works only with rgb_array and rgb_array_list render modes");
        }

        /// <summary>
        ///     A resize fits the largest same-aspect size into the window, as integers, always scaling from the original size,
        ///     so ten round trips do not drift.
        /// </summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L246">tests/utils/test_play.py (test_pygame_window_resize)</see>.</remarks>
        [TestMethod]
        public void TestPygameWindowResize() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            Assert.AreEqual((10, 10), game.OriginalVideoSize);

            game.ProcessEvent(new Event(EventType.WindowResized, new { x = 25, y = 30 }));
            Assert.AreEqual((25, 25), game.VideoSize);

            for (int i = 0; i < 10; i++) {
                game.ProcessEvent(new Event(EventType.WindowResized, new { x = 50, y = 50 }));
                game.ProcessEvent(new Event(EventType.WindowResized, new { x = 15, y = 15 }));
            }

            Assert.AreEqual((15, 15), game.VideoSize);
        }

        /// <summary>A release without a press (a focus change, key bounce) does nothing.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L267">tests/utils/test_play.py (test_keyboard_keyup_unpressed_key_no_crash)</see>.</remarks>
        [TestMethod]
        public void TestKeyboardKeyupUnpressedKeyNoCrash() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            Assert.AreEqual(0, game.PressedKeys.Count);
            game.ProcessEvent(new Event(EventType.KeyUp, new { key = RelevantKey1 }));
            Assert.AreEqual(0, game.PressedKeys.Count);
        }

        /// <summary>Two presses (auto-repeat) and one release leave nothing pressed.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L280">tests/utils/test_play.py (test_keyboard_duplicate_keydown_event)</see>.</remarks>
        [TestMethod]
        public void TestKeyboardDuplicateKeydownEvent() {
            using GenericTestEnv env = PlayableEnv();
            var game = new PlayableGame(env, DummyKeysToAction());
            var down = new Event(EventType.KeyDown, new { key = RelevantKey1 });
            game.ProcessEvent(down);
            game.ProcessEvent(down);
            game.ProcessEvent(new Event(EventType.KeyUp, new { key = RelevantKey1 }));
            Assert.AreEqual(0, game.PressedKeys.Count);
        }

        /// <summary>A negative or zero zoom is refused at construction, with Gymnasium's message.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/utils/test_play.py#L295">tests/utils/test_play.py (test_invalid_zoom_validation)</see>.</remarks>
        [TestMethod]
        public void TestInvalidZoomValidation() {
            using GenericTestEnv env = PlayableEnv();
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => new PlayableGame(env, DummyKeysToAction(), zoom: -1.5)).Message, "Zoom must be a positive float");
            StringAssert.Contains(Assert.ThrowsExactly<ValueError>(() => new PlayableGame(env, DummyKeysToAction(), zoom: 0.0)).Message, "Zoom must be a positive float");
        }

        /// <summary>
        ///     <c>PlayableEnv</c>: a <see cref="GenericTestEnv"/> rendering <c>rgb_array</c> frames of ones, <c>(10, 10, 3)</c>
        ///     float64, as Gymnasium's <c>np.ones((10, 10, 3))</c>.
        /// </summary>
        /// <returns>The env; the caller disposes it.</returns>
        private static GenericTestEnv PlayableEnv() =>
            new(metadata: new Metadata { [Metadata.RenderModesKey] = new List<string> { "rgb_array" } },
                renderFunc: _ => new RenderResult.Rgb(np.ones(new Shape(10, 10, 3))), renderMode: "rgb_array");

        /// <summary><c>dummy_keys_to_action()</c>: <c>{(97,): 0, (100,): 1}</c>.</summary>
        /// <returns>A new mapping.</returns>
        private static Dictionary<object, object?> DummyKeysToAction() => new() { [ValueTuple.Create(RelevantKey1)] = 0, [ValueTuple.Create(RelevantKey2)] = 1 };

        /// <summary><c>dummy_keys_to_action_int()</c>: <c>{97: 0, 100: 1}</c>.</summary>
        /// <returns>A new mapping.</returns>
        private static Dictionary<object, object?> DummyKeysToActionInt() => new() { [RelevantKey1] = 0, [RelevantKey2] = 1 };

        /// <summary><c>dummy_keys_to_action_str()</c>: <c>{'a': 0, 'd': 1}</c>.</summary>
        /// <returns>A new mapping.</returns>
        private static Dictionary<object, object?> DummyKeysToActionStr() => new() { ["a"] = 0, ["d"] = 1 };

        /// <summary><c>env.render().shape[:2]</c> as <c>(width, height)</c>, the order <see cref="PlayableGame.VideoSize"/> uses (the frame is square here, as in Gymnasium's test).</summary>
        /// <param name="env">The env.</param>
        /// <returns>The size.</returns>
        private static (int Width, int Height) FrameSize(IEnv env) {
            using RenderResult rendered = env.Render();
            NDArray frame = ((RenderResult.Rgb)rendered).Frame;
            return ((int)frame.shape[1], (int)frame.shape[0]);
        }

        /// <summary>Asserts that a key set holds exactly the given key codes.</summary>
        /// <param name="keys">The set.</param>
        /// <param name="expected">The codes.</param>
        private static void AssertKeys(ISet<object?> keys, params int[] expected) {
            Assert.AreEqual(expected.Length, keys.Count, $"keys: {string.Join(", ", keys)}");
            foreach (int key in expected)
                Assert.IsTrue(keys.Contains(key), $"missing {key}");
        }

        /// <summary>
        ///     Gymnasium's <c>KeysToActionWrapper</c>: a wrapper whose <c>get_keys_to_action</c> returns a given mapping. The
        ///     method is typed (DR-45), key combinations to the env's actions, so a string key is held as its characters' codes
        ///     and a tuple as its items' codes, which is what <c>play</c> makes of either; play's behaviour, the test's subject,
        ///     is unchanged.
        /// </summary>
        private sealed class KeysToActionWrapper : Wrapper<NDArray, NDArray> {
            /// <summary>The mapping as key combinations and the actions' values, in the given order.</summary>
            private readonly List<(KeyCombination Keys, long Action)> _mapping;

            /// <summary>Wraps <paramref name="env"/>.</summary>
            /// <param name="env">The env.</param>
            /// <param name="keysToAction">The mapping to return: tuple, string or int keys, int actions.</param>
            /// <exception cref="ArgumentException">Thrown for a key of another kind.</exception>
            public KeysToActionWrapper(Env<NDArray, NDArray> env, Dictionary<object, object?> keysToAction) : base(env) =>
                _mapping = keysToAction.Select(entry => (Combination(entry.Key), Convert.ToInt64(entry.Value))).ToList();

            /// <summary>The wrapper's <c>get_keys_to_action</c>: the mapping, with new action arrays on every call (the caller owns them).</summary>
            /// <returns>The mapping.</returns>
            public override IReadOnlyDictionary<KeyCombination, NDArray> GetKeysToAction() =>
                _mapping.ToDictionary(entry => entry.Keys, entry => NDArray.Scalar(entry.Action));

            /// <summary>A key as the codes <c>play</c> reads from it: a string's characters, an int, a tuple's items.</summary>
            /// <param name="key">The key.</param>
            /// <returns>The combination.</returns>
            /// <exception cref="ArgumentException">Thrown for a key of another kind.</exception>
            private static KeyCombination Combination(object key) => key switch {
                string text => KeyCombination.FromText(text),
                int code => new KeyCombination(code),
                System.Runtime.CompilerServices.ITuple tuple => new KeyCombination(Enumerable.Range(0, tuple.Length)
                    .Select(i => tuple[i] is string character ? char.ConvertToUtf32(character, 0) : Convert.ToInt32(tuple[i])).ToArray()),
                _ => throw new ArgumentException($"A key is a string, an int or a tuple, not {key.GetType().Name}.", nameof(key)),
            };
        }

        /// <summary>
        ///     Gymnasium's <c>PlayStatus</c>: runs a data callback per step, sums the rewards and keeps the last observation, a
        ///     copy, since <c>play</c>'s arguments are borrowed.
        /// </summary>
        private sealed class PlayStatus {
            /// <summary>The data callback.</summary>
            private readonly Func<object?, object?, object?, double, bool, bool, Info, object?[]> _dataCallback;

            /// <summary>Wraps a data callback.</summary>
            /// <param name="callback">Returns the transition as a tuple.</param>
            public PlayStatus(Func<object?, object?, object?, double, bool, bool, Info, object?[]> callback) => _dataCallback = callback;

            /// <summary>The sum of the rewards (<c>cumulative_reward</c>).</summary>
            public double CumulativeReward { get; private set; }

            /// <summary>A copy of the last observation (<c>last_observation</c>); the caller disposes it.</summary>
            public NDArray? LastObservation { get; private set; }

            /// <summary>The <c>play</c> callback.</summary>
            /// <param name="obsT">The observation before.</param>
            /// <param name="obsTp1">The observation after.</param>
            /// <param name="action">The action.</param>
            /// <param name="rew">The reward.</param>
            /// <param name="terminated">The termination flag.</param>
            /// <param name="truncated">The truncation flag.</param>
            /// <param name="info">The info.</param>
            public void Callback(object? obsT, object? obsTp1, object? action, double rew, bool terminated, bool truncated, Info info) {
                object?[] transition = _dataCallback(obsT, obsTp1, action, rew, terminated, truncated, info);
                CumulativeReward += (double)transition[3]!;
                LastObservation?.Dispose();
                LastObservation = ((NDArray)transition[1]!).copy();
            }
        }
    }
}
