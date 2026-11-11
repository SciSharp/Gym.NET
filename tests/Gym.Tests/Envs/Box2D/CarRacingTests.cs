#nullable enable
using System;
using System.Collections.Generic;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.Box2D {
    /// <summary>
    ///     What the parity suite cannot compare live about <see cref="CarRacingEnv"/>: its demo keyboard handler
    ///     (<c>register_input</c> exists only when Gymnasium's module runs as a script), the score's text format, and the
    ///     frames without the score's glyph data (DR-27's data does not ship yet).
    /// </summary>
    /// <remarks>
    ///     The events go through the process-wide <see cref="EventQueue"/>, so the class runs its tests one at a time
    ///     (MSTest's default within a class) and clears the queue around each.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L801">gymnasium/envs/box2d/car_racing.py</see>.
    /// </remarks>
    [TestClass]
    [DoNotParallelize]
    public class CarRacingTests {
        /// <summary>Empties the shared event queue before each test, so no earlier test's events reach this one.</summary>
        [TestInitialize]
        public void ClearQueue() => EventQueue.Clear();

        /// <summary>
        ///     The demo's controls follow the arrow keys as Gymnasium's handler sets them: left and right steer -1 and +1, up gives
        ///     gas 1, down brakes 0.8; each release clears its control; Return asks for a restart, Escape and the close button for
        ///     quitting, and neither flag is ever cleared by the handler.
        /// </summary>
        [TestMethod]
        public void RegisterInput_SetsTheDemosControlsAndFlags() {
            using NDArray a = np.array(new[] { 0.0, 0.0, 0.0 });
            bool quit = false, restart = false;

            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Left));
            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Up));
            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Down));
            CarRacingEnv.RegisterInput(a, ref quit, ref restart);
            CollectionAssert.AreEqual(new[] { -1.0, 1.0, 0.8 }, a.ToArray<double>());
            Assert.IsFalse(quit || restart);

            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Right));
            EventQueue.Post(KeyEvent(EventType.KeyUp, KeyCode.Up));
            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Return));
            CarRacingEnv.RegisterInput(a, ref quit, ref restart);
            CollectionAssert.AreEqual(new[] { 1.0, 0.0, 0.8 }, a.ToArray<double>());
            Assert.IsTrue(restart);
            Assert.IsFalse(quit);

            EventQueue.Post(KeyEvent(EventType.KeyUp, KeyCode.Right));
            EventQueue.Post(KeyEvent(EventType.KeyUp, KeyCode.Down));
            EventQueue.Post(new Event(EventType.Quit));
            CarRacingEnv.RegisterInput(a, ref quit, ref restart);
            CollectionAssert.AreEqual(new[] { 0.0, 0.0, 0.0 }, a.ToArray<double>());
            Assert.IsTrue(quit && restart, "Neither flag is cleared by the handler.");

            quit = false;
            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Escape));
            CarRacingEnv.RegisterInput(a, ref quit, ref restart);
            Assert.IsTrue(quit);
        }

        /// <summary>The handler refuses an action it cannot write raw float64 controls into, before reading any event.</summary>
        [TestMethod]
        public void RegisterInput_RefusesANonFloat64OrShortAction() {
            bool quit = false, restart = false;
            using NDArray float32 = np.array(new[] { 0f, 0f, 0f });
            using NDArray shortAction = np.array(new[] { 0.0, 0.0 });
            EventQueue.Post(KeyEvent(EventType.KeyDown, KeyCode.Escape));
            Assert.ThrowsExactly<ArgumentException>(() => CarRacingEnv.RegisterInput(float32, ref quit, ref restart));
            Assert.ThrowsExactly<ArgumentException>(() => CarRacingEnv.RegisterInput(shortAction, ref quit, ref restart));
            Assert.AreEqual(1, EventQueue.Count, "The refused calls read no event.");
        }

        /// <summary>
        ///     The score's text is Python's <c>f"{reward:04.0f}"</c>: rounded half to even, a negative zero keeps its sign, and
        ///     zeros pad after the sign to four characters.
        /// </summary>
        [TestMethod]
        public void FormatScore_IsPythonsZeroPaddedFormat() {
            // Pairs, not a dictionary: 0.0 and -0.0 are equal keys, and the negative zero is a case of its own.
            var cases = new List<(double Value, string Expected)> {
                (0.0, "0000"), (-0.0, "-000"), (-0.1, "-000"), (-0.5, "-000"), (-0.6, "-001"), (12.5, "0012"), (13.5, "0014"),
                (999.7, "1000"), (1000.0, "1000"), (-100.0, "-100"), (-1234.4, "-1234"), (54321.0, "54321"),
            };
            foreach ((double value, string expected) in cases)
                Assert.AreEqual(expected, CarRacingEnv.FormatScore(value), $"format({value:R}, '04.0f')");
        }

        /// <summary>
        ///     Without glyph data CarRacing still steps and renders, leaving the score's area black: the observation stays in
        ///     its space and the 600×400 frame has its size.
        /// </summary>
        [TestMethod]
        public void WithoutScoreGlyphs_TheEnvStillStepsAndRenders() {
            ScoreFont? installed = ScoreFont.Current;
            ScoreFont.Current = null;
            try {
                using var env = new CarRacingEnv(renderMode: "rgb_array");
                using NDArray observation = env.Reset(1).Observation;
                Assert.IsTrue(env.ObservationSpace.Contains(observation));
                using NDArray action = np.array(new[] { 0.0f, 1.0f, 0.0f });
                StepResult<NDArray> step = env.Step(action);
                step.Observation.Dispose();
                var frame = (RenderResult.Rgb)env.Render();
                using (frame.Frame)
                    CollectionAssert.AreEqual(new long[] { 400, 600, 3 }, frame.Frame.shape);
            } finally {
                ScoreFont.Current = installed;
            }
        }

        /// <summary>A key event as the interactive viewers post it: the type and pygame's <c>key</c> attribute.</summary>
        /// <param name="type">KeyDown or KeyUp.</param>
        /// <param name="key">The key code.</param>
        /// <returns>The event.</returns>
        private static Event KeyEvent(int type, int key) => new(type, new Info { ["key"] = key });
    }
}
