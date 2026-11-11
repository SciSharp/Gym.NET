#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Envs.Box2D {
    /// <summary>
    ///     What runs without Python about <see cref="CarRacingEnv"/>: its demo keyboard handler (<c>register_input</c> exists
    ///     only when Gymnasium's module runs as a script), the score's text format against Python's own outputs, and the
    ///     embedded score font over every text the score can show (DR-48); the parity suite compares the pixels live.
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
        ///     The score's text is Python's <c>f"{reward:04.0f}"</c>, each expectation printed by CPython 3.12: rounded half to
        ///     even, a negative zero keeps its sign, zeros pad after the sign to four characters, a large double shows every digit
        ///     of the integer it holds (not .NET's 15 significant digits), a NaN of either sign is <c>0nan</c> and the infinities
        ///     <c>0inf</c> and <c>-inf</c>.
        /// </summary>
        [TestMethod]
        public void FormatScore_IsPythonsZeroPaddedFormat() {
            const string maxValue = "179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368";
            // Pairs, not a dictionary: 0.0 and -0.0 are equal keys, and the negative zero is a case of its own.
            var cases = new List<(double Value, string Expected)> {
                (0.0, "0000"), (-0.0, "-000"), (-0.1, "-000"), (-0.5, "-000"), (-0.6, "-001"), (12.5, "0012"), (13.5, "0014"),
                (999.7, "1000"), (1000.0, "1000"), (-100.0, "-100"), (-1234.4, "-1234"), (54321.0, "54321"),
                (2.5, "0002"), (-2.5, "-002"), (9999.5, "10000"), (-999.5, "-1000"), (double.Epsilon, "0000"), (-double.Epsilon, "-000"),
                (9007199254740993.0, "9007199254740992"), (1e23, "99999999999999991611392"), (-1e20, "-100000000000000000000"),
                (double.MaxValue, maxValue), (double.MinValue, "-" + maxValue),
                (double.NaN, "0nan"), (BitConverter.Int64BitsToDouble(0x7FF8000000000000), "0nan"),
                (double.PositiveInfinity, "0inf"), (double.NegativeInfinity, "-inf"),
            };
            foreach ((double value, string expected) in cases)
                Assert.AreEqual(expected, CarRacingEnv.FormatScore(value), $"format({value:R}, '04.0f')");
        }

        /// <summary>
        ///     The embedded score font holds every character the score can show and lays out every kind of score text, up to
        ///     the 309 characters of <see cref="double.MaxValue"/>: one row of the line skip's 46 pixels each time. It refuses
        ///     the ligature sequence <c>fi</c> (pygame draws one glyph the data does not hold) and a character it has no glyph
        ///     for.
        /// </summary>
        [TestMethod]
        public void EmbeddedScoreFont_LaysOutEveryTextTheScoreCanShow() {
            ScoreFont font = ScoreFont.Embedded;
            foreach (string text in new[] { "0123456789-", "0nan", "0inf", "-inf", "-000", CarRacingEnv.FormatScore(double.MinValue) }) {
                (byte[] levels, int width, int height) = font.RenderShaded(text);
                Assert.AreEqual(46, height, $"'{text}' height");
                Assert.AreEqual(width * height, levels.Length, $"'{text}' levels");
                Assert.IsTrue(levels.Any(level => level == 255), $"'{text}' draws full coverage somewhere.");
            }

            Assert.ThrowsExactly<ArgumentException>(() => font.RenderShaded("fi"));
            Assert.ThrowsExactly<ArgumentException>(() => font.RenderShaded("0x"));
        }

        /// <summary>
        ///     CarRacing renders the observation and the 600×400 frame for every kind of score: NaN, both infinities and the
        ///     widest finite scores, whose box is clipped at both edges of the frame.
        /// </summary>
        [TestMethod]
        public void ExtremeScores_RenderInBothFrameKinds() {
            using var env = new CarRacingEnv(renderMode: "rgb_array");
            using NDArray observation = env.Reset(1).Observation;
            foreach (double score in new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity, double.MaxValue, double.MinValue }) {
                env.Reward = score;
                using NDArray action = np.array(new[] { 0.0f, 0.0f, 0.0f });
                StepResult<NDArray> step = env.Step(action);
                using (step.Observation)
                    Assert.IsTrue(env.ObservationSpace.Contains(step.Observation), $"Observation at score {score:R}.");
                var frame = (RenderResult.Rgb)env.Render();
                using (frame.Frame)
                    CollectionAssert.AreEqual(new long[] { 400, 600, 3 }, frame.Frame.shape, $"Frame at score {score:R}.");
            }
        }

        /// <summary>A key event as the interactive viewers post it: the type and pygame's <c>key</c> attribute.</summary>
        /// <param name="type">KeyDown or KeyUp.</param>
        /// <param name="key">The key code.</param>
        /// <returns>The event.</returns>
        private static Event KeyEvent(int type, int key) => new(type, new Info { ["key"] = key });
    }
}
