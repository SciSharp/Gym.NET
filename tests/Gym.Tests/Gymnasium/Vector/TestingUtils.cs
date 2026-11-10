#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Gymnasium.Vector {
    /// <summary>Gymnasium's testing utilities for the vector envs (<c>tests/vector/testing_utils.py</c>).</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/vector/testing_utils.py">tests/vector/testing_utils.py</see>.</remarks>
    internal static class TestingUtils {
        /// <summary>The frame height of <see cref="SlowEnv"/>'s observations (<c>HEIGHT</c>).</summary>
        internal const int Height = 64;

        /// <summary>The frame width of <see cref="SlowEnv"/>'s observations (<c>WIDTH</c>).</summary>
        internal const int Width = 64;

        /// <summary>
        ///     <c>make_env(env_name, seed, **kwargs)</c>: a function that makes the env without the checker, seeds its action
        ///     space and resets it with <paramref name="seed"/>.
        /// </summary>
        /// <param name="envName">The id.</param>
        /// <param name="seed">The seed.</param>
        /// <param name="kwargs">The env's constructor arguments by Python name, or <see langword="null"/>.</param>
        /// <returns>The function; every call builds a new env.</returns>
        internal static Func<IEnv> MakeEnv(string envName, long seed, object? kwargs = null) => () => {
            IEnv env = Make(envName, kwargs, disableEnvChecker: true);
            env.ActionSpace.Seed(seed);
            Release(env.Reset(seed));
            return env;
        };

        /// <summary><c>make_slow_env(slow_reset, seed)</c>: a function that makes a <see cref="SlowEnv"/> and resets it with <paramref name="seed"/>.</summary>
        /// <param name="slowReset">Seconds every reset sleeps.</param>
        /// <param name="seed">The seed.</param>
        /// <returns>The function.</returns>
        internal static Func<IEnv> MakeSlowEnv(double slowReset, long seed) => () => {
            var env = new SlowEnv(slowReset);
            Release(env.Reset(seed));
            return env;
        };

        /// <summary><c>make_custom_space_env(seed)</c>: a function that makes a <see cref="CustomSpaceEnv"/> and resets it.</summary>
        /// <param name="seed">The seed.</param>
        /// <returns>The function.</returns>
        internal static Func<IEnv> MakeCustomSpaceEnv(long seed) => () => {
            var env = new CustomSpaceEnv();
            env.Reset(seed);
            return env;
        };

        /// <summary>
        ///     <c>assert_rng_equal(rng_1, rng_2)</c>: the two generators are in the same state, field by field of NumPy's
        ///     <c>bit_generator.state</c> dict (<c>state</c>, <c>inc</c>, <c>has_uint32</c>, <c>uinteger</c>).
        /// </summary>
        /// <param name="rng1">The first generator.</param>
        /// <param name="rng2">The second.</param>
        /// <exception cref="AssertFailedException">Thrown when the states differ.</exception>
        /// <exception cref="InvalidCastException">Thrown for a generator over another bit generator than PCG64, which no Gymnasium space uses.</exception>
        /// <remarks>
        ///     NumSharp's <see cref="PCG64.State"/> is a class without value equality, so comparing the state objects (as
        ///     Python compares the dicts) would compare references: the fields are compared instead.
        /// </remarks>
        internal static void AssertRngEqual(Generator rng1, Generator rng2) {
            var state1 = (PCG64.State)rng1.bit_generator.state;
            var state2 = (PCG64.State)rng2.bit_generator.state;
            Assert.IsTrue(state1.state == state2.state && state1.inc == state2.inc && state1.has_uint32 == state2.has_uint32 && state1.uinteger == state2.uinteger,
                $"the generators' states differ: ({state1.state}, {state1.inc}, {state1.has_uint32}, {state1.uinteger}) vs ({state2.state}, {state2.inc}, {state2.has_uint32}, {state2.uinteger})");
        }

        /// <summary>Disposes the observation of a reset the caller does not read.</summary>
        /// <typeparam name="TObs">The observation type.</typeparam>
        /// <param name="result">The reset.</param>
        private static void Release<TObs>(ResetResult<TObs> result) => (result.Observation as IDisposable)?.Dispose();

        /// <summary>
        ///     A custom slow environment (<c>SlowEnv</c>): every reset sleeps <c>slow_reset</c> seconds, every step sleeps as
        ///     many seconds as its action; observations are random 64 × 64 frames.
        /// </summary>
        internal sealed class SlowEnv : Env<NDArray, NDArray> {
            /// <summary>A slow env (<c>SlowEnv(slow_reset=0.3)</c>).</summary>
            /// <param name="slowReset">Seconds every reset sleeps.</param>
            public SlowEnv(double slowReset = 0.3) {
                SlowReset = slowReset;
                ObservationSpace = new Box(0, 255, new Shape(Height, Width, 3), np.uint8);
                ActionSpace = new Box(0.0, 1.0, NumSharp.Shape.Scalar, np.float32);
            }

            /// <summary>Seconds every reset sleeps (<c>slow_reset</c>).</summary>
            public double SlowReset { get; }

            /// <summary>Resets the environment with a time sleep.</summary>
            /// <param name="seed">The seed.</param>
            /// <param name="options">Ignored.</param>
            /// <returns>A random frame and an empty info.</returns>
            public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                SeedNpRandom(seed);
                if (SlowReset > 0)
                    Thread.Sleep(TimeSpan.FromSeconds(SlowReset));
                return new ResetResult<NDArray>(ObservationSpace.Sample(), new Info());
            }

            /// <summary>Steps through the environment with a time sleep of <paramref name="action"/> seconds.</summary>
            /// <param name="action">The seconds to sleep.</param>
            /// <returns>A random frame, a reward of 0, and no end.</returns>
            public override StepResult<NDArray> Step(NDArray action) {
                Thread.Sleep(TimeSpan.FromSeconds(PyBuiltins.Float(action)));
                return new StepResult<NDArray>(ObservationSpace.Sample(), 0.0, false, false, new Info());
            }
        }

        /// <summary>A minimal custom observation space (<c>CustomSpace</c>): samples are integers in [0, 10); equal to any other custom space.</summary>
        internal sealed class CustomSpace : Space<NDArray> {
            /// <summary>A space with no shape or dtype, as Gymnasium's bare <c>Space()</c>.</summary>
            public CustomSpace() : base(null, null, null) { }

            /// <summary>A custom space is not flattenable.</summary>
            public override bool IsNpFlattenable => false;

            /// <summary>Generates a sample from the custom space: <c>np_random.integers(0, 10, ())</c>.</summary>
            /// <param name="mask">Ignored.</param>
            /// <param name="probability">Ignored.</param>
            /// <returns>A 0-d int64 array.</returns>
            public override NDArray Sample(object? mask = null, object? probability = null) => NpRandom.integers(0, 10, NumSharp.Shape.Scalar);

            /// <summary>Check if the element <paramref name="x"/> is contained within the space: <c>0 &lt;= x &lt;= 10</c>.</summary>
            /// <param name="x">The element.</param>
            /// <returns>Whether it is a number in [0, 10].</returns>
            public override bool Contains(object? x) {
                double value = PyBuiltins.Float(x);
                return 0 <= value && value <= 10;
            }

            /// <summary>Check if the two spaces are equal: any custom space is.</summary>
            /// <param name="obj">The other object.</param>
            /// <returns>Whether it is a <see cref="CustomSpace"/>.</returns>
            public override bool Equals(object? obj) => obj is CustomSpace;

            /// <summary>A hash consistent with <see cref="Equals(object)"/>.</summary>
            /// <returns>The same value for every custom space.</returns>
            public override int GetHashCode() => typeof(CustomSpace).GetHashCode();
        }

        /// <summary>An environment with custom spaces for observation and action spaces (<c>CustomSpaceEnv</c>): observations are strings.</summary>
        internal sealed class CustomSpaceEnv : Env<object?, object?> {
            /// <summary>Initialise the environment.</summary>
            public CustomSpaceEnv() {
                ((IEnv)this).ObservationSpace = new CustomSpace();
                ((IEnv)this).ActionSpace = new CustomSpace();
            }

            /// <summary>Resets the environment.</summary>
            /// <param name="seed">The seed.</param>
            /// <param name="options">Ignored.</param>
            /// <returns><c>"reset"</c> and an empty info.</returns>
            public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
                SeedNpRandom(seed);
                return new ResetResult<object?>("reset", new Info());
            }

            /// <summary>Steps through the environment: the observation names the action (<c>f"step({action:s})"</c>).</summary>
            /// <param name="action">The action, a string.</param>
            /// <returns><c>"step(action)"</c>, a reward of 0, and no end.</returns>
            public override StepResult<object?> Step(object? action) => new($"step({(string)action!})", 0.0, false, false, new Info());
        }
    }
}
