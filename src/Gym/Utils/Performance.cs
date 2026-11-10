#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Vector;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Runtime benchmarks for finding performance problems (Gymnasium's <c>gymnasium.utils.performance</c>): how many
    ///     steps, vector steps, constructions or renders an env manages per second over a target duration. Each runs until
    ///     the clock passes the target, so it runs slightly over it, and reports the count divided by the time measured.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The clock is <see cref="Time"/>, Python's <c>time.time()</c> by default; Gymnasium's tests replace
    ///     <c>time.time</c> to script the timestamps, and Gym.NET's replace <see cref="Time"/> the same way (a process-wide
    ///     setting: restore it afterwards). Unlike Python, the benchmarks dispose what they drop (the samples, observations
    ///     and frames), so the garbage collector's work does not leak into the measurement.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/performance.py">gymnasium/utils/performance.py</see>.</para>
    /// </remarks>
    public static class Performance {
        /// <summary>The Unix epoch, which <see cref="Time"/>'s default counts from as <c>time.time()</c> does.</summary>
        private static readonly DateTime Epoch = new(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        /// <summary>
        ///     The clock in seconds (Python's <c>time.time()</c>, wall-clock seconds since the Unix epoch); replaceable, as
        ///     Gymnasium's tests monkeypatch <c>performance.time.time</c>. Process-wide: a test that replaces it restores it.
        /// </summary>
        public static Func<double> Time { get; set; } = () => (DateTime.UtcNow - Epoch).Ticks / (double)TimeSpan.TicksPerSecond;

        /// <summary>
        ///     Measures the step throughput of an env (<c>benchmark_step</c>): resets with <paramref name="seed"/>, then steps
        ///     with sampled actions (resetting at every episode end) until <paramref name="targetDuration"/> seconds passed.
        /// </summary>
        /// <param name="env">The env; its action space is sampled once before the clock starts, as Gymnasium does.</param>
        /// <param name="targetDuration">The duration in seconds; the benchmark goes slightly over it.</param>
        /// <param name="seed">The first reset's seed (the action space is not seeded, as in Gymnasium).</param>
        /// <returns>The average steps per second.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="DivideByZeroException">Thrown when no time passed (a negative target), as Python's <c>float division by zero</c>.</exception>
        public static double BenchmarkStep(IEnv env, double targetDuration = 5, BigInteger? seed = null) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            long steps = 0;
            double end;
            Release(env.Reset(seed));
            Release(env.ActionSpace.Sample());
            double start = Time();

            while (true) {
                steps++;
                object? action = env.ActionSpace.Sample();
                StepResult<object?> result = env.Step(action);
                Release(action);
                Release(result.Observation);

                if (result.Terminated || result.Truncated)
                    Release(env.Reset());

                end = Time();
                if (end - start > targetDuration)
                    break;
            }

            return Divide(steps, end - start);
        }

        /// <summary>
        ///     Measures the step throughput of a vector env in sub-env steps per second (<c>benchmark_vector_step</c>): seeds
        ///     the action space, resets, takes one warm-up step outside the measurement and resets again, then steps until
        ///     <paramref name="targetDuration"/> seconds passed. Under <c>NEXT_STEP</c> the steps that only reset a sub-env do
        ///     not count; under <c>DISABLED</c> the ended sub-envs are reset through <c>reset_mask</c>.
        /// </summary>
        /// <param name="env">The vector env.</param>
        /// <param name="targetDuration">The duration in seconds; the benchmark can run longer while its last step completes.</param>
        /// <param name="seed">The seed of the action space and of both resets.</param>
        /// <returns>The number of individual environment steps per second.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="DivideByZeroException">Thrown when no time passed, as Python's <c>float division by zero</c>.</exception>
        public static double BenchmarkVectorStep(IVectorEnv env, double targetDuration = 5, BigInteger? seed = null) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            env.ActionSpace.Seed(seed);
            // Some environments lazily initialize during their first reset.
            Release(env.Reset(seed));

            // Warm up lazy initialization, including JIT compilation, outside the benchmark.
            object? warmUp = env.ActionSpace.Sample();
            Release(env.Step(warmUp));
            Release(warmUp);
            Release(env.Reset(seed));

            long steps = 0;
            double end;
            double start = Time();
            object? autoresetMode = env.Metadata.TryGetValue("autoreset_mode", out object? mode) ? mode : AutoresetMode.NextStep;
            var previousDone = new bool[env.NumEnvs];

            while (true) {
                object? actions = env.ActionSpace.Sample();
                VectorStepResult<object?> result = env.Step(actions);
                Release(actions);

                using NDArray done = np.logical_or(result.Terminations, result.Truncations);
                if (Equals(autoresetMode, AutoresetMode.NextStep)) {
                    steps += env.NumEnvs - Count(previousDone);
                    previousDone = done.ToArray<bool>();
                } else {
                    steps += env.NumEnvs;
                }

                if (Equals(autoresetMode, AutoresetMode.Disabled) && np.any(done))
                    Release(env.Reset(options: new Info { ["reset_mask"] = done }));

                Release(result);
                end = Time();
                if (end - start > targetDuration)
                    break;
            }

            return Divide(steps, end - start);
        }

        /// <summary>
        ///     Measures construction and first reset (<c>benchmark_init</c>): builds an env with <paramref name="envLambda"/>
        ///     and resets it, again and again, until <paramref name="targetDuration"/> seconds passed.
        /// </summary>
        /// <param name="envLambda">Builds an env; each is disposed after its reset (Python leaves them to the garbage collector).</param>
        /// <param name="targetDuration">The duration in seconds; the benchmark goes slightly over it.</param>
        /// <param name="seed">The seed of each first reset.</param>
        /// <returns>The constructions per second.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envLambda"/> is <see langword="null"/>.</exception>
        /// <exception cref="DivideByZeroException">Thrown when no time passed, as Python's <c>float division by zero</c>.</exception>
        public static double BenchmarkInit(Func<IEnv> envLambda, double targetDuration = 5, BigInteger? seed = null) {
            if (envLambda is null)
                throw new ArgumentNullException(nameof(envLambda));
            long inits = 0;
            double end;
            double start = Time();
            while (true) {
                inits++;
                using (IEnv env = envLambda())
                    Release(env.Reset(seed));

                end = Time();
                if (end - start > targetDuration)
                    break;
            }

            return Divide(inits, end - start);
        }

        /// <summary>Measures <c>render()</c> (<c>benchmark_render</c>): renders again and again until <paramref name="targetDuration"/> seconds passed.</summary>
        /// <param name="env">The env, built with a render mode that returns frames (not <c>human</c>, as in Gymnasium).</param>
        /// <param name="targetDuration">The duration in seconds; the benchmark goes slightly over it.</param>
        /// <returns>The renders per second.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="DivideByZeroException">Thrown when no time passed, as Python's <c>float division by zero</c>.</exception>
        public static double BenchmarkRender(IEnv env, double targetDuration = 5) {
            if (env is null)
                throw new ArgumentNullException(nameof(env));
            long renders = 0;
            double end;
            double start = Time();
            while (true) {
                renders++;
                Release(env.Render());

                end = Time();
                if (end - start > targetDuration)
                    break;
            }

            return Divide(renders, end - start);
        }

        /// <summary>Python's <c>count / length</c> of an int by a float: <c>float division by zero</c> for a zero length.</summary>
        /// <param name="count">The count.</param>
        /// <param name="length">The seconds measured.</param>
        /// <returns>The rate.</returns>
        /// <exception cref="DivideByZeroException">Thrown for a zero <paramref name="length"/>.</exception>
        private static double Divide(long count, double length) {
            if (length == 0)
                throw new DivideByZeroException("float division by zero");
            return count / length;
        }

        /// <summary><c>int(np.count_nonzero(flags))</c>.</summary>
        /// <param name="flags">The flags.</param>
        /// <returns>How many are set.</returns>
        private static int Count(bool[] flags) {
            int count = 0;
            foreach (bool flag in flags) {
                if (flag)
                    count++;
            }

            return count;
        }

        /// <summary>Disposes a dropped value's arrays: an array, or the arrays inside a tuple, a list of results or a dict.</summary>
        /// <param name="value">The value.</param>
        private static void Release(object? value) {
            switch (value) {
                case NDArray array:
                    array.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        Release(item);
                    break;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? item in dict.Values)
                        Release(item);
                    break;
                case RenderResult.Rgb rgb:
                    rgb.Frame.Dispose();
                    break;
                case RenderResult.Frames frames:
                    foreach (RenderResult frame in frames.Items)
                        Release(frame);
                    break;
                case IReadOnlyList<RenderResult> renders:
                    foreach (RenderResult render in renders)
                        Release(render);
                    break;
            }
        }

        /// <summary>Disposes a dropped reset's observation and infos.</summary>
        /// <param name="result">The reset.</param>
        private static void Release(ResetResult<object?> result) {
            Release(result.Observation);
            Release(result.Info);
        }

        /// <summary>Disposes a dropped vector reset's observations and infos.</summary>
        /// <param name="result">The reset.</param>
        private static void Release(VectorResetResult<object?> result) {
            Release(result.Observations);
            Release(result.Infos);
        }

        /// <summary>Disposes a dropped vector step's arrays.</summary>
        /// <param name="result">The step.</param>
        private static void Release(VectorStepResult<object?> result) {
            Release(result.Observations);
            result.Rewards.Dispose();
            result.Terminations.Dispose();
            result.Truncations.Dispose();
            Release(result.Infos);
        }
    }
}
