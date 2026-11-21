#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Utils;
using Gym.Vector;
using NumSharp;

namespace Gym.Wrappers.Vector {
    /// <summary>
    ///     Records each sub-env's episode return, length and duration, as Gymnasium's vector <c>RecordEpisodeStatistics</c>
    ///     does: a step that ends episodes adds <c>infos["episode"] = {"r", "l", "t"}</c> (zeros for the sub-envs that did not
    ///     end) and <c>infos["_episode"]</c> (which did), and the last <c>buffer_length</c> episodes stay in the queues.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     With <c>NEXT_STEP</c> the autoreset step after an episode's end starts the next episode's statistics instead of
    ///     counting; with <c>SAME_STEP</c> every step counts and an ended episode's statistics restart at once. A
    ///     <c>reset_mask</c> resets only the masked sub-envs' statistics, after Gymnasium's checks of the mask. The time is
    ///     wall-clock seconds (Python's <c>time.perf_counter()</c>, rounded to 6 digits), so it differs between runs: parity
    ///     comparisons leave the <c>t</c> entry out.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/common.py#L19">gymnasium/wrappers/vector/common.py</see>.</para>
    /// </remarks>
    [PyName("RecordEpisodeStatistics")]
    public class RecordEpisodeStatisticsVector<TObs, TAct> : VectorWrapper<TObs, TAct> {
        /// <summary>The info key the statistics go under (<c>_stats_key</c>).</summary>
        [PyName("_stats_key")]
        private readonly string _statsKey;

        /// <summary>The env's autoreset mode (<c>_autoreset_mode</c>).</summary>
        [PyName("_autoreset_mode")]
        private readonly AutoresetMode _autoresetMode;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="bufferLength">How many episodes the queues keep.</param>
        /// <param name="statsKey">The info key for the statistics.</param>
        /// <exception cref="TypeError">Thrown when the env's autoreset mode is not an <see cref="AutoresetMode"/>, with Gymnasium's message, or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for a negative <paramref name="bufferLength"/>: <c>maxlen must be non-negative</c>.</exception>
        public RecordEpisodeStatisticsVector(VectorEnv<TObs, TAct> env, int bufferLength = 100, string statsKey = "episode") : base(env) {
            _statsKey = statsKey;
            _autoresetMode = Vectorizing.AutoresetModeOf(env,
                $"{this} is missing `autoreset_mode` tag in its metadata, therefore, `RecordEpisodeStatistics` is assuming that the environment uses `AutoresetMode.NEXT_STEP`. See `https://farama.org/Vector-Autoreset-Mode` for more information on autoreset modes.");
            EpisodeStartTimes = np.zeros(new Shape(NumEnvs));
            EpisodeReturns = np.zeros(new Shape(NumEnvs));
            EpisodeLengths = np.zeros(new Shape(NumEnvs), np.int64);
            PrevDones = np.zeros(new Shape(NumEnvs), np.@bool);
            TimeQueue = new Deque<double>(bufferLength);
            ReturnQueue = new Deque<double>(bufferLength);
            LengthQueue = new Deque<long>(bufferLength);
        }

        /// <summary>The number of episodes ended so far (<c>episode_count</c>).</summary>
        public long EpisodeCount { get; private set; }

        /// <summary>When each sub-env's episode began, in <c>perf_counter</c> seconds (<c>episode_start_times</c>).</summary>
        public NDArray EpisodeStartTimes { get; private set; }

        /// <summary>Each sub-env's return so far, float64 (<c>episode_returns</c>).</summary>
        public NDArray EpisodeReturns { get; private set; }

        /// <summary>Each sub-env's length so far, int64 (<c>episode_lengths</c>).</summary>
        public NDArray EpisodeLengths { get; private set; }

        /// <summary>Which sub-envs ended at the last step (<c>prev_dones</c>).</summary>
        public NDArray PrevDones { get; private set; }

        /// <summary>The durations of the last episodes (<c>time_queue</c>).</summary>
        public Deque<double> TimeQueue { get; }

        /// <summary>The returns of the last episodes (<c>return_queue</c>).</summary>
        public Deque<double> ReturnQueue { get; }

        /// <summary>The lengths of the last episodes (<c>length_queue</c>).</summary>
        public Deque<long> LengthQueue { get; }

        /// <summary>Resets the env and the statistics: all of them, or only the masked sub-envs' for a <c>reset_mask</c>.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="TypeError">Thrown for a mask that is not a bool array, with Gymnasium's messages.</exception>
        /// <exception cref="ValueError">Thrown for a mask of the wrong shape or with no true value, with Gymnasium's messages.</exception>
        public override VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = base.Reset(seed, options);
            ResetStatistics(options);
            return result;
        }

        /// <summary>Resets the env with one seed per sub-env, and the statistics as <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/> does.</summary>
        /// <param name="seeds">The seeds, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="TypeError">Thrown for a mask that is not a bool array.</exception>
        /// <exception cref="ValueError">Thrown for a mask of the wrong shape or with no true value.</exception>
        public override VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = base.Reset(seeds, options);
            ResetStatistics(options);
            return result;
        }

        /// <summary>Steps the env, updates the statistics, and adds them to the infos of a step that ends episodes.</summary>
        /// <param name="actions">The actions, passed on.</param>
        /// <returns>The env's transition, its infos extended when episodes ended.</returns>
        /// <exception cref="ValueError">Thrown when the infos already hold the statistics key, with Gymnasium's message.</exception>
        public override VectorStepResult<TObs> Step(TAct actions) {
            VectorStepResult<TObs> result = base.Step(actions);
            Info infos = result.Infos;

            if (_autoresetMode == AutoresetMode.SameStep) {
                // Sub-envs reset within the step that ends them, so every step counts towards an episode.
                using (NDArray returns = EpisodeReturns + result.Rewards)
                    Replace(EpisodeReturns, returns);
                using (NDArray lengths = EpisodeLengths + 1)
                    Replace(EpisodeLengths, lengths);
            } else {
                // The step after an end resets the sub-env and starts the next episode's statistics.
                using NDArray continuing = np.logical_not(PrevDones);
                EpisodeReturns[PrevDones] = 0;
                using (NDArray kept = EpisodeReturns[continuing])
                using (NDArray rewards = result.Rewards[continuing])
                using (NDArray summed = kept + rewards)
                    EpisodeReturns[continuing] = summed;
                EpisodeLengths[PrevDones] = 0;
                using (NDArray lengths = EpisodeLengths[continuing])
                using (NDArray incremented = lengths + 1)
                    EpisodeLengths[continuing] = incremented;
                EpisodeStartTimes[PrevDones] = PerfCounter();
            }

            NDArray dones = np.logical_or(result.Terminations, result.Truncations);
            PrevDones.Dispose();
            PrevDones = dones;
            long numDones = dones.ToArray<bool>().LongCount(done => done);
            if (numDones > 0) {
                if (infos.ContainsKey(_statsKey) || infos.ContainsKey($"_{_statsKey}"))
                    throw new ValueError($"Attempted to add episode stats with key '{_statsKey}' but this key already exists in info: {PyBuiltins.Repr(infos.Keys.ToList())}");

                // np.round(time.perf_counter() - self.episode_start_times, 6): NumPy's scaled round, not Python's.
                using NDArray now = np.full(new Shape(NumEnvs), PerfCounter());
                using NDArray elapsed = now - EpisodeStartTimes;
                using NDArray times = np.around(elapsed, 6);
                double[] episodeTimeLength = times.ToArray<double>();
                infos[_statsKey] = new Info {
                    ["r"] = np.where(dones, EpisodeReturns, 0.0),
                    ["l"] = np.where(dones, EpisodeLengths, 0),
                    ["t"] = np.where(dones, times, 0.0),
                };
                infos[$"_{_statsKey}"] = dones.copy();
                EpisodeCount += numDones;

                bool[] ended = dones.ToArray<bool>();
                double[] returnsNow = EpisodeReturns.ToArray<double>();
                long[] lengthsNow = EpisodeLengths.ToArray<long>();
                for (int i = 0; i < NumEnvs; i++) {
                    if (!ended[i])
                        continue;
                    TimeQueue.Append(episodeTimeLength[i]);
                    ReturnQueue.Append(returnsNow[i]);
                    LengthQueue.Append(lengthsNow[i]);
                }

                if (_autoresetMode == AutoresetMode.SameStep) {
                    // The done sub-envs have already been reset: their statistics restart now, not at the next step.
                    EpisodeReturns[dones] = 0;
                    EpisodeLengths[dones] = 0;
                    EpisodeStartTimes[dones] = PerfCounter();
                }
            }

            return result;
        }

        /// <summary>Releases the statistics' arrays, then disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            EpisodeStartTimes.Dispose();
            EpisodeReturns.Dispose();
            EpisodeLengths.Dispose();
            PrevDones.Dispose();
        }

        /// <summary>Resets the statistics after a reset: the masked sub-envs' for a <c>reset_mask</c> (popped from the options), else all.</summary>
        /// <param name="options">The reset options.</param>
        /// <exception cref="TypeError">Thrown for a mask that is not a bool array.</exception>
        /// <exception cref="ValueError">Thrown for a mask of the wrong shape or with no true value.</exception>
        private void ResetStatistics(IReadOnlyDictionary<string, object?>? options) {
            if (options is not null && options.TryGetValue("reset_mask", out object? value)) {
                // options.pop("reset_mask"): the caller's dict loses the key when it can.
                (options as IDictionary<string, object?>)?.Remove("reset_mask");
                if (value is not NDArray mask)
                    throw new TypeError($"`options['reset_mask']` must be a numpy array, got {PyBuiltins.TypeRepr(value)}");
                if (mask.ndim != 1 || mask.shape[0] != NumEnvs)
                    throw new ValueError($"`options['reset_mask']` must have shape `({NumEnvs},)`, got {PyBuiltins.Repr(mask.shape.Select(d => (object?)d).ToArray())}");
                if (mask.typecode != NPTypeCode.Boolean)
                    throw new TypeError($"`options['reset_mask']` must have `dtype=np.bool_`, got {mask.dtype.name}");
                if (!np.any(mask))
                    throw new ValueError($"`options['reset_mask']` must contain a boolean array with at least one True value, got reset_mask={PyBuiltins.Str(mask)}");
                EpisodeStartTimes[mask] = PerfCounter();
                EpisodeReturns[mask] = 0;
                EpisodeLengths[mask] = 0;
                PrevDones[mask] = false;
                return;
            }

            Replace(EpisodeStartTimes, null, PerfCounter());
            Replace(EpisodeReturns, null, 0.0);
            Replace(EpisodeLengths, null, 0);
            PrevDones[":"] = false;
        }

        /// <summary>Overwrites an array's values in place, from another array or with one value.</summary>
        /// <param name="target">The array to overwrite.</param>
        /// <param name="source">The new values, or <see langword="null"/> to fill with <paramref name="fill"/>.</param>
        /// <param name="fill">The value when <paramref name="source"/> is <see langword="null"/>.</param>
        private static void Replace(NDArray target, NDArray? source, double fill = 0) {
            if (source is not null)
                np.copyto(target, source, casting: "unsafe");
            else
                target[":"] = fill;
        }

        /// <summary>Python's <c>time.perf_counter()</c>: a monotonic clock in seconds, meaningful only as a difference.</summary>
        /// <returns>The time.</returns>
        private static double PerfCounter() => System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency;
    }
}
