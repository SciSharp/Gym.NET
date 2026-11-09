#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Wrappers {
    /// <summary>
    ///     Adds the episode's elapsed steps to every observation, as Gymnasium's <c>TimeAwareObservation</c> does: a
    ///     <c>Dict</c> observation gets a <c>time</c> key, a <c>Tuple</c> one a last item, anything else becomes
    ///     <c>{"obs": obs, "time": time}</c>, and by default the result is flattened into one array.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The time is the step count since the last reset, as an int32 array <c>[t]</c>, or with <c>normalize_time</c> a
    ///     float32 <c>[t / max_episode_steps]</c>; the limit comes from the env's spec or, failing that, a <c>TimeLimit</c> in
    ///     the stack. The observation type depends on <c>flatten</c> at run time (an array when flattened, a dict or tuple
    ///     otherwise), so the wrapper's observations are <see langword="object"/>s.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py#L106">gymnasium/wrappers/stateful_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class TimeAwareObservation<TObs, TAct> : ObservationWrapper<object?, TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The observation space before flattening, which <see cref="Observation"/> flattens by; owned by this wrapper.</summary>
        private readonly Space _unflattenedSpace;

        /// <summary>
        ///     How the time joins an observation, decided by the env's space at construction (Python's <c>_append_data_func</c>):
        ///     into a <c>Dict</c> observation, onto a <c>Tuple</c> one, or beside any other in <c>{"obs": obs, "time": time}</c>.
        /// </summary>
        private readonly Func<TObs, NDArray, object> _appendData;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its spec or a <c>TimeLimit</c> in its stack must give the step limit.</param>
        /// <param name="flatten">Whether the observation and the time are flattened into one array (Gymnasium's default).</param>
        /// <param name="normalizeTime">Whether the time is the fraction of the limit elapsed (float32) instead of the step count (int32).</param>
        /// <param name="dictTimeKey">The time's key in a <c>Dict</c> observation.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when no step limit is known, or <paramref name="dictTimeKey"/> already is a key of the env's <c>Dict</c> space, with Gymnasium's messages.</exception>
        public TimeAwareObservation(Env<TObs, TAct> env, bool flatten = true, bool normalizeTime = false, string dictTimeKey = "time") : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["flatten"] = flatten, ["normalize_time"] = normalizeTime, ["dict_time_key"] = dictTimeKey });
            Flatten = flatten;
            NormalizeTime = normalizeTime;
            DictTimeKey = dictTimeKey;
            MaxTimesteps = StepLimit(env);

            // Find the normalized time space.
            Space timeSpace = normalizeTime ? new Box(0.0, 1.0) : new Box(0, MaxTimesteps, dtype: np.int32);

            // Find the observation space: through an untyped env's view, so its Dict or Tuple is recognized as one.
            Space envSpace = ObjectSpace.Unview(env.ObservationSpace)!;
            Space observationSpace;
            switch (envSpace) {
                case Dict dict:
                    _appendData = AppendToDict;
                    if (dict.Keys.Contains(dictTimeKey)) {
                        timeSpace.Dispose();
                        throw new ValueError($"The `dict_time_key` ({PyBuiltins.Repr(dictTimeKey)}) already exists in the observation space.");
                    }

                    // {dict_time_key: time_space, **env.observation_space.spaces}: the time first, then the env's keys.
                    var spaces = new Info { [dictTimeKey] = timeSpace };
                    foreach (KeyValuePair<string, Space> pair in dict)
                        spaces[pair.Key] = pair.Value.DeepCopy();
                    observationSpace = new Dict(spaces.ToDictionary(p => p.Key, p => (Space)p.Value!), sortKeys: dict.SortKeys);
                    break;
                case Tuple tuple:
                    _appendData = (obs, time) => ((object?[])(object)obs!).Append(time).ToArray();
                    observationSpace = new Tuple(tuple.Spaces.Select(s => s.DeepCopy()).Append(timeSpace));
                    break;
                default:
                    _appendData = (obs, time) => new Info { ["obs"] = obs, ["time"] = time };
                    // Dict(obs=..., time=...): keyword arguments, so Gymnasium sorts them.
                    observationSpace = new Dict(new Dictionary<string, Space> { ["obs"] = envSpace.DeepCopy(), ["time"] = timeSpace });
                    break;
            }

            // If to flatten the observation space.
            _unflattenedSpace = observationSpace;
            ObservationSpace = new ObjectSpace(flatten ? SpaceUtils.FlattenSpace(observationSpace) : observationSpace.DeepCopy());
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>Whether observations are flattened (<c>flatten</c>).</summary>
        public bool Flatten { get; }

        /// <summary>Whether the time is normalized by the limit (<c>normalize_time</c>).</summary>
        public bool NormalizeTime { get; }

        /// <summary>The time's key in a <c>Dict</c> observation.</summary>
        public string DictTimeKey { get; }

        /// <summary>The step limit the time is measured against (<c>max_timesteps</c>).</summary>
        public long MaxTimesteps { get; }

        /// <summary>The steps since the last reset (<c>timesteps</c>).</summary>
        public long Timesteps { get; set; }

        /// <summary>Adds the time to an observation (<c>observation(observation)</c>), then flattens when <see cref="Flatten"/> is set.</summary>
        /// <param name="observation">The env's observation.</param>
        /// <returns>A new array when flattening; otherwise a new dict or tuple holding the observation and the time array.</returns>
        public override object? Observation(TObs observation) {
            NDArray time = NormalizeTime
                ? np.array(new[] { (float)((double)Timesteps / MaxTimesteps) })
                : np.array(new[] { (int)Timesteps });
            object appended = _appendData(observation, time);
            if (!Flatten)
                return appended;
            try {
                return SpaceUtils.Flatten(_unflattenedSpace, appended);
            } finally {
                // The flattened array copied the time; the intermediate container and time array are this call's own.
                time.Dispose();
            }
        }

        /// <summary>Steps the env and counts the step before the observation is built.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The transition with the time added to its observation.</returns>
        public override StepResult<object?> Step(TAct action) {
            Timesteps += 1;
            return base.Step(action);
        }

        /// <summary>Resets the env and the step count.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The first observation with time 0.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            Timesteps = 0;
            return base.Reset(seed, options);
        }

        /// <summary>Disposes the unflattened space with the stack.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (disposing)
                _unflattenedSpace.Dispose();
        }

        /// <summary>
        ///     The step limit: the env's spec's <c>max_episode_steps</c>, else the first <c>TimeLimit</c> found walking down the
        ///     stack.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <returns>The limit.</returns>
        /// <exception cref="ValueError">Thrown when neither gives one.</exception>
        private static long StepLimit(Env<TObs, TAct> env) {
            // We don't need to keep if a TimeLimit wrapper exists as `spec` will do that work for us now.
            if (env.Spec?.MaxEpisodeSteps is long limit)
                return limit;
            // Else we need to loop through the environment stack to check if a `TimeLimit` wrapper exists.
            for (IEnv? wrapped = env; wrapped is IWrapper wrapper; wrapped = wrapper.Env) {
                Type type = wrapped.GetType();
                if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(TimeLimit<,>))
                    return (long)PyAttributes.GetAttr(wrapped, "_max_episode_steps")!;
            }

            throw new ValueError("The environment must be wrapped by a TimeLimit wrapper or the spec specify a `max_episode_steps`.");
        }

        /// <summary><c>{dict_time_key: time, **obs}</c>: the time first, then the observation's entries.</summary>
        /// <param name="observation">The env's dict observation.</param>
        /// <param name="time">The time array.</param>
        /// <returns>The new dict.</returns>
        /// <exception cref="TypeError">Thrown when the observation is not a dict.</exception>
        private Info AppendToDict(TObs observation, NDArray time) {
            var result = new Info { [DictTimeKey] = time };
            if (observation is not IEnumerable<KeyValuePair<string, object?>> entries)
                throw new TypeError($"'{PyBuiltins.TypeName(observation)}' object is not a mapping");
            foreach (KeyValuePair<string, object?> entry in entries)
                result[entry.Key] = entry.Value;
            return result;
        }
    }

    /// <summary>
    ///     Normalizes every observation to roughly zero mean and unit variance by running statistics of the observations seen
    ///     so far, as Gymnasium's <c>NormalizeObservation</c> does, byte for byte: float32 statistics, float32 results.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type (an array, or a number).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The statistics (<see cref="ObsRms"/>) start at mean 0 and variance 1 and are float32 like the new observation space
    ///     (an unbounded float32 <c>Box</c> of the env's shape); an observation of a wider dtype promotes them to float64, as
    ///     NumPy does. Every observation, the reset's included, updates them unless <see cref="UpdateRunningMean"/> is off.
    ///     Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py#L465">gymnasium/wrappers/stateful_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class NormalizeObservation<TObs, TAct> : ObservationWrapper<NDArray, TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env, whose observation space must have a shape.</param>
        /// <param name="epsilon">Added to the variance under the square root, strictly positive.</param>
        /// <exception cref="InvalidBound">Thrown for a non-positive <paramref name="epsilon"/>: <c>`epsilon` should be strictly positive. Received 0.0</c>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when the observation space has no shape.</exception>
        public NormalizeObservation(Env<TObs, TAct> env, double epsilon = 1e-8) : base(CheckEpsilon(env, epsilon)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["epsilon"] = epsilon });
            Shape shape = env.ObservationSpace.Shape ?? throw new ValueError("NormalizeObservation wrapper requires the observation space to have a shape.");
            ObservationSpace = new Box(double.NegativeInfinity, double.PositiveInfinity, shape, np.float32);
            ObsRms = new RunningMeanStd(shape: shape, dtype: np.float32);
            Epsilon = epsilon;
        }

        /// <summary>The recorded constructor arguments, <c>{'epsilon': 1e-08}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The running statistics of the observations (<c>obs_rms</c>); owned by this wrapper.</summary>
        public RunningMeanStd ObsRms { get; }

        /// <summary>The variance's offset under the square root (<c>epsilon</c>).</summary>
        public double Epsilon { get; set; }

        /// <summary>Whether observations update the statistics (<c>update_running_mean</c>); turn off to evaluate with frozen statistics.</summary>
        public bool UpdateRunningMean { get; set; } = true;

        /// <summary>
        ///     Updates the statistics with the observation, then returns <c>np.float32((obs - mean) / np.sqrt(var + epsilon))</c>,
        ///     each operation in the dtype NumPy computes it in.
        /// </summary>
        /// <param name="observation">The env's observation; left to the env's caller.</param>
        /// <returns>A new float32 array of the observation's shape.</returns>
        public override NDArray Observation(TObs observation) {
            using NDArray obs = AsArray(observation);
            if (UpdateRunningMean) {
                // np.array([observation]): a batch of one, with a new first axis.
                using NDArray batch = np.expand_dims(obs, 0);
                ObsRms.Update(batch);
            }

            NDArray mean = ObsRms.Mean, var = ObsRms.Var;
            bool differenceSingle = np.promote_types(obs.dtype, mean.dtype).typecode == NPTypeCode.Single;
            bool rootSingle = var.typecode == NPTypeCode.Single;
            bool quotientSingle = differenceSingle && rootSingle;
            double[] x = Values(obs), m = Values(mean), v = Values(var);
            var result = new float[x.Length];
            for (int i = 0; i < x.Length; i++) {
                // Each operation rounded to its dtype: a float32 operation computed in double and rounded once is exact.
                double difference = Round(x[i] - m[i], differenceSingle);
                double root = Round(Math.Sqrt(Round(v[i] + (rootSingle ? (float)Epsilon : Epsilon), rootSingle)), rootSingle);
                result[i] = (float)Round(difference / root, quotientSingle);
            }

            using NDArray flat = np.array(result);
            return flat.reshape(obs.Shape).copy();
        }

        /// <summary>Disposes the statistics with the stack.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (disposing)
                ObsRms.Dispose();
        }

        /// <summary>Gymnasium's check of <paramref name="epsilon"/>, made before the wrapper is built.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="epsilon">The offset.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="InvalidBound">Thrown for a non-positive <paramref name="epsilon"/>.</exception>
        private static Env<TObs, TAct> CheckEpsilon(Env<TObs, TAct> env, double epsilon) {
            // `epsilon` is added under the square root to keep the division away from zero, so a non-positive value defeats
            // its purpose and, once it exceeds the variance, silently turns every normalized observation into NaN.
            if (epsilon <= 0)
                throw new InvalidBound($"`epsilon` should be strictly positive. Received {PyBuiltins.Str(epsilon)}");
            return env;
        }

        /// <summary>An observation as an array of its own dtype: an array as a copy, a number as a 0-d array (a Python int as int64, a float as float64).</summary>
        /// <param name="observation">The observation.</param>
        /// <returns>A new array the caller owns.</returns>
        /// <exception cref="TypeError">Thrown for an observation that is no number or array.</exception>
        private static NDArray AsArray(TObs observation) => observation switch {
            NDArray array => array.copy(),
            bool flag => NDArray.Scalar(flag),
            double real => NDArray.Scalar(real),
            float single => NDArray.Scalar(single),
            long or int or short or sbyte or byte or ushort or uint => NDArray.Scalar(Convert.ToInt64(observation)),
            _ => throw new TypeError($"NormalizeObservation normalizes arrays and numbers, not {PyBuiltins.TypeRepr(observation)}"),
        };

        /// <summary>An array's values widened to double, in C order (exact for float32, float64 and integers up to 2^53).</summary>
        /// <param name="array">The array.</param>
        /// <returns>The values.</returns>
        private static double[] Values(NDArray array) {
            using NDArray widened = array.astype(np.float64);
            return widened.ToArray<double>();
        }

        /// <summary>A double rounded to float32 when the operation's dtype is float32.</summary>
        /// <param name="value">The value.</param>
        /// <param name="single">Whether the dtype is float32.</param>
        /// <returns>The value in that precision.</returns>
        private static double Round(double value, bool single) => single ? (float)value : value;
    }
}
