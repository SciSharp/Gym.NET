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

    /// <summary>
    ///     Delays every observation by <c>delay</c> steps, as Gymnasium's <c>DelayObservation</c> does: until the delay has
    ///     passed, the wrapper returns the space's zero (<see cref="WrapperFunctions.CreateZeroArray"/>), then the observation
    ///     the env returned <c>delay</c> steps earlier. Use it to test an agent's tolerance of stale observations.
    /// </summary>
    /// <typeparam name="TObs">The observation type, unchanged.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The reset's observation is delayed too: with a delay of 2, the reset and the first step return zeros. A delayed
    ///     observation is the env's own object, handed on when its turn comes; each zero is a new element the caller owns.
    ///     The observations still queued at a reset are dropped, not released: the env's info may hold them, so they are
    ///     left to NumSharp's finalizer, as elsewhere in Gym.NET.
    ///     </para>
    ///     <para>
    ///     Gymnasium refuses a non-integer delay with <c>TypeError</c>; here the parameter is a <see cref="long"/>, so the
    ///     compiler refuses it instead.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py#L35">gymnasium/wrappers/stateful_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class DelayObservation<TObs, TAct> : ObservationWrapper<TObs, TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="delay">The number of steps observations are delayed; 0 delays nothing.</param>
        /// <exception cref="ValueError">Thrown for a negative delay: <c>The delay needs to be greater than zero, actual value: -1</c>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public DelayObservation(Env<TObs, TAct> env, long delay) : base(CheckDelay(env, delay)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["delay"] = delay });
            Delay = delay;
        }

        /// <summary>The recorded constructor arguments, <c>{'delay': 3}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The delay in steps (<c>delay</c>).</summary>
        public long Delay { get; }

        /// <summary>The observations not returned yet, oldest first (<c>observation_queue</c>); the reset clears it.</summary>
        public Queue<TObs> ObservationQueue { get; } = new();

        /// <summary>Clears the queued observations, then resets the env.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The space's zero (the env's first observation when the delay is 0) and the env's info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ObservationQueue.Clear();
            return base.Reset(seed, options);
        }

        /// <summary>Queues the env's observation and returns the one <see cref="Delay"/> steps old, or the space's zero before then.</summary>
        /// <param name="observation">The env's observation; the wrapper holds it until its turn.</param>
        /// <returns>The delayed observation (the env's own object), or a new zero element.</returns>
        public override TObs Observation(TObs observation) {
            // Queued observations outlive this step and the caller's NumSharp scope around it (DR-96); the one leaving the queue
            // goes back to the caller, whose scope (if one is open) adopts it as it adopts what it builds.
            ObservationQueue.Enqueue(ScopeSafety.Keep(observation));
            if (ObservationQueue.Count > Delay)
                return ScopeSafety.HandOver(ObservationQueue.Dequeue());
            return (TObs)WrapperFunctions.CreateZeroArray(ObservationSpace)!;
        }

        /// <summary>Gymnasium's check of <paramref name="delay"/>, made before the wrapper is built, as Python checks it before <c>__init__</c> runs.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="delay">The delay.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="ValueError">Thrown for a negative delay.</exception>
        private static Env<TObs, TAct> CheckDelay(Env<TObs, TAct> env, long delay) {
            if (!(0 <= delay))
                throw new ValueError($"The delay needs to be greater than zero, actual value: {delay}");
            return env;
        }
    }

    /// <summary>
    ///     Stacks the last <c>stack_size</c> observations along a new first axis, as Gymnasium's <c>FrameStackObservation</c>
    ///     does: the space is <c>batch_space(env.observation_space, stack_size)</c>, and after a reset the stack is padded
    ///     with the reset's observation (<c>"reset"</c>, the default), the space's zero (<c>"zero"</c>), or a given element.
    ///     Use it to give a memoryless agent motion (velocities from frames).
    /// </summary>
    /// <typeparam name="TObs">
    ///     The env's observation type, which the stacks share: an array space stacks into an array, a <c>Dict</c> into a
    ///     dict, a <c>Tuple</c> into a tuple. A space whose stack is of another type (a <c>Text</c> space stacks into a
    ///     tuple of strings) needs the untyped form, <see langword="object"/>; the constructor refuses the typed one.
    /// </typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Every stack returned is a deep copy of the wrapper's buffer, so the caller owns it and later steps never change
    ///     it. The wrapper keeps the env's last <c>stack_size</c> observations (Python's deque) and releases only what it
    ///     created: the buffer, and the zero padding (when the reset's padding replaces it, or with the wrapper). The env's
    ///     observations that fall out of the stack are left to NumSharp's finalizer, since the env's info may hold them; a
    ///     custom padding stays the caller's.
    ///     </para>
    ///     <para>
    ///     Gymnasium refuses a non-integer <c>stack_size</c> with <c>TypeError</c>; here it is a <see cref="long"/>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py#L305">gymnasium/wrappers/stateful_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class FrameStackObservation<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The env's observation space, unviewed: what each stack is concatenated by.</summary>
        private readonly Space _envSpace;

        /// <summary>The buffer each stack is concatenated into (<c>stacked_obs</c>, from <c>create_empty_array</c>); owned.</summary>
        private readonly object? _stackedObs;

        /// <summary>The zero padding this wrapper created and still owns, or <see langword="null"/> once a reset's observation replaced it.</summary>
        private object? _ownedPadding;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="stackSize">The number of observations stacked; positive.</param>
        /// <param name="paddingType">
        ///     <c>"reset"</c> (<see langword="null"/>, the default: pad with each episode's first observation), <c>"zero"</c>
        ///     (pad with the space's zero), or an element of the env's observation space to pad with, which stays the
        ///     caller's.
        /// </param>
        /// <exception cref="ValueError">
        ///     Thrown for a stack size below 1 (<c>The stack_size needs to be greater than zero, actual value: 0</c>), or a
        ///     padding that is neither a mode nor an element of the space (<c>Unexpected `padding_type`, ...</c>).
        /// </exception>
        /// <exception cref="TypeError">Thrown when the stacks are not of <typeparamref name="TObs"/>, or <paramref name="env"/> is <see langword="null"/>.</exception>
        public FrameStackObservation(Env<TObs, TAct> env, long stackSize, object? paddingType = null) : base(env) {
            paddingType ??= "reset";
            // Python records the arguments before it checks them, so a refused wrapper has recorded them too.
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["stack_size"] = stackSize, ["padding_type"] = paddingType });
            if (!(0 < stackSize))
                throw new ValueError($"The stack_size needs to be greater than zero, actual value: {stackSize}");

            _envSpace = ObjectSpace.Unview(env.ObservationSpace)!;
            string mode;
            if (paddingType is "reset" or "zero") {
                _ownedPadding = PaddingValue = WrapperFunctions.CreateZeroArray(_envSpace);
                mode = (string)paddingType;
            } else if (_envSpace.Contains(paddingType)) {
                PaddingValue = paddingType;
                mode = "_custom";
            } else if (paddingType is string text) {
                // We are guessing that the user just entered the "reset" or "zero" wrong.
                throw new ValueError($"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {PyBuiltins.Repr(text)}");
            } else {
                throw new ValueError($"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {PyBuiltins.Repr(paddingType)} not an instance of env observation ({_envSpace})");
            }

            Space batched = Gym.Vector.VectorUtils.BatchSpace(_envSpace, checked((int)stackSize));
            try {
                ObservationSpace = ObservationWrapperFunctions.Typed<TObs>(batched);
            } catch {
                // The wrapper never got the batched space or the zero padding: release both before the refusal leaves.
                batched.Dispose();
                DisposeElement(_ownedPadding);
                throw;
            }

            StackSize = stackSize;
            PaddingType = mode;
            for (long i = 0; i < stackSize; i++)
                ObsQueue.AddLast(PaddingValue);
            _stackedObs = Gym.Vector.VectorUtils.CreateEmptyArray(_envSpace, checked((int)stackSize));
        }

        /// <summary>The recorded constructor arguments, <c>{'stack_size': 3, 'padding_type': 'reset'}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The number of observations stacked (<c>stack_size</c>).</summary>
        public long StackSize { get; }

        /// <summary>The padding mode: <c>reset</c>, <c>zero</c>, or <c>_custom</c> for a given element (<c>padding_type</c>).</summary>
        public string PaddingType { get; }

        /// <summary>The element the stack is padded with (<c>padding_value</c>): the space's zero, the given one, or the last reset's observation.</summary>
        public object? PaddingValue { get; private set; }

        /// <summary>The last <see cref="StackSize"/> observations, oldest first (<c>obs_queue</c>, a deque of fixed length).</summary>
        public LinkedList<object?> ObsQueue { get; } = new();

        /// <summary>Steps the env and returns the stack with its new observation last.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>A new stack (a deep copy), and the env's reward, flags and info.</returns>
        public override StepResult<TObs> Step(TAct action) {
            (TObs obs, double reward, bool terminated, bool truncated, Info info) = Env.Step(action);
            Append(obs);
            return new StepResult<TObs>(Stacked(), reward, terminated, truncated, info);
        }

        /// <summary>Resets the env and returns <c>stack_size - 1</c> paddings followed by the first observation.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>A new stack and the env's info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            (TObs obs, Info info) = Env.Reset(seed, options);
            if (PaddingType == "reset")
                PaddingValue = obs;
            for (long i = 0; i < StackSize - 1; i++)
                Append(PaddingValue);
            Append(obs);
            // The stack_size appends pushed every earlier slot out, so the first zero padding is now referenced by nothing
            // when the reset's observation replaced it.
            if (_ownedPadding is not null && !ReferenceEquals(_ownedPadding, PaddingValue)) {
                DisposeElement(_ownedPadding);
                _ownedPadding = null;
            }

            return new ResetResult<TObs>(Stacked(), info);
        }

        /// <summary>Releases the buffer and the zero padding this wrapper created, then disposes the wrapper as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                ObsQueue.Clear();
                DisposeElement(_stackedObs);
                DisposeElement(_ownedPadding);
                _ownedPadding = null;
            }

            base.Dispose(disposing);
        }

        /// <summary>Appends an observation to the fixed-length queue (<c>deque(maxlen=stack_size).append</c>): the oldest falls out.</summary>
        /// <param name="observation">The observation.</param>
        private void Append(object? observation) {
            // The stack keeps the env's observations (and a reset's padding, the same object) for StackSize steps, past the
            // caller's NumSharp scopes (DR-96); the caller gets a deep copy of the stack, never these.
            ObsQueue.AddLast(ScopeSafety.Keep(observation));
            if (ObsQueue.Count > StackSize)
                ObsQueue.RemoveFirst();
        }

        /// <summary><c>deepcopy(concatenate(env.observation_space, obs_queue, stacked_obs))</c>.</summary>
        /// <returns>A new stack the caller owns.</returns>
        private TObs Stacked() {
            object? concatenated = Gym.Vector.VectorUtils.Concatenate(_envSpace, ObsQueue, _stackedObs);
            return (TObs)PyCopy.DeepCopy(concatenated)!;
        }

        /// <summary>Releases the arrays of an element the wrapper created: an array, or the arrays inside a tuple or dict.</summary>
        /// <param name="element">The element, or <see langword="null"/>.</param>
        private static void DisposeElement(object? element) {
            switch (element) {
                case IDisposable disposable:
                    disposable.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        DisposeElement(item);
                    break;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? value in dict.Values)
                        DisposeElement(value);
                    break;
            }
        }
    }

    /// <summary>
    ///     Repeats each action for <c>skip</c> steps, sums the rewards, and returns the element-wise maximum of the last two
    ///     observations, as Gymnasium's <c>MaxAndSkipObservation</c> does: the frame skipping of Atari agents, where the
    ///     maximum removes the flicker of sprites drawn on alternate frames.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding arrays or numbers).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     An episode that ends early stops the repetition, and the maximum then reads what the earlier steps left in the
    ///     buffer, as in Gymnasium: after a one-step episode the second slot still holds the previous episode's frame. A
    ///     reset is the env's own, not a maximum. The rewards are summed as Python floats. Each step returns a new array of
    ///     the space's dtype; the env's frames are copied into the buffer and left to their owner.
    ///     </para>
    ///     <para>
    ///     Gymnasium refuses a non-integer <c>skip</c> with <c>TypeError</c>; here it is a <see cref="long"/>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_observation.py#L565">gymnasium/wrappers/stateful_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class MaxAndSkipObservation<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The last two observations of a step (<c>_obs_buffer</c>), <c>(2, *shape)</c> in the space's dtype; owned.</summary>
        private readonly NDArray _obsBuffer;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must have a shape.</param>
        /// <param name="skip">The number of steps each action is repeated; at least 2.</param>
        /// <exception cref="ValueError">
        ///     Thrown for a skip below 2 (<c>The skip value needs to be equal or greater than two, actual value: 0</c>), or a
        ///     space without a shape (<c>The observation space must have the shape attribute.</c>).
        /// </exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public MaxAndSkipObservation(Env<TObs, TAct> env, long skip = 4) : base(env) {
            // Python records the arguments before it checks them.
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["skip"] = skip });
            if (skip < 2)
                throw new ValueError($"The skip value needs to be equal or greater than two, actual value: {skip}");
            Space space = ObjectSpace.Unview(env.ObservationSpace)!;
            if (space.Shape is not Shape shape)
                throw new ValueError("The observation space must have the shape attribute.");
            Skip = skip;
            _obsBuffer = np.zeros(new Shape(new[] { 2L }.Concat(shape.dimensions).ToArray()), space.DType!);
        }

        /// <summary>The recorded constructor arguments, <c>{'skip': 4}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The number of steps each action is repeated (<c>_skip</c>).</summary>
        [PyName("_skip")]
        public long Skip { get; }

        /// <summary>Steps the env <see cref="Skip"/> times with the action (fewer when the episode ends) and returns the maximum of the last two observations.</summary>
        /// <param name="action">The action, repeated.</param>
        /// <returns>The maximum frame (a new array), the summed reward, and the last step's flags and info.</returns>
        public override StepResult<TObs> Step(TAct action) {
            double totalReward = 0.0;
            bool terminated = false, truncated = false;
            Info info = new();
            for (long i = 0; i < Skip; i++) {
                (TObs obs, double reward, bool stepTerminated, bool stepTruncated, Info stepInfo) = Env.Step(action);
                (terminated, truncated, info) = (stepTerminated, stepTruncated, stepInfo);
                if (i == Skip - 2)
                    Store(0, obs);
                if (i == Skip - 1)
                    Store(1, obs);
                totalReward += reward;
                if (terminated || truncated)
                    break;
            }

            NDArray maxFrame = np.max(_obsBuffer, axis: 0);
            return new StepResult<TObs>((TObs)(object)maxFrame, totalReward, terminated, truncated, info);
        }

        /// <summary>Releases the buffer, then disposes the wrapper as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing)
                _obsBuffer.Dispose();
            base.Dispose(disposing);
        }

        /// <summary><c>self._obs_buffer[slot] = obs</c>: the observation cast into the buffer's dtype, as NumPy's setitem casts it.</summary>
        /// <param name="slot">0 or 1.</param>
        /// <param name="observation">The env's observation, an array or a number.</param>
        /// <exception cref="TypeError">Thrown for an observation that is neither.</exception>
        private void Store(int slot, TObs observation) {
            NDArray frame = ObservationWrapperFunctions.AsArray(observation);
            try {
                _obsBuffer[slot] = frame;
            } finally {
                // Only the 0-d array made for a number is this method's; the env's array stays its own.
                if (!ReferenceEquals(frame, observation))
                    frame.Dispose();
            }
        }
    }
}
