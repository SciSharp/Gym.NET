#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     With probability <c>repeat_action_probability</c>, replaces the agent's action by the previous one for a run of
    ///     <c>repeat_action_duration</c> steps, as Gymnasium's <c>StickyAction</c> does (Machado et al., 2018, §5.2): the
    ///     stochasticity that keeps an agent from memorising a deterministic env's action sequence.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, unchanged.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The draws come from the env's generator (<c>self.np_random</c>), so they shift the env's own stream: a seeded
    ///     reset makes them reproducible, and an env that draws in its steps sees other numbers than without the wrapper.
    ///     A run starts only when there is a previous action (never at an episode's first step); its length is drawn from
    ///     <c>[low, high]</c> when it starts.
    ///     </para>
    ///     <para>
    ///     Python keeps the previous action object itself. Here the caller owns its actions and may dispose one after the
    ///     step, so the wrapper keeps a deep copy of its own (<see cref="LastAction"/>) and hands the env a fresh copy of it
    ///     at every repeated step, which the env may keep or release; the agent's own actions pass through as they are. A
    ///     caller that mutates an action array after the step therefore sees its old values repeated, where Python would
    ///     repeat the mutated array.
    ///     </para>
    ///     <para>
    ///     <c>repeat_action_duration</c> is a Python int (any .NET integer) or a tuple of two (<see langword="object"/>?[] or a
    ///     C# tuple); a list (any other <see cref="System.Collections.IList"/>) is refused as Gymnasium refuses it. Bounds
    ///     that are floats are accepted as Gymnasium accepts them, and the draw truncates them as NumPy's <c>integers</c> does.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_action.py#L16">gymnasium/wrappers/stateful_action.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class StickyAction<TObs, TAct> : ActionWrapper<TObs, TAct, TAct>, IRecordConstructorArgs {
        /// <summary>Whether <see cref="LastAction"/> holds an action of this episode (Python tests <c>last_action is not None</c>; a value-type action needs the flag).</summary>
        private bool _hasLastAction;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="repeatActionProbability">The probability that a run of repeats starts at a step; in [0, 1).</param>
        /// <param name="repeatActionDuration">
        ///     The length of a run: an integer (<see langword="null"/> is Gymnasium's default, 1), or a tuple <c>(low, high)</c>
        ///     to draw each run's length from.
        /// </param>
        /// <exception cref="InvalidProbability">Thrown for a probability outside [0, 1) or NaN: <c>`repeat_action_probability` should be in the interval [0,1). Received 1.0</c>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown for a duration that is neither an integer nor a tuple, a tuple whose length is not 2, or a bound below 1,
        ///     with Gymnasium's messages.
        /// </exception>
        /// <exception cref="InvalidBound">Thrown when the low bound exceeds the high one: <c>`repeat_action_duration` is not a valid bound. Received (4, 2)</c>.</exception>
        /// <exception cref="TypeError">Thrown when a bound is not a number (Python's comparison refuses it), or <paramref name="env"/> is <see langword="null"/>.</exception>
        public StickyAction(Env<TObs, TAct> env, double repeatActionProbability, object? repeatActionDuration = null)
            : base(CheckArguments(env, repeatActionProbability, repeatActionDuration ?? 1L, out object?[] range)) {
            repeatActionDuration ??= 1L;
            SavedKwargs = RecordConstructorArgs.Record(new Info {
                ["repeat_action_probability"] = repeatActionProbability, ["repeat_action_duration"] = repeatActionDuration,
            });
            RepeatActionProbability = repeatActionProbability;
            RepeatActionDurationRange = range;
        }

        /// <summary>The recorded constructor arguments, the duration as the caller gave it (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The probability that a run of repeats starts (<c>repeat_action_probability</c>).</summary>
        public double RepeatActionProbability { get; set; }

        /// <summary>The bounds a run's length is drawn from, a tuple of two (<c>repeat_action_duration_range</c>).</summary>
        public object?[] RepeatActionDurationRange { get; }

        /// <summary>
        ///     A copy of the action the env got at the last step (<c>last_action</c>), owned by the wrapper (do not dispose it);
        ///     the default before the episode's first step.
        /// </summary>
        public TAct? LastAction { get; private set; }

        /// <summary>Whether a run of repeats is in progress (<c>is_sticky_actions</c>).</summary>
        public bool IsStickyActions { get; private set; }

        /// <summary>The length of the current run, drawn when it starts; 0 outside a run (<c>num_repeats</c>).</summary>
        public long NumRepeats { get; private set; }

        /// <summary>The repeats the current run has taken (<c>repeats_taken</c>).</summary>
        public long RepeatsTaken { get; private set; }

        /// <summary>Forgets the last action and any run, then resets the env.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's reset.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ReleaseLastAction();
            IsStickyActions = false;
            NumRepeats = 0;
            RepeatsTaken = 0;
            return base.Reset(seed, options);
        }

        /// <summary>The action the env gets: the agent's, or a copy of the previous one while a run of repeats lasts.</summary>
        /// <param name="action">The agent's action; it stays the caller's.</param>
        /// <returns><paramref name="action"/> itself, or a new copy of the previous action that the env may keep.</returns>
        public override TAct Action(TAct action) {
            bool repeated = false;
            // Either the agent is already stuck in a run, or a new run is triggered; Python's `or` short-circuits, so the
            // uniform draw happens only outside a run and only when there is a previous action.
            if (IsStickyActions || (_hasLastAction && LastAction is not null && DrawUniform() < RepeatActionProbability)) {
                // A new run draws its length when it starts.
                if (NumRepeats == 0)
                    NumRepeats = DrawDuration();
                // The env gets its own copy: the wrapper's must survive whatever the env, or its caller, does with it.
                action = (TAct)PyCopy.DeepCopy(LastAction)!;
                repeated = true;
                IsStickyActions = true;
                RepeatsTaken += 1;
            }

            // The run is over: the next step is the agent's again.
            if (IsStickyActions && NumRepeats == RepeatsTaken) {
                IsStickyActions = false;
                NumRepeats = 0;
                RepeatsTaken = 0;
            }

            // self.last_action = action: a repeated step keeps the copy it already holds (the same values); the agent's
            // action is copied, since the caller may dispose it once the step returns.
            if (!repeated) {
                ReleaseLastAction();
                // The copy outlives this step and the caller's NumSharp scope around it (DR-96).
                LastAction = ScopeSafety.Keep((TAct?)PyCopy.DeepCopy(action));
            }

            _hasLastAction = true;
            return action;
        }

        /// <summary>Releases the wrapper's copy of the last action, then disposes the wrapper as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing)
                ReleaseLastAction();
            base.Dispose(disposing);
        }

        /// <summary>Forgets the last action (<c>last_action = None</c>), releasing the wrapper's copy, which nothing else holds.</summary>
        private void ReleaseLastAction() {
            DisposeCopy(LastAction);
            LastAction = default;
            _hasLastAction = false;
        }

        /// <summary>Releases the arrays of a copy the wrapper made: an array, or the arrays inside a tuple or dict.</summary>
        /// <param name="copy">The copy, or <see langword="null"/>.</param>
        private static void DisposeCopy(object? copy) {
            switch (copy) {
                case IDisposable disposable:
                    disposable.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        DisposeCopy(item);
                    break;
                case IReadOnlyDictionary<string, object?> dict:
                    foreach (object? value in dict.Values)
                        DisposeCopy(value);
                    break;
            }
        }

        /// <summary><c>self.np_random.uniform()</c>: one float64 draw from the env's generator.</summary>
        /// <returns>The draw, in [0, 1).</returns>
        private double DrawUniform() {
            using NDArray drawn = NpRandom.uniform();
            return (double)drawn;
        }

        /// <summary><c>self.np_random.integers(low, high + 1)</c>: a run's length, the bounds read as NumPy's <c>integers</c> reads them (a float truncated).</summary>
        /// <returns>The length, in [low, high].</returns>
        private long DrawDuration() {
            BigInteger low = IntegerBound(RepeatActionDurationRange[0]);
            // high + 1 is computed before the truncation, as Python adds 1 to the bound it was given.
            BigInteger high = RepeatActionDurationRange[1] is double or float
                ? PyBuiltins.Int(PyBuiltins.Float(RepeatActionDurationRange[1]) + 1.0)
                : IntegerBound(RepeatActionDurationRange[1]) + 1;
            using NDArray drawn = NpRandom.integers(low, high);
            return (long)drawn;
        }

        /// <summary>A bound as NumPy's <c>integers</c> takes it: an integer exactly, a bool as 0 or 1, a float truncated.</summary>
        /// <param name="bound">The bound.</param>
        /// <returns>The integer.</returns>
        private static BigInteger IntegerBound(object? bound) => bound switch {
            bool flag => flag ? BigInteger.One : BigInteger.Zero,
            BigInteger big => big,
            double or float => PyBuiltins.Int(PyBuiltins.Float(bound)),
            _ => new BigInteger(Convert.ToInt64(bound)),
        };

        /// <summary>
        ///     Gymnasium's checks of the probability and the duration, in its order, made before the wrapper is built (Python
        ///     checks them before <c>__init__</c> records anything).
        /// </summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="probability">The probability.</param>
        /// <param name="duration">The duration as given.</param>
        /// <param name="range">The bounds, a tuple of two.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="InvalidProbability">Thrown for a probability outside [0, 1).</exception>
        /// <exception cref="ValueError">Thrown for a duration of the wrong kind or length, or a bound below 1.</exception>
        /// <exception cref="InvalidBound">Thrown for reversed bounds.</exception>
        /// <exception cref="TypeError">Thrown for bounds that are not numbers.</exception>
        private static Env<TObs, TAct> CheckArguments(Env<TObs, TAct> env, double probability, object duration, out object?[] range) {
            // `not 0 <= p < 1` is also true for NaN, which fails both comparisons.
            if (!(0 <= probability && probability < 1))
                throw new InvalidProbability($"`repeat_action_probability` should be in the interval [0,1). Received {PyBuiltins.Str(probability)}");

            // The pair of bounds that the number of repeats is drawn from; Python's bool is an int, so it pairs too.
            if (duration is bool || ObservationWrapperFunctions.IsPythonInt(duration) || duration is BigInteger)
                range = new[] { duration, duration };
            else if (duration is object?[] items && PyBuiltins.IsTupleArray(items))
                range = items;
            else if (duration is ITuple tuple)
                range = ObservationWrapperFunctions.AsTuple(tuple);
            else
                throw new ValueError($"`repeat_action_duration` should be either an integer or a tuple. Received {PyBuiltins.Str(duration)}");

            if (range.Length != 2)
                throw new ValueError($"`repeat_action_duration` should be a tuple or a list of two integers. Received {PyBuiltins.Str(range)}");
            if (Compare(range[0], range[1]) > 0)
                throw new InvalidBound($"`repeat_action_duration` is not a valid bound. Received {PyBuiltins.Str(range)}");
            if (Compare(range[0], 1L) < 0 || Compare(range[1], 1L) < 0)
                throw new ValueError($"`repeat_action_duration` should be larger or equal than 1. Received {PyBuiltins.Str(range)}");
            return env;
        }

        /// <summary>Python's ordering of two numbers: exact for integers (bools included), as float64 when either is a float.</summary>
        /// <param name="left">The left number.</param>
        /// <param name="right">The right number.</param>
        /// <returns>Negative, zero or positive, as <paramref name="left"/> is less than, equal to or greater than <paramref name="right"/>.</returns>
        /// <exception cref="TypeError">Thrown when either is not a number: <c>'&gt;' not supported between instances of 'str' and 'int'</c>.</exception>
        private static int Compare(object? left, object? right) {
            static bool IsInteger(object? value) => value is bool or BigInteger || ObservationWrapperFunctions.IsPythonInt(value);
            static bool IsReal(object? value) => value is double or float;
            if (IsInteger(left) && IsInteger(right))
                return IntegerBound(left).CompareTo(IntegerBound(right));
            if ((IsInteger(left) || IsReal(left)) && (IsInteger(right) || IsReal(right)))
                return PyBuiltins.Float(left).CompareTo(PyBuiltins.Float(right));
            throw new TypeError($"'>' not supported between instances of '{PyBuiltins.TypeName(left)}' and '{PyBuiltins.TypeName(right)}'");
        }
    }

    /// <summary>
    ///     Steps the env <c>num_repeats</c> times with each action, sums the rewards, and returns the last step's
    ///     observation, flags and info, as Gymnasium's <c>RepeatAction</c> does: a lower decision frequency, for envs whose
    ///     agent need not act at every step. An episode that ends stops the repetition early.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Unlike <see cref="StickyAction{TObs, TAct}"/>, which replaces the agent's action by the previous one at random,
    ///     this repeats the current action deterministically. The rewards are summed as Python floats; the intermediate
    ///     observations are the env's and left to their owner. Gymnasium refuses a non-integer <c>num_repeats</c> with
    ///     <c>TypeError</c>; here it is a <see cref="long"/>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/stateful_action.py#L145">gymnasium/wrappers/stateful_action.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class RepeatAction<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="numRepeats">The number of steps each action is taken; at least 1.</param>
        /// <exception cref="ValueError">Thrown for a count below 1: <c>The num_repeats value needs to be equal or greater than one, actual value: 0</c>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public RepeatAction(Env<TObs, TAct> env, long numRepeats) : base(CheckRepeats(env, numRepeats)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["num_repeats"] = numRepeats });
            NumRepeats = numRepeats;
        }

        /// <summary>The recorded constructor arguments, <c>{'num_repeats': 4}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The number of steps each action is taken (<c>num_repeats</c>).</summary>
        public long NumRepeats { get; }

        /// <summary>Steps the env <see cref="NumRepeats"/> times with the action, stopping when the episode ends.</summary>
        /// <param name="action">The action, repeated.</param>
        /// <returns>The last step's observation, flags and info, and the summed reward.</returns>
        public override StepResult<TObs> Step(TAct action) {
            double totalReward = 0.0;
            bool terminated = false, truncated = false;
            Info info = new();
            TObs obs = default!;
            for (long i = 0; i < NumRepeats; i++) {
                (obs, double reward, terminated, truncated, info) = Env.Step(action);
                totalReward += reward;
                if (terminated || truncated)
                    break;
            }

            return new StepResult<TObs>(obs, totalReward, terminated, truncated, info);
        }

        /// <summary>Gymnasium's check of <paramref name="numRepeats"/>, made before the wrapper is built, as Python checks it before <c>__init__</c> records anything.</summary>
        /// <param name="env">The env, passed through.</param>
        /// <param name="numRepeats">The count.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="ValueError">Thrown for a count below 1.</exception>
        private static Env<TObs, TAct> CheckRepeats(Env<TObs, TAct> env, long numRepeats) {
            if (numRepeats < 1)
                throw new ValueError($"The num_repeats value needs to be equal or greater than one, actual value: {numRepeats}");
            return env;
        }
    }
}
