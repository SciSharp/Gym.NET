#nullable enable
using System;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>Applies a function to every reward <c>Step</c> returns, as Gymnasium's <c>TransformReward</c> does.</summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Rewards are <see cref="double"/>s throughout Gym.NET, so the function maps a <see cref="double"/> to a
    ///     <see cref="double"/>; Gymnasium's takes and returns anything that supports <c>float()</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_reward.py#L21">gymnasium/wrappers/transform_reward.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class TransformReward<TObs, TAct> : RewardWrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="func">The transformation; <see langword="null"/> fails at the first step, as Python's <c>None</c> does when called.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public TransformReward(Env<TObs, TAct> env, Func<double, double> func) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["func"] = func });
            Func = func;
        }

        /// <summary>Wraps <paramref name="env"/> for a subclass that assigns <see cref="Func"/> and <see cref="SavedKwargs"/> in its own constructor.</summary>
        /// <param name="env">The env.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected TransformReward(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = new Info();
            Func = null!;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>): a subclass's own, as Python records the first call's.</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; protected set; }

        /// <summary>The transformation (<c>func</c>).</summary>
        [PyName("func")]
        public Func<double, double> Func { get; protected set; }

        /// <summary>Applies <see cref="Func"/> (<c>reward(reward)</c>).</summary>
        /// <param name="reward">The env's reward.</param>
        /// <returns>The transformed reward.</returns>
        /// <exception cref="TypeError">Thrown when the function is <see langword="null"/>: <c>'NoneType' object is not callable</c>.</exception>
        public override double Reward(double reward) =>
            Func is null ? throw new TypeError("'NoneType' object is not callable") : Func(reward);
    }

    /// <summary>
    ///     Clips every reward into <c>[min_reward, max_reward]</c>, as Gymnasium's <c>ClipReward</c> does; either bound may be
    ///     left out.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The clip is <c>np.clip(reward, a_min=min_reward, a_max=max_reward)</c>: a NaN reward stays NaN. Bounds may be
    ///     numbers or arrays, as in Gymnasium, and are checked the same way; a reward is a <see cref="double"/> here, so an
    ///     array bound must clip it to a single value (a 0-d or one-element array) — the vector wrapper is the one for
    ///     per-env bounds. Gymnasium's <c>np.clip</c> returns an <c>np.float64</c> (or an <c>np.int64</c> for an int reward
    ///     and int bounds); the value is the same.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_reward.py#L64">gymnasium/wrappers/transform_reward.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class ClipReward<TObs, TAct> : TransformReward<TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="minReward">The lower bound, a number or an array; <see langword="null"/> for none.</param>
        /// <param name="maxReward">The upper bound, a number or an array; <see langword="null"/> for none.</param>
        /// <exception cref="InvalidBound">Thrown when both bounds are <see langword="null"/>, or a lower bound exceeds its upper bound, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when a bound is neither a number nor an array, or <paramref name="env"/> is <see langword="null"/>.</exception>
        public ClipReward(Env<TObs, TAct> env, object? minReward = null, object? maxReward = null) : base(env) {
            if (minReward is null && maxReward is null)
                throw new InvalidBound("Both `min_reward` and `max_reward` cannot be None");
            if (minReward is not null && maxReward is not null) {
                using NDArray low = ObservationWrapperFunctions.AsArray(minReward);
                using NDArray high = ObservationWrapperFunctions.AsArray(maxReward);
                using NDArray span = high - low;
                using NDArray reversed = span < 0;
                if (np.any(reversed))
                    throw new InvalidBound($"Min reward ({PyBuiltins.Str(minReward)}) must be smaller than max reward ({PyBuiltins.Str(maxReward)})");
            }

            SavedKwargs = RecordConstructorArgs.Record(new Info { ["min_reward"] = minReward, ["max_reward"] = maxReward });
            MinReward = minReward;
            MaxReward = maxReward;
            Func = reward => Clip(reward, minReward, maxReward);
        }

        /// <summary>The lower bound as given (<c>min_reward</c>), or <see langword="null"/>.</summary>
        public object? MinReward { get; }

        /// <summary>The upper bound as given (<c>max_reward</c>), or <see langword="null"/>.</summary>
        public object? MaxReward { get; }

        /// <summary><c>np.clip(reward, a_min, a_max)</c> with optional bounds.</summary>
        /// <param name="reward">The reward.</param>
        /// <param name="minimum">The lower bound, or <see langword="null"/>.</param>
        /// <param name="maximum">The upper bound, or <see langword="null"/>.</param>
        /// <returns>The clipped reward.</returns>
        /// <exception cref="TypeError">Thrown when array bounds clip the reward to more than one value.</exception>
        /// <remarks>
        ///     Always NumPy's clip through NumSharp, even for number bounds: its one-sided forms are <c>np.maximum</c> /
        ///     <c>np.minimum</c>, and the three disagree on a signed zero (<c>clip(-0.0, 0, 1)</c> keeps <c>-0.0</c>,
        ///     <c>clip(-0.0, 0, None)</c> gives <c>+0.0</c>), which a byte-exact reward must reproduce.
        /// </remarks>
        private static double Clip(double reward, object? minimum, object? maximum) {
            using NDArray rewardArray = NDArray.Scalar(reward);
            using NDArray? low = minimum is null ? null : ObservationWrapperFunctions.AsArray(minimum);
            using NDArray? high = maximum is null ? null : ObservationWrapperFunctions.AsArray(maximum);
            using NDArray clipped = np.clip(rewardArray, low, high);
            if (clipped.size != 1)
                throw new TypeError($"ClipReward's array bounds clip a single reward into {clipped.size} values; a reward is one double, use the vector wrapper for per-env bounds");
            return Convert.ToDouble(clipped.GetAtIndex(0));
        }
    }
}
