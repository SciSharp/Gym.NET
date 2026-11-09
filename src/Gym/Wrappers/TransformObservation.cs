#nullable enable
using Gym.Envs;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     Flattens every observation into a 1-D array (<c>spaces.utils.flatten</c>), with the flattened space
    ///     (<c>flatten_space</c>) as the observation space, as Gymnasium's <c>FlattenObservation</c> does: a <c>Dict</c> or
    ///     <c>Tuple</c> of arrays becomes one array, a <c>Discrete</c> a one-hot vector.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Typed for array results: a space whose flat form is not an array (<c>Graph</c>, <c>Sequence</c>, whose
    ///     <c>flatten_space</c> keeps them composite) is refused at construction with a <see cref="NumSharp.TypeError"/> rather
    ///     than returning a non-array observation.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L225">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class FlattenObservation<TObs, TAct> : ObservationWrapper<NDArray, TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The env's observation space at construction, which observations are flattened by (Python's captured <c>env.observation_space</c>).</summary>
        private readonly Space _envObservationSpace;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <exception cref="NumSharp.TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>, or its space's flat form is not an array.</exception>
        public FlattenObservation(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info());
            _envObservationSpace = env.ObservationSpace;
            Space flattened = SpaceUtils.FlattenSpace(_envObservationSpace);
            if (flattened is not Space<NDArray> arraySpace) {
                flattened.Dispose();
                throw new TypeError($"FlattenObservation returns arrays, but {_envObservationSpace} flattens to {flattened.GetType().Name}");
            }

            ObservationSpace = arraySpace;
        }

        /// <summary>The recorded constructor arguments, <c>{}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>Flattens an observation (<c>spaces.utils.flatten(env.observation_space, obs)</c>).</summary>
        /// <param name="observation">The env's observation; left to the env's caller.</param>
        /// <returns>A new 1-D array.</returns>
        public override NDArray Observation(TObs observation) => (NDArray)SpaceUtils.Flatten(_envObservationSpace, observation)!;
    }
}
