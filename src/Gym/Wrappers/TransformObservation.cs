#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Gym.Envs;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Wrappers {
    /// <summary>
    ///     Applies a function to every observation <c>Reset</c> and <c>Step</c> return, as Gymnasium's
    ///     <c>TransformObservation</c> does. Give it the transformed observations' space when they leave the env's.
    /// </summary>
    /// <typeparam name="TWObs">The transformed observation type.</typeparam>
    /// <typeparam name="TObs">The env's observation type.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Without an observation space the wrapper forwards the env's, which only works when <typeparamref name="TWObs"/>
    ///     is the env's observation type: reading it otherwise raises <see cref="TypeError"/>. The base of Gymnasium's
    ///     stateless observation wrappers, which compute their space and function in their constructors; they set
    ///     <see cref="Func"/> and <see cref="SavedKwargs"/> themselves, as Python's subclasses record their own arguments
    ///     before this class's constructor runs.
    ///     </para>
    ///     <para>
    ///     The function's result is returned as it is: whether it shares memory with the env's observation (a reshape view,
    ///     a dict holding the same arrays) is the function's business, as in Python.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L42">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class TransformObservation<TWObs, TObs, TAct> : ObservationWrapper<TWObs, TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env.</param>
        /// <param name="func">The transformation; <see langword="null"/> fails at the first observation, as Python's <c>None</c> does when called.</param>
        /// <param name="observationSpace">The transformed observations' space, owned by the wrapper from now on; <see langword="null"/> forwards the env's.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public TransformObservation(Env<TObs, TAct> env, Func<TObs, TWObs> func, Space<TWObs>? observationSpace) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["func"] = func, ["observation_space"] = observationSpace });
            if (observationSpace is not null)
                ObservationSpace = observationSpace;
            Func = func;
        }

        /// <summary>
        ///     Wraps <paramref name="env"/> for a subclass that computes its function and space in its own constructor and
        ///     assigns <see cref="Func"/>, <see cref="Env{TObs, TAct}.ObservationSpace"/> and <see cref="SavedKwargs"/> there.
        /// </summary>
        /// <param name="env">The env.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected TransformObservation(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = new Info();
            Func = null!;
        }

        /// <summary>The recorded constructor arguments (Python's <c>_saved_kwargs</c>): a subclass's own, as Python records the first call's.</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; protected set; }

        /// <summary>The transformation (<c>func</c>).</summary>
        [PyName("func")]
        public Func<TObs, TWObs> Func { get; protected set; }

        /// <summary>Applies <see cref="Func"/> (<c>observation(observation)</c>).</summary>
        /// <param name="observation">The env's observation.</param>
        /// <returns>The function's result, as it returns it.</returns>
        /// <exception cref="TypeError">Thrown when the function is <see langword="null"/>: <c>'NoneType' object is not callable</c>.</exception>
        public override TWObs Observation(TObs observation) =>
            Func is null ? throw new TypeError("'NoneType' object is not callable") : Func(observation);
    }

    /// <summary>
    ///     Keeps only some keys of a <c>Dict</c> observation, or some indices of a <c>Tuple</c> one, as Gymnasium's
    ///     <c>FilterObservation</c> does; the space is filtered the same way.
    /// </summary>
    /// <typeparam name="TObs">The observation type: <see cref="Info"/> for a <c>Dict</c> env, <see langword="object"/>?[] for a <c>Tuple</c> env, or <see langword="object"/>.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The filtered observation is a new dict or tuple holding the env's values themselves, not copies. The filtered
    ///     space holds copies of the env's subspaces (Python shares the subspace objects), so the wrapper can own and
    ///     dispose it; sampling it does not advance the env's subspaces' generators.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L100">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class FilterObservation<TObs, TAct> : TransformObservation<TObs, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must be a <c>Dict</c> or a <c>Tuple</c>.</param>
        /// <param name="filterKeys">
        ///     The keys to keep, as a Python sequence: an <see langword="object"/>?[] (a tuple) or a list of strings for a
        ///     <c>Dict</c>, of integers for a <c>Tuple</c> (negative indices count from the end). A string is a sequence of
        ///     one-character keys, as in Python.
        /// </param>
        /// <exception cref="TypeError">
        ///     Thrown, with Gymnasium's messages, when <paramref name="filterKeys"/> is no sequence, or its keys are not strings
        ///     for a <c>Dict</c> or not integers for a <c>Tuple</c>; also when <paramref name="env"/> is <see langword="null"/>.
        /// </exception>
        /// <exception cref="NumSharp.ValueError">
        ///     Thrown, with Gymnasium's messages, for keys the space lacks, duplicate tuple indices, a filter that keeps
        ///     nothing, or a space that is neither a <c>Dict</c> nor a <c>Tuple</c>.
        /// </exception>
        public FilterObservation(Env<TObs, TAct> env, object filterKeys) : base(env) {
            IReadOnlyList<object?> keys = ObservationWrapperFunctions.SequenceItems(filterKeys)
                                          ?? throw new TypeError($"Expects `filter_keys` to be a Sequence, actual type: {PyBuiltins.TypeRepr(filterKeys)}");
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["filter_keys"] = filterKeys });
            string shown = PyBuiltins.Str(filterKeys);

            switch (ObjectSpace.Unview(env.ObservationSpace)) {
                case Dict dict: {
                    // Filters for dictionary space.
                    if (!keys.All(key => key is string))
                        throw new TypeError($"All filter keys must be strings for a Dict space, got {shown}");
                    string[] names = keys.Cast<string>().ToArray();
                    string[] spaceKeys = dict.Keys.ToArray();
                    string[] missing = names.Where(key => !spaceKeys.Contains(key)).ToArray();
                    if (missing.Length > 0)
                        throw new ValueError("All the `filter_keys` must be included in the observation space.\n" +
                                             $"Filter keys: {shown}\n" +
                                             $"Observation keys: {PyBuiltins.Repr(spaceKeys.ToList())}\n" +
                                             $"Missing keys: {PyBuiltins.Repr(missing.ToList())}");

                    // {key: space[key] for key in filter_keys}: a dict comprehension, so a repeated key appears once, and a
                    // mapping, so Dict sorts its keys when the env's space does (sort_keys).
                    var subspaces = new Dictionary<string, Space>();
                    foreach (string key in names.Distinct())
                        subspaces[key] = dict[key].DeepCopy();
                    var filtered = new Dict(subspaces, sortKeys: dict.SortKeys);
                    if (filtered.Count == 0) {
                        filtered.Dispose();
                        throw new ValueError("The observation space is empty due to filtering all of the keys.");
                    }

                    ObservationSpace = ObservationWrapperFunctions.Typed<TObs>(filtered);
                    Func = observation => {
                        var source = ObservationWrapperFunctions.AsDict(observation);
                        var kept = new Info();
                        foreach (string key in names)
                            kept[key] = source[key];
                        return (TObs)(object)kept;
                    };
                    break;
                }
                case Tuple tuple: {
                    // Filter for tuple observation.
                    if (!keys.All(ObservationWrapperFunctions.IsPythonInt))
                        throw new TypeError($"All filter keys must be integers for a Tuple space, got {shown}");
                    long[] indices = keys.Select(key => Convert.ToInt64(key)).ToArray();
                    if (indices.Distinct().Count() != indices.Length)
                        throw new ValueError($"Duplicate keys exist, filter_keys: {shown}");
                    long length = tuple.Spaces.Count;
                    // Gymnasium's own range check: 0 < key and key >= len(space), so negative indices pass it.
                    long[] outOfRange = indices.Where(key => 0 < key && key >= length).ToArray();
                    if (outOfRange.Length > 0)
                        throw new ValueError("All the `filter_keys` must be included in the length of the observation space.\n" +
                                             $"Filter keys: {shown}, length of observation: {length}, " +
                                             $"missing indexes: {PyBuiltins.Repr(outOfRange.Select(i => (object?)i).ToList())}");

                    int[] positions = indices.Select(key => (int)(key < 0 ? key + length : key)).ToArray();
                    var filtered = new Tuple(positions.Select(position => tuple.Spaces[position].DeepCopy()));
                    if (filtered.Spaces.Count == 0) {
                        filtered.Dispose();
                        throw new ValueError("The observation space is empty due to filtering all keys.");
                    }

                    ObservationSpace = ObservationWrapperFunctions.Typed<TObs>(filtered);
                    Func = observation => {
                        object?[] source = ObservationWrapperFunctions.AsTuple(observation);
                        return (TObs)(object)positions.Select(position => source[position]).ToArray();
                    };
                    break;
                }
                case var other:
                    throw new ValueError($"FilterObservation wrapper is only usable with `Dict` and `Tuple` observations, actual type: {PyBuiltins.TypeRepr(other)}");
            }

            FilterKeys = filterKeys;
        }

        /// <summary>The keys kept, as given (<c>filter_keys</c>).</summary>
        public object FilterKeys { get; }
    }

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
    public partial class FlattenObservation<TObs, TAct> : TransformObservation<NDArray, TObs, TAct> {
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
            Func = observation => (NDArray)SpaceUtils.Flatten(_envObservationSpace, observation)!;
        }
    }

    /// <summary>
    ///     Converts an RGB image observation to grayscale (<c>0.2125 R + 0.7154 G + 0.0721 B</c>, truncated to uint8), as
    ///     Gymnasium's <c>GrayscaleObservation</c> does; <c>keep_dim</c> keeps a channel axis of length 1.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The weighted sum is NumPy's: the uint8 pixels times float64 weights, summed over the channel axis in float64, then
    ///     cast to uint8, which truncates. Every observation is a new array.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L259">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class GrayscaleObservation<TObs, TAct> : TransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must be a uint8 <c>Box</c> of shape <c>(h, w, 3)</c> with bounds 0 and 255.</param>
        /// <param name="keepDim">Whether observations keep a channel axis, <c>(h, w, 1)</c>, instead of <c>(h, w)</c>.</param>
        /// <exception cref="TypeError">Thrown when the space is no <c>Box</c> (Gymnasium's message), or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when the space is not a 3-channel image, or not uint8 within 0 and 255, with Gymnasium's messages.</exception>
        public GrayscaleObservation(Env<TObs, TAct> env, bool keepDim = false) : base(env) {
            if (ObjectSpace.Unview(env.ObservationSpace) is not Box box)
                throw new TypeError($"GrayscaleObservation requires a Box observation space, got {PyBuiltins.TypeRepr(ObjectSpace.Unview(env.ObservationSpace))}");
            long[] shape = box.Shape.dimensions;
            if (shape.Length != 3 || shape[^1] != 3)
                throw new ValueError($"GrayscaleObservation requires an image with 3 channels, got shape {PyBuiltins.Repr(WrapperFunctions.ShapeTuple(box.Shape))}");
            if (!ObservationWrapperFunctions.IsByteImage(box))
                throw new ValueError("GrayscaleObservation requires observation bounds 0 to 255 and dtype uint8");
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["keep_dim"] = keepDim });

            KeepDim = keepDim;
            Shape spaceShape = keepDim ? new Shape(shape[0], shape[1], 1L) : new Shape(shape[0], shape[1]);
            ObservationSpace = new Box(0, 255, spaceShape, np.uint8);
            Func = observation => {
                NDArray gray = Gray(ObservationWrapperFunctions.AsArray(observation));
                if (!keepDim)
                    return gray;
                using (gray)
                    return np.expand_dims(gray, -1);
            };
        }

        /// <summary>Whether the channel axis is kept (<c>keep_dim</c>).</summary>
        public bool KeepDim { get; }

        /// <summary><c>np.sum(np.multiply(obs, np.array([0.2125, 0.7154, 0.0721])), axis=-1).astype(np.uint8)</c>.</summary>
        /// <param name="image">The RGB image.</param>
        /// <returns>A new <c>(h, w)</c> uint8 array.</returns>
        private static NDArray Gray(NDArray image) {
            using NDArray weights = np.array(new[] { 0.2125, 0.7154, 0.0721 });
            using NDArray weighted = np.multiply(image, weights);
            using NDArray summed = np.sum(weighted, axis: -1);
            return summed.astype(np.uint8);
        }
    }

    /// <summary>Reshapes every array observation, as Gymnasium's <c>ReshapeObservation</c> does; the space's bounds are reshaped with it.</summary>
    /// <typeparam name="TObs">The env's observation type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Each observation is <c>np.reshape(obs, shape)</c>, a view of the env's observation where NumPy gives one, as in
    ///     Python: disposing it does not free the env's array.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L449">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class ReshapeObservation<TObs, TAct> : TransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must be a <c>Box</c>.</param>
        /// <param name="shape">The new shape; its product must equal the space's (Gymnasium's check, which a <c>-1</c> element therefore never passes).</param>
        /// <exception cref="TypeError">Thrown when the space is no <c>Box</c> (Gymnasium's message), or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="shape"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when the products differ, or an element is neither positive nor -1, with Gymnasium's messages.</exception>
        public ReshapeObservation(Env<TObs, TAct> env, long[] shape) : base(env) {
            if (ObjectSpace.Unview(env.ObservationSpace) is not Box box)
                throw new TypeError($"ReshapeObservation requires a Box observation space, got {PyBuiltins.TypeRepr(ObjectSpace.Unview(env.ObservationSpace))}");
            if (shape is null)
                throw new ArgumentNullException(nameof(shape));
            // np.prod of the new shape against np.prod of the space's (1.0 for an empty shape): the same number or not.
            long newProduct = shape.Aggregate(1L, (product, d) => product * d);
            long oldProduct = box.Shape.dimensions.Aggregate(1L, (product, d) => product * d);
            if (newProduct != oldProduct)
                throw new ValueError("ReshapeObservation requires the product of the new shape to match the original shape");
            object?[] shapeTuple = shape.Select(d => (object?)d).ToArray();
            if (!shape.All(x => x > 0 || x == -1))
                throw new ValueError($"ReshapeObservation requires shape elements to be greater than 0 or -1, got {PyBuiltins.Repr(shapeTuple)}");

            using (NDArray flatLow = np.ravel(box.Low))
            using (NDArray flatHigh = np.ravel(box.High))
            using (NDArray low = np.reshape(flatLow, new Shape(shape)))
            using (NDArray high = np.reshape(flatHigh, new Shape(shape))) {
                ObservationSpace = new Box(low, high, new Shape(shape), box.DType);
            }

            Shape = (long[])shape.Clone();
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["shape"] = shapeTuple });
            var target = new Shape(Shape);
            Func = observation => np.reshape(ObservationWrapperFunctions.AsArray(observation), target);
        }

        /// <summary>The new shape (<c>shape</c>).</summary>
        public long[] Shape { get; }
    }

    /// <summary>
    ///     Rescales a <c>Box</c> observation affinely onto <c>[min_obs, max_obs]</c>, as Gymnasium's
    ///     <c>RescaleObservation</c> does; unbounded components must stay unbounded and are only shifted.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Each observation is <c>gradient * obs + intercept</c>, both arrays in the space's dtype, computed as
    ///     <see cref="WrapperFunctions.RescaleBox"/> does (Windows NumPy's float64 path).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L513">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class RescaleObservation<TObs, TAct> : TransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must be a <c>Box</c>.</param>
        /// <param name="minObs">The new lower bound: an array of the space's shape, or a number (<see cref="WrapperFunctions.RescaleBox"/>).</param>
        /// <param name="maxObs">The new upper bound, in the same forms.</param>
        /// <exception cref="TypeError">Thrown when the space is no <c>Box</c> or a bound has the wrong type (Gymnasium's messages), or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when the bounds do not fit the space (see <see cref="WrapperFunctions.RescaleBox"/>).</exception>
        public RescaleObservation(Env<TObs, TAct> env, object minObs, object maxObs) : base(env) {
            if (ObjectSpace.Unview(env.ObservationSpace) is not Box box)
                throw new TypeError($"RescaleObservation requires a Box observation space, got {PyBuiltins.TypeRepr(ObjectSpace.Unview(env.ObservationSpace))}");
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["min_obs"] = minObs, ["max_obs"] = maxObs });
            (Box rescaled, Func<NDArray, NDArray> forward, _) = WrapperFunctions.RescaleBox(box, minObs, maxObs);
            ObservationSpace = rescaled;
            Func = observation => forward(ObservationWrapperFunctions.AsArray(observation));
        }
    }

    /// <summary>
    ///     Converts every observation to a dtype, as Gymnasium's <c>DtypeObservation</c> does: a <c>Box</c>,
    ///     <c>MultiDiscrete</c> or <c>MultiBinary</c> observation keeps its shape, a <c>Discrete</c> one becomes a 0-d array
    ///     of the dtype (a NumPy scalar), and the space follows.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type: arrays or numbers (<see cref="NDArray"/>, or <see langword="object"/>).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The conversion is NumPy's <c>dtype.type(obs)</c>, an unsafe cast: float to integer truncates, and out-of-range
    ///     values wrap. A <c>Discrete(n, start)</c> space becomes <c>Box(start, start + n - 1, (), dtype)</c>, a
    ///     <c>MultiBinary</c> one <c>Box(0, 1, shape, dtype)</c>, a <c>MultiDiscrete</c> one keeps its <c>nvec</c> and
    ///     <c>start</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L564">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class DtypeObservation<TObs, TAct> : TransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must be a <c>Box</c>, <c>Discrete</c>, <c>MultiDiscrete</c> or <c>MultiBinary</c>.</param>
        /// <param name="dtype">The dtype: a <see cref="DType"/> (<c>np.float32</c>), its name (<c>"float32"</c>), an <see cref="NPTypeCode"/> or a .NET <see cref="Type"/>.</param>
        /// <exception cref="TypeError">Thrown when the space is of another kind (Gymnasium's message), the dtype cannot be read, or <paramref name="env"/> is <see langword="null"/>.</exception>
        public DtypeObservation(Env<TObs, TAct> env, object dtype) : base(env) {
            Space envSpace = ObjectSpace.Unview(env.ObservationSpace)!;
            if (envSpace is not (Box or Discrete or MultiDiscrete or MultiBinary))
                throw new TypeError($"DtypeObservation requires a Box, Discrete, MultiDiscrete, or MultiBinary space, got {PyBuiltins.TypeRepr(envSpace)}");

            Dtype = dtype;
            DType target = ObservationWrapperFunctions.ToDType(dtype);
            ObservationSpace = envSpace switch {
                Box box => new Box(box.Low, box.High, box.Shape, target),
                Discrete discrete => new Box((double)discrete.Start, (double)(discrete.Start + discrete.N - 1), Shape.Scalar, target),
                MultiDiscrete multiDiscrete => new MultiDiscrete(multiDiscrete.Nvec, dtype: target, start: multiDiscrete.Start),
                _ => new Box(0, 1, envSpace.Shape, target),
            };

            SavedKwargs = RecordConstructorArgs.Record(new Info { ["dtype"] = dtype });
            Func = observation => ObservationWrapperFunctions.Cast(observation, target);
        }

        /// <summary>The dtype as given (<c>dtype</c>).</summary>
        public object Dtype { get; }
    }

    /// <summary>
    ///     Puts the rendered frame into every observation, as Gymnasium's <c>AddRenderObservation</c> does: with
    ///     <c>render_only</c> the frame replaces the observation, otherwise it joins it in a dict (beside a <c>Dict</c>'s keys,
    ///     or as <c>{obs_key: obs, render_key: frame}</c>).
    /// </summary>
    /// <typeparam name="TObs">The env's observation type.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The constructor resets the env and renders once to learn the frame's shape, as Gymnasium's does: the reset draws
    ///     from the env's generator (seeding it from entropy if it was never seeded), so seed the next reset to reproduce a
    ///     run. Each observation renders again; the env's render mode must give arrays (<c>rgb_array</c>, <c>depth_array</c>).
    ///     The observation type depends on <c>render_only</c> at run time (an array, or a dict), so observations are
    ///     <see langword="object"/>s.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L634">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class AddRenderObservation<TObs, TAct> : TransformObservation<object?, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its render mode must be set and not <c>human</c>.</param>
        /// <param name="renderOnly">Whether the frame replaces the observation (the default) instead of joining it in a dict.</param>
        /// <param name="renderKey">The frame's key in the dict.</param>
        /// <param name="obsKey">The observation's key when the env's space is no <c>Dict</c>.</param>
        /// <exception cref="NumSharp.ValueError">Thrown when the render mode is unset or <c>human</c>, or <paramref name="renderKey"/> already is a key of the env's <c>Dict</c> space, with Gymnasium's messages.</exception>
        /// <exception cref="TypeError">Thrown when rendering gives no array (Gymnasium's message), or <paramref name="env"/> is <see langword="null"/>.</exception>
        public AddRenderObservation(Env<TObs, TAct> env, bool renderOnly = true, string renderKey = "pixels", string obsKey = "state") : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["render_only"] = renderOnly, ["render_key"] = renderKey, ["obs_key"] = obsKey });

            if (env.RenderMode is null or "human")
                throw new ValueError($"AddRenderObservation requires render_mode to be not None and not 'human', got {PyBuiltins.Str(env.RenderMode)}");
            // Learn the frame's shape from one render after a reset, as Gymnasium does; the reset's observation is not kept.
            ResetResult<TObs> first = env.Reset();
            (first.Observation as IDisposable)?.Dispose();
            Shape frameShape;
            using (RenderResult rendered = env.Render()) {
                NDArray? pixels = ObservationWrapperFunctions.RenderedArray(rendered);
                if (pixels is null)
                    throw new TypeError($"AddRenderObservation expects env.render() to return a numpy array, got {ObservationWrapperFunctions.RenderTypeRepr(rendered)}");
                frameShape = new Shape(pixels.Shape.dimensions);
            }

            var pixelSpace = new Box(0, 255, frameShape, np.uint8);
            Space envSpace = ObjectSpace.Unview(env.ObservationSpace)!;
            if (renderOnly) {
                ObservationSpace = new ObjectSpace(pixelSpace);
                Func = _ => RenderFrame();
            } else if (envSpace is Dict dict) {
                if (dict.Keys.Contains(renderKey)) {
                    pixelSpace.Dispose();
                    throw new ValueError($"AddRenderObservation render_key '{renderKey}' already exists in the observation space");
                }

                // {render_key: pixel_space, **env.observation_space.spaces}: a mapping, the frame first, then the env's keys,
                // which Dict sorts when the env's space does (sort_keys).
                var spaces = new Dictionary<string, Space> { [renderKey] = pixelSpace };
                foreach (KeyValuePair<string, Space> pair in dict)
                    spaces[pair.Key] = pair.Value.DeepCopy();
                ObservationSpace = new ObjectSpace(new Dict(spaces, sortKeys: dict.SortKeys));
                Func = observation => {
                    var joined = new Info { [renderKey] = RenderFrame() };
                    foreach (KeyValuePair<string, object?> pair in ObservationWrapperFunctions.AsDict(observation))
                        joined[pair.Key] = pair.Value;
                    return joined;
                };
            } else {
                // Dict(...) of a plain dict sorts its keys, as Gymnasium's default sort_keys=True does.
                ObservationSpace = new ObjectSpace(new Dict(new Dictionary<string, Space> { [obsKey] = envSpace.DeepCopy(), [renderKey] = pixelSpace }));
                Func = observation => new Info { [obsKey] = observation, [renderKey] = RenderFrame() };
            }

            RenderOnly = renderOnly;
            RenderKey = renderKey;
            ObsKey = obsKey;
        }

        /// <summary>Whether the frame replaces the observation (<c>render_only</c>).</summary>
        public bool RenderOnly { get; }

        /// <summary>The frame's key (<c>render_key</c>).</summary>
        public string RenderKey { get; }

        /// <summary>The observation's key beside the frame (<c>obs_key</c>).</summary>
        public string ObsKey { get; }

        /// <summary>This wrapper's <c>self.render()</c> as an array: the frame the env renders now.</summary>
        /// <returns>The frame; the caller owns it.</returns>
        /// <exception cref="TypeError">Thrown when the rendering gives no array.</exception>
        private NDArray RenderFrame() {
            RenderResult rendered = Render();
            return ObservationWrapperFunctions.RenderedArray(rendered)
                   ?? throw new TypeError($"AddRenderObservation expects env.render() to return a numpy array, got {ObservationWrapperFunctions.RenderTypeRepr(rendered)}");
        }
    }

    /// <summary>
    ///     Discretizes a finite <c>Box</c> observation uniformly, as Gymnasium's <c>DiscretizeObservation</c> does: each of
    ///     the box's elements into <c>bins</c> equal bins, the indices combined into one <c>Discrete</c> number (row-major)
    ///     or kept apart as a <c>MultiDiscrete</c> vector.
    /// </summary>
    /// <typeparam name="TObs">The env's observation type: arrays (<see cref="NDArray"/>, or <see langword="object"/> holding them).</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The bin edges are NumPy's <c>np.linspace(low[i], high[i], bins[i] + 1)</c> in the box's dtype; an observation is
    ///     clipped to <c>[low, high - 1e-8]</c> (the margin in the box's dtype, so for float32 it often vanishes) and each
    ///     element located with <c>np.digitize</c>. A <c>Discrete</c> observation is a 0-d int64 array (Python's
    ///     <c>int</c>).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/transform_observation.py#L762">gymnasium/wrappers/transform_observation.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class DiscretizeObservation<TObs, TAct> : TransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env; its observation space must be a finite <c>Box</c>.</param>
        /// <param name="bins">The bins per element: one number for every element, or a sequence of one per element.</param>
        /// <param name="multidiscrete">Whether observations are <c>MultiDiscrete</c> vectors instead of one <c>Discrete</c> number.</param>
        /// <exception cref="TypeError">Thrown when the space is no <c>Box</c> (Gymnasium's message), or <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown when a bound is infinite, or the bins do not match the elements, with Gymnasium's messages.</exception>
        public DiscretizeObservation(Env<TObs, TAct> env, object bins, bool multidiscrete = false) : base(env) {
            if (ObjectSpace.Unview(env.ObservationSpace) is not Box box)
                throw new TypeError("DiscretizeObservation is only compatible with Box continuous observations.");

            Low = np.ravel(box.Low);
            High = np.ravel(box.High);
            NDims = (int)Low.size;
            ObservationWrapperFunctions.RequireFinite(Low, High, "observation", () => {
                Low.Dispose();
                High.Dispose();
            });

            Multidiscrete = multidiscrete;
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["bins"] = bins, ["multidiscrete"] = multidiscrete });
            Bins = ObservationWrapperFunctions.BinsPerDimension(bins, NDims, "action");

            // np.linspace(low[i], high[i], bins[i] + 1)[1:-1]: the inner edges.
            BinEdges = new NDArray[NDims];
            for (int i = 0; i < NDims; i++) {
                using NDArray start = Low[i];
                using NDArray stop = High[i];
                using NDArray edges = WrapperFunctions.Linspace(start, stop, Bins[i] + 1);
                BinEdges[i] = edges[$"1:-1"].copy();
            }

            if (multidiscrete) {
                ObservationSpace = new MultiDiscrete(Bins);
            } else {
                ObservationSpace = new Discrete(Bins.Aggregate(1L, (product, b) => product * b));
            }

            Func = Discretize;
        }

        /// <summary>The box's lower bounds, flattened (<c>low</c>).</summary>
        public NDArray Low { get; }

        /// <summary>The box's upper bounds, flattened (<c>high</c>).</summary>
        public NDArray High { get; }

        /// <summary>The number of elements discretized (<c>n_dims</c>).</summary>
        [PyName("n_dims")]
        public int NDims { get; }

        /// <summary>Whether observations are <c>MultiDiscrete</c> (<c>multidiscrete</c>).</summary>
        public bool Multidiscrete { get; }

        /// <summary>The bins of each element (<c>bins</c>).</summary>
        public long[] Bins { get; }

        /// <summary>The inner bin edges of each element (<c>bin_edges</c>), in the box's dtype.</summary>
        public NDArray[] BinEdges { get; }

        /// <summary>
        ///     The bin bounds a discretized observation came from (<c>revert_observation</c>): for each element, the edges
        ///     <c>edges[idx]</c> and <c>edges[idx + 1]</c> of its bin.
        /// </summary>
        /// <param name="observation">A discretized observation: a number (or 0-d array) for <c>Discrete</c>, an array for <c>MultiDiscrete</c>.</param>
        /// <returns>New arrays of the lower and upper edges, in the env's observation dtype; the caller owns them.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown when an index is beyond its element's bins.</exception>
        public (NDArray Low, NDArray High) RevertObservation(object observation) {
            long[] indices = Multidiscrete ? ObservationWrapperFunctions.Longs(observation) : UnflattenIndex(ObservationWrapperFunctions.Long(observation));
            var lows = new NDArray[NDims];
            var highs = new NDArray[NDims];
            try {
                for (int i = 0; i < indices.Length; i++) {
                    using NDArray start = Low[i];
                    using NDArray stop = High[i];
                    using NDArray edges = WrapperFunctions.Linspace(start, stop, Bins[i] + 1);
                    lows[i] = edges[indices[i]].copy();
                    highs[i] = edges[indices[i] + 1].copy();
                }

                DType dtype = ObjectSpace.Unview(Env.ObservationSpace)!.DType!;
                return (np.stack(lows).astype(dtype), np.stack(highs).astype(dtype));
            } finally {
                foreach (NDArray? edge in lows.Concat(highs))
                    edge?.Dispose();
            }
        }

        /// <summary>Discretizes one observation (<c>observation(observation)</c>).</summary>
        /// <param name="observation">The env's observation.</param>
        /// <returns>A new 0-d int64 array (the flat index) or <c>(n_dims,)</c> int64 array (one index per element).</returns>
        private NDArray Discretize(TObs observation) {
            // np.clip(np.ravel(observation), low, high - 1e-8): the margin is a weak Python float, so it takes the box's dtype.
            using NDArray flat = np.ravel(ObservationWrapperFunctions.AsArray(observation));
            using NDArray upper = High - 1e-8;
            using NDArray clipped = np.clip(flat, Low, upper);
            var indices = new long[NDims];
            for (int i = 0; i < NDims; i++) {
                using NDArray element = clipped[i];
                using NDArray index = np.digitize(element, BinEdges[i]);
                indices[i] = Convert.ToInt64(index.GetAtIndex(0));
            }

            if (Multidiscrete)
                return np.array(indices);
            // int(self._flatten_indices(indices)): a Python int.
            long flatIndex = 0;
            for (int i = 0; i < NDims; i++)
                flatIndex = flatIndex * Bins[i] + indices[i];
            return NDArray.Scalar(flatIndex);
        }

        /// <summary>The per-element indices of a flat index (<c>_unflatten_index</c>), last element varying fastest.</summary>
        /// <param name="flatIndex">The flat index.</param>
        /// <returns>One index per element.</returns>
        private long[] UnflattenIndex(long flatIndex) {
            var indices = new long[NDims];
            for (int i = NDims - 1; i >= 0; i--) {
                // Python's % and // on non-negative ints.
                indices[i] = PyMath.Mod(flatIndex, Bins[i]);
                flatIndex = PyMath.FloorDiv(flatIndex, Bins[i]);
            }

            return indices;
        }

        /// <summary>Releases the bounds and edges, then disposes the wrapper (and the env beneath) as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                // Low and High are views of the env's space bounds: released before the env disposes its space.
                Low.Dispose();
                High.Dispose();
                foreach (NDArray edges in BinEdges)
                    edges?.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    /// <summary>
    ///     The conversions Gymnasium's observation and action wrappers get from Python's dynamic typing: reading an
    ///     observation as an array, dict or tuple whatever its static type, reading Python sequences and dtypes, and the
    ///     checks the discretizing wrappers share.
    /// </summary>
    internal static class ObservationWrapperFunctions {
        /// <summary>An observation as an array: an <see cref="NDArray"/> as it is, a .NET number as a 0-d array (a NumPy scalar).</summary>
        /// <param name="value">The observation.</param>
        /// <returns>The array; a new one only for a number.</returns>
        /// <exception cref="TypeError">Thrown for anything else.</exception>
        internal static NDArray AsArray(object? value) => value switch {
            NDArray array => array,
            bool or sbyte or byte or short or ushort or int or uint or long or ulong or float or double => NDArray.Scalar(value),
            _ => throw new TypeError($"expected a numpy array, got {PyBuiltins.TypeRepr(value)}"),
        };

        /// <summary>An observation as a dict.</summary>
        /// <param name="value">The observation.</param>
        /// <returns>The dict.</returns>
        /// <exception cref="TypeError">Thrown when it is no dict.</exception>
        internal static IReadOnlyDictionary<string, object?> AsDict(object? value) =>
            value as IReadOnlyDictionary<string, object?> ?? throw new TypeError($"expected a dict observation, got {PyBuiltins.TypeRepr(value)}");

        /// <summary>An observation as a tuple: an <see langword="object"/>?[] as it is, any other <see cref="System.Runtime.CompilerServices.ITuple"/> by its items.</summary>
        /// <param name="value">The observation.</param>
        /// <returns>The items.</returns>
        /// <exception cref="TypeError">Thrown when it is no tuple.</exception>
        internal static object?[] AsTuple(object? value) {
            switch (value) {
                case object?[] items:
                    return items;
                case System.Runtime.CompilerServices.ITuple tuple: {
                    var items = new object?[tuple.Length];
                    for (int i = 0; i < items.Length; i++)
                        items[i] = tuple[i];
                    return items;
                }
                default:
                    throw new TypeError($"expected a tuple observation, got {PyBuiltins.TypeRepr(value)}");
            }
        }

        /// <summary>
        ///     The items of a Python sequence (<c>isinstance(x, Sequence)</c>): a string's one-character strings, a tuple's or
        ///     list's items; <see langword="null"/> for anything else (a set, a number, a bool).
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>The items, or <see langword="null"/> when the value is no sequence.</returns>
        internal static IReadOnlyList<object?>? SequenceItems(object? value) {
            switch (value) {
                case string text:
                    return Text.CodePoints(text).Cast<object?>().ToList();
                case object?[] items:
                    return items;
                case System.Runtime.CompilerServices.ITuple tuple:
                    return AsTuple(tuple);
                case IList list:
                    return list.Cast<object?>().ToList();
                default:
                    return null;
            }
        }

        /// <summary>Whether a value is a Python <c>int</c> (a .NET integer, not a bool).</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for an integer.</returns>
        internal static bool IsPythonInt(object? value) => value is sbyte or byte or short or ushort or int or uint or long or ulong;

        /// <summary>A space typed for an env's observation type: the space itself when it holds <typeparamref name="T"/>, else its untyped view.</summary>
        /// <typeparam name="T">The observation type.</typeparam>
        /// <param name="space">The space; the result owns it.</param>
        /// <returns>The typed space.</returns>
        /// <exception cref="TypeError">Thrown when the space holds another type and <typeparamref name="T"/> is not <see langword="object"/>.</exception>
        internal static Space<T> Typed<T>(Space space) {
            if (space is Space<T> typed)
                return typed;
            if (typeof(T) == typeof(object))
                return (Space<T>)(Space)new ObjectSpace(space);
            throw new TypeError($"{space} does not hold {typeof(T).Name} observations");
        }

        /// <summary>Whether a <c>Box</c> is a uint8 image within 0 and 255 (<c>np.all(low == 0) and np.all(high == 255) and dtype == uint8</c>).</summary>
        /// <param name="box">The box.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        internal static bool IsByteImage(Box box) {
            using NDArray lowZero = box.Low == 0;
            using NDArray highFull = box.High == 255;
            return np.all(lowZero) && np.all(highFull) && box.DType == np.uint8;
        }

        /// <summary>A dtype given as a <see cref="DType"/>, its name, an <see cref="NPTypeCode"/> or a .NET type, as <c>np.dtype(x)</c> reads it.</summary>
        /// <param name="dtype">The dtype.</param>
        /// <returns>The dtype.</returns>
        /// <exception cref="TypeError">Thrown for anything else.</exception>
        internal static DType ToDType(object dtype) => dtype switch {
            DType d => d,
            string name => np.dtype(name),
            NPTypeCode code => np.dtype(code),
            Type type => np.dtype(type),
            _ => throw new TypeError($"Cannot interpret {PyBuiltins.Repr(dtype)} as a data type"),
        };

        /// <summary><c>dtype.type(value)</c>: an unsafe cast to the dtype, a number becoming a 0-d array.</summary>
        /// <param name="value">The observation.</param>
        /// <param name="dtype">The dtype.</param>
        /// <returns>A new array the caller owns.</returns>
        /// <exception cref="TypeError">Thrown when the observation is neither an array nor a number.</exception>
        internal static NDArray Cast(object? value, DType dtype) {
            if (value is NDArray array)
                return array.astype(dtype);
            using NDArray scalar = AsArray(value);
            return scalar.astype(dtype);
        }

        /// <summary>The array a rendering holds (<c>isinstance(render(), np.ndarray)</c>): an <c>rgb_array</c> or <c>depth_array</c> frame; <see langword="null"/> otherwise.</summary>
        /// <param name="rendered">The rendering.</param>
        /// <returns>The frame, owned with the rendering, or <see langword="null"/>.</returns>
        internal static NDArray? RenderedArray(RenderResult rendered) => rendered switch {
            RenderResult.Rgb rgb => rgb.Frame,
            RenderResult.Depth depth => depth.Frame,
            _ => null,
        };

        /// <summary>Python's <c>type(env.render())</c> for a rendering that is no array: <c>NoneType</c>, <c>str</c>, <c>tuple</c> or <c>list</c>.</summary>
        /// <param name="rendered">The rendering.</param>
        /// <returns>The type's repr.</returns>
        internal static string RenderTypeRepr(RenderResult rendered) => rendered switch {
            RenderResult.NoneResult => "<class 'NoneType'>",
            RenderResult.Ansi => "<class 'str'>",
            RenderResult.RgbDepth => "<class 'tuple'>",
            RenderResult.Frames => "<class 'list'>",
            _ => "<class 'numpy.ndarray'>",
        };

        /// <summary>
        ///     The discretizing wrappers' finiteness check: <c>Discretization requires {what} space to be finite</c> when a
        ///     bound is infinite, after running <paramref name="release"/> for the arrays the caller made.
        /// </summary>
        /// <param name="low">The flattened lower bounds.</param>
        /// <param name="high">The flattened upper bounds.</param>
        /// <param name="what"><c>observation</c> or <c>action</c>.</param>
        /// <param name="release">Releases the caller's arrays before throwing.</param>
        /// <exception cref="NumSharp.ValueError">Thrown when a bound is infinite, with Gymnasium's message.</exception>
        internal static void RequireFinite(NDArray low, NDArray high, string what, Action release) {
            using NDArray lowInfinite = np.isinf(low);
            using NDArray highInfinite = np.isinf(high);
            if (np.any(lowInfinite) || np.any(highInfinite)) {
                string message = $"Discretization requires {what} space to be finite. Found: low={PyBuiltins.Str(low)}, high={PyBuiltins.Str(high)}";
                release();
                throw new ValueError(message);
            }
        }

        /// <summary>
        ///     The discretizing wrappers' bins: one number repeated for every element, or a sequence of one per element
        ///     (Gymnasium's message says "action dimensions" for both wrappers).
        /// </summary>
        /// <param name="bins">The bins.</param>
        /// <param name="nDims">The number of elements.</param>
        /// <param name="what">The word in the message (<c>action</c>, as both of Gymnasium's wrappers print).</param>
        /// <returns>The bins of each element.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown when a sequence's length is not <paramref name="nDims"/>.</exception>
        /// <exception cref="TypeError">Thrown when the bins are neither a number nor a sequence of numbers.</exception>
        internal static long[] BinsPerDimension(object bins, int nDims, string what) {
            if (IsPythonInt(bins))
                return Enumerable.Repeat(Convert.ToInt64(bins), nDims).ToArray();
            IReadOnlyList<object?> items = bins is NDArray array
                ? array.ToArray<long>().Cast<object?>().ToList()
                : SequenceItems(bins) ?? throw new TypeError($"object of type {PyBuiltins.Repr(PyBuiltins.TypeName(bins))} has no len()");
            if (items.Count != nDims)
                throw new ValueError($"bins must match {what} dimensions: expected {nDims}, got {items.Count}");
            return items.Select(Convert.ToInt64).ToArray();
        }

        /// <summary>A number, or a 0-d array holding one, as a <see cref="long"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The integer.</returns>
        internal static long Long(object value) => value is NDArray array ? Convert.ToInt64(array.GetAtIndex(0)) : Convert.ToInt64(value);

        /// <summary>An array or sequence of integers as <see cref="long"/>s (<c>np.asarray(x, dtype=int)</c>).</summary>
        /// <param name="value">The value.</param>
        /// <returns>The integers.</returns>
        /// <exception cref="TypeError">Thrown when the value is neither.</exception>
        internal static long[] Longs(object value) {
            if (value is NDArray array) {
                using NDArray asLong = array.astype(np.int64);
                return asLong.ToArray<long>();
            }

            return (SequenceItems(value) ?? throw new TypeError($"expected a sequence of integers, got {PyBuiltins.TypeRepr(value)}")).Select(Convert.ToInt64).ToArray();
        }
    }
}
