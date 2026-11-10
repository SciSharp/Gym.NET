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
using Gym.Vector;
using NumSharp;

namespace Gym.Wrappers.Vector {
    /// <summary>
    ///     A single-env transform wrapper's function on one element, untyped: what the vectorizing wrappers call per
    ///     sub-env (Python's <c>self.wrapper.func(obs)</c>).
    /// </summary>
    internal interface IVectorizableTransform {
        /// <summary>Applies the wrapper's <c>func</c> to one element.</summary>
        /// <param name="value">The element: an observation, an action or a reward (a boxed <see cref="double"/>).</param>
        /// <returns>The transformed element.</returns>
        /// <exception cref="TypeError">Thrown when the wrapper has no function, or the element is not of its type.</exception>
        object? ApplyFunc(object? value);
    }

    /// <summary>
    ///     The stand-in single env a vectorizing wrapper builds its single-env wrapper over (Python's <c>_SingleEnv</c>, or a
    ///     bare <c>Env()</c> for rewards): it holds the sub-envs' single space and nothing else.
    /// </summary>
    /// <remarks>Stepping or resetting it is an error: the single-env wrapper is only asked for its spaces and its function.</remarks>
    internal sealed class VectorizedSingleEnv : Env<object?, object?> {
        /// <summary>A stand-in with the given spaces (an untyped view of each).</summary>
        /// <param name="observationSpace">The single observation space, or <see langword="null"/> to leave it unset.</param>
        /// <param name="actionSpace">The single action space, or <see langword="null"/> to leave it unset.</param>
        public VectorizedSingleEnv(Space? observationSpace, Space? actionSpace) {
            // The wrapper built over this env owns copies, never these: the vector env keeps its own spaces.
            if (observationSpace is not null)
                ((IEnv)this).ObservationSpace = observationSpace.DeepCopy();
            if (actionSpace is not null)
                ((IEnv)this).ActionSpace = actionSpace.DeepCopy();
        }

        /// <summary>Not supported: the stand-in only lends its spaces.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Never.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<object?> Step(object? action) => throw new NotImplementedError("The single env of a vectorized wrapper cannot step.");

        /// <summary>Not supported: the stand-in only lends its spaces.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Never.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) =>
            throw new NotImplementedError("The single env of a vectorized wrapper cannot reset.");
    }

    /// <summary>The construction and the per-element work the vectorizing wrappers share.</summary>
    internal static class Vectorizing {
        /// <summary>
        ///     Builds the single-env wrapper over the stand-in env (Python's <c>wrapper(self._SingleEnv(space), **kwargs)</c>):
        ///     an open generic class closes over the stand-in's untyped types.
        /// </summary>
        /// <param name="wrapper">The single-env wrapper class, open or closed.</param>
        /// <param name="single">The stand-in env; the built wrapper owns it.</param>
        /// <param name="kwargs">The wrapper's other arguments by Python name, or <see langword="null"/>.</param>
        /// <returns>The single-env wrapper.</returns>
        /// <exception cref="TypeError">Thrown when the class is not a transform wrapper, or the arguments do not bind.</exception>
        internal static IEnv Build(Type wrapper, VectorizedSingleEnv single, object? kwargs) {
            Info arguments = PyKwargs.From(kwargs);
            arguments["env"] = single;
            object built = PyKwargs.Construct(wrapper, arguments);
            if (built is not IVectorizableTransform || built is not IEnv env) {
                (built as IDisposable)?.Dispose();
                throw new TypeError($"{PyAttributes.PyTypeName(wrapper)} is not a transform wrapper that can be vectorized");
            }

            return env;
        }

        /// <summary>
        ///     Reads <c>metadata["autoreset_mode"]</c>: a missing key warns and means <c>NEXT_STEP</c>; anything but an
        ///     <see cref="AutoresetMode"/> is refused.
        /// </summary>
        /// <param name="env">The vector env.</param>
        /// <param name="missingWarning">The warning for a missing key.</param>
        /// <returns>The mode.</returns>
        /// <exception cref="TypeError">Thrown when the value is not an <see cref="AutoresetMode"/>, with Gymnasium's message.</exception>
        internal static AutoresetMode AutoresetModeOf(IVectorEnv env, string missingWarning) {
            if (!env.Metadata.TryGetValue("autoreset_mode", out object? mode)) {
                Logger.Warn(missingWarning);
                return AutoresetMode.NextStep;
            }

            return mode as AutoresetMode ?? throw new TypeError($"Expected env.metadata['autoreset_mode'] to be an AutoresetMode, got {PyBuiltins.TypeRepr(mode)}");
        }

        /// <summary>
        ///     <c>concatenate(space, tuple(func(item) for item in iterate(batched, items)), out)</c>: each sub-env's element
        ///     transformed, then stacked into <paramref name="out"/>.
        /// </summary>
        /// <param name="transform">The single-env wrapper.</param>
        /// <param name="iterated">The batched space the items are read by.</param>
        /// <param name="single">The single space the results are stacked by.</param>
        /// <param name="items">The batched elements.</param>
        /// <param name="out">The buffer to stack into: the batched elements themselves, or the wrapper's own.</param>
        /// <returns>What <see cref="VectorUtils.Concatenate"/> returns (<paramref name="out"/> filled, for arrays).</returns>
        internal static object? Concatenate(IVectorizableTransform transform, Space iterated, Space single, object? items, object? @out) {
            List<object?> transformed = VectorUtils.Iterate(iterated, items).Select(transform.ApplyFunc).ToList();
            return VectorUtils.Concatenate(single, transformed, @out);
        }

        /// <summary>Releases the arrays of a buffer the wrapper made: an array, or the arrays inside a tuple or dict.</summary>
        /// <param name="value">The buffer, or <see langword="null"/>.</param>
        internal static void Release(object? value) {
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
            }
        }
    }

    /// <summary>
    ///     Applies one function to the whole batch of observations, as Gymnasium's vector <c>TransformObservation</c> does;
    ///     give the spaces when the function changes them. Use <see cref="VectorizeTransformObservation{TWObs, TObs, TAct}"/>
    ///     to apply a single-env wrapper's function per sub-env instead.
    /// </summary>
    /// <typeparam name="TWObs">The wrapper's batched observation type.</typeparam>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Without spaces the env's are kept; with only a single space, the batched one is <c>batch_space(single,
    ///     num_envs)</c>. When the batched space is not the batched single space, a warning says so (Gymnasium's text, which
    ///     prints the env's observation space).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L20">gymnasium/wrappers/vector/vectorize_observation.py</see>.</para>
    /// </remarks>
    [PyName("TransformObservation")]
    public class TransformObservationVector<TWObs, TObs, TAct> : VectorObservationWrapper<TWObs, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="func">The function of the batched observations.</param>
        /// <param name="observationSpace">The wrapper's batched space, or <see langword="null"/>.</param>
        /// <param name="singleObservationSpace">The wrapper's single space, or <see langword="null"/>.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>, or a space does not hold <typeparamref name="TWObs"/>.</exception>
        /// <exception cref="ValueError">Thrown for a <c>SAME_STEP</c> env (the vector observation wrappers' rule).</exception>
        public TransformObservationVector(VectorEnv<TObs, TAct> env, Func<TObs, TWObs> func, Space<TWObs>? observationSpace = null, Space? singleObservationSpace = null) : base(env) {
            if (observationSpace is null) {
                if (singleObservationSpace is not null) {
                    SingleObservationSpace = singleObservationSpace;
                    ObservationSpace = ObservationWrapperFunctions.Typed<TWObs>(VectorUtils.BatchSpace(singleObservationSpace, NumEnvs));
                }
            } else {
                ObservationSpace = observationSpace;
                if (singleObservationSpace is not null)
                    SingleObservationSpace = singleObservationSpace;
            }

            using (Space batched = VectorUtils.BatchSpace(SingleObservationSpace, NumEnvs)) {
                if (!Equals(ObjectSpace.Unview(ObservationSpace), batched))
                    Logger.Warn($"For {env}, the observation space and the batched single observation space don't match as expected, observation_space={env.ObservationSpace}, batched single_observation_space={batched}");
            }

            Func = func;
        }

        /// <summary>The function of the batched observations (<c>func</c>).</summary>
        public Func<TObs, TWObs> Func { get; set; }

        /// <summary>Applies <see cref="Func"/> to the batch.</summary>
        /// <param name="observations">The env's batched observations.</param>
        /// <returns>The function's result.</returns>
        public override TWObs Observations(TObs observations) => Func(observations);
    }

    /// <summary>
    ///     Vectorizes a single-env observation transform (Gymnasium's <c>VectorizeTransformObservation</c>): the wrapper class
    ///     is built over a stand-in env of the single space, and its function is applied to each sub-env's observation, the
    ///     results stacked into the batch.
    /// </summary>
    /// <typeparam name="TWObs">The wrapper's batched observation type.</typeparam>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     When the wrapper's batched space equals the env's (<c>same_out</c>), the results are written into the env's own
    ///     batch, in place, as in Gymnasium; otherwise into the wrapper's buffer, and a deep copy is returned. A missing
    ///     <c>autoreset_mode</c> warns and means <c>NEXT_STEP</c>.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L104">gymnasium/wrappers/vector/vectorize_observation.py</see>.</para>
    /// </remarks>
    public class VectorizeTransformObservation<TWObs, TObs, TAct> : VectorObservationWrapper<TWObs, TObs, TAct> {
        /// <summary>The wrapper's buffer (<c>out</c>), from <c>create_empty_array</c>; owned.</summary>
        private readonly object? _out;

        /// <summary>Wraps <paramref name="env"/>, building <paramref name="wrapper"/> over its single observation space.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="wrapper">The single-env transform wrapper class (<c>typeof(FilterObservation&lt;,&gt;)</c>).</param>
        /// <param name="kwargs">The wrapper's other arguments by Python name (a dictionary or an anonymous object), or <see langword="null"/>.</param>
        /// <exception cref="TypeError">Thrown when the class cannot be vectorized or built, the mode is not an <see cref="AutoresetMode"/>, or a space does not hold <typeparamref name="TWObs"/>.</exception>
        /// <exception cref="ValueError">Thrown for a <c>SAME_STEP</c> env, or what the wrapper's constructor refuses.</exception>
        public VectorizeTransformObservation(VectorEnv<TObs, TAct> env, Type wrapper, object? kwargs = null) : base(env) {
            AutoresetMode = Vectorizing.AutoresetModeOf(env, $"Vector environment ({env}) is missing `autoreset_mode` metadata key.");
            Wrapper = Vectorizing.Build(wrapper, new VectorizedSingleEnv(ObjectSpace.Unview(env.SingleObservationSpace), null), kwargs);
            SingleObservationSpace = ObjectSpace.Unview(Wrapper.ObservationSpace)!.DeepCopy();
            ObservationSpace = ObservationWrapperFunctions.Typed<TWObs>(VectorUtils.BatchSpace(SingleObservationSpace, NumEnvs));
            SameOut = Equals(ObjectSpace.Unview(ObservationSpace), ObjectSpace.Unview(env.ObservationSpace));
            _out = VectorUtils.CreateEmptyArray(SingleObservationSpace, NumEnvs);
        }

        /// <summary>The env's autoreset mode (<c>autoreset_mode</c>).</summary>
        public AutoresetMode AutoresetMode { get; }

        /// <summary>The single-env wrapper whose function is applied (<c>wrapper</c>); owned.</summary>
        public IEnv Wrapper { get; }

        /// <summary>Whether the wrapper's batched space equals the env's, so results go into the env's batch (<c>same_out</c>).</summary>
        public bool SameOut { get; }

        /// <summary>Steps the env and transforms the observations; with <c>SAME_STEP</c>, each <c>final_obs</c> too.</summary>
        /// <param name="actions">The actions, passed on.</param>
        /// <returns>The transition with transformed observations.</returns>
        public override VectorStepResult<TWObs> Step(TAct actions) {
            VectorStepResult<TObs> result = Env.Step(actions);
            TWObs observations = Observations(result.Observations);
            if (ReferenceEquals(AutoresetMode, AutoresetMode.SameStep) && result.Infos.TryGetValue("final_obs", out object? finalObs)
                && finalObs is object?[] finals && result.Infos.TryGetValue("_final_obs", out object? hasFinal) && hasFinal is NDArray mask) {
                bool[] present = mask.ToArray<bool>();
                for (int i = 0; i < finals.Length; i++) {
                    if (present[i])
                        finals[i] = ((IVectorizableTransform)Wrapper).ApplyFunc(finals[i]);
                }
            }

            return new VectorStepResult<TWObs>(observations, result.Rewards, result.Terminations, result.Truncations, result.Infos);
        }

        /// <summary>Each sub-env's observation through the wrapper's function, stacked again.</summary>
        /// <param name="observations">The env's batched observations.</param>
        /// <returns>The env's batch filled in place (<see cref="SameOut"/>), or a deep copy of the wrapper's buffer.</returns>
        public override TWObs Observations(TObs observations) {
            var transform = (IVectorizableTransform)Wrapper;
            if (SameOut)
                return (TWObs)Vectorizing.Concatenate(transform, ObjectSpace.Unview(ObservationSpace)!, SingleObservationSpace, observations, observations)!;
            object? filled = Vectorizing.Concatenate(transform, ObjectSpace.Unview(Env.ObservationSpace)!, SingleObservationSpace, observations, _out);
            return (TWObs)PyCopy.DeepCopy(filled)!;
        }

        /// <summary>Releases the buffer and the single-env wrapper, then disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                Vectorizing.Release(_out);
                Wrapper.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    /// <summary>Keeps the chosen keys of a <c>Dict</c> or positions of a <c>Tuple</c> observation, per sub-env (Gymnasium's vector <c>FilterObservation</c>).</summary>
    /// <typeparam name="TObs">The batched observation type: <see cref="Info"/>, <see langword="object"/>?[] or <see langword="object"/>.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L246">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("FilterObservation")]
    public class FilterObservationVector<TObs, TAct> : VectorizeTransformObservation<TObs, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="filterKeys">The keys or positions to keep, as <see cref="FilterObservation{TObs, TAct}"/> takes them.</param>
        /// <exception cref="TypeError">Thrown for a space that is neither a <c>Dict</c> nor a <c>Tuple</c>, with Gymnasium's message.</exception>
        /// <exception cref="ValueError">Thrown for keys the space lacks, or no key left, with Gymnasium's messages.</exception>
        public FilterObservationVector(VectorEnv<TObs, TAct> env, object filterKeys) : base(env, typeof(FilterObservation<,>), new Info { ["filter_keys"] = filterKeys }) { }
    }

    /// <summary>Flattens each sub-env's observation (Gymnasium's vector <c>FlattenObservation</c>).</summary>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L287">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("FlattenObservation")]
    public class FlattenObservationVector<TObs, TAct> : VectorizeTransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single space flattens to an array.</param>
        /// <exception cref="TypeError">Thrown for a space whose flat form is not an array.</exception>
        public FlattenObservationVector(VectorEnv<TObs, TAct> env) : base(env, typeof(FlattenObservation<,>)) { }
    }

    /// <summary>Turns each sub-env's RGB image gray (Gymnasium's vector <c>GrayscaleObservation</c>).</summary>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L314">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("GrayscaleObservation")]
    public class GrayscaleObservationVector<TObs, TAct> : VectorizeTransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single space is a uint8 RGB image box.</param>
        /// <param name="keepDim">Whether the gray image keeps a channel axis of 1.</param>
        /// <exception cref="ValueError">Thrown for a space that is not an RGB image, with Gymnasium's messages.</exception>
        public GrayscaleObservationVector(VectorEnv<TObs, TAct> env, bool keepDim = false) : base(env, typeof(GrayscaleObservation<,>), new Info { ["keep_dim"] = keepDim }) { }
    }

    /// <summary>Resizes each sub-env's image by OpenCV's area interpolation (Gymnasium's vector <c>ResizeObservation</c>).</summary>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L346">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("ResizeObservation")]
    public class ResizeObservationVector<TObs, TAct> : VectorizeTransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single space is a uint8 image box.</param>
        /// <param name="shape">The new <c>(height, width)</c>.</param>
        /// <exception cref="TypeError">Thrown for a space that is not a <c>Box</c>.</exception>
        /// <exception cref="ValueError">Thrown for a space or shape <see cref="ResizeObservation{TObs, TAct}"/> refuses.</exception>
        public ResizeObservationVector(VectorEnv<TObs, TAct> env, long[] shape) : base(env, typeof(ResizeObservation<,>), new Info { ["shape"] = shape }) { }
    }

    /// <summary>Reshapes each sub-env's observation (Gymnasium's vector <c>ReshapeObservation</c>).</summary>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L376">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("ReshapeObservation")]
    public class ReshapeObservationVector<TObs, TAct> : VectorizeTransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single space is a <c>Box</c>.</param>
        /// <param name="shape">The new single shape.</param>
        /// <exception cref="TypeError">Thrown for a space that is not a <c>Box</c>.</exception>
        /// <exception cref="ValueError">Thrown for a shape of another size, with Gymnasium's message.</exception>
        public ReshapeObservationVector(VectorEnv<TObs, TAct> env, long[] shape) : base(env, typeof(ReshapeObservation<,>), new Info { ["shape"] = shape }) { }
    }

    /// <summary>Rescales each sub-env's <c>Box</c> observation affinely onto new bounds (Gymnasium's vector <c>RescaleObservation</c>).</summary>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L406">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("RescaleObservation")]
    public class RescaleObservationVector<TObs, TAct> : VectorizeTransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single space is a <c>Box</c>.</param>
        /// <param name="minObs">The new lower bound: a number or an array.</param>
        /// <param name="maxObs">The new upper bound: a number or an array.</param>
        /// <exception cref="TypeError">Thrown for a space that is not a <c>Box</c>.</exception>
        /// <exception cref="ValueError">Thrown for bounds <c>rescale_box</c> refuses.</exception>
        public RescaleObservationVector(VectorEnv<TObs, TAct> env, object minObs, object maxObs)
            : base(env, typeof(RescaleObservation<,>), new Info { ["min_obs"] = minObs, ["max_obs"] = maxObs }) { }
    }

    /// <summary>Casts each sub-env's observation to a dtype (Gymnasium's vector <c>DtypeObservation</c>).</summary>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_observation.py#L449">gymnasium/wrappers/vector/vectorize_observation.py</see>.</remarks>
    [PyName("DtypeObservation")]
    public class DtypeObservationVector<TObs, TAct> : VectorizeTransformObservation<NDArray, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single space is an array space.</param>
        /// <param name="dtype">The dtype, as <see cref="DtypeObservation{TObs, TAct}"/> takes it.</param>
        /// <exception cref="TypeError">Thrown for a space without arrays or a dtype that cannot be read.</exception>
        public DtypeObservationVector(VectorEnv<TObs, TAct> env, object dtype) : base(env, typeof(DtypeObservation<,>), new Info { ["dtype"] = dtype }) { }
    }

    /// <summary>
    ///     Applies one function to the whole batch of actions, as Gymnasium's vector <c>TransformAction</c> does; give the
    ///     spaces when the function's input differs from the env's actions.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TWAct">The wrapper's batched action type.</typeparam>
    /// <typeparam name="TAct">The env's batched action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_action.py#L19">gymnasium/wrappers/vector/vectorize_action.py</see>.</remarks>
    [PyName("TransformAction")]
    public class TransformActionVector<TObs, TWAct, TAct> : VectorActionWrapper<TObs, TWAct, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="func">The function from the wrapper's actions to the env's.</param>
        /// <param name="actionSpace">The wrapper's batched space, or <see langword="null"/>.</param>
        /// <param name="singleActionSpace">The wrapper's single space, or <see langword="null"/>.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>, or a space does not hold <typeparamref name="TWAct"/>.</exception>
        public TransformActionVector(VectorEnv<TObs, TAct> env, Func<TWAct, TAct> func, Space<TWAct>? actionSpace = null, Space? singleActionSpace = null) : base(env) {
            if (actionSpace is null) {
                if (singleActionSpace is not null) {
                    SingleActionSpace = singleActionSpace;
                    ActionSpace = ObservationWrapperFunctions.Typed<TWAct>(VectorUtils.BatchSpace(singleActionSpace, NumEnvs));
                }
            } else {
                ActionSpace = actionSpace;
                if (singleActionSpace is not null)
                    SingleActionSpace = singleActionSpace;
            }

            using (Space batched = VectorUtils.BatchSpace(SingleActionSpace, NumEnvs)) {
                if (!Equals(ObjectSpace.Unview(ActionSpace), batched))
                    Logger.Warn($"For {env}, the action space and the batched single action space don't match as expected, action_space={env.ActionSpace}, batched single_action_space={batched}");
            }

            Func = func;
        }

        /// <summary>The function of the batched actions (<c>func</c>).</summary>
        public Func<TWAct, TAct> Func { get; set; }

        /// <summary>Applies <see cref="Func"/> to the batch.</summary>
        /// <param name="actions">The wrapper's batched actions.</param>
        /// <returns>The env's batched actions.</returns>
        public override TAct Actions(TWAct actions) => Func(actions);
    }

    /// <summary>
    ///     Vectorizes a single-env action transform (Gymnasium's <c>VectorizeTransformAction</c>): the wrapper class is built
    ///     over a stand-in env of the single action space, and its function is applied to each sub-env's action.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TWAct">The wrapper's batched action type.</typeparam>
    /// <typeparam name="TAct">The env's batched action type.</typeparam>
    /// <remarks>
    ///     <para>
    ///     When the wrapper's batched space equals the env's (<c>same_out</c>), the results are written into the caller's
    ///     action batch in place, as in Gymnasium; otherwise into the wrapper's buffer, and a deep copy is passed on.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_action.py#L94">gymnasium/wrappers/vector/vectorize_action.py</see>.</para>
    /// </remarks>
    public class VectorizeTransformAction<TObs, TWAct, TAct> : VectorActionWrapper<TObs, TWAct, TAct> {
        /// <summary>The wrapper's buffer (<c>out</c>), of the env's single action space; owned.</summary>
        private readonly object? _out;

        /// <summary>Wraps <paramref name="env"/>, building <paramref name="wrapper"/> over its single action space.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="wrapper">The single-env transform wrapper class (<c>typeof(ClipAction&lt;,&gt;)</c>).</param>
        /// <param name="kwargs">The wrapper's other arguments by Python name, or <see langword="null"/>.</param>
        /// <exception cref="TypeError">Thrown when the class cannot be vectorized or built, or a space does not hold <typeparamref name="TWAct"/>.</exception>
        /// <exception cref="ValueError">Thrown for what the wrapper's constructor refuses.</exception>
        public VectorizeTransformAction(VectorEnv<TObs, TAct> env, Type wrapper, object? kwargs = null) : base(env) {
            Wrapper = Vectorizing.Build(wrapper, new VectorizedSingleEnv(null, ObjectSpace.Unview(env.SingleActionSpace)), kwargs);
            SingleActionSpace = ObjectSpace.Unview(Wrapper.ActionSpace)!.DeepCopy();
            ActionSpace = ObservationWrapperFunctions.Typed<TWAct>(VectorUtils.BatchSpace(SingleActionSpace, NumEnvs));
            SameOut = Equals(ObjectSpace.Unview(ActionSpace), ObjectSpace.Unview(env.ActionSpace));
            _out = VectorUtils.CreateEmptyArray(ObjectSpace.Unview(env.SingleActionSpace)!, NumEnvs);
        }

        /// <summary>The single-env wrapper whose function is applied (<c>wrapper</c>); owned.</summary>
        public IEnv Wrapper { get; }

        /// <summary>Whether the wrapper's batched space equals the env's (<c>same_out</c>).</summary>
        public bool SameOut { get; }

        /// <summary>Each sub-env's action through the wrapper's function, stacked again.</summary>
        /// <param name="actions">The wrapper's batched actions.</param>
        /// <returns>The caller's batch filled in place (<see cref="SameOut"/>), or a deep copy of the wrapper's buffer.</returns>
        public override TAct Actions(TWAct actions) {
            var transform = (IVectorizableTransform)Wrapper;
            Space envSingle = ObjectSpace.Unview(Env.SingleActionSpace)!;
            if (SameOut)
                return (TAct)Vectorizing.Concatenate(transform, ObjectSpace.Unview(ActionSpace)!, envSingle, actions, actions)!;
            object? filled = Vectorizing.Concatenate(transform, ObjectSpace.Unview(ActionSpace)!, envSingle, actions, _out);
            return (TAct)PyCopy.DeepCopy(filled)!;
        }

        /// <summary>Releases the buffer and the single-env wrapper, then disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                Vectorizing.Release(_out);
                Wrapper.Dispose();
            }

            base.Dispose(disposing);
        }
    }

    /// <summary>Clips each sub-env's action into the env's <c>Box</c> (Gymnasium's vector <c>ClipAction</c>).</summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The env's batched action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_action.py#L196">gymnasium/wrappers/vector/vectorize_action.py</see>.</remarks>
    [PyName("ClipAction")]
    public class ClipActionVector<TObs, TAct> : VectorizeTransformAction<TObs, NDArray, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single action space is a <c>Box</c>.</param>
        /// <exception cref="TypeError">Thrown for a space that is not a <c>Box</c>.</exception>
        public ClipActionVector(VectorEnv<TObs, TAct> env) : base(env, typeof(ClipAction<,>)) { }
    }

    /// <summary>Rescales each sub-env's action from new bounds onto the env's <c>Box</c> (Gymnasium's vector <c>RescaleAction</c>).</summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The env's batched action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_action.py#L225">gymnasium/wrappers/vector/vectorize_action.py</see>.</remarks>
    [PyName("RescaleAction")]
    public class RescaleActionVector<TObs, TAct> : VectorizeTransformAction<TObs, NDArray, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env, whose single action space is a <c>Box</c>.</param>
        /// <param name="minAction">The new lower bound: a number or an array.</param>
        /// <param name="maxAction">The new upper bound: a number or an array.</param>
        /// <exception cref="TypeError">Thrown for a space that is not a <c>Box</c>.</exception>
        /// <exception cref="InvalidBound">Thrown for equal bounds.</exception>
        /// <exception cref="ValueError">Thrown for bounds <c>rescale_box</c> refuses.</exception>
        public RescaleActionVector(VectorEnv<TObs, TAct> env, object minAction, object maxAction)
            : base(env, typeof(RescaleAction<,>), new Info { ["min_action"] = minAction, ["max_action"] = maxAction }) { }
    }

    /// <summary>Applies one function to the whole batch of rewards (Gymnasium's vector <c>TransformReward</c>).</summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_reward.py#L21">gymnasium/wrappers/vector/vectorize_reward.py</see>.</remarks>
    [PyName("TransformReward")]
    public class TransformRewardVector<TObs, TAct> : VectorRewardWrapper<TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="func">The function of the float64 reward array.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public TransformRewardVector(VectorEnv<TObs, TAct> env, Func<NDArray, NDArray> func) : base(env) => Func = func;

        /// <summary>The function of the batched rewards (<c>func</c>).</summary>
        public Func<NDArray, NDArray> Func { get; set; }

        /// <summary>Applies <see cref="Func"/> to the batch.</summary>
        /// <param name="rewards">The env's rewards.</param>
        /// <returns>The function's result.</returns>
        public override NDArray Rewards(NDArray rewards) => Func(rewards);
    }

    /// <summary>
    ///     Vectorizes a single-env reward transform (Gymnasium's <c>VectorizeTransformReward</c>): the wrapper class is built
    ///     over a bare stand-in env, and its function is applied to each sub-env's reward, in place.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>The rewards are a float64 array, updated in place: <c>rewards[i] = wrapper.func(rewards[i])</c>.</para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_reward.py#L57">gymnasium/wrappers/vector/vectorize_reward.py</see>.</para>
    /// </remarks>
    public class VectorizeTransformReward<TObs, TAct> : VectorRewardWrapper<TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>, building <paramref name="wrapper"/> over a bare stand-in env.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="wrapper">The single-env transform wrapper class (<c>typeof(ClipReward&lt;,&gt;)</c>).</param>
        /// <param name="kwargs">The wrapper's other arguments by Python name, or <see langword="null"/>.</param>
        /// <exception cref="TypeError">Thrown when the class cannot be vectorized or built.</exception>
        /// <exception cref="InvalidBound">Thrown for what the wrapper's constructor refuses.</exception>
        public VectorizeTransformReward(VectorEnv<TObs, TAct> env, Type wrapper, object? kwargs = null) : base(env) {
            Wrapper = Vectorizing.Build(wrapper, new VectorizedSingleEnv(null, null), kwargs);
        }

        /// <summary>The single-env wrapper whose function is applied (<c>wrapper</c>); owned.</summary>
        public IEnv Wrapper { get; }

        /// <summary>Each reward through the wrapper's function, written back into the array.</summary>
        /// <param name="rewards">The env's float64 rewards; updated in place.</param>
        /// <returns><paramref name="rewards"/>.</returns>
        public override NDArray Rewards(NDArray rewards) {
            var transform = (IVectorizableTransform)Wrapper;
            for (int i = 0; i < rewards.size; i++) {
                double reward = rewards.GetDouble(i);
                rewards.SetDouble((double)transform.ApplyFunc(reward)!, i);
            }

            return rewards;
        }

        /// <summary>Releases the single-env wrapper, then disposes as every vector wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="IDisposable.Dispose"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing)
                Wrapper.Dispose();
            base.Dispose(disposing);
        }
    }

    /// <summary>Clips each sub-env's reward between bounds (Gymnasium's vector <c>ClipReward</c>).</summary>
    /// <typeparam name="TObs">The batched observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The batched action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/vector/vectorize_reward.py#L104">gymnasium/wrappers/vector/vectorize_reward.py</see>.</remarks>
    [PyName("ClipReward")]
    public class ClipRewardVector<TObs, TAct> : VectorizeTransformReward<TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env.</param>
        /// <param name="minReward">The lower bound, or <see langword="null"/>.</param>
        /// <param name="maxReward">The upper bound, or <see langword="null"/>.</param>
        /// <exception cref="InvalidBound">Thrown for no bounds or reversed ones, with Gymnasium's messages.</exception>
        public ClipRewardVector(VectorEnv<TObs, TAct> env, object? minReward = null, object? maxReward = null)
            : base(env, typeof(ClipReward<,>), new Info { ["min_reward"] = minReward, ["max_reward"] = maxReward }) { }
    }
}

