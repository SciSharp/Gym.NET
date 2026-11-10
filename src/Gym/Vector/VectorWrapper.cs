#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     One layer of a vector-wrapper stack seen without its generic types, so code inside Gym.NET can walk down a stack
    ///     whose layers are typed differently (a single-env stack has the public <see cref="IWrapper"/> for this).
    /// </summary>
    internal interface IVectorWrapperLayer {
        /// <summary>The vector env this layer wraps, possibly another wrapper.</summary>
        IVectorEnv InnerEnv { get; }
    }

    /// <summary>
    ///     Wraps a vector env to change its behaviour, forwarding everything it does not change: the spaces (until the wrapper
    ///     assigns its own), the metadata, the spec, the generator, <see cref="VectorEnv{TObs, TAct}.NumEnvs"/>, reset, step,
    ///     render and close. Typed by the wrapper's batched types and the wrapped env's.
    /// </summary>
    /// <typeparam name="TWObs">The wrapper's batched observation type.</typeparam>
    /// <typeparam name="TWAct">The wrapper's batched action type.</typeparam>
    /// <typeparam name="TObs">The wrapped env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The wrapped env's batched action type.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Unlike a single-env wrapper, a vector wrapper adds nothing to the spec: <see cref="Spec"/> is the wrapped env's.
    ///     When the wrapper's types differ from the env's, override <see cref="Step"/> and <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/>
    ///     (or derive from <see cref="VectorObservationWrapper{TWObs, TObs, TAct}"/>, <see cref="VectorActionWrapper{TObs, TWAct, TAct}"/>):
    ///     the default forwarding passes values through and throws <see cref="InvalidCastException"/> for a value of only one type.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L361">gymnasium/vector/vector_env.py</see>.</para>
    /// </remarks>
    public class VectorWrapper<TWObs, TWAct, TObs, TAct> : VectorEnv<TWObs, TWAct>, IVectorWrapperLayer {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env to wrap, possibly another wrapper.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>: <c>Expected env to be a `gymnasium.vector.VectorEnv` but got &lt;class 'NoneType'&gt;</c>.</exception>
        public VectorWrapper(VectorEnv<TObs, TAct> env) {
            Env = env ?? throw new TypeError($"Expected env to be a `gymnasium.vector.VectorEnv` but got {PyBuiltins.TypeRepr(null)}");
        }

        /// <summary>The wrapped vector env (<c>env</c>).</summary>
        public VectorEnv<TObs, TAct> Env { get; }

        /// <summary>The env's metadata, unless this wrapper was assigned its own (Python's <c>_metadata</c>); assigning <see langword="null"/> restores forwarding.</summary>
        public override Metadata Metadata {
            get => WrapperMetadata ?? Env.Metadata;
            set => WrapperMetadata = value;
        }

        /// <summary>The env's spec: a vector wrapper records nothing in it.</summary>
        /// <exception cref="AttributeError">Thrown when assigned, as Python's property has no setter.</exception>
        public override EnvSpec? Spec {
            get => Env.Spec;
            set => throw new AttributeError($"property 'spec' of '{PyAttributes.PyTypeName(GetType())}' object has no setter");
        }

        /// <summary>The env's render mode; read-only on a wrapper.</summary>
        /// <exception cref="AttributeError">Thrown when assigned.</exception>
        public override string? RenderMode {
            get => Env.RenderMode;
            protected set => throw new AttributeError($"property 'render_mode' of '{PyAttributes.PyTypeName(GetType())}' object has no setter");
        }

        /// <summary>The env's closed flag; setting it sets the env's (Python's property forwards both ways).</summary>
        public override bool Closed {
            get => Env.Closed;
            set => Env.Closed = value;
        }

        /// <summary>The env's number of sub-envs.</summary>
        /// <exception cref="AttributeError">Thrown when assigned, as Python's property has no setter.</exception>
        public override int NumEnvs {
            get => Env.NumEnvs;
            protected set => throw new AttributeError($"property 'num_envs' of '{PyAttributes.PyTypeName(GetType())}' object has no setter");
        }

        /// <summary>The env's batched observation space, unless this wrapper was assigned its own (<c>_observation_space</c>).</summary>
        /// <exception cref="TypeError">Thrown by the getter when the env's space holds another element type and this wrapper has none of its own.</exception>
        public override Space<TWObs> ObservationSpace {
            get => WrapperObservationSpace ?? Env.ObservationSpace as Space<TWObs> ?? throw ForwardedSpaceMismatch("observation_space", typeof(TWObs));
            set => WrapperObservationSpace = value;
        }

        /// <summary>The env's batched action space, unless this wrapper was assigned its own (<c>_action_space</c>).</summary>
        /// <exception cref="TypeError">Thrown by the getter when the env's space holds another element type and this wrapper has none of its own.</exception>
        public override Space<TWAct> ActionSpace {
            get => WrapperActionSpace ?? Env.ActionSpace as Space<TWAct> ?? throw ForwardedSpaceMismatch("action_space", typeof(TWAct));
            set => WrapperActionSpace = value;
        }

        /// <summary>The env's single observation space, unless this wrapper was assigned its own (<c>_single_observation_space</c>).</summary>
        public override Space SingleObservationSpace {
            get => WrapperSingleObservationSpace ?? Env.SingleObservationSpace;
            set => WrapperSingleObservationSpace = value;
        }

        /// <summary>The env's single action space, unless this wrapper was assigned its own (<c>_single_action_space</c>).</summary>
        public override Space SingleActionSpace {
            get => WrapperSingleActionSpace ?? Env.SingleActionSpace;
            set => WrapperSingleActionSpace = value;
        }

        /// <summary>The env's generator; assigning one assigns the env's.</summary>
        public override Generator NpRandom {
            get => Env.NpRandom;
            set => Env.NpRandom = value;
        }

        /// <summary>The env's generator seed.</summary>
        public override BigInteger NpRandomSeed => Env.NpRandomSeed;

        /// <summary>The innermost vector env.</summary>
        public override IVectorEnv Unwrapped => Env.Unwrapped;

        /// <summary>The backing slot of <see cref="WrapperObservationSpace"/>.</summary>
        private Space<TWObs>? _wrapperObservationSpace;

        /// <summary>The backing slot of <see cref="WrapperActionSpace"/>.</summary>
        private Space<TWAct>? _wrapperActionSpace;

        /// <summary>The vector env this wrapper wraps, untyped, so <see cref="OwnedBeneath"/> can walk a stack of differently typed layers.</summary>
        IVectorEnv IVectorWrapperLayer.InnerEnv => Env;

        /// <summary>
        ///     This wrapper's own batched observation space, <see langword="null"/> to forward the env's. The wrapper owns and
        ///     disposes a space it is given (DR-40), except one a vector env beneath it already owns, which it only references.
        /// </summary>
        /// <exception cref="InvalidOperationException">Thrown by the setter for a space an env outside this stack, not yet disposed, owns.</exception>
        [PyName("_observation_space")]
        protected Space<TWObs>? WrapperObservationSpace {
            get => _wrapperObservationSpace;
            set => _wrapperObservationSpace = Space.Assign(this, value, _wrapperObservationSpace, _wrapperActionSpace, "observation_space", OwnedBeneath);
        }

        /// <summary>This wrapper's own batched action space, <see langword="null"/> to forward the env's; owned as <see cref="WrapperObservationSpace"/> is.</summary>
        /// <exception cref="InvalidOperationException">Thrown by the setter for a space an env outside this stack, not yet disposed, owns.</exception>
        [PyName("_action_space")]
        protected Space<TWAct>? WrapperActionSpace {
            get => _wrapperActionSpace;
            set => _wrapperActionSpace = Space.Assign(this, value, _wrapperActionSpace, _wrapperObservationSpace, "action_space", OwnedBeneath);
        }

        /// <summary>This wrapper's own single observation space, <see langword="null"/> to forward the env's.</summary>
        [PyName("_single_observation_space")]
        protected Space? WrapperSingleObservationSpace { get; set; }

        /// <summary>This wrapper's own single action space, <see langword="null"/> to forward the env's.</summary>
        [PyName("_single_action_space")]
        protected Space? WrapperSingleActionSpace { get; set; }

        /// <summary>This wrapper's own metadata, <see langword="null"/> to forward the env's.</summary>
        [PyName("_metadata")]
        protected Metadata? WrapperMetadata { get; set; }

        /// <summary>Forwards to the env (<c>self.env.reset(seed=seed, options=options)</c>); valid only when the wrapper does not change the observation type.</summary>
        /// <param name="seed">The seed.</param>
        /// <param name="options">The options.</param>
        /// <returns>The env's result, its observations passed through.</returns>
        /// <exception cref="InvalidCastException">Thrown when the observation types differ and the observations are not of both.</exception>
        public override VectorResetResult<TWObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = Env.Reset(seed, options);
            return new VectorResetResult<TWObs>(PassThrough<TObs, TWObs>(result.Observations, "observations"), result.Infos);
        }

        /// <summary>Forwards a reset with one seed per sub-env to the env.</summary>
        /// <param name="seeds">The seeds.</param>
        /// <param name="options">The options.</param>
        /// <returns>The env's result, its observations passed through.</returns>
        /// <exception cref="InvalidCastException">Thrown when the observation types differ and the observations are not of both.</exception>
        public override VectorResetResult<TWObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = Env.Reset(seeds, options);
            return new VectorResetResult<TWObs>(PassThrough<TObs, TWObs>(result.Observations, "observations"), result.Infos);
        }

        /// <summary>Forwards to the env (<c>self.env.step(actions)</c>); valid only when the wrapper changes neither type.</summary>
        /// <param name="actions">The actions, passed through.</param>
        /// <returns>The env's result, its observations passed through.</returns>
        /// <exception cref="InvalidCastException">Thrown when the types differ and a value is not of both.</exception>
        public override VectorStepResult<TWObs> Step(TWAct actions) {
            VectorStepResult<TObs> result = Env.Step(PassThrough<TWAct, TAct>(actions, "actions"));
            return new VectorStepResult<TWObs>(PassThrough<TObs, TWObs>(result.Observations, "observations"), result.Rewards, result.Terminations, result.Truncations, result.Infos);
        }

        /// <summary>Forwards to the env (<c>self.env.render()</c>).</summary>
        /// <returns>The env's renders.</returns>
        public override IReadOnlyList<RenderResult>? Render() => Env.Render();

        /// <summary>Closes the env beneath (<c>self.env.close()</c>).</summary>
        public override void Close() => Env.Close();

        /// <summary>The env's <c>close_extras</c>.</summary>
        protected override void CloseExtras() => Env.CloseExtrasCore();

        /// <summary>Closes, releases this wrapper's own spaces, and disposes the env beneath: disposing the outermost wrapper disposes the stack.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="VectorEnv{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            Close();
            if (!disposing)
                return;
            // Only what this wrapper owns: a batched space passed through from a vector env beneath is that env's (DR-40).
            if (WrapperObservationSpace is { } observationSpace && observationSpace.IsOwnedBy(this))
                observationSpace.Dispose();
            if (WrapperActionSpace is { } actionSpace && actionSpace.IsOwnedBy(this))
                actionSpace.Dispose();
            Env.Dispose();
        }

        /// <summary>
        ///     Whether a vector env beneath this wrapper (the one it wraps, or any layer below) owns <paramref name="space"/>,
        ///     which makes the space a pass-through this wrapper references without owning (DR-40).
        /// </summary>
        /// <param name="space">The space assigned to this wrapper.</param>
        /// <returns><see langword="true"/> when a lower layer owns it.</returns>
        private bool OwnedBeneath(Space space) {
            for (IVectorEnv? env = Env; env is not null; env = (env as IVectorWrapperLayer)?.InnerEnv) {
                if (space.IsOwnedBy(env))
                    return true;
            }

            return false;
        }

        /// <summary>Python's <c>repr</c>: <c>&lt;ClipReward, SyncVectorEnv(CartPole-v1, num_envs=3)&gt;</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"<{PyAttributes.PyTypeName(GetType())}, {Env}>";

        /// <summary>A value passed through a wrapper whose types are the env's, or an error naming what does not fit.</summary>
        /// <typeparam name="TFrom">The value's declared type.</typeparam>
        /// <typeparam name="TTo">The type it is passed on as.</typeparam>
        /// <param name="value">The value.</param>
        /// <param name="what">What the value is, for the error.</param>
        /// <returns>The value as <typeparamref name="TTo"/>.</returns>
        /// <exception cref="InvalidCastException">Thrown when the value is not a <typeparamref name="TTo"/>.</exception>
        private TTo PassThrough<TFrom, TTo>(TFrom value, string what) {
            if (value is TTo same)
                return same;
            if (value is null && default(TTo) is null)
                return default!;
            throw new InvalidCastException($"{PyAttributes.PyTypeName(GetType())} passes {what} through unchanged, but {typeof(TFrom).Name} is not {typeof(TTo).Name}; override the method that converts them.");
        }

        /// <summary>The error for forwarding an env space whose elements are not this wrapper's type.</summary>
        /// <param name="attribute"><c>action_space</c> or <c>observation_space</c>.</param>
        /// <param name="expected">The wrapper's element type.</param>
        /// <returns>The error to throw.</returns>
        private TypeError ForwardedSpaceMismatch(string attribute, Type expected) =>
            new($"{PyAttributes.PyTypeName(GetType())} forwards the env's {attribute}, whose elements are not {expected.Name}; assign the wrapper its own {attribute}.");
    }

    /// <summary>A vector wrapper of the env's own types (<c>VectorWrapper[ObsType, ActType]</c>), the base of wrappers that change neither.</summary>
    /// <typeparam name="TObs">The batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type.</typeparam>
    public class VectorWrapper<TObs, TAct> : VectorWrapper<TObs, TAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env to wrap.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public VectorWrapper(VectorEnv<TObs, TAct> env) : base(env) { }
    }

    /// <summary>
    ///     A vector wrapper that changes only the observations, through <see cref="Observations"/>, on reset and on every
    ///     step. It refuses a <c>SAME_STEP</c> env: there the batch mixes new and final observations, which a transform of
    ///     the batch cannot tell apart.
    /// </summary>
    /// <typeparam name="TWObs">The wrapper's batched observation type.</typeparam>
    /// <typeparam name="TObs">The env's batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L543">gymnasium/vector/vector_env.py</see>.</remarks>
    public abstract class VectorObservationWrapper<TWObs, TObs, TAct> : VectorWrapper<TWObs, TAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>, checking its autoreset mode.</summary>
        /// <param name="env">The vector env to wrap.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when the env's <c>autoreset_mode</c> is neither <c>NEXT_STEP</c> nor <c>DISABLED</c>, with Gymnasium's message.</exception>
        protected VectorObservationWrapper(VectorEnv<TObs, TAct> env) : base(env) {
            if (!env.Metadata.TryGetValue("autoreset_mode", out object? mode)) {
                Logger.Warn($"Vector environment ({env}) is missing `autoreset_mode` metadata key.");
            } else if (!ReferenceEquals(mode, AutoresetMode.NextStep) && !ReferenceEquals(mode, AutoresetMode.Disabled)) {
                // An f-string shows the member's str, AutoresetMode.SAME_STEP.
                throw new ValueError($"Expected autoreset_mode to be NEXT_STEP or DISABLED, got {(mode is AutoresetMode member ? member.Str : PyBuiltins.Str(mode))}");
            }
        }

        /// <summary>Resets the env and transforms its observations.</summary>
        /// <param name="seed">The seed.</param>
        /// <param name="options">The options.</param>
        /// <returns>The transformed observations and the env's infos.</returns>
        public override VectorResetResult<TWObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = Env.Reset(seed, options);
            return new VectorResetResult<TWObs>(Observations(result.Observations), result.Infos);
        }

        /// <summary>Resets the env with one seed per sub-env and transforms its observations.</summary>
        /// <param name="seeds">The seeds.</param>
        /// <param name="options">The options.</param>
        /// <returns>The transformed observations and the env's infos.</returns>
        public override VectorResetResult<TWObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            VectorResetResult<TObs> result = Env.Reset(seeds, options);
            return new VectorResetResult<TWObs>(Observations(result.Observations), result.Infos);
        }

        /// <summary>Steps the env and transforms its observations; the rewards, flags and infos pass through.</summary>
        /// <param name="actions">The actions.</param>
        /// <returns>The transition with transformed observations.</returns>
        public override VectorStepResult<TWObs> Step(TAct actions) {
            VectorStepResult<TObs> result = Env.Step(actions);
            return new VectorStepResult<TWObs>(Observations(result.Observations), result.Rewards, result.Terminations, result.Truncations, result.Infos);
        }

        /// <summary>The transform (<c>observations(observations)</c>).</summary>
        /// <param name="observations">The env's batched observations.</param>
        /// <returns>The wrapper's batched observations.</returns>
        public abstract TWObs Observations(TObs observations);
    }

    /// <summary>A vector wrapper that changes only the actions, through <see cref="Actions"/>, before every step.</summary>
    /// <typeparam name="TObs">The batched observation type.</typeparam>
    /// <typeparam name="TWAct">The wrapper's batched action type.</typeparam>
    /// <typeparam name="TAct">The env's batched action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L603">gymnasium/vector/vector_env.py</see>.</remarks>
    public abstract class VectorActionWrapper<TObs, TWAct, TAct> : VectorWrapper<TObs, TWAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env to wrap.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected VectorActionWrapper(VectorEnv<TObs, TAct> env) : base(env) { }

        /// <summary>Transforms the actions and steps the env with them.</summary>
        /// <param name="actions">The wrapper's batched actions.</param>
        /// <returns>The env's transition.</returns>
        public override VectorStepResult<TObs> Step(TWAct actions) => Env.Step(Actions(actions));

        /// <summary>The transform (<c>actions(actions)</c>).</summary>
        /// <param name="actions">The wrapper's batched actions.</param>
        /// <returns>The env's batched actions.</returns>
        public abstract TAct Actions(TWAct actions);
    }

    /// <summary>A vector wrapper that changes only the rewards, through <see cref="Rewards"/>, after every step.</summary>
    /// <typeparam name="TObs">The batched observation type.</typeparam>
    /// <typeparam name="TAct">The batched action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L637">gymnasium/vector/vector_env.py</see>.</remarks>
    public abstract class VectorRewardWrapper<TObs, TAct> : VectorWrapper<TObs, TAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The vector env to wrap.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected VectorRewardWrapper(VectorEnv<TObs, TAct> env) : base(env) { }

        /// <summary>Steps the env and transforms its rewards.</summary>
        /// <param name="actions">The actions.</param>
        /// <returns>The transition with transformed rewards.</returns>
        public override VectorStepResult<TObs> Step(TAct actions) {
            VectorStepResult<TObs> result = Env.Step(actions);
            return new VectorStepResult<TObs>(result.Observations, Rewards(result.Rewards), result.Terminations, result.Truncations, result.Infos);
        }

        /// <summary>The transform (<c>rewards(rewards)</c>).</summary>
        /// <param name="rewards">The env's <c>(num_envs,)</c> rewards.</param>
        /// <returns>The wrapper's rewards.</returns>
        public abstract NDArray Rewards(NDArray rewards);
    }
}
