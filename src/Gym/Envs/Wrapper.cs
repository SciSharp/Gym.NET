#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     A wrapper of any observation and action types, Python's <c>isinstance(env, gymnasium.Wrapper)</c>: lets code walk
    ///     a stack of wrappers (<see cref="Env"/> down to <see cref="IEnv.Unwrapped"/>) without knowing each layer's types.
    /// </summary>
    public interface IWrapper : IEnv {
        /// <summary>The env this wrapper wraps (<c>env</c>), possibly another wrapper.</summary>
        IEnv Env { get; }

        /// <summary>The wrapper's class name (<c>class_name()</c>), as its <c>WrapperSpec</c> records it.</summary>
        /// <returns>The name, without generic arity: <c>TimeLimit</c>.</returns>
        string ClassName();
    }

    /// <summary>
    ///     Wraps an env to change parts of its behaviour while forwarding the rest. Every member forwards to
    ///     <see cref="Env"/> unless the wrapper overrides it; spaces and metadata can be replaced on the wrapper alone, without
    ///     touching the env beneath.
    /// </summary>
    /// <typeparam name="TWObs">The observation type the wrapper returns.</typeparam>
    /// <typeparam name="TWAct">The action type the wrapper takes.</typeparam>
    /// <typeparam name="TObs">The wrapped env's observation type.</typeparam>
    /// <typeparam name="TAct">The wrapped env's action type.</typeparam>
    /// <remarks>
    ///     <para>
    ///     <see cref="Spec"/>, <see cref="RenderMode"/>, <see cref="NpRandom"/> and <see cref="NpRandomSeed"/> always read the
    ///     env beneath; <see cref="Spec"/> extends the env's spec with this wrapper's <see cref="WrapperSpec"/> (recording its
    ///     constructor arguments when it implements <see cref="IRecordConstructorArgs"/>) and caches it. Python attributes
    ///     are not forwarded: <c>wrapper.gravity</c> raises, and <see cref="GetWrapperAttr"/> is the way to reach an
    ///     attribute of a lower layer.
    ///     </para>
    ///     <para>
    ///     When the wrapper's types differ from the env's, override <see cref="Step"/> and <see cref="Reset"/> (or derive from
    ///     <see cref="ObservationWrapper{TWObs, TObs, TAct}"/> or <see cref="ActionWrapper{TObs, TWAct, TAct}"/>): the default
    ///     forwarding passes values through unchanged and throws <see cref="InvalidCastException"/> when a value is not also of
    ///     the other type.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L290">gymnasium/core.py</see>.</para>
    /// </remarks>
    public class Wrapper<TWObs, TWAct, TObs, TAct> : Env<TWObs, TWAct>, IWrapper {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env to wrap, possibly another wrapper.</param>
        /// <exception cref="TypeError">
        ///     Thrown when <paramref name="env"/> is <see langword="null"/>, with Gymnasium's message
        ///     (<c>Expected env to be a `gymnasium.Env` but got &lt;class 'NoneType'&gt;</c>).
        /// </exception>
        public Wrapper(Env<TObs, TAct> env) {
            Env = env ?? throw new TypeError($"Expected env to be a `gymnasium.Env` but got {PyBuiltins.TypeRepr(null)}");
        }

        /// <summary>The wrapped env (<c>env</c>).</summary>
        public Env<TObs, TAct> Env { get; }

        /// <inheritdoc />
        IEnv IWrapper.Env => Env;

        /// <summary>
        ///     The env's metadata, unless this wrapper was assigned its own (Python's <c>_metadata</c>); assigning
        ///     <see langword="null"/> restores forwarding, as setting <c>_metadata = None</c> does.
        /// </summary>
        public override Metadata Metadata {
            get => WrapperMetadata ?? Env.Metadata;
            set => WrapperMetadata = value;
        }

        /// <summary>The env's render mode; read-only on a wrapper (Python's property has no setter).</summary>
        /// <exception cref="AttributeError">Thrown when assigned (through <see cref="PyAttributes.SetAttr"/>).</exception>
        public override string? RenderMode {
            get => Env.RenderMode;
            protected set => throw new AttributeError($"property 'render_mode' of '{PyAttributes.PyTypeName(GetType())}' object has no setter");
        }

        /// <summary>
        ///     The env's spec extended with this wrapper (Python's <c>spec</c> property): a deep copy of the env's spec whose
        ///     <see cref="EnvSpec.AdditionalWrappers"/> ends with this wrapper's <see cref="WrapperSpec"/>, or
        ///     <see langword="null"/> when the env has no spec. Computed once, then cached (<c>_cached_spec</c>).
        /// </summary>
        /// <exception cref="AttributeError">Thrown when assigned: <c>property 'spec' of 'X' object has no setter</c>.</exception>
        public override EnvSpec? Spec {
            get {
                if (CachedSpec is not null)
                    return CachedSpec;

                EnvSpec? envSpec = Env.Spec;
                if (envSpec is not null) {
                    // See if the wrapper inherits from RecordConstructorArgs, then add the kwargs; otherwise None, which make
                    // refuses to re-create.
                    Info? kwargs;
                    if (this is IRecordConstructorArgs recorded) {
                        kwargs = recorded.SavedKwargs;
                        if (kwargs.ContainsKey("env")) {
                            kwargs = kwargs.DeepCopy();
                            kwargs.Remove("env");
                        }
                    } else {
                        kwargs = null;
                    }

                    string typeName = PyAttributes.PyTypeName(GetType());
                    var wrapperSpec = new WrapperSpec(ClassName(), $"{GetType().Namespace}:{typeName}", kwargs);

                    // To avoid reference issues, deep-copy the prior env's spec and add the new information.
                    try {
                        envSpec = envSpec.DeepCopy();
                        var wrappers = new List<WrapperSpec>(envSpec.AdditionalWrappers) { wrapperSpec };
                        envSpec.AdditionalWrappers = wrappers.ToArray();
                    } catch (Exception e) {
                        Logger.Warn($"An exception occurred ({e.Message}) while copying the environment spec={envSpec}");
                        return null;
                    }
                }

                CachedSpec = envSpec;
                return envSpec;
            }
            set => throw new AttributeError($"property 'spec' of '{PyAttributes.PyTypeName(GetType())}' object has no setter");
        }

        /// <summary>The env's action space, unless this wrapper was assigned its own (Python's <c>_action_space</c>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter when neither this wrapper nor the env has one.</exception>
        public override Space ActionSpace {
            get => WrapperActionSpace ?? Env.ActionSpace;
            set => WrapperActionSpace = value;
        }

        /// <summary>The env's observation space, unless this wrapper was assigned its own (Python's <c>_observation_space</c>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter when neither this wrapper nor the env has one.</exception>
        public override Space ObservationSpace {
            get => WrapperObservationSpace ?? Env.ObservationSpace;
            set => WrapperObservationSpace = value;
        }

        /// <summary>The env's generator; assigning one assigns the env's (so the innermost env's seed becomes -1).</summary>
        public override Generator NpRandom {
            get => Env.NpRandom;
            set => Env.NpRandom = value;
        }

        /// <summary>The env's <see cref="IEnv.NpRandomSeed"/>, which a seeded reset through the wrapper sets on the innermost env.</summary>
        public override BigInteger NpRandomSeed => Env.NpRandomSeed;

        /// <summary>The innermost env, beneath every layer.</summary>
        public override IEnv Unwrapped => Env.Unwrapped;

        /// <summary>
        ///     Python's <c>_np_random</c> on a wrapper: an error, because a wrapper holds no generator of its own; read
        ///     <c>Unwrapped</c>'s or <see cref="NpRandom"/> instead.
        /// </summary>
        /// <exception cref="AttributeError">
        ///     Thrown by the getter with Gymnasium's message (<c>Can't access `_np_random` of a wrapper, use
        ///     `.unwrapped._np_random` or `.np_random`.</c>) and by the setter as by a setter-less Python property.
        /// </exception>
        [PyName("_np_random")]
        protected override Generator? RawNpRandom {
            get => throw new AttributeError("Can't access `_np_random` of a wrapper, use `.unwrapped._np_random` or `.np_random`.");
            set => throw new AttributeError($"property '_np_random' of '{PyAttributes.PyTypeName(GetType())}' object has no setter");
        }

        /// <summary>This wrapper's own action space (Python's <c>_action_space</c>), <see langword="null"/> to forward the env's.</summary>
        [PyName("_action_space")]
        protected Space? WrapperActionSpace { get; set; }

        /// <summary>This wrapper's own observation space (Python's <c>_observation_space</c>), <see langword="null"/> to forward the env's.</summary>
        [PyName("_observation_space")]
        protected Space? WrapperObservationSpace { get; set; }

        /// <summary>This wrapper's own metadata (Python's <c>_metadata</c>), <see langword="null"/> to forward the env's.</summary>
        [PyName("_metadata")]
        protected Metadata? WrapperMetadata { get; set; }

        /// <summary>
        ///     The spec computed by the first read of <see cref="Spec"/> (Python's <c>_cached_spec</c>). Wrappers that override
        ///     <see cref="Spec"/> use it the same way; clear it to recompute after the env's spec changed.
        /// </summary>
        [PyName("_cached_spec")]
        protected EnvSpec? CachedSpec { get; set; }

        /// <summary>Forwards to the env (<c>self.env.step(action)</c>); override to change the transition.</summary>
        /// <param name="action">The action, passed through unchanged.</param>
        /// <returns>The env's transition, its observation passed through unchanged.</returns>
        /// <exception cref="InvalidCastException">Thrown when the wrapper's and the env's types differ and a value is not of both.</exception>
        public override StepResult<TWObs> Step(TWAct action) {
            StepResult<TObs> result = Env.Step(PassThrough<TWAct, TAct>(action, "action"));
            return new StepResult<TWObs>(PassThrough<TObs, TWObs>(result.Observation, "observation"), result.Reward, result.Terminated, result.Truncated, result.Info);
        }

        /// <summary>Forwards to the env (<c>self.env.reset(seed=seed, options=options)</c>); override to change the result.</summary>
        /// <param name="seed">The seed, passed on to the env.</param>
        /// <param name="options">The options, passed on to the env.</param>
        /// <returns>The env's first observation, passed through unchanged, and info.</returns>
        /// <exception cref="Error">Thrown for a negative seed (by the innermost env).</exception>
        /// <exception cref="InvalidCastException">Thrown when the wrapper's and the env's observation types differ and the observation is not of both.</exception>
        public override ResetResult<TWObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> result = Env.Reset(seed, options);
            return new ResetResult<TWObs>(PassThrough<TObs, TWObs>(result.Observation, "observation"), result.Info);
        }

        /// <summary>Forwards to the env (<c>self.env.render()</c>).</summary>
        /// <returns>The env's rendering.</returns>
        public override RenderResult Render() => Env.Render();

        /// <summary>Closes the env beneath (<c>self.env.close()</c>).</summary>
        public override void Close() => Env.Close();

        /// <summary>The wrapper's class name (<c>class_name()</c>): the type name without generic arity, <c>TimeLimit</c>.</summary>
        /// <returns>The name its <see cref="WrapperSpec"/> records.</returns>
        public virtual string ClassName() => PyAttributes.PyTypeName(GetType());

        /// <summary>Whether this wrapper or any layer beneath it has the Python attribute <paramref name="name"/>.</summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns><see langword="true"/> when some layer has it.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L408">gymnasium/core.py</see>.</remarks>
        public override bool HasWrapperAttr(string name) {
            if (PyAttributes.HasAttr(this, name))
                return true;
            return Env.HasWrapperAttr(name);
        }

        /// <summary>
        ///     The Python attribute <paramref name="name"/> from this wrapper when it has it, else from the first layer beneath
        ///     that does.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns>The value, boxed.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">
        ///     Thrown when no layer has it: <c>wrapper TimeLimit has no attribute 'x'</c>. Python chains the lower layer's error
        ///     as the cause (<c>raise ... from e</c>); NumSharp's <see cref="AttributeError"/> has no constructor taking an
        ///     inner exception, so the chain is not kept.
        /// </exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L415">gymnasium/core.py</see>.</remarks>
        public override object? GetWrapperAttr(string name) {
            if (PyAttributes.HasAttr(this, name))
                return PyAttributes.GetAttr(this, name);
            try {
                return Env.GetWrapperAttr(name);
            } catch (AttributeError) {
                throw new AttributeError($"wrapper {ClassName()} has no attribute {PyBuiltins.Repr(name)}");
            }
        }

        /// <summary>
        ///     Assigns the Python attribute <paramref name="name"/> on this wrapper when it has it, else on the first layer
        ///     beneath that has it (asking them with <c>force=False</c>); when no layer has it and <paramref name="force"/> is
        ///     set, creates it on this wrapper.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <param name="value">The value.</param>
        /// <param name="force">Create the attribute on this wrapper when no layer has it.</param>
        /// <returns><see langword="true"/> when some layer was assigned.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">Thrown when the attribute found is read-only.</exception>
        /// <exception cref="TypeError">Thrown when the value does not fit the attribute's type.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L434">gymnasium/core.py</see>.</remarks>
        public override bool SetWrapperAttr(string name, object? value, bool force = true) {
            if (PyAttributes.HasAttr(this, name)) {
                PyAttributes.SetAttr(this, name, value);
                return true;
            }

            bool alreadySet = Env.SetWrapperAttr(name, value, force: false);
            if (alreadySet)
                return true;
            if (force) {
                PyAttributes.SetAttr(this, name, value);
                return true;
            }

            return false;
        }

        /// <summary>The stack's description: <c>&lt;TimeLimit&lt;OrderEnforcing&lt;CartPoleEnv&lt;CartPole-v1&gt;&gt;&gt;&gt;</c>.</summary>
        /// <returns>The wrapper's class name followed by the env's text.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L459">gymnasium/core.py</see>.</remarks>
        public override string ToString() => $"<{PyAttributes.PyTypeName(GetType())}{Env}>";

        /// <summary>
        ///     Passes a value between the wrapper's and the env's type parameters, which are the same type in every wrapper
        ///     that does not convert; the JIT removes the boxing then.
        /// </summary>
        /// <typeparam name="TFrom">The type the value has.</typeparam>
        /// <typeparam name="TTo">The type it must have.</typeparam>
        /// <param name="value">The value.</param>
        /// <param name="what">What the value is, for the error message.</param>
        /// <returns>The same value as <typeparamref name="TTo"/>.</returns>
        /// <exception cref="InvalidCastException">Thrown when the value is not a <typeparamref name="TTo"/>.</exception>
        private TTo PassThrough<TFrom, TTo>(TFrom value, string what) {
            if (value is TTo same)
                return same;
            if (value is null && default(TTo) is null)
                return default!;
            throw new InvalidCastException($"{ClassName()} passes the {what} through unchanged, but {PyBuiltins.Repr(value)} is not a {typeof(TTo).Name}; override Step and Reset to convert between {typeof(TFrom).Name} and {typeof(TTo).Name}.");
        }
    }

    /// <summary>
    ///     A wrapper that keeps its env's observation and action types (<c>Wrapper[ObsType, ActType, ObsType, ActType]</c>),
    ///     the base of wrappers such as <c>TimeLimit</c> and <c>OrderEnforcing</c>.
    /// </summary>
    /// <typeparam name="TObs">The observation type of the env and the wrapper.</typeparam>
    /// <typeparam name="TAct">The action type of the env and the wrapper.</typeparam>
    public class Wrapper<TObs, TAct> : Wrapper<TObs, TAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env to wrap.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public Wrapper(Env<TObs, TAct> env) : base(env) { }
    }

    /// <summary>
    ///     Transforms every observation <see cref="Reset"/> and <see cref="Step"/> return through <see cref="Observation"/>.
    ///     If the transformation changes the observation space, assign the new one to
    ///     <see cref="Env{TObs, TAct}.ObservationSpace"/> in the constructor.
    /// </summary>
    /// <typeparam name="TWObs">The transformed observation type.</typeparam>
    /// <typeparam name="TObs">The env's observation type.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L534">gymnasium/core.py</see>.</remarks>
    public abstract class ObservationWrapper<TWObs, TObs, TAct> : Wrapper<TWObs, TAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env whose observations are transformed.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected ObservationWrapper(Env<TObs, TAct> env) : base(env) { }

        /// <summary>Resets the env and returns the transformed first observation with the env's info.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns><c>(self.observation(obs), info)</c>.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TWObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            (TObs obs, Info info) = Env.Reset(seed, options);
            return new ResetResult<TWObs>(Observation(obs), info);
        }

        /// <summary>Steps the env and returns its transition with the observation transformed.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns><c>(self.observation(observation), reward, terminated, truncated, info)</c>.</returns>
        public override StepResult<TWObs> Step(TAct action) {
            (TObs observation, double reward, bool terminated, bool truncated, Info info) = Env.Step(action);
            return new StepResult<TWObs>(Observation(observation), reward, terminated, truncated, info);
        }

        /// <summary>The transformation (<c>observation(observation)</c>).</summary>
        /// <param name="observation">The env's observation.</param>
        /// <returns>The transformed observation.</returns>
        public abstract TWObs Observation(TObs observation);
    }

    /// <summary>Transforms every reward <see cref="Step"/> returns through <see cref="Reward"/>.</summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L578">gymnasium/core.py</see>.</remarks>
    public abstract class RewardWrapper<TObs, TAct> : Wrapper<TObs, TAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env whose rewards are transformed.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected RewardWrapper(Env<TObs, TAct> env) : base(env) { }

        /// <summary>Steps the env and returns its transition with the reward transformed.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns><c>(observation, self.reward(reward), terminated, truncated, info)</c>.</returns>
        public override StepResult<TObs> Step(TAct action) {
            (TObs observation, double reward, bool terminated, bool truncated, Info info) = Env.Step(action);
            return new StepResult<TObs>(observation, Reward(reward), terminated, truncated, info);
        }

        /// <summary>The transformation (<c>reward(reward)</c>).</summary>
        /// <param name="reward">The env's reward.</param>
        /// <returns>The transformed reward.</returns>
        public abstract double Reward(double reward);
    }

    /// <summary>
    ///     Transforms every action through <see cref="Action"/> before the env sees it. If the wrapper takes actions from
    ///     another space, assign it to <see cref="Env{TObs, TAct}.ActionSpace"/> in the constructor.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TWAct">The action type the wrapper takes.</typeparam>
    /// <typeparam name="TAct">The env's action type.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L613">gymnasium/core.py</see>.</remarks>
    public abstract class ActionWrapper<TObs, TWAct, TAct> : Wrapper<TObs, TWAct, TObs, TAct> {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env whose actions are transformed.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        protected ActionWrapper(Env<TObs, TAct> env) : base(env) { }

        /// <summary>Steps the env with the transformed action (<c>self.env.step(self.action(action))</c>).</summary>
        /// <param name="action">The wrapper's action.</param>
        /// <returns>The env's transition.</returns>
        public override StepResult<TObs> Step(TWAct action) => Env.Step(Action(action));

        /// <summary>The transformation (<c>action(action)</c>).</summary>
        /// <param name="action">The wrapper's action.</param>
        /// <returns>The action for the env, an element of the env's action space.</returns>
        public abstract TAct Action(TWAct action);
    }
}
