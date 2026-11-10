#nullable enable
using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Experimental.Functional {
    /// <summary>
    ///     A function transformation for <c>FuncEnv.Transform</c> (DR-43): applied to each of a functional env's functions,
    ///     generic over every signature, as <c>jax.jit</c> and <c>jax.vmap</c> transform any function they are given.
    /// </summary>
    /// <remarks>
    ///     A transform returns a delegate of the type it was given, so the env stores it back in the same slot; one that
    ///     wraps the function (logging, timing, caching) builds a delegate of <typeparamref name="TDelegate"/> around it, an
    ///     expression tree being the general way.
    /// </remarks>
    public interface IFunctionTransform {
        /// <summary>Transforms one function.</summary>
        /// <typeparam name="TDelegate">The function's delegate type, which the result keeps.</typeparam>
        /// <param name="function">The function as the env holds it now.</param>
        /// <returns>The function to hold instead.</returns>
        TDelegate Apply<TDelegate>(TDelegate function) where TDelegate : Delegate;
    }

    /// <summary>
    ///     The base of functional envs (Gymnasium's <c>FuncEnv</c>): an env as pure functions of an explicit state, for
    ///     stateless use, where the caller threads the state through <see cref="Initial"/>, <see cref="Transition"/>,
    ///     <see cref="Observation"/>, <see cref="Reward"/> and <see cref="Terminal"/>. Constants live on the subclass and
    ///     options set them by Python name. Gymnasium marks the API as internal and likely to change; so is this one.
    /// </summary>
    /// <typeparam name="TState">The state passed between the functions.</typeparam>
    /// <typeparam name="TObs">The observation type.</typeparam>
    /// <typeparam name="TAct">The action type.</typeparam>
    /// <typeparam name="TReward">The reward type.</typeparam>
    /// <typeparam name="TTerminal">The terminal flag's type.</typeparam>
    /// <typeparam name="TRenderState">The render state passed between the render functions.</typeparam>
    /// <typeparam name="TParams">The parameters every function takes, <see langword="null"/> or <c>default</c> for <see cref="DefaultParams"/>.</typeparam>
    /// <remarks>
    ///     <para>
    ///     In Python the functions are attributes that hold functions, which is what lets <c>transform</c> reassign them.
    ///     Here they are settable delegate properties under Gymnasium's names (DR-43): <c>env.Transition(state, action,
    ///     rng)</c> reads like the method call it stands for, assigning one replaces the function for every caller (C# code
    ///     and <see cref="PyAttributes"/> alike), and <see cref="Transform(IFunctionTransform)"/> is Python's seven
    ///     assignments. Each property starts bound to a protected <c>…Core</c> method, which is what a subclass overrides;
    ///     a subclass may also assign the properties itself.
    ///     </para>
    ///     <para>
    ///     Every function raises <see cref="NotImplementedError"/> until a subclass overrides it, as Gymnasium's do, except
    ///     the two info functions (empty infos) and <see cref="GetDefaultParams"/> (none). No JAX: <c>rng</c> is whatever
    ///     generator the subclass uses (DR-36). Gymnasium's quirk is kept: <c>transform</c> stores the transformed
    ///     <c>transition_info</c> as <see cref="StepInfo"/> and leaves <see cref="TransitionInfo"/> as it was.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/experimental/functional.py#L21">gymnasium/experimental/functional.py</see>.</para>
    /// </remarks>
    public abstract class FuncEnv<TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams> : IDynamicAttributes {
        /// <summary>The <c>step_info</c> function <see cref="Transform(IFunctionTransform)"/> created, or <see langword="null"/> before one ran.</summary>
        private TransitionInfoFunction? _stepInfo;

        /// <summary>
        ///     Binds every function to its <c>…Core</c> method, then initializes the environment constants: each option sets the
        ///     attribute of its Python name, then <see cref="DefaultParams"/> is read.
        /// </summary>
        /// <param name="options">Attribute values by Python name (<c>self.__dict__.update(options)</c>), or <see langword="null"/>.</param>
        /// <exception cref="AttributeError">Thrown for a read-only attribute.</exception>
        /// <exception cref="TypeError">Thrown when a value does not fit a declared attribute's type.</exception>
        protected FuncEnv(IReadOnlyDictionary<string, object?>? options = null) {
            // Bound first, so an option may replace a function as Python's __dict__.update can; binding a virtual method here
            // binds the subclass's override, since the object is of the derived type already.
            Initial = InitialCore;
            Transition = TransitionCore;
            Observation = ObservationCore;
            Reward = RewardCore;
            Terminal = TerminalCore;
            StateInfo = StateInfoCore;
            TransitionInfo = TransitionInfoCore;

            if (options is not null) {
                foreach (KeyValuePair<string, object?> option in options)
                    PyAttributes.SetAttr(this, option.Key, option.Value);
            }

            // A virtual call from the constructor, as Python's __init__ calls get_default_params: an override must not
            // depend on its own constructor having run.
            DefaultParams = GetDefaultParams(null);
        }

        /// <summary>The signature of <see cref="Initial"/>: the initial state from a generator.</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The initial state.</returns>
        public delegate TState InitialFunction(object? rng, TParams? @params = default);

        /// <summary>The signature of <see cref="Transition"/>: the next state.</summary>
        /// <param name="state">The state.</param>
        /// <param name="action">The action.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The next state.</returns>
        public delegate TState TransitionFunction(TState state, TAct action, object? rng, TParams? @params = default);

        /// <summary>The signature of <see cref="Observation"/>: a state's observation.</summary>
        /// <param name="state">The state.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The observation.</returns>
        public delegate TObs ObservationFunction(TState state, object? rng, TParams? @params = default);

        /// <summary>The signature of <see cref="Reward"/>: a transition's reward.</summary>
        /// <param name="state">The state before.</param>
        /// <param name="action">The action.</param>
        /// <param name="nextState">The state after.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The reward.</returns>
        public delegate TReward RewardFunction(TState state, TAct action, TState nextState, object? rng, TParams? @params = default);

        /// <summary>The signature of <see cref="Terminal"/>: whether a state is terminal.</summary>
        /// <param name="state">The state.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The terminal flag.</returns>
        public delegate TTerminal TerminalFunction(TState state, object? rng, TParams? @params = default);

        /// <summary>The signature of <see cref="StateInfo"/>: a state's info.</summary>
        /// <param name="state">The state.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>A new info.</returns>
        public delegate Info StateInfoFunction(TState state, TParams? @params = default);

        /// <summary>The signature of <see cref="TransitionInfo"/> and <see cref="StepInfo"/>: a transition's info.</summary>
        /// <param name="state">The state before.</param>
        /// <param name="action">The action.</param>
        /// <param name="nextState">The state after.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>A new info.</returns>
        public delegate Info TransitionInfoFunction(TState state, TAct action, TState nextState, TParams? @params = default);

        /// <summary>The observation space (<c>observation_space</c>), set by the subclass; <see langword="null"/> until then.</summary>
        public Space? ObservationSpace { get; set; }

        /// <summary>The action space (<c>action_space</c>), set by the subclass; <see langword="null"/> until then.</summary>
        public Space? ActionSpace { get; set; }

        /// <summary>The parameters <see cref="GetDefaultParams"/> gave at construction (<c>default_params</c>).</summary>
        public TParams? DefaultParams { get; set; }

        /// <summary>The run-time attributes: the options the class does not declare.</summary>
        public IDictionary<string, object?> DynamicAttributes { get; } = new Dictionary<string, object?>(StringComparer.Ordinal);

        /// <summary>Generates the initial state of the environment with a random number generator (<c>initial</c>); <see cref="InitialCore"/> until replaced.</summary>
        public InitialFunction Initial { get; set; }

        /// <summary>Updates (transitions) the state with an action and random number generator (<c>transition</c>); <see cref="TransitionCore"/> until replaced.</summary>
        public TransitionFunction Transition { get; set; }

        /// <summary>Generates an observation for a given state of an environment (<c>observation</c>); <see cref="ObservationCore"/> until replaced.</summary>
        public ObservationFunction Observation { get; set; }

        /// <summary>Computes the reward for a given transition (<c>reward</c>); <see cref="RewardCore"/> until replaced.</summary>
        public RewardFunction Reward { get; set; }

        /// <summary>Returns if the state is a final terminal state (<c>terminal</c>); <see cref="TerminalCore"/> until replaced.</summary>
        public TerminalFunction Terminal { get; set; }

        /// <summary>Info dict about a single state (<c>state_info</c>); <see cref="StateInfoCore"/> until replaced.</summary>
        public StateInfoFunction StateInfo { get; set; }

        /// <summary>Info dict about a full transition (<c>transition_info</c>); <see cref="TransitionInfoCore"/> until replaced. <see cref="Transform(IFunctionTransform)"/> leaves it as it was.</summary>
        public TransitionInfoFunction TransitionInfo { get; set; }

        /// <summary>
        ///     The transformed <c>transition_info</c> (<c>step_info</c>), which Gymnasium's <c>transform</c> creates under this
        ///     name: absent until <see cref="Transform(IFunctionTransform)"/> runs, as the attribute is in Python.
        /// </summary>
        /// <exception cref="AttributeError">Thrown by the getter before a transform: <c>'Bare' object has no attribute 'step_info'</c>.</exception>
        /// <exception cref="ArgumentNullException">Thrown by the setter for <see langword="null"/>.</exception>
        public TransitionInfoFunction StepInfo {
            get => _stepInfo ?? throw new AttributeError($"'{PyAttributes.PyTypeName(GetType())}' object has no attribute 'step_info'");
            set => _stepInfo = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>The initial state, what <see cref="Initial"/> runs until replaced; override it in a subclass.</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The initial state.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        protected virtual TState InitialCore(object? rng, TParams? @params) => throw new NotImplementedError();

        /// <summary>The next state, what <see cref="Transition"/> runs until replaced; override it in a subclass.</summary>
        /// <param name="state">The state.</param>
        /// <param name="action">The action.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The next state.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        protected virtual TState TransitionCore(TState state, TAct action, object? rng, TParams? @params) => throw new NotImplementedError();

        /// <summary>A state's observation, what <see cref="Observation"/> runs until replaced; override it in a subclass.</summary>
        /// <param name="state">The state.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The observation.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        protected virtual TObs ObservationCore(TState state, object? rng, TParams? @params) => throw new NotImplementedError();

        /// <summary>A transition's reward, what <see cref="Reward"/> runs until replaced; override it in a subclass.</summary>
        /// <param name="state">The state before.</param>
        /// <param name="action">The action.</param>
        /// <param name="nextState">The state after.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The reward.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        protected virtual TReward RewardCore(TState state, TAct action, TState nextState, object? rng, TParams? @params) => throw new NotImplementedError();

        /// <summary>Whether a state is terminal, what <see cref="Terminal"/> runs until replaced; override it in a subclass.</summary>
        /// <param name="state">The state.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>Whether the state is terminal.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        protected virtual TTerminal TerminalCore(TState state, object? rng, TParams? @params) => throw new NotImplementedError();

        /// <summary>A state's info, what <see cref="StateInfo"/> runs until replaced: empty unless overridden.</summary>
        /// <param name="state">The state.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>A new info.</returns>
        protected virtual Info StateInfoCore(TState state, TParams? @params) => new();

        /// <summary>A transition's info, what <see cref="TransitionInfo"/> runs until replaced: empty unless overridden.</summary>
        /// <param name="state">The state before.</param>
        /// <param name="action">The action.</param>
        /// <param name="nextState">The state after.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>A new info.</returns>
        protected virtual Info TransitionInfoCore(TState state, TAct action, TState nextState, TParams? @params) => new();

        /// <summary>
        ///     Functional transformations (<c>transform(func)</c>): replaces each function with what <paramref name="transform"/>
        ///     makes of it, for every caller, in Gymnasium's order: <c>initial</c>, <c>transition</c>, <c>observation</c>,
        ///     <c>reward</c>, <c>terminal</c>, <c>state_info</c>, and <c>transition_info</c> stored as <see cref="StepInfo"/>.
        /// </summary>
        /// <param name="transform">The transformation, generic over the functions' signatures.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="transform"/> is <see langword="null"/>, or returns <see langword="null"/>.</exception>
        public void Transform(IFunctionTransform transform) {
            if (transform is null)
                throw new ArgumentNullException(nameof(transform));
            Initial = Checked(transform.Apply(Initial));
            Transition = Checked(transform.Apply(Transition));
            Observation = Checked(transform.Apply(Observation));
            Reward = Checked(transform.Apply(Reward));
            Terminal = Checked(transform.Apply(Terminal));
            StateInfo = Checked(transform.Apply(StateInfo));
            // Gymnasium assigns the transformed transition_info to step_info and leaves transition_info as it was.
            StepInfo = Checked(transform.Apply(TransitionInfo));
        }

        /// <summary>
        ///     Functional transformations with an untyped function, as Python's <c>transform(lambda f: ...)</c> takes any
        ///     callable: <paramref name="func"/> receives each function as a delegate and returns its replacement, which must
        ///     be of the same delegate type or have the same signature (a lambda typed <c>Func&lt;...&gt;</c> is rebound).
        /// </summary>
        /// <param name="func">The transformation.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="func"/> is <see langword="null"/>, or returns <see langword="null"/>.</exception>
        /// <exception cref="TypeError">Thrown when a replacement's signature cannot stand for the function it replaces.</exception>
        public void Transform(Func<Delegate, Delegate> func) {
            if (func is null)
                throw new ArgumentNullException(nameof(func));
            Transform(new UntypedFunctionTransform(func));
        }

        /// <summary>Show the state (<c>render_image</c>): the next render state and the frame.</summary>
        /// <param name="state">The state.</param>
        /// <param name="renderState">The render state.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The next render state and the frame.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual (TRenderState RenderState, NDArray Frame) RenderImage(TState state, TRenderState renderState, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Initialize the render state (<c>render_init</c>).</summary>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <param name="kwargs">The renderer's options by Python name, or <see langword="null"/>.</param>
        /// <returns>The render state.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual TRenderState RenderInit(TParams? @params = default, IReadOnlyDictionary<string, object?>? kwargs = null) => throw new NotImplementedError();

        /// <summary>Close the render state (<c>render_close</c>).</summary>
        /// <param name="renderState">The render state.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual void RenderClose(TRenderState renderState, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Get the default params (<c>get_default_params</c>): none unless overridden; called once by the constructor.</summary>
        /// <param name="kwargs">The parameters' options by Python name, or <see langword="null"/>.</param>
        /// <returns>The parameters, or <c>default</c>.</returns>
        public virtual TParams? GetDefaultParams(IReadOnlyDictionary<string, object?>? kwargs = null) => default;

        /// <summary>A transformed function, refused when the transform returned none.</summary>
        /// <typeparam name="TDelegate">The function's delegate type.</typeparam>
        /// <param name="function">What the transform returned.</param>
        /// <returns>The function.</returns>
        /// <exception cref="ArgumentNullException">Thrown for <see langword="null"/>: a slot never holds no function.</exception>
        private static TDelegate Checked<TDelegate>(TDelegate? function) where TDelegate : Delegate =>
            function ?? throw new ArgumentNullException(nameof(function), "The transform returned no function.");

        /// <summary><see cref="Transform(Func{Delegate, Delegate})"/>'s untyped function as an <see cref="IFunctionTransform"/>.</summary>
        private sealed class UntypedFunctionTransform : IFunctionTransform {
            /// <summary>The untyped function.</summary>
            private readonly Func<Delegate, Delegate> _func;

            /// <summary>Wraps <paramref name="func"/>.</summary>
            /// <param name="func">The untyped transformation.</param>
            public UntypedFunctionTransform(Func<Delegate, Delegate> func) => _func = func;

            /// <summary>The untyped function's result, as the function's own delegate type.</summary>
            /// <typeparam name="TDelegate">The function's delegate type.</typeparam>
            /// <param name="function">The function.</param>
            /// <returns>The replacement.</returns>
            /// <exception cref="ArgumentNullException">Thrown when the function returns <see langword="null"/>.</exception>
            /// <exception cref="TypeError">Thrown when the result's signature cannot stand for <typeparamref name="TDelegate"/>.</exception>
            public TDelegate Apply<TDelegate>(TDelegate function) where TDelegate : Delegate {
                Delegate result = _func(function) ?? throw new ArgumentNullException(nameof(function), "The transform returned no function.");
                if (result is TDelegate typed)
                    return typed;
                try {
                    // Same signature, another delegate type (a lambda typed Func<...>): bound again to the slot's type.
                    return (TDelegate)Delegate.CreateDelegate(typeof(TDelegate), result.Target, result.Method);
                } catch (ArgumentException) {
                    throw new TypeError($"transform returned a {result.GetType().Name}, which cannot stand for a {typeof(TDelegate).Name}");
                }
            }
        }
    }

    /// <summary>
    ///     <see cref="FuncEnv{TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams}"/> untyped, as Gymnasium's own
    ///     generic <c>FuncEnv</c> is used unparameterized: every type is <see langword="object"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/experimental/functional.py#L21">gymnasium/experimental/functional.py</see>.</remarks>
    public abstract class FuncEnv : FuncEnv<object?, object?, object?, object?, object?, object?, object?> {
        /// <summary>Initialize the environment constants from <paramref name="options"/>.</summary>
        /// <param name="options">Attribute values by Python name, or <see langword="null"/>.</param>
        /// <exception cref="AttributeError">Thrown for a read-only attribute.</exception>
        /// <exception cref="TypeError">Thrown when a value does not fit a declared attribute's type.</exception>
        protected FuncEnv(IReadOnlyDictionary<string, object?>? options = null) : base(options) { }
    }
}
