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
    ///     Every function raises <see cref="NotImplementedError"/> until a subclass overrides it, as Gymnasium's do, except
    ///     the two info functions (empty infos) and <see cref="GetDefaultParams"/> (none). No JAX: <c>rng</c> is whatever
    ///     generator the subclass uses (DR-36).
    ///     </para>
    ///     <para>
    ///     <see cref="Transform"/> cannot rebind methods as Python's attribute assignment does: it stores the transformed
    ///     functions as run-time attributes by Python name (<c>initial</c>, <c>transition</c>, ..., and <c>step_info</c> for
    ///     the transformed <c>transition_info</c>, Gymnasium's own naming), which <see cref="PyAttributes.GetAttr"/> returns;
    ///     calling the methods directly still runs the originals.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/experimental/functional.py#L21">gymnasium/experimental/functional.py</see>.</para>
    /// </remarks>
    public abstract class FuncEnv<TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams> : IDynamicAttributes {
        /// <summary>Initialize the environment constants: each option sets the attribute of its Python name, then <see cref="DefaultParams"/> is read.</summary>
        /// <param name="options">Attribute values by Python name (<c>self.__dict__.update(options)</c>), or <see langword="null"/>.</param>
        /// <exception cref="AttributeError">Thrown for a read-only attribute.</exception>
        /// <exception cref="TypeError">Thrown when a value does not fit a declared attribute's type.</exception>
        protected FuncEnv(IReadOnlyDictionary<string, object?>? options = null) {
            if (options is not null) {
                foreach (KeyValuePair<string, object?> option in options)
                    PyAttributes.SetAttr(this, option.Key, option.Value);
            }

            // A virtual call from the constructor, as Python's __init__ calls get_default_params: an override must not
            // depend on its own constructor having run.
            DefaultParams = GetDefaultParams(null);
        }

        /// <summary>The observation space (<c>observation_space</c>), set by the subclass; <see langword="null"/> until then.</summary>
        public Space? ObservationSpace { get; set; }

        /// <summary>The action space (<c>action_space</c>), set by the subclass; <see langword="null"/> until then.</summary>
        public Space? ActionSpace { get; set; }

        /// <summary>The parameters <see cref="GetDefaultParams"/> gave at construction (<c>default_params</c>).</summary>
        public TParams? DefaultParams { get; set; }

        /// <summary>The run-time attributes: the options the class does not declare, and the functions <see cref="Transform"/> stored.</summary>
        public IDictionary<string, object?> DynamicAttributes { get; } = new Dictionary<string, object?>(StringComparer.Ordinal);

        /// <summary>Generates the initial state of the environment with a random number generator (<c>initial</c>).</summary>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The initial state.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual TState Initial(object? rng, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Updates (transitions) the state with an action and random number generator (<c>transition</c>).</summary>
        /// <param name="state">The state.</param>
        /// <param name="action">The action.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The next state.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual TState Transition(TState state, TAct action, object? rng, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Generates an observation for a given state of an environment (<c>observation</c>).</summary>
        /// <param name="state">The state.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The observation.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual TObs Observation(TState state, object? rng, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Computes the reward for a given transition between <paramref name="state"/>, <paramref name="action"/> to <paramref name="nextState"/> (<c>reward</c>).</summary>
        /// <param name="state">The state before.</param>
        /// <param name="action">The action.</param>
        /// <param name="nextState">The state after.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>The reward.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual TReward Reward(TState state, TAct action, TState nextState, object? rng, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Returns if the state is a final terminal state (<c>terminal</c>).</summary>
        /// <param name="state">The state.</param>
        /// <param name="rng">The generator.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>Whether the state is terminal.</returns>
        /// <exception cref="NotImplementedError">Thrown until a subclass overrides it.</exception>
        public virtual TTerminal Terminal(TState state, object? rng, TParams? @params = default) => throw new NotImplementedError();

        /// <summary>Info dict about a single state (<c>state_info</c>): empty unless overridden.</summary>
        /// <param name="state">The state.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>A new info.</returns>
        public virtual Info StateInfo(TState state, TParams? @params = default) => new();

        /// <summary>Info dict about a full transition (<c>transition_info</c>): empty unless overridden.</summary>
        /// <param name="state">The state before.</param>
        /// <param name="action">The action.</param>
        /// <param name="nextState">The state after.</param>
        /// <param name="params">The parameters, or <c>default</c>.</param>
        /// <returns>A new info.</returns>
        public virtual Info TransitionInfo(TState state, TAct action, TState nextState, TParams? @params = default) => new();

        /// <summary>
        ///     Functional transformations (<c>transform(func)</c>): <paramref name="func"/> receives each function as a delegate
        ///     and its result is stored as the run-time attribute of that function's Python name (see the remarks of
        ///     <see cref="FuncEnv{TState, TObs, TAct, TReward, TTerminal, TRenderState, TParams}"/>).
        /// </summary>
        /// <param name="func">The transformation, from a function to its transformed form.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="func"/> is <see langword="null"/>.</exception>
        public void Transform(Func<Delegate, Delegate> func) {
            if (func is null)
                throw new ArgumentNullException(nameof(func));
            DynamicAttributes["initial"] = func(new Func<object?, TParams?, TState>(Initial));
            DynamicAttributes["transition"] = func(new Func<TState, TAct, object?, TParams?, TState>(Transition));
            DynamicAttributes["observation"] = func(new Func<TState, object?, TParams?, TObs>(Observation));
            DynamicAttributes["reward"] = func(new Func<TState, TAct, TState, object?, TParams?, TReward>(Reward));
            DynamicAttributes["terminal"] = func(new Func<TState, object?, TParams?, TTerminal>(Terminal));
            DynamicAttributes["state_info"] = func(new Func<TState, TParams?, Info>(StateInfo));
            // Gymnasium assigns the transformed transition_info to step_info, and so does this.
            DynamicAttributes["step_info"] = func(new Func<TState, TAct, TState, TParams?, Info>(TransitionInfo));
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
