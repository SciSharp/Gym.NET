#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     The env contract over <see cref="object"/>: what code that holds envs of any observation and action type (wrappers
    ///     over unknown envs, the registry, vector envs, the legacy adapter) programs against. Every env is an
    ///     <see cref="Env{TObs, TAct}"/>, which implements it; its typed members are the ones to call when the types are known.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Members follow Gymnasium's <c>Env</c> one to one with the port plan's naming (§3.8): <c>step</c> is
    ///     <see cref="Step"/>, <c>np_random_seed</c> is <see cref="NpRandomSeed"/>, <c>get_wrapper_attr</c> is
    ///     <see cref="GetWrapperAttr"/>. <see cref="IDisposable.Dispose"/> calls <see cref="Close"/>, as leaving a Python
    ///     <c>with</c> block does.
    ///     </para>
    ///     <para>Not thread-safe: step, reset and render an instance from one thread at a time.</para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/core.py#L25">gymnasium/core.py</see>.</para>
    /// </remarks>
    public interface IEnv : IDisposable {
        /// <summary>
        ///     The env's metadata, <c>{"render_modes": [...], "render_fps": ...}</c> with Python key spelling. A wrapper returns
        ///     its env's unless it was assigned its own.
        /// </summary>
        Metadata Metadata { get; set; }

        /// <summary>The render mode fixed at construction (<c>"human"</c>, <c>"rgb_array"</c>, ...), or <see langword="null"/> when the env does not render.</summary>
        string? RenderMode { get; }

        /// <summary>
        ///     The spec <c>make</c> attached, or <see langword="null"/> for an env built with its constructor. A wrapper's is its
        ///     env's, extended with the wrapper (and read-only: assigning it raises <see cref="AttributeError"/>).
        /// </summary>
        EnvSpec? Spec { get; set; }

        /// <summary>The space every action must belong to.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while the env has not set one, as reading the missing Python attribute does.</exception>
        Space ActionSpace { get; set; }

        /// <summary>The space every observation belongs to.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while the env has not set one.</exception>
        Space ObservationSpace { get; set; }

        /// <summary>
        ///     The env's generator (<c>np_random</c>): created from OS entropy on first use, replaced by a seeded
        ///     <see cref="Reset"/>; assigning one makes <see cref="NpRandomSeed"/> <c>-1</c>. A wrapper forwards to the
        ///     innermost env.
        /// </summary>
        Generator NpRandom { get; set; }

        /// <summary>
        ///     The seed <see cref="NpRandom"/> was built from (<c>np_random_seed</c>): the seed of the last seeded reset, the
        ///     entropy of an unseeded generator (creating it first if needed), or <c>-1</c> after a generator was assigned.
        ///     Read-only, as in Gymnasium.
        /// </summary>
        BigInteger NpRandomSeed { get; }

        /// <summary>The innermost env, beneath every wrapper (<c>unwrapped</c>); an env that is not a wrapper returns itself.</summary>
        IEnv Unwrapped { get; }

        /// <summary>
        ///     Runs one timestep with an action of any type; the env converts it to its action type first (a typed env's
        ///     <c>Step(TAct)</c> takes it directly).
        /// </summary>
        /// <param name="action">An element of <see cref="ActionSpace"/>.</param>
        /// <returns>The transition, its observation boxed.</returns>
        /// <exception cref="Exception">Whatever the env raises for the action: an <see cref="AssertionError"/> for an invalid one, a <see cref="ResetNeeded"/> from <c>OrderEnforcing</c>, ....</exception>
        StepResult<object?> Step(object? action);

        /// <summary>
        ///     Steps asynchronously with an action of any type (see <see cref="Env{TObs, TAct}.StepAsync"/>): by default the step
        ///     runs on the calling thread and the task completes synchronously; an action that cannot be converted, like any
        ///     error of the step, faults the task.
        /// </summary>
        /// <param name="action">An element of <see cref="ActionSpace"/>.</param>
        /// <param name="cancellationToken">Checked before the step starts.</param>
        /// <returns>A value task that completes with the transition, its observation boxed.</returns>
        ValueTask<StepResult<object?>> StepAsync(object? action, CancellationToken cancellationToken = default);

        /// <summary>Starts an episode (<c>reset(seed=seed, options=options)</c>).</summary>
        /// <param name="seed">
        ///     Reseeds <see cref="NpRandom"/> first when given (any non-negative integer); <see langword="null"/> keeps the
        ///     generator, so a stream seeded once runs on across episodes.
        /// </param>
        /// <param name="options">Env-specific options (for example <c>"low"</c>/<c>"high"</c> in classic_control), or <see langword="null"/>.</param>
        /// <returns>The first observation, boxed, and the info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null);

        /// <summary>Renders as <see cref="RenderMode"/> says (<c>render()</c>).</summary>
        /// <returns>A frame, text, collected frames, or <see cref="RenderResult.None"/> (human mode, no render mode).</returns>
        /// <exception cref="NotImplementedError">Thrown by an env that does not render.</exception>
        RenderResult Render();

        /// <summary>Releases what the env holds, windows included (<c>close()</c>); closing twice does nothing more.</summary>
        void Close();

        /// <summary>
        ///     Python's <c>has_wrapper_attr(name)</c>: whether this env, or for a wrapper any env beneath it, has the Python
        ///     attribute <paramref name="name"/> (<c>"gravity"</c>, <c>"_disable_render_order_enforcing"</c>).
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns><see langword="true"/> when some layer has it.</returns>
        bool HasWrapperAttr(string name);

        /// <summary>
        ///     Python's <c>get_wrapper_attr(name)</c>: the attribute from the outermost layer that has it.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns>The value, boxed.</returns>
        /// <exception cref="AttributeError">Thrown when no layer has it (<c>wrapper TimeLimit has no attribute 'x'</c>).</exception>
        object? GetWrapperAttr(string name);

        /// <summary>
        ///     Python's <c>set_wrapper_attr(name, value, force=force)</c>: assigns the attribute on the outermost layer that has
        ///     it; when none has it, <paramref name="force"/> creates it on this layer.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <param name="value">The value; numbers convert to a numeric attribute's type.</param>
        /// <param name="force">Create the attribute here when no layer has it (Gymnasium's default, <see langword="true"/>).</param>
        /// <returns><see langword="true"/> when some layer was assigned.</returns>
        /// <exception cref="AttributeError">Thrown when the attribute is read-only (<c>np_random_seed</c>).</exception>
        /// <exception cref="TypeError">Thrown when the value does not fit the attribute's type.</exception>
        bool SetWrapperAttr(string name, object? value, bool force = true);
    }
}
