#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     Truncates the episode once <see cref="MaxEpisodeSteps"/> steps have run since the last reset. This is where the
    ///     <c>truncated</c> signal comes from for envs that have no limit of their own, as opposed to <c>terminated</c>, which
    ///     the MDP decides. <c>make</c> adds it when the spec has <c>max_episode_steps</c>.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     Stepping before the first reset steps the env and then raises Python's <see cref="TypeError"/> (the step counter is
    ///     still <c>None</c>), exactly as Gymnasium does; <c>OrderEnforcing</c> beneath it catches that mistake first in a
    ///     <c>make</c>-built stack. The step that reaches the limit returns <c>truncated=True</c> even when it also terminates.
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/wrappers/common.py#L42">gymnasium/wrappers/common.py</see>.
    /// </remarks>
    public class TimeLimit<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/> with a step limit.</summary>
        /// <param name="env">The env to limit.</param>
        /// <param name="maxEpisodeSteps">
        ///     The step at which the episode is truncated (<c>elapsed &gt;= max_episode_steps</c>). Typed as nullable so that a
        ///     spec's <c>None</c> reaches Gymnasium's check, as <c>TimeLimit(env, max_episode_steps=None)</c> does.
        /// </param>
        /// <exception cref="TypeError">
        ///     Thrown when <paramref name="maxEpisodeSteps"/> is <see langword="null"/>: <c>Expect the `max_episode_steps` to be
        ///     an integer, actually: &lt;class 'NoneType'&gt;</c>. Python's other non-integers cannot be passed in C#.
        /// </exception>
        /// <exception cref="ValueError">Thrown when it is not positive: <c>Expect the `max_episode_steps` to be positive, actually: -1</c>.</exception>
        /// <remarks>The arguments are checked before the env, as in Python, so a bad limit is reported even for a <see langword="null"/> env.</remarks>
        public TimeLimit(Env<TObs, TAct> env, long? maxEpisodeSteps) : base(CheckArguments(env, maxEpisodeSteps)) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["max_episode_steps"] = maxEpisodeSteps });
            MaxEpisodeSteps = maxEpisodeSteps!.Value;
            ElapsedSteps = null;
        }

        /// <summary>The recorded constructor arguments, <c>{'max_episode_steps': n}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The step limit (Python's <c>_max_episode_steps</c>); assigning it changes the limit of the running episode too.</summary>
        [PyName("_max_episode_steps")]
        public long MaxEpisodeSteps { get; set; }

        /// <summary>
        ///     The steps since the last reset (Python's <c>_elapsed_steps</c>), <see langword="null"/> until the first reset.
        /// </summary>
        [PyName("_elapsed_steps")]
        public long? ElapsedSteps { get; set; }

        /// <summary>
        ///     The env's spec with <see cref="EnvSpec.MaxEpisodeSteps"/> set to this wrapper's limit (a deep copy, cached), or
        ///     <see langword="null"/> when the env has none. Unlike other wrappers, no <see cref="WrapperSpec"/> is added: the
        ///     limit is part of the spec itself.
        /// </summary>
        /// <exception cref="AttributeError">Thrown when assigned, as for every wrapper.</exception>
        public override EnvSpec? Spec {
            get {
                if (CachedSpec is not null)
                    return CachedSpec;

                EnvSpec? envSpec = Env.Spec;
                if (envSpec is not null) {
                    try {
                        envSpec = envSpec.DeepCopy();
                        envSpec.MaxEpisodeSteps = MaxEpisodeSteps;
                    } catch (Exception e) {
                        Logger.Warn($"An exception occurred ({e.Message}) while copying the environment spec={envSpec}");
                        return null;
                    }
                }

                CachedSpec = envSpec;
                return envSpec;
            }
            set => base.Spec = value;
        }

        /// <summary>
        ///     Steps the env, counts the step, and sets <c>truncated</c> once the count reaches <see cref="MaxEpisodeSteps"/>.
        /// </summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition, truncated at the limit.</returns>
        /// <exception cref="TypeError">
        ///     Thrown after the env stepped when the wrapper was never reset: <c>unsupported operand type(s) for +=:
        ///     'NoneType' and 'int'</c>, Python's error for <c>None += 1</c>.
        /// </exception>
        public override StepResult<TObs> Step(TAct action) {
            (TObs observation, double reward, bool terminated, bool truncated, Info info) = Env.Step(action);
            // self._elapsed_steps += 1, with _elapsed_steps still None before the first reset.
            if (ElapsedSteps is null)
                throw new TypeError("unsupported operand type(s) for +=: 'NoneType' and 'int'");
            ElapsedSteps += 1;

            if (ElapsedSteps >= MaxEpisodeSteps)
                truncated = true;

            return new StepResult<TObs>(observation, reward, terminated, truncated, info);
        }

        /// <summary>Resets the step count to zero, then resets the env.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ElapsedSteps = 0;
            return base.Reset(seed, options);
        }

        /// <summary>Gymnasium's argument checks, run before the base constructor sees the env.</summary>
        /// <param name="env">The env, returned unchanged.</param>
        /// <param name="maxEpisodeSteps">The step limit.</param>
        /// <returns><paramref name="env"/>.</returns>
        /// <exception cref="TypeError">Thrown for a missing limit.</exception>
        /// <exception cref="ValueError">Thrown for a limit that is not positive.</exception>
        private static Env<TObs, TAct> CheckArguments(Env<TObs, TAct> env, long? maxEpisodeSteps) {
            if (maxEpisodeSteps is null)
                throw new TypeError($"Expect the `max_episode_steps` to be an integer, actually: {PyBuiltins.TypeRepr(null)}");
            if (maxEpisodeSteps <= 0)
                throw new ValueError($"Expect the `max_episode_steps` to be positive, actually: {PyBuiltins.Str(maxEpisodeSteps.Value)}");
            return env;
        }
    }

    /// <summary>
    ///     Raises <see cref="ResetNeeded"/> when <c>step</c> or <c>render</c> is called before the first <c>reset</c>, a
    ///     mistake that would otherwise fail somewhere inside the env with a less helpful error. <c>make</c> adds it unless the
    ///     spec disables <c>order_enforce</c>.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/9e04324f6b0adbe19112206dfe247edc4142e7ec/gymnasium/wrappers/common.py#L344">gymnasium/wrappers/common.py</see>.</remarks>
    public class OrderEnforcing<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Whether <see cref="Reset"/> ran (Python's <c>_has_reset</c>).</summary>
        [PyName("_has_reset")]
        private bool _hasReset;

        /// <summary>Whether <see cref="Render"/> may run before a reset (Python's <c>_disable_render_order_enforcing</c>).</summary>
        [PyName("_disable_render_order_enforcing")]
        private bool _disableRenderOrderEnforcing;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env whose call order is enforced.</param>
        /// <param name="disableRenderOrderEnforcing">Allow <see cref="Render"/> before the first reset.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public OrderEnforcing(Env<TObs, TAct> env, bool disableRenderOrderEnforcing = false) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["disable_render_order_enforcing"] = disableRenderOrderEnforcing });
            _hasReset = false;
            _disableRenderOrderEnforcing = disableRenderOrderEnforcing;
        }

        /// <summary>The recorded constructor arguments, <c>{'disable_render_order_enforcing': False}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>Whether the env has been reset through this wrapper (<c>has_reset</c>).</summary>
        public bool HasReset => _hasReset;

        /// <summary>
        ///     The env's spec with <see cref="EnvSpec.OrderEnforce"/> set (a deep copy, cached), or <see langword="null"/> when
        ///     the env has none. No <see cref="WrapperSpec"/> is added: order enforcing is part of the spec itself.
        /// </summary>
        /// <exception cref="AttributeError">Thrown when assigned, as for every wrapper.</exception>
        public override EnvSpec? Spec {
            get {
                if (CachedSpec is not null)
                    return CachedSpec;

                EnvSpec? envSpec = Env.Spec;
                if (envSpec is not null) {
                    try {
                        envSpec = envSpec.DeepCopy();
                        envSpec.OrderEnforce = true;
                    } catch (Exception e) {
                        Logger.Warn($"An exception occurred ({e.Message}) while copying the environment spec={envSpec}");
                        return null;
                    }
                }

                CachedSpec = envSpec;
                return envSpec;
            }
            set => base.Spec = value;
        }

        /// <summary>Steps the env, once it has been reset.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition.</returns>
        /// <exception cref="ResetNeeded">Thrown before the first reset: <c>Cannot call env.step() before calling env.reset()</c>.</exception>
        public override StepResult<TObs> Step(TAct action) {
            if (!_hasReset)
                throw new ResetNeeded("Cannot call env.step() before calling env.reset()");
            return base.Step(action);
        }

        /// <summary>Records the reset, then resets the env.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        /// <remarks>The reset counts even when the env's reset then fails, as in Python, where the flag is set first.</remarks>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            _hasReset = true;
            return base.Reset(seed, options);
        }

        /// <summary>Renders the env, once it has been reset or when render order enforcing is disabled.</summary>
        /// <returns>The env's rendering.</returns>
        /// <exception cref="ResetNeeded">
        ///     Thrown before the first reset: <c>Cannot call `env.render()` before calling `env.reset()`, if this is an intended
        ///     action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.</c>
        /// </exception>
        public override RenderResult Render() {
            if (!_disableRenderOrderEnforcing && !_hasReset)
                throw new ResetNeeded("Cannot call `env.render()` before calling `env.reset()`, if this is an intended action, set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.");
            return base.Render();
        }
    }
}
