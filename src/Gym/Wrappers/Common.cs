#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
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
    ///     Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/common.py#L43">gymnasium/wrappers/common.py</see>.
    /// </remarks>
    [GymOptions]
    public partial class TimeLimit<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
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
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/common.py#L371">gymnasium/wrappers/common.py</see>.</remarks>
    [GymOptions]
    public partial class OrderEnforcing<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
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

    /// <summary>
    ///     Runs <see cref="PassiveChecker"/>'s checks around the env so that a broken env is reported where it breaks: the
    ///     spaces at construction, then the first reset, step and render, and whether the reset and the first step, then the
    ///     first and second steps, returned shared data. Every later call is forwarded unchanged, so the checks cost nothing
    ///     for the rest of the run. <c>make</c> adds it beneath <c>OrderEnforcing</c> unless <c>disable_env_checker</c> is set.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Problems are warnings through <see cref="Logger"/>, or Gymnasium's assertions for what cannot work; the env's
    ///     results are never changed. The reuse check holds views of the reset's and first step's observation and info (see
    ///     <see cref="RetainedData"/>) until it has run, then lets them go, so the env's memory is not kept for the run.
    ///     </para>
    ///     <para>
    ///     As in Gymnasium, stepping before the first reset steps the env and then fails with Python's unpacking
    ///     <see cref="TypeError"/>, since there is no reset data to compare with; <c>OrderEnforcing</c> above it in a
    ///     <c>make</c>-built stack catches that mistake first.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/common.py#L225">gymnasium/wrappers/common.py</see>.</para>
    /// </remarks>
    public class PassiveEnvChecker<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/> and checks its action and observation spaces.</summary>
        /// <param name="env">The env to check.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">
        ///     Thrown when the env has no action space (<c>The environment must specify an action space.
        ///     https://gymnasium.farama.org/introduction/create_custom_env/</c>) or no observation space (the same with
        ///     "observation"), the action space being checked first.
        /// </exception>
        /// <exception cref="AssertionError">Thrown for a space that cannot work (<see cref="PassiveChecker.CheckSpace"/>).</exception>
        public PassiveEnvChecker(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info());

            PassiveChecker.CheckActionSpace(RequireSpace(() => env.ActionSpace, "action"));
            PassiveChecker.CheckObservationSpace(RequireSpace(() => env.ObservationSpace, "observation"));

            CheckedReset = false;
            CheckedStep = false;
            CheckedRender = false;
            CloseCalled = false;

            // For checking that reset and step do not reuse their observation and info objects, comparing the reset with
            // the first step, then the first step with the second step.
            CheckedDataReuse = false;
            _previousObs = null;
            _previousInfo = null;
        }

        /// <summary>The retained observation of the reset or first step (half of Python's <c>_previous_data</c>), owned.</summary>
        private RetainedData? _previousObs;

        /// <summary>The retained info of the reset or first step (the other half of <c>_previous_data</c>), owned.</summary>
        private RetainedData? _previousInfo;

        /// <summary>The recorded constructor arguments, <c>{}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>Whether the first reset was checked (<c>checked_reset</c>); setting it back to <see langword="false"/> checks the next reset.</summary>
        public bool CheckedReset { get; set; }

        /// <summary>Whether the first step was checked (<c>checked_step</c>); setting it back to <see langword="false"/> checks the next step.</summary>
        public bool CheckedStep { get; set; }

        /// <summary>Whether the first render was checked (<c>checked_render</c>); setting it back to <see langword="false"/> checks the next render.</summary>
        public bool CheckedRender { get; set; }

        /// <summary>Whether <see cref="Close"/> ran (<c>close_called</c>); a later close that throws is reported before the error propagates.</summary>
        public bool CloseCalled { get; set; }

        /// <summary>Whether the data-reuse check of the first two steps ran (<c>checked_data_reuse</c>).</summary>
        public bool CheckedDataReuse { get; set; }

        /// <summary>
        ///     The reset's or first step's observation and info, retained for the reuse check of the next step (Python's
        ///     <c>_previous_data</c>); <see langword="null"/> before the first checked reset and once the check is done. The
        ///     wrapper owns the retained views and releases them when it replaces them or is disposed: read, never dispose.
        /// </summary>
        [PyName("_previous_data")]
        public (RetainedData Obs, RetainedData Info)? PreviousData => _previousObs is null ? null : (_previousObs, _previousInfo!);

        /// <summary>
        ///     The env's spec with <see cref="EnvSpec.DisableEnvChecker"/> cleared (a deep copy, cached), or
        ///     <see langword="null"/> when the env has none. No <see cref="WrapperSpec"/> is added: the checker is part of the
        ///     spec itself.
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
                        envSpec.DisableEnvChecker = false;
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
        ///     Steps the env. The first step runs <see cref="PassiveChecker.EnvStepPassiveChecker{TObs, TAct}"/> and compares
        ///     its data with the reset's; the second compares its data with the first step's; later steps are forwarded.
        /// </summary>
        /// <param name="action">The action, passed on unchecked (some envs accept out-of-bounds actions).</param>
        /// <returns>The env's transition, unchanged.</returns>
        /// <exception cref="AssertionError">Thrown by the first step's checks for a malformed observation or a <see langword="null"/> info.</exception>
        /// <exception cref="TypeError">
        ///     Thrown after the env stepped when there is no reset data to compare with (stepped before the first reset):
        ///     <c>cannot unpack non-iterable NoneType object</c>, as in Gymnasium.
        /// </exception>
        public override StepResult<TObs> Step(TAct action) {
            if (!CheckedStep) {
                CheckedStep = true;
                StepResult<TObs> result = PassiveChecker.EnvStepPassiveChecker(Env, action);

                (RetainedData Obs, RetainedData Info) previous = PreviousData ?? throw new TypeError("cannot unpack non-iterable NoneType object");
                PassiveChecker.EnvDataReusePassiveChecker(previous.Obs, previous.Info, (result.Observation, result.Info), "reset", "step");
                ReplacePreviousData((PassiveChecker.Retain(result.Observation), PassiveChecker.Retain(result.Info)));
                return result;
            }

            StepResult<TObs> stepped = Env.Step(action);
            if (!CheckedDataReuse) {
                (RetainedData Obs, RetainedData Info) previous = PreviousData ?? throw new TypeError("cannot unpack non-iterable NoneType object");
                PassiveChecker.EnvDataReusePassiveChecker(previous.Obs, previous.Info, (stepped.Observation, stepped.Info), "step", "step");
                // The check is done: stop holding the user's data.
                ReplacePreviousData(null);
                CheckedDataReuse = true;
            }

            return stepped;
        }

        /// <summary>
        ///     Resets the env. The first reset runs <see cref="PassiveChecker.EnvResetPassiveChecker{TObs, TAct}"/> and keeps
        ///     its data for the first step's reuse check; later resets are forwarded.
        /// </summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info, unchanged.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        /// <exception cref="AssertionError">Thrown by the first reset's checks for a malformed observation or a <see langword="null"/> info.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            if (!CheckedReset) {
                CheckedReset = true;
                ResetResult<TObs> result = PassiveChecker.EnvResetPassiveChecker(Env, seed, options);
                ReplacePreviousData((PassiveChecker.Retain(result.Observation), PassiveChecker.Retain(result.Info)));
                return result;
            }

            return Env.Reset(seed, options);
        }

        /// <summary>Renders the env; the first render runs <see cref="PassiveChecker.EnvRenderPassiveChecker"/>, later ones are forwarded.</summary>
        /// <returns>The env's rendering, unchanged.</returns>
        /// <exception cref="AssertionError">Thrown by the first render's checks for metadata that contradicts the render mode.</exception>
        public override RenderResult Render() {
            if (!CheckedRender) {
                CheckedRender = true;
                return PassiveChecker.EnvRenderPassiveChecker(Env);
            }

            return Env.Render();
        }

        /// <summary>
        ///     Closes the env. Closing again must be allowed: when a later close throws, a warning says so before the error
        ///     propagates.
        /// </summary>
        public override void Close() {
            if (!CloseCalled) {
                CloseCalled = true;
                Env.Close();
                return;
            }

            try {
                Env.Close();
            } catch (Exception) {
                Logger.Warn("Calling `env.close()` on the closed environment should be allowed, but it raised the following exception.");
                throw;
            }
        }

        /// <summary>Releases the retained reuse-check data, then closes and disposes the env as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing) {
                _previousObs?.Dispose();
                _previousInfo?.Dispose();
                _previousObs = null;
                _previousInfo = null;
            }

            base.Dispose(disposing);
        }

        /// <summary>Swaps the retained data, releasing the views of the data it replaces.</summary>
        /// <param name="data">The new data, or <see langword="null"/>.</param>
        private void ReplacePreviousData((RetainedData Obs, RetainedData Info)? data) {
            _previousObs?.Dispose();
            _previousInfo?.Dispose();
            _previousObs = data?.Obs;
            _previousInfo = data?.Info;
        }

        /// <summary>Python's <c>hasattr(env, "action_space")</c> guard: reads a space, turning "no such attribute" into Gymnasium's message.</summary>
        /// <param name="read">Reads the space.</param>
        /// <param name="kind"><c>action</c> or <c>observation</c>.</param>
        /// <returns>The space.</returns>
        /// <exception cref="AttributeError">Thrown when the env has no such space, with Gymnasium's message.</exception>
        private static Space RequireSpace(Func<Space> read, string kind) {
            try {
                return read();
            } catch (AttributeError) {
                throw new AttributeError($"The environment must specify an {kind} space. https://gymnasium.farama.org/introduction/create_custom_env/");
            }
        }
    }

    /// <summary>
    ///     Resets the env automatically, on the step after an episode ends (Gymnasium's vector autoreset order): that step
    ///     ignores its action, resets the env and returns the first observation with reward 0 and both signals
    ///     <see langword="false"/>, so a loop never needs to call reset after the first.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     Since Gymnasium 1.0 the reset happens one step late and the infos carry no <c>final_observation</c> or
    ///     <c>final_info</c>: the terminating step's own result is the episode's last. The reset is unseeded and without
    ///     options. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/common.py#L169">gymnasium/wrappers/common.py</see>.
    /// </remarks>
    public class Autoreset<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env to reset automatically.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        public Autoreset(Env<TObs, TAct> env) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info());
            AutoresetPending = false;
        }

        /// <summary>The recorded constructor arguments, <c>{}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>
        ///     Whether the last step ended the episode, so the next step resets instead (Python's <c>autoreset</c>, renamed in
        ///     C#, where a member cannot share its class's name).
        /// </summary>
        [PyName("autoreset")]
        public bool AutoresetPending { get; set; }

        /// <summary>Clears the pending reset, then resets the env.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            AutoresetPending = false;
            return base.Reset(seed, options);
        }

        /// <summary>Steps the env, or resets it when the previous step ended the episode.</summary>
        /// <param name="action">The action; ignored on a resetting step.</param>
        /// <returns>The env's transition, or the reset's observation and info with reward 0 and no signals.</returns>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> result;
            if (AutoresetPending) {
                (TObs obs, Info info) = Env.Reset();
                result = new StepResult<TObs>(obs, 0.0, false, false, info);
            } else {
                result = Env.Step(action);
            }

            AutoresetPending = result.Terminated || result.Truncated;
            return result;
        }
    }

    /// <summary>
    ///     Records each episode's return, length and duration: the step that ends an episode gets
    ///     <c>info["episode"] = {"r": return, "l": length, "t": seconds}</c>, and the last <c>buffer_length</c> episodes stay
    ///     in <see cref="ReturnQueue"/>, <see cref="LengthQueue"/> and <see cref="TimeQueue"/>.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The statistics go into the info the env returned, which is edited in place. The time is wall-clock seconds since
    ///     the reset (Python's <c>time.perf_counter()</c>, rounded to 6 digits), so it differs between runs: parity and golden
    ///     comparisons leave the <c>t</c> entry out.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/common.py#L454">gymnasium/wrappers/common.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class RecordEpisodeStatistics<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>The info key the statistics go under (Python's <c>_stats_key</c>).</summary>
        [PyName("_stats_key")]
        private readonly string _statsKey;

        /// <summary>Wraps <paramref name="env"/>.</summary>
        /// <param name="env">The env whose episodes to record.</param>
        /// <param name="bufferLength">How many episodes the queues keep (Python's <c>deque(maxlen=buffer_length)</c>).</param>
        /// <param name="statsKey">The info key for the statistics.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for a negative <paramref name="bufferLength"/>: <c>maxlen must be non-negative</c>.</exception>
        public RecordEpisodeStatistics(Env<TObs, TAct> env, int bufferLength = 100, string statsKey = "episode") : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["buffer_length"] = (long)bufferLength, ["stats_key"] = statsKey });

            _statsKey = statsKey;

            EpisodeCount = 0;
            EpisodeStartTime = -1;
            EpisodeReturns = 0.0;
            EpisodeLengths = 0;

            TimeQueue = new Deque<double>(bufferLength);
            ReturnQueue = new Deque<double>(bufferLength);
            LengthQueue = new Deque<long>(bufferLength);
        }

        /// <summary>The recorded constructor arguments, <c>{'buffer_length': 100, 'stats_key': 'episode'}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The number of episodes finished (<c>episode_count</c>).</summary>
        public long EpisodeCount { get; set; }

        /// <summary>When the current episode began, in <c>perf_counter</c> seconds (<c>episode_start_time</c>); -1 before the first reset.</summary>
        public double EpisodeStartTime { get; set; }

        /// <summary>The current episode's return so far (<c>episode_returns</c>).</summary>
        public double EpisodeReturns { get; set; }

        /// <summary>The current episode's length so far (<c>episode_lengths</c>).</summary>
        public long EpisodeLengths { get; set; }

        /// <summary>The durations of the last episodes, oldest first (<c>time_queue</c>).</summary>
        public Deque<double> TimeQueue { get; }

        /// <summary>The returns of the last episodes, oldest first (<c>return_queue</c>).</summary>
        public Deque<double> ReturnQueue { get; }

        /// <summary>The lengths of the last episodes, oldest first (<c>length_queue</c>).</summary>
        public Deque<long> LengthQueue { get; }

        /// <summary>Steps the env and counts the step; the step that ends an episode records it and adds the statistics to the info.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition, its info extended when the episode ended.</returns>
        /// <exception cref="AssertionError">Thrown, empty as Python's bare assert, when the info already has the statistics key.</exception>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> result = base.Step(action);

            EpisodeReturns += result.Reward;
            EpisodeLengths += 1;

            if (result.Terminated || result.Truncated) {
                if (result.Info.ContainsKey(_statsKey))
                    throw new AssertionError("");

                double episodeTimeLength = PyMath.Round(PerfCounter() - EpisodeStartTime, 6);
                result.Info[_statsKey] = new Info {
                    ["r"] = EpisodeReturns,
                    ["l"] = EpisodeLengths,
                    ["t"] = episodeTimeLength
                };

                TimeQueue.Append(episodeTimeLength);
                ReturnQueue.Append(EpisodeReturns);
                LengthQueue.Append(EpisodeLengths);

                EpisodeCount += 1;
                EpisodeStartTime = PerfCounter();
            }

            return result;
        }

        /// <summary>Resets the env and starts a new episode's statistics.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info.</returns>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> result = base.Reset(seed, options);

            EpisodeStartTime = PerfCounter();
            EpisodeReturns = 0.0;
            EpisodeLengths = 0;

            return result;
        }

        /// <summary>Python's <c>time.perf_counter()</c>: a monotonic clock in seconds, meaningful only as a difference.</summary>
        /// <returns>The time.</returns>
        private static double PerfCounter() => System.Diagnostics.Stopwatch.GetTimestamp() / (double)System.Diagnostics.Stopwatch.Frequency;
    }
}
