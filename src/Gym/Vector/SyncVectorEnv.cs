#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     A vector env that steps its sub-envs one after another on the calling thread: the reference implementation of the
    ///     vector API, byte for byte what Gymnasium's <c>SyncVectorEnv</c> returns for the same sub-envs, seeds and actions.
    ///     Typed by the batched types (<see cref="NDArray"/> for array spaces); <see cref="SyncVectorEnv"/> is the untyped one.
    /// </summary>
    /// <typeparam name="TObs">The batched observation type: the element type of the single observation space batched.</typeparam>
    /// <typeparam name="TAct">The batched action type.</typeparam>
    /// <remarks>
    ///     <para>
    ///     The sub-envs are built by the constructor from <see cref="EnvFns"/>, owned by this vector env and closed with it.
    ///     When a sub-env's episode ends, <see cref="AutoresetMode"/> decides: <see cref="Gym.Vector.AutoresetMode.NextStep"/>
    ///     resets it at the next step (whose reward is 0 and whose action is ignored), <see cref="Gym.Vector.AutoresetMode.SameStep"/>
    ///     resets it at once and moves the last observation and info into <c>final_obs</c>/<c>final_info</c>,
    ///     <see cref="Gym.Vector.AutoresetMode.Disabled"/> leaves it to <c>reset(options={"reset_mask": ...})</c>.
    ///     </para>
    ///     <para>
    ///     Ownership: with <see cref="Copy"/> (the default) every observation batch returned is a new deep copy the caller
    ///     owns; without it, the batch is this env's own buffer, overwritten by the next call, which the caller must neither
    ///     keep nor dispose. Rewards and flags are always new arrays. Not thread-safe.
    ///     </para>
    ///     <para>
    ///     Deviations: an options dict's <c>reset_mask</c> is not removed from the caller's dict (Python pops it); the
    ///     sub-envs get the options without it, as in Gymnasium. Gymnasium's <c>np_random</c> and <c>np_random_seed</c> of a
    ///     sync vector env are tuples of the sub-envs' generators and seeds (<c>get_attr</c>); the typed
    ///     <see cref="VectorEnv{TObs, TAct}.NpRandom"/> and <see cref="VectorEnv{TObs, TAct}.NpRandomSeed"/> cannot hold a
    ///     tuple, so they keep the base's meaning, the vector env's own generator and seed, and <c>GetAttr("np_random")</c>
    ///     and <c>GetAttr("np_random_seed")</c> give the sub-envs' values.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/sync_vector_env.py#L34">gymnasium/vector/sync_vector_env.py</see>.</para>
    /// </remarks>
    public class SyncVectorEnv<TObs, TAct> : VectorEnv<TObs, TAct> {
        /// <summary>The single observation space batched, untyped: what the batching functions dispatch on, whatever <typeparamref name="TObs"/> is.</summary>
        private readonly Space _batchedObservationSpace;

        /// <summary>The single action space batched, untyped: what <c>iterate</c> splits the actions with.</summary>
        private readonly Space _batchedActionSpace;

        /// <summary>The sub-envs' last observations, unbatched (Python's <c>_env_obs</c>).</summary>
        private readonly object?[] _envObs;

        /// <summary>The batch the observations are concatenated into (Python's <c>_observations</c>), returned as is when <see cref="Copy"/> is off.</summary>
        private object? _observations;

        /// <summary>The rewards of the last step (Python's <c>_rewards</c>), float64; copied out on every step.</summary>
        private NDArray _rewards;

        /// <summary>The terminations of the last step (Python's <c>_terminations</c>), bool.</summary>
        private NDArray _terminations;

        /// <summary>The truncations of the last step (Python's <c>_truncations</c>), bool.</summary>
        private NDArray _truncations;

        /// <summary>Which sub-envs ended at the last step and reset at the next one (Python's <c>_autoreset_envs</c>), bool.</summary>
        private NDArray _autoresetEnvs;

        /// <summary>Builds the sub-envs and the batched spaces (<c>SyncVectorEnv(env_fns, copy, observation_mode, autoreset_mode)</c>).</summary>
        /// <param name="envFns">One function per sub-env, called once each, in order; the vector env owns what they return.</param>
        /// <param name="copy">Whether every returned observation batch is a deep copy (the default) or the env's own buffer.</param>
        /// <param name="observationMode">
        ///     <c>"same"</c> (<see langword="null"/>, the default): every sub-env has the first's observation space, batched;
        ///     <c>"different"</c>: the spaces may differ but share shape and dtype, batched with <c>batch_differing_spaces</c>;
        ///     or a tuple (<see langword="object"/>?[] or a value tuple) of the batched and the single space to use.
        /// </param>
        /// <param name="autoresetMode">An <see cref="Gym.Vector.AutoresetMode"/> or its value as a string; <see langword="null"/> is <see cref="Gym.Vector.AutoresetMode.NextStep"/>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envFns"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for an unknown autoreset mode or observation mode, with Gymnasium's message.</exception>
        /// <exception cref="IndexError">Thrown when there are no sub-envs (<c>list index out of range</c>).</exception>
        /// <exception cref="TypeError">Thrown when a tuple observation mode holds something else than spaces, or a batched space's elements are not <typeparamref name="TObs"/>/<typeparamref name="TAct"/>.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not match the first's, with Gymnasium's message.</exception>
        public SyncVectorEnv(IEnumerable<Func<IEnv>> envFns, bool copy = true, object? observationMode = null, object? autoresetMode = null) {
            if (envFns is null)
                throw new ArgumentNullException(nameof(envFns));
            EnvFns = envFns.ToArray();
            Copy = copy;
            ObservationMode = observationMode ?? "same";
            AutoresetMode = Gym.Vector.AutoresetMode.From(autoresetMode);

            // Initialise all sub-environments
            var envs = new List<IEnv>(EnvFns.Count);
            try {
                foreach (Func<IEnv> envFn in EnvFns)
                    envs.Add(envFn());
                if (envs.Count == 0)
                    throw new IndexError("list index out of range");

                // Define core attributes using the sub-environments.
                Envs = envs;
                NumEnvs = envs.Count;
                // Copy the metadata such that we don't modify the sub-environment's metadata.
                Metadata = new Metadata(envs[0].Metadata) { ["autoreset_mode"] = AutoresetMode };
                RenderMode = envs[0].RenderMode;
                SingleActionSpace = envs[0].ActionSpace;
                _batchedActionSpace = VectorUtils.BatchSpace(SingleActionSpace, NumEnvs);
                ActionSpace = Typed<TAct>(_batchedActionSpace, "action_space");

                (Space batched, Space single) = ObservationSpaces(ObservationMode, envs);
                _batchedObservationSpace = batched;
                SingleObservationSpace = single;
                ObservationSpace = Typed<TObs>(batched, "observation_space");

                CheckSubEnvSpaces(envs);
            } catch {
                // Nothing else holds the sub-envs yet: release the ones built before the error.
                foreach (IEnv env in envs)
                    env.Dispose();
                throw;
            }

            // Initialise attributes used in `step` and `reset`.
            _envObs = new object?[NumEnvs];
            _observations = VectorUtils.CreateEmptyArray(SingleObservationSpace, NumEnvs, (shape, dtype) => np.zeros(shape, dtype));
            _rewards = np.zeros(new Shape(NumEnvs), NPTypeCode.Double);
            _terminations = np.zeros(new Shape(NumEnvs), NPTypeCode.Boolean);
            _truncations = np.zeros(new Shape(NumEnvs), NPTypeCode.Boolean);
            _autoresetEnvs = np.zeros(new Shape(NumEnvs), NPTypeCode.Boolean);
        }

        /// <summary>The functions the sub-envs were built from (<c>env_fns</c>).</summary>
        public IReadOnlyList<Func<IEnv>> EnvFns { get; }

        /// <summary>Whether returned observation batches are deep copies (<c>copy</c>).</summary>
        public bool Copy { get; }

        /// <summary>The observation mode given (<c>observation_mode</c>): <c>"same"</c>, <c>"different"</c>, or the tuple of spaces.</summary>
        public object ObservationMode { get; }

        /// <summary>When ended sub-envs reset (<c>autoreset_mode</c>), also in <c>metadata["autoreset_mode"]</c>.</summary>
        public AutoresetMode AutoresetMode { get; }

        /// <summary>The sub-envs (<c>envs</c>), in order; owned by this vector env.</summary>
        public IReadOnlyList<IEnv> Envs { get; } = Array.Empty<IEnv>();

        /// <summary>
        ///     Resets the sub-envs (<c>reset(seed=seed, options=options)</c>): every one, or with <c>options["reset_mask"]</c>
        ///     (a <c>(num_envs,)</c> bool array) only the masked ones. An integer seed seeds sub-env <c>i</c> with <c>seed + i</c>.
        /// </summary>
        /// <param name="seed">The first sub-env's seed, or <see langword="null"/> to keep every generator.</param>
        /// <param name="options">The options passed to every reset sub-env, without <c>reset_mask</c>.</param>
        /// <returns>The batched observations (a copy unless <see cref="Copy"/> is off) and the merged reset infos.</returns>
        /// <exception cref="TypeError">Thrown when <c>reset_mask</c> is not an array or not bool.</exception>
        /// <exception cref="ValueError">Thrown when <c>reset_mask</c> has another shape or no <see langword="true"/>.</exception>
        /// <exception cref="Error">Thrown for a negative seed, by the sub-env.</exception>
        public override VectorResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            var seeds = new BigInteger?[NumEnvs];
            for (int i = 0; i < seeds.Length; i++)
                seeds[i] = seed is null ? null : seed.Value + i;
            return ResetCore(seeds, options);
        }

        /// <summary>Resets the sub-envs with a seed each (<c>reset(seed=[...])</c>); see <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/>.</summary>
        /// <param name="seeds">One seed per sub-env, <see langword="null"/> keeping that sub-env's generator.</param>
        /// <param name="options">The options passed to every reset sub-env, without <c>reset_mask</c>.</param>
        /// <returns>The batched observations and the merged reset infos.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="seeds"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown when there is not one seed per sub-env, with Gymnasium's message, or for a bad <c>reset_mask</c>.</exception>
        /// <exception cref="TypeError">Thrown for a <c>reset_mask</c> that is not a bool array.</exception>
        public override VectorResetResult<TObs> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null) {
            if (seeds is null)
                throw new ArgumentNullException(nameof(seeds));
            if (seeds.Count != NumEnvs)
                throw new ValueError($"If seeds are passed as a list the length must match num_envs={NumEnvs} but got length={seeds.Count}.");
            return ResetCore(seeds, options);
        }

        /// <summary>
        ///     Steps every sub-env with its action (<c>step(actions)</c>), applying <see cref="AutoresetMode"/> to the ones whose
        ///     episode ended at the previous step (<c>NEXT_STEP</c>) or at this one (<c>SAME_STEP</c>).
        /// </summary>
        /// <param name="actions">The batched actions, one per sub-env, split by the batched action space (<c>iterate</c>).</param>
        /// <returns>The batched observations (a copy unless <see cref="Copy"/> is off), new reward and flag arrays, and the merged infos.</returns>
        /// <exception cref="ValueError">Thrown when the actions are not one per sub-env (Python's <c>zip(strict=True)</c> text), after the paired sub-envs stepped.</exception>
        /// <exception cref="AssertionError">Thrown in <c>DISABLED</c> mode when an ended sub-env is stepped without a reset.</exception>
        public override VectorStepResult<TObs> Step(TAct actions) {
            IEnumerator<object?> actionsIter = VectorUtils.Iterate(_batchedActionSpace, actions).GetEnumerator();
            var infos = new Info();
            try {
                int i = 0;
                for (; i < NumEnvs; i++) {
                    if (!actionsIter.MoveNext())
                        throw new ValueError("zip() argument 2 is longer than argument 1");
                    StepOne(i, actionsIter.Current, infos);
                }

                if (actionsIter.MoveNext())
                    throw new ValueError("zip() argument 2 is shorter than argument 1");
            } finally {
                actionsIter.Dispose();
            }

            // Concatenate the observations.
            _observations = VectorUtils.Concatenate(SingleObservationSpace, _envObs, _observations);
            NDArray autoresetEnvs = np.logical_or(_terminations, _truncations);
            _autoresetEnvs.Dispose();
            _autoresetEnvs = autoresetEnvs;
            return new VectorStepResult<TObs>(ObservationsOut(), _rewards.copy(), _terminations.copy(), _truncations.copy(), infos);
        }

        /// <summary>Renders every sub-env (<c>render()</c>), in order.</summary>
        /// <returns>One result per sub-env, as each renders by its own mode.</returns>
        public override IReadOnlyList<RenderResult>? Render() => Envs.Select(env => env.Render()).ToArray();

        /// <summary>
        ///     Calls a sub-env member by its Python name on every sub-env (<c>call(name, *args)</c>), through the wrappers as
        ///     <c>get_wrapper_attr</c> finds it: an attribute's value (a delegate one is called with the arguments), or a method,
        ///     called with the arguments in order.
        /// </summary>
        /// <param name="name">The Python name: <c>render</c>, <c>np_random</c>, a run-time attribute.</param>
        /// <param name="args">The positional arguments for a callable member.</param>
        /// <returns>One result per sub-env.</returns>
        /// <exception cref="AttributeError">Thrown when a sub-env has no such attribute or method.</exception>
        /// <exception cref="TypeError">Thrown when no method of that name takes the arguments given.</exception>
        public IReadOnlyList<object?> Call(string name, params object?[] args) {
            var results = new object?[NumEnvs];
            for (int i = 0; i < Envs.Count; i++)
                results[i] = CallOne(Envs[i], name, args ?? Array.Empty<object?>());
            return results;
        }

        /// <summary>Reads an attribute of every sub-env (<c>get_attr(name)</c>), as <see cref="Call"/> does without arguments.</summary>
        /// <param name="name">The Python name.</param>
        /// <returns>One value per sub-env.</returns>
        /// <exception cref="AttributeError">Thrown when a sub-env has no such attribute.</exception>
        public IReadOnlyList<object?> GetAttr(string name) => Call(name);

        /// <summary>
        ///     Sets an attribute of every sub-env (<c>set_attr(name, values)</c>) through its wrappers
        ///     (<c>set_wrapper_attr</c>): a list or tuple gives one value per sub-env, anything else is set on all of them.
        /// </summary>
        /// <param name="name">The Python name.</param>
        /// <param name="values">A list or <see langword="object"/>?[] of one value per sub-env, or one value for all (an array is one value).</param>
        /// <exception cref="ValueError">Thrown when a list's length is not the number of sub-envs, with Gymnasium's message.</exception>
        public void SetAttr(string name, object? values) {
            IList perEnv;
            if (values is IList list and not NDArray) {
                perEnv = list;
            } else {
                var repeated = new object?[NumEnvs];
                Array.Fill(repeated, values);
                perEnv = repeated;
            }

            if (perEnv.Count != NumEnvs)
                throw new ValueError($"Values must be a list or tuple with length equal to the number of environments. Got `{perEnv.Count}` values for {NumEnvs} environments.");
            for (int i = 0; i < Envs.Count; i++)
                Envs[i].SetWrapperAttr(name, perEnv[i]);
        }

        /// <summary>Closes every sub-env (<c>close_extras</c>).</summary>
        protected override void CloseExtras() {
            foreach (IEnv env in Envs)
                env.Close();
        }

        /// <summary>Closes, then disposes the sub-envs (which this vector env built and owns) and its buffers.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="VectorEnv{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            base.Dispose(disposing);
            if (!disposing)
                return;
            foreach (IEnv env in Envs)
                env.Dispose();
            _batchedObservationSpace.Dispose();
            _batchedActionSpace.Dispose();
            Release(_observations);
            _rewards.Dispose();
            _terminations.Dispose();
            _truncations.Dispose();
            _autoresetEnvs.Dispose();
        }

        /// <summary>The reset of <c>reset</c> once the seeds are one per sub-env: the masked or full reset, then the batch.</summary>
        /// <param name="seeds">One seed per sub-env.</param>
        /// <param name="options">The options, possibly with <c>reset_mask</c>.</param>
        /// <returns>The batched observations and the merged infos.</returns>
        /// <exception cref="TypeError">Thrown when <c>reset_mask</c> is not an array or not bool.</exception>
        /// <exception cref="ValueError">Thrown when <c>reset_mask</c> has another shape or no <see langword="true"/>.</exception>
        private VectorResetResult<TObs> ResetCore(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options) {
            var infos = new Info();
            if (options is not null && options.TryGetValue("reset_mask", out object? maskValue)) {
                // The check hands back the caller's own mask, which the caller keeps and disposes.
#pragma warning disable NDW012
                NDArray resetMask = CheckResetMask(maskValue);
#pragma warning restore NDW012
                // The sub-envs get the options without the mask, as after Python's options.pop("reset_mask").
                var subOptions = new Info(options.Where(entry => entry.Key != "reset_mask"));
                for (int i = 0; i < NumEnvs; i++) {
                    if (!resetMask.GetBoolean(i))
                        continue;
                    _terminations.SetValue(false, i);
                    _truncations.SetValue(false, i);
                    _autoresetEnvs.SetValue(false, i);
                }

                for (int i = 0; i < NumEnvs; i++) {
                    if (!resetMask.GetBoolean(i))
                        continue;
                    (object? observation, Info envInfo) = Envs[i].Reset(seeds[i], subOptions);
                    _envObs[i] = observation;
                    infos = AddInfo(infos, envInfo, i);
                }
            } else {
                ReplaceFlags();
                for (int i = 0; i < NumEnvs; i++) {
                    (object? observation, Info envInfo) = Envs[i].Reset(seeds[i], options);
                    _envObs[i] = observation;
                    infos = AddInfo(infos, envInfo, i);
                }
            }

            // Concatenate the observations.
            _observations = VectorUtils.Concatenate(SingleObservationSpace, _envObs, _observations);
            return new VectorResetResult<TObs>(ObservationsOut(), infos);
        }

        /// <summary>Steps (or autoresets) sub-env <paramref name="i"/> by the autoreset mode and merges its info.</summary>
        /// <param name="i">The sub-env.</param>
        /// <param name="action">Its action.</param>
        /// <param name="infos">The infos merged so far.</param>
        /// <exception cref="AssertionError">Thrown in <c>DISABLED</c> mode for an ended sub-env.</exception>
        private void StepOne(int i, object? action, Info infos) {
            IEnv env = Envs[i];
            Info envInfo;
            if (ReferenceEquals(AutoresetMode, Gym.Vector.AutoresetMode.NextStep)) {
                if (_autoresetEnvs.GetBoolean(i)) {
                    // The episode ended at the previous step: reset now, ignoring the action, with a reward of 0.
                    (object? observation, Info resetInfo) = env.Reset();
                    _envObs[i] = observation;
                    envInfo = resetInfo;
                    _rewards.SetValue(0.0, i);
                    _terminations.SetValue(false, i);
                    _truncations.SetValue(false, i);
                } else {
                    envInfo = Store(i, env.Step(action));
                }
            } else if (ReferenceEquals(AutoresetMode, Gym.Vector.AutoresetMode.Disabled)) {
                // assumes that the user has correctly autoreset
                if (_autoresetEnvs.GetBoolean(i))
                    throw new AssertionError($"self._autoreset_envs={PyBuiltins.Repr(_autoresetEnvs)}");
                envInfo = Store(i, env.Step(action));
            } else {
                envInfo = Store(i, env.Step(action));
                if (_terminations.GetBoolean(i) || _truncations.GetBoolean(i)) {
                    // SAME_STEP: the ended episode's last observation and info move into final_obs/final_info, and the
                    // sub-env starts its next episode within this step.
                    AddInfo(infos, new Info { ["final_obs"] = _envObs[i], ["final_info"] = envInfo }, i);
                    (object? observation, Info resetInfo) = env.Reset();
                    _envObs[i] = observation;
                    envInfo = resetInfo;
                }
            }

            AddInfo(infos, envInfo, i);
        }

        /// <summary>Stores a sub-env's transition in this step's slots (<c>(self._env_obs[i], self._rewards[i], ...) = env.step(action)</c>).</summary>
        /// <param name="i">The sub-env.</param>
        /// <param name="result">Its transition.</param>
        /// <returns>Its info.</returns>
        private Info Store(int i, Observations.StepResult<object?> result) {
            _envObs[i] = result.Observation;
            _rewards.SetValue(result.Reward, i);
            _terminations.SetValue(result.Terminated, i);
            _truncations.SetValue(result.Truncated, i);
            return result.Info;
        }

        /// <summary>The observation batch handed out: a deep copy the caller owns, or the buffer itself without <see cref="Copy"/>.</summary>
        /// <returns>The batch.</returns>
        private TObs ObservationsOut() {
            object? batch = Copy ? PyCopy.DeepCopy(_observations) : _observations;
            return (TObs)batch!;
        }

        /// <summary>A full reset's fresh, all-False flag arrays (Python assigns new arrays), releasing the old ones.</summary>
        private void ReplaceFlags() {
            _terminations.Dispose();
            _truncations.Dispose();
            _autoresetEnvs.Dispose();
            _terminations = np.zeros(new Shape(NumEnvs), NPTypeCode.Boolean);
            _truncations = np.zeros(new Shape(NumEnvs), NPTypeCode.Boolean);
            _autoresetEnvs = np.zeros(new Shape(NumEnvs), NPTypeCode.Boolean);
        }

        /// <summary>Gymnasium's checks of <c>options["reset_mask"]</c>: an array, of shape <c>(num_envs,)</c>, of bool, with a <see langword="true"/>.</summary>
        /// <param name="value">The mask given.</param>
        /// <returns>The mask.</returns>
        /// <exception cref="TypeError">Thrown when it is not an array, or not bool.</exception>
        /// <exception cref="ValueError">Thrown when its shape is wrong or nothing is masked.</exception>
        private NDArray CheckResetMask(object? value) {
            if (value is not NDArray resetMask)
                throw new TypeError($"`options['reset_mask']` must be a numpy array, got {PyBuiltins.TypeRepr(value)}");
            if (resetMask.ndim != 1 || resetMask.shape[0] != NumEnvs)
                throw new ValueError($"`options['reset_mask']` must have shape `({NumEnvs},)`, got {PyBuiltins.Repr(ShapeTuple(resetMask))}");
            if (resetMask.typecode != NPTypeCode.Boolean)
                throw new TypeError($"`options['reset_mask']` must have `dtype=np.bool_`, got {resetMask.dtype.name}");
            if (!np.any(resetMask))
                throw new ValueError($"`options['reset_mask']` must contain a boolean array with at least one True value, got reset_mask={PyBuiltins.Str(resetMask)}");
            return resetMask;
        }

        /// <summary>
        ///     The batched and single observation spaces of an observation mode: the first sub-env's space batched
        ///     (<c>"same"</c>), the sub-envs' spaces batched together (<c>"different"</c>), or the pair given.
        /// </summary>
        /// <param name="mode">The mode.</param>
        /// <param name="envs">The sub-envs.</param>
        /// <returns>The batched space and the single one.</returns>
        /// <exception cref="TypeError">Thrown when a pair holds something else than spaces.</exception>
        /// <exception cref="ValueError">Thrown for any other mode.</exception>
        private static (Space Batched, Space Single) ObservationSpaces(object mode, IReadOnlyList<IEnv> envs) {
            object?[]? pair = mode switch {
                object?[] array when array.Length == 2 => array,
                ITuple { Length: 2 } tuple => new[] { tuple[0], tuple[1] },
                _ => null,
            };
            if (pair is not null) {
                if (pair[0] is not Space batched || pair[1] is not Space single)
                    throw new TypeError($"Expected both elements of observation_mode to be Spaces, got {PyBuiltins.TypeRepr(pair[0])} and {PyBuiltins.TypeRepr(pair[1])}");
                // Python unpacks the pair as (observation_space, single_observation_space).
                return (batched, single);
            }

            if (mode is "same")
                return (VectorUtils.BatchSpace(envs[0].ObservationSpace, envs.Count), envs[0].ObservationSpace);
            if (mode is "different")
                return (VectorUtils.BatchDifferingSpaces(envs.Select(env => env.ObservationSpace).ToArray()), envs[0].ObservationSpace);
            throw new ValueError($"Invalid `observation_mode`, expected: 'same' or 'different' or tuple of single and batch observation space, actual got {PyBuiltins.Str(mode)}");
        }

        /// <summary>Gymnasium's check that every sub-env's spaces fit the batch, by the observation mode.</summary>
        /// <param name="envs">The sub-envs.</param>
        /// <exception cref="RuntimeError">Thrown at the first sub-env that does not fit, with Gymnasium's message.</exception>
        private void CheckSubEnvSpaces(IReadOnlyList<IEnv> envs) {
            foreach (IEnv env in envs) {
                if (ObservationMode is "same") {
                    if (!env.ObservationSpace.Equals(SingleObservationSpace))
                        throw new RuntimeError("SyncVectorEnv(..., observation_mode='same') however the sub-environments observation spaces are not equivalent. "
                                               + $"single_observation_space={SingleObservationSpace}, sub-environment observation_space={env.ObservationSpace}. "
                                               + "If this is intentional, use `observation_mode='different'` instead.");
                } else if (!SpaceUtils.IsSpaceDtypeShapeEquiv(env.ObservationSpace, SingleObservationSpace)) {
                    throw new RuntimeError("SyncVectorEnv(..., observation_mode='different' or custom space) however the sub-environments observation spaces do not share a common shape and dtype, "
                                           + $"single_observation_space={SingleObservationSpace}, sub-environment observation space={env.ObservationSpace}");
                }

                if (!env.ActionSpace.Equals(SingleActionSpace))
                    throw new RuntimeError($"Sub-environment action space doesn't make the `single_action_space`, action_space={env.ActionSpace}, single_action_space={SingleActionSpace}");
            }
        }

        /// <summary>
        ///     One sub-env's <c>call</c>: an attribute through the wrappers (called when it is a delegate), else a public method
        ///     whose Python name is <paramref name="name"/>, on the outermost object as Python's bound method would be.
        /// </summary>
        /// <param name="env">The sub-env.</param>
        /// <param name="name">The Python name.</param>
        /// <param name="args">The positional arguments.</param>
        /// <returns>The value or the call's result.</returns>
        /// <exception cref="AttributeError">Thrown when there is no such attribute or method.</exception>
        /// <exception cref="TypeError">Thrown when no method of that name takes the arguments.</exception>
        private static object? CallOne(IEnv env, string name, object?[] args) {
            if (env.HasWrapperAttr(name)) {
                object? function = env.GetWrapperAttr(name);
                return function is Delegate callable ? callable.DynamicInvoke(args) : function;
            }

            MethodInfo[] methods = env.GetType().GetMethods(BindingFlags.Public | BindingFlags.Instance)
                .Where(method => PyAttributes.SnakeCase(method.Name) == name && !method.IsGenericMethodDefinition).ToArray();
            if (methods.Length == 0)
                // get_wrapper_attr's own error, from the innermost env.
                return env.GetWrapperAttr(name);
            foreach (MethodInfo method in methods) {
                ParameterInfo[] parameters = method.GetParameters();
                if (args.Length > parameters.Length || parameters.Skip(args.Length).Any(p => !p.IsOptional))
                    continue;
                object?[] arguments = parameters.Select((p, i) => i < args.Length ? args[i] : p.DefaultValue).ToArray();
                try {
                    return method.Invoke(env, arguments);
                } catch (TargetInvocationException e) when (e.InnerException is not null) {
                    // Reflection wraps what the method threw; the caller sees the method's own exception.
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(e.InnerException).Throw();
                    throw;
                }
            }

            throw new TypeError($"{PyAttributes.PyTypeName(env.GetType())}.{name}() takes no {args.Length} positional arguments");
        }

        /// <summary>A batched space as the typed property needs it: as it is, or viewed as <see langword="object"/> elements for the untyped env.</summary>
        /// <typeparam name="T">The batched element type.</typeparam>
        /// <param name="space">The batched space.</param>
        /// <param name="attribute">The Python attribute, for the error.</param>
        /// <returns>The typed space.</returns>
        /// <exception cref="TypeError">Thrown when the space's elements are not <typeparamref name="T"/>.</exception>
        private Space<T> Typed<T>(Space space, string attribute) {
            if (space is Space<T> typed)
                return typed;
            if (typeof(T) == typeof(object))
                return (Space<T>)(object)new ObjectSpace(space);
            throw new TypeError($"{PyAttributes.PyTypeName(GetType())}<{typeof(TObs).Name}, {typeof(TAct).Name}> needs an {attribute} of {typeof(T).Name} elements, but batching gives {space}; use the untyped SyncVectorEnv or the batched types");
        }

        /// <summary>An array's shape as Python's tuple, for messages.</summary>
        /// <param name="array">The array.</param>
        /// <returns>The shape as <see langword="object"/>?[] of longs.</returns>
        private static object?[] ShapeTuple(NDArray array) => array.shape.Select(dim => (object?)dim).ToArray();

        /// <summary>Disposes the arrays of an internal batch: an array, or the arrays inside a tuple or dict batch.</summary>
        /// <param name="value">The batch.</param>
        private static void Release(object? value) {
            switch (value) {
                case NDArray array:
                    array.Dispose();
                    break;
                case object?[] items:
                    foreach (object? item in items)
                        Release(item);
                    break;
                case Info dict:
                    foreach (KeyValuePair<string, object?> entry in dict)
                        Release(entry.Value);
                    break;
            }
        }
    }

    /// <summary>
    ///     <see cref="SyncVectorEnv{TObs, TAct}"/> untyped: observations and actions are <see langword="object"/>s, whatever
    ///     the spaces, as Gymnasium's own. The untyped <c>MakeVec</c> builds this one.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/sync_vector_env.py#L34">gymnasium/vector/sync_vector_env.py</see>.</remarks>
    public class SyncVectorEnv : SyncVectorEnv<object?, object?> {
        /// <summary>Builds the sub-envs and the batched spaces; see <see cref="SyncVectorEnv{TObs, TAct}"/>'s constructor.</summary>
        /// <param name="envFns">One function per sub-env, called once each, in order.</param>
        /// <param name="copy">Whether every returned observation batch is a deep copy.</param>
        /// <param name="observationMode"><c>"same"</c> (<see langword="null"/>), <c>"different"</c>, or the pair of batched and single spaces.</param>
        /// <param name="autoresetMode">An <see cref="AutoresetMode"/> or its value; <see langword="null"/> is <c>NEXT_STEP</c>.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="envFns"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for an unknown autoreset or observation mode.</exception>
        /// <exception cref="IndexError">Thrown when there are no sub-envs.</exception>
        /// <exception cref="TypeError">Thrown when a pair observation mode holds something else than spaces.</exception>
        /// <exception cref="RuntimeError">Thrown when a sub-env's spaces do not fit.</exception>
        public SyncVectorEnv(IEnumerable<Func<IEnv>> envFns, bool copy = true, object? observationMode = null, object? autoresetMode = null)
            : base(envFns, copy, observationMode, autoresetMode) { }
    }
}
