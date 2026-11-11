#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     The base class of every environment: an MDP driven through <see cref="Step"/> and <see cref="Reset"/>, with spaces,
    ///     metadata, a render mode, a lazily created generator and the attribute protocol (<see cref="GetWrapperAttr"/>, ...).
    /// </summary>
    /// <typeparam name="TObs">
    ///     The observation type: <see cref="NDArray"/> for array spaces, <c>Discrete</c> included (its elements are 0-d
    ///     int64 arrays, DR-08); a tuple type for a composite space (Blackjack's <c>object?[]</c>, DR-13).
    /// </typeparam>
    /// <typeparam name="TAct">The action type, likewise: every built-in env takes <see cref="NDArray"/> actions.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Implementing an env: set <see cref="ActionSpace"/>, <see cref="ObservationSpace"/> and, when it renders,
    ///     <see cref="Metadata"/> and <see cref="RenderMode"/> in the constructor; begin <see cref="Reset"/> with
    ///     <see cref="SeedNpRandom"/> (Gymnasium's <c>super().reset(seed=seed)</c>), and draw every random number from
    ///     <see cref="NpRandom"/>. Skipping the seeding call breaks reproducibility, which every parity test relies on.
    ///     </para>
    ///     <para>
    ///     Python attributes map to properties (port plan §3.8), which Gymnasium's attribute functions reach by their Python
    ///     names through <see cref="PyAttributes"/>: <c>np_random_seed</c> is <see cref="NpRandomSeed"/>, and non-public
    ///     state carries <see cref="PyNameAttribute"/> (<c>_np_random</c>). Like a Python instance, an env also holds
    ///     attributes created at run time (<c>set_wrapper_attr(name, value, force=True)</c>, <c>env.x = 1</c> through
    ///     <see cref="PyAttributes.SetAttr"/>).
    ///     </para>
    ///     <para>
    ///     A deviation that follows from C#: <see cref="Metadata"/> is per instance (Python's class attribute is shared by
    ///     every instance of the class, so mutating one env's changed all of them). The spaces are typed by the env's element
    ///     types, <see cref="Space{T}"/> of <typeparamref name="TAct"/> and <typeparamref name="TObs"/> (DR-07).
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L28">gymnasium/core.py</see>.</para>
    /// </remarks>
    public abstract class Env<TObs, TAct> : IEnv, IDynamicAttributes, IPythonAttributePresence {
        /// <summary>The metadata; a fresh <c>{"render_modes": []}</c> until the env assigns its own.</summary>
        private Metadata _metadata = new() { [Metadata.RenderModesKey] = new List<string>() };

        /// <summary>The action space, <see langword="null"/> while unset (reading it then raises <see cref="AttributeError"/>).</summary>
        private Space<TAct>? _actionSpace;

        /// <summary>The observation space, <see langword="null"/> while unset.</summary>
        private Space<TObs>? _observationSpace;

        /// <summary>Python's <c>_np_random</c>: <see langword="null"/> until first used or seeded.</summary>
        private Generator? _npRandom;

        /// <summary>Python's <c>_np_random_seed</c>: <see langword="null"/> until first used or seeded, -1 once a generator was assigned.</summary>
        private BigInteger? _npRandomSeed;

        /// <summary>The attributes created at run time, allocated on first use.</summary>
        private Info? _dynamicAttributes;

        /// <summary>
        ///     The class-level metadata, Python's class attribute <c>metadata</c> that <c>make</c> reads before the env exists
        ///     to decide on <c>HumanRendering</c> or <c>RenderCollection</c>: <c>{"render_modes": []}</c> here, as on
        ///     <c>gymnasium.Env</c>. An env class that renders declares its own with <c>new</c>
        ///     (<c>public new static Metadata ClassMetadata =&gt; ...</c>) and sets its instance <see cref="Metadata"/> from it.
        /// </summary>
        /// <remarks>
        ///     A new dict on every read: Python shares one dict between the class and its instances, so editing it there edits
        ///     every env; here each reader gets its own.
        /// </remarks>
        public static Metadata ClassMetadata => new() { [Metadata.RenderModesKey] = new List<string>() };

        /// <summary>
        ///     The metadata (<c>metadata</c>), with Python key spelling: <c>{"render_modes": ["human", "rgb_array"],
        ///     "render_fps": 50}</c>. Defaults to <c>{"render_modes": []}</c>, as <c>Env.metadata</c> does.
        /// </summary>
        /// <exception cref="ArgumentNullException">Thrown when assigning <see langword="null"/>.</exception>
        public virtual Metadata Metadata {
            get => _metadata;
            set => _metadata = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>
        ///     The render mode given to the constructor (<c>render_mode</c>), <see langword="null"/> (the default) when the env
        ///     should not render. Fixed for the env's life: rendering another way means building another env.
        /// </summary>
        public virtual string? RenderMode { get; protected set; }

        /// <summary>The spec <c>make</c> attached (<c>spec</c>), or <see langword="null"/> (the default) for an env built with its constructor.</summary>
        public virtual EnvSpec? Spec { get; set; }

        /// <summary>
        ///     The space every action must belong to (<c>action_space</c>), typed by the action type (DR-07): a
        ///     <see cref="Discrete"/> or <see cref="Box"/> for <see cref="NDArray"/> actions. The env sets it in its constructor
        ///     and owns it: <see cref="Dispose()"/> disposes it, so a space must not be shared between envs, and the setter
        ///     refuses one that another env not yet disposed owns (DR-40). Replacing it lets the old space go undisposed, as in
        ///     Python, where whoever still holds it keeps using it.
        /// </summary>
        /// <exception cref="AttributeError">
        ///     Thrown by the getter while unset, as reading Python's annotated-only attribute does
        ///     (<c>'GenericTestEnv' object has no attribute 'action_space'</c>).
        /// </exception>
        /// <exception cref="InvalidOperationException">Thrown by the setter for a space another env not yet disposed owns; the message names the fix (a new space, or <c>DeepCopy()</c>).</exception>
        public virtual Space<TAct> ActionSpace {
            get => _actionSpace ?? throw MissingAttribute("action_space");
            set => _actionSpace = Space.Assign(this, value, _actionSpace, _observationSpace, "action_space");
        }

        /// <summary>
        ///     The space every observation belongs to (<c>observation_space</c>), typed by the observation type. The env sets it
        ///     in its constructor and owns it, like <see cref="ActionSpace"/>, the same refusal included.
        /// </summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="InvalidOperationException">Thrown by the setter for a space another env not yet disposed owns.</exception>
        public virtual Space<TObs> ObservationSpace {
            get => _observationSpace ?? throw MissingAttribute("observation_space");
            set => _observationSpace = Space.Assign(this, value, _observationSpace, _actionSpace, "observation_space");
        }

        /// <summary>The action space, untyped, for code that handles any env (<see cref="IEnv"/>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="TypeError">Thrown by the setter for a space whose elements are not <typeparamref name="TAct"/>.</exception>
        Space IEnv.ActionSpace {
            get => ActionSpace;
            set => ActionSpace = Typed<TAct>(value, "action_space");
        }

        /// <summary>The observation space, untyped, for code that handles any env.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        /// <exception cref="TypeError">Thrown by the setter for a space whose elements are not <typeparamref name="TObs"/>.</exception>
        Space IEnv.ObservationSpace {
            get => ObservationSpace;
            set => ObservationSpace = Typed<TObs>(value, "observation_space");
        }

        /// <summary>
        ///     The env's generator (<c>np_random</c>). The first read creates one from OS entropy, as Gymnasium does, which is
        ///     why an unseeded env still draws reproducibly within itself; a seeded <see cref="Reset"/> replaces it. Assigning a
        ///     generator shares it deliberately and makes <see cref="NpRandomSeed"/> -1, because a generator's seed cannot be
        ///     recovered.
        /// </summary>
        public virtual Generator NpRandom {
            get {
                if (_npRandom is null)
                    (_npRandom, _npRandomSeed) = Seeding.NpRandom();
                return _npRandom;
            }
            set {
                _npRandom = value;
                // Setting a numpy rng with -1 would cause a ValueError, which marks the seed as unknown.
                _npRandomSeed = -1;
            }
        }

        /// <summary>
        ///     The seed of <see cref="NpRandom"/> (<c>np_random_seed</c>): the seed of the last seeded reset, the 128-bit
        ///     entropy of an entropy-seeded generator (created by this read if none exists yet), or -1 after a generator was
        ///     assigned. Read-only: Python raises <see cref="AttributeError"/> on assignment, and
        ///     <see cref="PyAttributes.SetAttr"/> does too.
        /// </summary>
        public virtual BigInteger NpRandomSeed {
            get {
                if (_npRandomSeed is null)
                    (_npRandom, _npRandomSeed) = Seeding.NpRandom();
                return _npRandomSeed.Value;
            }
        }

        /// <summary>The innermost env (<c>unwrapped</c>): the env itself; wrappers override it to walk down their stack.</summary>
        public virtual IEnv Unwrapped => this;

        /// <summary>
        ///     Python's <c>_np_random</c>: the generator as it is, <see langword="null"/> before first use, without creating one.
        ///     Gymnasium's tests read it to check laziness; a wrapper raises <see cref="AttributeError"/> instead.
        /// </summary>
        [PyName("_np_random")]
        protected virtual Generator? RawNpRandom {
            get => _npRandom;
            set => _npRandom = value;
        }

        /// <summary>Python's <c>_np_random_seed</c>: the recorded seed as it is, <see langword="null"/> before first use.</summary>
        [PyName("_np_random_seed")]
        protected virtual BigInteger? RawNpRandomSeed {
            get => _npRandomSeed;
            set => _npRandomSeed = value;
        }

        /// <summary>The run-time attributes, Python's instance <c>__dict__</c> entries the class does not declare.</summary>
        IDictionary<string, object?> IDynamicAttributes.DynamicAttributes => _dynamicAttributes ??= new Info();

        /// <summary>
        ///     Runs one timestep of the dynamics with <paramref name="action"/> (<c>step(action)</c>). Once the result is
        ///     terminated or truncated, call <see cref="Reset"/> before stepping again.
        /// </summary>
        /// <param name="action">An element of <see cref="ActionSpace"/>.</param>
        /// <returns>The observation, reward, <c>terminated</c>, <c>truncated</c> and info.</returns>
        /// <exception cref="AssertionError">Thrown by Gym's envs for an invalid action or a step before the first reset, where Gymnasium's envs fail an <c>assert</c>.</exception>
        /// <remarks>
        ///     Every call must return a new observation and a new info, never a buffer an earlier call returned and this one
        ///     mutated: callers commonly keep every timestep (a replay or rollout buffer), and a reused buffer silently
        ///     rewrites the history they kept. Gymnasium's <c>check_env</c> and <c>PassiveEnvChecker</c> enforce this
        ///     since v1.4.0. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L79">gymnasium/core.py</see>.
        /// </remarks>
        public abstract StepResult<TObs> Step(TAct action);

        /// <summary>
        ///     Steps asynchronously: completes with what <see cref="Step"/> returns. The default runs <see cref="Step"/> on the
        ///     calling thread and completes synchronously, with no thread-pool hop and no allocation; an env whose dynamics are
        ///     genuinely asynchronous (a remote simulator, a batched GPU step) overrides it.
        /// </summary>
        /// <param name="action">An element of <see cref="ActionSpace"/>.</param>
        /// <param name="cancellationToken">Checked before the step starts; a step that has started runs to completion.</param>
        /// <returns>
        ///     A value task that completes with the transition, is faulted with what <see cref="Step"/> threw, or is canceled
        ///     when <paramref name="cancellationToken"/> was already canceled.
        /// </returns>
        /// <remarks>
        ///     Exceptions surface when the task is awaited, never from this call, as for any asynchronous method. Wrappers
        ///     inherit this default, so awaiting a wrapper runs the wrapper's own <see cref="Step"/> (and, through it, the env
        ///     beneath, synchronously); a wrapper that must keep an asynchronous env asynchronous overrides <c>StepAsync</c> as
        ///     well. Not thread-safe, like <see cref="Step"/>: await each call before issuing the next on the same env.
        /// </remarks>
        public virtual ValueTask<StepResult<TObs>> StepAsync(TAct action, CancellationToken cancellationToken = default) {
            if (cancellationToken.IsCancellationRequested)
                return ValueTask.FromCanceled<StepResult<TObs>>(cancellationToken);
            try {
                return new ValueTask<StepResult<TObs>>(Step(action));
            } catch (Exception exception) {
                // An asynchronous method reports failure through its task, so the caller's await, not this call, throws.
                return ValueTask.FromException<StepResult<TObs>>(exception);
            }
        }

        /// <summary>
        ///     Starts an episode (<c>reset(seed=seed, options=options)</c>). Implementations call <see cref="SeedNpRandom"/>
        ///     first, then draw the initial state from <see cref="NpRandom"/>.
        /// </summary>
        /// <param name="seed">
        ///     Reseeds <see cref="NpRandom"/> with Gymnasium's <c>Generator(PCG64(SeedSequence(seed)))</c> when given (any
        ///     non-negative integer); <see langword="null"/> keeps the current generator. Seed once, right after
        ///     construction, then reset without a seed.
        /// </param>
        /// <param name="options">Env-specific reset options, or <see langword="null"/>.</param>
        /// <returns>The first observation and the info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <remarks>
        ///     As with <see cref="Step"/>, the observation and info must be new objects, not a buffer a previous call
        ///     returned and this one mutated in place. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L124">gymnasium/core.py</see>.
        /// </remarks>
        public abstract ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null);

        /// <summary>
        ///     Renders as <see cref="RenderMode"/> says (<c>render()</c>): <c>rgb_array</c> returns a frame, <c>ansi</c> text,
        ///     <c>human</c> draws to a window during <see cref="Step"/> and returns <see cref="RenderResult.None"/>. The base
        ///     raises <see cref="NotImplementedError"/>, as <c>Env.render</c> does.
        /// </summary>
        /// <returns>The rendering.</returns>
        /// <exception cref="NotImplementedError">Thrown unless the env overrides it.</exception>
        public virtual RenderResult Render() => throw new NotImplementedError();

        /// <summary>
        ///     The keyboard mapping <c>play</c> uses when the caller gives none (<c>get_keys_to_action()</c>): key combinations
        ///     to the actions they trigger, MountainCar's <c>{(): 1, (276,): 0, (275,): 2, (275, 276): 1}</c>. Absent until an
        ///     env overrides it (DR-45): this base raises Python's <see cref="AttributeError"/>, as calling a method the class
        ///     lacks does, and <see cref="PyAttributes"/> reports the attribute only on an env that overrides it.
        /// </summary>
        /// <returns>A new mapping the caller owns, its actions included: an override builds fresh actions on every call.</returns>
        /// <exception cref="AttributeError">
        ///     Thrown by this base (<c>'CartPoleEnv' object has no attribute 'get_keys_to_action'</c>), and by an override in a
        ///     state that has no mapping (MountainCar's continuous mode).
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     Wrappers inherit this absent base: Gymnasium 1.x's wrappers forward no attributes, and <c>play</c> asks the layers
        ///     beneath through <c>has_wrapper_attr</c>. An override whose presence depends on the instance also overrides
        ///     <see cref="HasPythonAttribute"/>, so <c>hasattr</c> answers as Python's would for that state.
        ///     </para>
        ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/utils/play.py#L39">gymnasium/utils/play.py</see>.</para>
        /// </remarks>
        [PyName("get_keys_to_action")]
        public virtual IReadOnlyDictionary<KeyCombination, TAct> GetKeysToAction() =>
            throw new AttributeError($"'{PythonClassName}' object has no attribute 'get_keys_to_action'");

        /// <summary>
        ///     The class name Python prints for this instance in messages: the class's own name (<see cref="PyAttributes.PyTypeName"/>)
        ///     unless an env whose Python class depends on the instance says otherwise (the merged MountainCar's
        ///     <c>Continuous_MountainCarEnv</c>, DR-32). <see cref="PyAttributes"/> names it in its <see cref="AttributeError"/>s.
        /// </summary>
        protected internal virtual string PythonClassName => PyAttributes.PyTypeName(GetType());

        /// <summary>The class name, as <see cref="PyAttributes"/> reaches it on any env.</summary>
        string IPythonAttributePresence.PythonClassName => PythonClassName;

        /// <summary>
        ///     Whether this instance has the Python attribute <paramref name="name"/>, for attributes whose presence varies by
        ///     instance (DR-45): <see cref="PyAttributes"/> asks it before reporting a <see cref="PyNameAttribute"/> method that
        ///     the env overrides. The default leaves the answer to that override check.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling (<c>get_keys_to_action</c>).</param>
        /// <returns><see langword="false"/> to hide the attribute on this instance; <see langword="true"/> here.</returns>
        protected internal virtual bool HasPythonAttribute(string name) => true;

        /// <summary>The presence hook, as <see cref="PyAttributes"/> reaches it on any env.</summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns>What <see cref="HasPythonAttribute"/> answers.</returns>
        bool IPythonAttributePresence.HasPythonAttribute(string name) => HasPythonAttribute(name);

        /// <summary>
        ///     Releases what the env holds (<c>close()</c>): windows, native handles, connections. Calling it on a closed env
        ///     does nothing more. The base holds nothing.
        /// </summary>
        public virtual void Close() { }

        /// <summary>
        ///     0.2's name for <see cref="Close"/>, kept so 0.2's teardown line compiles until 3.0 (DR-58): it calls
        ///     <see cref="Close"/>, the env's own override included.
        /// </summary>
        [Obsolete("Close() is Gymnasium's name (or Dispose(), through using); CloseEnvironment is 0.2's. Removed in 3.0 with Gym.Legacy.")]
        public void CloseEnvironment() => Close();

        /// <summary>
        ///     Calls <see cref="Close"/>, so <c>using</c> plays the role of Python's <c>with</c> statement (<c>__exit__</c>), then
        ///     releases the spaces this env owns (D19). Idempotent in effect: a second call closes again, which every env's
        ///     <see cref="Close"/> tolerates, and finds the spaces already released.
        /// </summary>
        public void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>
        ///     What <see cref="Dispose()"/> does: <see cref="Close"/>, then the env's own action and observation spaces. A wrapper
        ///     overrides it to release its own spaces and the env it wraps.
        /// </summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Dispose()"/>; envs have no finalizers.</param>
        protected virtual void Dispose(bool disposing) {
            Close();
            if (!disposing)
                return;
            // The fields, not the virtual properties: a wrapper's properties return the wrapped env's spaces, which that env
            // releases itself.
            _actionSpace?.Dispose();
            _observationSpace?.Dispose();
        }

        /// <summary>
        ///     Whether this env has the Python attribute <paramref name="name"/> (Python's <c>hasattr</c>: declared members by
        ///     Python name, then run-time attributes).
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns><see langword="true"/> when reading it would succeed.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L282">gymnasium/core.py</see>.</remarks>
        public virtual bool HasWrapperAttr(string name) => PyAttributes.HasAttr(this, name);

        /// <summary>The Python attribute <paramref name="name"/> of this env, as Python's <c>getattr(self, name)</c> reads it.</summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <returns>The value, boxed.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">Thrown when the env has no such attribute (<c>'CartPoleEnv' object has no attribute 'x'</c>).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L286">gymnasium/core.py</see>.</remarks>
        public virtual object? GetWrapperAttr(string name) => PyAttributes.GetAttr(this, name);

        /// <summary>
        ///     Assigns the Python attribute <paramref name="name"/> when <paramref name="force"/> is set or the env has it,
        ///     creating a run-time attribute for a new name.
        /// </summary>
        /// <param name="name">The attribute, in Python spelling.</param>
        /// <param name="value">The value; numbers convert to a numeric attribute's type.</param>
        /// <param name="force">Assign even when the env does not have the attribute yet.</param>
        /// <returns><see langword="true"/> when the attribute was assigned.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="name"/> is <see langword="null"/>.</exception>
        /// <exception cref="AttributeError">Thrown for a read-only attribute (<c>property 'np_random_seed' of 'CartPoleEnv' object has no setter</c>).</exception>
        /// <exception cref="TypeError">Thrown when the value does not fit the attribute's type.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L290">gymnasium/core.py</see>.</remarks>
        public virtual bool SetWrapperAttr(string name, object? value, bool force = true) {
            if (force || PyAttributes.HasAttr(this, name)) {
                PyAttributes.SetAttr(this, name, value);
                return true;
            }

            return false;
        }

        /// <summary>
        ///     The env's description: <c>&lt;CartPoleEnv instance&gt;</c> without a spec, <c>&lt;CartPoleEnv&lt;CartPole-v1&gt;&gt;</c>
        ///     with one; wrapper descriptions embed it.
        /// </summary>
        /// <returns>The text, the same as Gymnasium prints for the env.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L261">gymnasium/core.py</see>.</remarks>
        public override string ToString() {
            EnvSpec? spec = Spec;
            string name = PyAttributes.PyTypeName(GetType());
            return spec is null ? $"<{name} instance>" : $"<{name}<{spec.Id}>>";
        }

        /// <summary>
        ///     The first line of every <see cref="Reset"/> (Gymnasium's <c>super().reset(seed=seed)</c>): reseeds
        ///     <see cref="NpRandom"/> and records <see cref="NpRandomSeed"/> when a seed is given; without one, the generator,
        ///     and with it the random stream, carries on.
        /// </summary>
        /// <param name="seed">The seed passed to <see cref="Reset"/>, or <see langword="null"/>.</param>
        /// <exception cref="Error">Thrown for a negative seed: <c>Seed must be greater or equal to zero, actual value: -1</c>.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L124">gymnasium/core.py</see>.</remarks>
        protected void SeedNpRandom(BigInteger? seed) {
            // Initialize the RNG if the seed is manually passed.
            if (seed is not null)
                (_npRandom, _npRandomSeed) = Seeding.NpRandom(seed);
        }

        /// <summary>
        ///     Converts an action given to the untyped <see cref="IEnv.Step"/> to <typeparamref name="TAct"/>: an instance of it
        ///     as is; for an integer or enum action type, any .NET integer, <see langword="bool"/>, enum member or 0-d integer
        ///     array (what <c>Discrete.Sample()</c> returns); for a floating-point action type, any number or 0-d array; for an
        ///     <see cref="NDArray"/> action type, what <c>np.asarray</c> makes of it: a .NET integer (or <see langword="bool"/>,
        ///     Python's <c>int</c> subclass) a 0-d int64 array, a <see langword="double"/> or <see langword="float"/> a 0-d float64
        ///     or float32 array, a .NET array an array of its element type.
        /// </summary>
        /// <param name="action">The action as given.</param>
        /// <returns>The typed action.</returns>
        /// <exception cref="TypeError">Thrown when the action cannot denote a <typeparamref name="TAct"/>.</exception>
        /// <exception cref="OverflowException">Thrown when an integer does not fit an integer action type.</exception>
        /// <remarks>
        ///     Python has no such step: an env receives whatever it is given and validates it itself. Envs whose Python source
        ///     asserts on the action override this to assert first, so an unconvertible action fails with the source's message
        ///     (CartPole: <c>'x' (&lt;class 'str'&gt;) invalid</c>) rather than this one.
        /// </remarks>
        protected virtual TAct ConvertAction(object? action) {
            if (action is TAct typed)
                return typed;
            Type target = Nullable.GetUnderlyingType(typeof(TAct)) ?? typeof(TAct);
            if (action is null) {
                // null is None: fine for a reference or nullable action type, which the check above cannot match.
                if (default(TAct) is null)
                    return default!;
            } else if (target.IsEnum) {
                if (Discrete.TryToInt64(action, out long value))
                    return (TAct)Enum.ToObject(target, value);
            } else if (IsIntegerType(target)) {
                if (Discrete.TryToInt64(action, out long value))
                    return (TAct)Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
            } else if (target == typeof(double) || target == typeof(float)) {
                if (action is not string && (action is IConvertible || action is NDArray { ndim: 0 })) {
                    double real = PyBuiltins.Float(action);
                    return (TAct)(target == typeof(float) ? (object)(float)real : real);
                }
            } else if (target == typeof(NDArray)) {
                // A Discrete action space's element is a 0-d int64 array (DR-08), so a plain .NET integer becomes one, as
                // Gymnasium's envs accept a Python int where they sample an np.int64.
                switch (action) {
                    case double d:
                        return (TAct)(object)NDArray.Scalar<double>(d);
                    case float f:
                        return (TAct)(object)NDArray.Scalar<float>(f);
                    case Array array:
                        return (TAct)(object)np.array(array);
                    default:
                        if (action is not NDArray && Discrete.TryToInt64(action, out long integer))
                            return (TAct)(object)NDArray.Scalar<long>(integer);
                        break;
                }
            }

            throw new TypeError($"{PyAttributes.PyTypeName(GetType())} takes actions of type {target.Name}, not {PyBuiltins.Repr(action)} ({PyBuiltins.TypeRepr(action)})");
        }

        /// <summary>The untyped step: converts the action (<see cref="ConvertAction"/>), steps, and boxes the observation.</summary>
        /// <param name="action">The action.</param>
        /// <returns>The transition with its observation boxed.</returns>
        StepResult<object?> IEnv.Step(object? action) {
            TAct typed = ConvertAction(action);
            try {
                StepResult<TObs> result = Step(typed);
                return new StepResult<object?>(result.Observation, result.Reward, result.Terminated, result.Truncated, result.Info);
            } finally {
                ReleaseConverted(typed, action);
            }
        }

        /// <summary>
        ///     The untyped asynchronous step: converts the action and awaits <see cref="StepAsync(TAct, CancellationToken)"/>, so
        ///     an action that cannot be converted faults the returned task instead of throwing here.
        /// </summary>
        /// <param name="action">The action.</param>
        /// <param name="cancellationToken">Checked before the step starts.</param>
        /// <returns>A value task that completes with the transition, its observation boxed.</returns>
        async ValueTask<StepResult<object?>> IEnv.StepAsync(object? action, CancellationToken cancellationToken) {
            TAct typed = ConvertAction(action);
            try {
                StepResult<TObs> result = await StepAsync(typed, cancellationToken).ConfigureAwait(false);
                return new StepResult<object?>(result.Observation, result.Reward, result.Terminated, result.Truncated, result.Info);
            } finally {
                ReleaseConverted(typed, action);
            }
        }

        /// <summary>The untyped reset: resets and boxes the observation.</summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <param name="options">The options, or <see langword="null"/>.</param>
        /// <returns>The first observation, boxed, and the info.</returns>
        ResetResult<object?> IEnv.Reset(BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            ResetResult<TObs> result = Reset(seed, options);
            return new ResetResult<object?>(result.Observation, result.Info);
        }

        /// <summary>
        ///     Releases an action <see cref="ConvertAction"/> made (a 0-d array from an integer) once the step is over; an action
        ///     the caller passed as is stays the caller's. Actions are borrowed for the step (D19): an env that keeps one past
        ///     <see cref="Step"/> must copy it.
        /// </summary>
        /// <param name="converted">The typed action the step received.</param>
        /// <param name="given">The action as the caller gave it.</param>
        private static void ReleaseConverted(TAct converted, object? given) {
            if (!ReferenceEquals(converted, given) && converted is IDisposable disposable)
                disposable.Dispose();
        }

        /// <summary>
        ///     An untyped space as the env's typed one: the space itself when its elements are <typeparamref name="T"/>, or an
        ///     <see cref="ObjectSpace"/> view of it when <typeparamref name="T"/> is <see cref="object"/>, so an
        ///     <c>Env&lt;object?, object?&gt;</c> can hold any space.
        /// </summary>
        /// <typeparam name="T">The env's element type.</typeparam>
        /// <param name="space">The space assigned.</param>
        /// <param name="attribute">The Python attribute, for the error.</param>
        /// <returns>The typed space.</returns>
        /// <exception cref="TypeError">Thrown for a space whose elements are another type.</exception>
        private Space<T> Typed<T>(Space? space, string attribute) {
            if (space is Space<T> typed)
                return typed;
            if (space is not null && typeof(T) == typeof(object))
                return (Space<T>)(object)new ObjectSpace(space);
            throw SpaceTypeMismatch(space, typeof(T), attribute);
        }

        /// <summary>The error for assigning an untyped space whose elements are not the env's type.</summary>
        /// <param name="space">The space assigned.</param>
        /// <param name="expected">The element type the env requires.</param>
        /// <param name="attribute">The Python attribute, <c>action_space</c> or <c>observation_space</c>.</param>
        /// <returns>The error to throw.</returns>
        private TypeError SpaceTypeMismatch(Space? space, Type expected, string attribute) =>
            new($"{PyAttributes.PyTypeName(GetType())}.{attribute} must be a space of {expected.Name}, not {(space is null ? "None" : space.GetType().Name)}");

        /// <summary>The error Python raises for an attribute that was declared but never assigned.</summary>
        /// <param name="name">The attribute.</param>
        /// <returns>The error to throw: <c>'X' object has no attribute 'name'</c>.</returns>
        private AttributeError MissingAttribute(string name) => new($"'{PyAttributes.PyTypeName(GetType())}' object has no attribute '{name}'");

        /// <summary>Whether a type is one of .NET's integer primitives.</summary>
        /// <param name="type">The type.</param>
        /// <returns><see langword="true"/> for the signed and unsigned integers of every width.</returns>
        private static bool IsIntegerType(Type type) {
            return type == typeof(long) || type == typeof(int) || type == typeof(short) || type == typeof(sbyte) || type == typeof(ulong) || type == typeof(uint)
                   || type == typeof(ushort) || type == typeof(byte);
        }
    }
}
