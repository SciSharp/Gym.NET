#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Tests.Ported {
    /// <summary>
    ///     Port of <c>tests/testing_env.py</c> @ 9e04324: <see cref="GenericTestEnv"/>, the env Gymnasium's tests build with
    ///     custom reset, step and render functions, and the default functions.
    /// </summary>
    /// <remarks>
    ///     <c>old_reset_func</c> and <c>old_step_func</c> (the pre-0.26 API, for the env checker's tests, P3) and
    ///     <c>GenericTestVectorEnv</c> (P6) come with the code they test.
    /// </remarks>
    public static class TestingEnv {
        /// <summary>
        ///     Port of <c>basic_reset_func</c>: seeds the env as <c>super().reset(seed=seed)</c> does, seeds the observation
        ///     space with the env's <c>np_random_seed</c>, and returns a sample with <c>{"options": options}</c>.
        /// </summary>
        /// <param name="self">The env.</param>
        /// <param name="seed">The reset seed.</param>
        /// <param name="options">The reset options, echoed in the info.</param>
        /// <returns>A sample of the observation space and the info.</returns>
        public static ResetResult<object?> BasicResetFunc(GenericTestEnv self, BigInteger? seed, IReadOnlyDictionary<string, object?>? options) {
            self.BaseReset(seed);
            self.ObservationSpace.Seed(self.NpRandomSeed);
            return new ResetResult<object?>(self.ObservationSpace.Sample(), new Info { ["options"] = options });
        }

        /// <summary>Port of <c>basic_step_func</c>: a sample of the observation space, reward 0, not terminated or truncated, <c>{}</c>.</summary>
        /// <param name="self">The env.</param>
        /// <param name="action">Ignored.</param>
        /// <returns>The transition.</returns>
        public static StepResult<object?> BasicStepFunc(GenericTestEnv self, object? action) {
            return new StepResult<object?>(self.ObservationSpace.Sample(), 0, false, false, new Info());
        }

        /// <summary>Port of <c>basic_render_func</c>: renders nothing.</summary>
        /// <param name="self">The env.</param>
        /// <returns><see cref="RenderResult.None"/>.</returns>
        public static RenderResult BasicRenderFunc(GenericTestEnv self) => RenderResult.None;
    }

    /// <summary>
    ///     Port of <c>GenericTestEnv</c> (<c>tests/testing_env.py:55</c> @ 9e04324): an env whose reset, step and render are
    ///     the functions given to the constructor, with <c>Box(0, 1, (1,))</c> spaces and the spec <c>TestingEnv-v0</c>
    ///     (100 steps) unless told otherwise.
    /// </summary>
    /// <remarks>
    ///     Observations and actions are <see cref="object"/>, as the Python env takes and returns anything. Python binds the
    ///     functions with <c>types.MethodType</c>; here they are delegates that receive the env. A <see langword="null"/>
    ///     function keeps the method that raises <see cref="NotImplementedError"/>, as not binding one does.
    /// </remarks>
    public sealed class GenericTestEnv : Env<object?, object?> {
        /// <summary>The sentinel of <c>DEFAULT_SPACE</c>: use <c>Box(0, 1, (1,))</c>.</summary>
        public static readonly Space DefaultSpace = new DefaultTestSpace();

        /// <summary>The bound reset function, or <see langword="null"/>.</summary>
        private readonly Func<GenericTestEnv, BigInteger?, IReadOnlyDictionary<string, object?>?, ResetResult<object?>>? _resetFunc;

        /// <summary>The bound step function, or <see langword="null"/>.</summary>
        private readonly Func<GenericTestEnv, object?, StepResult<object?>>? _stepFunc;

        /// <summary>The bound render function, or <see langword="null"/>.</summary>
        private readonly Func<GenericTestEnv, RenderResult>? _renderFunc;

        /// <summary>
        ///     Port of <c>GenericTestEnv.__init__</c> with Python's defaults: a <see langword="null"/> argument means the default
        ///     (<c>DEFAULT_SPACE</c>, the basic functions). To leave a space unset or a function unbound, as passing Python's
        ///     <c>None</c> does, use the constructor that takes <c>explicitNulls</c>.
        /// </summary>
        /// <param name="actionSpace">The action space; <see langword="null"/> or <see cref="DefaultSpace"/> is <c>Box(0, 1, (1,))</c>.</param>
        /// <param name="observationSpace">The observation space; <see langword="null"/> or <see cref="DefaultSpace"/> is <c>Box(0, 1, (1,))</c>.</param>
        /// <param name="resetFunc">The reset function; the default is <see cref="TestingEnv.BasicResetFunc"/>.</param>
        /// <param name="stepFunc">The step function; the default is <see cref="TestingEnv.BasicStepFunc"/>.</param>
        /// <param name="renderFunc">The render function; the default is <see cref="TestingEnv.BasicRenderFunc"/>.</param>
        /// <param name="metadata">The metadata; <see langword="null"/> is <c>{"render_modes": []}</c>.</param>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="spec">The spec; <see langword="null"/> is <c>EnvSpec("TestingEnv-v0", "tests.testing_env:GenericTestEnv", max_episode_steps=100)</c>.</param>
        public GenericTestEnv(Space? actionSpace = null, Space? observationSpace = null,
            Func<GenericTestEnv, BigInteger?, IReadOnlyDictionary<string, object?>?, ResetResult<object?>>? resetFunc = null,
            Func<GenericTestEnv, object?, StepResult<object?>>? stepFunc = null, Func<GenericTestEnv, RenderResult>? renderFunc = null,
            Metadata? metadata = null, string? renderMode = null, EnvSpec? spec = null) : this(true, actionSpace ?? DefaultSpace, observationSpace ?? DefaultSpace,
            resetFunc ?? TestingEnv.BasicResetFunc, stepFunc ?? TestingEnv.BasicStepFunc, renderFunc ?? TestingEnv.BasicRenderFunc, metadata, renderMode, spec) { }

        /// <summary>The constructor proper, where <see langword="null"/> spaces and functions mean "leave unset/unbound", Python's <c>None</c>.</summary>
        /// <param name="explicitNulls">Distinguishes this overload; always <see langword="true"/>.</param>
        /// <param name="actionSpace"><see cref="DefaultSpace"/>, <see langword="null"/> (unset), or a space.</param>
        /// <param name="observationSpace"><see cref="DefaultSpace"/>, <see langword="null"/> (unset), or a space.</param>
        /// <param name="resetFunc">The reset function, or <see langword="null"/> to keep the one that raises.</param>
        /// <param name="stepFunc">The step function, or <see langword="null"/> to keep the one that raises.</param>
        /// <param name="renderFunc">The render function, or <see langword="null"/> to keep the one that raises.</param>
        /// <param name="metadata">The metadata; <see langword="null"/> is <c>{"render_modes": []}</c>.</param>
        /// <param name="renderMode">The render mode.</param>
        /// <param name="spec">The spec; <see langword="null"/> is the default <c>TestingEnv-v0</c> spec.</param>
        public GenericTestEnv(bool explicitNulls, Space? actionSpace, Space? observationSpace,
            Func<GenericTestEnv, BigInteger?, IReadOnlyDictionary<string, object?>?, ResetResult<object?>>? resetFunc,
            Func<GenericTestEnv, object?, StepResult<object?>>? stepFunc, Func<GenericTestEnv, RenderResult>? renderFunc,
            Metadata? metadata = null, string? renderMode = null, EnvSpec? spec = null) {
            if (ReferenceEquals(actionSpace, DefaultSpace))
                ActionSpace = new Box(0f, 1f, new Shape(1));
            else if (actionSpace is not null)
                ActionSpace = actionSpace;
            // If actionSpace is null, don't set the attribute (for testing error cases).

            if (ReferenceEquals(observationSpace, DefaultSpace))
                ObservationSpace = new Box(0f, 1f, new Shape(1));
            else if (observationSpace is not null)
                ObservationSpace = observationSpace;

            _resetFunc = resetFunc;
            _stepFunc = stepFunc;
            _renderFunc = renderFunc;

            Metadata = metadata ?? new Metadata { [Metadata.RenderModesKey] = new List<string>() };
            RenderMode = renderMode;
            Spec = spec ?? new EnvSpec("TestingEnv-v0", "tests.testing_env:GenericTestEnv", maxEpisodeSteps: 100);
        }

        /// <summary><c>super(GenericTestEnv, self).reset(seed=seed)</c>: <c>Env.reset</c>'s seeding, for the reset functions.</summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <exception cref="Error">Thrown for a negative seed.</exception>
        public void BaseReset(BigInteger? seed) => SeedNpRandom(seed);

        /// <summary>Calls the reset function.</summary>
        /// <param name="seed">The seed.</param>
        /// <param name="options">The options.</param>
        /// <returns>What the function returns.</returns>
        /// <exception cref="NotImplementedError">Thrown when no reset function is bound: <c>TestingEnv reset_fn is not set.</c></exception>
        public override ResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            if (_resetFunc is null)
                throw new NotImplementedError("TestingEnv reset_fn is not set.");
            return _resetFunc(this, seed, options);
        }

        /// <summary>Calls the step function.</summary>
        /// <param name="action">The action.</param>
        /// <returns>What the function returns.</returns>
        /// <exception cref="NotImplementedError">Thrown when no step function is bound: <c>TestingEnv step_fn is not set.</c></exception>
        public override StepResult<object?> Step(object? action) {
            if (_stepFunc is null)
                throw new NotImplementedError("TestingEnv step_fn is not set.");
            return _stepFunc(this, action);
        }

        /// <summary>Calls the render function.</summary>
        /// <returns>What the function returns.</returns>
        /// <exception cref="NotImplementedError">Thrown when no render function is bound: <c>testingEnv render_fn is not set.</c></exception>
        public override RenderResult Render() {
            if (_renderFunc is null)
                throw new NotImplementedError("testingEnv render_fn is not set.");
            return _renderFunc(this);
        }

        /// <summary>Port of <c>DefaultTestSpace</c>: a sentinel type standing for "the default space", never sampled.</summary>
        private sealed class DefaultTestSpace : Space {
            /// <summary>The sentinel.</summary>
            public DefaultTestSpace() : base(Shape.Scalar, np.float32) { }

            /// <summary>Never called: the sentinel is replaced before use.</summary>
            /// <param name="mask">Ignored.</param>
            /// <returns>Nothing; always throws.</returns>
            /// <exception cref="NotSupportedException">Always.</exception>
            public override NDArray Sample(NDArray? mask = null) => throw new NotSupportedException("DEFAULT_SPACE is a sentinel.");

            /// <summary>The sentinel contains nothing.</summary>
            /// <param name="ndArray">Ignored.</param>
            /// <returns><see langword="false"/>.</returns>
            public override bool Contains(object ndArray) => false;

            /// <summary>Nothing to seed.</summary>
            /// <param name="seed">Ignored.</param>
            public override void Seed(BigInteger seed) { }
        }
    }
}
