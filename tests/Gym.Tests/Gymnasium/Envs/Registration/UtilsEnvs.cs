#nullable enable
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Spaces;
using NumSharp;

namespace Gym.Tests.Gymnasium.Envs.Registration {
    /// <summary>
    ///     An env whose constructor takes three required arguments, for the kwargs tests of <c>make</c>
    ///     (Gymnasium's <c>ArgumentEnv</c>). Its step and reset are the base env's, which raise.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/utils_envs.py#L6">tests/envs/registration/utils_envs.py (ArgumentEnv)</see>.</remarks>
    public sealed class ArgumentEnv : Env<NDArray, NDArray> {
        /// <summary>Records the three arguments; the spaces are Gymnasium's class attributes, <c>Box(-1, 1, (1,))</c>.</summary>
        /// <param name="arg1">The first argument.</param>
        /// <param name="arg2">The second argument.</param>
        /// <param name="arg3">The third argument.</param>
        public ArgumentEnv(object? arg1, object? arg2, object? arg3) {
            ObservationSpace = new Box(-1, 1, new Shape(1L));
            ActionSpace = new Box(-1, 1, new Shape(1L));
            Arg1 = arg1;
            Arg2 = arg2;
            Arg3 = arg3;
        }

        /// <summary>The first argument (<c>arg1</c>).</summary>
        public object? Arg1 { get; }

        /// <summary>The second argument (<c>arg2</c>).</summary>
        public object? Arg2 { get; }

        /// <summary>The third argument (<c>arg3</c>).</summary>
        public object? Arg3 { get; }

        /// <summary>Not implemented, as Gymnasium's base <c>Env.step</c>.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new NotImplementedError();

        /// <summary>Not implemented, as Gymnasium's base <c>Env.reset</c> only seeds.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotImplementedError();
    }

    /// <summary>An env that renders <c>rgb_array</c> but not <c>human</c> (Gymnasium's <c>NoHuman</c>), so <c>make</c> would add <c>HumanRendering</c>.</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/utils_envs.py#L17">tests/envs/registration/utils_envs.py (NoHuman)</see>.</remarks>
    public sealed class NoHuman : Env<NDArray, NDArray> {
        /// <summary>Asserts the render mode is one the class metadata lists, as Gymnasium's does.</summary>
        /// <param name="renderMode">The render mode.</param>
        /// <exception cref="AssertionError">Thrown for a mode the metadata does not list.</exception>
        public NoHuman(string? renderMode = null) {
            ObservationSpace = new Box(-1, 1, new Shape(1L));
            ActionSpace = new Box(-1, 1, new Shape(1L));
            Metadata = ClassMetadata;
            if (!Metadata.RenderModes!.Contains(renderMode!))
                throw new AssertionError("");
            RenderMode = renderMode;
        }

        /// <summary><c>{"render_modes": ["rgb_array"], "render_fps": 4}</c>.</summary>
        public static new Metadata ClassMetadata => new() { [Metadata.RenderModesKey] = new List<string> { "rgb_array" }, [Metadata.RenderFpsKey] = 4L };

        /// <summary>Not implemented.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new NotImplementedError();

        /// <summary>Not implemented.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotImplementedError();
    }

    /// <summary>
    ///     An env on the old rendering API: its constructor takes no <c>render_mode</c> (Gymnasium's <c>NoHumanOldAPI</c>), so
    ///     <c>make</c>'s render-mode keyword makes it raise <see cref="TypeError"/>.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/utils_envs.py#L31">tests/envs/registration/utils_envs.py (NoHumanOldAPI)</see>.</remarks>
    public sealed class NoHumanOldApi : Env<NDArray, NDArray> {
        /// <summary>Takes nothing, as the old API's constructors did.</summary>
        public NoHumanOldApi() {
            ObservationSpace = new Box(-1, 1, new Shape(1L));
            ActionSpace = new Box(-1, 1, new Shape(1L));
            Metadata = ClassMetadata;
        }

        /// <summary><c>{"render_modes": ["rgb_array_list"], "render_fps": 4}</c>.</summary>
        public static new Metadata ClassMetadata => new() { [Metadata.RenderModesKey] = new List<string> { "rgb_array_list" }, [Metadata.RenderFpsKey] = 4L };

        /// <summary>Not implemented.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new NotImplementedError();

        /// <summary>Not implemented.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotImplementedError();
    }

    /// <summary>An env that renders neither <c>human</c> nor <c>rgb_array</c>, only <c>ascii</c> (Gymnasium's <c>NoHumanNoRGB</c>).</summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/utils_envs.py#L43">tests/envs/registration/utils_envs.py (NoHumanNoRGB)</see>.</remarks>
    public sealed class NoHumanNoRgb : Env<NDArray, NDArray> {
        /// <summary>Asserts the render mode is one the class metadata lists.</summary>
        /// <param name="renderMode">The render mode.</param>
        /// <exception cref="AssertionError">Thrown for a mode the metadata does not list.</exception>
        public NoHumanNoRgb(string? renderMode = null) {
            ObservationSpace = new Box(-1, 1, new Shape(1L));
            ActionSpace = new Box(-1, 1, new Shape(1L));
            Metadata = ClassMetadata;
            if (!Metadata.RenderModes!.Contains(renderMode!))
                throw new AssertionError("");
            RenderMode = renderMode;
        }

        /// <summary><c>{"render_modes": ["ascii"], "render_fps": 4}</c>.</summary>
        public static new Metadata ClassMetadata => new() { [Metadata.RenderModesKey] = new List<string> { "ascii" }, [Metadata.RenderFpsKey] = 4L };

        /// <summary>Not implemented.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new NotImplementedError();

        /// <summary>Not implemented.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotImplementedError();
    }

    /// <summary>
    ///     An env that renders but kept the base metadata, <c>{"render_modes": []}</c> (Gymnasium's
    ///     <c>NoRenderModesMetadata</c>): <c>make</c> warns that its render mode is not listed, and builds it anyway.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/tests/envs/registration/utils_envs.py#L55">tests/envs/registration/utils_envs.py (NoRenderModesMetadata)</see>.</remarks>
    public sealed class NoRenderModesMetadata : Env<NDArray, NDArray> {
        /// <summary>Takes a required render mode and <c>Box(0, 1)</c> spaces.</summary>
        /// <param name="renderMode">The render mode, not checked.</param>
        public NoRenderModesMetadata(string? renderMode) {
            RenderMode = renderMode;
            ObservationSpace = new Box(0, 1);
            ActionSpace = new Box(0, 1);
        }

        /// <summary>Not implemented.</summary>
        /// <param name="action">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override StepResult<NDArray> Step(NDArray action) => throw new NotImplementedError();

        /// <summary>Not implemented.</summary>
        /// <param name="seed">Ignored.</param>
        /// <param name="options">Ignored.</param>
        /// <returns>Nothing; always throws.</returns>
        /// <exception cref="NotImplementedError">Always.</exception>
        public override ResetResult<NDArray> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) => throw new NotImplementedError();
    }
}
