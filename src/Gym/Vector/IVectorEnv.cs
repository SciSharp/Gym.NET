#nullable enable
using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Spaces;
using NumSharp;

namespace Gym.Vector {
    /// <summary>
    ///     A vector env seen without its types (DR-21): what registry-wide code, the untyped <c>MakeVec</c> and tools that
    ///     handle any vector env work with. Observations, actions and spaces are <see langword="object"/>s and untyped
    ///     <see cref="Space"/>s; the rewards and flags are arrays either way. <see cref="VectorEnv{TObs, TAct}"/> implements it.
    /// </summary>
    /// <remarks>
    ///     A vector env runs <see cref="NumEnvs"/> copies of an env in lockstep: every step takes one action per sub-env
    ///     (batched by <see cref="ActionSpace"/>) and returns every sub-env's result batched along the first axis. When a
    ///     sub-env's episode ends, the env's <c>metadata["autoreset_mode"]</c> (<see cref="AutoresetMode"/>) says when it is
    ///     reset. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/vector/vector_env.py#L45">gymnasium/vector/vector_env.py</see>.
    /// </remarks>
    public interface IVectorEnv : IDisposable {
        /// <summary>The metadata (<c>metadata</c>): the render modes and frame rate, and <c>autoreset_mode</c>.</summary>
        Metadata Metadata { get; set; }

        /// <summary>The spec <c>MakeVec</c> attached (<c>spec</c>), or <see langword="null"/> for a vector env built with its constructor.</summary>
        EnvSpec? Spec { get; set; }

        /// <summary>The sub-envs' render mode (<c>render_mode</c>), or <see langword="null"/>.</summary>
        string? RenderMode { get; }

        /// <summary>Whether <see cref="Close"/> ran (<c>closed</c>); a second close does nothing.</summary>
        bool Closed { get; set; }

        /// <summary>The number of sub-envs (<c>num_envs</c>), fixed at construction.</summary>
        int NumEnvs { get; }

        /// <summary>The batched observation space (<c>observation_space</c>): the single space batched <see cref="NumEnvs"/> times.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        Space ObservationSpace { get; set; }

        /// <summary>The batched action space (<c>action_space</c>), whose samples are one action per sub-env.</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        Space ActionSpace { get; set; }

        /// <summary>One sub-env's observation space (<c>single_observation_space</c>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        Space SingleObservationSpace { get; set; }

        /// <summary>One sub-env's action space (<c>single_action_space</c>).</summary>
        /// <exception cref="AttributeError">Thrown by the getter while unset.</exception>
        Space SingleActionSpace { get; set; }

        /// <summary>
        ///     The vector env's own generator (<c>np_random</c>), created from OS entropy on first read; the sub-envs have
        ///     theirs. Only vector envs that draw themselves (<c>CartPoleVectorEnv</c>) use it.
        /// </summary>
        Generator NpRandom { get; set; }

        /// <summary>The seed of <see cref="NpRandom"/> (<c>np_random_seed</c>); -1 after a generator was assigned.</summary>
        BigInteger NpRandomSeed { get; }

        /// <summary>The innermost vector env under any vector wrappers (<c>unwrapped</c>).</summary>
        IVectorEnv Unwrapped { get; }

        /// <summary>Resets every sub-env (<c>reset(seed=seed, options=options)</c>); an integer seed seeds sub-env <c>i</c> with <c>seed + i</c>.</summary>
        /// <param name="seed">The seed, or <see langword="null"/> to keep the generators.</param>
        /// <param name="options">The reset options, passed to every sub-env; <c>reset_mask</c> picks which sub-envs reset.</param>
        /// <returns>The batched observations and merged infos.</returns>
        VectorResetResult<object?> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null);

        /// <summary>Resets every sub-env with a seed each (<c>reset(seed=[...])</c>), <see langword="null"/> keeping a sub-env's generator.</summary>
        /// <param name="seeds">One seed per sub-env.</param>
        /// <param name="options">The reset options, as for <see cref="Reset(BigInteger?, IReadOnlyDictionary{string, object})"/>.</param>
        /// <returns>The batched observations and merged infos.</returns>
        VectorResetResult<object?> Reset(IReadOnlyList<BigInteger?> seeds, IReadOnlyDictionary<string, object?>? options = null);

        /// <summary>Steps every sub-env with its action (<c>step(actions)</c>).</summary>
        /// <param name="actions">One action per sub-env, as the batched action space samples them.</param>
        /// <returns>The batched transition.</returns>
        VectorStepResult<object?> Step(object? actions);

        /// <summary>Renders every sub-env (<c>render()</c>): one result each, or <see langword="null"/> where Python returns <c>None</c>.</summary>
        /// <returns>The frames or texts, in sub-env order, or <see langword="null"/>.</returns>
        IReadOnlyList<RenderResult>? Render();

        /// <summary>Closes the vector env and its sub-envs (<c>close()</c>); a second call does nothing.</summary>
        void Close();
    }
}
