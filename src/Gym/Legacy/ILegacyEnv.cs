#nullable enable
using System;
using System.Threading.Tasks;
using Gym.Collections;
using Gym.Spaces;
using NumSharp;
using SixLabors.ImageSharp;

namespace Gym.Legacy {
    /// <summary>
    ///     The env contract Gym.NET had before the core-API work (P1), <c>gym</c> 0.21's: <c>Reset()</c> returns only the
    ///     observation, <c>Step</c> the 4-tuple <see cref="Step"/>, <c>Seed</c> reseeds separately, and <c>Render</c> takes the
    ///     mode per call. Code written against it runs on any current env through <see cref="LegacyEnvAdapter"/>.
    /// </summary>
    /// <remarks>
    ///     This was <c>Gym.Envs.IEnv</c>; that name now belongs to the current, Gymnasium-shaped contract. Kept for one major
    ///     version (port plan §16), with the migration: <c>Reset()</c> → <c>Reset(seed)</c>, <c>Step(a).Done</c> →
    ///     <c>terminated || truncated</c>, <c>Seed(s)</c> → <c>Reset(seed: s)</c>, <c>Render("rgb_array")</c> →
    ///     <c>render_mode</c> at construction.
    /// </remarks>
    [Obsolete("The pre-0.26 env API is legacy: program against Gym.Envs.IEnv / Env<TObs, TAct> (Reset(seed) returns (obs, info), Step returns StepResult<TObs>). Gym.Legacy is removed in 3.0.")]
    public interface ILegacyEnv : IDisposable {
        /// <summary>The env's metadata as the legacy <see cref="Dict"/>.</summary>
        Dict Metadata { get; set; }

        /// <summary>The range rewards fall in, <c>gym</c>'s <c>reward_range</c> (Gymnasium 1.x removed it), in float64 like the rewards.</summary>
        (double From, double To) RewardRange { get; set; }

        /// <summary>The action space.</summary>
        Space ActionSpace { get; set; }

        /// <summary>The observation space.</summary>
        Space ObservationSpace { get; set; }

        /// <summary>Starts an episode and returns its first observation.</summary>
        /// <returns>The observation.</returns>
        NDArray Reset();

        /// <summary>Runs one step.</summary>
        /// <param name="action">The action.</param>
        /// <returns>The 4-tuple transition.</returns>
        Step Step(object action);

        /// <summary>Runs one step asynchronously; by default it completes synchronously, on the calling thread.</summary>
        /// <param name="action">The action.</param>
        /// <returns>A value task completing with the transition, or faulted with what the step threw.</returns>
        ValueTask<Step> StepAsync(object action);

        /// <summary>Renders the current state.</summary>
        /// <param name="mode">The render mode.</param>
        /// <returns>The frame, or <see langword="null"/> when the mode returns none.</returns>
        Image? Render(string mode = "human");

        /// <summary>Releases the env's resources.</summary>
        void CloseEnvironment();

        /// <summary>Seeds the episodes that follow.</summary>
        /// <param name="seed">A non-negative seed.</param>
        void Seed(long seed);
    }
}
