using System;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Oracle;

namespace Gym.Tests.Parity.Pair {
    /// <summary>
    ///     The .NET half of a <see cref="GymPair"/>: anything that can be driven like a Gymnasium env and describe what it
    ///     did. <see cref="GymNetEnv"/> adapts Gym.NET's envs; test-local reference transcriptions (<c>Reference/</c>)
    ///     implement it directly, as positive and negative controls of the harness.
    /// </summary>
    /// <remarks>
    ///     Implementations describe their results immediately (<see cref="ValueFacts.Of"/>) and release what they own, so no
    ///     NumSharp array outlives a call inside the harness. Capabilities an implementation lacks throw
    ///     <see cref="NotSupportedException"/>; the pair only calls them when a test asks for them.
    /// </remarks>
    public interface IDotNetEnv : IDisposable {
        /// <summary>How reports name this side, for example <c>Gym.NET CartPoleEnv</c>.</summary>
        string Name { get; }

        /// <summary>
        ///     Contract gaps the adapter bridges, reported with every divergence and in the run summary, for example "the legacy
        ///     API returns no reset info; reported as {}". Empty for a side that implements Gymnasium's contract itself.
        /// </summary>
        string[] Adaptations { get; }

        /// <summary>Seeds and resets: Gymnasium's <c>reset(seed=seed)</c>.</summary>
        /// <param name="seed">The seed.</param>
        /// <returns>The described <c>(observation, info)</c>, or the exception the reset threw.</returns>
        CallResult Reset(long seed);

        /// <summary>Steps with an action: Gymnasium's <c>step(action)</c>.</summary>
        /// <param name="action">
        ///     The .NET action (a boxed <see langword="long"/>, or an <c>NDArray</c> that stays owned by the caller and is only
        ///     read during the call).
        /// </param>
        /// <returns>The described 5-tuple, or the exception the step threw.</returns>
        CallResult Step(object action);

        /// <summary>Describes the env generator's state (<c>np_random.bit_generator.state</c>), when this side can expose it.</summary>
        /// <param name="state">The described state, when <see langword="true"/> is returned.</param>
        /// <param name="reason">Why the state is not observable, when <see langword="false"/> is returned.</param>
        /// <returns><see langword="true"/> when the state was read.</returns>
        bool TryGetRngState(out ValueFacts? state, out string reason);

        /// <summary>Overwrites a state attribute with float64 values, for state injection (<c>env.unwrapped.state = ...</c>).</summary>
        /// <param name="name">The attribute, as Gymnasium names it (<c>state</c>).</param>
        /// <param name="values">The values, bit for bit what the Python side receives.</param>
        /// <exception cref="NotSupportedException">Thrown when this side has no such attribute.</exception>
        void SetAttribute(string name, double[] values);

        /// <summary>Describes a state attribute, for the before/after dumps of a divergence report.</summary>
        /// <param name="name">The attribute, as Gymnasium names it.</param>
        /// <returns>The described value, or <see langword="null"/> when this side has no such attribute.</returns>
        ValueFacts? GetAttribute(string name);

        /// <summary>Renders the current state in <c>rgb_array</c> form.</summary>
        /// <returns>The frame.</returns>
        /// <exception cref="NotSupportedException">Thrown when this side cannot render.</exception>
        Frame Render();

        /// <summary>Seeds the action space: Gymnasium's <c>action_space.seed(seed)</c>.</summary>
        /// <param name="seed">The seed.</param>
        void SeedActionSpace(long seed);

        /// <summary>Samples the action space: Gymnasium's <c>action_space.sample()</c>.</summary>
        /// <returns>The sample, owned by the caller (dispose it when it is an <c>NDArray</c>).</returns>
        object SampleAction();
    }
}
