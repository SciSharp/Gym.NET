#nullable enable

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     One possible outcome of an action in a toy_text env's transition table <c>P[s][a]</c>: Gymnasium's 4-tuple
    ///     <c>(probability, next_state, reward, terminated)</c>, in the order Gymnasium lists the outcomes.
    /// </summary>
    /// <param name="Probability">The outcome's probability; the outcomes of one action sum to 1 (up to rounding).</param>
    /// <param name="NextState">The state the outcome leads to.</param>
    /// <param name="Reward">The reward, as a float64 (Gymnasium's are Python ints in FrozenLake, CliffWalking and Taxi, of the same value).</param>
    /// <param name="Terminated">Whether the outcome ends the episode.</param>
    /// <remarks>
    ///     Steps draw an outcome with <see cref="ToyTextUtils.CategoricalSample"/> over the probabilities, one float64 from the
    ///     env's generator, so a table with a single certain outcome still draws.
    /// </remarks>
    public sealed record ToyTextTransition(double Probability, long NextState, double Reward, bool Terminated);
}
