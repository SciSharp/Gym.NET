#nullable enable
using Gym.Envs;
using Gym.Utils;
using NumSharp;

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     The base of the toy_text envs' typed info records (DR-91, DR-95): the transition's probability, <c>prob</c>, which
    ///     FrozenLake, CliffWalking and Taxi send on every reset and step, as their transition tables give it
    ///     (<c>P[s][a] = [(prob, next, reward, terminated)]</c>). Blackjack sends <c>{}</c> and has no record.
    /// </summary>
    /// <param name="Prob">
    ///     The probability of the transition just taken (<c>prob</c>): 1 at a reset (a Python int there), and the table's
    ///     probability after a step (1.0 on a deterministic map, 1/3 or the success rate on a slippery one); read as a
    ///     <see langword="double"/>.
    /// </param>
    /// <remarks>
    ///     The keys are Gymnasium v1.4.0's, measured on every toy_text id (a reset with seed 0, then 300 random steps); every
    ///     key comes on every reset and step, so no member is nullable.
    /// </remarks>
    public abstract record ToyTextInfo(double Prob) : EnvInfo;

    /// <summary>FrozenLake's info (<see cref="FrozenLakeEnv"/>, FrozenLake-v1 and FrozenLake8x8-v1), typed (DR-91): <c>{"prob": ...}</c>.</summary>
    /// <param name="Prob">The probability of the transition just taken (<c>prob</c>); 1 at a reset.</param>
    [GymDict]
    public sealed partial record FrozenLakeInfo(double Prob) : ToyTextInfo(Prob);

    /// <summary>CliffWalking's info (<see cref="CliffWalkingEnv"/>, CliffWalking-v1 and CliffWalkingSlippery-v1), typed (DR-91): <c>{"prob": ...}</c>.</summary>
    /// <param name="Prob">The probability of the transition just taken (<c>prob</c>); 1 at a reset.</param>
    [GymDict]
    public sealed partial record CliffWalkingInfo(double Prob) : ToyTextInfo(Prob);

    /// <summary>Taxi's info (<see cref="TaxiEnv"/>, Taxi-v4), typed (DR-91): <c>{"prob": ..., "action_mask": ...}</c> on every reset and step.</summary>
    /// <param name="Prob">The probability of the transition just taken (<c>prob</c>), a Python float, 1.0 unless the taxi is rainy or the passenger fickle.</param>
    /// <param name="ActionMask">
    ///     Which of the six actions change the state (<c>action_mask</c>): an int8 array of six 0s and 1s, the step result's own
    ///     array (disposed with it).
    /// </param>
    [GymDict]
    public sealed partial record TaxiInfo(double Prob, NDArray ActionMask) : ToyTextInfo(Prob);
}
