#nullable enable
using Gym.Envs;
using Gym.Utils;

namespace Gym.Environments.Envs.Box2D {
    /// <summary>
    ///     CarRacing's info (<see cref="CarRacingEnv"/>, CarRacing-v3), typed (DR-91, DR-95): <c>{}</c> on every reset and on
    ///     most steps, and <c>{"lap_finished": ...}</c> on the step that ends an episode by finishing the lap or leaving the
    ///     playfield. LunarLander and BipedalWalker send <c>{}</c> and have no record.
    /// </summary>
    /// <param name="LapFinished">
    ///     Whether the episode ended with the lap done (<c>lap_finished</c>): <see langword="true"/> once every tile was
    ///     visited, <see langword="false"/> when the car left the playfield; <see langword="null"/> on any other step and at
    ///     reset, where the key is absent.
    /// </param>
    [GymDict]
    public sealed partial record CarRacingInfo(bool? LapFinished) : EnvInfo;
}
