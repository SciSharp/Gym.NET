using Gym.Agents;
using Gym.Envs;
using NumSharp;

namespace Gym.Showreel;

/// <summary>
///     An <see cref="IPolicy"/> driving an env with a published expert agent (DR-109): the observation, whatever its
///     dtype, goes through the agent's network and the float32 action comes back as an array.
/// </summary>
/// <remarks>
///     Stateless: the agent maps each observation to its action on its own, so one instance per player is only the
///     <see cref="IPolicy"/> contract, and the shared <see cref="ExpertPolicy"/> is thread-safe.
/// </remarks>
internal sealed class ExpertAgent : IPolicy {
    /// <summary>The agent (shared, read-only).</summary>
    private readonly ExpertPolicy _policy;

    /// <summary>Wraps an agent.</summary>
    /// <param name="policy">The agent.</param>
    public ExpertAgent(ExpertPolicy policy) => _policy = policy;

    /// <inheritdoc />
    /// <exception cref="System.ArgumentException">
    ///     The observation's length is not the agent's: the env was made without the agent's <c>make_kwargs</c>.
    /// </exception>
    public object Act(IEnv env, object? observation) {
        // Float64 MuJoCo observations and float32 Box2D ones alike: read as float64, the agent casts to float32 itself,
        // as Stable-Baselines3 does.
        using NDArray values = ((NDArray)observation!).astype(NPTypeCode.Double);
        return np.array(_policy.Act(values.ToArray<double>()));
    }
}
