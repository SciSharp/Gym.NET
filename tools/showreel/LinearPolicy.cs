using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Gym.Envs;
using Gym.Spaces;
using NumSharp;

namespace Gym.Showreel;

/// <summary>
///     A linear policy on normalized observations, the kind Augmented Random Search trains: <c>a_i = clip(sum_j W[i,j] *
///     (s_j - mean_j) / scale_j, -1, 1) * high_i</c>, cast to float32. It plays the MuJoCo envs in the showreel.
/// </summary>
/// <remarks>
///     <para>
///     The JSON format is the one <c>examples/HalfCheetah/policy.json</c> uses, so that autopilot
///     (trained on the same physics) loads as it is. Scaling by the action space's bound is this class's addition: it
///     is 1 for HalfCheetah, which keeps that policy's actions unchanged, and lets one formula drive envs whose bounds are
///     ±3 (InvertedPendulum), ±2 (Pusher) or ±0.4 (the humanoids).
///     </para>
///     <para>
///     Not thread-safe through <see cref="Act(ReadOnlySpan{double}, Span{float}, ReadOnlySpan{double})"/>'s callers'
///     buffers; the static overload is pure and is what the trainer's worker threads call.
///     </para>
/// </remarks>
internal sealed class LinearPolicy {
    /// <summary>Creates a zero policy (all actions 0) with identity normalization.</summary>
    /// <param name="inputs">Observation size.</param>
    /// <param name="outputs">Action size.</param>
    /// <exception cref="ArgumentOutOfRangeException">A size is not positive.</exception>
    public LinearPolicy(int inputs, int outputs) {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputs);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputs);
        Inputs = inputs;
        Outputs = outputs;
        Weights = new double[inputs * outputs];
        Mean = new double[inputs];
        Scale = Enumerable.Repeat(1.0, inputs).ToArray();
    }

    /// <summary>Observation size.</summary>
    public int Inputs { get; }

    /// <summary>Action size.</summary>
    public int Outputs { get; }

    /// <summary>The weight matrix, row-major, <see cref="Outputs"/> rows of <see cref="Inputs"/>; trained in place.</summary>
    public double[] Weights { get; }

    /// <summary>The observation mean the inputs are centred on.</summary>
    public double[] Mean { get; }

    /// <summary>The observation standard deviation the inputs are divided by (never 0: a small epsilon is added).</summary>
    public double[] Scale { get; }

    /// <summary>Provenance saved with the policy: trainer, settings, evaluation returns.</summary>
    public JsonObject Info { get; set; } = new();

    /// <summary>The action for an observation.</summary>
    /// <param name="observation">The observation, <see cref="Inputs"/> values.</param>
    /// <param name="action">Receives <see cref="Outputs"/> float32 actions.</param>
    /// <param name="high">The action space's upper bound per action (its lower bound is assumed to be its negation).</param>
    public void Act(ReadOnlySpan<double> observation, Span<float> action, ReadOnlySpan<double> high) =>
        Act(Weights, Mean, Scale, observation, action, high);

    /// <summary>The policy's formula over explicit parameters, for perturbed weights during training.</summary>
    /// <param name="weights">The weights, row-major.</param>
    /// <param name="mean">The observation mean.</param>
    /// <param name="scale">The observation scale.</param>
    /// <param name="observation">The observation.</param>
    /// <param name="action">Receives the actions.</param>
    /// <param name="high">The action bounds.</param>
    public static void Act(ReadOnlySpan<double> weights, ReadOnlySpan<double> mean, ReadOnlySpan<double> scale, ReadOnlySpan<double> observation,
        Span<float> action, ReadOnlySpan<double> high) {
        int n = observation.Length;
        for (int i = 0; i < action.Length; i++) {
            double acc = 0;
            for (int j = 0; j < n; j++)
                acc += weights[i * n + j] * ((observation[j] - mean[j]) / scale[j]);
            // Clipped by comparisons rather than Math.Clamp so a NaN passes through as the POC's autopilot passes it.
            double clipped = acc < -1.0 ? -1.0 : acc > 1.0 ? 1.0 : acc;
            action[i] = (float)(clipped * high[i]);
        }
    }

    /// <summary>A deep copy (the trainer keeps the best snapshot while it goes on changing the weights).</summary>
    /// <returns>The copy, with a copy of <see cref="Info"/>.</returns>
    public LinearPolicy Clone() {
        var copy = new LinearPolicy(Inputs, Outputs) { Info = (JsonObject)Info.DeepClone() };
        Weights.CopyTo(copy.Weights, 0);
        Mean.CopyTo(copy.Mean, 0);
        Scale.CopyTo(copy.Scale, 0);
        return copy;
    }

    /// <summary>Loads a policy file.</summary>
    /// <param name="path">The JSON file.</param>
    /// <returns>The policy.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="JsonException">The file is not JSON.</exception>
    /// <exception cref="InvalidDataException">An array has the wrong length.</exception>
    public static LinearPolicy Load(string path) {
        JsonNode root = JsonNode.Parse(File.ReadAllText(path)) ?? throw new JsonException($"{path}: empty");
        int inputs = root["inputs"]!.GetValue<int>(), outputs = root["outputs"]!.GetValue<int>();
        var policy = new LinearPolicy(inputs, outputs);
        Fill(policy.Weights, root["weights"], "weights", path);
        Fill(policy.Mean, root["mean"], "mean", path);
        Fill(policy.Scale, root["scale"], "scale", path);
        if (root["info"] is JsonObject info)
            policy.Info = (JsonObject)info.DeepClone();
        return policy;
    }

    /// <summary>Copies a JSON number array into a target of the expected length.</summary>
    /// <param name="target">The array to fill.</param>
    /// <param name="node">The JSON array.</param>
    /// <param name="name">Its key, for the error.</param>
    /// <param name="path">The file, for the error.</param>
    /// <exception cref="InvalidDataException">Not an array of the target's length.</exception>
    private static void Fill(double[] target, JsonNode? node, string name, string path) {
        if (node is not JsonArray array || array.Count != target.Length)
            throw new InvalidDataException($"{path}: '{name}' must hold {target.Length} numbers");
        for (int i = 0; i < target.Length; i++)
            target[i] = array[i]!.GetValue<double>();
    }

    /// <summary>Writes the policy as JSON (doubles round-trip exactly through System.Text.Json).</summary>
    /// <param name="path">The file; overwritten.</param>
    /// <exception cref="IOException">The file cannot be written.</exception>
    public void Save(string path) {
        var root = new JsonObject {
            ["format"] = "a = clip(sum_j W[i,j]*((s_j - mean_j)/scale_j), -1, 1) * high_i as float32; W row-major, outputs x inputs",
            ["inputs"] = Inputs,
            ["outputs"] = Outputs,
            ["weights"] = new JsonArray(Weights.Select(v => (JsonNode)v).ToArray()),
            ["mean"] = new JsonArray(Mean.Select(v => (JsonNode)v).ToArray()),
            ["scale"] = new JsonArray(Scale.Select(v => (JsonNode)v).ToArray()),
            ["info"] = Info.DeepClone(),
        };
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    /// <summary>The upper bound of each action of a Box action space, read once per player or worker.</summary>
    /// <param name="env">The env.</param>
    /// <returns>The bounds as float64.</returns>
    /// <exception cref="InvalidOperationException">The action space is not a Box.</exception>
    public static double[] ActionHigh(IEnv env) {
        if (env.ActionSpace is not Box box)
            throw new InvalidOperationException($"{env.Unwrapped.GetType().Name}'s action space is {env.ActionSpace}, not a Box");
        using NDArray high = box.High.astype(NPTypeCode.Double);
        return high.ToArray<double>();
    }
}

/// <summary>An <see cref="IPolicy"/> driving an env with a trained <see cref="LinearPolicy"/>.</summary>
internal sealed class LinearPolicyAgent : IPolicy {
    /// <summary>The trained policy.</summary>
    private readonly LinearPolicy _policy;

    /// <summary>The action buffer, reused each step.</summary>
    private readonly float[] _action;

    /// <summary>The action bounds, read from the env on the first step.</summary>
    private double[]? _high;

    /// <summary>Wraps a policy.</summary>
    /// <param name="policy">The policy (shared, read-only here).</param>
    public LinearPolicyAgent(LinearPolicy policy) {
        _policy = policy;
        _action = new float[policy.Outputs];
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">The observation's size is not the policy's.</exception>
    public object Act(IEnv env, object? observation) {
        _high ??= LinearPolicy.ActionHigh(env);
        var o = (NDArray)observation!;
        using NDArray d = o.astype(NPTypeCode.Double);
        double[] s = d.ToArray<double>();
        if (s.Length != _policy.Inputs)
            throw new InvalidOperationException($"the observation has {s.Length} values, the policy takes {_policy.Inputs}");
        _policy.Act(s, _action, _high);
        return np.array((float[])_action.Clone());
    }
}
