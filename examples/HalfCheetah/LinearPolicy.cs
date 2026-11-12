using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace MujocoPoc.Play;

/// <summary>
///     The autopilot: a linear policy on normalized observations, <c>a_i = clip(sum_j W[i,j]*((s_j - mean_j)/scale_j), -1,
///     1)</c> cast to float32, as Augmented Random Search trains it (<see cref="Ars"/>).
/// </summary>
/// <remarks>
///     The arithmetic order is part of the contract: each term is <c>W*((s - mean)/scale)</c>, summed from 0 in column
///     order, clipped in double, then rounded to float32. replay_in_gymnasium.py evaluates it in the same order in plain
///     Python floats, so the same policy produces the same actions, and therefore the same episode, in Gymnasium on
///     MuJoCo. NumPy's matrix product would sum in another order and lose that.
/// </remarks>
public sealed class LinearPolicy {
    /// <summary>The observation size.</summary>
    public int Inputs { get; }

    /// <summary>The action size.</summary>
    public int Outputs { get; }

    /// <summary>The weights, row-major: <see cref="Outputs"/> rows of <see cref="Inputs"/>.</summary>
    public double[] Weights { get; }

    /// <summary>The observation mean subtracted before the weights.</summary>
    public double[] Mean { get; }

    /// <summary>The observation scale divided by before the weights (a standard deviation plus 1e-8).</summary>
    public double[] Scale { get; }

    /// <summary>What produced the policy (training settings and scores), kept with it in its file.</summary>
    public JsonObject Info { get; set; } = new();

    /// <summary>A zero policy (every action 0) with the identity normalization.</summary>
    /// <param name="inputs">The observation size.</param>
    /// <param name="outputs">The action size.</param>
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

    /// <summary>The action for an observation.</summary>
    /// <param name="observation">The observation.</param>
    /// <param name="action">Receives the float32 action.</param>
    public void Act(ReadOnlySpan<double> observation, Span<float> action) => Act(Weights, Mean, Scale, observation, action);

    /// <summary>
    ///     The policy's arithmetic with explicit weights and normalization, as training evaluates its perturbed copies.
    /// </summary>
    /// <param name="weights">Row-major weights, <c>action.Length</c> rows of <c>observation.Length</c>.</param>
    /// <param name="mean">The observation mean.</param>
    /// <param name="scale">The observation scale.</param>
    /// <param name="observation">The observation.</param>
    /// <param name="action">Receives the action: each output clipped to [-1, 1] in double, then rounded to float32.</param>
    public static void Act(ReadOnlySpan<double> weights, ReadOnlySpan<double> mean, ReadOnlySpan<double> scale,
                           ReadOnlySpan<double> observation, Span<float> action) {
        int n = observation.Length;
        for (int i = 0; i < action.Length; i++) {
            double acc = 0;
            for (int j = 0; j < n; j++)
                acc += weights[i * n + j] * ((observation[j] - mean[j]) / scale[j]);
            // written out rather than Math.Clamp so that a NaN passes through exactly as Python's comparisons pass it
            action[i] = (float)(acc < -1.0 ? -1.0 : acc > 1.0 ? 1.0 : acc);
        }
    }

    /// <summary>Reads a policy file written by <see cref="Save"/>.</summary>
    /// <param name="path">The JSON file.</param>
    /// <returns>The policy.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="JsonException">The file is not a policy.</exception>
    /// <exception cref="InvalidDataException">The arrays do not match the sizes.</exception>
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

    /// <summary>Copies a JSON number array into a buffer of the expected length.</summary>
    /// <param name="target">The buffer.</param>
    /// <param name="node">The array node.</param>
    /// <param name="name">Its name, for the error.</param>
    /// <param name="path">The file, for the error.</param>
    /// <exception cref="InvalidDataException">The array is missing or has the wrong length.</exception>
    private static void Fill(double[] target, JsonNode? node, string name, string path) {
        if (node is not JsonArray array || array.Count != target.Length)
            throw new InvalidDataException($"{path}: '{name}' must hold {target.Length} numbers");
        for (int i = 0; i < target.Length; i++)
            target[i] = array[i]!.GetValue<double>();
    }

    /// <summary>
    ///     Writes the policy as JSON. Doubles are written in their shortest round-trip form, so a reader in any language gets
    ///     the same bits back.
    /// </summary>
    /// <param name="path">The file, overwritten.</param>
    /// <exception cref="IOException">The file cannot be written.</exception>
    public void Save(string path) {
        var root = new JsonObject {
            ["format"] = "a = clip(sum_j W[i,j]*((s_j - mean_j)/scale_j), -1, 1) as float32; W row-major, outputs x inputs",
            ["inputs"] = Inputs,
            ["outputs"] = Outputs,
            ["weights"] = new JsonArray(Weights.Select(v => (JsonNode)v).ToArray()),
            ["mean"] = new JsonArray(Mean.Select(v => (JsonNode)v).ToArray()),
            ["scale"] = new JsonArray(Scale.Select(v => (JsonNode)v).ToArray()),
            ["info"] = Info.DeepClone(),
        };
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
    }
}
