using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using NumSharp;

namespace MujocoPoc.Play;

/// <summary>
///     A published expert agent read from <c>expert.json</c> (DR-109): the deterministic actor network of a
///     Stable-Baselines3 model, which maps an observation to the action its <c>predict(deterministic=True)</c> takes.
/// </summary>
/// <remarks>
///     <para>
///     The file holds the network's dense layers (base64 of little-endian float32 values, each weight row-major
///     <c>outputs x inputs</c> as PyTorch's <c>nn.Linear</c> stores it), the activation after each, how the last layer's
///     values become the action (<c>tanh-rescale</c> for SAC and TQC, <c>clip</c> for PPO), and the constructor
///     arguments the env must be made with for the agent to see what it was trained on (<c>make_kwargs</c>). It is a copy
///     of the agent in <c>tests/Gym.Tests/Agents</c>, written by <c>tools/agents/import_agents.py</c>, and this reader a
///     copy of that folder's <c>ExpertPolicy</c>: every example keeps its agent's code to itself (DR-74).
///     </para>
///     <para>
///     It computes as PyTorch does: the observation is cast to float32, each dot product is summed in float64, and every
///     layer's output is rounded to float32. An action can still differ from PyTorch's in its last bit, and over a long
///     episode of chaotic physics a last bit grows into a different trajectory, as good as the first.
///     </para>
/// </remarks>
internal sealed class ExpertPolicy {
    /// <summary>The widest layer the stack scratch holds; the published agents' widest is 400.</summary>
    private const int MaxWidth = 1024;

    /// <summary>The dense layers, input layer first.</summary>
    private readonly Layer[] _layers;

    /// <summary>The action space's lower bounds, as the model saved them.</summary>
    private readonly float[] _low;

    /// <summary>The action space's upper bounds.</summary>
    private readonly float[] _high;

    /// <summary>Whether the output is squashed by tanh and rescaled to the bounds (SAC, TQC), rather than clipped (PPO).</summary>
    private readonly bool _squash;

    /// <summary>The constructor arguments the agent's env needs (<c>make_kwargs</c>); none for most agents.</summary>
    private readonly Dictionary<string, object?> _makeKwargs;

    /// <summary>Creates an agent from its parsed parts; <see cref="Load"/> is the way in.</summary>
    /// <param name="layers">The dense layers.</param>
    /// <param name="low">The lower bounds.</param>
    /// <param name="high">The upper bounds.</param>
    /// <param name="squash">Whether the output is tanh-squashed.</param>
    /// <param name="makeKwargs">The constructor arguments.</param>
    private ExpertPolicy(Layer[] layers, float[] low, float[] high, bool squash, Dictionary<string, object?> makeKwargs) {
        _layers = layers;
        _low = low;
        _high = high;
        _squash = squash;
        _makeKwargs = makeKwargs;
    }

    /// <summary>Reads the agent's file.</summary>
    /// <param name="path">The <c>policy.json</c> file.</param>
    /// <returns>The agent.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="JsonException">It is not JSON.</exception>
    /// <exception cref="InvalidDataException">
    ///     It is not a <c>gym.net-agent/1</c> file, a layer's sizes do not chain, an array has the wrong length, or an
    ///     activation, output mode or constructor argument is of an unknown kind.
    /// </exception>
    public static ExpertPolicy Load(string path) {
        JsonNode root = JsonNode.Parse(File.ReadAllText(path)) ?? throw new JsonException($"{path}: empty");
        if (root["format"]?.GetValue<string>() != "gym.net-agent/1")
            throw new InvalidDataException($"{path}: not a gym.net-agent/1 file");
        Layer[] layers = root["layers"]!.AsArray().Select(n => Layer.Parse(n!, path)).ToArray();
        for (int i = 1; i < layers.Length; i++)
            if (layers[i].Inputs != layers[i - 1].Outputs)
                throw new InvalidDataException($"{path}: layer {i} takes {layers[i].Inputs} inputs, layer {i - 1} gives {layers[i - 1].Outputs}");
        if (layers.Any(l => l.Outputs > MaxWidth))
            throw new InvalidDataException($"{path}: a layer is wider than {MaxWidth}");
        float[] Bounds(string name) {
            if (root[name] is not JsonArray array || array.Count != layers[^1].Outputs)
                throw new InvalidDataException($"{path}: '{name}' must hold {layers[^1].Outputs} numbers");
            return array.Select(v => (float)v!.GetValue<double>()).ToArray();
        }

        bool squash = root["output"]?.GetValue<string>() switch {
            "tanh-rescale" => true,
            "clip" => false,
            var other => throw new InvalidDataException($"{path}: unknown output '{other}'"),
        };
        var kwargs = new Dictionary<string, object?>();
        if (root["make_kwargs"] is JsonObject make)
            foreach (var (name, value) in make)
                kwargs[name] = value?.GetValueKind() switch {
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    JsonValueKind.Number => value.GetValue<double>(),
                    JsonValueKind.String => value.GetValue<string>(),
                    _ => throw new InvalidDataException($"{path}: make_kwargs '{name}' must be a bool, number or string"),
                };
        return new ExpertPolicy(layers, Bounds("action_low"), Bounds("action_high"), squash, kwargs);
    }

    /// <summary>
    ///     The keyword arguments to make the env with: the agent's own (<c>make_kwargs</c>: HumanoidStandup's expert sees
    ///     positions and velocities only), and for a window, human mode and the view's arguments.
    /// </summary>
    /// <param name="human">Whether the env renders into a window (<c>render_mode="human"</c>).</param>
    /// <param name="view">Arguments that change only what the window shows (a camera), added in human mode; or null.</param>
    /// <returns>A new dictionary for <c>Gymnasium.Make</c>.</returns>
    public Dictionary<string, object?> MakeArguments(bool human, IReadOnlyDictionary<string, object?>? view = null) {
        var arguments = new Dictionary<string, object?>(_makeKwargs);
        if (human) {
            arguments["render_mode"] = "human";
            if (view is not null)
                foreach (var (name, value) in view)
                    arguments[name] = value;
        }
        return arguments;
    }

    /// <summary>The action for an observation.</summary>
    /// <param name="observation">The env's observation (float64 for MuJoCo, float32 for Box2D).</param>
    /// <returns>A new float32 array of one command per motor.</returns>
    /// <exception cref="ArgumentException">The observation's length is not the network's.</exception>
    public NDArray Act(NDArray observation) {
        float[] input;
        // Cast to float32 first, as Stable-Baselines3 casts what the env returns.
        using (NDArray values = observation.astype(NPTypeCode.Single))
            input = values.ToArray<float>();
        if (input.Length != _layers[0].Inputs)
            throw new ArgumentException($"the agent reads {_layers[0].Inputs} numbers, the observation has {input.Length}", nameof(observation));
        // Two stack buffers alternate as layer input and output: a forward pass allocates nothing but the action.
        Span<float> a = stackalloc float[MaxWidth], b = stackalloc float[MaxWidth];
        scoped ReadOnlySpan<float> x = input;
        for (int l = 0; l < _layers.Length; l++) {
            Span<float> y = (l % 2 == 0 ? a : b)[.._layers[l].Outputs];
            _layers[l].Forward(x, y);
            x = y;
        }
        var action = new float[_low.Length];
        // Stable-Baselines3's deterministic predict: unscale a tanh-squashed output, or clip an unsquashed one.
        for (int i = 0; i < action.Length; i++)
            action[i] = _squash ? _low[i] + 0.5f * (MathF.Tanh(x[i]) + 1f) * (_high[i] - _low[i]) : Math.Clamp(x[i], _low[i], _high[i]);
        return np.array(action);
    }

    /// <summary>One dense layer: <c>y = activation(W x + b)</c> in float32, each sum taken in float64.</summary>
    /// <param name="Inputs">Its input width.</param>
    /// <param name="Outputs">Its output width.</param>
    /// <param name="Weight">The weights, row-major <c>Outputs x Inputs</c>.</param>
    /// <param name="Bias">The biases.</param>
    /// <param name="Activation"><c>relu</c>, <c>tanh</c> or <c>none</c>.</param>
    private sealed record Layer(int Inputs, int Outputs, float[] Weight, float[] Bias, string Activation) {
        /// <summary>Computes the layer's output.</summary>
        /// <param name="input">The input, <see cref="Inputs"/> values.</param>
        /// <param name="output">Receives <see cref="Outputs"/> values.</param>
        public void Forward(ReadOnlySpan<float> input, Span<float> output) {
            for (int o = 0; o < Outputs; o++) {
                double sum = Bias[o];
                for (int i = 0; i < Inputs; i++)
                    sum += (double)Weight[o * Inputs + i] * input[i];
                // Rounded to float32 before the activation, as a float32 tensor holds the pre-activation.
                float y = (float)sum;
                output[o] = Activation switch {
                    "relu" => y > 0 ? y : 0,
                    "tanh" => MathF.Tanh(y),
                    _ => y,
                };
            }
        }

        /// <summary>Reads a layer from its JSON object.</summary>
        /// <param name="node">The layer object.</param>
        /// <param name="path">The file, for errors.</param>
        /// <returns>The layer.</returns>
        /// <exception cref="InvalidDataException">The sizes and the arrays disagree, or the activation is unknown.</exception>
        /// <exception cref="FormatException">An array is not base64.</exception>
        public static Layer Parse(JsonNode node, string path) {
            int inputs = node["inputs"]!.GetValue<int>(), outputs = node["outputs"]!.GetValue<int>();
            string activation = node["activation"]!.GetValue<string>();
            if (activation is not ("relu" or "tanh" or "none"))
                throw new InvalidDataException($"{path}: unknown activation '{activation}'");
            float[] Decode(string field, int count) {
                byte[] bytes = Convert.FromBase64String(node[field]!.GetValue<string>());
                if (bytes.Length != count * 4)
                    throw new InvalidDataException($"{path}: '{field}' holds {bytes.Length / 4} values, not {count}");
                var values = new float[count];
                // Little-endian in the file; .NET runs little-endian on every platform it supports.
                Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
                return values;
            }

            return new Layer(inputs, outputs, Decode("weight", inputs * outputs), Decode("bias", outputs), activation);
        }
    }
}
