#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Gym.Agents;

/// <summary>
///     A published expert agent for one of Gym.NET's ids, read from <c>tests/Gym.Tests/Agents/&lt;id&gt;.json</c>: the
///     deterministic actor network of a Stable-Baselines3 model (Farama's MuJoCo experts, RL Zoo's hardcore walker),
///     mapping an observation to the action its <c>predict(deterministic=True)</c> takes (DR-109).
/// </summary>
/// <remarks>
///     <para>
///     The file holds the actor's dense layers (base64 of little-endian float32, each weight row-major
///     <c>outputs x inputs</c> as PyTorch's <c>nn.Linear</c> stores it), the activation after each, and how the last
///     layer's values become the action: <c>tanh-rescale</c> (SAC and TQC: <c>low + (tanh(y) + 1) / 2 * (high - low)</c>)
///     or <c>clip</c> (PPO: <c>y</c> clipped to the bounds). <c>tools/agents/import_agents.py</c> writes it from the pinned
///     Hugging Face revision recorded under <c>source</c>.
///     </para>
///     <para>
///     The computation follows PyTorch's float32 forward pass: the observation is cast to float32 first, as
///     Stable-Baselines3 does, and every layer's output is rounded to float32. Each dot product is summed in float64,
///     so a result can differ from PyTorch's blocked float32 sums in the last bit or two; <see cref="Checks"/> holds
///     PyTorch's own actions for four observations to bound that difference. A last-bit difference in an action can grow
///     over a long MuJoCo episode into a visibly different (equally good) trajectory, so an agent's episode in Gym.NET is
///     the agent's, not a replay of Python's.
///     </para>
///     <para>
///     Instances are immutable and <see cref="Act(ReadOnlySpan{float}, Span{float})"/> keeps its scratch on the stack,
///     so one policy can drive many envs on many threads at once.
///     </para>
/// </remarks>
internal sealed class ExpertPolicy {
    /// <summary>The largest hidden layer the stack scratch holds; the imported agents' widest is 400.</summary>
    private const int MaxWidth = 1024;

    /// <summary>The dense layers, input layer first.</summary>
    private readonly Layer[] _layers;

    /// <summary>Creates a policy from parsed parts; <see cref="Load"/> is the way in.</summary>
    /// <param name="id">The registered id the agent plays.</param>
    /// <param name="makeKwargs">The constructor arguments it must be made with.</param>
    /// <param name="layers">The dense layers.</param>
    /// <param name="low">The action space's lower bounds.</param>
    /// <param name="high">The action space's upper bounds.</param>
    /// <param name="squash">Whether the output is <c>tanh-rescale</c> (else <c>clip</c>).</param>
    /// <param name="source">The provenance object.</param>
    /// <param name="checks">PyTorch's actions for fixed observations.</param>
    private ExpertPolicy(string id, IReadOnlyDictionary<string, object?> makeKwargs, Layer[] layers, float[] low, float[] high, bool squash,
        JsonObject source, IReadOnlyList<(float[] Observation, float[] Action)> checks) {
        Id = id;
        MakeKwargs = makeKwargs;
        _layers = layers;
        ActionLow = low;
        ActionHigh = high;
        Squash = squash;
        Source = source;
        Checks = checks;
    }

    /// <summary>The registered id the agent was trained on, e.g. <c>Humanoid-v5</c>.</summary>
    public string Id { get; }

    /// <summary>
    ///     Constructor arguments the env must be made with for the agent to see what it was trained on (HumanoidStandup's
    ///     expert observes positions and velocities only); empty for most. They change the observation, never the dynamics.
    /// </summary>
    public IReadOnlyDictionary<string, object?> MakeKwargs { get; }

    /// <summary>The observation length the network takes.</summary>
    public int ObservationSize => _layers[0].Inputs;

    /// <summary>The action length it returns.</summary>
    public int ActionSize => _layers[^1].Outputs;

    /// <summary>The action space's lower bounds, as the model saved them.</summary>
    public float[] ActionLow { get; }

    /// <summary>The action space's upper bounds.</summary>
    public float[] ActionHigh { get; }

    /// <summary>Whether the output is squashed by tanh and rescaled to the bounds (SAC, TQC), rather than clipped (PPO).</summary>
    public bool Squash { get; }

    /// <summary>Where the agent comes from: repository, pinned revision, algorithm, reported return, licence note.</summary>
    public JsonObject Source { get; }

    /// <summary>Observations with the actions PyTorch computes for them, from the import; the reader's correctness check.</summary>
    public IReadOnlyList<(float[] Observation, float[] Action)> Checks { get; }

    /// <summary>The number of weights and biases, for reporting.</summary>
    public int Parameters => _layers.Sum(l => l.Weight.Length + l.Bias.Length);

    /// <summary>The action for an observation, as float32.</summary>
    /// <param name="observation">The observation, <see cref="ObservationSize"/> values (float64 values are cast first).</param>
    /// <param name="action">Receives <see cref="ActionSize"/> values.</param>
    /// <exception cref="ArgumentException">A span has the wrong length.</exception>
    public void Act(ReadOnlySpan<float> observation, Span<float> action) {
        if (observation.Length != ObservationSize)
            throw new ArgumentException($"{Id}'s agent takes {ObservationSize} observation values, not {observation.Length}", nameof(observation));
        if (action.Length != ActionSize)
            throw new ArgumentException($"{Id}'s agent returns {ActionSize} actions, not {action.Length}", nameof(action));
        // Two stack buffers alternate as layer input and output, so a forward pass allocates nothing and threads share nothing.
        Span<float> a = stackalloc float[MaxWidth], b = stackalloc float[MaxWidth];
        // Scoped: after the first layer it points into the stack buffers, which must not outlive this call.
        scoped ReadOnlySpan<float> input = observation;
        for (int l = 0; l < _layers.Length; l++) {
            Layer layer = _layers[l];
            Span<float> output = (l % 2 == 0 ? a : b)[..layer.Outputs];
            layer.Forward(input, output);
            input = output;
        }
        for (int i = 0; i < action.Length; i++) {
            float y = input[i];
            float low = ActionLow[i], high = ActionHigh[i];
            // Stable-Baselines3's deterministic predict: unscale a tanh-squashed output (float32 arithmetic, as numpy does
            // it on float32 bounds), or clip an unsquashed one to the bounds.
            action[i] = Squash ? low + 0.5f * (MathF.Tanh(y) + 1f) * (high - low) : Math.Clamp(y, low, high);
        }
    }

    /// <summary>The action for a float64 observation (Gym.NET's MuJoCo observations), cast to float32 first.</summary>
    /// <param name="observation">The observation.</param>
    /// <returns>A new action array.</returns>
    /// <exception cref="ArgumentException">The observation has the wrong length.</exception>
    public float[] Act(ReadOnlySpan<double> observation) {
        if (observation.Length != ObservationSize)
            throw new ArgumentException($"{Id}'s agent takes {ObservationSize} observation values, not {observation.Length}", nameof(observation));
        var cast = new float[observation.Length];
        for (int i = 0; i < cast.Length; i++)
            cast[i] = (float)observation[i];
        var action = new float[ActionSize];
        Act(cast, action);
        return action;
    }

    /// <summary>
    ///     The largest difference between this reader's actions and PyTorch's on <see cref="Checks"/>, relative to each
    ///     action's range.
    /// </summary>
    /// <returns>The worst relative difference (0 when they agree exactly).</returns>
    public double CheckError() {
        double worst = 0;
        var action = new float[ActionSize];
        foreach (var (observation, expected) in Checks) {
            Act(observation, action);
            for (int i = 0; i < action.Length; i++)
                worst = Math.Max(worst, Math.Abs(action[i] - expected[i]) / Math.Max(1e-6, ActionHigh[i] - ActionLow[i]));
        }
        return worst;
    }

    /// <summary>Reads an agent file.</summary>
    /// <param name="path">The <c>&lt;id&gt;.json</c> file.</param>
    /// <returns>The policy.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="JsonException">The file is not JSON.</exception>
    /// <exception cref="InvalidDataException">
    ///     The format is not <c>gym.net-agent/1</c>, a layer's sizes do not chain, an array has the wrong length, or an
    ///     activation or output mode is unknown.
    /// </exception>
    public static ExpertPolicy Load(string path) => Parse(File.ReadAllText(path), path);

    /// <summary>
    ///     Reads an agent from its JSON text, for agents shipped as embedded resources (the test projects embed them, so a
    ///     test never depends on the working directory).
    /// </summary>
    /// <param name="json">The file's text.</param>
    /// <param name="path">Its name, for error messages.</param>
    /// <returns>The policy.</returns>
    /// <exception cref="JsonException">The text is not JSON.</exception>
    /// <exception cref="InvalidDataException">
    ///     The format is not <c>gym.net-agent/1</c>, a layer's sizes do not chain, an array has the wrong length, or an
    ///     activation or output mode is unknown.
    /// </exception>
    public static ExpertPolicy Parse(string json, string path) {
        JsonNode root = JsonNode.Parse(json) ?? throw new JsonException($"{path}: empty");
        if (root["format"]?.GetValue<string>() != "gym.net-agent/1")
            throw new InvalidDataException($"{path}: not a gym.net-agent/1 file");
        string id = root["id"]!.GetValue<string>();
        var layers = root["layers"]!.AsArray().Select(n => Layer.Parse(n!, path)).ToArray();
        for (int i = 1; i < layers.Length; i++)
            if (layers[i].Inputs != layers[i - 1].Outputs)
                throw new InvalidDataException($"{path}: layer {i} takes {layers[i].Inputs} inputs, layer {i - 1} gives {layers[i - 1].Outputs}");
        if (layers.Any(l => l.Outputs > MaxWidth))
            throw new InvalidDataException($"{path}: a layer is wider than {MaxWidth}");
        float[] low = Floats(root["action_low"], layers[^1].Outputs, "action_low", path);
        float[] high = Floats(root["action_high"], layers[^1].Outputs, "action_high", path);
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
        var checks = new List<(float[], float[])>();
        if (root["checks"] is JsonArray list)
            foreach (JsonNode? check in list)
                checks.Add((Floats(check!["observation"], layers[0].Inputs, "checks.observation", path),
                    Floats(check["action"], layers[^1].Outputs, "checks.action", path)));
        var source = root["source"] as JsonObject ?? new JsonObject();
        return new ExpertPolicy(id, kwargs, layers, low, high, squash, (JsonObject)source.DeepClone(), checks);
    }

    /// <summary>Reads a JSON number array of an expected length as float32.</summary>
    /// <param name="node">The array.</param>
    /// <param name="length">The expected length.</param>
    /// <param name="name">Its key, for the error.</param>
    /// <param name="path">The file, for the error.</param>
    /// <returns>The values.</returns>
    /// <exception cref="InvalidDataException">Not an array of that length.</exception>
    private static float[] Floats(JsonNode? node, int length, string name, string path) {
        if (node is not JsonArray array || array.Count != length)
            throw new InvalidDataException($"{path}: '{name}' must hold {length} numbers");
        return array.Select(v => (float)v!.GetValue<double>()).ToArray();
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
                ReadOnlySpan<float> row = Weight.AsSpan(o * Inputs, Inputs);
                double sum = Bias[o];
                for (int i = 0; i < Inputs; i++)
                    sum += (double)row[i] * input[i];
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
        public static Layer Parse(JsonNode node, string path) {
            int inputs = node["inputs"]!.GetValue<int>(), outputs = node["outputs"]!.GetValue<int>();
            string activation = node["activation"]!.GetValue<string>();
            if (activation is not ("relu" or "tanh" or "none"))
                throw new InvalidDataException($"{path}: unknown activation '{activation}'");
            float[] weight = Decode(node["weight"]!.GetValue<string>(), inputs * outputs, "weight", path);
            float[] bias = Decode(node["bias"]!.GetValue<string>(), outputs, "bias", path);
            return new Layer(inputs, outputs, weight, bias, activation);
        }

        /// <summary>Decodes base64 little-endian float32 values.</summary>
        /// <param name="text">The base64 text.</param>
        /// <param name="count">The expected number of values.</param>
        /// <param name="name">The field, for the error.</param>
        /// <param name="path">The file, for the error.</param>
        /// <returns>The values.</returns>
        /// <exception cref="InvalidDataException">The decoded length is not <paramref name="count"/> values.</exception>
        /// <exception cref="FormatException">The text is not base64.</exception>
        private static float[] Decode(string text, int count, string name, string path) {
            byte[] bytes = Convert.FromBase64String(text);
            if (bytes.Length != count * 4)
                throw new InvalidDataException($"{path}: '{name}' holds {bytes.Length / 4} values, not {count}");
            var values = new float[count];
            // The file is little-endian whatever the machine; .NET's supported platforms are little-endian too, but the
            // explicit read keeps the format's contract visible.
            for (int i = 0; i < count; i++)
                values[i] = BitConverter.ToSingle(BitConverter.IsLittleEndian ? bytes.AsSpan(i * 4, 4) : bytes.AsSpan(i * 4, 4).ToArray().Reverse().ToArray());
            return values;
        }
    }
}
