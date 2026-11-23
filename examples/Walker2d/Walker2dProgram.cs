#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using NumSharp;

namespace Gym.Examples.Walker2d {
    /// <summary>
    ///     Plays Walker2d-v5 with Farama's SAC expert, a published Stable-Baselines3 agent. <c>dotnet run --project
    ///     examples/Walker2d -c Release</c> plays one seeded episode headless and prints its steps and return; <c>--
    ///     --render</c> plays it in a window.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <c>policy.json</c> holds the agent (<see cref="ExpertPolicy"/>): Farama's SAC expert, the agent the Farama
    ///     Foundation, Gymnasium's maintainers, trained on this env to record the Minari datasets, imported from
    ///     Hugging Face (<c>farama-minari/Walker2d-v5-SAC-expert</c> at revision <c>2b0ab40</c>) by
    ///     <c>tools/agents/import_agents.py</c>. A Stable-Baselines3 network of two dense layers of 256 with relu: each
    ///     step it casts the observation to float32 and squashes the result into the motors' range with tanh. Over
    ///     seeds 0-9 it scores 6,949.0 in Gym.NET (every seed runs all 1,000 steps); the model card reports 6956.61 +/-
    ///     15.95.
    ///     </para>
    ///     <para>
    ///     The model repository states no licence; its training code is MIT. Gym.NET commits the weights with this
    ///     attribution, on the maintainer's decision (DR-109).
    ///     </para>
    ///     <para>
    ///     The default seed, 0, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class Walker2dProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "Walker2d-v5";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 0;

        /// <summary>Plays the episodes the arguments ask for with the agent.</summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> is refused, with the reason
        ///     this env cannot be played from the keyboard. Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played, 2 when an option's value is not a whole number, or for <c>--play</c>.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> sets Avalonia's viewer before the env first draws.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                Console.Error.WriteLine($"{Id}: its frames take about 0.1 s each to draw in software and a step is 8 ms of simulated time, so keyboard play would run about 12 times slower than real time; --render shows the agent instead.");
                return 2;
            }
            // Human mode draws inside every reset and step into the app-wide viewer (DR-20); without one it raises
            // DependencyNotInstalled at the first frame, as Gymnasium does without pygame.
            if (render)
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
            double sum = 0;
            for (long episode = 0; episode < episodes; episode++) {
                (int steps, double total, string ending) = PlayEpisode(seed + episode, render);
                sum += total;
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{Id}, seed {seed + episode}: {steps} step{(steps == 1 ? "" : "s")}, return {total:F1}, {ending}"));
            }
            if (episodes > 1)
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"Mean return over {episodes} episodes: {sum / episodes:F1}"));
            return 0;
        }

        /// <summary>Plays one episode with the agent and reports how it went.</summary>
        /// <param name="seed">The reset seed, which draws the starting state.</param>
        /// <param name="render">
        ///     Whether to show the episode in a window, in human mode, paced at the env's frame rate.
        /// </param>
        /// <returns>
        ///     The steps taken, the return (float64, as every Gym.NET reward), and how the episode ended.
        /// </returns>
        /// <remarks>
        ///     The arrays a reset or a step returns belong to the caller (D19). The method is <c>[NDScoped]</c> (DR-80):
        ///     NumSharp.Build releases every array the episode creates (the results, the actions) when it returns. The env
        ///     is built and disposed inside it, the guides' rule for an env in a scoped method.
        /// </remarks>
        [NDScoped]
        private static (int Steps, double Return, string Ending) PlayEpisode(long seed, bool render) {
            // The agent first: its file names the constructor arguments the env needs for the agent to see what it
            // was trained on.
            ExpertPolicy policy = ExpertPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policy.json"));
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, policy.MakeArguments(render));
            NDArray observation = env.Reset(seed: seed).Observation;
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(policy.Act(observation));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? "fell over" : "truncated by the time limit");
            }
        }

        /// <summary>Reads a whole-number option's value, or the fallback when the option is absent.</summary>
        /// <param name="args">The command line.</param>
        /// <param name="name">The option, <c>--seed</c> or <c>--episodes</c>.</param>
        /// <param name="fallback">The value when the option is absent.</param>
        /// <param name="value">The value read, or <paramref name="fallback"/>.</param>
        /// <returns>
        ///     False, after printing why, when the option has no value or one that is not a non-negative whole number.
        /// </returns>
        private static bool TryNumber(string[] args, string name, long fallback, out long value) {
            value = fallback;
            int at = Array.IndexOf(args, name);
            if (at < 0)
                return true;
            // Gymnasium's seeds are non-negative, and so is a count of episodes: NumberStyles.None takes digits only.
            if (at + 1 < args.Length && long.TryParse(args[at + 1], NumberStyles.None, CultureInfo.InvariantCulture, out value))
                return true;
            Console.Error.WriteLine($"{name} needs a non-negative whole number");
            return false;
        }
    }

    /// <summary>
    ///     A published expert agent read from <c>policy.json</c> (DR-109): the deterministic actor network of a
    ///     Stable-Baselines3 model, which maps an observation to the action its <c>predict(deterministic=True)</c>
    ///     takes.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The file holds the network's dense layers (base64 of little-endian float32 values, each weight row-major
    ///     <c>outputs x inputs</c> as PyTorch's <c>nn.Linear</c> stores it), the activation after each, how the last
    ///     layer's values become the action (<c>tanh-rescale</c> for SAC and TQC, <c>clip</c> for PPO), and the
    ///     constructor arguments the env must be made with for the agent to see what it was trained on
    ///     (<c>make_kwargs</c>). It is a copy of the agent in <c>tests/Gym.Tests/Agents</c>, written by
    ///     <c>tools/agents/import_agents.py</c>, and this reader a copy of that folder's <c>ExpertPolicy</c>: every
    ///     example keeps its agent's code to itself (DR-74).
    ///     </para>
    ///     <para>
    ///     It computes as PyTorch does: the observation is cast to float32, each dot product is summed in float64, and
    ///     every layer's output is rounded to float32. An action can still differ from PyTorch's in its last bit, and
    ///     over a long episode of chaotic physics a last bit grows into a different trajectory, as good as the first.
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

        /// <summary>
        ///     Whether the output is squashed by tanh and rescaled to the bounds (SAC, TQC), rather than clipped (PPO).
        /// </summary>
        private readonly bool _squash;

        /// <summary>
        ///     The constructor arguments the agent's env needs (<c>make_kwargs</c>); none for most agents.
        /// </summary>
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
        ///     It is not a <c>gym.net-agent/1</c> file, a layer's sizes do not chain, an array has the wrong length, or
        ///     an activation, output mode or constructor argument is of an unknown kind.
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
        ///     The keyword arguments to make the env with: the agent's own (<c>make_kwargs</c>: HumanoidStandup's
        ///     expert sees positions and velocities only), and for a window, human mode and the view's arguments.
        /// </summary>
        /// <param name="human">Whether the env renders into a window (<c>render_mode="human"</c>).</param>
        /// <param name="view">
        ///     Arguments that change only what the window shows (a camera), added in human mode; or null.
        /// </param>
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
            /// <exception cref="InvalidDataException">
            ///     The sizes and the arrays disagree, or the activation is unknown.
            /// </exception>
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
}
