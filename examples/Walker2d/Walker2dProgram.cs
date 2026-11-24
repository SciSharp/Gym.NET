#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.Walker2d {
    /// <summary>
    ///     Plays Walker2d-v5 with Farama's SAC expert, a published Stable-Baselines3 agent. <c>dotnet run --project
    ///     examples/Walker2d -c Release</c> plays one seeded episode headless and prints its steps and return; <c>--
    ///     --render</c> plays it in a window, and <c>-- --record &lt;folder&gt;</c> records the episode for
    ///     the env's page on the docs site (DR-112).
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
        ///     this env cannot be played from the keyboard; <c>--record &lt;folder&gt;</c> records the docs page's media
        ///     from seed 0, whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played or the recording was written, 2 when an option's value is not a whole number,
        ///     <c>--record</c> names no folder, or for <c>--play</c>.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> sets Avalonia's viewer before the env first draws.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     <c>--record</c> could not run ffmpeg, which <see cref="FfmpegVideoWriter"/> needs on PATH.
        /// </exception>
        /// <exception cref="IOException">
        ///     <c>--record</c> could not create the folder or write one of its files, or <c>policy.json</c> cannot be read.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            int record = Array.IndexOf(args, "--record");
            if (record >= 0) {
                // The folder is required: the page's media live in the docs tree, and a default would write them where no
                // page reads them.
                if (record + 1 >= args.Length || args[record + 1].StartsWith("--", StringComparison.Ordinal)) {
                    Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/mujoco/media");
                    return 2;
                }
                (int recordedSteps, double recordedReturn) = Record(args[record + 1]);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{Id}, seed 0: {recordedSteps} steps, return {recordedReturn:F1}, recorded into {args[record + 1]}"));
                return 0;
            }
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                Console.Error.WriteLine($"{Id}: its frames take about 0.2 s each to draw in software and a step is 8 ms of simulated time, so keyboard play would run about 25 times slower than real time; --render shows the agent instead.");
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

        // ============================================================================ record (the docs page's media)

        /// <summary>
        ///     The video's view (DR-112): the model's own <c>track</c> camera, 3 m to the side, which follows the body's centre of mass, as the front-page showreel plays it (DR-74).
        /// </summary>
        private static readonly Dictionary<string, object?> RecordView = new Dictionary<string, object?> { ["camera_name"] = "track" };

        /// <summary>
        ///     The trace's state columns, each a value of the simulation read before the step that acts on it: a name, the
        ///     array (<c>q</c> for <c>qpos</c>, <c>v</c> for <c>qvel</c>) and the index. Walker2d's: the torso's x, height and angle (<c>qpos[0..2]</c>), its forward velocity (<c>qvel[0]</c>), and the right leg's then the left leg's thigh, leg and foot angles (<c>qpos[3..8]</c>).
        /// </summary>
        private static readonly (string Name, char Array, int Index)[] TraceState = {
            ("x", 'q', 0),
            ("z", 'q', 1),
            ("angle", 'q', 2),
            ("vx", 'v', 0),
            ("thigh", 'q', 3),
            ("leg", 'q', 4),
            ("foot", 'q', 5),
            ("thigh_left", 'q', 6),
            ("leg_left", 'q', 7),
            ("foot_left", 'q', 8),
        };

        /// <summary>The info's reward terms the trace keeps for each step, in the info's own names.</summary>
        private static readonly string[] RewardTerms = { "reward_forward", "reward_ctrl", "reward_survive" };

        /// <summary>
        ///     How it works' frame (DR-114): 480×600 pixels of the reset pose without noise (<c>reset_noise_scale</c> 0), seen by MuJoCo's free camera from the side, 2.4 m from a point 0.8 m above the floor, at azimuth 90° and elevation -5°: the two legs, straight and side by side in the plane, one behind the other.
        /// </summary>
        private const int FigureWidth = 480, FigureHeight = 600;

        /// <summary>The figure's free camera: distance (m), azimuth and elevation (degrees), as <c>default_camera_config</c> takes them.</summary>
        private const double FigureDistance = 2.4, FigureAzimuth = 90.0, FigureElevation = -5.0;

        /// <summary>The point the figure's camera looks at, in world coordinates.</summary>
        private static readonly double[] FigureLookat = { 0.0, 0.0, 0.8 };

        /// <summary>
        ///     The joints How it works labels: each joint, the body it moves, the joint's anchor in that body's frame (its
        ///     <c>pos</c> in the model, which this model's bodies keep unrotated at the reset pose without noise, so the
        ///     anchor is the body's origin plus this offset), the observation's item for its angle (-1 for none) and the
        ///     action's item for its motor (-1 for none). Walker2d's model (<c>walker2d_v5.xml</c>): the right leg's hip, knee and ankle, then the left leg's, which share their places at the reset pose.
        /// </summary>
        private static readonly (string Joint, string Body, double[] Offset, int Observation, int Action)[] FigureJoints = {
            ("thigh_joint", "thigh", new[] { 0.0, 0.0, 0.0 }, 2, 0),
            ("leg_joint", "leg", new[] { 0.0, 0.0, 0.25 }, 3, 1),
            ("foot_joint", "foot", new[] { -0.2, 0.0, 0.1 }, 4, 2),
            ("thigh_left_joint", "thigh_left", new[] { 0.0, 0.0, 0.0 }, 5, 3),
            ("leg_left_joint", "leg_left", new[] { 0.0, 0.0, 0.25 }, 6, 4),
            ("foot_left_joint", "foot_left", new[] { -0.2, 0.0, 0.1 }, 7, 5),
        };

        /// <summary>
        ///     Records the docs page's media (DR-112) into <paramref name="folder"/>: the seed-0 episode as
        ///     <c>Walker2d-v5.episode.mp4</c> (one 480×480 frame per step at the env's own frame rate, through Gymnasium's
        ///     <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>), every step as
        ///     <c>Walker2d-v5.episode.json</c>, and How it works' frame as <c>Walker2d-v5.figure.png</c> with
        ///     <c>Walker2d-v5.figure.json</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/mujoco/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written, or the agent's file cannot be read.</exception>
        /// <exception cref="InvalidDataException">The agent's file is not a <c>gym.net-agent/1</c> agent.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the state the agent acted on, its commands and the step's reward with its terms:
        ///     <c>step, x, z, angle, vx, thigh, leg, foot, thigh_left, leg_left, foot_left, a0 ... a5, reward_forward, reward_ctrl, reward_survive, reward</c>. The state is
        ///     the simulation's own, rounded to 5 decimals (<see cref="TraceState"/>); the commands are the network's float32
        ///     values, each written as the float64 that holds it exactly, so the trace replays the episode bit for bit
        ///     (<c>action_dtype</c> says to cast them back); the reward terms are the step's info. The video's frame <c>t</c>
        ///     shows the state of row <c>t</c>, and its last frame the state after the last step, which the JSON keeps as
        ///     <c>final</c>.
        ///     </para>
        ///     <para>
        ///     Gym.NET draws MuJoCo's frames in software (DR-62), so the recording takes minutes. RecordVideo names its file
        ///     <c>Walker2d-v5-episode-0.mp4</c> and writes it when the env closes; it records into a scratch folder inside
        ///     <paramref name="folder"/> and the file is moved to the page's name, so a recording that fails halfway leaves
        ///     the published video as it was. <c>[NDScoped]</c> (DR-80) releases every array the episode creates.
        ///     </para>
        /// </remarks>
        [NDScoped]
        private static (int Steps, double Return) Record(string folder) {
            ExpertPolicy policy = ExpertPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policy.json"));
            Directory.CreateDirectory(folder);
            string scratch = Path.Combine(folder, ".recording");
            var rows = new List<double[]>();
            double total = 0;
            double[] final;
            long fps;
            // The agent's own constructor arguments, with the recorder's render mode and the page's view.
            Dictionary<string, object?> arguments = policy.MakeArguments(human: false);
            arguments["render_mode"] = "rgb_array";
            foreach (var (name, value) in RecordView)
                arguments[name] = value;
            using (Env<NDArray, NDArray> made = Gymnasium.Make<NDArray, NDArray>(Id, arguments))
            using (var env = new RecordVideo<NDArray, NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: Id,
                       videoWriter: new FfmpegVideoWriter())) {
                var mujoco = (MujocoEnv)env.Unwrapped;
                fps = Convert.ToInt64(env.Metadata["render_fps"], CultureInfo.InvariantCulture);
                NDArray observation = env.Reset(seed: 0).Observation;
                for (int t = 0; ; t++) {
                    NDArray action = policy.Act(observation);
                    // float32 to float64 is exact, so the JSON's number casts back to the very float32 the agent chose.
                    double[] commands = action.ToArray<float>().Select(v => (double)v).ToArray();
                    double[] state = RecordState(mujoco);
                    StepResult<NDArray> step = env.Step(action);
                    total += step.Reward;
                    double[] terms = RewardTerms.Select(term => Math.Round(step.Info.Get<double>(term), 5)).ToArray();
                    rows.Add(state.Prepend(t).Concat(commands).Concat(terms).Append(Math.Round(step.Reward, 5)).ToArray());
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = RecordState(mujoco);
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = 0, steps = rows.Count, @return = total, fps,
                agent = "Farama's SAC expert (examples/Walker2d)", action_dtype = "float32",
                columns = new[] { "step", "x", "z", "angle", "vx", "thigh", "leg", "foot", "thigh_left", "leg_left", "foot_left", "a0", "a1", "a2", "a3", "a4", "a5", "reward_forward", "reward_ctrl", "reward_survive", "reward" },
                rows,
                final,
            }) + "\n");
            SaveFigure(folder);
            return (rows.Count, total);
        }

        /// <summary>The trace's view of a state: <see cref="TraceState"/>'s values of the simulation, rounded to 5 decimals.</summary>
        /// <param name="mujoco">The unwrapped env, whose simulation holds the state.</param>
        /// <returns>One value per <see cref="TraceState"/> entry.</returns>
        private static double[] RecordState(MujocoEnv mujoco) {
            // The simulation's own arrays through views: copied out at once, the views released with the episode's scope.
            double[] qpos = mujoco.Data.Qpos.ToArray<double>(), qvel = mujoco.Data.Qvel.ToArray<double>();
            return TraceState.Select(s => Math.Round(s.Array == 'q' ? qpos[s.Index] : qvel[s.Index], 5)).ToArray();
        }

        /// <summary>
        ///     Saves How it works' frame as <c>Walker2d-v5.figure.png</c>, and as <c>Walker2d-v5.figure.json</c> the camera and the pixel
        ///     of each point its labels mark: the root body and each joint, projected through MuJoCo's free-camera geometry.
        /// </summary>
        /// <param name="folder">The output folder.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     The frame is the env's own render of its reset without noise (<c>reset_noise_scale</c> 0), which every seed
        ///     then gives alike. The projection is <c>mjv_updateCamera</c>'s for a free camera: <c>forward = (cos e cos a,
        ///     cos e sin a, sin e)</c>, <c>up = (-sin e cos a, -sin e sin a, cos e)</c>, <c>right = (sin a, -cos a, 0)</c> for
        ///     azimuth <c>a</c> and elevation <c>e</c>, the eye at <c>lookat - distance forward</c>, and a vertical field of
        ///     view of 45° (MuJoCo's <c>vis.global.fovy</c>, which every model Gymnasium bundles keeps) over the frame's
        ///     height: a point at camera coordinates <c>(x, y, z)</c> is the pixel <c>(W/2 + f x/z, H/2 - f y/z)</c>,
        ///     <c>f = (H/2) / tan(22.5°)</c>. Reference:
        ///     <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_vis_visualize.c">engine_vis_visualize.c</see>.
        ///     </para>
        ///     <para>
        ///     The frame is Gym.NET's drawing of MuJoCo's renderer (DR-98): equal to Gymnasium's for the inverted pendulums,
        ///     a few pixels off for the other models; the projected points do not depend on the drawing.
        ///     </para>
        /// </remarks>
        [NDScoped]
        private static void SaveFigure(string folder) {
            var arguments = new Dictionary<string, object?> {
                ["render_mode"] = "rgb_array", ["width"] = FigureWidth, ["height"] = FigureHeight, ["reset_noise_scale"] = 0.0,
                ["default_camera_config"] = new Dictionary<string, object?> {
                    ["distance"] = FigureDistance, ["azimuth"] = FigureAzimuth, ["elevation"] = FigureElevation, ["lookat"] = FigureLookat,
                },
            };
            ExpertPolicy policy = ExpertPolicy.Load(Path.Combine(AppContext.BaseDirectory, "policy.json"));
            // The agent's constructor arguments change what it observes, never the pose: kept so the figure's observation is the page's.
            foreach (var (name, value) in policy.MakeArguments(human: false))
                arguments[name] = value;
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, arguments);
            NDArray observation = env.Reset(seed: 0).Observation;
            using RenderResult rendered = env.Render();
            NDArray frame = (NDArray)rendered;
            using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), FigureWidth, FigureHeight))
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            var mujoco = (MujocoEnv)env.Unwrapped;
            double a = FigureAzimuth * Math.PI / 180, e = FigureElevation * Math.PI / 180;
            double[] forward = { Math.Cos(e) * Math.Cos(a), Math.Cos(e) * Math.Sin(a), Math.Sin(e) };
            double[] up = { -Math.Sin(e) * Math.Cos(a), -Math.Sin(e) * Math.Sin(a), Math.Cos(e) };
            double[] right = { Math.Sin(a), -Math.Cos(a), 0 };
            double[] eye = Enumerable.Range(0, 3).Select(k => FigureLookat[k] - FigureDistance * forward[k]).ToArray();
            double focal = FigureHeight / 2.0 / Math.Tan(45.0 / 2 * Math.PI / 180);
            double[] Pixel(string body, double[] offset) {
                double[] p = mujoco.Data.Body(body).Xpos.ToArray<double>();
                double[] v = { p[0] + offset[0] - eye[0], p[1] + offset[1] - eye[1], p[2] + offset[2] - eye[2] };
                double z = v[0] * forward[0] + v[1] * forward[1] + v[2] * forward[2];
                double x = v[0] * right[0] + v[1] * right[1] + v[2] * right[2];
                double y = v[0] * up[0] + v[1] * up[1] + v[2] * up[2];
                return new[] { Math.Round(FigureWidth / 2.0 + focal * x / z, 2), Math.Round(FigureHeight / 2.0 - focal * y / z, 2) };
            }
            double[] o = observation.ToArray<double>();
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, width = FigureWidth, height = FigureHeight,
                camera = new { distance = FigureDistance, azimuth = FigureAzimuth, elevation = FigureElevation, lookat = FigureLookat, fovy = 45.0 },
                observation = o.Take(40).Select(v => Math.Round(v, 5)).ToArray(),
                torso = Pixel("torso", new double[3]),
                joints = FigureJoints.Select(j => new { name = j.Joint, body = j.Body, offset = j.Offset, observation = j.Observation, action = j.Action, pixel = Pixel(j.Body, j.Offset) }).ToArray(),
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
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
