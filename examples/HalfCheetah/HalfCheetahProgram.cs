using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Gym;
using Gym.Environments;
using Gym.Environments.Envs.MuJoCo;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Gym.MuJoCo;

namespace MujocoPoc.Play;

/// <summary>
///     HalfCheetah-v5 played by Farama's TQC expert (DR-109): <c>dotnet run --project examples/HalfCheetah -c Release</c>
///     plays one seeded episode headless through <c>Gymnasium.Make</c> and prints its steps and return; <c>-- --render</c>
///     plays it in MuJoCo's window, <c>-- --autopilot</c> plays it on the autopilot instead (a linear policy trained by
///     Augmented Random Search), and <c>-- --play</c> opens the playable HalfCheetah, where the keyboard drives the six
///     motors and Tab hands them to the autopilot. Also trains the autopilot, records it to a video, and prints its exact
///     episode returns for comparison with Gymnasium on MuJoCo (replay_in_gymnasium.py).
/// </summary>
/// <remarks>
///     Usage:
///     <code>
///     HalfCheetah [--seed N] [--episodes N] [--render]    Farama's TQC expert through Gymnasium.Make (default; DR-74, DR-109)
///     HalfCheetah --record FOLDER                          the docs page's media from the expert's seed-0 episode (DR-112)
///     HalfCheetah --autopilot [--seed N] [--episodes N] [--render]    the same, on the autopilot
///     HalfCheetah --play [--seed N]                        the game, with the keyboard driving
///     HalfCheetah play [--seed N] [--manual]               the game, with the autopilot driving until Tab
///     HalfCheetah train [--iterations N] [--threads N] [--seed N] [--out policy.json]
///     HalfCheetah video FILE.mp4 [--seconds S] [--seed N]   needs ffmpeg on PATH
///     HalfCheetah frames FOLDER [--steps N] [--every N] [--seed N]
///     HalfCheetah score [--seeds A-B]
///     </code>
///     The model (half_cheetah.json, MuJoCo's compiled HalfCheetah as Gym.Environments embeds it), the expert (expert.json) and the
///     autopilot (policy.json) are read from the program's folder; <c>--expert FILE</c> and <c>--policy FILE</c> pick others.
///     <para>
///     The game moved here from <c>tools/parity/mujoco/halfcheetah_poc/play</c> (DR-74): it steps its own copy of the engine
///     and draws its own side view, cheap enough for real time. The default mode, the flags every example takes, plays the
///     product's env instead, the one <c>Gymnasium.Make</c> builds.
///     </para>
/// </remarks>
internal static class HalfCheetahProgram {
    private const string Title = "HalfCheetah-v5 - Gym.NET managed core, bit-exact with MuJoCo 4.0.0";

    /// <summary>Dispatches the command.</summary>
    /// <param name="args">The command and its options (see the class remarks).</param>
    /// <returns>0 on success, 1 on a failure the output explains, 2 on bad usage.</returns>
    private static int Main(string[] args) {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        var options = new Options(args.SkipWhile(a => !a.StartsWith("--", StringComparison.Ordinal)).ToArray());
        // With no command, the flags every example takes choose: --play opens the game with the keyboard driving, and
        // otherwise the autopilot plays seeded episodes through Gymnasium.Make, headless or in MuJoCo's window.
        bool commanded = args.Length > 0 && !args[0].StartsWith("--", StringComparison.Ordinal);
        string command = commanded ? args[0] : options.Flag("play") ? "play" : "episode";
        string folder = AppContext.BaseDirectory;
        MjModel model = MjModel.Load(Path.Combine(folder, "half_cheetah.json"));
        string policyPath = options.Text("policy", Path.Combine(folder, "policy.json"));
        string expertPath = options.Text("expert", Path.Combine(folder, "expert.json"));
        try {
            switch (command) {
                case "episode":
                    // --record writes the docs page's media from the expert's seed-0 episode (DR-112), whatever --seed says.
                    if (options.Flag("record"))
                        return Record(expertPath, options.Text("record", ""));
                    // The expert plays by default (DR-109); --autopilot keeps the linear policy whose returns Gymnasium
                    // reproduces to the last bit (replay_in_gymnasium.py).
                    return options.Flag("autopilot")
                        ? Episodes(policyPath, options.Int("seed", 0), options.Int("episodes", 1), options.Flag("render"))
                        : ExpertEpisodes(expertPath, options.Int("seed", 0), options.Int("episodes", 1), options.Flag("render"));
                case "play":
                    // --play hands the motors to the keyboard from the start; the play command keeps the autopilot driving.
                    bool manual = options.Flag("manual") || !commanded;
                    int code = Play(model, File.Exists(policyPath) ? LinearPolicy.Load(policyPath) : null, options.Int("seed", 0), manual);
                    // the Avalonia UI thread keeps the process alive; end it explicitly
                    Environment.Exit(code);
                    return code;
                case "train":
                    return Train(model, options);
                case "video":
                    return Video(model, LinearPolicy.Load(policyPath), Argument(args, "video"), options.Double("seconds", 30), options.Int("seed", 0));
                case "frames":
                    return Frames(model, LinearPolicy.Load(policyPath), Argument(args, "frames"), options.Int("steps", 200), options.Int("every", 20), options.Int("seed", 0));
                case "score":
                    return Score(model, LinearPolicy.Load(policyPath), options.Text("seeds", "0-4"));
                default:
                    Console.Error.WriteLine($"unknown command '{command}': play, train, video, frames or score (or none, for the episode flags)");
                    return 2;
            }
        } catch (ArgumentException e) {
            Console.Error.WriteLine(e.Message);
            return 2;
        }
    }

    /// <summary>
    ///     The mode every example shares (DR-74), as the expert plays it (DR-109): seeded episodes of HalfCheetah-v5 played
    ///     by Farama's TQC expert through <c>Gymnasium.Make</c>, headless or in MuJoCo's window, each one's steps and return
    ///     printed.
    /// </summary>
    /// <param name="expertPath">The expert's file, a copy of <c>tests/Gym.Tests/Agents/HalfCheetah-v5.json</c>.</param>
    /// <param name="seed">The first reset seed.</param>
    /// <param name="episodes">How many episodes, on consecutive seeds.</param>
    /// <param name="render">Whether to show them in MuJoCo's window, human mode with Gymnasium's controls (DR-70).</param>
    /// <returns>0.</returns>
    /// <exception cref="IOException">The expert's file cannot be read.</exception>
    /// <exception cref="InvalidDataException">The file is not a <c>gym.net-agent/1</c> agent.</exception>
    /// <remarks>
    ///     Seed 0 returns about 17,600, three times the autopilot's 5,797: at full gallop the runner covers 17 m a second.
    ///     Each episode is one <see cref="PlayExpertEpisode"/> call, whose scope releases its arrays when it returns.
    /// </remarks>
    private static int ExpertEpisodes(string expertPath, int seed, int episodes, bool render) {
        ExpertPolicy expert = ExpertPolicy.Load(expertPath);
        // Human mode draws inside every reset and step into the app-wide viewer (DR-20).
        if (render)
            Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
        double sum = 0;
        for (int episode = 0; episode < episodes; episode++) {
            (int steps, double total) = PlayExpertEpisode(expert, seed + episode, render);
            sum += total;
            Console.WriteLine($"HalfCheetah-v5, seed {seed + episode}: {steps} steps, return {total:F1}, truncated by the time limit");
        }
        if (episodes > 1)
            Console.WriteLine($"Mean return over {episodes} episodes: {sum / episodes:F1}");
        return 0;
    }

    /// <summary>One episode of HalfCheetah-v5 played by the expert, through <c>Gymnasium.Make</c>.</summary>
    /// <param name="expert">The expert.</param>
    /// <param name="seed">The reset seed.</param>
    /// <param name="render">Whether to show the episode in MuJoCo's window.</param>
    /// <returns>The steps taken and the return.</returns>
    /// <remarks>
    ///     <c>[NDScoped]</c> (DR-80): NumSharp.Build releases every array the episode creates (the results, the actions)
    ///     when it returns. The env is built and disposed inside it, the guides' rule for an env in a scoped method.
    /// </remarks>
    [NDScoped]
    private static (int Steps, double Return) PlayExpertEpisode(ExpertPolicy expert, int seed, bool render) {
        using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5", expert.MakeArguments(render));
        (NDArray observation, Info info) = env.Reset(seed: seed);
        int steps = 0;
        double total = 0;
        while (true) {
            StepResult<NDArray> step = env.Step(expert.Act(observation));
            observation = step.Observation;
            steps++;
            total += step.Reward;
            // HalfCheetah never terminates: every episode is cut by the 1,000-step time limit.
            if (step.Terminated || step.Truncated)
                return (steps, total);
        }
    }

    // ================================================================================== record (the docs page's media)

    /// <summary>The id the docs page's media come from.</summary>
    private const string Id = "HalfCheetah-v5";

    /// <summary>The docs page's episode (DR-112): the expert from seed 0, the episode the showreel plays.</summary>
    private const int RecordSeed = 0;

    /// <summary>
    ///     How it works' frame (DR-114): 760×380 pixels of the reset pose without noise (<c>reset_noise_scale</c> 0), seen
    ///     by MuJoCo's free camera from the side, 1.6 m from a point 0.45 m above the floor, at azimuth 90° and elevation -8°.
    /// </summary>
    private const int FigureWidth = 760, FigureHeight = 380;

    /// <summary>The figure's camera: distance (m), azimuth and elevation (degrees), as <c>default_camera_config</c> takes them.</summary>
    private const double FigureDistance = 1.6, FigureAzimuth = 90, FigureElevation = -8;

    /// <summary>The point the figure's camera looks at, in world coordinates: above the torso's start, 0.45 m up.</summary>
    private static readonly double[] FigureLookat = { 0, 0, 0.45 };

    /// <summary>
    ///     The joints How it works labels, each with the body it moves (whose frame origin is the joint's anchor in this
    ///     model: every hinge sits at its body's origin), the observation's item for its angle and the action's item for its
    ///     motor. HalfCheetah's model (<c>half_cheetah.xml</c>): the back leg's thigh, shin and foot, then the front leg's.
    /// </summary>
    private static readonly (string Joint, string Body, int Observation, int Action)[] FigureJoints = {
        ("bthigh", "bthigh", 2, 0), ("bshin", "bshin", 3, 1), ("bfoot", "bfoot", 4, 2),
        ("fthigh", "fthigh", 5, 3), ("fshin", "fshin", 6, 4), ("ffoot", "ffoot", 7, 5),
    };

    /// <summary>
    ///     Records the docs page's media (DR-112) into <paramref name="folder"/>: the expert's seed-0 episode as
    ///     <c>HalfCheetah-v5.episode.mp4</c> (one 480×480 frame per step at the env's 20 frames a second, through the
    ///     model's <c>track</c> camera, by Gymnasium's <see cref="RecordVideo{TObs, TAct}"/> and
    ///     <see cref="FfmpegVideoWriter"/>), every step as <c>HalfCheetah-v5.episode.json</c>, and How it works' frame as
    ///     <c>HalfCheetah-v5.figure.png</c> with <c>HalfCheetah-v5.figure.json</c>.
    /// </summary>
    /// <param name="expertPath">The expert's file.</param>
    /// <param name="folder">The output folder, <c>docs/environments/mujoco/media</c> for the site; created if missing.</param>
    /// <returns>0 when the media were written, 2 when no folder was given.</returns>
    /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
    /// <exception cref="IOException">A file cannot be written, or the expert's file cannot be read.</exception>
    /// <exception cref="InvalidDataException">The expert's file is not a <c>gym.net-agent/1</c> agent.</exception>
    /// <remarks>
    ///     <para>
    ///     Each JSON row is the state the expert acted on, its six commands and the step's reward with its two terms:
    ///     <c>step, x, z, pitch, vx, bthigh, bshin, bfoot, fthigh, fshin, ffoot, a0 ... a5, reward_forward, reward_ctrl,
    ///     reward</c>. The state is the simulation's own: the torso's x, height and pitch (<c>qpos[0..2]</c>), its forward
    ///     velocity (<c>qvel[0]</c>) and the six joint angles (<c>qpos[3..8]</c>), rounded to 5 decimals. The commands are
    ///     the network's float32 values, each written as the float64 that holds it exactly, so the trace replays the
    ///     episode bit for bit (<c>action_dtype</c> says to cast them back); the reward terms are the step's info. The
    ///     video's frame <c>t</c> shows the state of row <c>t</c>, and its last frame the state after the last step, which
    ///     the JSON keeps as <c>final</c>.
    ///     </para>
    ///     <para>
    ///     Gym.NET draws MuJoCo's frames in software (DR-62), so the recording takes a minute or two. RecordVideo names its
    ///     file <c>HalfCheetah-v5-episode-0.mp4</c> and writes it when the env closes; it records into a scratch folder
    ///     inside <paramref name="folder"/> and the file is moved to the page's name, so a recording that fails halfway
    ///     leaves the published video as it was. <c>[NDScoped]</c> (DR-80) releases every array the episode creates.
    ///     </para>
    /// </remarks>
    [NDScoped]
    private static int Record(string expertPath, string folder) {
        // The folder is required: the page's media live in the docs tree, and a default would write them where no page
        // reads them.
        if (string.IsNullOrEmpty(folder)) {
            Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/mujoco/media");
            return 2;
        }
        ExpertPolicy expert = ExpertPolicy.Load(expertPath);
        Directory.CreateDirectory(folder);
        string scratch = Path.Combine(folder, ".recording");
        var rows = new List<double[]>();
        double total = 0;
        double[] final;
        Dictionary<string, object?> arguments = expert.MakeArguments(human: false);
        arguments["render_mode"] = "rgb_array";
        // The model's own camera, which follows the body's centre of mass: with no camera named, Gymnasium 1.4 forces
        // MuJoCo's free camera, which never follows a body, and the runner leaves the frame within a second (the
        // showreel's choice, DR-74).
        arguments["camera_name"] = "track";
        using (Env<NDArray, NDArray> made = Gymnasium.Make<NDArray, NDArray>(Id, arguments))
        using (var env = new RecordVideo<NDArray, NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: Id,
                   videoWriter: new FfmpegVideoWriter())) {
            var cheetah = (MujocoEnv)env.Unwrapped;
            (NDArray observation, Info info) = env.Reset(seed: RecordSeed);
            for (int t = 0; ; t++) {
                NDArray action = expert.Act(observation);
                // float32 to float64 is exact, so the JSON's number casts back to the very float32 the expert chose.
                double[] commands = action.ToArray<float>().Select(v => (double)v).ToArray();
                double[] state = RecordState(cheetah);
                StepResult<NDArray> step = env.Step(action);
                total += step.Reward;
                double forward = step.Info.Get<double>("reward_forward"), control = step.Info.Get<double>("reward_ctrl");
                rows.Add(state.Prepend(t).Concat(commands)
                    .Append(Math.Round(forward, 5)).Append(Math.Round(control, 5)).Append(Math.Round(step.Reward, 5)).ToArray());
                observation = step.Observation;
                // HalfCheetah never terminates: the time limit ends the episode at 1,000 steps.
                if (step.Terminated || step.Truncated)
                    break;
            }
            final = RecordState(cheetah);
            // RecordVideo writes the episode's video when it stops recording, which closing the env does.
            env.Close();
        }
        string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
        if (!File.Exists(recorded))
            throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
        File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
        Directory.Delete(scratch, recursive: true);
        File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
            id = Id, seed = RecordSeed, steps = rows.Count, @return = total, fps = 20,
            agent = "Farama's TQC expert (examples/HalfCheetah)", action_dtype = "float32",
            columns = new[] { "step", "x", "z", "pitch", "vx", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot",
                "a0", "a1", "a2", "a3", "a4", "a5", "reward_forward", "reward_ctrl", "reward" },
            rows,
            final,
        }) + "\n");
        SaveFigure(folder);
        Console.WriteLine($"{Id}, seed {RecordSeed}: {rows.Count} steps, return {total:F1}, recorded into {folder}");
        return 0;
    }

    /// <summary>The trace's view of a state: the torso's x, height and pitch, its forward velocity, the six joint angles.</summary>
    /// <param name="cheetah">The unwrapped env, whose simulation holds the state.</param>
    /// <returns>Ten values, rounded to 5 decimals: <c>qpos[0..2]</c>, <c>qvel[0]</c>, <c>qpos[3..8]</c>.</returns>
    private static double[] RecordState(MujocoEnv cheetah) {
        // The simulation's own arrays through views: copied out at once, the views released with the episode's scope.
        double[] qpos = cheetah.Data.Qpos.ToArray<double>(), qvel = cheetah.Data.Qvel.ToArray<double>();
        double R(double v) => Math.Round(v, 5);
        return new[] { R(qpos[0]), R(qpos[1]), R(qpos[2]), R(qvel[0]), R(qpos[3]), R(qpos[4]), R(qpos[5]), R(qpos[6]), R(qpos[7]), R(qpos[8]) };
    }

    /// <summary>
    ///     Saves How it works' frame as <c>HalfCheetah-v5.figure.png</c>, and as <c>HalfCheetah-v5.figure.json</c> the camera
    ///     and the pixel of each point its labels mark: the torso and each joint, projected through MuJoCo's free-camera
    ///     geometry.
    /// </summary>
    /// <param name="folder">The output folder.</param>
    /// <exception cref="IOException">A file cannot be written.</exception>
    /// <remarks>
    ///     <para>
    ///     The frame is the env's own render of its reset without noise, which every seed then gives alike. The projection
    ///     is <c>mjv_updateCamera</c>'s for a free camera: <c>forward = (cos e cos a, cos e sin a, sin e)</c>,
    ///     <c>up = (-sin e cos a, -sin e sin a, cos e)</c>, <c>right = (sin a, -cos a, 0)</c> for azimuth <c>a</c> and
    ///     elevation <c>e</c>, the eye at <c>lookat - distance forward</c>, and a vertical field of view of 45° (MuJoCo's
    ///     <c>vis.global.fovy</c>, which every model Gymnasium bundles keeps) over the frame's height: a point at camera
    ///     coordinates <c>(x, y, z)</c> is the pixel <c>(W/2 + f x/z, H/2 - f y/z)</c>, <c>f = (H/2) / tan(22.5°)</c>.
    ///     Reference: <see href="https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_vis_visualize.c">engine_vis_visualize.c</see>.
    ///     </para>
    ///     <para>
    ///     The frame is Gym.NET's: its MuJoCo frames are drawn in software and equal Gymnasium's only for the inverted
    ///     pendulums so far (DR-98), so a few thousand of this frame's pixels differ from Gymnasium's own, along the
    ///     floor's checkers and the bodies' edges; the projected points do not depend on the drawing.
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
        using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, arguments);
        (NDArray observation, Info info) = env.Reset(seed: RecordSeed);
        using RenderResult rendered = env.Render();
        NDArray frame = (NDArray)rendered;
        using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), FigureWidth, FigureHeight))
            image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
        var cheetah = (MujocoEnv)env.Unwrapped;
        double a = FigureAzimuth * Math.PI / 180, e = FigureElevation * Math.PI / 180;
        double[] forward = { Math.Cos(e) * Math.Cos(a), Math.Cos(e) * Math.Sin(a), Math.Sin(e) };
        double[] up = { -Math.Sin(e) * Math.Cos(a), -Math.Sin(e) * Math.Sin(a), Math.Cos(e) };
        double[] right = { Math.Sin(a), -Math.Cos(a), 0 };
        double[] eye = Enumerable.Range(0, 3).Select(k => FigureLookat[k] - FigureDistance * forward[k]).ToArray();
        double focal = FigureHeight / 2.0 / Math.Tan(45.0 / 2 * Math.PI / 180);
        double[] Pixel(string body) {
            double[] p = cheetah.Data.Body(body).Xpos.ToArray<double>();
            double[] v = { p[0] - eye[0], p[1] - eye[1], p[2] - eye[2] };
            double z = v[0] * forward[0] + v[1] * forward[1] + v[2] * forward[2];
            double x = v[0] * right[0] + v[1] * right[1] + v[2] * right[2];
            double y = v[0] * up[0] + v[1] * up[1] + v[2] * up[2];
            return new[] { Math.Round(FigureWidth / 2.0 + focal * x / z, 2), Math.Round(FigureHeight / 2.0 - focal * y / z, 2) };
        }
        double[] o = observation.ToArray<double>();
        File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
            id = Id, width = FigureWidth, height = FigureHeight,
            camera = new { distance = FigureDistance, azimuth = FigureAzimuth, elevation = FigureElevation, lookat = FigureLookat, fovy = 45.0 },
            observation = o.Select(v => Math.Round(v, 5)).ToArray(),
            torso = Pixel("torso"),
            joints = FigureJoints.Select(j => new { name = j.Joint, body = j.Body, observation = j.Observation, action = j.Action, pixel = Pixel(j.Body) }).ToArray(),
        }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    /// <summary>
    ///     The same mode on the autopilot (<c>--autopilot</c>): seeded episodes of HalfCheetah-v5 on the autopilot through
    ///     <c>Gymnasium.Make</c>, the product's env rather than this program's game, headless or in MuJoCo's window, and prints
    ///     each one's steps and return.
    /// </summary>
    /// <param name="policyPath">The autopilot's file.</param>
    /// <param name="seed">The first reset seed.</param>
    /// <param name="episodes">How many episodes, on consecutive seeds.</param>
    /// <param name="render">Whether to show them in MuJoCo's window, human mode with Gymnasium's controls (DR-70).</param>
    /// <returns>0.</returns>
    /// <exception cref="IOException">The policy file cannot be read.</exception>
    /// <remarks>
    ///     Seed 0 returns 5,797.0, the bits <c>score</c> prints for the game's own copy of the engine, and the bits
    ///     replay_in_gymnasium.py gets from Gymnasium on MuJoCo 4.0.0. Each episode is one <see cref="PlayEpisode"/> call,
    ///     whose scope releases its arrays when it returns.
    /// </remarks>
    private static int Episodes(string policyPath, int seed, int episodes, bool render) {
        LinearPolicy policy = LinearPolicy.Load(policyPath);
        // Human mode draws inside every reset and step into the app-wide viewer (DR-20).
        if (render)
            Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
        double sum = 0;
        float[] action = new float[CheetahGame.ActionSize];
        for (int episode = 0; episode < episodes; episode++) {
            (int steps, double total) = PlayEpisode(policy, action, seed + episode, render);
            sum += total;
            Console.WriteLine($"HalfCheetah-v5, seed {seed + episode}: {steps} steps, return {total:F1}, truncated by the time limit");
        }
        if (episodes > 1)
            Console.WriteLine($"Mean return over {episodes} episodes: {sum / episodes:F1}");
        return 0;
    }

    /// <summary>One episode of HalfCheetah-v5 flown by the autopilot, through <c>Gymnasium.Make</c>.</summary>
    /// <param name="policy">The autopilot.</param>
    /// <param name="action">The buffer the autopilot writes its action into; reused across steps.</param>
    /// <param name="seed">The reset seed.</param>
    /// <param name="render">Whether to show the episode in MuJoCo's window.</param>
    /// <returns>The steps taken and the return.</returns>
    /// <remarks>
    ///     The arrays a reset or a step returns are the caller's (D19). The method is <c>[NDScoped]</c> (DR-80):
    ///     NumSharp.Build releases every array the episode creates (the results, the actions) when it returns, one episode's
    ///     worth at a time. The env is built and disposed inside it, the guides' rule for an env in a scoped method.
    /// </remarks>
    [NDScoped]
    private static (int Steps, double Return) PlayEpisode(LinearPolicy policy, float[] action, int seed, bool render) {
        using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>("HalfCheetah-v5", render ? new { render_mode = "human" } : null);
        (NDArray observation, Info info) = env.Reset(seed: seed);
        int steps = 0;
        double total = 0;
        while (true) {
            // HalfCheetah's motors take [-1, 1], the autopilot's own range: its outputs are the action as they are.
            policy.Act(observation.ToArray<double>(), action);
            StepResult<NDArray> step = env.Step(np.array(action));
            observation = step.Observation;
            steps++;
            total += step.Reward;
            // HalfCheetah never terminates: every episode is cut by the 1,000-step time limit.
            if (step.Terminated || step.Truncated)
                return (steps, total);
        }
    }

    /// <summary>The positional argument after a command (a file or folder).</summary>
    /// <param name="args">The arguments.</param>
    /// <param name="command">The command.</param>
    /// <returns>The argument.</returns>
    /// <exception cref="ArgumentException">It is missing.</exception>
    private static string Argument(string[] args, string command) =>
        args.Length > 1 && !args[1].StartsWith("--", StringComparison.Ordinal) ? args[1] : throw new ArgumentException($"{command} needs a path: {command} PATH");

    // ================================================================================== play

    /// <summary>
    ///     The window: the simulation runs in real time (or faster or slower with +/-); each env step reads the autopilot or
    ///     the keys, holds that action for five physics steps, and the view redraws up to 60 times a second.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot, or null when there is none (the keyboard drives).</param>
    /// <param name="seed">The first episode's Gymnasium seed; each new episode takes the next.</param>
    /// <param name="manual">Start with the keyboard driving.</param>
    /// <returns>0.</returns>
    /// <exception cref="InvalidOperationException">The viewer reports no keyboard input.</exception>
    private static int Play(MjModel model, LinearPolicy? policy, int seed, bool manual) {
        var view = new SideView();
        using var game = new CheetahGame(model);
        game.Reset(seed);
        // The factory returns once the window is up (DR-85).
        IEnvViewer viewer = AvaloniaEnvViewer.Factory(view.Width, view.Height, Title);
        var input = viewer as IInteractiveEnvViewer ?? throw new InvalidOperationException("the viewer reports no keys");
        input.BeginInput();
        Console.WriteLine("Window open. Tab: autopilot on/off; keys Q/A W/S E/D drive the back leg, U/J I/K O/L the front; " +
                          "R: new episode; Space: pause; +/-: speed; Esc: quit.");
        var held = new HashSet<int>();
        bool autopilot = policy != null && !manual, paused = false, quit = false;
        double speed = 1, lastScore = double.NaN, bannerUntil = 0, lastFrame = -1;
        string? banner = null;
        int nextSeed = seed + 1;
        var clock = Stopwatch.StartNew();
        double due = 0;
        Span<float> action = stackalloc float[CheetahGame.ActionSize];

        void Show() {
            double now = clock.Elapsed.TotalSeconds;
            Hud hud = new(autopilot ? "AUTOPILOT" : "YOU", paused, speed, lastScore, now < bannerUntil ? banner : null);
            Surface frame = view.Render(game, hud);
            // The viewer contract's (H, W, 3) uint8 frame, copied by the viewer before Render returns.
            using NDArray pixels = SurfArray.ToRgbArray(frame);
            viewer.Render(pixels);
            lastFrame = now;
        }

        while (!quit) {
            foreach (ViewerInput e in input.TakeInput()) {
                switch (e.Kind) {
                    case ViewerInputKind.Close:
                        quit = true;
                        break;
                    case ViewerInputKind.KeyUp:
                        held.Remove(e.Key);
                        break;
                    case ViewerInputKind.KeyDown:
                        held.Add(e.Key);
                        switch (e.Key) {
                            case KeyCode.Escape:
                                quit = true;
                                break;
                            case KeyCode.Tab when policy != null:
                                autopilot = !autopilot;
                                break;
                            case KeyCode.Space:
                                paused = !paused;
                                break;
                            case KeyCode.R:
                                game.Reset(nextSeed++);
                                break;
                            case KeyCode.EqualsSign or KeyCode.Plus or KeyCode.KeypadPlus:
                                speed = Math.Min(4, speed * 2);
                                break;
                            case KeyCode.Minus or KeyCode.KeypadMinus:
                                speed = Math.Max(0.125, speed / 2);
                                break;
                        }
                        break;
                }
            }
            if (quit)
                break;
            if (paused) {
                Show();
                Thread.Sleep(30);
                due = clock.Elapsed.TotalSeconds;
                continue;
            }
            if (autopilot && policy != null)
                policy.Act(game.Observation(), action);
            else
                KeysToAction(held, action);
            game.BeginStep(action);
            for (int k = 0; k < CheetahGame.FrameSkip; k++) {
                game.PhysicsStep();
                due += model.timestep / speed;
                double now = clock.Elapsed.TotalSeconds;
                if (now - due > 0.25) {
                    // far behind (the window was dragged, the machine is busy): skip the debt instead of racing to catch up
                    due = now;
                }
                if (now - lastFrame >= 1.0 / 60)
                    Show();
                while (clock.Elapsed.TotalSeconds < due)
                    Thread.Sleep(1);
            }
            game.EndStep();
            if (game.StepIndex >= CheetahGame.EpisodeSteps) {
                lastScore = game.EpisodeReturn;
                banner = $"EPISODE SCORE {lastScore.ToString("F0", CultureInfo.InvariantCulture)}";
                bannerUntil = clock.Elapsed.TotalSeconds + 3;
                game.Reset(nextSeed++);
            }
        }
        viewer.Close();
        return 0;
    }

    /// <summary>
    ///     The keyboard's action: per motor, +1 while its upper key is held, -1 while its lower key is, 0 for neither or both.
    ///     Q/A, W/S, E/D drive the back thigh, shin and foot; U/J, I/K, O/L the front ones.
    /// </summary>
    /// <param name="held">The SDL keycodes held down.</param>
    /// <param name="action">Receives the action.</param>
    private static void KeysToAction(HashSet<int> held, Span<float> action) {
        ReadOnlySpan<int> plus = [KeyCode.Q, KeyCode.W, KeyCode.E, KeyCode.U, KeyCode.I, KeyCode.O];
        ReadOnlySpan<int> minus = [KeyCode.A, KeyCode.S, KeyCode.D, KeyCode.J, KeyCode.K, KeyCode.L];
        for (int i = 0; i < CheetahGame.ActionSize; i++)
            action[i] = (held.Contains(plus[i]) ? 1f : 0f) - (held.Contains(minus[i]) ? 1f : 0f);
    }

    // ================================================================================== train, record, score

    /// <summary>Trains the autopilot with ARS and writes it as JSON.</summary>
    /// <param name="model">The model.</param>
    /// <param name="options">--iterations, --threads, --seed, --out.</param>
    /// <returns>0.</returns>
    private static int Train(MjModel model, Options options) {
        var settings = new Ars.Settings(Iterations: options.Int("iterations", 600), Threads: options.Int("threads", Math.Max(1, Environment.ProcessorCount / 2)),
                                        Seed: options.Int("seed", 1));
        string output = options.Text("out", "policy.json");
        Console.WriteLine($"Training with {settings}");
        LinearPolicy best = Ars.Train(model, settings, Console.Out);
        best.Save(output);
        Console.WriteLine($"best evaluation {best.Info["evaluation_mean"]} at iteration {best.Info["iteration"]}; written to {Path.GetFullPath(output)}");
        return 0;
    }

    /// <summary>
    ///     Records the autopilot to an H.264 video through ffmpeg: 50 frames a second (one every two physics steps), real
    ///     time, episodes chained with their score shown for two seconds.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot.</param>
    /// <param name="path">The video file, overwritten.</param>
    /// <param name="seconds">How many simulated seconds to record.</param>
    /// <param name="seed">The first episode's seed.</param>
    /// <returns>0, or ffmpeg's exit code when it failed.</returns>
    /// <exception cref="System.ComponentModel.Win32Exception">ffmpeg is not on PATH.</exception>
    private static int Video(MjModel model, LinearPolicy policy, string path, double seconds, int seed) {
        var view = new SideView();
        using var game = new CheetahGame(model);
        game.Reset(seed);
        const int fps = 50;
        int physicsPerFrame = (int)Math.Round(1.0 / (fps * model.timestep));
        var start = new ProcessStartInfo("ffmpeg") { RedirectStandardInput = true, UseShellExecute = false };
        foreach (string a in new[] { "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", $"{view.Width}x{view.Height}",
                                     "-r", fps.ToString(CultureInfo.InvariantCulture), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                                     "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path })
            start.ArgumentList.Add(a);
        using Process ffmpeg = Process.Start(start)!;
        Stream sink = ffmpeg.StandardInput.BaseStream;
        Span<float> action = stackalloc float[CheetahGame.ActionSize];
        double lastScore = double.NaN, bannerUntil = -1;
        string? banner = null;
        long physics = 0, total = (long)Math.Round(seconds / model.timestep);
        int episodeSeed = seed;
        while (physics < total) {
            policy.Act(game.Observation(), action);
            game.BeginStep(action);
            for (int k = 0; k < CheetahGame.FrameSkip; k++) {
                game.PhysicsStep();
                if (++physics % physicsPerFrame == 0) {
                    double t = physics * model.timestep;
                    Surface frame = view.Render(game, new Hud("AUTOPILOT", false, 1, lastScore, t < bannerUntil ? banner : null));
                    sink.Write(frame.ToRgbBytes());
                }
            }
            game.EndStep();
            if (game.StepIndex >= CheetahGame.EpisodeSteps) {
                lastScore = game.EpisodeReturn;
                banner = $"EPISODE SCORE {lastScore.ToString("F0", CultureInfo.InvariantCulture)}";
                bannerUntil = physics * model.timestep + 2;
                game.Reset(++episodeSeed);
            }
        }
        sink.Close();
        ffmpeg.WaitForExit();
        Console.WriteLine(ffmpeg.ExitCode == 0 ? $"{seconds} s of autopilot written to {Path.GetFullPath(path)}" : $"ffmpeg failed ({ffmpeg.ExitCode})");
        return ffmpeg.ExitCode;
    }

    /// <summary>Writes PNG frames of the autopilot's first env steps, every few steps, for a quick look without a window.</summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot.</param>
    /// <param name="folder">The output folder (created).</param>
    /// <param name="steps">Env steps to play.</param>
    /// <param name="every">Env steps between frames.</param>
    /// <param name="seed">The episode's seed.</param>
    /// <returns>0.</returns>
    private static int Frames(MjModel model, LinearPolicy policy, string folder, int steps, int every, int seed) {
        Directory.CreateDirectory(folder);
        var view = new SideView();
        using var game = new CheetahGame(model);
        game.Reset(seed);
        Span<float> action = stackalloc float[CheetahGame.ActionSize];
        for (int t = 0; t <= steps; t++) {
            if (t % every == 0) {
                Surface frame = view.Render(game, new Hud("AUTOPILOT", false, 1, double.NaN, null));
                using Image<Rgb24> image = Image.LoadPixelData<Rgb24>(frame.ToRgbBytes(), frame.Width, frame.Height);
                image.SaveAsPng(Path.Combine(folder, $"step{t:D4}.png"));
            }
            policy.Act(game.Observation(), action);
            game.Step(action);
        }
        Console.WriteLine($"frames written to {Path.GetFullPath(folder)}");
        return 0;
    }

    /// <summary>
    ///     Plays full episodes from Gymnasium's seeded resets and prints each return with its exact bits, to compare with
    ///     replay_in_gymnasium.py's run of the same policy in Gymnasium on MuJoCo.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="policy">The autopilot.</param>
    /// <param name="seeds">A range "A-B" or a single seed.</param>
    /// <returns>0.</returns>
    /// <exception cref="ArgumentException">The range is malformed.</exception>
    private static int Score(MjModel model, LinearPolicy policy, string seeds) {
        string[] parts = seeds.Split('-');
        if (!int.TryParse(parts[0], out int first) || !int.TryParse(parts[^1], out int last) || last < first)
            throw new ArgumentException($"--seeds wants A-B or A, got '{seeds}'");
        double[] results = new double[last - first + 1];
        Parallel.For(first, last + 1, s => results[s - first] = Ars.Play(model, policy, s));
        for (int s = first; s <= last; s++) {
            double r = results[s - first];
            Console.WriteLine($"seed {s}: return {r:R} (0x{BitConverter.DoubleToInt64Bits(r):X16})");
        }
        return 0;
    }

    /// <summary>The <c>--name value</c> options after the command.</summary>
    private sealed class Options {
        private readonly Dictionary<string, string?> values = new();

        /// <summary>Parses <c>--name value</c> pairs and bare <c>--flag</c>s.</summary>
        /// <param name="args">The option arguments.</param>
        public Options(string[] args) {
            for (int i = 0; i < args.Length; i++) {
                if (!args[i].StartsWith("--", StringComparison.Ordinal))
                    continue;
                bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal);
                values[args[i][2..]] = hasValue ? args[++i] : null;
            }
        }

        /// <summary>Whether a flag was given.</summary>
        /// <param name="name">The flag without dashes.</param>
        /// <returns>Whether it is present.</returns>
        public bool Flag(string name) => values.ContainsKey(name);

        /// <summary>A text option.</summary>
        /// <param name="name">The option without dashes.</param>
        /// <param name="fallback">The value when it is absent.</param>
        /// <returns>The value.</returns>
        public string Text(string name, string fallback) => values.TryGetValue(name, out string? v) && v != null ? v : fallback;

        /// <summary>An integer option.</summary>
        /// <param name="name">The option without dashes.</param>
        /// <param name="fallback">The value when it is absent.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentException">The value is not an integer.</exception>
        public int Int(string name, int fallback) =>
            values.TryGetValue(name, out string? v) && v != null
                ? int.TryParse(v, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) ? n : throw new ArgumentException($"--{name} wants an integer, got '{v}'")
                : fallback;

        /// <summary>A number option.</summary>
        /// <param name="name">The option without dashes.</param>
        /// <param name="fallback">The value when it is absent.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentException">The value is not a number.</exception>
        public double Double(string name, double fallback) =>
            values.TryGetValue(name, out string? v) && v != null
                ? double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out double x) ? x : throw new ArgumentException($"--{name} wants a number, got '{v}'")
                : fallback;
    }
}
