#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using Gym.Box2D;
using Gym.Environments.Envs.Box2D;
using Gym.Envs;
using Gym.Observations;
using Gym.Rendering.Avalonia;
using Gym.Utils;
using Gym.Wrappers;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Examples.CarRacing {
    /// <summary>
    ///     Drives CarRacing-v3 by pure pursuit on the track's centre line. <c>dotnet run --project examples/CarRacing
    ///     -c Release</c> plays one seeded episode headless and prints its steps and return; <c>-- --render</c> plays
    ///     it in a window, <c>-- --play</c> hands the car to the W, A, S and D keys, and <c>-- --record &lt;folder&gt;</c>
    ///     records the episode for Car Racing's page on the docs site (DR-112).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The agent reads the track and the car from the unwrapped env rather than from the 96x96 pixels: it aims at
    ///     the centre-line point five track tiles past the nearest one, steers by the angle to it, keeps the throttle
    ///     while the speed is under a limit that drops with that angle, and brakes into sharp bends. Seed 11 drives the
    ///     full 1,000 steps.
    ///     </para>
    ///     <para>
    ///     The default seed, 11, is the episode the front-page showreel plays (<c>tools/showreel</c>, DR-74).
    ///     </para>
    /// </remarks>
    internal static class CarRacingProgram {
        /// <summary>The environment this example plays, by its registered id.</summary>
        private const string Id = "CarRacing-v3";

        /// <summary>The first reset seed when <c>--seed</c> is not given: the episode the showreel plays.</summary>
        private const long DefaultSeed = 11;

        /// <summary>
        ///     The moment How it works annotates on Car Racing's page (DR-114): the frame after this many steps of the
        ///     seed-11 episode, the car turning into a left bend along its red and white kerb, every indicator lit.
        /// </summary>
        private const int FigureStep = 640;

        /// <summary>
        ///     Plays the episodes the arguments ask for with the agent, or hands the env to the keyboard, or records the
        ///     docs page's media.
        /// </summary>
        /// <param name="args">
        ///     <c>--seed N</c> starts from reset seed N; <c>--episodes N</c> plays N episodes on consecutive seeds and
        ///     prints their mean; <c>--render</c> shows them in a window; <c>--play</c> lets you play the env from the
        ///     keyboard instead, until Escape or the window's close button; <c>--record &lt;folder&gt;</c> records the
        ///     page's episode, seed 11 whatever <c>--seed</c> says (<see cref="Record"/>). Other arguments are ignored.
        /// </param>
        /// <returns>
        ///     0 when the episodes were played, the keyboard game ended or the recording was written, 2 when an option's
        ///     value is not a whole number or <c>--record</c> names no folder.
        /// </returns>
        /// <exception cref="Gym.Exceptions.DependencyNotInstalled">
        ///     Not thrown in practice: <c>--render</c> and <c>--play</c> set Avalonia's viewer before the env first
        ///     draws.
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     <c>--record</c> could not run ffmpeg, which <see cref="FfmpegVideoWriter"/> needs on PATH.
        /// </exception>
        /// <exception cref="System.IO.IOException">
        ///     <c>--record</c> could not create the folder or write one of its files.
        /// </exception>
        private static int Main(string[] args) {
            if (!TryNumber(args, "--seed", DefaultSeed, out long seed) || !TryNumber(args, "--episodes", 1, out long episodes))
                return 2;
            int record = Array.IndexOf(args, "--record");
            if (record >= 0) {
                // The folder is required: the page's media live in the docs tree, and a default would write them where no
                // page reads them.
                if (record + 1 >= args.Length || args[record + 1].StartsWith("--", StringComparison.Ordinal)) {
                    Console.Error.WriteLine("--record needs a folder, e.g. --record docs/environments/box2d/media");
                    return 2;
                }
                (int recordedSteps, double recordedReturn) = Record(args[record + 1]);
                Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                    $"{Id}, seed {DefaultSeed}: {recordedSteps} steps, return {recordedReturn:F1}, recorded into {args[record + 1]}"));
                return 0;
            }
            bool render = args.Contains("--render");
            if (args.Contains("--play")) {
                // play draws the env's frames into a window of the app-wide viewer and reads the keys from it.
                Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory;
                PlayWithKeyboard(seed);
                return 0;
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
            // make's stack: TimeLimit over OrderEnforcing over PassiveEnvChecker over the env. The render mode is a
            // constructor argument, fixed for the env's life, as in Gymnasium.
            using Env<NDArray, NDArray> env = Gymnasium.Make<NDArray, NDArray>(Id, render ? new { render_mode = "human" } : null);
            (NDArray observation, Info info) = env.Reset(seed: seed);
            int steps = 0;
            double total = 0;
            while (true) {
                StepResult<NDArray> step = env.Step(Act(env));
                observation = step.Observation;
                steps++;
                total += step.Reward;
                if (step.Terminated || step.Truncated)
                    return (steps, total, step.Terminated ? total > 0 ? "finished the lap" : "left the playfield" : "truncated by the time limit");
            }
        }

        /// <summary>
        ///     Records the seed-11 episode for Car Racing's page on the docs site (DR-112): the whole episode as
        ///     <c>CarRacing-v3.episode.mp4</c>, one 600×400 frame per step at the env's 50 frames a second, through
        ///     Gymnasium's <see cref="RecordVideo{TObs, TAct}"/> and <see cref="FfmpegVideoWriter"/>; every step as
        ///     <c>CarRacing-v3.episode.json</c>; How it works' frame as <c>CarRacing-v3.figure.png</c> with
        ///     <c>CarRacing-v3.figure.json</c>; and the agent's own view of that moment, the 96×96 observation, as
        ///     <c>CarRacing-v3.observation.png</c>.
        /// </summary>
        /// <param name="folder">The output folder, <c>docs/environments/box2d/media</c> for the site; created if missing.</param>
        /// <returns>The steps taken and the return.</returns>
        /// <exception cref="InvalidOperationException">ffmpeg could not be run, or wrote no video.</exception>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     Each JSON row is the state the agent acted on, its three commands and the reward: <c>step, x, y, angle,
        ///     speed, tiles, steer, gas, brake, reward</c>: the hull's world position and angle, its speed (the length of
        ///     its velocity, the white indicator's value) and the tiles visited so far (of <c>tiles_total</c>). The
        ///     commands are the agent's float32 values, each written as the float64 that holds it exactly, so the trace
        ///     replays the episode bit for bit (<c>action_dtype</c> says to cast them back). The video's frame <c>t</c>
        ///     shows the state of row <c>t</c>, and its last frame the state after the last step, which the JSON keeps as
        ///     <c>final</c>.
        ///     </para>
        ///     <para>
        ///     RecordVideo names its file <c>CarRacing-v3-episode-0.mp4</c> and writes it when the env closes; it records
        ///     into a scratch folder inside <paramref name="folder"/> and the file is moved to the page's name, so a
        ///     recording that fails halfway leaves the published video as it was. Each step renders twice, the 96×96
        ///     observation inside the step and the 600×400 frame for the video, as Gymnasium's RecordVideo does: the
        ///     recording runs at about a fifth of the headless episode's speed.
        ///     </para>
        /// </remarks>
        [NDScoped]   // releases every array the episode creates when it returns (DR-80); the envs are disposed inside
        private static (int Steps, double Return) Record(string folder) {
            Directory.CreateDirectory(folder);
            string scratch = Path.Combine(folder, ".recording");
            var rows = new List<double[]>();
            double total = 0;
            double[] final;
            int tilesTotal;
            using (Env<NDArray, NDArray> made = Gymnasium.Make<NDArray, NDArray>(Id, RenderMode.RgbArray))
            using (var env = new RecordVideo<NDArray, NDArray>(made, scratch, episodeTrigger: _ => true, namePrefix: Id,
                       videoWriter: new FfmpegVideoWriter())) {
                var race = (CarRacingEnv)env.Unwrapped;
                (NDArray observation, Info info) = env.Reset(seed: DefaultSeed);
                tilesTotal = race.Track.Count;
                for (int t = 0; ; t++) {
                    NDArray action = Act(env);
                    // float32 to float64 is exact, so the JSON's number casts back to the very float32 the agent chose.
                    double[] commands = action.ToArray<float>().Select(v => (double)v).ToArray();
                    // The frame now on screen shows the state the agent is about to act on (row t): the recorder's last,
                    // and the observation the agent sees of the same moment.
                    if (t == FigureStep)
                        SaveFigure(env, folder, race, observation);
                    double[] state = State(race);
                    StepResult<NDArray> step = env.Step(action);
                    total += step.Reward;
                    rows.Add(state.Prepend(t).Concat(commands).Append(Math.Round(step.Reward, 5)).ToArray());
                    observation = step.Observation;
                    if (step.Terminated || step.Truncated)
                        break;
                }
                final = State(race);
                // RecordVideo writes the episode's video when it stops recording, which closing the env does.
                env.Close();
            }
            string recorded = Path.Combine(scratch, $"{Id}-episode-0.mp4");
            if (!File.Exists(recorded))
                throw new InvalidOperationException($"RecordVideo wrote no {recorded}; is ffmpeg on PATH?");
            File.Move(recorded, Path.Combine(folder, $"{Id}.episode.mp4"), overwrite: true);
            Directory.Delete(scratch, recursive: true);
            File.WriteAllText(Path.Combine(folder, $"{Id}.episode.json"), JsonSerializer.Serialize(new {
                id = Id, seed = DefaultSeed, steps = rows.Count, @return = total, fps = CarRacingEnv.Fps,
                agent = "pure pursuit on the track's centre line (examples/CarRacing)", action_dtype = "float32",
                tiles_total = tilesTotal,
                columns = new[] { "step", "x", "y", "angle", "speed", "tiles", "steer", "gas", "brake", "reward" },
                rows,
                final,
            }) + "\n");
            return (rows.Count, total);
        }

        /// <summary>
        ///     The trace's view of a state: the hull's world position and angle, its speed, and the tiles visited so far.
        /// </summary>
        /// <param name="race">The unwrapped env.</param>
        /// <returns>Five values: x, y, angle (radians), speed (world units a second), tiles.</returns>
        private static double[] State(CarRacingEnv race) {
            b2Body hull = race.Car!.Hull!;
            b2Vec2 v = hull.LinearVelocity;
            double speed = Math.Sqrt((double)v.x * v.x + (double)v.y * v.y);
            return new[] { Math.Round(hull.Position.x, 3), Math.Round(hull.Position.y, 3), Math.Round(hull.Angle, 4), Math.Round(speed, 3), race.TileVisitedCount };
        }

        /// <summary>
        ///     Saves How it works' frame (the recorder's last, the state now) as <c>CarRacing-v3.figure.png</c>, the agent's
        ///     96×96 view of the same moment as <c>CarRacing-v3.observation.png</c>, and what the figure's labels point at
        ///     as <c>CarRacing-v3.figure.json</c>, in the frame's pixels: the car, the score and each indicator's bar.
        /// </summary>
        /// <param name="env">The recording env, whose last recorded frame is the figure.</param>
        /// <param name="folder">The output folder.</param>
        /// <param name="race">The unwrapped env, for the car and the score.</param>
        /// <param name="observation">The observation the agent acts on at this step, the 96×96 frame of the same moment.</param>
        /// <exception cref="IOException">A file cannot be written.</exception>
        /// <remarks>
        ///     <para>
        ///     The env draws a 1000×800 frame and shrinks it to 600×400 for <c>rgb_array</c> (and to 96×96 for the
        ///     observation), so a point of the big frame is the pixel <c>(0.6 x, 0.5 y)</c> here. The camera keeps the car
        ///     at the big frame's (500, 600), turned so the car points up: (300, 300) here, after the first second's
        ///     zoom.
        ///     </para>
        ///     <para>
        ///     The indicators are Gymnasium's <c>_render_indicators</c>: on the black strip at the bottom (its last 100 of
        ///     800 rows), with <c>s = 25</c> and <c>h = 20</c> big-frame pixels, a vertical bar <c>(place, value)</c> spans
        ///     x from <c>place s</c> to <c>(place + 1) s</c> and y from <c>780 - 20 value</c> to 780, a horizontal one x
        ///     from <c>place s</c> to <c>(place + value) s</c> and y from 720 to 760: the speed at place 5 (value
        ///     <c>0.02 speed</c>), the four wheels' ABS at 7 to 10 (<c>0.01 omega</c>), the steering at 20
        ///     (<c>-10 angle</c> of the front-left wheel's joint) and the gyroscope at 30 (<c>-0.8</c> times the hull's
        ///     angular velocity); a value within 1e-4 of zero draws nothing. The score is centred on (60, 750). Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/car_racing.py#L697-L763">gymnasium/envs/box2d/car_racing.py</see>.
        ///     </para>
        /// </remarks>
        private static void SaveFigure(RecordVideo<NDArray, NDArray> env, string folder, CarRacingEnv race, NDArray observation) {
            NDArray frame = env.RecordedFrames[^1];
            int height = (int)frame.shape[0], width = (int)frame.shape[1];
            using (var image = Image.LoadPixelData<Rgb24>(frame.ToArray<byte>(), width, height))
                image.SaveAsPng(Path.Combine(folder, $"{Id}.figure.png"));
            using (var view = Image.LoadPixelData<Rgb24>(observation.ToArray<byte>(), CarRacingEnv.StateW, CarRacingEnv.StateH))
                view.SaveAsPng(Path.Combine(folder, $"{Id}.observation.png"));
            // From the 1000x800 frame the env draws to this frame's pixels.
            double sx = (double)width / CarRacingEnv.WindowW, sy = (double)height / CarRacingEnv.WindowH;
            const double s = CarRacingEnv.WindowW / 40.0, h = CarRacingEnv.WindowH / 40.0;
            double[] Box(double x0, double y0, double x1, double y1) =>
                new[] { Math.Round(Math.Min(x0, x1) * sx, 2), Math.Round(Math.Min(y0, y1) * sy, 2), Math.Round(Math.Max(x0, x1) * sx, 2), Math.Round(Math.Max(y0, y1) * sy, 2) };
            double[]? Vertical(int place, double value) =>
                Math.Abs(value) > 1e-4 ? Box(place * s, CarRacingEnv.WindowH - (h + h * value), (place + 1) * s, CarRacingEnv.WindowH - h) : null;
            double[]? Horizontal(int place, double value) =>
                Math.Abs(value) > 1e-4 ? Box(place * s, CarRacingEnv.WindowH - 4 * h, (place + value) * s, CarRacingEnv.WindowH - 2 * h) : null;
            Car car = race.Car!;
            b2Vec2 v = car.Hull!.LinearVelocity;
            double speed = Math.Sqrt((double)v.x * v.x + (double)v.y * v.y);
            double steering = car.Wheels[0].Joint.Angle, turning = car.Hull.AngularVelocity;
            File.WriteAllText(Path.Combine(folder, $"{Id}.figure.json"), JsonSerializer.Serialize(new {
                id = Id, step = FigureStep, width, height,
                car = new[] { Math.Round(CarRacingEnv.WindowW / 2.0 * sx, 2), Math.Round((CarRacingEnv.WindowH - CarRacingEnv.WindowH / 4.0) * sy, 2) },
                score = new { center = new[] { Math.Round(60 * sx, 2), Math.Round((CarRacingEnv.WindowH - CarRacingEnv.WindowH * 2.5 / 40.0) * sy, 2) }, value = Math.Round(race.Reward, 3) },
                tiles = race.TileVisitedCount, tiles_total = race.Track.Count,
                speed = Math.Round(speed, 3), omegas = car.Wheels.Select(w => Math.Round(w.Omega, 3)).ToArray(),
                steering = Math.Round(steering, 4), angular_velocity = Math.Round(turning, 4),
                indicators = new {
                    strip = Box(0, CarRacingEnv.WindowH - 5 * h, CarRacingEnv.WindowW, CarRacingEnv.WindowH),
                    speed = Vertical(5, 0.02 * speed),
                    abs = car.Wheels.Select((w, i) => Vertical(7 + i, 0.01 * w.Omega)).ToArray(),
                    steering = Horizontal(20, -10.0 * steering),
                    gyroscope = Horizontal(30, -0.8 * turning),
                },
            }, new JsonSerializerOptions { WriteIndented = true }) + "\n");
        }

        /// <summary>The agent: pure pursuit of a point on the centre line, five tiles ahead.</summary>
        /// <param name="env">The env; its unwrapped <see cref="CarRacingEnv"/> exposes the car and the track.</param>
        /// <returns>A float32 <c>[steer, gas, brake]</c>.</returns>
        private static NDArray Act(IEnv env) {
            var race = (CarRacingEnv)env.Unwrapped;
            b2Body hull = race.Car!.Hull!;
            b2Vec2 p = hull.Position;
            double angle = hull.Angle;
            IReadOnlyList<(double Alpha, double Beta, double X, double Y)> track = race.Track;
            int nearest = 0;
            double best = double.MaxValue;
            for (int i = 0; i < track.Count; i++) {
                double dx0 = track[i].X - p.x, dy0 = track[i].Y - p.y, d = dx0 * dx0 + dy0 * dy0;
                if (d < best) {
                    best = d;
                    nearest = i;
                }
            }
            (double Alpha, double Beta, double X, double Y) target = track[(nearest + 5) % track.Count];
            double dx = target.X - p.x, dy = target.Y - p.y;
            // The hull's forward axis is its local +y: (-sin a, cos a) in world coordinates.
            double fx = -Math.Sin(angle), fy = Math.Cos(angle);
            double error = Math.Atan2(fx * dy - fy * dx, fx * dx + fy * dy);
            b2Vec2 v = hull.LinearVelocity;
            double speed = Math.Sqrt(v.x * v.x + v.y * v.y);
            float steer = (float)Math.Clamp(-error * 2.5, -1, 1);
            float gas = speed < 45 - 30 * Math.Min(1, Math.Abs(error)) ? 0.6f : 0f;
            float brake = Math.Abs(error) > 0.6 && speed > 30 ? 0.4f : 0f;
            return np.array(new[] { steer, gas, brake });
        }

        /// <summary>
        ///     The keys for <c>--play</c>, the map Gymnasium's own <c>play</c> documentation gives for CarRacing: W
        ///     gas, A and D steer, S brake, and the four two-key combinations.
        /// </summary>
        /// <returns>
        ///     The map from keys to float32 <c>[steer, gas, brake]</c> actions; the caller disposes them.
        /// </returns>
        private static Dictionary<object, object?> Keys() => new() {
            ["w"] = np.array(new[] { 0f, 0.7f, 0f }),
            ["a"] = np.array(new[] { -1f, 0f, 0f }),
            ["s"] = np.array(new[] { 0f, 0f, 1f }),
            ["d"] = np.array(new[] { 1f, 0f, 0f }),
            ["wa"] = np.array(new[] { -1f, 0.7f, 0f }),
            ["dw"] = np.array(new[] { 1f, 0.7f, 0f }),
            ["ds"] = np.array(new[] { 1f, 0f, 1f }),
            ["as"] = np.array(new[] { -1f, 0f, 1f }),
        };

        /// <summary>
        ///     Hands the env to the keyboard, Gymnasium's play, until Escape or the window's close button.
        /// </summary>
        /// <param name="seed">
        ///     The seed of every reset: each new game starts as the last one did, as Gymnasium's play does.
        /// </param>
        private static void PlayWithKeyboard(long seed) {
            // play reads the env's rgb_array frames and draws them into its own window.
            using IEnv env = Gymnasium.Make(Id, new { render_mode = "rgb_array" });
            Dictionary<object, object?> keys = Keys();
            NDArray noop = np.array(new float[3]);
            try {
                Play.Run(env, keysToAction: keys, noop: noop, seed: seed);
            } finally {
                // play only borrows a map it is given: its actions, and the no-op, stay ours to release.
                foreach (object? action in keys.Values)
                    (action as IDisposable)?.Dispose();
                noop.Dispose();
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
}
