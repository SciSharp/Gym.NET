using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Gym.Envs;
using Gym.Spaces;
using NumSharp;

namespace Gym.Showreel;

/// <summary>
///     Where the parity segment's Python half runs: the interpreter with the parity suite's reference stack, the Gymnasium
///     checkout to put on <c>PYTHONPATH</c>, the replay script and a folder for the exchanged files.
/// </summary>
/// <param name="Python">The interpreter (<c>python</c> on <c>PATH</c> by default).</param>
/// <param name="Gymnasium">The <c>refs/Gymnasium</c> checkout.</param>
/// <param name="Script">The path of <c>gymnasium_replay.py</c>.</param>
/// <param name="WorkDir">A folder for the actions and replay files (hundreds of MB while the tool runs).</param>
internal sealed record PythonSide(string Python, string Gymnasium, string Script, string WorkDir);

/// <summary>
///     One episode of the parity segment, proven equal: the actions Gym.NET's policy took, replayed in Gymnasium, with
///     every frame, observation, reward and flag compared to Gym.NET's own replay of them. Gives the video Gymnasium's
///     frames and the per-step pixel counts the badges show.
/// </summary>
/// <remarks>
///     <para>
///     <see cref="Create"/> fails rather than return a replay that differs: a non-zero count would never reach the video.
///     The counts it keeps are therefore all zero, and the badges print them rather than a constant so the claim on screen
///     is the measurement.
///     </para>
///     <para>
///     Both sides of the segment show the frames that were compared: Gymnasium's from its replay file, Gym.NET's from a
///     second file written while verifying. A live player would not do: LunarLander ages its exhaust particles inside
///     <c>render()</c> (Gymnasium's own behaviour, which Gym.NET keeps), so a panel rendering only the displayed steps
///     would show other particles than the every-step replay that was proven equal.
///     </para>
///     <para>
///     The files stay open for random access while the video renders, one frame read per displayed step;
///     <see cref="Dispose"/> closes and deletes them.
///     </para>
/// </remarks>
internal sealed class ParityReplay : IDisposable {
    /// <summary>The replay file's magic number.</summary>
    private static readonly byte[] Magic = "GYMR"u8.ToArray();

    /// <summary>The header's size: the magic and five int32 fields.</summary>
    private const int HeaderBytes = 4 + 5 * 4;

    /// <summary>Gymnasium's replay file, open for reading.</summary>
    private readonly FileStream _file;

    /// <summary>Its path, deleted on dispose.</summary>
    private readonly string _path;

    /// <summary>Gym.NET's frames of the same steps (raw, one frame after another), open for reading.</summary>
    private readonly FileStream _gymNetFile;

    /// <summary>Its path, deleted on dispose.</summary>
    private readonly string _gymNetPath;

    /// <summary>Bytes of one record: frame, observation, reward and the two flags.</summary>
    private readonly long _recordBytes;

    /// <summary>The differing pixels of each record's frame (all zero once created).</summary>
    private readonly int[] _different;

    /// <summary>The return accumulated up to each record.</summary>
    private readonly double[] _returns;

    /// <summary>The buffer <see cref="GymnasiumFrame"/> reads into.</summary>
    private readonly RgbImage _frame;

    /// <summary>Wraps a verified replay file and Gym.NET's frames of the same steps.</summary>
    /// <param name="recipe">The episode.</param>
    /// <param name="path">The replay file.</param>
    /// <param name="gymNetPath">Gym.NET's frames, written while verifying.</param>
    /// <param name="steps">Records in it (the reset and each step).</param>
    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    /// <param name="observationBytes">Bytes of one observation.</param>
    /// <param name="different">The per-record pixel counts.</param>
    /// <param name="returns">The per-record returns.</param>
    /// <param name="ended">Whether the last record terminated or truncated the episode.</param>
    /// <param name="finalStamp">The stamp that ending earns.</param>
    private ParityReplay(EpisodeRecipe recipe, string path, string gymNetPath, int steps, int width, int height, int observationBytes, int[] different,
        double[] returns, bool ended, string? finalStamp) {
        Recipe = recipe;
        _path = path;
        _file = File.OpenRead(path);
        _gymNetPath = gymNetPath;
        _gymNetFile = File.OpenRead(gymNetPath);
        Steps = steps;
        _recordBytes = (long)width * height * 3 + observationBytes + 10;
        _different = different;
        _returns = returns;
        _frame = new RgbImage(width, height);
        Ended = ended;
        FinalStamp = finalStamp;
    }

    /// <summary>The episode.</summary>
    public EpisodeRecipe Recipe { get; }

    /// <summary>Records in the replay: the reset frame and one per step.</summary>
    public int Steps { get; }

    /// <summary>Whether the replay's last step ended the episode.</summary>
    public bool Ended { get; }

    /// <summary>The stamp the ending earns, shown on both sides at the last step.</summary>
    public string? FinalStamp { get; }

    /// <summary>The number of pixels whose RGB differs between Gymnasium's and Gym.NET's frame at a step.</summary>
    /// <param name="step">The record, 0 for the reset.</param>
    /// <returns>The count (always 0: <see cref="Create"/> refuses a replay with any other).</returns>
    public int DifferentPixels(int step) => _different[Math.Clamp(step, 0, Steps - 1)];

    /// <summary>The return accumulated up to a step.</summary>
    /// <param name="step">The record.</param>
    /// <returns>The return.</returns>
    public double ReturnAt(int step) => _returns[Math.Clamp(step, 0, Steps - 1)];

    /// <summary>Gymnasium's frame at a step, read from the replay file.</summary>
    /// <param name="step">The record.</param>
    /// <returns>The frame; the buffer is reused by the next call.</returns>
    /// <exception cref="IOException">The file is shorter than its header says.</exception>
    public RgbImage GymnasiumFrame(int step) {
        _file.Position = HeaderBytes + Math.Clamp(step, 0, Steps - 1) * _recordBytes;
        _file.ReadExactly(_frame.Pixels);
        return _frame;
    }

    /// <summary>Gym.NET's frame at a step, the one that was compared with <see cref="GymnasiumFrame"/>.</summary>
    /// <param name="step">The record.</param>
    /// <returns>The frame; the buffer is shared with <see cref="GymnasiumFrame"/> and reused by the next call.</returns>
    /// <exception cref="IOException">The file is shorter than expected.</exception>
    public RgbImage GymNetFrame(int step) {
        _gymNetFile.Position = Math.Clamp(step, 0, Steps - 1) * (long)_frame.Pixels.Length;
        _gymNetFile.ReadExactly(_frame.Pixels);
        return _frame;
    }

    /// <summary>Closes and deletes both frame files.</summary>
    public void Dispose() {
        _file.Dispose();
        _gymNetFile.Dispose();
        File.Delete(_path);
        File.Delete(_gymNetPath);
    }

    /// <summary>
    ///     Plays a recipe with its policy for up to <paramref name="maxSteps"/> steps, replays the actions in Gymnasium,
    ///     replays them again in Gym.NET with rendering, and compares the two record by record.
    /// </summary>
    /// <param name="recipe">The episode.</param>
    /// <param name="maxSteps">The most steps to record (the segment shows about six seconds).</param>
    /// <param name="python">Where Python runs.</param>
    /// <param name="log">Progress lines.</param>
    /// <returns>The verified replay.</returns>
    /// <exception cref="InvalidOperationException">
    ///     Python failed, or Gym.NET and Gymnasium differ in any frame pixel, observation byte, reward bit or flag; the
    ///     message names the first difference.
    /// </exception>
    /// <exception cref="IOException">The exchanged files cannot be written or read.</exception>
    public static ParityReplay Create(EpisodeRecipe recipe, int maxSteps, PythonSide python, TextWriter log) {
        Directory.CreateDirectory(python.WorkDir);
        string stem = Path.Combine(python.WorkDir, recipe.Id);
        string actionsPath = stem + ".actions.json", replayPath = stem + ".gymnasium.bin", gymNetPath = stem + ".gymnet.bin";
        List<object> actions = RecordActions(recipe, maxSteps, out bool discrete);
        WriteActions(actionsPath, recipe, actions, discrete);
        log.WriteLine($"parity: {recipe.Id} seed {recipe.Seed}, {actions.Count} steps; replaying in Gymnasium");
        try {
            RunPython(python, actionsPath, replayPath);
            var (steps, width, height, observationBytes) = ReadHeader(replayPath);
            if (steps != actions.Count + 1)
                throw new InvalidOperationException($"{recipe.Id}: Gymnasium wrote {steps} records for {actions.Count} actions");
            var (different, returns, ended, stamp) = Verify(recipe, actions, replayPath, gymNetPath, width, height, observationBytes);
            log.WriteLine($"parity: {recipe.Id}: {steps} frames, observations, rewards and flags identical to Gymnasium's");
            return new ParityReplay(recipe, replayPath, gymNetPath, steps, width, height, observationBytes, different, returns, ended, stamp);
        } catch {
            // A failed run leaves no frame files behind: they are hundreds of MB and nothing will read them.
            File.Delete(replayPath);
            File.Delete(gymNetPath);
            throw;
        } finally {
            File.Delete(actionsPath);
        }
    }

    /// <summary>Plays the recipe's policy and keeps its actions as plain values (an int64, or a float32 array).</summary>
    /// <param name="recipe">The episode.</param>
    /// <param name="maxSteps">The most steps.</param>
    /// <param name="discrete">Whether the action space is discrete.</param>
    /// <returns>The actions in order.</returns>
    private static List<object> RecordActions(EpisodeRecipe recipe, int maxSteps, out bool discrete) {
        // render_mode as in the video, though nothing is rendered here: the env is configured exactly as when it plays.
        using IEnv env = Gymnasium.Make(recipe.Id, new { render_mode = "rgb_array" });
        discrete = env.ActionSpace is Discrete;
        IPolicy policy = recipe.Policy();
        object? observation = env.Reset(seed: recipe.Seed).Observation;
        var actions = new List<object>();
        for (int i = 0; i < maxSteps; i++) {
            var action = (NDArray)policy.Act(env, observation);
            actions.Add(discrete ? Convert.ToInt64(action.GetValue(), CultureInfo.InvariantCulture) : action.ToArray<float>());
            var (next, _, terminated, truncated, _) = env.Step(action);
            observation = next;
            if (terminated || truncated)
                break;
        }
        return actions;
    }

    /// <summary>
    ///     Writes the actions for the Python side: integers for a discrete space; for a box, each float32's bit pattern as
    ///     an unsigned integer, so no decimal rendering can round a value.
    /// </summary>
    /// <param name="path">The JSON file.</param>
    /// <param name="recipe">The episode (id and seed).</param>
    /// <param name="actions">The recorded actions.</param>
    /// <param name="discrete">Whether they are discrete.</param>
    private static void WriteActions(string path, EpisodeRecipe recipe, List<object> actions, bool discrete) {
        using var stream = File.Create(path);
        using var json = new Utf8JsonWriter(stream);
        json.WriteStartObject();
        json.WriteString("env", recipe.Id);
        json.WriteNumber("seed", recipe.Seed);
        json.WriteString("kind", discrete ? "discrete" : "box");
        json.WriteStartArray(discrete ? "actions" : "bits");
        foreach (object a in actions) {
            if (discrete) {
                json.WriteNumberValue((long)a);
                continue;
            }
            json.WriteStartArray();
            foreach (float v in (float[])a)
                json.WriteNumberValue(BitConverter.SingleToUInt32Bits(v));
            json.WriteEndArray();
        }
        json.WriteEndArray();
        json.WriteEndObject();
    }

    /// <summary>Runs the replay script with Gymnasium on <c>PYTHONPATH</c> and pygame headless.</summary>
    /// <param name="python">Where Python runs.</param>
    /// <param name="actionsPath">The actions file.</param>
    /// <param name="replayPath">The replay file to write.</param>
    /// <exception cref="InvalidOperationException">The interpreter cannot start or the script fails; carries its stderr.</exception>
    private static void RunPython(PythonSide python, string actionsPath, string replayPath) {
        var start = new ProcessStartInfo(python.Python) {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };
        start.ArgumentList.Add(python.Script);
        start.ArgumentList.Add("--actions");
        start.ArgumentList.Add(actionsPath);
        start.ArgumentList.Add("--out");
        start.ArgumentList.Add(replayPath);
        string? existing = Environment.GetEnvironmentVariable("PYTHONPATH");
        start.Environment["PYTHONPATH"] = string.IsNullOrEmpty(existing) ? python.Gymnasium : python.Gymnasium + Path.PathSeparator + existing;
        start.Environment["SDL_VIDEODRIVER"] = "dummy";
        start.Environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1";
        using Process process = Process.Start(start) ?? throw new InvalidOperationException($"could not start '{python.Python}'");
        // Read both pipes concurrently: a full stderr pipe would otherwise block the script while we wait on stdout.
        var stderr = process.StandardError.ReadToEndAsync();
        process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        if (process.ExitCode != 0)
            throw new InvalidOperationException($"gymnasium_replay.py failed ({process.ExitCode}) with '{python.Python}':{Environment.NewLine}{stderr.Result}");
    }

    /// <summary>Reads the replay file's header.</summary>
    /// <param name="path">The file.</param>
    /// <returns>Records, frame width and height, observation bytes.</returns>
    /// <exception cref="InvalidOperationException">The magic number or version is wrong.</exception>
    private static (int Steps, int Width, int Height, int ObservationBytes) ReadHeader(string path) {
        using var file = File.OpenRead(path);
        Span<byte> header = stackalloc byte[HeaderBytes];
        file.ReadExactly(header);
        if (!header[..4].SequenceEqual(Magic) || BinaryPrimitives.ReadInt32LittleEndian(header[4..]) != 1)
            throw new InvalidOperationException($"{path} is not a version-1 replay file");
        return (BinaryPrimitives.ReadInt32LittleEndian(header[8..]), BinaryPrimitives.ReadInt32LittleEndian(header[16..]),
            BinaryPrimitives.ReadInt32LittleEndian(header[12..]), BinaryPrimitives.ReadInt32LittleEndian(header[20..]));
    }

    /// <summary>
    ///     Replays the actions in Gym.NET with rendering and compares every record with Gymnasium's: frame pixels,
    ///     observation bytes, reward bits, terminated and truncated. Gym.NET's frames are kept in a file of their own, so
    ///     the video shows exactly the frames compared here.
    /// </summary>
    /// <param name="recipe">The episode.</param>
    /// <param name="actions">The recorded actions.</param>
    /// <param name="replayPath">Gymnasium's records.</param>
    /// <param name="gymNetPath">Where Gym.NET's frames are written.</param>
    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    /// <param name="observationBytes">Bytes of one observation.</param>
    /// <returns>Per-record pixel counts and returns, whether the episode ended, and the stamp it earned.</returns>
    /// <exception cref="InvalidOperationException">Any difference, described.</exception>
    private static (int[] Different, double[] Returns, bool Ended, string? Stamp) Verify(EpisodeRecipe recipe, List<object> actions, string replayPath,
        string gymNetPath, int width, int height, int observationBytes) {
        using var file = File.OpenRead(replayPath);
        using var gymNet = File.Create(gymNetPath);
        file.Position = HeaderBytes;
        int frameBytes = width * height * 3;
        var record = new byte[frameBytes + observationBytes + 10];
        var different = new int[actions.Count + 1];
        var returns = new double[actions.Count + 1];
        using IEnv env = Gymnasium.Make(recipe.Id, new { render_mode = "rgb_array" });
        object? observation = env.Reset(seed: recipe.Seed).Observation;
        double total = 0;
        bool ended = false;
        string? stamp = null;
        for (int k = 0; k <= actions.Count; k++) {
            double reward = 0;
            bool terminated = false, truncated = false;
            if (k > 0) {
                object a = actions[k - 1];
                using NDArray action = a is long d ? NDArray.Scalar(d) : np.array((float[])a);
                (observation, reward, terminated, truncated, _) = env.Step(action);
                total += reward;
            }
            file.ReadExactly(record);
            byte[] ours = Frame(env, width, height, recipe.Id);
            gymNet.Write(ours);
            different[k] = CountDifferent(ours, record.AsSpan(0, frameBytes));
            returns[k] = total;
            string at = $"{recipe.Id} seed {recipe.Seed}, step {k}";
            if (different[k] != 0)
                throw new InvalidOperationException($"{at}: {different[k]} pixels differ from Gymnasium's frame");
            if (!ObservationBytes(observation).AsSpan().SequenceEqual(record.AsSpan(frameBytes, observationBytes)))
                throw new InvalidOperationException($"{at}: the observation's bytes differ from Gymnasium's");
            double theirs = BinaryPrimitives.ReadDoubleLittleEndian(record.AsSpan(frameBytes + observationBytes));
            if (BitConverter.DoubleToInt64Bits(theirs) != BitConverter.DoubleToInt64Bits(reward))
                throw new InvalidOperationException($"{at}: reward {reward:R} differs from Gymnasium's {theirs:R}");
            if ((record[^2] != 0) != terminated || (record[^1] != 0) != truncated)
                throw new InvalidOperationException($"{at}: terminated/truncated {terminated}/{truncated} differ from Gymnasium's");
            if (terminated || truncated) {
                ended = true;
                stamp = recipe.Stamp(terminated, truncated, total);
            }
        }
        return (different, returns, ended, stamp);
    }

    /// <summary>Renders the env and returns the frame's bytes, checking its size against Gymnasium's.</summary>
    /// <param name="env">The env.</param>
    /// <param name="width">The expected width.</param>
    /// <param name="height">The expected height.</param>
    /// <param name="id">The id, for the error.</param>
    /// <returns>The RGB bytes.</returns>
    /// <exception cref="InvalidOperationException">No RGB frame, or a different size.</exception>
    private static byte[] Frame(IEnv env, int width, int height, string id) {
        using RenderResult r = env.Render();
        if (r is not RenderResult.Rgb rgb || (int)rgb.Frame.shape[0] != height || (int)rgb.Frame.shape[1] != width)
            throw new InvalidOperationException($"{id}: Gym.NET's frame is not {width}x{height} RGB like Gymnasium's");
        return rgb.Frame.ToArray<byte>();
    }

    /// <summary>Counts the pixels whose three channels are not all equal.</summary>
    /// <param name="a">One frame's bytes.</param>
    /// <param name="b">The other's, the same length.</param>
    /// <returns>The count.</returns>
    private static int CountDifferent(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b) {
        // The common case is identical frames: one vectorized comparison settles it before any per-pixel loop.
        if (a.SequenceEqual(b))
            return 0;
        int n = 0;
        for (int i = 0; i < a.Length; i += 3)
            if (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2])
                n++;
        return n;
    }

    /// <summary>An observation's bytes in C order, as NumPy's <c>tobytes()</c> writes them.</summary>
    /// <param name="observation">An <see cref="NDArray"/> observation.</param>
    /// <returns>The bytes.</returns>
    /// <exception cref="NotSupportedException">A dtype the parity envs never return.</exception>
    private static byte[] ObservationBytes(object? observation) {
        var a = (NDArray)observation!;
        return a.typecode switch {
            NPTypeCode.Single => MemoryMarshal.AsBytes(a.ToArray<float>().AsSpan()).ToArray(),
            NPTypeCode.Double => MemoryMarshal.AsBytes(a.ToArray<double>().AsSpan()).ToArray(),
            NPTypeCode.Byte => a.ToArray<byte>(),
            NPTypeCode.Int64 => MemoryMarshal.AsBytes(a.ToArray<long>().AsSpan()).ToArray(),
            _ => throw new NotSupportedException($"observation dtype {a.typecode}"),
        };
    }
}

/// <summary>The parity segment's two verified replays: the lander and the race car.</summary>
/// <param name="Lander">LunarLander-v3's replay.</param>
/// <param name="Racer">CarRacing-v3's replay.</param>
internal sealed record ParityReplays(ParityReplay Lander, ParityReplay Racer) : IDisposable {
    /// <summary>Records and verifies both, the lander's whole episode and the race car's first 330 steps.</summary>
    /// <param name="python">Where Python runs.</param>
    /// <param name="log">Progress lines.</param>
    /// <returns>The replays.</returns>
    /// <exception cref="InvalidOperationException">Python failed, or a replay differs.</exception>
    /// <exception cref="IOException">The exchanged files cannot be written or read.</exception>
    public static ParityReplays Create(PythonSide python, TextWriter log) {
        ParityReplay lander = ParityReplay.Create(Episodes.Get("LunarLander-v3"), 1000, python, log);
        try {
            return new ParityReplays(lander, ParityReplay.Create(Episodes.Get("CarRacing-v3"), 330, python, log));
        } catch {
            // The lander's replay file would otherwise stay on disk: nothing else owns it yet.
            lander.Dispose();
            throw;
        }
    }

    /// <summary>Deletes both replay files.</summary>
    public void Dispose() {
        Lander.Dispose();
        Racer.Dispose();
    }
}
