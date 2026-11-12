using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using Gym.Tests.Parity.Golden;

namespace Gym.Tests.Parity.Frames {
    /// <summary>
    ///     The checked-in measurement the MuJoCo frame cells are gated on (DR-71): for each id, render mode and state, how
    ///     many pixels (or depth values) of Gym.NET's frame differed from Gymnasium's when it was recorded, the largest
    ///     difference, and Gymnasium's own render-to-render noise at that state; and the GPU that drew Gymnasium's frames.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A frame passes while it differs in at most its recorded count plus the noise, the count Gymnasium's next render
    ///     can move by itself (Humanoid-v5's frames vary between renders on the reference GPU). The bar only moves when it is
    ///     re-recorded (<c>GYMNET_PARITY_RECORD_BASELINE=1</c>), which a deliberate improvement of the renderer does, so a
    ///     regression fails the cell instead of quietly raising it.
    ///     </para>
    ///     <para>
    ///     The measurement is the reference GPU's: part of Gymnasium's pixels are the GPU's own choices
    ///     (<c>tools/parity/mujoco/gpu_probe.py</c>), so on another GPU or driver the cells are Inconclusive rather than gated
    ///     on numbers that do not apply. Record from one target framework (<c>-f net10.0</c>): the cells rewrite this one
    ///     file, and two test hosts recording at once would race on it.
    ///     </para>
    /// </remarks>
    internal sealed class MujocoFrameBaseline {
        /// <summary>The variable that turns the cells into recorders: <c>1</c> writes what they measure into the file.</summary>
        internal const string RecordVariable = "GYMNET_PARITY_RECORD_BASELINE";

        /// <summary>One state's measurement.</summary>
        /// <param name="Differing">Pixels (RGB) or values (depth) of Gym.NET's frame that differed from Gymnasium's.</param>
        /// <param name="Largest">The largest channel delta (RGB, 0–255) or ulp distance (depth); reported, not gated.</param>
        /// <param name="Noise">The most pixels or values in which Gymnasium's own renders of the state differed from each other.</param>
        internal sealed record Entry(long Differing, long Largest, long Noise);

        /// <summary>The entries by key (<c>id mode label</c>), sorted so the file's diffs stay readable.</summary>
        private readonly SortedDictionary<string, Entry> _entries = new(StringComparer.Ordinal);

        /// <summary>Whether this run records (<see cref="RecordVariable"/> is <c>1</c>).</summary>
        internal static bool RecordRequested => Environment.GetEnvironmentVariable(RecordVariable) == "1";

        /// <summary>The file: <c>tests/Gym.Tests.Parity/Baselines/mujoco_frames.json</c> of the checkout.</summary>
        /// <exception cref="InvalidOperationException">Thrown when the test binaries are not inside a Gym.NET checkout.</exception>
        internal static string FilePath => GoldenStore.CheckoutPath("tests", "Gym.Tests.Parity", "Baselines", "mujoco_frames.json");

        /// <summary>The GPU (its <c>GL_RENDERER</c> and <c>GL_VERSION</c>) Gymnasium's frames were drawn on when this was recorded.</summary>
        internal string Gpu { get; private set; } = "";

        /// <summary>The day it was recorded, <c>yyyy-MM-dd</c>.</summary>
        internal string Recorded { get; private set; } = "";

        /// <summary>Reads the file, or starts an empty baseline when there is none yet.</summary>
        /// <returns>The baseline.</returns>
        /// <exception cref="InvalidDataException">Thrown when the file is not a baseline this reader understands.</exception>
        internal static MujocoFrameBaseline Load() {
            var baseline = new MujocoFrameBaseline();
            if (!File.Exists(FilePath))
                return baseline;
            try {
                using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(FilePath));
                JsonElement root = document.RootElement;
                baseline.Gpu = root.GetProperty("gpu").GetString() ?? "";
                baseline.Recorded = root.GetProperty("recorded").GetString() ?? "";
                foreach (JsonProperty frame in root.GetProperty("frames").EnumerateObject()) {
                    JsonElement e = frame.Value;
                    baseline._entries[frame.Name] = new Entry(e.GetProperty("differing").GetInt64(), e.GetProperty("largest").GetInt64(), e.GetProperty("noise").GetInt64());
                }
            } catch (Exception exception) when (exception is JsonException or KeyNotFoundException or InvalidOperationException or FormatException) {
                throw new InvalidDataException($"{FilePath} is not a MuJoCo frame baseline: {exception.Message}", exception);
            }

            return baseline;
        }

        /// <summary>The recorded measurement of a state.</summary>
        /// <param name="key">The state's key, <c>id mode label</c>.</param>
        /// <param name="entry">The measurement, when there is one.</param>
        /// <returns>Whether the baseline holds the state.</returns>
        internal bool TryGet(string key, out Entry entry) {
            bool found = _entries.TryGetValue(key, out Entry? value);
            entry = value!;
            return found;
        }

        /// <summary>Records a state's measurement, and the GPU and day it was taken on.</summary>
        /// <param name="key">The state's key.</param>
        /// <param name="entry">The measurement.</param>
        /// <param name="gpu">The GPU that drew Gymnasium's frame.</param>
        internal void Set(string key, Entry entry, string gpu) {
            _entries[key] = entry;
            Gpu = gpu;
            Recorded = DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        }

        /// <summary>Writes the file atomically, entries sorted by key, one per line.</summary>
        internal void Save() {
            var text = new StringBuilder();
            text.Append("{\n");
            text.Append("  \"about\": \"DR-71: the MuJoCo frame cells' bar. Gym.NET's frame of each state differed from Gymnasium's in 'differing' pixels (rgb_array) or values (depth_array); a cell fails above differing + noise, noise being how much Gymnasium's own renders of the state differed. Re-record with GYMNET_PARITY_RECORD_BASELINE=1 (-f net10.0) on the reference GPU.\",\n");
            text.Append($"  \"gpu\": {JsonSerializer.Serialize(Gpu)},\n");
            text.Append($"  \"recorded\": {JsonSerializer.Serialize(Recorded)},\n");
            text.Append("  \"frames\": {\n");
            int i = 0;
            foreach ((string key, Entry e) in _entries) {
                text.Append(CultureInfo.InvariantCulture,
                    $"    {JsonSerializer.Serialize(key)}: {{ \"differing\": {e.Differing}, \"largest\": {e.Largest}, \"noise\": {e.Noise} }}");
                text.Append(++i < _entries.Count ? ",\n" : "\n");
            }

            text.Append("  }\n}\n");
            GoldenStore.WriteAtomically(FilePath, Encoding.UTF8.GetBytes(text.ToString()));
        }
    }
}
