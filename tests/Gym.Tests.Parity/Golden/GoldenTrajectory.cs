using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;

namespace Gym.Tests.Parity.Golden {
    /// <summary>
    ///     What a golden says about its own recording: the case it records and the stack it was recorded on (D15's manifest,
    ///     carried by every file, so a file is self-describing even when only some goldens are re-recorded).
    /// </summary>
    /// <param name="Suite">The case's suite.</param>
    /// <param name="Case">The case's name.</param>
    /// <param name="EnvId">The Gymnasium id.</param>
    /// <param name="Kwargs">The constructor kwargs, described, in order.</param>
    /// <param name="Seed">The reset seed.</param>
    /// <param name="Actions">The action source's description.</param>
    /// <param name="ReplaySamples">Whether replay samples the .NET action space rather than feeding the recorded actions.</param>
    /// <param name="MaxSteps">The step limit.</param>
    /// <param name="FrameEvery">The frame interval; 0 for none.</param>
    /// <param name="StateAttribute">The compared state attribute, or <see langword="null"/>.</param>
    /// <param name="GymnasiumCommit">The <c>refs/Gymnasium</c> commit the trajectory was recorded from.</param>
    /// <param name="Platform">The runtime identifier it was recorded on (<c>win-x64</c>).</param>
    /// <param name="Stack">The live stack's versions at recording time, keyed as <c>tools/parity/manifest.json</c> keys them.</param>
    public sealed record GoldenHeader(
        string Suite,
        string Case,
        string EnvId,
        IReadOnlyList<KeyValuePair<string, ValueFacts>> Kwargs,
        long Seed,
        string Actions,
        bool ReplaySamples,
        int MaxSteps,
        int FrameEvery,
        string? StateAttribute,
        string GymnasiumCommit,
        string Platform,
        IReadOnlyList<KeyValuePair<string, string>> Stack) {
        /// <summary>The format tag of the first line; a reader refuses any other, so a format change cannot be misread.</summary>
        public const string Format = "gym.net-golden/1";

        /// <summary>The header a recording of <paramref name="case"/> on the current stack gets.</summary>
        /// <param name="case">The case.</param>
        /// <param name="gymnasiumCommit">The checked-out Gymnasium commit.</param>
        /// <param name="platform">This process's runtime identifier.</param>
        /// <param name="stack">The live stack's versions.</param>
        /// <returns>The header.</returns>
        public static GoldenHeader For(GoldenCase @case, string gymnasiumCommit, string platform, IEnumerable<KeyValuePair<string, string>> stack) {
            return new GoldenHeader(@case.Suite, @case.Name, @case.EnvId, @case.DescribedKwargs(), @case.Seed, @case.Actions.Description,
                @case.Actions.ReplaySamples, @case.MaxSteps, @case.FrameEvery, @case.StateAttribute, gymnasiumCommit, platform,
                stack.OrderBy(entry => entry.Key, StringComparer.Ordinal).ToList());
        }

        /// <summary>
        ///     The differences between what this golden recorded and what <paramref name="case"/> asks for today, one line per
        ///     differing field; empty when the golden still records the case.
        /// </summary>
        /// <param name="case">The catalog's case.</param>
        /// <returns>The differences, for a "re-record" message.</returns>
        public IReadOnlyList<string> DifferencesFrom(GoldenCase @case) {
            var differences = new List<string>();
            void Check(string field, string recorded, string wanted) {
                if (!string.Equals(recorded, wanted, StringComparison.Ordinal))
                    differences.Add($"{field}: recorded {recorded}, the catalog asks for {wanted}");
            }

            Check("env id", EnvId, @case.EnvId);
            Check("kwargs", RenderKwargs(Kwargs), RenderKwargs(@case.DescribedKwargs()));
            Check("seed", Seed.ToString(CultureInfo.InvariantCulture), @case.Seed.ToString(CultureInfo.InvariantCulture));
            Check("actions", Actions, @case.Actions.Description);
            Check("max steps", MaxSteps.ToString(CultureInfo.InvariantCulture), @case.MaxSteps.ToString(CultureInfo.InvariantCulture));
            Check("frame every", FrameEvery.ToString(CultureInfo.InvariantCulture), @case.FrameEvery.ToString(CultureInfo.InvariantCulture));
            Check("state attribute", StateAttribute ?? "none", @case.StateAttribute ?? "none");
            return differences;
        }

        /// <summary>Renders kwargs canonically, so two descriptions of the same kwargs compare equal as text.</summary>
        /// <param name="kwargs">The described kwargs.</param>
        /// <returns>For example <c>{sutton_barto_reward=True}</c>.</returns>
        private static string RenderKwargs(IReadOnlyList<KeyValuePair<string, ValueFacts>> kwargs) {
            return "{" + string.Join(", ", kwargs.Select(k => $"{k.Key}={k.Value.Canonical()}")) + "}";
        }
    }

    /// <summary>One line of a golden after its header: a verified call or a verified frame, in the order they happened.</summary>
    public abstract record GoldenEntry {
        /// <summary>Restricts the hierarchy to calls and frames, the two things a replay knows how to repeat.</summary>
        private GoldenEntry() { }

        /// <summary>The trajectory position: 0 for the reset and what follows it, n after the n-th step.</summary>
        public abstract int Step { get; }

        /// <summary>
        ///     A <c>reset</c> or <c>step</c> call and what Gymnasium returned, which matched Gym.NET when it was recorded.
        /// </summary>
        /// <param name="Label">The call as reports name it (<c>reset(seed=0)</c>, <c>step #3: step(1)</c>).</param>
        /// <param name="Step">The trajectory position.</param>
        /// <param name="Action">The action as Gymnasium received it; <see langword="null"/> for the reset.</param>
        /// <param name="Result">What the call returned or raised.</param>
        /// <param name="State">The state attribute after the call, when the live cell compared it.</param>
        /// <param name="Rng">The env generator's state after the call, when the live cell compared it.</param>
        public sealed record Call(string Label, int Step, ValueFacts? Action, CallResult Result, ValueFacts? State, ValueFacts? Rng) : GoldenEntry {
            /// <inheritdoc />
            public override int Step { get; } = Step;

            /// <summary>Whether this is the trajectory's reset rather than a step.</summary>
            public bool IsReset => Action is null && Step == 0;
        }

        /// <summary>A frame both sides rendered identically, stored as a PNG next to the trajectory.</summary>
        /// <param name="Label">Where in the trajectory (<c>after reset</c>, <c>after step 50</c>).</param>
        /// <param name="Step">The trajectory position.</param>
        /// <param name="File">The PNG's path relative to the suite's directory, with forward slashes.</param>
        public sealed record Frame(string Label, int Step, string File) : GoldenEntry {
            /// <inheritdoc />
            public override int Step { get; } = Step;
        }
    }

    /// <summary>
    ///     A golden trajectory: its header and its entries. On disk it is gzip-compressed JSON lines (decision D15): the header
    ///     on the first line, then one line per verified call or frame; frames themselves are PNG files beside it.
    /// </summary>
    /// <remarks>
    ///     The text is UTF-8 with <c>\n</c> line endings and no indentation, written in a fixed property order, so the same
    ///     trajectory always serializes to the same bytes. Only the compressed form may differ between runtimes (zlib versions
    ///     differ between .NET 8 and 10), which is why <see cref="GoldenStore"/> compares content, not files, before writing.
    /// </remarks>
    public sealed class GoldenTrajectory {
        /// <summary>
        ///     The writer options: no indentation, and dtypes such as <c>&lt;f4</c> kept readable instead of being escaped as a
        ///     <c>\u</c> sequence, which the default encoder does to every <c>&lt;</c>.
        /// </summary>
        private static readonly JsonWriterOptions WriterOptions = new() {
            Indented = false,
            // Golden files are never embedded in HTML, so the relaxed encoder's only effect is readable dtypes and messages.
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        /// <summary>Creates a trajectory.</summary>
        /// <param name="header">The header.</param>
        /// <param name="entries">The entries, in order.</param>
        public GoldenTrajectory(GoldenHeader header, IReadOnlyList<GoldenEntry> entries) {
            Header = header;
            Entries = entries;
        }

        /// <summary>The header.</summary>
        public GoldenHeader Header { get; }

        /// <summary>The verified calls and frames, in the order they happened.</summary>
        public IReadOnlyList<GoldenEntry> Entries { get; }

        /// <summary>The trajectory as JSON lines, uncompressed.</summary>
        /// <returns>UTF-8 text, one JSON object per line, each line ending in <c>\n</c>.</returns>
        public byte[] ToJsonLines() {
            using var buffer = new MemoryStream();
            WriteLine(buffer, WriteHeader);
            foreach (GoldenEntry entry in Entries)
                WriteLine(buffer, writer => WriteEntry(writer, entry));
            return buffer.ToArray();
        }

        /// <summary>Reads a trajectory from JSON lines.</summary>
        /// <param name="jsonLines">The uncompressed UTF-8 text.</param>
        /// <returns>The trajectory.</returns>
        /// <exception cref="InvalidDataException">Thrown when the text is empty, its format tag is not <see cref="GoldenHeader.Format"/>, or a line is malformed.</exception>
        public static GoldenTrajectory FromJsonLines(byte[] jsonLines) {
            string[] lines = Encoding.UTF8.GetString(jsonLines).Split('\n', StringSplitOptions.RemoveEmptyEntries);
            if (lines.Length == 0)
                throw new InvalidDataException("A golden trajectory has no header line.");

            GoldenHeader header;
            using (JsonDocument document = JsonDocument.Parse(lines[0]))
                header = ReadHeader(document.RootElement);
            var entries = new List<GoldenEntry>(lines.Length - 1);
            for (int i = 1; i < lines.Length; i++) {
                using JsonDocument document = JsonDocument.Parse(lines[i]);
                entries.Add(ReadEntry(document.RootElement, i + 1));
            }

            return new GoldenTrajectory(header, entries);
        }

        /// <summary>Compresses JSON lines with gzip at the optimal level.</summary>
        /// <param name="jsonLines">The text.</param>
        /// <returns>The gzip stream's bytes.</returns>
        public static byte[] Compress(byte[] jsonLines) {
            using var output = new MemoryStream();
            using (var gzip = new GZipStream(output, CompressionLevel.SmallestSize, leaveOpen: true))
                gzip.Write(jsonLines);
            return output.ToArray();
        }

        /// <summary>Decompresses a golden file's bytes.</summary>
        /// <param name="gzip">The file's bytes.</param>
        /// <returns>The JSON lines.</returns>
        /// <exception cref="InvalidDataException">Thrown when the bytes are not a gzip stream.</exception>
        public static byte[] Decompress(byte[] gzip) {
            using var input = new GZipStream(new MemoryStream(gzip), CompressionMode.Decompress);
            using var output = new MemoryStream();
            input.CopyTo(output);
            return output.ToArray();
        }

        /// <summary>Writes one JSON object and a <c>\n</c>.</summary>
        /// <param name="stream">The destination.</param>
        /// <param name="write">Writes the object.</param>
        private static void WriteLine(Stream stream, Action<Utf8JsonWriter> write) {
            using (var writer = new Utf8JsonWriter(stream, WriterOptions))
                write(writer);
            stream.WriteByte((byte)'\n');
        }

        /// <summary>Writes the header object.</summary>
        /// <param name="writer">The writer.</param>
        private void WriteHeader(Utf8JsonWriter writer) {
            GoldenHeader h = Header;
            writer.WriteStartObject();
            writer.WriteString("format", GoldenHeader.Format);
            writer.WriteString("suite", h.Suite);
            writer.WriteString("case", h.Case);
            writer.WriteString("env_id", h.EnvId);
            writer.WriteStartArray("kwargs");
            foreach (KeyValuePair<string, ValueFacts> kwarg in h.Kwargs) {
                writer.WriteStartArray();
                writer.WriteStringValue(kwarg.Key);
                GoldenCodec.Write(writer, kwarg.Value);
                writer.WriteEndArray();
            }

            writer.WriteEndArray();
            writer.WriteNumber("seed", h.Seed);
            writer.WriteString("actions", h.Actions);
            writer.WriteBoolean("replay_samples", h.ReplaySamples);
            writer.WriteNumber("max_steps", h.MaxSteps);
            writer.WriteNumber("frame_every", h.FrameEvery);
            if (h.StateAttribute is null)
                writer.WriteNull("state_attribute");
            else
                writer.WriteString("state_attribute", h.StateAttribute);
            writer.WriteString("gymnasium_commit", h.GymnasiumCommit);
            writer.WriteString("platform", h.Platform);
            writer.WriteStartObject("stack");
            foreach (KeyValuePair<string, string> entry in h.Stack)
                writer.WriteString(entry.Key, entry.Value);
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        /// <summary>Writes one entry's object.</summary>
        /// <param name="writer">The writer.</param>
        /// <param name="entry">The entry.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown for an entry type other than a call or a frame, which the private constructor rules out.</exception>
        private static void WriteEntry(Utf8JsonWriter writer, GoldenEntry entry) {
            writer.WriteStartObject();
            switch (entry) {
                case GoldenEntry.Call call:
                    writer.WriteString("call", call.Label);
                    writer.WriteNumber("step", call.Step);
                    GoldenCodec.WriteOptional(writer, "action", call.Action);
                    GoldenCodec.WriteResult(writer, call.Result);
                    GoldenCodec.WriteOptional(writer, "state", call.State);
                    GoldenCodec.WriteOptional(writer, "rng", call.Rng);
                    break;
                case GoldenEntry.Frame frame:
                    writer.WriteString("frame", frame.Label);
                    writer.WriteNumber("step", frame.Step);
                    writer.WriteString("png", frame.File);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(entry), entry.GetType().Name, "Only calls and frames are golden entries.");
            }

            writer.WriteEndObject();
        }

        /// <summary>Reads the header line.</summary>
        /// <param name="root">The line's object.</param>
        /// <returns>The header.</returns>
        /// <exception cref="InvalidDataException">Thrown for another format tag or a missing member.</exception>
        private static GoldenHeader ReadHeader(JsonElement root) {
            string format = GoldenCodec.RequiredString(root, "format");
            if (format != GoldenHeader.Format)
                throw new InvalidDataException($"The golden's format is '{format}'; this reader reads '{GoldenHeader.Format}'. Re-record it.");

            var kwargs = new List<KeyValuePair<string, ValueFacts>>();
            foreach (JsonElement pair in GoldenCodec.Required(root, "kwargs").EnumerateArray())
                kwargs.Add(new KeyValuePair<string, ValueFacts>(pair[0].GetString()!, GoldenCodec.Read(pair[1])));
            var stack = GoldenCodec.Required(root, "stack").EnumerateObject()
                .Select(p => new KeyValuePair<string, string>(p.Name, p.Value.GetString()!)).ToList();
            JsonElement state = GoldenCodec.Required(root, "state_attribute");

            return new GoldenHeader(
                GoldenCodec.RequiredString(root, "suite"),
                GoldenCodec.RequiredString(root, "case"),
                GoldenCodec.RequiredString(root, "env_id"),
                kwargs,
                GoldenCodec.Required(root, "seed").GetInt64(),
                GoldenCodec.RequiredString(root, "actions"),
                GoldenCodec.Required(root, "replay_samples").GetBoolean(),
                GoldenCodec.Required(root, "max_steps").GetInt32(),
                GoldenCodec.Required(root, "frame_every").GetInt32(),
                state.ValueKind == JsonValueKind.Null ? null : state.GetString(),
                GoldenCodec.RequiredString(root, "gymnasium_commit"),
                GoldenCodec.RequiredString(root, "platform"),
                stack);
        }

        /// <summary>Reads one entry line.</summary>
        /// <param name="root">The line's object.</param>
        /// <param name="lineNumber">The line's 1-based number, for the error message.</param>
        /// <returns>The entry.</returns>
        /// <exception cref="InvalidDataException">Thrown when the line is neither a call nor a frame, or is malformed.</exception>
        private static GoldenEntry ReadEntry(JsonElement root, int lineNumber) {
            try {
                if (root.TryGetProperty("call", out JsonElement call)) {
                    return new GoldenEntry.Call(call.GetString()!, GoldenCodec.Required(root, "step").GetInt32(),
                        GoldenCodec.ReadOptional(root, "action"), GoldenCodec.ReadResult(root),
                        GoldenCodec.ReadOptional(root, "state"), GoldenCodec.ReadOptional(root, "rng"));
                }

                if (root.TryGetProperty("frame", out JsonElement frame))
                    return new GoldenEntry.Frame(frame.GetString()!, GoldenCodec.Required(root, "step").GetInt32(), GoldenCodec.RequiredString(root, "png"));
            } catch (InvalidDataException exception) {
                throw new InvalidDataException($"Golden line {lineNumber}: {exception.Message}", exception);
            }

            throw new InvalidDataException($"Golden line {lineNumber} is neither a call nor a frame.");
        }
    }
}
