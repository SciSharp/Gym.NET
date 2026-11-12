using System;
using System.IO;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Session;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Tests.Parity.Golden {
    /// <summary>
    ///     Where goldens live (<c>tests/Golden/&lt;suite&gt;/&lt;case&gt;.jsonl.gz</c>, frames in
    ///     <c>tests/Golden/&lt;suite&gt;/&lt;case&gt;/</c>), whether this run records them, and how they are written: only when
    ///     their content changed, and atomically.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     The store finds the checkout by walking up to <c>Gym.NET.sln</c> on its own, without the Python session, so a
    ///     replay runs on a machine with no Python and no initialized <c>refs/Gymnasium</c>.
    ///     </para>
    ///     <para>
    ///     Writes compare content, not bytes: the gzip stream of the same text differs between .NET 8 and .NET 10 (their
    ///     zlib builds differ), and a PNG encoder may change, so an unchanged recording must not rewrite its files and churn
    ///     the history. Both target frameworks may record in parallel (<c>dotnet test</c> runs them as two processes), so a
    ///     file is written to a temporary name and moved into place, never left half-written.
    ///     </para>
    /// </remarks>
    public static class GoldenStore {
        /// <summary>
        ///     Set to <c>1</c> to make the live golden cells write what they verified (DR-04's record mode). Anything else,
        ///     including unset, only verifies.
        /// </summary>
        public const string RecordVariable = "GYMNET_PARITY_RECORD";

        /// <summary>The checkout root, found once; <see langword="null"/> when no <c>Gym.NET.sln</c> is above the test binaries.</summary>
        private static readonly Lazy<string?> RepositoryRoot = new(() => FindRepositoryRoot(AppContext.BaseDirectory));

        /// <summary>Whether this run records goldens (<see cref="RecordVariable"/> is <c>1</c>).</summary>
        public static bool RecordRequested => Environment.GetEnvironmentVariable(RecordVariable) == "1";

        /// <summary><c>tests/Golden</c> of the checkout.</summary>
        /// <exception cref="InvalidOperationException">Thrown when the test binaries are not inside a Gym.NET checkout.</exception>
        public static string Root => Path.Combine(RepositoryRoot.Value ?? throw new InvalidOperationException(
            $"No Gym.NET.sln in {AppContext.BaseDirectory} or above it: goldens are read from the checkout's tests/Golden."), "tests", "Golden");

        /// <summary><c>tools/parity/manifest.json</c>: the pinned Gymnasium commit a golden must have been recorded from.</summary>
        /// <exception cref="InvalidOperationException">Thrown when the test binaries are not inside a Gym.NET checkout.</exception>
        public static string ManifestPath => Path.Combine(RepositoryRoot.Value ?? throw new InvalidOperationException(
            $"No Gym.NET.sln in {AppContext.BaseDirectory} or above it."), "tools", "parity", "manifest.json");

        /// <summary>A path inside the checkout, for the parity data that is not a golden (the MuJoCo frame baseline).</summary>
        /// <param name="parts">The path's parts below the checkout's root.</param>
        /// <returns>The absolute path.</returns>
        /// <exception cref="InvalidOperationException">Thrown when the test binaries are not inside a Gym.NET checkout.</exception>
        public static string CheckoutPath(params string[] parts) => Path.Combine(new[] {
            RepositoryRoot.Value ?? throw new InvalidOperationException($"No Gym.NET.sln in {AppContext.BaseDirectory} or above it."),
        }.Concat(parts).ToArray());

        /// <summary>The trajectory file of a case.</summary>
        /// <param name="case">The case.</param>
        /// <returns><c>tests/Golden/&lt;suite&gt;/&lt;name&gt;.jsonl.gz</c>.</returns>
        public static string TrajectoryPath(GoldenCase @case) => Path.Combine(Root, @case.Suite, @case.Name + ".jsonl.gz");

        /// <summary>The directory a case's frames are stored in.</summary>
        /// <param name="case">The case.</param>
        /// <returns><c>tests/Golden/&lt;suite&gt;/&lt;name&gt;</c>.</returns>
        public static string FramesDirectory(GoldenCase @case) => Path.Combine(Root, @case.Suite, @case.Name);

        /// <summary>Resolves a frame entry's file, which the trajectory names relative to its suite's directory.</summary>
        /// <param name="case">The case.</param>
        /// <param name="relative">The entry's path, with forward slashes.</param>
        /// <returns>The absolute path.</returns>
        public static string FramePath(GoldenCase @case, string relative) => Path.Combine(Root, @case.Suite, relative.Replace('/', Path.DirectorySeparatorChar));

        /// <summary>Reads a case's committed golden.</summary>
        /// <param name="case">The case.</param>
        /// <returns>The trajectory, or <see langword="null"/> when none was recorded yet.</returns>
        /// <exception cref="InvalidDataException">Thrown when the file exists but is not a golden this reader understands.</exception>
        public static GoldenTrajectory? TryLoad(GoldenCase @case) {
            string path = TrajectoryPath(@case);
            if (!File.Exists(path))
                return null;
            return GoldenTrajectory.FromJsonLines(GoldenTrajectory.Decompress(File.ReadAllBytes(path)));
        }

        /// <summary>
        ///     Writes a trajectory unless the committed file already holds the same text.
        /// </summary>
        /// <param name="case">The case it records.</param>
        /// <param name="trajectory">The trajectory.</param>
        /// <returns><see langword="true"/> when the file was written, <see langword="false"/> when it was already current.</returns>
        public static bool SaveTrajectory(GoldenCase @case, GoldenTrajectory trajectory) {
            string path = TrajectoryPath(@case);
            byte[] text = trajectory.ToJsonLines();
            if (File.Exists(path)) {
                byte[] current;
                try {
                    current = GoldenTrajectory.Decompress(File.ReadAllBytes(path));
                } catch (InvalidDataException) {
                    // A truncated or foreign file is replaced, not trusted.
                    current = Array.Empty<byte>();
                }

                if (current.AsSpan().SequenceEqual(text))
                    return false;
            }

            WriteAtomically(path, GoldenTrajectory.Compress(text));
            return true;
        }

        /// <summary>Writes a frame as a PNG unless the file already holds the same pixels.</summary>
        /// <param name="path">The PNG's path.</param>
        /// <param name="frame">The frame.</param>
        /// <returns><see langword="true"/> when the file was written.</returns>
        public static bool SaveFrame(string path, Frame frame) {
            if (File.Exists(path)) {
                try {
                    Frame current = LoadFrame(path);
                    if (current.Height == frame.Height && current.Width == frame.Width && current.Rgb.AsSpan().SequenceEqual(frame.Rgb))
                        return false;
                } catch (Exception exception) when (exception is UnknownImageFormatException or InvalidImageContentException) {
                    // An unreadable PNG is replaced below.
                }
            }

            using var buffer = new MemoryStream();
            using (Image<Rgb24> image = Image.LoadPixelData<Rgb24>(frame.Rgb, frame.Width, frame.Height))
                image.SaveAsPng(buffer);
            WriteAtomically(path, buffer.ToArray());
            return true;
        }

        /// <summary>Reads a golden frame.</summary>
        /// <param name="path">The PNG's path.</param>
        /// <returns>The frame, in <c>rgb_array</c> layout.</returns>
        /// <exception cref="FileNotFoundException">Thrown when the file is missing.</exception>
        /// <exception cref="UnknownImageFormatException">Thrown when the file is not an image.</exception>
        public static Frame LoadFrame(string path) {
            using Image image = Image.Load(path);
            return Frame.FromImage(image);
        }

        /// <summary>Whether this process runs on the reference platform of <c>tools/parity/manifest.json</c>, read without the Python session.</summary>
        /// <returns><see langword="true"/> on the reference platform.</returns>
        public static bool OnReferencePlatform() {
            StackManifest manifest = StackManifest.Load(ManifestPath);
            return string.Equals(GymPythonSession.Platform, manifest.ReferencePlatform, StringComparison.Ordinal);
        }

        /// <summary>The pinned Gymnasium commit, read without the Python session.</summary>
        /// <returns>The full commit id <c>tools/parity/manifest.json</c> pins.</returns>
        /// <exception cref="InvalidDataException">Thrown when the manifest pins no commit.</exception>
        public static string PinnedGymnasiumCommit() {
            StackManifest manifest = StackManifest.Load(ManifestPath);
            if (!manifest.Stack.TryGetValue("gymnasium.commit", out string? commit))
                throw new InvalidDataException($"{ManifestPath} pins no gymnasium.commit.");
            return commit;
        }

        /// <summary>Writes a file through a temporary sibling and a move, so readers and a parallel writer never see half of it.</summary>
        /// <param name="path">The destination.</param>
        /// <param name="content">The bytes.</param>
        internal static void WriteAtomically(string path, byte[] content) {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            // A per-process name, so the two target frameworks recording at once never share a temporary file.
            string temporary = $"{path}.{Environment.ProcessId}.tmp";
            File.WriteAllBytes(temporary, content);
            File.Move(temporary, path, overwrite: true);
        }

        /// <summary>Walks up from a directory to the one holding <c>Gym.NET.sln</c>.</summary>
        /// <param name="start">Where to start, the test assembly's directory.</param>
        /// <returns>The checkout's root, or <see langword="null"/> when there is none above.</returns>
        private static string? FindRepositoryRoot(string start) {
            for (var directory = new DirectoryInfo(start); directory != null; directory = directory.Parent) {
                if (File.Exists(Path.Combine(directory.FullName, "Gym.NET.sln")))
                    return directory.FullName;
            }

            return null;
        }
    }
}
