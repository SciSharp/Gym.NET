using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     The Gym.NET checkout the test assembly was built from: where the pinned Gymnasium, the stack manifest and the
    ///     optional reference venv live. It is found by walking up from the test binaries, so the suite runs the same from
    ///     <c>dotnet test</c>, an IDE, or the test executable, whatever the working directory.
    /// </summary>
    public sealed class RepositoryLayout {
        /// <summary>Creates the layout of the checkout rooted at <paramref name="root"/>; <see cref="TryDiscover"/> finds the root.</summary>
        /// <param name="root">The directory that holds <c>Gym.NET.sln</c>.</param>
        private RepositoryLayout(string root) {
            Root = root;
        }

        /// <summary>The checkout's root directory, the one holding <c>Gym.NET.sln</c>.</summary>
        public string Root { get; }

        /// <summary>
        ///     The <c>refs/Gymnasium</c> submodule. Its directory goes first on the embedded interpreter's <c>sys.path</c>, so
        ///     <c>import gymnasium</c> loads the pinned source rather than any installed copy.
        /// </summary>
        public string GymnasiumRoot => Path.Combine(Root, "refs", "Gymnasium");

        /// <summary>The <c>gymnasium</c> package inside <see cref="GymnasiumRoot"/>; the session checks <c>gymnasium.__file__</c> lies in it.</summary>
        public string GymnasiumPackage => Path.Combine(GymnasiumRoot, "gymnasium");

        /// <summary><c>tools/parity</c>: the manifest, the requirements and the setup scripts of the reference stack.</summary>
        public string ParityTools => Path.Combine(Root, "tools", "parity");

        /// <summary><c>tools/parity/manifest.json</c>, the reference stack every run is checked against.</summary>
        public string ManifestPath => Path.Combine(ParityTools, "manifest.json");

        /// <summary>
        ///     <c>tools/parity/.venv</c>, created by <c>setup_venv.ps1</c>/<c>setup_venv.sh</c>. When it exists and
        ///     <c>GYMNET_PARITY_VENV</c> is unset, the session runs on it instead of probing for an interpreter.
        /// </summary>
        public string DefaultVenv => Path.Combine(ParityTools, ".venv");

        /// <summary>
        ///     Finds the checkout that contains <paramref name="startDirectory"/>, and checks that its Gymnasium submodule is
        ///     initialized, because an empty <c>refs/Gymnasium</c> would otherwise surface later as an import error.
        /// </summary>
        /// <param name="startDirectory">Where to start walking up; the test assembly's directory.</param>
        /// <param name="reason">Why no layout was returned, for the session's failure message; empty on success.</param>
        /// <returns>The layout, or <see langword="null"/> when there is no <c>Gym.NET.sln</c> above the start or the submodule is empty.</returns>
        public static RepositoryLayout? TryDiscover(string startDirectory, out string reason) {
            for (var directory = new DirectoryInfo(startDirectory); directory != null; directory = directory.Parent) {
                if (!File.Exists(Path.Combine(directory.FullName, "Gym.NET.sln")))
                    continue;

                var layout = new RepositoryLayout(directory.FullName);
                if (!File.Exists(Path.Combine(layout.GymnasiumPackage, "__init__.py"))) {
                    reason = $"{layout.GymnasiumRoot} is not initialized. Run: git submodule update --init refs/Gymnasium";
                    return null;
                }

                reason = "";
                return layout;
            }

            reason = $"no Gym.NET.sln in {startDirectory} or any directory above it";
            return null;
        }

        /// <summary>
        ///     The commit <c>refs/Gymnasium</c> has checked out, read from the submodule's git directory without running git,
        ///     so the pin check needs no git on <c>PATH</c>.
        /// </summary>
        /// <param name="detail">Why the commit could not be read, when <see langword="null"/> is returned; empty otherwise.</param>
        /// <returns>The 40-character commit id, or <see langword="null"/> when the repository metadata cannot be read.</returns>
        /// <remarks>
        ///     A submodule's <c>.git</c> is a file (<c>gitdir: ../../.git/modules/refs/Gymnasium</c>); a plain clone has a
        ///     directory. Its <c>HEAD</c> is a commit id when detached, as a submodule checkout normally is, or a symbolic ref,
        ///     which is resolved through the loose ref file and then <c>packed-refs</c>.
        /// </remarks>
        public string? ReadGymnasiumHead(out string detail) {
            try {
                string dotGit = Path.Combine(GymnasiumRoot, ".git");
                string gitDirectory;
                if (Directory.Exists(dotGit)) {
                    gitDirectory = dotGit;
                } else if (File.Exists(dotGit)) {
                    string pointer = File.ReadAllText(dotGit).Trim();
                    const string prefix = "gitdir:";
                    if (!pointer.StartsWith(prefix, StringComparison.Ordinal)) {
                        detail = $"{dotGit} is not a gitdir pointer: '{pointer}'";
                        return null;
                    }

                    gitDirectory = Path.GetFullPath(Path.Combine(GymnasiumRoot, pointer.Substring(prefix.Length).Trim()));
                } else {
                    detail = $"{GymnasiumRoot} has no .git";
                    return null;
                }

                string head = File.ReadAllText(Path.Combine(gitDirectory, "HEAD")).Trim();
                const string symbolic = "ref: ";
                if (!head.StartsWith(symbolic, StringComparison.Ordinal)) {
                    detail = "";
                    return head;
                }

                string reference = head.Substring(symbolic.Length).Trim();
                string loose = Path.Combine(gitDirectory, reference.Replace('/', Path.DirectorySeparatorChar));
                if (File.Exists(loose)) {
                    detail = "";
                    return File.ReadAllText(loose).Trim();
                }

                // packed-refs lines are "<commit> <ref>"; comment and peeled-tag lines start with '#' and '^'.
                string packed = Path.Combine(gitDirectory, "packed-refs");
                string? line = File.Exists(packed)
                    ? File.ReadLines(packed).FirstOrDefault(l => l.EndsWith(" " + reference, StringComparison.Ordinal))
                    : null;
                if (line != null) {
                    detail = "";
                    return line.Substring(0, line.IndexOf(' '));
                }

                detail = $"HEAD names {reference}, which is neither a loose nor a packed ref";
                return null;
            } catch (IOException e) {
                detail = $"{e.GetType().Name}: {e.Message}";
                return null;
            } catch (UnauthorizedAccessException e) {
                detail = $"{e.GetType().Name}: {e.Message}";
                return null;
            }
        }

        /// <summary>
        ///     Lists local modifications of tracked files in <c>refs/Gymnasium</c>. An edited oracle would silently change
        ///     what "Gymnasium's answer" means, so the session treats any modification as stack drift.
        /// </summary>
        /// <param name="changes">
        ///     The first lines of <c>git status --porcelain</c> when the checkout is modified; empty when it is clean; the
        ///     reason when it could not be checked.
        /// </param>
        /// <returns>
        ///     <see langword="true"/> when git ran and reported the state (clean or not); <see langword="false"/> when git is
        ///     unavailable, in which case the session reports the cleanliness as unverified instead of failing the run.
        /// </returns>
        /// <remarks>
        ///     Untracked files are ignored (<c>--untracked-files=no</c>): <c>__pycache__</c> directories and the like are
        ///     never imported as Gymnasium source. The session also sets <c>sys.dont_write_bytecode</c>, so the suite itself
        ///     never creates them.
        /// </remarks>
        public bool TryListGymnasiumChanges(out string changes) {
            var start = new ProcessStartInfo("git") {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            foreach (string argument in new[] { "-C", GymnasiumRoot, "status", "--porcelain", "--untracked-files=no" })
                start.ArgumentList.Add(argument);

            try {
                using var process = Process.Start(start);
                if (process is null) {
                    changes = "git did not start";
                    return false;
                }

                // Both pipes are drained before waiting: a full stderr pipe would otherwise block git forever.
                var stderr = process.StandardError.ReadToEndAsync();
                string stdout = process.StandardOutput.ReadToEnd();
                if (!process.WaitForExit(30_000)) {
                    process.Kill(entireProcessTree: true);
                    changes = "git status timed out";
                    return false;
                }

                if (process.ExitCode != 0) {
                    changes = $"git status failed ({process.ExitCode}): {stderr.Result.Trim()}";
                    return false;
                }

                string[] lines = stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                changes = string.Join("; ", lines.Take(5)) + (lines.Length > 5 ? $"; and {lines.Length - 5} more" : "");
                return true;
            } catch (System.ComponentModel.Win32Exception e) {
                changes = $"git is not available: {e.Message}";
                return false;
            }
        }
    }
}
