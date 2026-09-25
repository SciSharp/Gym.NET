using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Gym.Tests.Parity.Session {
    /// <summary>
    ///     The Python the session embeds: the shared library pythonnet loads and the module path the embedded interpreter
    ///     starts with.
    /// </summary>
    /// <param name="LibraryPath">The <c>python3XY.dll</c> or <c>libpython3.XY.so</c> to load (<c>Runtime.PythonDLL</c>).</param>
    /// <param name="Interpreter">The executable that was probed, or <see langword="null"/> when only <c>PYTHONNET_PYDLL</c> was given.</param>
    /// <param name="SysPath">
    ///     The probed interpreter's <c>sys.path</c>, which the embedded one adopts so that it resolves modules exactly like
    ///     that executable (a venv's site-packages included); <see langword="null"/> keeps the embedded default.
    /// </param>
    /// <param name="Version">The probed <c>major.minor.micro</c>, or "unknown" when nothing was probed.</param>
    /// <param name="Source">How it was found, for reports: "venv ...", an executable, or "PYTHONNET_PYDLL".</param>
    public sealed record PythonInstallation(string LibraryPath, string? Interpreter, IReadOnlyList<string>? SysPath, string Version, string Source);

    /// <summary>
    ///     Finds a Python with NumPy that pythonnet can embed, following NumSharp's discovery
    ///     (<c>refs/NumSharp/test/NumSharp.Tests.Interop/PythonSession.cs</c>) and extending it with the parity venv of
    ///     docs/plans/GYM_PARITY_INTEROP_TESTING.md §3–§4.
    /// </summary>
    /// <remarks>
    ///     Order: a venv (<c>GYMNET_PARITY_VENV</c>, else <c>tools/parity/.venv</c> when present) is the only candidate
    ///     when there is one, so a configured venv that is broken fails loudly instead of silently falling back to another
    ///     Python. Without one: <c>python</c>, <c>python3</c>, <c>~/.claude/python</c>, then the Windows per-user installs.
    ///     <c>PYTHONNET_PYDLL</c> overrides only the library; the interpreter is still probed for <c>sys.path</c> when possible.
    /// </remarks>
    internal static class PythonDiscovery {
        /// <summary>The oldest Python pythonnet 3.0.5 drives.</summary>
        private static readonly Version MinimumPython = new(3, 7);

        /// <summary>The newest Python pythonnet 3.0.5 drives; 3.14 needs pythonnet 3.1.0.</summary>
        private static readonly Version MaximumPython = new(3, 13);

        /// <summary>
        ///     Runs in the candidate interpreter. NumPy is imported to reject interpreters without it; the rest is what the
        ///     session needs to embed the same environment: where the base installation lives (a venv has no DLL of its own),
        ///     the version, the unix library hints, and <c>sys.path</c>.
        /// </summary>
        private const string ProbeScript =
            "import json, sys, sysconfig, numpy\n" +
            "print(json.dumps({'base_prefix': sys.base_prefix, 'version': list(sys.version_info[:3]), " +
            "'instsoname': sysconfig.get_config_var('INSTSONAME') or '', 'libdir': sysconfig.get_config_var('LIBDIR') or '', " +
            "'path': sys.path}))";

        /// <summary>Finds the Python to embed.</summary>
        /// <param name="layout">The checkout, whose <c>tools/parity/.venv</c> is used when present.</param>
        /// <param name="reason">Every candidate tried and why it was rejected, when <see langword="null"/> is returned; empty otherwise.</param>
        /// <returns>The installation to embed, or <see langword="null"/> when no candidate qualifies.</returns>
        public static PythonInstallation? Discover(RepositoryLayout layout, out string reason) {
            var attempts = new List<string>();
            string? library = Environment.GetEnvironmentVariable("PYTHONNET_PYDLL");
            if (!string.IsNullOrEmpty(library) && Path.IsPathRooted(library) && !File.Exists(library)) {
                reason = $"PYTHONNET_PYDLL='{library}' does not exist";
                return null;
            }

            string? venv = Environment.GetEnvironmentVariable("GYMNET_PARITY_VENV");
            if (string.IsNullOrEmpty(venv) && Directory.Exists(layout.DefaultVenv))
                venv = layout.DefaultVenv;

            IEnumerable<string> candidates = venv != null ? new[] { VenvInterpreter(venv) } : DefaultCandidates();
            foreach (string candidate in candidates) {
                if (!Probe(candidate, out Probed? probed, out string detail)) {
                    attempts.Add($"{candidate}: {detail}");
                    continue;
                }

                reason = "";
                string source = venv != null ? $"venv {venv} ({candidate})" : candidate;
                // PYTHONNET_PYDLL replaces the library only; the probed sys.path still describes the environment to embed.
                return new PythonInstallation(string.IsNullOrEmpty(library) ? probed.Library : library, candidate, probed.SysPath, probed.Version, source);
            }

            if (!string.IsNullOrEmpty(library) && venv == null) {
                // No interpreter could be probed, but the library was named explicitly: embed it with its default sys.path
                // and let the manifest check report whatever that environment lacks.
                reason = "";
                return new PythonInstallation(library, null, null, "unknown", "PYTHONNET_PYDLL");
            }

            reason = attempts.Count == 0 ? "no Python interpreter candidates" : string.Join(" | ", attempts);
            return null;
        }

        /// <summary>The interpreter inside a venv: <c>Scripts\python.exe</c> on Windows, <c>bin/python</c> elsewhere.</summary>
        /// <param name="venv">The venv's root directory.</param>
        /// <returns>The path of its interpreter, whether or not it exists; the probe reports a missing one.</returns>
        private static string VenvInterpreter(string venv) {
            return OperatingSystem.IsWindows() ? Path.Combine(venv, "Scripts", "python.exe") : Path.Combine(venv, "bin", "python");
        }

        /// <summary>The interpreters tried when no venv is configured, in NumSharp's order.</summary>
        /// <returns>Candidate executables; names without a directory are resolved through <c>PATH</c>.</returns>
        private static IEnumerable<string> DefaultCandidates() {
            yield return "python";
            yield return "python3";

            string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            string claudePython = Path.Combine(home, ".claude", "python", OperatingSystem.IsWindows() ? "python.exe" : "python3");
            if (File.Exists(claudePython))
                yield return claudePython;

            if (!OperatingSystem.IsWindows())
                yield break;

            string programs = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Programs", "Python");
            if (!Directory.Exists(programs))
                yield break;

            foreach (string directory in Directory.GetDirectories(programs, "Python3*").OrderByDescending(d => d, StringComparer.OrdinalIgnoreCase)) {
                string executable = Path.Combine(directory, "python.exe");
                if (File.Exists(executable))
                    yield return executable;
            }
        }

        /// <summary>What a successful probe learned.</summary>
        /// <param name="Library">The shared library to embed.</param>
        /// <param name="SysPath">The interpreter's <c>sys.path</c>, empty entries (the probe's working directory) removed.</param>
        /// <param name="Version">The interpreter's <c>major.minor.micro</c>.</param>
        private sealed record Probed(string Library, IReadOnlyList<string> SysPath, string Version);

        /// <summary>
        ///     Runs <paramref name="executable"/> once to prove NumPy imports, to find the shared library pythonnet must load,
        ///     and to reject versions pythonnet 3.0.5 cannot drive.
        /// </summary>
        /// <param name="executable">The interpreter to run.</param>
        /// <param name="probed">What was learned, when <see langword="true"/> is returned.</param>
        /// <param name="detail">Why the candidate was rejected, when <see langword="false"/> is returned.</param>
        /// <returns><see langword="true"/> when the candidate can be embedded.</returns>
        private static bool Probe(string executable, [NotNullWhen(true)] out Probed? probed, out string detail) {
            probed = null;
            var start = new ProcessStartInfo(executable) {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            // ArgumentList quotes each element for the platform, so the script's quotes survive intact.
            start.ArgumentList.Add("-c");
            start.ArgumentList.Add(ProbeScript);

            try {
                using var process = Process.Start(start);
                if (process is null) {
                    detail = "did not start";
                    return false;
                }

                // Drain both pipes before waiting, or a chatty stderr could fill its pipe and block the probe.
                var stderr = process.StandardError.ReadToEndAsync();
                string stdout = process.StandardOutput.ReadToEnd();
                if (!process.WaitForExit(30_000)) {
                    process.Kill(entireProcessTree: true);
                    detail = "probe timed out";
                    return false;
                }

                if (process.ExitCode != 0) {
                    detail = $"probe failed: {LastLine(stderr.Result)}";
                    return false;
                }

                using var json = JsonDocument.Parse(stdout);
                JsonElement root = json.RootElement;
                int[] parts = root.GetProperty("version").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                var version = new Version(parts[0], parts[1]);
                if (version < MinimumPython || version > MaximumPython) {
                    detail = $"Python {parts[0]}.{parts[1]} is outside pythonnet 3.0.5's supported range ({MinimumPython}-{MaximumPython})";
                    return false;
                }

                string basePrefix = root.GetProperty("base_prefix").GetString() ?? "";
                string? library = OperatingSystem.IsWindows()
                    ? Path.Combine(basePrefix, $"python{parts[0]}{parts[1]}.dll")
                    : FindUnixLibrary(basePrefix, root.GetProperty("instsoname").GetString() ?? "", root.GetProperty("libdir").GetString() ?? "", parts[0], parts[1]);
                if (library is null || !File.Exists(library)) {
                    detail = $"no shared library for base prefix '{basePrefix}' (pythonnet needs a shared build)";
                    return false;
                }

                string[] sysPath = root.GetProperty("path").EnumerateArray()
                    .Select(e => e.GetString() ?? "")
                    .Where(p => p.Length > 0)
                    .ToArray();
                probed = new Probed(library, sysPath, $"{parts[0]}.{parts[1]}.{parts[2]}");
                detail = "";
                return true;
            } catch (System.ComponentModel.Win32Exception e) {
                detail = e.Message;
                return false;
            } catch (JsonException e) {
                detail = $"unreadable probe output: {e.Message}";
                return false;
            }
        }

        /// <summary>
        ///     Finds libpython on Linux and macOS: <c>LIBDIR/INSTSONAME</c> for ordinary shared builds, the file name of
        ///     <c>INSTSONAME</c> under the base prefix for macOS framework builds (where <c>INSTSONAME</c> is a relative
        ///     framework path, the trap NumSharp's discovery documents), then the plain <c>libpythonX.Y</c> spellings.
        /// </summary>
        /// <param name="prefix">The base prefix.</param>
        /// <param name="soname">The <c>INSTSONAME</c> config variable, possibly empty.</param>
        /// <param name="libdir">The <c>LIBDIR</c> config variable, possibly empty.</param>
        /// <param name="major">The major version.</param>
        /// <param name="minor">The minor version.</param>
        /// <returns>The first candidate that exists, or <see langword="null"/>.</returns>
        private static string? FindUnixLibrary(string prefix, string soname, string libdir, int major, int minor) {
            string extension = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? "dylib" : "so";
            var candidates = new List<string>();
            if (soname.Length > 0 && libdir.Length > 0)
                candidates.Add(Path.Combine(libdir, soname));
            if (soname.Length > 0 && prefix.Length > 0)
                candidates.Add(Path.Combine(prefix, Path.GetFileName(soname)));
            if (libdir.Length > 0)
                candidates.Add(Path.Combine(libdir, $"libpython{major}.{minor}.{extension}"));
            if (prefix.Length > 0)
                candidates.Add(Path.Combine(prefix, "lib", $"libpython{major}.{minor}.{extension}"));
            return candidates.FirstOrDefault(File.Exists);
        }

        /// <summary>The last non-empty line of a probe's stderr: the exception line of a Python traceback.</summary>
        /// <param name="text">The captured stderr.</param>
        /// <returns>That line, or "no stderr".</returns>
        private static string LastLine(string text) {
            string? last = text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).LastOrDefault();
            return last ?? "no stderr";
        }
    }
}
