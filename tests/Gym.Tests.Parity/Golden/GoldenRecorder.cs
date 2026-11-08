using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Gym.Tests.Parity.Session;

namespace Gym.Tests.Parity.Golden {
    /// <summary>
    ///     Collects what a live <see cref="GymPair"/> verified (DR-04's record mode) and saves it as the case's golden. Attach
    ///     it through <see cref="GymPair.Recorder"/>; the pair hands it Gymnasium's side of every call and frame after the
    ///     two sides matched, so a golden only ever holds verified data.
    /// </summary>
    /// <remarks>
    ///     Nothing is written until <see cref="Save"/>, which the cell calls only after the whole trajectory passed: a run
    ///     that diverges halfway leaves the committed golden as it was.
    /// </remarks>
    public sealed class GoldenRecorder {
        /// <summary>The case being recorded.</summary>
        private readonly GoldenCase _case;

        /// <summary>The verified calls and frames, in order.</summary>
        private readonly List<GoldenEntry> _entries = new();

        /// <summary>The frames to write, by their path relative to the suite's directory.</summary>
        private readonly List<(string Relative, Frame Frame)> _frames = new();

        /// <summary>Starts a recording.</summary>
        /// <param name="case">The case the pair runs.</param>
        public GoldenRecorder(GoldenCase @case) {
            _case = @case;
        }

        /// <summary>Records one verified call.</summary>
        /// <param name="call">The call as reports name it.</param>
        /// <param name="step">The trajectory position.</param>
        /// <param name="action">The action Gymnasium received; <see langword="null"/> for a reset.</param>
        /// <param name="result">Gymnasium's result, equal to Gym.NET's.</param>
        /// <param name="state">The state attribute the pair compared after the call, or <see langword="null"/>.</param>
        /// <param name="rng">The generator state the pair compared after the call, or <see langword="null"/>.</param>
        public void Call(string call, int step, ValueFacts? action, CallResult result, ValueFacts? state, ValueFacts? rng) {
            _entries.Add(new GoldenEntry.Call(call, step, action, result, state, rng));
        }

        /// <summary>Records one frame both sides drew identically.</summary>
        /// <param name="label">Where in the trajectory, for example <c>after step 50</c>.</param>
        /// <param name="step">The trajectory position.</param>
        /// <param name="frame">Gymnasium's frame, equal to Gym.NET's.</param>
        public void Frame(string label, int step, Frame frame) {
            // Numbered by order, not by label, so the names stay stable and sort the way the trajectory runs.
            string relative = $"{_case.Name}/frame-{_frames.Count.ToString("D3", CultureInfo.InvariantCulture)}.png";
            _frames.Add((relative, frame));
            _entries.Add(new GoldenEntry.Frame(label, step, relative));
        }

        /// <summary>
        ///     Writes the golden: the trajectory, its frames, and the removal of frames an earlier recording left that this one
        ///     no longer has. Unchanged files are left untouched.
        /// </summary>
        /// <returns>The files written or deleted, for the test's output; empty when the committed golden was already current.</returns>
        /// <exception cref="InvalidOperationException">
        ///     Thrown when the session is not up, or the checked-out Gymnasium commit cannot be read: the header records both,
        ///     and a golden without them could not be checked for staleness.
        /// </exception>
        public IReadOnlyList<string> Save() {
            if (GymPythonSession.Stack is null || GymPythonSession.Repository is null || GymPythonSession.Manifest is null)
                throw new InvalidOperationException("A golden is recorded from a live run: the Python session must be up.");
            string commit = GymPythonSession.Repository.ReadGymnasiumHead(out string detail)
                ?? throw new InvalidOperationException($"The checked-out Gymnasium commit cannot be read ({detail}).");

            // Only the manifest's keys: the report's other rows are machine paths (the Python library, numpy's origin) that
            // would make the same recording differ between machines.
            StackManifest manifest = GymPythonSession.Manifest;
            var stack = GymPythonSession.Stack.Items.Where(item => manifest.Stack.ContainsKey(item.Key))
                .Select(item => new KeyValuePair<string, string>(item.Key, item.Actual));
            var header = GoldenHeader.For(_case, commit, GymPythonSession.Platform, stack);
            var changed = new List<string>();
            if (GoldenStore.SaveTrajectory(_case, new GoldenTrajectory(header, _entries.ToList())))
                changed.Add(GoldenStore.TrajectoryPath(_case));

            var kept = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach ((string relative, Frame frame) in _frames) {
                string path = GoldenStore.FramePath(_case, relative);
                kept.Add(Path.GetFullPath(path));
                if (GoldenStore.SaveFrame(path, frame))
                    changed.Add(path);
            }

            // A re-recording with fewer frames must not leave the old extra PNGs behind as orphans nobody replays.
            string directory = GoldenStore.FramesDirectory(_case);
            if (Directory.Exists(directory)) {
                foreach (string stale in Directory.EnumerateFiles(directory, "frame-*.png").Where(f => !kept.Contains(Path.GetFullPath(f))).ToList()) {
                    File.Delete(stale);
                    changed.Add($"deleted {stale}");
                }
            }

            return changed;
        }
    }
}
