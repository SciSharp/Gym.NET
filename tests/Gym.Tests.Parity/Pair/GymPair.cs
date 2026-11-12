using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Golden;
using Gym.Tests.Parity.Oracle;

namespace Gym.Tests.Parity.Pair {
    /// <summary>
    ///     Drives one live Gymnasium env and its .NET counterpart in lockstep and verifies every call bit for bit (plan §5).
    ///     The first divergence throws a <see cref="ParityException"/> whose report names the call, every differing field,
    ///     the actions that led there and both sides' state before and after.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Both sides are reset with the same seed and fed the same actions (<see cref="ParityAction"/> carries identical bits
    ///     to both). Per call the comparison order is fixed: <c>raised</c>, <c>observation</c>, <c>reward</c> (float64 bits),
    ///     <c>terminated</c>, <c>truncated</c>, <c>info</c> (keys in order, values exactly), then the optional state attribute
    ///     (float64, exactly), then <c>rng</c>, the env generator's full PCG64 state, which catches a missing or extra draw at
    ///     the call it happens even when no value differs yet.
    ///     </para>
    ///     <para>
    ///     Owns both envs and disposes them. Every Python call takes the GIL inside the Python half; the .NET half runs
    ///     outside it.
    ///     </para>
    /// </remarks>
    public sealed class GymPair : IDisposable {
        /// <summary>The actions of the current trajectory, rendered, for reports.</summary>
        private readonly List<string> _actions = new();

        /// <summary>Qualifications reported with every divergence (adapter gaps, fields one side cannot expose).</summary>
        private readonly List<string> _notes = new();

        /// <summary>The state attribute compared after every call and dumped in reports, or <see langword="null"/>.</summary>
        private readonly string? _stateAttribute;

        /// <summary>The last described state of each side, the "before" of the next call's report.</summary>
        private ValueFacts? _pythonState, _dotnetState;

        /// <summary>
        ///     Whether this pair has reset both sides with a seed. Until then both generators are entropy-seeded (and reading
        ///     Gymnasium's <c>np_random</c> would even create one), so their states are unrelated and are not compared, and
        ///     there is no trajectory position to report.
        /// </summary>
        private bool _seeded;

        /// <summary>Pairs two envs; the pair owns and disposes both.</summary>
        /// <param name="python">The live Gymnasium env.</param>
        /// <param name="dotnet">The .NET env.</param>
        /// <param name="stateAttribute">
        ///     An attribute both sides expose (<c>state</c> for the classic-control envs), compared exactly after every call as
        ///     float64 values (a float32 array or a tuple of numbers on Gymnasium's side is read as its float64 array) and
        ///     dumped before and after a divergence; <see langword="null"/> when there is none.
        /// </param>
        public GymPair(PyGymEnv python, IDotNetEnv dotnet, string? stateAttribute = null) {
            Python = python;
            DotNet = dotnet;
            _stateAttribute = stateAttribute;
            foreach (string adaptation in dotnet.Adaptations)
                _notes.Add($"{dotnet.Name}: {adaptation}");
        }

        /// <summary>The Gymnasium half.</summary>
        public PyGymEnv Python { get; }

        /// <summary>The .NET half.</summary>
        public IDotNetEnv DotNet { get; }

        /// <summary>What is compared, the report's case line.</summary>
        public string Case => $"{Python.Description} vs {DotNet.Name}";

        /// <summary>Steps taken since the last reset.</summary>
        public int StepsTaken { get; private set; }

        /// <summary>Whether the last verified step terminated or truncated the episode (on both sides, since they matched).</summary>
        public bool Ended { get; private set; }

        /// <summary>Qualifications of this comparison, printed with any divergence and in <see cref="PairRun"/>.</summary>
        public IReadOnlyList<string> Notes => _notes;

        /// <summary>
        ///     Receives every call and frame this pair has verified, so a golden is always a recording of a live cell that
        ///     passed (DR-04); <see langword="null"/> (the default) records nothing.
        /// </summary>
        /// <remarks>
        ///     The recorder is handed Gymnasium's side only after the two sides matched, so what it keeps is also what Gym.NET
        ///     produced. A call that diverges throws before it reaches the recorder, and the test then fails without saving.
        /// </remarks>
        public GoldenRecorder? Recorder { get; set; }

        /// <summary>
        ///     Whether frames are measured instead of required equal (DR-71, the MuJoCo cases): <see cref="CompareFrame"/>
        ///     then notes how Gym.NET's frame differs from Gymnasium's and hands the recorder Gym.NET's own frame, which the
        ///     Python-free replay requires bit for bit. The default, <see langword="false"/>, keeps frames byte-exact.
        /// </summary>
        /// <remarks>
        ///     Gymnasium's MuJoCo frames carry its GPU's own rounding (DR-67), so no golden of them could be replayed against
        ///     Gym.NET; how far Gym.NET's frames may drift from them is the frame cells' gate
        ///     (<c>MujocoFrameLiveParityTests</c>), not this pair's.
        /// </remarks>
        public bool MeasureFrames { get; set; }

        /// <summary>The trajectory position for a report: the step count after a seeded reset, none before one.</summary>
        private int? Position => _seeded ? StepsTaken : null;

        /// <summary>Resets both sides with <paramref name="seed"/> and verifies the results.</summary>
        /// <param name="seed">The seed of both <c>reset(seed=...)</c> calls.</param>
        /// <returns>Gymnasium's result, which equals Gym.NET's.</returns>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        public CallResult Reset(long seed) {
            StepsTaken = 0;
            Ended = false;
            _seeded = true;
            _actions.Clear();
            CallResult python = Python.Reset(seed);
            CallResult dotnet = DotNet.Reset(seed);
            return Verify($"reset(seed={seed.ToString(CultureInfo.InvariantCulture)})", 0, python, dotnet, action: null);
        }

        /// <summary>Steps both sides with the same action and verifies the results.</summary>
        /// <param name="action">The action; it stays owned by the caller.</param>
        /// <returns>Gymnasium's result, which equals Gym.NET's.</returns>
        /// <exception cref="ParityException">Thrown at the first difference, including a difference in raised exceptions.</exception>
        public CallResult Step(ParityAction action) {
            StepsTaken++;
            _actions.Add(action.ToString());
            CallResult python = Python.Step(action);
            CallResult dotnet = DotNet.Step(action.DotNet);
            return Verify($"step #{StepsTaken}: step({action})", StepsTaken, python, dotnet, action.Facts);
        }

        /// <summary>
        ///     Runs a trajectory: a reset with <paramref name="seed"/>, then steps until the episode ends, both sides raise, or
        ///     <paramref name="maxSteps"/> steps are taken, optionally comparing frames along the way.
        /// </summary>
        /// <param name="seed">The reset seed.</param>
        /// <param name="actions">Where actions come from.</param>
        /// <param name="maxSteps">The most steps to take.</param>
        /// <param name="frameEvery">Compare frames every this many steps (and after the reset); 0 never renders.</param>
        /// <returns>What the run covered, for the test's output.</returns>
        /// <exception cref="ParityException">Thrown at the first difference.</exception>
        public PairRun Run(long seed, IActionSource actions, int maxSteps, int frameEvery = 0) {
            Reset(seed);
            actions.Start(this);
            if (frameEvery > 0)
                CompareFrame("after reset");
            bool raised = false;
            while (StepsTaken < maxSteps && !Ended && !raised) {
                using ParityAction action = actions.Next(this);
                // Identical exceptions on both sides are parity too, but the trajectory cannot go on after them.
                raised = Step(action).Raised != null;
                if (frameEvery > 0 && StepsTaken % frameEvery == 0)
                    CompareFrame($"after step {StepsTaken}");
            }

            return new PairRun(Case, seed, actions.Description, StepsTaken, Ended, raised, Notes.ToList());
        }

        /// <summary>
        ///     Writes the same float64 values into a state attribute on both sides (plan §5, state injection), which isolates one
        ///     step of dynamics from reset and the generator.
        /// </summary>
        /// <param name="attribute">The attribute, for example <c>state</c>.</param>
        /// <param name="values">The values, bit for bit the same on both sides.</param>
        /// <exception cref="NotSupportedException">Thrown when the .NET side has no such attribute.</exception>
        public void InjectState(string attribute, params double[] values) {
            Python.SetAttribute(attribute, values);
            DotNet.SetAttribute(attribute, values);
            _actions.Add($"<{attribute} := [{string.Join(", ", values.Select(v => ValueFacts.FormatNumber(v)))}]>");
            ReadStates();
        }

        /// <summary>
        ///     Renders both sides and compares the frames pixel for pixel (plan §6), writing <c>python.png</c>,
        ///     <c>dotnet.png</c> and <c>diff.png</c> when they differ; with <see cref="MeasureFrames"/>, measures them and
        ///     records Gym.NET's frame instead (frames of different sizes still fail).
        /// </summary>
        /// <param name="label">Where in the trajectory, for the report and the artifacts' directory.</param>
        /// <exception cref="ParityException">Thrown when the frames differ, with field <c>frame</c>.</exception>
        public void CompareFrame(string label) {
            Frame python = Python.Render();
            Frame dotnet = DotNet.Render();
            FrameComparison comparison = FrameDiff.Compare(python, dotnet);
            if (MeasureFrames && comparison.SameSize) {
                // measured, not gated (DR-71): the note reports the difference, the golden keeps Gym.NET's own frame
                _notes.Add($"frame {label} (step {StepsTaken}): {comparison.Describe()}");
                Recorder?.Frame(label, StepsTaken, dotnet);
                return;
            }

            if (comparison.Identical) {
                // Only an identical frame is recorded, so a golden frame is always one both sides drew.
                Recorder?.Frame(label, StepsTaken, python);
                return;
            }
            IReadOnlyList<string> artifacts = FrameDiff.WriteArtifacts(python, dotnet, comparison, $"{label} step {StepsTaken}");
            throw new ParityException(new Divergence {
                Case = Case,
                Call = $"render() {label}",
                Step = Position,
                Fields = new[] { new FieldDivergence("frame", "frame", comparison.Describe()) },
                Actions = _actions.ToList(),
                Notes = _notes.ToList(),
                PythonStateAfter = _pythonState?.Render(),
                DotNetStateAfter = _dotnetState?.Render(),
                Artifacts = artifacts,
            });
        }

        /// <summary>Throws a divergence found outside <see cref="Reset"/>/<see cref="Step"/> (an action source's sample comparison).</summary>
        /// <param name="call">What was compared.</param>
        /// <param name="fields">The differences; not empty.</param>
        /// <exception cref="ParityException">Always.</exception>
        internal void Fail(string call, IReadOnlyList<FieldDivergence> fields) {
            throw new ParityException(new Divergence {
                Case = Case,
                Call = call,
                Step = Position,
                Fields = fields,
                Actions = _actions.ToList(),
                Notes = _notes.ToList(),
                PythonStateAfter = _pythonState?.Render(),
                DotNetStateAfter = _dotnetState?.Render(),
            });
        }

        /// <summary>Disposes both envs; the Python one under the GIL.</summary>
        public void Dispose() {
            try {
                Python.Dispose();
            } finally {
                DotNet.Dispose();
            }
        }

        /// <summary>Compares one call's results in the fixed field order and throws at the first call with any difference.</summary>
        /// <param name="call">The call, for the report.</param>
        /// <param name="step">The trajectory position.</param>
        /// <param name="python">Gymnasium's result.</param>
        /// <param name="dotnet">Gym.NET's result.</param>
        /// <param name="action">The step's action as Gymnasium received it, for the <see cref="Recorder"/>; <see langword="null"/> for a reset.</param>
        /// <returns>Gymnasium's result, when the two match.</returns>
        /// <exception cref="ParityException">Thrown when anything differs.</exception>
        private CallResult Verify(string call, int step, CallResult python, CallResult dotnet, ValueFacts? action) {
            ValueFacts? pythonBefore = _pythonState, dotnetBefore = _dotnetState;
            ReadStates();

            var fields = new List<FieldDivergence>();
            // What this call compared beyond the transition, for the recorder: a golden holds exactly what was verified.
            ValueFacts? comparedState = null, comparedRng = null;
            if (python.Raised != null || dotnet.Raised != null) {
                fields.AddRange(ByteContract.Compare("raised", python.ToOutcome(), dotnet.ToOutcome()));
            } else {
                Transition p = python.Transition!, d = dotnet.Transition!;
                fields.AddRange(ByteContract.Compare("observation", p.Observation, d.Observation));
                fields.AddRange(ByteContract.Compare("reward", p.Reward, d.Reward, FieldPolicy.Float64Value));
                fields.AddRange(ByteContract.Compare("terminated", p.Terminated, d.Terminated));
                fields.AddRange(ByteContract.Compare("truncated", p.Truncated, d.Truncated));
                fields.AddRange(ByteContract.Compare("info", p.Info, d.Info));
                if (_stateAttribute != null) {
                    fields.AddRange(ByteContract.Compare("state", _pythonState, _dotnetState));
                    comparedState = _pythonState;
                }

                fields.AddRange(CompareRng(out comparedRng));
            }

            if (fields.Count > 0) {
                throw new ParityException(new Divergence {
                    Case = Case,
                    Call = call,
                    Step = _seeded ? step : null,
                    Fields = fields,
                    Actions = _actions.ToList(),
                    Notes = _notes.ToList(),
                    PythonStateBefore = pythonBefore?.Render(),
                    DotNetStateBefore = dotnetBefore?.Render(),
                    PythonStateAfter = _pythonState?.Render(),
                    DotNetStateAfter = _dotnetState?.Render(),
                });
            }

            Ended = python.Transition?.Ended ?? false;
            Recorder?.Call(call, step, action, python, comparedState, comparedRng);
            return python;
        }

        /// <summary>
        ///     Compares the generators' states, or records once why the .NET side cannot expose its generator, so a skipped
        ///     check is visible in every report rather than silent.
        /// </summary>
        /// <param name="python">Gymnasium's state when it was compared, for the <see cref="Recorder"/>; <see langword="null"/> when the check was skipped.</param>
        /// <returns>The differences; empty when they match or cannot be compared.</returns>
        private IReadOnlyList<FieldDivergence> CompareRng(out ValueFacts? python) {
            python = null;
            if (!_seeded) {
                const string unseeded = "rng not compared: no seeded reset yet, so both generators are entropy-seeded and unrelated";
                if (!_notes.Contains(unseeded))
                    _notes.Add(unseeded);
                return Array.Empty<FieldDivergence>();
            }

            if (!DotNet.TryGetRngState(out ValueFacts? dotnet, out string reason)) {
                string note = $"rng not compared: {reason}";
                if (!_notes.Contains(note))
                    _notes.Add(note);
                return Array.Empty<FieldDivergence>();
            }

            python = Python.RngState();
            return ByteContract.Compare("rng", python, dotnet);
        }

        /// <summary>
        ///     Reads both sides' state attribute, when one is configured, into the "after" slots: Gymnasium's as its float64
        ///     values (<see cref="PyGymEnv.GetStateValues"/>, since its envs change the state's container between calls),
        ///     Gym.NET's as the float64 array its <c>double[]</c> holds.
        /// </summary>
        private void ReadStates() {
            if (_stateAttribute is null)
                return;
            _pythonState = Python.GetStateValues(_stateAttribute).Value;
            _dotnetState = DotNet.GetAttribute(_stateAttribute);
        }
    }

    /// <summary>What a <see cref="GymPair.Run"/> covered, all of it verified.</summary>
    /// <param name="Case">What was compared.</param>
    /// <param name="Seed">The reset seed.</param>
    /// <param name="Actions">Where the actions came from.</param>
    /// <param name="Steps">Steps taken.</param>
    /// <param name="Ended">Whether the episode ended (terminated or truncated) within the run.</param>
    /// <param name="Raised">Whether it stopped because both sides raised the same exception.</param>
    /// <param name="Notes">The pair's qualifications.</param>
    public sealed record PairRun(string Case, long Seed, string Actions, int Steps, bool Ended, bool Raised, IReadOnlyList<string> Notes) {
        /// <summary>Renders the run for the test's output.</summary>
        /// <returns>One line, plus one per note.</returns>
        public override string ToString() {
            string end = Raised ? "stopped: both raised" : Ended ? "episode ended" : "step limit";
            return $"{Case}, seed {Seed}, {Actions}: {Steps} steps identical ({end})" + string.Concat(Notes.Select(n => Environment.NewLine + "  note: " + n));
        }
    }
}
