using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Tests.Parity.Frames;
using Gym.Tests.Parity.Oracle;
using Gym.Tests.Parity.Pair;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace Gym.Tests.Parity.Golden {
    /// <summary>What a replay covered, all of it identical to the golden.</summary>
    /// <param name="Key">The case.</param>
    /// <param name="Env">The .NET side's name.</param>
    /// <param name="Steps">Steps replayed after the reset.</param>
    /// <param name="Frames">Frames compared.</param>
    /// <param name="Notes">Qualifications, for example that the golden was recorded on another platform.</param>
    public sealed record GoldenReplayRun(string Key, string Env, int Steps, int Frames, IReadOnlyList<string> Notes) {
        /// <summary>Renders the run for the test's output.</summary>
        /// <returns>One line, plus one per note.</returns>
        public override string ToString() =>
            $"golden {Key} vs {Env}: reset + {Steps} steps and {Frames} frames identical" + string.Concat(Notes.Select(n => Environment.NewLine + "  note: " + n));
    }

    /// <summary>
    ///     Replays a golden without Python (DR-04, plan §9's replay mode): drives a .NET env through the recorded trajectory
    ///     and compares every call with what Gymnasium returned when the golden was recorded, in <see cref="GymPair"/>'s field
    ///     order and through the same <see cref="ByteContract"/>, so a replay is as strict as the live cell it came from.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Actions are fed back from the recording (open-loop), except for a <see cref="GoldenActions.Sampled"/> case: its
    ///     replay seeds the .NET action space after the reset and requires each sample to equal the recorded one, so space
    ///     sampling stays checked. Frames are compared pixel for pixel with the recorded PNGs.
    ///     </para>
    ///     <para>
    ///     Divergence reports name Gymnasium's side "python", as live reports do; in a replay it is the recording, which the
    ///     report's notes say.
    ///     </para>
    /// </remarks>
    public static class GoldenAssert {
        /// <summary>
        ///     Replays a case's committed golden against a fresh .NET env built by the case, after checking that the golden
        ///     still records this case at the pinned Gymnasium commit.
        /// </summary>
        /// <param name="case">The case.</param>
        /// <returns>What the replay covered.</returns>
        /// <exception cref="AssertFailedException">
        ///     Thrown when the case has no golden yet, or its golden is stale: it records other parameters than the catalog's,
        ///     or another Gymnasium commit than <c>tools/parity/manifest.json</c> pins. The message says how to re-record.
        /// </exception>
        /// <exception cref="ParityException">Thrown at the first call or frame that differs from the golden.</exception>
        public static GoldenReplayRun Replay(GoldenCase @case) {
            GoldenTrajectory golden = LoadCurrent(@case);
            using IDotNetEnv env = @case.CreateDotNet(@case.RenderMode);
            return Replay(@case, golden, env);
        }

        /// <summary>Reads a case's golden and fails unless it is present and current.</summary>
        /// <param name="case">The case.</param>
        /// <returns>The golden.</returns>
        /// <exception cref="AssertFailedException">Thrown when the golden is missing or stale, with the re-record command.</exception>
        public static GoldenTrajectory LoadCurrent(GoldenCase @case) {
            string record = $"Record it with: {GoldenStore.RecordVariable}=1 dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -f net10.0 " +
                            $"--filter \"FullyQualifiedName~GoldenLiveTests\" (needs the reference stack).";
            GoldenTrajectory golden = GoldenStore.TryLoad(@case)
                ?? throw new AssertFailedException($"{@case.Key} has no golden at {GoldenStore.TrajectoryPath(@case)}. {record}");

            IReadOnlyList<string> differences = golden.Header.DifferencesFrom(@case);
            if (differences.Count > 0)
                Assert.Fail($"The golden of {@case.Key} is stale:{Environment.NewLine}  {string.Join(Environment.NewLine + "  ", differences)}{Environment.NewLine}{record}");

            string pinned = GoldenStore.PinnedGymnasiumCommit();
            if (!string.Equals(golden.Header.GymnasiumCommit, pinned, StringComparison.OrdinalIgnoreCase))
                Assert.Fail($"The golden of {@case.Key} was recorded at Gymnasium {golden.Header.GymnasiumCommit}, but {pinned} is pinned. " +
                            $"A re-pin re-records every golden. {record}");
            return golden;
        }

        /// <summary>Replays a golden against a given env, which the caller owns (the harness self-tests pass a deliberately broken one).</summary>
        /// <param name="case">The case the golden records.</param>
        /// <param name="golden">The golden.</param>
        /// <param name="env">The .NET side, freshly built; it is reset and stepped but not disposed.</param>
        /// <returns>What the replay covered.</returns>
        /// <exception cref="ParityException">Thrown at the first call or frame that differs from the golden.</exception>
        public static GoldenReplayRun Replay(GoldenCase @case, GoldenTrajectory golden, IDotNetEnv env) {
            var actions = new List<string>();
            var notes = new List<string> {
                $"python = the golden recorded from Gymnasium {Short(golden.Header.GymnasiumCommit)} on {golden.Header.Platform}",
            };
            string caseLine = $"golden {@case.Key} vs {env.Name}";
            int steps = 0, frames = 0;

            foreach (GoldenEntry entry in golden.Entries) {
                switch (entry) {
                    case GoldenEntry.Call { IsReset: true } reset:
                        Verify(caseLine, reset, env.Reset(golden.Header.Seed), env, @case, actions, notes);
                        // The live sampled source seeds both action spaces right after the reset; the replay does the same.
                        if (@case.Actions is GoldenActions.Sampled sampled)
                            env.SeedActionSpace(sampled.Seed);
                        break;
                    case GoldenEntry.Call step:
                        Verify(caseLine, step, StepWith(caseLine, step, env, @case, actions, notes), env, @case, actions, notes);
                        steps++;
                        break;
                    case GoldenEntry.Frame frame:
                        CompareFrame(caseLine, frame, env, @case, actions, notes);
                        frames++;
                        break;
                }
            }

            return new GoldenReplayRun(@case.Key, env.Name, steps, frames, notes);
        }

        /// <summary>Takes one recorded step: picks the action (recorded, or sampled and checked), steps, and releases the action.</summary>
        /// <param name="caseLine">The report's case line.</param>
        /// <param name="step">The recorded call.</param>
        /// <param name="env">The .NET side.</param>
        /// <param name="case">The case.</param>
        /// <param name="actions">The action history, extended here.</param>
        /// <param name="notes">The report's notes.</param>
        /// <returns>What the .NET step returned or raised.</returns>
        /// <exception cref="ParityException">Thrown when a sampled action differs from the recorded one.</exception>
        /// <exception cref="InvalidOperationException">Thrown when a recorded step has no action, which only a hand-edited golden can hold.</exception>
        private static CallResult StepWith(string caseLine, GoldenEntry.Call step, IDotNetEnv env, GoldenCase @case, List<string> actions, List<string> notes) {
            ValueFacts recorded = step.Action ?? throw new InvalidOperationException($"{step.Label} has no recorded action.");
            actions.Add(recorded.Render());
            object action;
            if (@case.Actions.ReplaySamples) {
                action = env.SampleAction();
                var differences = ByteContract.Compare("action", recorded, ValueFacts.Of(action));
                if (differences.Count > 0) {
                    (action as IDisposable)?.Dispose();
                    throw new ParityException(new Divergence {
                        Case = caseLine,
                        Call = $"action_space.sample() for step #{step.Step}",
                        Step = step.Step - 1,
                        Fields = differences,
                        Actions = actions.ToList(),
                        Notes = notes.ToList(),
                    });
                }
            } else {
                action = ToDotNetAction(recorded);
            }

            try {
                return env.Step(action);
            } finally {
                // Both an action this replay built and a sample are owned here; the env only read it during the step.
                (action as IDisposable)?.Dispose();
            }
        }

        /// <summary>Compares one call's results with the recording in <see cref="GymPair"/>'s field order.</summary>
        /// <param name="caseLine">The report's case line.</param>
        /// <param name="recorded">The recorded call.</param>
        /// <param name="dotnet">What the .NET side returned or raised.</param>
        /// <param name="env">The .NET side, for the state attribute and generator.</param>
        /// <param name="case">The case.</param>
        /// <param name="actions">The action history.</param>
        /// <param name="notes">The report's notes.</param>
        /// <exception cref="ParityException">Thrown when anything differs.</exception>
        private static void Verify(string caseLine, GoldenEntry.Call recorded, CallResult dotnet, IDotNetEnv env, GoldenCase @case, List<string> actions, List<string> notes) {
            var fields = new List<FieldDivergence>();
            ValueFacts? dotnetState = null;
            if (recorded.Result.Raised != null || dotnet.Raised != null) {
                fields.AddRange(ByteContract.Compare("raised", recorded.Result.ToOutcome(), dotnet.ToOutcome()));
            } else {
                Transition p = recorded.Result.Transition!, d = dotnet.Transition!;
                fields.AddRange(ByteContract.Compare("observation", p.Observation, d.Observation));
                fields.AddRange(ByteContract.Compare("reward", p.Reward, d.Reward, FieldPolicy.Float64Value));
                fields.AddRange(ByteContract.Compare("terminated", p.Terminated, d.Terminated));
                fields.AddRange(ByteContract.Compare("truncated", p.Truncated, d.Truncated));
                fields.AddRange(ByteContract.Compare("info", p.Info, d.Info));
                // State and generator are compared exactly when the live cell compared them, no more and no less.
                if (recorded.State != null && @case.StateAttribute != null) {
                    dotnetState = env.GetAttribute(@case.StateAttribute);
                    fields.AddRange(ByteContract.Compare("state", recorded.State, dotnetState));
                }

                if (recorded.Rng != null) {
                    if (env.TryGetRngState(out ValueFacts? rng, out string reason))
                        fields.AddRange(ByteContract.Compare("rng", recorded.Rng, rng));
                    else
                        fields.Add(new FieldDivergence("rng", "rng", $"compared when the golden was recorded, but {reason}"));
                }
            }

            if (fields.Count == 0)
                return;
            throw new ParityException(new Divergence {
                Case = caseLine,
                Call = recorded.Label,
                Step = recorded.Step,
                Fields = fields,
                Actions = actions.ToList(),
                Notes = notes.ToList(),
                PythonStateAfter = recorded.State?.Render(),
                DotNetStateAfter = dotnetState?.Render(),
            });
        }

        /// <summary>Renders the .NET side and compares the frame with the recorded PNG, writing the artifacts when they differ.</summary>
        /// <param name="caseLine">The report's case line.</param>
        /// <param name="frame">The recorded frame entry.</param>
        /// <param name="env">The .NET side.</param>
        /// <param name="case">The case, which locates the PNG.</param>
        /// <param name="actions">The action history.</param>
        /// <param name="notes">The report's notes.</param>
        /// <exception cref="ParityException">Thrown when the frames differ, with field <c>frame</c>.</exception>
        private static void CompareFrame(string caseLine, GoldenEntry.Frame frame, IDotNetEnv env, GoldenCase @case, List<string> actions, List<string> notes) {
            Frame expected = GoldenStore.LoadFrame(GoldenStore.FramePath(@case, frame.File));
            Frame actual = env.Render();
            FrameComparison comparison = FrameDiff.Compare(expected, actual);
            if (comparison.Identical)
                return;
            IReadOnlyList<string> artifacts = FrameDiff.WriteArtifacts(expected, actual, comparison, $"golden {@case.Name} {frame.Label}");
            throw new ParityException(new Divergence {
                Case = caseLine,
                Call = $"render() {frame.Label}",
                Step = frame.Step,
                Fields = new[] { new FieldDivergence("frame", "frame", comparison.Describe()) },
                Actions = actions.ToList(),
                Notes = notes.ToList(),
                Artifacts = artifacts,
            });
        }

        /// <summary>
        ///     The .NET form of a recorded action, as the live harness hands it to Gym.NET: a boxed <see langword="long"/> for
        ///     an integer, an <see cref="NDArray"/> with the recorded dtype, shape and bytes for an array.
        /// </summary>
        /// <param name="facts">The recorded action.</param>
        /// <returns>The action; an array is owned by the caller.</returns>
        /// <exception cref="NotSupportedException">Thrown for an action that is neither an integer nor an array, which the live harness cannot produce either.</exception>
        internal static object ToDotNetAction(ValueFacts facts) {
            switch (facts.Kind) {
                case FactKind.Int:
                    return (long)facts.Integer;
                case FactKind.Array: {
                    // frombuffer views and pins its buffer, so it gets a copy of the recorded bytes, and the result is a
                    // compact copy that owns its memory: the env may keep the action after the step.
                    using NDArray flat = np.frombuffer((byte[])facts.Bytes.Clone(), np.dtype(facts.DType));
                    using NDArray shaped = flat.reshape(facts.Shape);
                    return shaped.copy();
                }
                default:
                    throw new NotSupportedException($"A recorded action of kind {facts.Kind} ({facts.Render()}) cannot be replayed; only integer and array actions are.");
            }
        }

        /// <summary>The first eight characters of a commit id, for notes.</summary>
        /// <param name="commit">The commit id.</param>
        /// <returns>Its abbreviation.</returns>
        private static string Short(string commit) => commit.Length > 8 ? commit.Substring(0, 8) : commit;
    }
}
