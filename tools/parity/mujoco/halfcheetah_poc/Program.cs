using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp;
using Gym.MuJoCo;

namespace MujocoPoc;

/// <summary>
///     Checks the core against MuJoCo 4.0.0 and Gymnasium bit for bit, for the 21 MuJoCo ids that run on MuJoCo 4 (the
///     rules in <see cref="EnvRules"/>): the reset noise and the sampled actions (NumSharp's NumPy generator), every
///     mjData array oracle.py recorded after each physics step and each env step, and the env's observation, reward,
///     flags and info after each env step.
/// </summary>
/// <remarks>
///     Two passes over each recorded episode. Closed loop: the core runs on its own from its own reset, as an env would, so
///     any difference compounds. Resync: before each physics step the core is put back on MuJoCo's recorded state, so a
///     difference shows in the step that makes it. Exit code 0 when every episode is exact in both passes and the env layer.
///     An id without rules (a MuJoCo env outside Gymnasium's eleven) is checked physics only: the closed loop starts from
///     Gymnasium's recorded reset and replays its actions, and its env layer is not checked (its report says so). A model
///     outside the core's scope is reported by the feature <see cref="MjModel.Load"/> names, and counts as failed.
/// </remarks>
internal static class Program {
    /// <summary>The names of the per-step records, in oracle.py's order after the state block.</summary>
    private static readonly string[] StateNames =
        ["qpos", "qvel", "qacc", "qacc_warmstart", "qacc_smooth", "qfrc_constraint", "qfrc_passive", "qfrc_bias", "qfrc_actuator", "actuator_force"];

    private static readonly string[] EfcNames =
        ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss", "efc_diagA", "efc_KBIP", "efc_D", "efc_R", "efc_vel", "efc_aref", "efc_b", "efc_force", "efc_state"];

    private static readonly string[] DebugNames =
        ["xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "subtree_com", "cinert", "cdof", "crb", "M", "qLD", "qLDiagInv", "cvel", "cdof_dot"];

    private static readonly string[] PostNames = ["cfrc_ext", "ten_length", "ten_velocity", "site_xpos", "site_xmat", "xpos"];

    /// <summary>
    ///     Checks every recorded trace given and prints one report per trace, then the totals, then the core's speed on
    ///     the first trace of each env.
    /// </summary>
    /// <param name="args">
    ///     Folders, each holding one oracle.py recording (model.json, run.json, trace.bin, env.bin); default: the current
    ///     folder. <c>--verbose</c> prints each pass's tally and first difference.
    /// </param>
    /// <returns>0 when every trace is bit-exact in every pass, 1 otherwise (a model the core refuses counts as failed).</returns>
    /// <exception cref="IOException">A trace file cannot be read.</exception>
    private static int Main(string[] args) {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        bool verbose = args.Contains("--verbose");
        string[] dirs = args.Where(a => a != "--verbose").ToArray();
        if (dirs.Length == 0)
            dirs = [Directory.GetCurrentDirectory()];
        int failed = 0;
        long physicsSteps = 0, envSteps = 0, comparisons = 0;
        // the first trace of each env is the one its speed is measured on, in the order the envs first appear
        var benchmarks = new List<(MjModel Model, RunInfo Run, string Dir)>();
        foreach (string dir in dirs) {
            MjModel model;
            try {
                model = MjModel.Load(Path.Combine(dir, "model.json"));
            } catch (NotSupportedException e) {
                // a model beyond the core's scope is a finding, not a crash: report it and go on with the other folders
                Console.WriteLine($"{Path.GetFileName(dir.TrimEnd('/', '\\'))}: refused by the core: {e.Message}");
                failed++;
                continue;
            }
            RunInfo run = RunInfo.Load(Path.Combine(dir, "run.json"));
            TraceResult r = CheckTrace(dir, model, run, verbose);
            Console.WriteLine($"{Path.GetFileName(dir.TrimEnd('/', '\\'))}: {r.Summary}");
            failed += r.Exact ? 0 : 1;
            physicsSteps += r.PhysicsSteps;
            envSteps += r.EnvSteps;
            comparisons += r.Comparisons;
            if (benchmarks.All(b => b.Run.Env != run.Env))
                benchmarks.Add((model, run, dir));
        }
        Console.WriteLine($"total: {dirs.Length - failed}/{dirs.Length} traces bit-exact; {envSteps} env steps, {physicsSteps} physics steps, "
                          + $"{comparisons} array comparisons per pass");
        foreach ((MjModel model, RunInfo run, string dir) in benchmarks)
            Benchmark(model, run, dir);
        return failed == 0 ? 0 : 1;
    }

    /// <summary>What one trace's checks found.</summary>
    /// <param name="Exact">Whether every reset, action, pass and env step was bit-exact.</param>
    /// <param name="Summary">The one-line report.</param>
    /// <param name="EnvSteps">Env steps checked.</param>
    /// <param name="PhysicsSteps">Physics steps checked.</param>
    /// <param name="Comparisons">Array comparisons made in the closed loop.</param>
    private sealed record TraceResult(bool Exact, string Summary, long EnvSteps, long PhysicsSteps, long Comparisons);


    /// <summary>
    ///     Checks one recording: per episode the reset and actions drawn with NumSharp, the closed loop with the env layer,
    ///     and the resync pass. For an env the checker has no rules for, physics only: the closed loop starts from
    ///     Gymnasium's recorded reset and replays the recorded actions, and the env layer is not checked.
    /// </summary>
    /// <param name="dir">The recording's folder.</param>
    /// <param name="model">The model.</param>
    /// <param name="run">The run description.</param>
    /// <param name="verbose">Whether to print each pass's tally.</param>
    /// <returns>The trace's results.</returns>
    /// <exception cref="IOException">A trace file cannot be read.</exception>
    /// <exception cref="InvalidDataException">The trace is not oracle.py's format 3.</exception>
    /// <remarks>
    ///     A physics-only pass proves the core's arithmetic for the model, not the env: the reset noise, the action
    ///     sampling and the env's observation, reward and info stay unverified until the env has rules here.
    /// </remarks>
    private static TraceResult CheckTrace(string dir, MjModel model, RunInfo run, bool verbose) {
        List<double[]> trace = Records.ReadAll(Path.Combine(dir, "trace.bin"));
        List<double[]> envRecords = Records.ReadAll(Path.Combine(dir, "env.bin"));
        if (trace[0].Length != 2 || trace[0][0] != 3)
            throw new InvalidDataException($"{dir}: not an oracle.py format-3 trace");
        EnvRules? rules = EnvRules.TryFor(model, run);
        var closed = new Checker("closed loop");
        var resync = new Checker("resync");
        var env = new Checker("env layer");
        int resetsOk = 0, actionsOk = 0, envStepsTotal = 0;
        long constraintSteps = 0, iterations = 0;
        var touched = new SortedSet<int>();
        int rec = 1, envRec = 0;
        // one data for the whole run, reset per episode by mj_resetData, as Gymnasium's env resets its one mjData: the
        // closed loop then proves that a reset leaves nothing of the previous episode behind
        using var core = new MjData(model);
        for (int e = 0; e < run.Seeds.Length; e++) {
            int seed = run.Seeds[e];
            int steps = (int)trace[rec][0];
            bool debug = trace[rec][1] != 0;
            rec++;
            int start = rec;
            rec += 4;
            // ---- closed loop with the env layer, from the core's own reset and actions
            core.ResetData();
            double[][]? actions = null;
            if (rules != null) {
                rules.Reset(core, seed, run.Flip);
                if (Program.SameBits(core.qpos, trace[start]) && Program.SameBits(core.qvel, trace[start + 1])
                    && Program.SameBits(core.qacc_warmstart, trace[start + 2]))
                    resetsOk++;
                env.Check("reset obs", rules.Obs(core), envRecords[envRec++], -1);
                actions = SampleActions(rules, seed, steps);
            } else {
                // physics only: Gymnasium's reset as recorded (Step's own forward pass recomputes everything else)
                Array.Copy(trace[start], core.qpos, model.nq);
                Array.Copy(trace[start + 1], core.qvel, model.nv);
                Array.Copy(trace[start + 2], core.qacc_warmstart, model.nv);
                core.time = trace[start + 3][0];
            }
            int recClosed = rec;
            for (int t = 0; t < steps; t++) {
                // physics only replays the recorded action, so the action check is trivially exact there
                double[] action = actions?[t] ?? trace[recClosed];
                if (SameBits(action, trace[recClosed]))
                    actionsOk++;
                recClosed++;
                rules?.Before(core, action);
                Array.Copy(action, core.ctrl, model.nu);
                for (int k = 0; k < run.FrameSkip; k++) {
                    core.Step();
                    CompareStep(closed, core, trace, ref recClosed, debug, t * run.FrameSkip + k);
                    constraintSteps += core.nefc > 0 ? 1 : 0;
                    iterations += core.solver_niter;
                    for (int c = 0; c < core.ncon; c++) {
                        touched.Add(core.con_geom1[c]);
                        touched.Add(core.con_geom2[c]);
                    }
                }
                core.RnePostConstraint();
                ComparePost(closed, core, trace, ref recClosed, t);
                if (rules != null) {
                    (double[] obs, double[] row) = rules.After(core, action, t, run.InfoKeys);
                    env.Check("obs", obs, envRecords[envRec++], t);
                    env.Check("reward, terminated, truncated, info", row, envRecords[envRec++], t);
                }
            }
            // ---- resync: every physics step from MuJoCo's recorded pre-step state
            using var fresh = new MjData(model);
            double[] prevQpos = trace[start], prevQvel = trace[start + 1], prevWarm = trace[start + 2];
            double prevTime = trace[start + 3][0];
            for (int t = 0; t < steps; t++) {
                double[] action = trace[rec++];
                for (int k = 0; k < run.FrameSkip; k++) {
                    int stepStart = rec;
                    Array.Copy(prevQpos, fresh.qpos, model.nq);
                    Array.Copy(prevQvel, fresh.qvel, model.nv);
                    Array.Copy(prevWarm, fresh.qacc_warmstart, model.nv);
                    Array.Copy(action, fresh.ctrl, model.nu);
                    fresh.time = prevTime;
                    fresh.Step();
                    CompareStep(resync, fresh, trace, ref rec, debug, t * run.FrameSkip + k);
                    prevQpos = trace[stepStart];
                    prevQvel = trace[stepStart + 1];
                    prevWarm = trace[stepStart + 3];
                    prevTime = trace[stepStart + StateNames.Length][0];
                }
                fresh.RnePostConstraint();
                ComparePost(resync, fresh, trace, ref rec, t);
            }
            envStepsTotal += steps;
        }
        if (verbose) {
            closed.Report();
            resync.Report();
            env.Report();
        }
        bool envExact = rules == null || (resetsOk == run.Seeds.Length && actionsOk == envStepsTotal && env.Clean);
        bool exact = envExact && closed.Clean && resync.Clean;
        string envPart = rules == null
            ? "env not checked (physics only: no env rules)"
            : $"resets {resetsOk}/{run.Seeds.Length}, actions {actionsOk}/{envStepsTotal}, env {(env.Clean ? "exact" : "DIFFERS")}";
        string summary = $"{run.Env}{(run.Flip ? " flipped" : "")}{(run.Terminate ? "" : " no-terminate")}, {run.Seeds.Length} episodes "
                         + $"({envStepsTotal} env steps): closed loop {closed.ExactSteps}/{closed.Steps}, resync {resync.ExactSteps}/{resync.Steps}, "
                         + $"{envPart}; {constraintSteps} steps with constraints, {iterations} solver iterations, "
                         + $"geoms touching {string.Join(",", touched)}"
                         + (exact ? "" : $"; first difference: {closed.First ?? resync.First ?? env.First}");
        return new TraceResult(exact, summary, envStepsTotal, closed.Steps, closed.Comparisons);
    }

    /// <summary>
    ///     An episode's actions, as <c>action_space.seed(seed)</c> then <c>sample()</c> per step draws them: float64
    ///     <c>uniform(low, high)</c> over the float32 bounds, cast to float32.
    /// </summary>
    /// <param name="rules">The env, for its action bounds.</param>
    /// <param name="seed">The action-space seed.</param>
    /// <param name="steps">How many actions.</param>
    /// <returns>The actions, widened back to float64 as MuJoCo's ctrl receives them.</returns>
    private static double[][] SampleActions(EnvRules rules, int seed, int steps) {
        var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
        using NDArray low = np.array(rules.ActionLow);
        using NDArray high = np.array(rules.ActionHigh);
        var actions = new double[steps][];
        for (int t = 0; t < steps; t++) {
            using NDArray drawn = rng.uniform(low, high, new Shape(rules.ActionLow.Length));
            actions[t] = drawn.ToArray<double>().Select(v => (double)(float)v).ToArray();
        }
        return actions;
    }

    /// <summary>Reads one physics step's records and compares each with the core's arrays.</summary>
    /// <param name="c">The checker.</param>
    /// <param name="core">The core after the step.</param>
    /// <param name="trace">The trace.</param>
    /// <param name="rec">The cursor, advanced past the step's records.</param>
    /// <param name="debug">Whether the episode recorded the dual matrix and the position/velocity-stage arrays.</param>
    /// <param name="step">The physics step index, for the report.</param>
    private static void CompareStep(Checker c, MjData core, List<double[]> trace, ref int rec, bool debug, int step) {
        int nv = core.qvel.Length, nefc = core.nefc, ncon = core.ncon;
        double[][] state = [core.qpos, core.qvel, core.qacc, core.qacc_warmstart, core.qacc_smooth, core.qfrc_constraint,
                            core.qfrc_passive, core.qfrc_bias, core.qfrc_actuator, core.actuator_force];
        c.BeginStep();
        for (int i = 0; i < StateNames.Length; i++)
            c.Check(StateNames[i], state[i], trace[rec++], step);
        c.Check("time, ncon, nefc, niter, nisland",
                [core.time, ncon, nefc, core.nisland > 0 ? core.solver_niter : 0, core.nisland], trace[rec++], step);
        c.Check("contact.dist", core.con_dist.AsSpan(0, ncon).ToArray(), trace[rec++], step);
        c.Check("contact.geom", Enumerable.Range(0, ncon).SelectMany(i => new double[] { core.con_geom1[i], core.con_geom2[i] }).ToArray(), trace[rec++], step);
        c.Check("contact.pos", core.con_pos.AsSpan(0, 3 * ncon).ToArray(), trace[rec++], step);
        c.Check("contact.frame", core.con_frame.AsSpan(0, 9 * ncon).ToArray(), trace[rec++], step);
        c.Check("efc_J", core.efc_J.AsSpan(0, nefc * nv).ToArray(), trace[rec++], step);
        double[] I(int[] v) => v.Take(nefc).Select(x => (double)x).ToArray();
        double[] E(double[] v, int stride = 1) => v.AsSpan(0, stride * nefc).ToArray();
        double[][] efc = [I(core.efc_type), I(core.efc_id), E(core.efc_pos), E(core.efc_margin), E(core.efc_frictionloss),
                          E(core.efc_diagA), E(core.efc_KBIP, 4), E(core.efc_D), E(core.efc_R), E(core.efc_vel),
                          E(core.efc_aref), E(core.efc_b), E(core.efc_force), I(core.efc_state)];
        for (int i = 0; i < EfcNames.Length; i++)
            c.Check(EfcNames[i], efc[i], trace[rec++], step);
        if (debug) {
            c.Check("efc_AR", core.efc_AR.Length == 0 ? [] : core.efc_AR.AsSpan(0, nefc * nefc).ToArray(), trace[rec++], step);
            double[][] dbg = [core.xpos, core.xquat, core.xmat, core.xipos, core.ximat, core.geom_xpos, core.geom_xmat,
                              core.subtree_com, core.cinert, core.cdof, core.crb, core.M, core.qLD, core.qLDiagInv, core.cvel, core.cdof_dot];
            for (int i = 0; i < DebugNames.Length; i++)
                c.Check(DebugNames[i], dbg[i], trace[rec++], step);
        }
        c.EndStep();
    }

    /// <summary>Compares the arrays recorded after an env step's <c>mj_rnePostConstraint</c>.</summary>
    /// <param name="c">The checker.</param>
    /// <param name="core">The core after <see cref="MjData.RnePostConstraint"/>.</param>
    /// <param name="trace">The trace.</param>
    /// <param name="rec">The cursor, advanced past the records.</param>
    /// <param name="t">The env step index.</param>
    private static void ComparePost(Checker c, MjData core, List<double[]> trace, ref int rec, int t) {
        double[][] post = [core.cfrc_ext, core.ten_length, core.ten_velocity, core.site_xpos, core.site_xmat, core.xpos];
        for (int i = 0; i < PostNames.Length; i++)
            c.Check(PostNames[i], post[i], trace[rec++], t);
    }

    /// <summary>Whether two arrays have the same length and the same bits in every entry.</summary>
    /// <param name="a">The first array.</param>
    /// <param name="b">The second array.</param>
    /// <returns><see langword="true"/> when identical.</returns>
    public static bool SameBits(double[] a, double[] b) =>
        a.Length == b.Length && a.Zip(b).All(p => BitConverter.DoubleToInt64Bits(p.First) == BitConverter.DoubleToInt64Bits(p.Second));

    /// <summary>
    ///     Times the core alone (no env layer, no comparisons) over a recording's first episode, best of five runs. The
    ///     time per step depends on the contacts: a body lying on the ground costs several times one standing.
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run, for the env, seed and length.</param>
    /// <param name="dir">
    ///     The recording's folder: an env without rules is timed on the reset and actions recorded there, the others on
    ///     their own NumSharp reset and actions.
    /// </param>
    /// <exception cref="IOException">The trace of an env without rules cannot be read.</exception>
    private static void Benchmark(MjModel model, RunInfo run, string dir) {
        EnvRules? rules = EnvRules.TryFor(model, run);
        (double[] Qpos, double[] Qvel, double[] Warm, double Time, double[][] Actions)? recorded =
            rules == null ? FirstEpisode(Records.ReadAll(Path.Combine(dir, "trace.bin")), run.FrameSkip) : null;
        double[][] actions = recorded?.Actions ?? SampleActions(rules!, run.Seeds[0], run.Steps[0]);
        double best = double.MaxValue;
        for (int r = 0; r < 5; r++) {
            using var core = new MjData(model);
            if (rules != null) {
                rules.Reset(core, run.Seeds[0], run.Flip);
            } else {
                Array.Copy(recorded!.Value.Qpos, core.qpos, model.nq);
                Array.Copy(recorded.Value.Qvel, core.qvel, model.nv);
                Array.Copy(recorded.Value.Warm, core.qacc_warmstart, model.nv);
                core.time = recorded.Value.Time;
            }
            var sw = Stopwatch.StartNew();
            foreach (double[] a in actions) {
                Array.Copy(a, core.ctrl, model.nu);
                for (int k = 0; k < run.FrameSkip; k++)
                    core.Step();
            }
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMicroseconds / (actions.Length * run.FrameSkip));
        }
        Console.WriteLine($"speed ({run.Env}): {best:F2} us per physics step (managed core, best of 5 runs of one episode)");
    }

    /// <summary>
    ///     A trace's first episode as recorded: Gymnasium's reset state and the actions it sampled, for timing an env the
    ///     checker has no rules for.
    /// </summary>
    /// <param name="trace">The trace's records.</param>
    /// <param name="frameSkip">The env's physics steps per env step (run.json's <c>frame_skip</c>).</param>
    /// <returns>The reset's qpos, qvel, qacc_warmstart and time, and the episode's actions.</returns>
    /// <remarks>
    ///     The records of an env step are its action, then per physics step 30 records (47 in a debug episode), then the
    ///     six recorded after <c>mj_rnePostConstraint</c> (<see cref="PostNames"/>); a change to oracle.py's layout must
    ///     change the stride here.
    /// </remarks>
    private static (double[] Qpos, double[] Qvel, double[] Warm, double Time, double[][] Actions) FirstEpisode(
        List<double[]> trace, int frameSkip) {
        int steps = (int)trace[1][0];
        bool debug = trace[1][1] != 0;
        // the state block, the meta record, 4 contact records, efc_J and the efc arrays; then efc_AR and DEBUG
        int perStep = StateNames.Length + 6 + EfcNames.Length + (debug ? 1 + DebugNames.Length : 0);
        int stride = 1 + frameSkip * perStep + PostNames.Length;
        var actions = new double[steps][];
        for (int t = 0; t < steps; t++)
            actions[t] = trace[6 + t * stride];
        return (trace[2], trace[3], trace[4], trace[5][0], actions);
    }
}

/// <summary>oracle.py's run.json: which env, which episodes, and the env's info keys in their recorded order.</summary>
/// <param name="Env">The Gymnasium id.</param>
/// <param name="Seeds">Each episode's reset and action-space seed.</param>
/// <param name="Steps">Each episode's env steps.</param>
/// <param name="Flip">Whether the episodes start upside down.</param>
/// <param name="Terminate">Whether the env terminates when unhealthy.</param>
/// <param name="InfoKeys">The info entries and their sizes, in env.bin's order.</param>
/// <param name="MaxEpisodeSteps">The env's TimeLimit (<c>max_episode_steps</c>), the step that truncates.</param>
/// <param name="FrameSkip">The env's physics steps per env step (<c>frame_skip</c>), an env setting rather than the model's.</param>
internal sealed record RunInfo(string Env, int[] Seeds, int[] Steps, bool Flip, bool Terminate, (string Key, int Size)[] InfoKeys,
                                 int MaxEpisodeSteps, int FrameSkip) {
    /// <summary>Reads run.json.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The run.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="JsonException">The file is not the expected JSON.</exception>
    public static RunInfo Load(string path) {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonElement r = doc.RootElement;
        return new RunInfo(
            r.GetProperty("env").GetString() ?? "",
            r.GetProperty("seeds").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            r.GetProperty("steps").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
            r.GetProperty("flip").GetBoolean(),
            r.GetProperty("terminate").GetBoolean(),
            r.GetProperty("info_keys").EnumerateArray().Select(e => (e[0].GetString() ?? "", e[1].GetInt32())).ToArray(),
            r.GetProperty("max_episode_steps").GetInt32(),
            r.GetProperty("frame_skip").GetInt32());
    }
}

/// <summary>Reads oracle.py's binary records: each an int32 count followed by that many little-endian float64.</summary>
internal static class Records {
    /// <summary>Reads every record of a file.</summary>
    /// <param name="path">The file.</param>
    /// <returns>The records in file order.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="InvalidDataException">The file ends inside a record.</exception>
    public static List<double[]> ReadAll(string path) {
        byte[] data = File.ReadAllBytes(path);
        var list = new List<double[]>();
        int pos = 0;
        while (pos < data.Length) {
            int n = BitConverter.ToInt32(data, pos);
            pos += 4;
            if (pos + 8L * n > data.Length)
                throw new InvalidDataException($"{path}: truncated record at byte {pos}");
            var v = new double[n];
            Buffer.BlockCopy(data, pos, v, 0, 8 * n);
            pos += 8 * n;
            list.Add(v);
        }
        return list;
    }
}

/// <summary>Tallies bitwise comparisons: per array and per step, with the first difference in detail.</summary>
internal sealed class Checker {
    private readonly string name;
    private readonly Dictionary<string, int> mismatchesByArray = new();
    private int steps, exactSteps, comparisons;
    private bool stepClean = true;
    private string? first;

    /// <summary>Creates a checker for one pass.</summary>
    /// <param name="name">The pass name, for the report.</param>
    public Checker(string name) => this.name = name;

    /// <summary>Whether every comparison so far was exact.</summary>
    public bool Clean => mismatchesByArray.Count == 0;

    /// <summary>Physics steps compared.</summary>
    public int Steps => steps;

    /// <summary>Physics steps whose every array was exact.</summary>
    public int ExactSteps => exactSteps;

    /// <summary>Array comparisons made.</summary>
    public int Comparisons => comparisons;

    /// <summary>The first difference found, or <see langword="null"/>.</summary>
    public string? First => first;

    /// <summary>Starts a physics step (its arrays are counted together).</summary>
    public void BeginStep() => stepClean = true;

    /// <summary>Ends a physics step, counting it exact when all its arrays were.</summary>
    public void EndStep() {
        steps++;
        if (stepClean)
            exactSteps++;
    }

    /// <summary>Compares one array and records a difference.</summary>
    /// <param name="what">The array's name.</param>
    /// <param name="ours">The core's values.</param>
    /// <param name="theirs">The recorded values.</param>
    /// <param name="step">The step index (-1 for the reset).</param>
    public void Check(string what, double[] ours, double[] theirs, int step) {
        comparisons++;
        if (Program.SameBits(ours, theirs))
            return;
        stepClean = false;
        mismatchesByArray[what] = mismatchesByArray.GetValueOrDefault(what) + 1;
        if (first != null)
            return;
        if (ours.Length != theirs.Length) {
            first = $"step {step}: {what} has {ours.Length} values, MuJoCo {theirs.Length}";
            return;
        }
        int i = Enumerable.Range(0, ours.Length).First(k => BitConverter.DoubleToInt64Bits(ours[k]) != BitConverter.DoubleToInt64Bits(theirs[k]));
        long ulps = Math.Abs(BitConverter.DoubleToInt64Bits(ours[i]) - BitConverter.DoubleToInt64Bits(theirs[i]));
        first = $"step {step}: {what}[{i}] = {ours[i]:R}, MuJoCo {theirs[i]:R} ({ulps} ulp)";
    }

    /// <summary>Prints the pass's tally.</summary>
    public void Report() {
        string stepsPart = steps > 0 ? $"{exactSteps}/{steps} steps exact, " : "";
        Console.WriteLine($"{name}: {stepsPart}{comparisons - mismatchesByArray.Values.Sum()}/{comparisons} array comparisons exact");
        if (first != null) {
            Console.WriteLine($"  first difference: {first}");
            foreach (var (array, count) in mismatchesByArray.OrderByDescending(p => p.Value).Take(12))
                Console.WriteLine($"  {array}: {count} differing");
        }
    }
}
