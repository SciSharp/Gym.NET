using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp;

namespace MujocoPoc;

/// <summary>
///     Checks the core against MuJoCo 3.15.0 and Gymnasium bit for bit, for HalfCheetah-v5, Hopper-v5 and Humanoid-v5: the
///     reset noise and the sampled actions (NumSharp's NumPy generator), every mjData array oracle.py recorded after each
///     physics step and each env step, and the env's observation, reward, flags and info after each env step.
/// </summary>
/// <remarks>
///     Two passes over each recorded episode. Closed loop: the core runs on its own from its own reset, as an env would, so
///     any difference compounds. Resync: before each physics step the core is put back on MuJoCo's recorded state, so a
///     difference shows in the step that makes it. Exit code 0 when every episode is exact in both passes and the env layer.
/// </remarks>
internal static class Program {
    /// <summary>
    ///     The three envs' <c>max_episode_steps</c>: Gymnasium's TimeLimit truncates at this step, so a shorter episode
    ///     never sees <c>truncated</c> set.
    /// </summary>
    internal const int MaxEpisodeSteps = 1000;

    /// <summary>The names of the per-step records, in oracle.py's order after the state block.</summary>
    private static readonly string[] StateNames =
        ["qpos", "qvel", "qacc", "qacc_warmstart", "qacc_smooth", "qfrc_constraint", "qfrc_passive", "qfrc_bias", "qfrc_actuator", "actuator_force"];

    private static readonly string[] EfcNames =
        ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss", "efc_diagA", "efc_KBIP", "efc_D", "efc_R", "efc_vel", "efc_aref", "efc_b", "efc_force", "efc_state"];

    private static readonly string[] DebugNames =
        ["xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "subtree_com", "cinert", "cdof", "crb", "M", "qLD", "qLDiagInv", "cvel", "cdof_dot"];

    private static readonly string[] PostNames = ["cfrc_ext", "ten_length", "ten_velocity"];

    /// <summary>
    ///     Checks every recorded trace given and prints one report per trace, then the totals, then the core's speed on
    ///     the first trace of each env.
    /// </summary>
    /// <param name="args">
    ///     Folders, each holding one oracle.py recording (model.json, run.json, trace.bin, env.bin); default: the current
    ///     folder. <c>--verbose</c> prints each pass's tally and first difference.
    /// </param>
    /// <returns>0 when every trace is bit-exact in every pass, 1 otherwise.</returns>
    /// <exception cref="IOException">A trace file cannot be read.</exception>
    /// <exception cref="NotSupportedException">A model is outside the core's scope, or the env is not one of the three.</exception>
    private static int Main(string[] args) {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        bool verbose = args.Contains("--verbose");
        string[] dirs = args.Where(a => a != "--verbose").ToArray();
        if (dirs.Length == 0)
            dirs = [Directory.GetCurrentDirectory()];
        int failed = 0;
        long physicsSteps = 0, envSteps = 0, comparisons = 0;
        // the first trace of each env is the one its speed is measured on, in the order the envs first appear
        var benchmarks = new List<(CheetahModel Model, RunInfo Run)>();
        foreach (string dir in dirs) {
            CheetahModel model = CheetahModel.Load(Path.Combine(dir, "model.json"));
            RunInfo run = RunInfo.Load(Path.Combine(dir, "run.json"));
            TraceResult r = CheckTrace(dir, model, run, verbose);
            Console.WriteLine($"{Path.GetFileName(dir.TrimEnd('/', '\\'))}: {r.Summary}");
            failed += r.Exact ? 0 : 1;
            physicsSteps += r.PhysicsSteps;
            envSteps += r.EnvSteps;
            comparisons += r.Comparisons;
            if (benchmarks.All(b => b.Run.Env != run.Env))
                benchmarks.Add((model, run));
        }
        Console.WriteLine($"total: {dirs.Length - failed}/{dirs.Length} traces bit-exact; {envSteps} env steps, {physicsSteps} physics steps, "
                          + $"{comparisons} array comparisons per pass");
        foreach ((CheetahModel model, RunInfo run) in benchmarks)
            Benchmark(model, run);
        return failed == 0 ? 0 : 1;
    }

    /// <summary>What one trace's checks found.</summary>
    /// <param name="Exact">Whether every reset, action, pass and env step was bit-exact.</param>
    /// <param name="Summary">The one-line report.</param>
    /// <param name="EnvSteps">Env steps checked.</param>
    /// <param name="PhysicsSteps">Physics steps checked.</param>
    /// <param name="Comparisons">Array comparisons made in the closed loop.</param>
    private sealed record TraceResult(bool Exact, string Summary, long EnvSteps, long PhysicsSteps, long Comparisons);

    /// <summary>oracle.py's run.json: which env, which episodes, and the env's info keys in their recorded order.</summary>
    /// <param name="Env">The Gymnasium id.</param>
    /// <param name="Seeds">Each episode's reset and action-space seed.</param>
    /// <param name="Steps">Each episode's env steps.</param>
    /// <param name="Flip">Whether the episodes start upside down.</param>
    /// <param name="Terminate">Whether the env terminates when unhealthy.</param>
    /// <param name="InfoKeys">The info entries and their sizes, in env.bin's order.</param>
    private sealed record RunInfo(string Env, int[] Seeds, int[] Steps, bool Flip, bool Terminate, (string Key, int Size)[] InfoKeys) {
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
                r.GetProperty("info_keys").EnumerateArray().Select(e => (e[0].GetString() ?? "", e[1].GetInt32())).ToArray());
        }
    }

    /// <summary>
    ///     Checks one recording: per episode the reset and actions drawn with NumSharp, the closed loop with the env layer,
    ///     and the resync pass.
    /// </summary>
    /// <param name="dir">The recording's folder.</param>
    /// <param name="model">The model.</param>
    /// <param name="run">The run description.</param>
    /// <param name="verbose">Whether to print each pass's tally.</param>
    /// <returns>The trace's results.</returns>
    /// <exception cref="IOException">A trace file cannot be read.</exception>
    /// <exception cref="InvalidDataException">The trace is not oracle.py's format 2.</exception>
    private static TraceResult CheckTrace(string dir, CheetahModel model, RunInfo run, bool verbose) {
        List<double[]> trace = Records.ReadAll(Path.Combine(dir, "trace.bin"));
        List<double[]> envRecords = Records.ReadAll(Path.Combine(dir, "env.bin"));
        if (trace[0].Length != 2 || trace[0][0] != 2)
            throw new InvalidDataException($"{dir}: not an oracle.py format-2 trace");
        EnvRules rules = EnvRules.For(run.Env, model, run.Terminate);
        var closed = new Checker("closed loop");
        var resync = new Checker("resync");
        var env = new Checker("env layer");
        int resetsOk = 0, actionsOk = 0, envStepsTotal = 0;
        long constraintSteps = 0, iterations = 0;
        var touched = new SortedSet<int>();
        int rec = 1, envRec = 0;
        for (int e = 0; e < run.Seeds.Length; e++) {
            int seed = run.Seeds[e];
            int steps = (int)trace[rec][0];
            bool debug = trace[rec][1] != 0;
            rec++;
            int start = rec;
            rec += 4;
            // ---- closed loop with the env layer, from the core's own reset and actions
            var core = new CheetahCore(model);
            rules.Reset(core, seed, run.Flip);
            if (Program.SameBits(core.qpos, trace[start]) && Program.SameBits(core.qvel, trace[start + 1])
                && Program.SameBits(core.qacc_warmstart, trace[start + 2]))
                resetsOk++;
            env.Check("reset obs", rules.Obs(core), envRecords[envRec++], -1);
            double[][] actions = SampleActions(rules, seed, steps);
            int recClosed = rec;
            for (int t = 0; t < steps; t++) {
                if (SameBits(actions[t], trace[recClosed]))
                    actionsOk++;
                recClosed++;
                rules.Before(core);
                Array.Copy(actions[t], core.ctrl, model.nu);
                for (int k = 0; k < model.frame_skip; k++) {
                    core.Step();
                    CompareStep(closed, core, trace, ref recClosed, debug, t * model.frame_skip + k);
                    constraintSteps += core.nefc > 0 ? 1 : 0;
                    iterations += core.solver_niter;
                    for (int c = 0; c < core.ncon; c++) {
                        touched.Add(core.con_geom1[c]);
                        touched.Add(core.con_geom2[c]);
                    }
                }
                core.RnePostConstraint();
                ComparePost(closed, core, trace, ref recClosed, t);
                (double[] obs, double[] row) = rules.After(core, actions[t], t, run.InfoKeys);
                env.Check("obs", obs, envRecords[envRec++], t);
                env.Check("reward, terminated, truncated, info", row, envRecords[envRec++], t);
            }
            // ---- resync: every physics step from MuJoCo's recorded pre-step state
            var fresh = new CheetahCore(model);
            double[] prevQpos = trace[start], prevQvel = trace[start + 1], prevWarm = trace[start + 2];
            double prevTime = trace[start + 3][0];
            for (int t = 0; t < steps; t++) {
                double[] action = trace[rec++];
                for (int k = 0; k < model.frame_skip; k++) {
                    int stepStart = rec;
                    Array.Copy(prevQpos, fresh.qpos, model.nq);
                    Array.Copy(prevQvel, fresh.qvel, model.nv);
                    Array.Copy(prevWarm, fresh.qacc_warmstart, model.nv);
                    Array.Copy(action, fresh.ctrl, model.nu);
                    fresh.time = prevTime;
                    fresh.Step();
                    CompareStep(resync, fresh, trace, ref rec, debug, t * model.frame_skip + k);
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
        bool exact = resetsOk == run.Seeds.Length && actionsOk == envStepsTotal && closed.Clean && resync.Clean && env.Clean;
        string summary = $"{run.Env}{(run.Flip ? " flipped" : "")}{(run.Terminate ? "" : " no-terminate")}, {run.Seeds.Length} episodes "
                         + $"({envStepsTotal} env steps): resets {resetsOk}/{run.Seeds.Length}, actions {actionsOk}/{envStepsTotal}, "
                         + $"closed loop {closed.ExactSteps}/{closed.Steps}, resync {resync.ExactSteps}/{resync.Steps}, env {(env.Clean ? "exact" : "DIFFERS")}; "
                         + $"{constraintSteps} steps with constraints, {iterations} solver iterations, geoms touching {string.Join(",", touched)}"
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
    private static void CompareStep(Checker c, CheetahCore core, List<double[]> trace, ref int rec, bool debug, int step) {
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
    /// <param name="core">The core after <see cref="CheetahCore.RnePostConstraint"/>.</param>
    /// <param name="trace">The trace.</param>
    /// <param name="rec">The cursor, advanced past the records.</param>
    /// <param name="t">The env step index.</param>
    private static void ComparePost(Checker c, CheetahCore core, List<double[]> trace, ref int rec, int t) {
        double[][] post = [core.cfrc_ext, core.ten_length, core.ten_velocity];
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
    private static void Benchmark(CheetahModel model, RunInfo run) {
        EnvRules rules = EnvRules.For(run.Env, model, run.Terminate);
        double[][] actions = SampleActions(rules, run.Seeds[0], run.Steps[0]);
        double best = double.MaxValue;
        for (int r = 0; r < 5; r++) {
            var core = new CheetahCore(model);
            rules.Reset(core, run.Seeds[0], run.Flip);
            var sw = Stopwatch.StartNew();
            foreach (double[] a in actions) {
                Array.Copy(a, core.ctrl, model.nu);
                for (int k = 0; k < model.frame_skip; k++)
                    core.Step();
            }
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMicroseconds / (actions.Length * model.frame_skip));
        }
        Console.WriteLine($"speed ({run.Env}): {best:F2} us per physics step (managed core, best of 5 runs of one episode)");
    }
}

/// <summary>
///     An env's rules over the core, as Gymnasium's v5 class has them: the reset noise, the action bounds, the
///     observation, and per step the reward, the termination and the info entries. Each reproduces NumPy's arithmetic
///     where it decides bits (NEP 50 float32 costs, NumPy's pairwise sums, einsum's sequential sum).
/// </summary>
internal abstract class EnvRules {
    /// <summary>The model the rules read.</summary>
    protected readonly CheetahModel M;

    /// <summary>The control step, <c>timestep * frame_skip</c>.</summary>
    protected readonly double Dt;

    /// <summary>Whether an unhealthy state terminates the episode.</summary>
    protected readonly bool Terminate;

    /// <summary>The action space's lower bounds (the control ranges rounded to float32).</summary>
    public readonly double[] ActionLow;

    /// <summary>The action space's upper bounds.</summary>
    public readonly double[] ActionHigh;

    /// <summary>Sets up the rules for a model.</summary>
    /// <param name="model">The model.</param>
    /// <param name="terminate">Whether an unhealthy state terminates the episode.</param>
    protected EnvRules(CheetahModel model, bool terminate) {
        M = model;
        Terminate = terminate;
        Dt = model.timestep * model.frame_skip;
        ActionLow = Enumerable.Range(0, model.nu).Select(i => (double)(float)model.actuator_ctrlrange[2 * i]).ToArray();
        ActionHigh = Enumerable.Range(0, model.nu).Select(i => (double)(float)model.actuator_ctrlrange[2 * i + 1]).ToArray();
    }

    /// <summary>The rules of an env id.</summary>
    /// <param name="env">HalfCheetah-v5, Hopper-v5 or Humanoid-v5.</param>
    /// <param name="model">The model.</param>
    /// <param name="terminate">Whether an unhealthy state terminates the episode.</param>
    /// <returns>The rules.</returns>
    /// <exception cref="NotSupportedException">Another env.</exception>
    public static EnvRules For(string env, CheetahModel model, bool terminate) => env switch {
        "HalfCheetah-v5" => new HalfCheetahRules(model),
        "Hopper-v5" => new HopperRules(model, terminate),
        "Humanoid-v5" => new HumanoidRules(model, terminate),
        _ => throw new NotSupportedException($"no env rules for {env}"),
    };

    /// <summary>
    ///     Gymnasium's reset after <c>mj_resetData</c>: the env's noise from <c>Generator(PCG64(SeedSequence(seed)))</c>
    ///     added to the reference configuration (and zero velocity), the optional flip, then <c>mj_forward</c>
    ///     (<c>set_state</c>).
    /// </summary>
    /// <param name="core">A new core, whose state is set.</param>
    /// <param name="seed">The reset seed.</param>
    /// <param name="flip">Whether to turn the model upside down (<c>qpos[2] += pi</c>).</param>
    public void Reset(CheetahCore core, int seed, bool flip) {
        var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
        (double[] dq, double[] dv) = Noise(rng);
        for (int i = 0; i < M.nq; i++)
            core.qpos[i] = M.qpos0[i] + dq[i];
        for (int i = 0; i < M.nv; i++)
            core.qvel[i] = 0.0 + dv[i];
        if (flip)
            core.qpos[2] += Math.PI;
        core.Forward();
    }

    /// <summary>The env's reset noise, in its draw order.</summary>
    /// <param name="rng">The seeded generator.</param>
    /// <returns>The position and velocity noise.</returns>
    protected abstract (double[] Qpos, double[] Qvel) Noise(Generator rng);

    /// <summary>The observation.</summary>
    /// <param name="core">The core.</param>
    /// <returns>The observation.</returns>
    public abstract double[] Obs(CheetahCore core);

    /// <summary>Records what the step's reward needs from before the physics steps.</summary>
    /// <param name="core">The core before the step.</param>
    public abstract void Before(CheetahCore core);

    /// <summary>The step's results after the physics steps and <c>mj_rnePostConstraint</c>.</summary>
    /// <param name="core">The core after the step.</param>
    /// <param name="action">The action (float32 values).</param>
    /// <param name="t">The step index within the episode.</param>
    /// <param name="infoKeys">The info entries in env.bin's order.</param>
    /// <returns>The observation, and [reward, terminated, truncated, the info values].</returns>
    /// <exception cref="KeyNotFoundException">An info key the rules do not produce.</exception>
    public (double[] Obs, double[] Row) After(CheetahCore core, double[] action, int t, (string Key, int Size)[] infoKeys) {
        (double reward, bool terminated, Dictionary<string, double[]> info) = Results(core, action);
        var row = new List<double> { reward, terminated ? 1 : 0, t + 1 >= Program.MaxEpisodeSteps ? 1 : 0 };
        foreach ((string key, _) in infoKeys)
            row.AddRange(info[key]);
        return (Obs(core), row.ToArray());
    }

    /// <summary>The reward, the termination and the info entries.</summary>
    /// <param name="core">The core after the step.</param>
    /// <param name="action">The action.</param>
    /// <returns>The results.</returns>
    protected abstract (double Reward, bool Terminated, Dictionary<string, double[]> Info) Results(CheetahCore core, double[] action);

    /// <summary>
    ///     <c>np.sum(np.square(a))</c> of a float32 array: float32 squares, then NumPy's add reduction (the identity 0 plus
    ///     the pairwise sum, a plain loop below 8 entries).
    /// </summary>
    /// <param name="a">The float32 values (widened).</param>
    /// <returns>The float32 sum.</returns>
    protected static float SumSquaresF32(double[] a) {
        var sq = new float[a.Length];
        for (int i = 0; i < a.Length; i++) {
            float f = (float)a[i];
            sq[i] = f * f;
        }
        return 0f + PairwiseF32(sq);
    }

    /// <summary>NumPy's float32 pairwise sum for up to 128 entries (8 accumulators beyond 7 entries).</summary>
    /// <param name="v">The values.</param>
    /// <returns>The sum.</returns>
    private static float PairwiseF32(float[] v) {
        int n = v.Length;
        if (n < 8) {
            float r = 0;
            foreach (float x in v)
                r += x;
            return r;
        }
        var acc = new float[8];
        Array.Copy(v, acc, 8);
        int i = 8;
        for (; i < n - n % 8; i += 8)
            for (int j = 0; j < 8; j++)
                acc[j] += v[i + j];
        float res = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; i++)
            res += v[i];
        return res;
    }

    /// <summary>NumPy's float64 sum of up to 128 entries: the identity 0 plus the pairwise sum.</summary>
    /// <param name="v">The values.</param>
    /// <returns>The sum.</returns>
    protected static double SumF64(double[] v) {
        int n = v.Length;
        double res;
        if (n < 8) {
            res = 0;
            foreach (double x in v)
                res += x;
            return 0.0 + res;
        }
        var acc = new double[8];
        Array.Copy(v, acc, 8);
        int i = 8;
        for (; i < n - n % 8; i += 8)
            for (int j = 0; j < 8; j++)
                acc[j] += v[i + j];
        res = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; i < n; i++)
            res += v[i];
        return 0.0 + res;
    }
}

/// <summary>HalfCheetah-v5: forward velocity reward minus a float32 control cost; it never terminates.</summary>
internal sealed class HalfCheetahRules : EnvRules {
    private double xBefore;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    public HalfCheetahRules(CheetahModel model) : base(model, false) { }

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) Noise(Generator rng) {
        using NDArray u = rng.uniform(-0.1, 0.1, new Shape(M.nq));
        using NDArray z = rng.standard_normal(new Shape(M.nv));
        return (u.ToArray<double>(), z.ToArray<double>().Select(v => 0.1 * v).ToArray());
    }

    /// <inheritdoc/>
    public override double[] Obs(CheetahCore core) => core.qpos.Skip(1).Concat(core.qvel).ToArray();

    /// <inheritdoc/>
    public override void Before(CheetahCore core) => xBefore = core.qpos[0];

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(CheetahCore core, double[] action) {
        double xAfter = core.qpos[0];
        double xVelocity = (xAfter - xBefore) / Dt;
        double forward = 1.0 * xVelocity;
        float ctrlCost = 0.1f * SumSquaresF32(action);
        return (forward - ctrlCost, false, new Dictionary<string, double[]> {
            ["x_position"] = [xAfter], ["x_velocity"] = [xVelocity], ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost],
        });
    }
}

/// <summary>
///     Hopper-v5: forward velocity plus a healthy reward minus a float32 control cost; unhealthy (fallen, tilted, or a state
///     beyond ±100) terminates; velocities are clipped to ±10 in the observation.
/// </summary>
internal sealed class HopperRules : EnvRules {
    private double xBefore;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="terminate">Whether an unhealthy state terminates.</param>
    public HopperRules(CheetahModel model, bool terminate) : base(model, terminate) { }

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) Noise(Generator rng) {
        using NDArray q = rng.uniform(-5e-3, 5e-3, new Shape(M.nq));
        using NDArray v = rng.uniform(-5e-3, 5e-3, new Shape(M.nv));
        return (q.ToArray<double>(), v.ToArray<double>());
    }

    /// <inheritdoc/>
    public override double[] Obs(CheetahCore core) =>
        core.qpos.Skip(1).Concat(core.qvel.Select(v => v < -10 ? -10.0 : v > 10 ? 10.0 : v)).ToArray();

    /// <inheritdoc/>
    public override void Before(CheetahCore core) => xBefore = core.qpos[0];

    /// <summary>Whether the hopper is healthy: state within ±100 (beyond the x and z positions), z above 0.7, angle within ±0.2.</summary>
    /// <param name="core">The core.</param>
    /// <returns>The health.</returns>
    private bool Healthy(CheetahCore core) {
        bool state = core.qpos.Skip(2).Concat(core.qvel).All(s => -100.0 < s && s < 100.0);
        double z = core.qpos[1], angle = core.qpos[2];
        return state && 0.7 < z && z < double.PositiveInfinity && -0.2 < angle && angle < 0.2;
    }

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(CheetahCore core, double[] action) {
        double xAfter = core.qpos[0];
        double xVelocity = (xAfter - xBefore) / Dt;
        double forward = 1.0 * xVelocity;
        bool healthy = Healthy(core);
        double healthyReward = healthy ? 1.0 : 0.0;
        double rewards = forward + healthyReward;
        float ctrlCost = 1e-3f * SumSquaresF32(action);
        return (rewards - ctrlCost, !healthy && Terminate, new Dictionary<string, double[]> {
            ["x_position"] = [xAfter], ["z_distance_from_origin"] = [core.qpos[1] - M.qpos0[1]], ["x_velocity"] = [xVelocity],
            ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost], ["reward_survive"] = [healthyReward],
        });
    }
}

/// <summary>
///     Humanoid-v5: the mass centre's forward velocity (weight 1.25) plus a healthy reward, minus float64 control and
///     contact costs; a torso height outside (1, 2) terminates; the observation adds the body inertias, velocities,
///     actuator forces and external forces to the state.
/// </summary>
internal sealed class HumanoidRules : EnvRules {
    private double[] xyBefore = [0, 0];

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="terminate">Whether an unhealthy state terminates.</param>
    public HumanoidRules(CheetahModel model, bool terminate) : base(model, terminate) { }

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) Noise(Generator rng) {
        using NDArray q = rng.uniform(-1e-2, 1e-2, new Shape(M.nq));
        using NDArray v = rng.uniform(-1e-2, 1e-2, new Shape(M.nv));
        return (q.ToArray<double>(), v.ToArray<double>());
    }

    /// <inheritdoc/>
    public override double[] Obs(CheetahCore core) =>
        core.qpos.Skip(2).Concat(core.qvel).Concat(core.cinert.Skip(10)).Concat(core.cvel.Skip(6))
            .Concat(core.qfrc_actuator.Skip(6)).Concat(core.cfrc_ext.Skip(6)).ToArray();

    /// <summary>
    ///     <c>mass_center</c>: <c>einsum("b,bj->j", body_mass, xipos)</c> (each column summed in body order from 0) over
    ///     <c>body_mass.sum()</c> (NumPy's pairwise sum), the x and y components.
    /// </summary>
    /// <param name="core">The core, whose last forward pass set xipos.</param>
    /// <returns>The mass centre's x and y.</returns>
    private double[] MassCenter(CheetahCore core) {
        double denom = SumF64(M.body_mass);
        var res = new double[2];
        for (int j = 0; j < 2; j++) {
            double num = 0;
            for (int b = 0; b < M.nbody; b++)
                num += M.body_mass[b] * core.xipos[3 * b + j];
            res[j] = num / denom;
        }
        return res;
    }

    /// <inheritdoc/>
    public override void Before(CheetahCore core) => xyBefore = MassCenter(core);

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(CheetahCore core, double[] action) {
        double[] xyAfter = MassCenter(core);
        double xVelocity = (xyAfter[0] - xyBefore[0]) / Dt, yVelocity = (xyAfter[1] - xyBefore[1]) / Dt;
        double forward = 1.25 * xVelocity;
        double z = core.qpos[2];
        bool healthy = 1.0 < z && z < 2.0;
        double healthyReward = healthy ? 5.0 : 0.0;
        double rewards = forward + healthyReward;
        // float64 costs: data.ctrl holds the action widened to float64; cfrc_ext's 84 squares
        double ctrlCost = 0.1 * SumF64(core.ctrl.Select(c => c * c).ToArray());
        double contactCost = 5e-7 * SumF64(core.cfrc_ext.Select(c => c * c).ToArray());
        contactCost = Math.Min(Math.Max(contactCost, double.NegativeInfinity), 10.0);
        double costs = ctrlCost + contactCost;
        double x = core.qpos[0], y = core.qpos[1];
        return (rewards - costs, !healthy && Terminate, new Dictionary<string, double[]> {
            ["x_position"] = [x], ["y_position"] = [y],
            ["tendon_length"] = (double[])core.ten_length.Clone(), ["tendon_velocity"] = (double[])core.ten_velocity.Clone(),
            ["distance_from_origin"] = [Math.Sqrt(x * x + y * y)],
            ["x_velocity"] = [xVelocity], ["y_velocity"] = [yVelocity],
            ["reward_survive"] = [healthyReward], ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost],
            ["reward_contact"] = [-contactCost],
        });
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
