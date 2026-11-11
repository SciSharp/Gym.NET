using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using NumSharp;

namespace MujocoPoc;

/// <summary>
///     Checks the HalfCheetah core against MuJoCo 3.15.0 and Gymnasium, bit for bit: the reset noise and the sampled
///     actions (NumSharp's NumPy generator), every mjData array oracle.py recorded after each of the 5,000 physics steps,
///     and the env's observation, reward and info after each of the 1,000 env steps.
/// </summary>
/// <remarks>
///     Two passes over the same trace. Closed loop: the core runs on its own from its own reset, as an env would, so any
///     difference compounds. Resync: before each physics step the core is put back on MuJoCo's recorded state, so a
///     difference shows in the step that makes it. Exit code 0 when both passes and the env layer are exact.
/// </remarks>
internal static class Program {
    /// <summary>The names of the per-step records, in oracle.py's order after the state block.</summary>
    private static readonly string[] StateNames =
        ["qpos", "qvel", "qacc", "qacc_warmstart", "qacc_smooth", "qfrc_constraint", "qfrc_passive", "qfrc_bias", "qfrc_actuator", "actuator_force"];

    private static readonly string[] EfcNames =
        ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss", "efc_diagA", "efc_KBIP", "efc_D", "efc_R", "efc_vel", "efc_aref", "efc_b", "efc_force", "efc_state"];

    private static readonly string[] DebugNames =
        ["xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "subtree_com", "cinert", "cdof", "crb", "M", "qLD", "qLDiagInv", "cvel", "cdof_dot"];

    /// <summary>Checks every recorded episode given and prints one report per episode, then the totals.</summary>
    /// <param name="args">
    ///     Folders, each holding one episode's model.json, run.json, trace.bin and env.bin (default: the current folder).
    ///     <c>--verbose</c> prints each pass's tally and first difference.
    /// </param>
    /// <returns>0 when every episode is bit-exact in every pass, 1 otherwise.</returns>
    /// <exception cref="IOException">A trace file cannot be read.</exception>
    /// <exception cref="NotSupportedException">The model is outside the core's scope.</exception>
    private static int Main(string[] args) {
        CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
        bool verbose = args.Contains("--verbose");
        string[] dirs = args.Where(a => a != "--verbose").ToArray();
        if (dirs.Length == 0)
            dirs = [Directory.GetCurrentDirectory()];
        int failed = 0;
        long physicsSteps = 0, comparisons = 0, contactSteps = 0, iterations = 0;
        var touched = new SortedSet<int>();
        CheetahModel? model = null;
        double[][]? actions = null;
        foreach (string dir in dirs) {
            model = CheetahModel.Load(Path.Combine(dir, "model.json"));
            Episode e = RunEpisode(dir, model, verbose);
            Console.WriteLine($"{Path.GetFileName(dir.TrimEnd('/', '\\'))}: {e.Summary}");
            failed += e.Exact ? 0 : 1;
            physicsSteps += e.PhysicsSteps;
            comparisons += e.Comparisons;
            contactSteps += e.ContactSteps;
            iterations += e.SolverIterations;
            touched.UnionWith(e.TouchedGeoms);
            actions = e.Actions;
        }
        Console.WriteLine($"total: {dirs.Length - failed}/{dirs.Length} episodes bit-exact; {physicsSteps} physics steps "
                          + $"({contactSteps} with constraint rows, {iterations} Newton iterations), {comparisons} array comparisons; "
                          + $"geoms that touched the ground: {string.Join(",", touched)}");
        if (model != null && actions != null)
            Benchmark(model, actions);
        return failed == 0 ? 0 : 1;
    }

    /// <summary>What one episode's checks found.</summary>
    /// <param name="Exact">Whether the reset, the actions, both passes and the env layer were all bit-exact.</param>
    /// <param name="Summary">The one-line report.</param>
    /// <param name="PhysicsSteps">Physics steps checked.</param>
    /// <param name="Comparisons">Array comparisons made in the closed loop.</param>
    /// <param name="ContactSteps">Physics steps that had constraint rows.</param>
    /// <param name="SolverIterations">Newton iterations over the episode.</param>
    /// <param name="TouchedGeoms">Geoms that had a contact with the ground.</param>
    /// <param name="Actions">The episode's actions, for the benchmark.</param>
    private sealed record Episode(bool Exact, string Summary, long PhysicsSteps, long Comparisons, long ContactSteps,
                                  long SolverIterations, SortedSet<int> TouchedGeoms, double[][] Actions);

    /// <summary>
    ///     Checks one recorded episode: the reset and actions drawn with NumSharp, the closed loop with the env layer, and
    ///     the resync pass.
    /// </summary>
    /// <param name="dir">The episode's folder.</param>
    /// <param name="model">The model.</param>
    /// <param name="verbose">Whether to print each pass's tally.</param>
    /// <returns>The episode's results.</returns>
    /// <exception cref="IOException">A trace file cannot be read.</exception>
    private static Episode RunEpisode(string dir, CheetahModel model, bool verbose) {
        List<double[]> trace = Records.ReadAll(Path.Combine(dir, "trace.bin"));
        List<double[]> envRecords = Records.ReadAll(Path.Combine(dir, "env.bin"));
        using var run = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, "run.json")));
        int seed = run.RootElement.GetProperty("seed").GetInt32();
        bool flip = run.RootElement.GetProperty("flip").GetBoolean();
        int perStep = StateNames.Length + 6 + EfcNames.Length + DebugNames.Length;
        int envSteps = (trace.Count - 4) / (1 + model.frame_skip * perStep);

        // ---- reset and actions, drawn as Gymnasium draws them
        var core = new CheetahCore(model);
        bool resetOk = Reset(core, model, trace, seed, flip);
        double[][] actions = SampleActions(model.nu, envSteps, seed);
        int actionOk = 0;
        for (int t = 0; t < envSteps; t++)
            if (SameBits(actions[t], trace[4 + t * (1 + model.frame_skip * perStep)]))
                actionOk++;

        // ---- closed loop with the env layer
        var closed = new Checker("closed loop");
        var env = new Checker("env layer");
        long contactSteps = 0, iterations = 0;
        var touched = new SortedSet<int>();
        Compare(env, "reset obs", Obs(core), envRecords[0], -1);
        int rec = 4;
        double dt = model.timestep * model.frame_skip;
        for (int t = 0; t < envSteps; t++) {
            rec++; // the action record (checked above)
            double xBefore = core.qpos[0];
            for (int i = 0; i < model.nu; i++)
                core.ctrl[i] = actions[t][i];
            for (int k = 0; k < model.frame_skip; k++) {
                core.Step();
                CompareStep(closed, core, trace, ref rec, t * model.frame_skip + k);
                contactSteps += core.nefc > 0 ? 1 : 0;
                iterations += core.solver_niter;
                for (int c = 0; c < core.ncon; c++)
                    touched.Add(core.con_geom2[c]);
            }
            EnvCheck(env, core, actions[t], xBefore, dt, t, envSteps, envRecords);
        }

        // ---- resync: every physics step from MuJoCo's recorded pre-step state
        var resync = new Checker("resync");
        var fresh = new CheetahCore(model);
        rec = 4;
        double[] prevQpos = trace[0], prevQvel = trace[1], prevWarm = trace[2];
        double prevTime = trace[3][0];
        for (int t = 0; t < envSteps; t++) {
            double[] action = trace[rec++];
            for (int k = 0; k < model.frame_skip; k++) {
                int start = rec;
                Array.Copy(prevQpos, fresh.qpos, model.nq);
                Array.Copy(prevQvel, fresh.qvel, model.nv);
                Array.Copy(prevWarm, fresh.qacc_warmstart, model.nv);
                Array.Copy(action, fresh.ctrl, model.nu);
                fresh.time = prevTime;
                fresh.Step();
                CompareStep(resync, fresh, trace, ref rec, t * model.frame_skip + k);
                prevQpos = trace[start];
                prevQvel = trace[start + 1];
                prevWarm = trace[start + 3];
                prevTime = trace[start + StateNames.Length][0];
            }
        }

        if (verbose) {
            closed.Report();
            resync.Report();
            env.Report();
        }
        bool exact = resetOk && actionOk == envSteps && closed.Clean && resync.Clean && env.Clean;
        string summary = $"seed {seed}{(flip ? " flipped" : "")}: reset {(resetOk ? "exact" : "DIFFERS")}, actions {actionOk}/{envSteps}, "
                         + $"closed loop {closed.ExactSteps}/{closed.Steps}, resync {resync.ExactSteps}/{resync.Steps}, "
                         + $"env {(env.Clean ? "exact" : "DIFFERS")}; {contactSteps} steps with constraints, geoms {string.Join(",", touched)}"
                         + (exact ? "" : $"; first difference: {closed.First ?? resync.First ?? env.First}");
        return new Episode(exact, summary, closed.Steps, closed.Comparisons, contactSteps, iterations, touched, actions);
    }

    /// <summary>
    ///     Gymnasium's HalfCheetah reset: <c>qpos = init_qpos + uniform(-0.1, 0.1, nq)</c>, <c>qvel = init_qvel +
    ///     0.1*standard_normal(nv)</c> from <c>Generator(PCG64(SeedSequence(seed)))</c>, then oracle.py's optional flip
    ///     (<c>qpos[2] += pi</c>); compared with MuJoCo's recorded start.
    /// </summary>
    /// <param name="core">The core, whose state is set.</param>
    /// <param name="model">The model.</param>
    /// <param name="trace">The trace, whose first records are the starting qpos, qvel, qacc_warmstart and time.</param>
    /// <param name="seed">The reset seed.</param>
    /// <param name="flip">Whether the episode starts upside down.</param>
    /// <returns>Whether the drawn state equals the recorded one bit for bit.</returns>
    private static bool Reset(CheetahCore core, CheetahModel model, List<double[]> trace, int seed, bool flip) {
        var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
        using NDArray uniform = rng.uniform(-0.1, 0.1, new Shape(model.nq));
        using NDArray normal = rng.standard_normal(new Shape(model.nv));
        double[] u = uniform.ToArray<double>(), z = normal.ToArray<double>();
        for (int i = 0; i < model.nq; i++)
            core.qpos[i] = model.qpos0[i] + u[i];
        for (int i = 0; i < model.nv; i++)
            core.qvel[i] = 0.0 + 0.1 * z[i];
        if (flip)
            core.qpos[2] += Math.PI;
        Array.Clear(core.qacc_warmstart);
        core.time = 0;
        return SameBits(core.qpos, trace[0]) && SameBits(core.qvel, trace[1]) && SameBits(core.qacc_warmstart, trace[2]);
    }

    /// <summary>
    ///     The env's actions, as <c>action_space.seed(seed)</c> then <c>sample()</c> per step draws them: float64
    ///     <c>uniform(low, high)</c> over the float32 bounds, cast to float32.
    /// </summary>
    /// <param name="nu">The action size.</param>
    /// <param name="steps">How many actions.</param>
    /// <param name="seed">The action-space seed.</param>
    /// <returns>The actions, widened back to float64 as MuJoCo's ctrl receives them.</returns>
    private static double[][] SampleActions(int nu, int steps, int seed) {
        var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
        using NDArray low = np.array(Enumerable.Repeat(-1.0, nu).ToArray());
        using NDArray high = np.array(Enumerable.Repeat(1.0, nu).ToArray());
        var actions = new double[steps][];
        for (int t = 0; t < steps; t++) {
            using NDArray drawn = rng.uniform(low, high, new Shape(nu));
            actions[t] = drawn.ToArray<double>().Select(v => (double)(float)v).ToArray();
        }
        return actions;
    }

    /// <summary>The observation: qpos without the x position, then qvel.</summary>
    /// <param name="core">The core.</param>
    /// <returns>The 17-number observation.</returns>
    private static double[] Obs(CheetahCore core) => core.qpos.Skip(1).Concat(core.qvel).ToArray();

    /// <summary>
    ///     HalfCheetah-v5's step results after the physics steps: <c>x_velocity = (x_after - x_before)/dt</c>, reward
    ///     <c>1.0*x_velocity - ctrl_cost</c> where <c>ctrl_cost = 0.1*sum(action²)</c> is float32 under NEP 50.
    /// </summary>
    /// <param name="env">The checker.</param>
    /// <param name="core">The core after the step.</param>
    /// <param name="action">The action (float32 values).</param>
    /// <param name="xBefore">qpos[0] before the step.</param>
    /// <param name="dt">The env time step.</param>
    /// <param name="t">The step index.</param>
    /// <param name="steps">The episode length (TimeLimit truncates at its end).</param>
    /// <param name="envRecords">The recorded env results.</param>
    private static void EnvCheck(Checker env, CheetahCore core, double[] action, double xBefore, double dt, int t, int steps, List<double[]> envRecords) {
        double xAfter = core.qpos[0];
        double xVelocity = (xAfter - xBefore) / dt;
        double forward = 1.0 * xVelocity;
        float ctrlCost = 0.1f * NumpySumSquares(action);
        double reward = forward - ctrlCost;
        Compare(env, "obs", Obs(core), envRecords[1 + 2 * t], t);
        Compare(env, "reward, terminated, truncated, info",
                [reward, 0.0, t + 1 == steps ? 1.0 : 0.0, xAfter, xVelocity, forward, -ctrlCost], envRecords[2 + 2 * t], t);
    }

    /// <summary>
    ///     <c>np.sum(np.square(a))</c> of a float32 array of fewer than 9 entries: float32 squares, then NumPy's add
    ///     reduction, which starts from the identity 0 and adds the entries in order (its pairwise sum is a plain loop
    ///     below 8 entries).
    /// </summary>
    /// <param name="a">The float32 values (widened).</param>
    /// <returns>The float32 sum.</returns>
    private static float NumpySumSquares(double[] a) {
        float res = 0;
        foreach (double v in a) {
            float f = (float)v;
            res += f * f;
        }
        return 0f + res;
    }

    /// <summary>Reads one physics step's records and compares each with the core's arrays.</summary>
    /// <param name="c">The checker.</param>
    /// <param name="core">The core after the step.</param>
    /// <param name="trace">The trace.</param>
    /// <param name="rec">The cursor, advanced past the step's records.</param>
    /// <param name="step">The physics step index, for the report.</param>
    private static void CompareStep(Checker c, CheetahCore core, List<double[]> trace, ref int rec, int step) {
        int nv = core.qvel.Length, nefc = core.nefc, ncon = core.ncon;
        double[][] state = [core.qpos, core.qvel, core.qacc, core.qacc_warmstart, core.qacc_smooth, core.qfrc_constraint,
                            core.qfrc_passive, core.qfrc_bias, core.qfrc_actuator, core.actuator_force];
        c.BeginStep();
        for (int i = 0; i < StateNames.Length; i++)
            Compare(c, StateNames[i], state[i], trace[rec++], step);
        Compare(c, "time, ncon, nefc, niter, nisland",
                [core.time, ncon, nefc, core.nisland > 0 ? core.solver_niter : 0, core.nisland], trace[rec++], step);
        Compare(c, "contact.dist", core.con_dist.AsSpan(0, ncon).ToArray(), trace[rec++], step);
        Compare(c, "contact.geom", Enumerable.Range(0, ncon).SelectMany(i => new double[] { core.con_geom1[i], core.con_geom2[i] }).ToArray(), trace[rec++], step);
        Compare(c, "contact.pos", core.con_pos.AsSpan(0, 3 * ncon).ToArray(), trace[rec++], step);
        Compare(c, "contact.frame", core.con_frame.AsSpan(0, 9 * ncon).ToArray(), trace[rec++], step);
        Compare(c, "efc_J", core.efc_J.AsSpan(0, nefc * nv).ToArray(), trace[rec++], step);
        double[] I(int[] v) => v.Take(nefc).Select(x => (double)x).ToArray();
        double[] E(double[] v, int stride = 1) => v.AsSpan(0, stride * nefc).ToArray();
        double[][] efc = [I(core.efc_type), I(core.efc_id), E(core.efc_pos), E(core.efc_margin), E(core.efc_frictionloss),
                          E(core.efc_diagA), E(core.efc_KBIP, 4), E(core.efc_D), E(core.efc_R), E(core.efc_vel),
                          E(core.efc_aref), E(core.efc_b), E(core.efc_force), I(core.efc_state)];
        for (int i = 0; i < EfcNames.Length; i++)
            Compare(c, EfcNames[i], efc[i], trace[rec++], step);
        double[][] debug = [core.xpos, core.xquat, core.xmat, core.xipos, core.ximat, core.geom_xpos, core.geom_xmat,
                            core.subtree_com, core.cinert, core.cdof, core.crb, core.M, core.qLD, core.qLDiagInv, core.cvel, core.cdof_dot];
        for (int i = 0; i < DebugNames.Length; i++)
            Compare(c, DebugNames[i], debug[i], trace[rec++], step);
        c.EndStep();
    }

    /// <summary>Compares one array bit for bit and records the outcome.</summary>
    /// <param name="c">The checker.</param>
    /// <param name="name">The array's name.</param>
    /// <param name="ours">The core's values.</param>
    /// <param name="theirs">The recorded values.</param>
    /// <param name="step">The step index.</param>
    private static void Compare(Checker c, string name, double[] ours, double[] theirs, int step) => c.Check(name, ours, theirs, step);

    /// <summary>Whether two arrays have the same length and the same bits in every entry.</summary>
    /// <param name="a">The first array.</param>
    /// <param name="b">The second array.</param>
    /// <returns><see langword="true"/> when identical.</returns>
    public static bool SameBits(double[] a, double[] b) =>
        a.Length == b.Length && a.Zip(b).All(p => BitConverter.DoubleToInt64Bits(p.First) == BitConverter.DoubleToInt64Bits(p.Second));

    /// <summary>Times the core alone over the episode's 5,000 physics steps (best of five runs).</summary>
    /// <param name="model">The model.</param>
    /// <param name="actions">The actions.</param>
    private static void Benchmark(CheetahModel model, double[][] actions) {
        double best = double.MaxValue;
        for (int run = 0; run < 5; run++) {
            var core = new CheetahCore(model);
            Reset(core, model, [new double[model.nq], new double[model.nv], new double[model.nv], [0.0]], 42, false);
            var sw = Stopwatch.StartNew();
            foreach (double[] a in actions) {
                Array.Copy(a, core.ctrl, model.nu);
                for (int k = 0; k < model.frame_skip; k++)
                    core.Step();
            }
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMicroseconds / (actions.Length * model.frame_skip));
        }
        Console.WriteLine($"speed: {best:F2} us per physics step (managed core, best of 5 runs of the last episode)");
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
