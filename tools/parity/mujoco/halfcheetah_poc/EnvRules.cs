using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NumSharp;
using Gym.MuJoCo;

namespace MujocoPoc;

/// <summary>
///     An env's rules over the core, as Gymnasium's v4 and v5 classes have them: the reset (noise drawn from
///     <c>Generator(PCG64(SeedSequence(seed)))</c> in the env's order), the action bounds, the observation, and per step the
///     reward, the termination and the info entries, with the TimeLimit's truncation. Each reproduces NumPy's arithmetic
///     where it decides bits.
/// </summary>
/// <remarks>
///     <para>
///         The arithmetic the rules rely on was measured against NumPy 2.4.2 on win-x64, each case bit for bit over many
///         random trials:
///         <list type="bullet">
///             <item>float32 costs stay float32 (NEP 50: a Python float is weak).</item>
///             <item>
///                 <c>np.sum</c> is the identity 0 plus NumPy's pairwise sum: a plain loop below 8 entries, and from 8 on,
///                 8 accumulators combined as a tree.
///             </item>
///             <item>
///                 <c>einsum("b,bj->j")</c> sums each column left to right from 0, and <c>np.linalg.norm</c> of 2 or 3
///                 values is the square root of their squares added left to right.
///             </item>
///             <item>
///                 <c>np.sin</c> and <c>np.cos</c> on float64 arrays are the C runtime's, as <see cref="Math.Sin"/> is.
///             </item>
///             <item>
///                 A float64 scalar's <c>x**2</c> is the C runtime's <c>pow</c> (<see cref="Math.Pow"/>), which differs
///                 from <c>x*x</c> in about 0.05% of inputs.
///             </item>
///         </list>
///     </para>
///     <para>
///         Body positions are read from <see cref="MjData.xpos"/> as MuJoCo publishes them, so after an RK4 step they
///         are its last stage's, as Gymnasium's env code sees them. An env keeps per-episode state between
///         <see cref="Before"/> and <see cref="After"/>, so one rules object serves one episode at a time.
///     </para>
/// </remarks>
internal abstract class EnvRules {
    /// <summary>The model the rules read.</summary>
    protected readonly MjModel M;

    /// <summary>The control step, <c>timestep * frame_skip</c> (Gymnasium's <c>dt</c>).</summary>
    protected readonly double Dt;

    /// <summary>Whether an unhealthy state terminates the episode (<c>terminate_when_unhealthy</c>).</summary>
    protected readonly bool Terminate;

    /// <summary>The action space's lower bounds (the control ranges rounded to float32).</summary>
    public readonly double[] ActionLow;

    /// <summary>The action space's upper bounds.</summary>
    public readonly double[] ActionHigh;

    /// <summary>The TimeLimit's <c>max_episode_steps</c>: the step it truncates at.</summary>
    private readonly int maxEpisodeSteps;

    /// <summary>Sets up the rules for a model and a recorded run.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run, for the termination flag and the episode limit.</param>
    protected EnvRules(MjModel model, RunInfo run) {
        M = model;
        Terminate = run.Terminate;
        maxEpisodeSteps = run.MaxEpisodeSteps;
        Dt = model.timestep * run.FrameSkip;
        ActionLow = Enumerable.Range(0, model.nu).Select(i => (double)(float)model.actuator_ctrlrange[2 * i]).ToArray();
        ActionHigh = Enumerable.Range(0, model.nu).Select(i => (double)(float)model.actuator_ctrlrange[2 * i + 1]).ToArray();
    }

    /// <summary>
    ///     The rules of a run's env, or <see langword="null"/> for an env the checker has no rules for, which it then checks
    ///     physics only (from the recorded reset and actions, without the env layer).
    /// </summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run: its env id, termination flag and episode limit.</param>
    /// <returns>The rules, or <see langword="null"/>.</returns>
    /// <exception cref="KeyNotFoundException">A body the env reads by name is missing from the model's names.</exception>
    public static EnvRules? TryFor(MjModel model, RunInfo run) => run.Env switch {
        "HalfCheetah-v5" or "HalfCheetah-v4" => new HalfCheetahRules(model, run),
        "Hopper-v5" => HopperWalkerRules.Hopper(model, run, v4: false),
        "Hopper-v4" => HopperWalkerRules.Hopper(model, run, v4: true),
        "Walker2d-v5" => HopperWalkerRules.Walker2d(model, run, v4: false),
        "Walker2d-v4" => HopperWalkerRules.Walker2d(model, run, v4: true),
        "Humanoid-v5" => new HumanoidRules(model, run, v4: false),
        "Humanoid-v4" => new HumanoidRules(model, run, v4: true),
        "HumanoidStandup-v5" => new HumanoidStandupRules(model, run, v4: false),
        "HumanoidStandup-v4" => new HumanoidStandupRules(model, run, v4: true),
        "Ant-v5" => new AntRules(model, run, v4: false),
        "Ant-v4" => new AntRules(model, run, v4: true),
        "InvertedPendulum-v5" => new InvertedPendulumRules(model, run, v4: false),
        "InvertedPendulum-v4" => new InvertedPendulumRules(model, run, v4: true),
        "InvertedDoublePendulum-v5" => new InvertedDoublePendulumRules(model, run, v4: false),
        "InvertedDoublePendulum-v4" => new InvertedDoublePendulumRules(model, run, v4: true),
        "Reacher-v5" => new ReacherRules(model, run, v4: false),
        "Reacher-v4" => new ReacherRules(model, run, v4: true),
        "Swimmer-v5" or "Swimmer-v4" => new SwimmerRules(model, run),
        "Pusher-v5" => new PusherRules(model, run),
        _ => null,
    };

    /// <summary>
    ///     Gymnasium's reset after <c>mj_resetData</c>: the env's <c>reset_model</c> state from a generator seeded with
    ///     <paramref name="seed"/>, the optional flip, then <c>mj_forward</c> (<c>set_state</c>).
    /// </summary>
    /// <param name="core">A new data, or one just reset by <c>mj_resetData</c>, whose state is set.</param>
    /// <param name="seed">The reset seed.</param>
    /// <param name="flip">Whether to turn the model upside down (<c>qpos[2] += pi</c>, hinge-rooted models only).</param>
    public void Reset(MjData core, int seed, bool flip) {
        var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
        (double[] qpos, double[] qvel) = ResetState(rng);
        Array.Copy(qpos, core.qpos, M.nq);
        Array.Copy(qvel, core.qvel, M.nv);
        if (flip)
            core.qpos[2] += Math.PI;
        core.Forward();
    }

    /// <summary>The env's <c>reset_model</c> state, its draws in the env's order.</summary>
    /// <param name="rng">The seeded generator.</param>
    /// <returns>The positions and velocities <c>set_state</c> receives.</returns>
    protected abstract (double[] Qpos, double[] Qvel) ResetState(Generator rng);

    /// <summary>
    ///     The common reset: <c>init_qpos + uniform(-scale, scale, nq)</c>, then <c>init_qvel + uniform(-scale, scale, nv)</c>
    ///     (<c>init_qvel</c> is 0, and adding it turns a -0 into +0).
    /// </summary>
    /// <param name="rng">The seeded generator.</param>
    /// <param name="scale">The noise scale.</param>
    /// <returns>The reset state.</returns>
    protected (double[] Qpos, double[] Qvel) UniformReset(Generator rng, double scale) {
        double[] dq = Uniform(rng, -scale, scale, M.nq), dv = Uniform(rng, -scale, scale, M.nv);
        return (M.qpos0.Zip(dq, (q, d) => q + d).ToArray(), dv.Select(v => 0.0 + v).ToArray());
    }

    /// <summary>
    ///     The reset of HalfCheetah, Ant and InvertedDoublePendulum: uniform position noise, then velocities
    ///     <c>init_qvel + scale * standard_normal(nv)</c>.
    /// </summary>
    /// <param name="rng">The seeded generator.</param>
    /// <param name="scale">The noise scale.</param>
    /// <returns>The reset state.</returns>
    protected (double[] Qpos, double[] Qvel) NormalReset(Generator rng, double scale) {
        double[] dq = Uniform(rng, -scale, scale, M.nq);
        using NDArray z = rng.standard_normal(new Shape(M.nv));
        return (M.qpos0.Zip(dq, (q, d) => q + d).ToArray(), z.ToArray<double>().Select(v => 0.0 + scale * v).ToArray());
    }

    /// <summary>NumPy's <c>uniform(low, high, n)</c> on the env's generator.</summary>
    /// <param name="rng">The generator.</param>
    /// <param name="low">The lower bound.</param>
    /// <param name="high">The upper bound.</param>
    /// <param name="n">How many values.</param>
    /// <returns>The values.</returns>
    protected static double[] Uniform(Generator rng, double low, double high, int n) {
        using NDArray u = rng.uniform(low, high, new Shape(n));
        return u.ToArray<double>();
    }

    /// <summary>The observation.</summary>
    /// <param name="core">The core.</param>
    /// <returns>The observation.</returns>
    public abstract double[] Obs(MjData core);

    /// <summary>Records what the step's reward needs from before the physics steps (the default needs nothing).</summary>
    /// <param name="core">The core before the step.</param>
    /// <param name="action">The step's action (float32 values), which Reacher-v4 prices before stepping.</param>
    public virtual void Before(MjData core, double[] action) { }

    /// <summary>The step's results after the physics steps and <c>mj_rnePostConstraint</c>.</summary>
    /// <param name="core">The core after the step.</param>
    /// <param name="action">The action (float32 values).</param>
    /// <param name="t">The step index within the episode.</param>
    /// <param name="infoKeys">The info entries in env.bin's order.</param>
    /// <returns>The observation, and [reward, terminated, truncated, the info values].</returns>
    /// <exception cref="KeyNotFoundException">An info key the rules do not produce.</exception>
    public (double[] Obs, double[] Row) After(MjData core, double[] action, int t, (string Key, int Size)[] infoKeys) {
        (double reward, bool terminated, Dictionary<string, double[]> info) = Results(core, action);
        var row = new List<double> { reward, terminated ? 1 : 0, t + 1 >= maxEpisodeSteps ? 1 : 0 };
        foreach ((string key, _) in infoKeys)
            row.AddRange(info[key]);
        return (Obs(core), row.ToArray());
    }

    /// <summary>
    ///     The reward, the termination and the info entries. A rules class may return more entries than its version's
    ///     info holds (v4 and v5 name some differently); <see cref="After"/> takes the recorded keys.
    /// </summary>
    /// <param name="core">The core after the step.</param>
    /// <param name="action">The action.</param>
    /// <returns>The results.</returns>
    protected abstract (double Reward, bool Terminated, Dictionary<string, double[]> Info) Results(MjData core, double[] action);

    /// <summary>The index of a body by its name.</summary>
    /// <param name="name">The body name.</param>
    /// <returns>The body index.</returns>
    /// <exception cref="KeyNotFoundException">The model has no such body, or its dump has no names.</exception>
    protected int BodyId(string name) {
        int b = Array.IndexOf(M.body_names, name);
        return b >= 0 ? b : throw new KeyNotFoundException($"no body named {name} in the model's names");
    }

    /// <summary><c>np.clip</c> of one value: below <paramref name="lo"/> it is lo, above <paramref name="hi"/> hi; NaN stays.</summary>
    /// <param name="x">The value.</param>
    /// <param name="lo">The lower bound.</param>
    /// <param name="hi">The upper bound.</param>
    /// <returns>The clipped value.</returns>
    protected static double Clip(double x, double lo, double hi) => x < lo ? lo : x > hi ? hi : x;

    /// <summary><c>np.linalg.norm</c> of 2 float64 values: <c>sqrt(x*x + y*y)</c>.</summary>
    /// <param name="x">The first value.</param>
    /// <param name="y">The second value.</param>
    /// <returns>The norm.</returns>
    protected static double Norm2(double x, double y) => Math.Sqrt(x * x + y * y);

    /// <summary><c>np.linalg.norm</c> of 3 float64 values: <c>sqrt((x*x + y*y) + z*z)</c>.</summary>
    /// <param name="x">The first value.</param>
    /// <param name="y">The second value.</param>
    /// <param name="z">The third value.</param>
    /// <returns>The norm.</returns>
    protected static double Norm3(double x, double y, double z) => Math.Sqrt(x * x + y * y + z * z);

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

    /// <summary><c>np.sum(np.square(a))</c> of a float64 array (pairwise beyond 7 entries).</summary>
    /// <param name="a">The values.</param>
    /// <returns>The sum of squares.</returns>
    protected static double SumSquaresF64(IEnumerable<double> a) => SumF64(a.Select(v => v * v).ToArray());

    /// <summary>
    ///     The body inertias, velocities, actuator forces and external forces the humanoids observe: v5 drops the world
    ///     body's rows and the root's 6 actuator dofs, v4 keeps every row.
    /// </summary>
    /// <param name="core">The core.</param>
    /// <param name="v4">Whether the full v4 arrays are observed.</param>
    /// <returns>The observation's tail after the positions and velocities.</returns>
    protected static IEnumerable<double> HumanoidTail(MjData core, bool v4) => v4
        ? core.cinert.Concat(core.cvel).Concat(core.qfrc_actuator).Concat(core.cfrc_ext)
        : core.cinert.Skip(10).Concat(core.cvel.Skip(6)).Concat(core.qfrc_actuator.Skip(6)).Concat(core.cfrc_ext.Skip(6));
}

/// <summary>HalfCheetah-v5 and -v4: forward velocity minus a float32 control cost; never terminates.</summary>
internal sealed class HalfCheetahRules : EnvRules {
    private double xBefore;

    /// <summary>Sets up the rules (v4 and v5 differ only in an info name).</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    public HalfCheetahRules(MjModel model, RunInfo run) : base(model, run) { }

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => NormalReset(rng, 0.1);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) => core.qpos.Skip(1).Concat(core.qvel).ToArray();

    /// <inheritdoc/>
    public override void Before(MjData core, double[] action) => xBefore = core.qpos[0];

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double xAfter = core.qpos[0];
        double xVelocity = (xAfter - xBefore) / Dt;
        double forward = 1.0 * xVelocity;
        float ctrlCost = 0.1f * SumSquaresF32(action);
        return (forward - ctrlCost, false, new Dictionary<string, double[]> {
            ["x_position"] = [xAfter], ["x_velocity"] = [xVelocity], ["reward_forward"] = [forward], ["reward_run"] = [forward],
            ["reward_ctrl"] = [-ctrlCost],
        });
    }
}

/// <summary>
///     Hopper and Walker2d, v5 and v4: forward velocity plus a healthy reward minus a float32 control cost; an unhealthy
///     height or angle (and for Hopper a state beyond ±100) terminates; velocities are clipped to ±10 in the observation.
/// </summary>
/// <remarks>
///     v4 pays the healthy reward whenever <c>terminate_when_unhealthy</c> is set (<c>float(is_healthy or terminate)</c>),
///     v5 only while healthy; the arithmetic is otherwise the same.
/// </remarks>
internal sealed class HopperWalkerRules : EnvRules {
    private readonly bool v4, checkState;
    private readonly double zLow, zHigh, angleLow, angleHigh;
    private double xBefore;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is the v4.</param>
    /// <param name="checkState">Whether the health needs the state beyond x and z within ±100 (Hopper).</param>
    /// <param name="zLow">The exclusive lower bound of a healthy height.</param>
    /// <param name="zHigh">The exclusive upper bound of a healthy height.</param>
    /// <param name="angleLow">The exclusive lower bound of a healthy torso angle.</param>
    /// <param name="angleHigh">The exclusive upper bound of a healthy torso angle.</param>
    private HopperWalkerRules(MjModel model, RunInfo run, bool v4, bool checkState, double zLow, double zHigh, double angleLow,
                              double angleHigh) : base(model, run) {
        this.v4 = v4;
        this.checkState = checkState;
        this.zLow = zLow;
        this.zHigh = zHigh;
        this.angleLow = angleLow;
        this.angleHigh = angleHigh;
    }

    /// <summary>Hopper's rules: height in (0.7, inf), angle in (-0.2, 0.2), state within ±100.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is Hopper-v4.</param>
    /// <returns>The rules.</returns>
    public static HopperWalkerRules Hopper(MjModel model, RunInfo run, bool v4) =>
        new(model, run, v4, true, 0.7, double.PositiveInfinity, -0.2, 0.2);

    /// <summary>Walker2d's rules: height in (0.8, 2.0), angle in (-1, 1).</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is Walker2d-v4.</param>
    /// <returns>The rules.</returns>
    public static HopperWalkerRules Walker2d(MjModel model, RunInfo run, bool v4) =>
        new(model, run, v4, false, 0.8, 2.0, -1.0, 1.0);

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => UniformReset(rng, 5e-3);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) =>
        core.qpos.Skip(1).Concat(core.qvel.Select(v => Clip(v, -10, 10))).ToArray();

    /// <inheritdoc/>
    public override void Before(MjData core, double[] action) => xBefore = core.qpos[0];

    /// <summary>Whether the body is healthy.</summary>
    /// <param name="core">The core.</param>
    /// <returns>The health.</returns>
    private bool Healthy(MjData core) {
        bool state = !checkState || core.qpos.Skip(2).Concat(core.qvel).All(s => -100.0 < s && s < 100.0);
        double z = core.qpos[1], angle = core.qpos[2];
        return state && zLow < z && z < zHigh && angleLow < angle && angle < angleHigh;
    }

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double xAfter = core.qpos[0];
        double xVelocity = (xAfter - xBefore) / Dt;
        double forward = 1.0 * xVelocity;
        bool healthy = Healthy(core);
        double healthyReward = (v4 ? healthy || Terminate : healthy) ? 1.0 : 0.0;
        double rewards = forward + healthyReward;
        float ctrlCost = 1e-3f * SumSquaresF32(action);
        return (rewards - ctrlCost, !healthy && Terminate, new Dictionary<string, double[]> {
            ["x_position"] = [xAfter], ["z_distance_from_origin"] = [core.qpos[1] - M.qpos0[1]], ["x_velocity"] = [xVelocity],
            ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost], ["reward_survive"] = [healthyReward],
        });
    }
}

/// <summary>
///     Humanoid-v5 and -v4: the mass centre's forward velocity (weight 1.25) plus a healthy reward, minus float64 costs
///     (v5: control and contact; v4: control only); a torso height outside (1, 2) terminates.
/// </summary>
internal sealed class HumanoidRules : EnvRules {
    private readonly bool v4;
    private double[] xyBefore = [0, 0];

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is Humanoid-v4 (full arrays observed, no contact cost, mass-centre info).</param>
    public HumanoidRules(MjModel model, RunInfo run, bool v4) : base(model, run) => this.v4 = v4;

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => UniformReset(rng, 1e-2);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) =>
        core.qpos.Skip(2).Concat(core.qvel).Concat(HumanoidTail(core, v4)).ToArray();

    /// <summary>
    ///     <c>mass_center</c>: <c>einsum("b,bj->j", body_mass, xipos)</c> (each column summed in body order from 0) over
    ///     <c>body_mass.sum()</c> (NumPy's pairwise sum), the x and y components.
    /// </summary>
    /// <param name="core">The core, whose last forward pass set xipos.</param>
    /// <returns>The mass centre's x and y.</returns>
    private double[] MassCenter(MjData core) {
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
    public override void Before(MjData core, double[] action) => xyBefore = MassCenter(core);

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double[] xyAfter = MassCenter(core);
        double xVelocity = (xyAfter[0] - xyBefore[0]) / Dt, yVelocity = (xyAfter[1] - xyBefore[1]) / Dt;
        double forward = 1.25 * xVelocity;
        double z = core.qpos[2];
        bool healthy = 1.0 < z && z < 2.0;
        double healthyReward = (v4 ? healthy || Terminate : healthy) ? 5.0 : 0.0;
        double rewards = forward + healthyReward;
        // float64 costs: data.ctrl holds the action widened to float64; cfrc_ext's 84 squares
        double ctrlCost = 0.1 * SumSquaresF64(core.ctrl);
        if (v4) {
            return (rewards - ctrlCost, !healthy && Terminate, new Dictionary<string, double[]> {
                ["reward_linvel"] = [forward], ["reward_quadctrl"] = [-ctrlCost], ["reward_alive"] = [healthyReward],
                ["x_position"] = [xyAfter[0]], ["y_position"] = [xyAfter[1]],
                ["distance_from_origin"] = [Norm2(xyAfter[0], xyAfter[1])],
                ["x_velocity"] = [xVelocity], ["y_velocity"] = [yVelocity], ["forward_reward"] = [forward],
            });
        }
        double contactCost = 5e-7 * SumSquaresF64(core.cfrc_ext);
        contactCost = Math.Min(Math.Max(contactCost, double.NegativeInfinity), 10.0);
        double costs = ctrlCost + contactCost;
        double x = core.qpos[0], y = core.qpos[1];
        return (rewards - costs, !healthy && Terminate, new Dictionary<string, double[]> {
            ["x_position"] = [x], ["y_position"] = [y],
            ["tendon_length"] = (double[])core.ten_length.Clone(), ["tendon_velocity"] = (double[])core.ten_velocity.Clone(),
            ["distance_from_origin"] = [Norm2(x, y)],
            ["x_velocity"] = [xVelocity], ["y_velocity"] = [yVelocity],
            ["reward_survive"] = [healthyReward], ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost],
            ["reward_contact"] = [-contactCost],
        });
    }
}

/// <summary>
///     HumanoidStandup-v5 and -v4: the torso height over the time step, minus float64 control and impact costs, plus 1;
///     never terminates.
/// </summary>
/// <remarks>v5 caps the impact cost with <c>np.clip</c>, v4 with Python's <c>min(cost, 10)</c>; both give the same double.</remarks>
internal sealed class HumanoidStandupRules : EnvRules {
    private readonly bool v4;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is HumanoidStandup-v4 (full arrays observed, three info entries).</param>
    public HumanoidStandupRules(MjModel model, RunInfo run, bool v4) : base(model, run) => this.v4 = v4;

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => UniformReset(rng, 1e-2);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) =>
        core.qpos.Skip(2).Concat(core.qvel).Concat(HumanoidTail(core, v4)).ToArray();

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double posAfter = core.qpos[2];
        double uphCost = (posAfter - 0) / M.timestep;
        double quadCtrlCost = 0.1 * SumSquaresF64(core.ctrl);
        double quadImpactCost = 0.5e-6 * SumSquaresF64(core.cfrc_ext);
        // np.clip(x, -inf, 10) and min(x, 10) agree: x unless it exceeds 10 (NaN stays NaN)
        quadImpactCost = 10.0 < quadImpactCost ? 10.0 : quadImpactCost;
        double reward = uphCost - quadCtrlCost - quadImpactCost + 1;
        return (reward, false, new Dictionary<string, double[]> {
            ["x_position"] = [core.qpos[0]], ["y_position"] = [core.qpos[1]],
            ["z_distance_from_origin"] = [core.qpos[2] - M.qpos0[2]],
            ["tendon_length"] = (double[])core.ten_length.Clone(), ["tendon_velocity"] = (double[])core.ten_velocity.Clone(),
            ["reward_linup"] = [uphCost], ["reward_quadctrl"] = [-quadCtrlCost], ["reward_impact"] = [-quadImpactCost],
        });
    }
}

/// <summary>
///     Ant-v5 and -v4: the torso's forward velocity plus a healthy reward, minus a float32 control cost and (v5) a contact
///     cost on the clipped external forces; a non-finite state or a height outside [0.2, 1.0] is unhealthy.
/// </summary>
internal sealed class AntRules : EnvRules {
    private readonly bool v4;
    private readonly int torso;
    private double xBefore, yBefore;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is Ant-v4 (no contact forces observed or priced, torso-position info).</param>
    public AntRules(MjModel model, RunInfo run, bool v4) : base(model, run) {
        this.v4 = v4;
        torso = BodyId("torso");
    }

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => NormalReset(rng, 0.1);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) {
        IEnumerable<double> obs = core.qpos.Skip(2).Concat(core.qvel);
        return (v4 ? obs : obs.Concat(core.cfrc_ext.Skip(6).Select(f => Clip(f, -1.0, 1.0)))).ToArray();
    }

    /// <inheritdoc/>
    public override void Before(MjData core, double[] action) {
        xBefore = core.xpos[3 * torso];
        yBefore = core.xpos[3 * torso + 1];
    }

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double xAfter = core.xpos[3 * torso], yAfter = core.xpos[3 * torso + 1];
        double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;
        double z = core.qpos[2];
        bool healthy = core.qpos.Concat(core.qvel).All(double.IsFinite) && 0.2 <= z && z <= 1.0;
        double healthyReward = (v4 ? healthy || Terminate : healthy) ? 1.0 : 0.0;
        double forward = xVelocity;
        double rewards = forward + healthyReward;
        float ctrlCost = 0.5f * SumSquaresF32(action);
        if (v4) {
            return (rewards - ctrlCost, !healthy && Terminate, new Dictionary<string, double[]> {
                ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost], ["reward_survive"] = [healthyReward],
                ["x_position"] = [xAfter], ["y_position"] = [yAfter], ["distance_from_origin"] = [Norm2(xAfter, yAfter)],
                ["x_velocity"] = [xVelocity], ["y_velocity"] = [yVelocity], ["forward_reward"] = [forward],
            });
        }
        double contactCost = 5e-4 * SumSquaresF64(core.cfrc_ext.Select(f => Clip(f, -1.0, 1.0)));
        double costs = ctrlCost + contactCost;
        return (rewards - costs, !healthy && Terminate, new Dictionary<string, double[]> {
            ["x_position"] = [core.qpos[0]], ["y_position"] = [core.qpos[1]],
            ["distance_from_origin"] = [Norm2(core.qpos[0], core.qpos[1])],
            ["x_velocity"] = [xVelocity], ["y_velocity"] = [yVelocity],
            ["reward_forward"] = [forward], ["reward_ctrl"] = [-ctrlCost], ["reward_contact"] = [-contactCost],
            ["reward_survive"] = [healthyReward],
        });
    }
}

/// <summary>
///     InvertedPendulum-v5 and -v4: 1 per step while the pole stays within 0.2 rad (v5: 0 at the terminating step, v4: 1
///     always).
/// </summary>
internal sealed class InvertedPendulumRules : EnvRules {
    private readonly bool v4;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is InvertedPendulum-v4.</param>
    public InvertedPendulumRules(MjModel model, RunInfo run, bool v4) : base(model, run) => this.v4 = v4;

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => UniformReset(rng, 0.01);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) => core.qpos.Concat(core.qvel).ToArray();

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double[] obs = Obs(core);
        bool terminated = !obs.All(double.IsFinite) || Math.Abs(obs[1]) > 0.2;
        double reward = v4 ? 1.0 : terminated ? 0 : 1;
        return (reward, terminated, new Dictionary<string, double[]> { ["reward_survive"] = [reward] });
    }
}

/// <summary>
///     InvertedDoublePendulum-v5 and -v4: 10 per step while the tip stays above 1, minus distance and velocity penalties
///     squared with the C runtime's <c>pow</c> as NumPy does.
/// </summary>
internal sealed class InvertedDoublePendulumRules : EnvRules {
    private readonly bool v4;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is InvertedDoublePendulum-v4 (all constraint forces observed, no info).</param>
    public InvertedDoublePendulumRules(MjModel model, RunInfo run, bool v4) : base(model, run) => this.v4 = v4;

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => NormalReset(rng, 0.1);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) =>
        new[] { core.qpos[0] }.Concat(core.qpos.Skip(1).Select(Math.Sin)).Concat(core.qpos.Skip(1).Select(Math.Cos))
            .Concat(core.qvel.Select(v => Clip(v, -10, 10)))
            .Concat(core.qfrc_constraint.Take(v4 ? M.nv : 1).Select(f => Clip(f, -10, 10))).ToArray();

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double x = core.site_xpos[0], y = core.site_xpos[2];
        bool terminated = y <= 1;
        double v1 = core.qvel[1], v2 = core.qvel[2];
        // NumPy's float64 ** 2 is the C runtime's pow, not a multiplication
        double distPenalty = 0.01 * Math.Pow(x, 2) + Math.Pow(y - 2, 2);
        double velPenalty = 1e-3 * Math.Pow(v1, 2) + 5e-3 * Math.Pow(v2, 2);
        double aliveBonus = v4 ? 10 : 10.0 * (terminated ? 0 : 1);
        double reward = aliveBonus - distPenalty - velPenalty;
        return (reward, terminated, new Dictionary<string, double[]> {
            ["reward_survive"] = [aliveBonus], ["distance_penalty"] = [-distPenalty], ["velocity_penalty"] = [-velPenalty],
        });
    }
}

/// <summary>
///     Reacher-v5 and -v4: minus the fingertip's distance to the target, minus a float32 control cost. v4 prices the step
///     from the state before it; v5 from the state after.
/// </summary>
internal sealed class ReacherRules : EnvRules {
    private readonly bool v4;
    private readonly int fingertip, target;
    private double v4Dist;
    private float v4Ctrl;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    /// <param name="v4">Whether the env is Reacher-v4.</param>
    public ReacherRules(MjModel model, RunInfo run, bool v4) : base(model, run) {
        this.v4 = v4;
        fingertip = BodyId("fingertip");
        target = BodyId("target");
    }

    /// <summary>
    ///     Reacher's reset: arm noise <c>uniform(-0.1, 0.1, nq) + init_qpos</c>, a goal drawn in the disc of radius 0.2 by
    ///     rejection (<c>uniform(-0.2, 0.2, 2)</c> until its norm is below 0.2) into the target's two coordinates, then
    ///     velocity noise <c>uniform(-0.005, 0.005, nv)</c> with the target's zeroed.
    /// </summary>
    /// <param name="rng">The seeded generator.</param>
    /// <returns>The reset state.</returns>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) {
        double[] qpos = Uniform(rng, -0.1, 0.1, M.nq).Zip(M.qpos0, (u, q) => u + q).ToArray();
        double[] goal;
        do {
            goal = Uniform(rng, -0.2, 0.2, 2);
        } while (!(Norm2(goal[0], goal[1]) < 0.2));
        qpos[M.nq - 2] = goal[0];
        qpos[M.nq - 1] = goal[1];
        double[] qvel = Uniform(rng, -0.005, 0.005, M.nv).Select(v => 0.0 + v).ToArray();
        qvel[M.nv - 2] = 0;
        qvel[M.nv - 1] = 0;
        return (qpos, qvel);
    }

    /// <summary>The fingertip's position minus the target's, from the published body frames.</summary>
    /// <param name="core">The core.</param>
    /// <returns>The 3 components.</returns>
    private double[] Vec(MjData core) =>
        Enumerable.Range(0, 3).Select(k => core.xpos[3 * fingertip + k] - core.xpos[3 * target + k]).ToArray();

    /// <inheritdoc/>
    public override double[] Obs(MjData core) =>
        core.qpos.Take(2).Select(Math.Cos).Concat(core.qpos.Take(2).Select(Math.Sin)).Concat(core.qpos.Skip(2))
            .Concat(core.qvel.Take(2)).Concat(Vec(core).Take(v4 ? 3 : 2)).ToArray();

    /// <inheritdoc/>
    public override void Before(MjData core, double[] action) {
        if (!v4)
            return;
        double[] vec = Vec(core);
        v4Dist = -Norm3(vec[0], vec[1], vec[2]);
        v4Ctrl = -SumSquaresF32(action);
    }

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double rewardDist;
        float rewardCtrl;
        if (v4) {
            rewardDist = v4Dist;
            rewardCtrl = v4Ctrl;
        } else {
            double[] vec = Vec(core);
            rewardDist = -Norm3(vec[0], vec[1], vec[2]) * 1;
            rewardCtrl = -SumSquaresF32(action) * 1f;
        }
        return (rewardDist + rewardCtrl, false, new Dictionary<string, double[]> {
            ["reward_dist"] = [rewardDist], ["reward_ctrl"] = [rewardCtrl],
        });
    }
}

/// <summary>Swimmer-v5 and -v4: forward velocity minus a float32 control cost; never terminates.</summary>
internal sealed class SwimmerRules : EnvRules {
    private double xBefore, yBefore;

    /// <summary>Sets up the rules (v4 and v5 differ only in info names).</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    public SwimmerRules(MjModel model, RunInfo run) : base(model, run) { }

    /// <inheritdoc/>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) => UniformReset(rng, 0.1);

    /// <inheritdoc/>
    public override double[] Obs(MjData core) => core.qpos.Skip(2).Concat(core.qvel).ToArray();

    /// <inheritdoc/>
    public override void Before(MjData core, double[] action) {
        xBefore = core.qpos[0];
        yBefore = core.qpos[1];
    }

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double xAfter = core.qpos[0], yAfter = core.qpos[1];
        double xVelocity = (xAfter - xBefore) / Dt, yVelocity = (yAfter - yBefore) / Dt;
        double forward = 1.0 * xVelocity;
        float ctrlCost = 1e-4f * SumSquaresF32(action);
        return (forward - ctrlCost, false, new Dictionary<string, double[]> {
            ["x_position"] = [xAfter], ["y_position"] = [yAfter], ["distance_from_origin"] = [Norm2(xAfter, yAfter)],
            ["x_velocity"] = [xVelocity], ["y_velocity"] = [yVelocity], ["reward_forward"] = [forward], ["reward_fwd"] = [forward],
            ["forward_reward"] = [forward], ["reward_ctrl"] = [-ctrlCost],
        });
    }
}

/// <summary>
///     Pusher-v5: minus the object's distance to the goal, minus half the arm tip's distance to the object, minus a float32
///     control cost; never terminates.
/// </summary>
internal sealed class PusherRules : EnvRules {
    private readonly int tipsArm, obj, goal;

    /// <summary>Sets up the rules.</summary>
    /// <param name="model">The model.</param>
    /// <param name="run">The run.</param>
    public PusherRules(MjModel model, RunInfo run) : base(model, run) {
        tipsArm = BodyId("tips_arm");
        obj = BodyId("object");
        goal = BodyId("goal");
    }

    /// <summary>
    ///     Pusher's reset: the arm at its reference pose (the env assigns <c>init_qpos</c> itself, no noise), the object
    ///     drawn by rejection (<c>uniform(-0.3, 0, 1)</c> then <c>uniform(-0.2, 0.2, 1)</c> until farther than 0.17 from the
    ///     goal at the origin), then velocity noise <c>uniform(-0.005, 0.005, nv)</c> with the object's and goal's zeroed.
    /// </summary>
    /// <param name="rng">The seeded generator.</param>
    /// <returns>The reset state.</returns>
    protected override (double[] Qpos, double[] Qvel) ResetState(Generator rng) {
        double[] qpos = (double[])M.qpos0.Clone();
        double c0, c1;
        do {
            c0 = Uniform(rng, -0.3, 0.0, 1)[0];
            c1 = Uniform(rng, -0.2, 0.2, 1)[0];
        } while (!(Norm2(c0 - 0, c1 - 0) > 0.17));
        qpos[M.nq - 4] = c0;
        qpos[M.nq - 3] = c1;
        qpos[M.nq - 2] = 0;
        qpos[M.nq - 1] = 0;
        double[] qvel = Uniform(rng, -0.005, 0.005, M.nv).Select(v => 0.0 + v).ToArray();
        for (int i = M.nv - 4; i < M.nv; i++)
            qvel[i] = 0;
        return (qpos, qvel);
    }

    /// <inheritdoc/>
    public override double[] Obs(MjData core) =>
        core.qpos.Take(7).Concat(core.qvel.Take(7)).Concat(core.xpos.Skip(3 * tipsArm).Take(3))
            .Concat(core.xpos.Skip(3 * obj).Take(3)).Concat(core.xpos.Skip(3 * goal).Take(3)).ToArray();

    /// <inheritdoc/>
    protected override (double, bool, Dictionary<string, double[]>) Results(MjData core, double[] action) {
        double[] Diff(int a, int b) => Enumerable.Range(0, 3).Select(k => core.xpos[3 * a + k] - core.xpos[3 * b + k]).ToArray();
        double[] vec1 = Diff(obj, tipsArm), vec2 = Diff(obj, goal);
        double rewardNear = -Norm3(vec1[0], vec1[1], vec1[2]) * 0.5;
        double rewardDist = -Norm3(vec2[0], vec2[1], vec2[2]) * 1;
        float rewardCtrl = -SumSquaresF32(action) * 0.1f;
        double reward = rewardDist + rewardCtrl + rewardNear;
        return (reward, false, new Dictionary<string, double[]> {
            ["reward_dist"] = [rewardDist], ["reward_ctrl"] = [rewardCtrl], ["reward_near"] = [rewardNear],
        });
    }
}
