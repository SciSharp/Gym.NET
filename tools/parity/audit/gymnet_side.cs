#:project ../../../src/Gym.Environments/Gym.Environments.csproj
#:property PublishAot=false
// Gym.NET side of the Gymnasium parity audit (docs/plans/GYMNASIUM_PARITY_AUDIT.md).
//
// Prints the cases of gymnasium_side.py through Gym.NET, in the same "<case> <index> <value>" format, into two files:
//   gymnet.txt      what Gym.NET does today (its spaces, envs and legacy MT19937 generators);
//   projection.txt  what the same cases give when they are computed the way Gymnasium computes them, on NumSharp's
//                   Generator(PCG64(SeedSequence(seed))) and with a float64 transcription of Gymnasium's CartPole step.
//                   It is a measurement of how far F2 (seeding) and the P2/P4 re-ports would carry, not shipped code.
// It also renders CartPole at the states gymnasium_side.py rendered (frames/gymnet_cartpole_*.rgb) and lists Gym.NET's
// public types (gymnet_api.txt).
//
// Run after gymnasium_side.py, from the repository root (ReferenceNDArrayAnalyzer=false: a file-based app cannot restore
// NumSharp's netstandard2.0 analyzer, see eng/NumSharp.props):
//     dotnet run -p:ReferenceNDArrayAnalyzer=false tools/parity/audit/gymnet_side.cs -- <output-dir>
using System.Globalization;
using System.Reflection;
using Gym.Environments;
using Gym.Environments.Envs.Aether;
using Gym.Environments.Envs.Classic;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

string outDir = args[0];
long[] seeds = { 0, 1, 42, 1000 };
const int CartPoleSteps = 500;
const int LunarRandomSteps = 2000;
var gymnasiumLines = File.ReadAllLines(Path.Combine(outDir, "gymnasium.txt"));
var current = new List<string>();
var projection = new List<string>();
var stateField = typeof(CartPoleEnv).GetField("state", BindingFlags.NonPublic | BindingFlags.Instance)
    ?? throw new InvalidOperationException("CartPoleEnv.state not found; the probe injects Gymnasium's states through it.");

foreach (long seed in seeds)
{
    // ---- Seeding ----
    Emit(current, $"rng.random.s{seed}", LegacySeeding.RandomState(seed).random_sample(8).ToArray<double>().Select(F64));
    Emit(projection, $"rng.random.s{seed}", np.random.default_rng(seed).random(8).ToArray<double>().Select(F64));

    // ---- Discrete ----
    var space = new Discrete(4, seed: seed);
    Emit(current, $"discrete.sample.s{seed}", Enumerable.Range(0, 10).Select(_ => Long(space.Sample())));
    using (var mask = np.array(new sbyte[] { 1, 0, 1, 1 }))
        Emit(current, $"discrete.masked.s{seed}", Enumerable.Range(0, 5).Select(_ => Long(space.Sample(mask))));
    var started = new Discrete(3, seed: seed, start: 5);
    Emit(current, $"discrete.start.s{seed}", Enumerable.Range(0, 5).Select(_ => Long(started.Sample())));
    using (var one = space.Sample())
        Emit(current, $"discrete.meta.s{seed}", new[] { $"shape={PythonTuple(space.Shape.Dimensions)} dtype={space.DType.name} sample_type={(one.ndim == 0 ? "NDArray(0-d)" : "NDArray")}" });

    // Gymnasium: start + np_random.integers(n, dtype=int64); masked: start + choice(np.where(mask == 1)[0]).
    var rng = np.random.default_rng(seed);
    Emit(projection, $"discrete.sample.s{seed}", Enumerable.Range(0, 10).Select(_ => Long(rng.integers(4, dtype: np.int64))));
    using (var mask = np.array(new sbyte[] { 1, 0, 1, 1 }))
    using (var isValid = mask == 1)
    using (var valid = np.nonzero(isValid)[0])
        Emit(projection, $"discrete.masked.s{seed}", Enumerable.Range(0, 5).Select(_ => SingleElement(rng.choice(valid))));
    var startRng = np.random.default_rng(seed);
    Emit(projection, $"discrete.start.s{seed}", Enumerable.Range(0, 5).Select(_ => (5 + long.Parse(Long(startRng.integers(3, dtype: np.int64)), CultureInfo.InvariantCulture)).ToString(CultureInfo.InvariantCulture)));

    // ---- Box ----
    using (var s = new Box(-1f, 1f, new Shape(3), np.float32, seed: seed).Sample())
    {
        Emit(current, $"box.bounded.s{seed}", s.ToArray<float>().Select(F32));
        Emit(current, $"box.bounded.dtype.s{seed}", new[] { s.dtype.name });
    }
    using (var s = new Box(float.NegativeInfinity, float.PositiveInfinity, new Shape(3), np.float64, seed: seed).Sample())
        Emit(current, $"box.normal.s{seed}", s.ToArray<double>().Select(F64));
    using (var s = new Box(0f, float.PositiveInfinity, new Shape(3), np.float64, seed: seed).Sample())
        Emit(current, $"box.exponential.s{seed}", s.ToArray<double>().Select(F64));
    using (var s = new Box(0f, 5f, new Shape(8), np.int32, seed: seed).Sample())
        Emit(current, $"box.int.s{seed}", s.ToArray<int>().Select(v => v.ToString(CultureInfo.InvariantCulture)));

    // Gymnasium's Box.sample: bounded elements draw uniform(low, high) = low + (high - low) * next_double in float64;
    // unbounded ones a STANDARD normal; low-bounded ones exponential + low; integer boxes draw with high + 1, then floor,
    // clip to the dtype's range and cast.
    Emit(projection, $"box.bounded.s{seed}", BoundedDraw(np.random.default_rng(seed), -1.0, 1.0, 3).Select(v => F32((float)v)));
    Emit(projection, $"box.bounded.dtype.s{seed}", new[] { "float32" });
    Emit(projection, $"box.normal.s{seed}", np.random.default_rng(seed).normal(0.0, 1.0, 3).ToArray<double>().Select(F64));
    Emit(projection, $"box.exponential.s{seed}", np.random.default_rng(seed).exponential(1.0, 3).ToArray<double>().Select(v => F64(v + 0.0)));
    Emit(projection, $"box.int.s{seed}", BoundedDraw(np.random.default_rng(seed), 0.0, 5.0 + 1.0, 8)
        .Select(v => ((int)Math.Clamp(Math.Floor(v), int.MinValue, int.MaxValue)).ToString(CultureInfo.InvariantCulture)));

    // ---- CartPole ----
    var cartPole = new CartPoleEnv(NullEnvViewer.Factory, LegacySeeding.RandomState(seed));
    using (var reset = cartPole.Reset())
    {
        Emit(current, $"cartpole.reset.dtype.s{seed}", new[] { reset.dtype.name });
        Emit(current, $"cartpole.reset.obs.s{seed}", reset.ToArray<double>().Select(F64));
        Emit(current, $"cartpole.reset.state.s{seed}", reset.ToArray<double>().Select(F64));
    }
    // Dynamics from Gymnasium's own initial state, injected, so the comparison isolates the step function.
    double[] initial = Enumerable.Range(0, 4).Select(i => FromF64(Value(gymnasiumLines, $"cartpole.reset.state.s{seed}", i))).ToArray();
    stateField.SetValue(cartPole, np.array(initial));
    for (int t = 1; t <= CartPoleSteps; t++)
    {
        var step = cartPole.Step((long)Action(t - 1));
        var st = step.Observation.ToArray<double>();
        current.Add($"cartpole.traj.s{seed} {t} {F64(st[0])} {F64(st[1])} {F64(st[2])} {F64(st[3])} {F64(step.Reward)} {(step.Done ? 1 : 0)}");
    }

    // Projection: Gymnasium's reset on NumSharp's Generator, then Gymnasium's step transcribed in float64.
    double[] projected;
    using (var drawn = np.random.default_rng(seed).uniform(-0.05, 0.05, 4))
        projected = drawn.ToArray<double>();
    Emit(projection, $"cartpole.reset.dtype.s{seed}", new[] { "float32" });
    Emit(projection, $"cartpole.reset.obs.s{seed}", projected.Select(v => F32((float)v)));
    Emit(projection, $"cartpole.reset.state.s{seed}", projected.Select(F64));
    var pole = new CartPoleFloat64(projected);
    for (int t = 1; t <= CartPoleSteps; t++)
    {
        var (st, reward, terminated) = pole.Step(Action(t - 1));
        projection.Add($"cartpole.traj.s{seed} {t} {F64(st[0])} {F64(st[1])} {F64(st[2])} {F64(st[3])} {F64(reward)} {(terminated ? 1 : 0)}");
    }

    // ---- LunarLander ----
    foreach (bool continuous in new[] { false, true })
    {
        string mode = continuous ? "continuous" : "discrete";
        var lander = new LunarLanderEnv(NullEnvViewer.Factory, continuous: continuous, random_state: LegacySeeding.RandomState(seed));
        NDArray obs = lander.Reset();
        Emit(current, $"lunar.{mode}.reset.s{seed}", obs.ToArray<float>().Select(F32));
        // Gym.NET has no TimeLimit; cap at LunarLander-v3's max_episode_steps (1000) to compare like with like.
        double total = 0;
        int steps = 0;
        bool done = false;
        while (!done && steps < 1000)
        {
            var step = lander.Step(Pid(lander, obs.ToArray<float>()));
            total += step.Reward;
            steps++;
            done = step.Done;
            obs = step.Observation;
        }
        current.Add($"lunar.{mode}.heuristic.s{seed} 0 steps={steps} total={F64(total)} ({total.ToString("F6", CultureInfo.InvariantCulture)}) terminated={(done ? 1 : 0)}");
    }
    {
        var lander = new LunarLanderEnv(NullEnvViewer.Factory, random_state: LegacySeeding.RandomState(seed));
        lander.ActionSpace.Seed(seed);
        lander.Reset().Dispose();
        int outside = 0;
        for (int i = 0; i < LunarRandomSteps; i++)
        {
            using var action = lander.ActionSpace.Sample();
            var step = lander.Step(action);
            if (!lander.ObservationSpace.Contains(step.Observation))
                outside++;
            if (step.Done)
                lander.Reset().Dispose();
        }
        current.Add($"lunar.random.outside.s{seed} 0 {outside}/{LunarRandomSteps}");
    }
}

// ---- Observation-space bounds and the continuous-sample crash (B-20) ----
var cp = new CartPoleEnv(NullEnvViewer.Factory);
current.Add("cartpole.obs_space 0 " + string.Join(" ", ((Box)cp.ObservationSpace).Low.ToArray<float>().Select(F32)));
current.Add("cartpole.obs_space 1 " + string.Join(" ", ((Box)cp.ObservationSpace).High.ToArray<float>().Select(F32)));
var ll = new LunarLanderEnv(NullEnvViewer.Factory);
current.Add("lunar.obs_space 0 " + string.Join(" ", ((Box)ll.ObservationSpace).Low.ToArray<float>().Select(F32)));
current.Add("lunar.obs_space 1 " + string.Join(" ", ((Box)ll.ObservationSpace).High.ToArray<float>().Select(F32)));
{
    var lander = new LunarLanderEnv(NullEnvViewer.Factory, continuous: true, random_state: LegacySeeding.RandomState(0));
    lander.Reset().Dispose();
    string outcome;
    try
    {
        lander.Step(lander.ActionSpace.Sample());
        outcome = "ok";
    }
    catch (Exception e)
    {
        outcome = $"{e.GetType().Name}: {e.Message}";
    }
    current.Add($"lunar.continuous.step_sample 0 {outcome}");
}

// ---- CartPole frames at Gymnasium's states ----
Directory.CreateDirectory(Path.Combine(outDir, "frames"));
foreach (var line in File.ReadAllLines(Path.Combine(outDir, "cartpole_states.txt")).Where(l => l.Length > 0))
{
    var parts = line.Split(' ');
    var env = new CartPoleEnv(NullEnvViewer.Factory);
    stateField.SetValue(env, np.array(parts.Skip(2).Select(FromF64).ToArray()));
    using var image = (Image<Rgba32>)env.Render();
    var rgb = new byte[image.Width * image.Height * 3];
    image.ProcessPixelRows(rows =>
    {
        for (int y = 0; y < rows.Height; y++)
        {
            var row = rows.GetRowSpan(y);
            for (int x = 0; x < row.Length; x++)
            {
                int o = (y * image.Width + x) * 3;
                rgb[o] = row[x].R;
                rgb[o + 1] = row[x].G;
                rgb[o + 2] = row[x].B;
            }
        }
    });
    File.WriteAllBytes(Path.Combine(outDir, "frames", $"gymnet_cartpole_s{parts[0]}_t{parts[1]}.rgb"), rgb);
}

// ---- Public API surface ----
var api = new List<string>();
foreach (var assembly in new[] { typeof(Gym.Envs.Env).Assembly, typeof(CartPoleEnv).Assembly })
    foreach (var type in assembly.GetExportedTypes().OrderBy(t => t.FullName, StringComparer.Ordinal))
        api.Add($"{assembly.GetName().Name} {type.FullName}");

File.WriteAllLines(Path.Combine(outDir, "gymnet.txt"), current);
File.WriteAllLines(Path.Combine(outDir, "projection.txt"), projection);
File.WriteAllLines(Path.Combine(outDir, "gymnet_api.txt"), api);
Console.WriteLine($"{current.Count} current lines, {projection.Count} projection lines, {api.Count} public types -> {outDir}");

// ---- helpers ----

// The open-loop CartPole actions of gymnasium_side.py: 0, 0, 1, 1, 0, 0, ...
static int Action(int i) => (i / 2) % 2;

static string F64(double v) => BitConverter.DoubleToInt64Bits(v).ToString("x16", CultureInfo.InvariantCulture);
static string F32(float v) => BitConverter.SingleToInt32Bits(v).ToString("x8", CultureInfo.InvariantCulture);
static double FromF64(string hex) => BitConverter.Int64BitsToDouble(long.Parse(hex, NumberStyles.HexNumber, CultureInfo.InvariantCulture));

// Converts and releases a 0-d integer sample.
static string Long(NDArray scalar)
{
    using (scalar)
        return ((long)scalar).ToString(CultureInfo.InvariantCulture);
}

// Converts and releases a choice() result. NumSharp's Generator.choice(array) with size=None returns a one-element 1-D
// array where NumPy returns a scalar (Generator.Choice.Sampler.cs says so); the drawn value is the same.
static string SingleElement(NDArray drawn)
{
    using (drawn)
    {
        if (drawn.size != 1)
            throw new InvalidOperationException($"choice returned {drawn.size} elements");
        return Convert.ToInt64(drawn.GetAtIndex(0), CultureInfo.InvariantCulture).ToString(CultureInfo.InvariantCulture);
    }
}

static void Emit(List<string> lines, string caseName, IEnumerable<string> values)
{
    int index = 0;
    foreach (var value in values)
        lines.Add($"{caseName} {index++} {value}");
}

static string Value(string[] lines, string caseName, int index)
{
    string prefix = $"{caseName} {index} ";
    return lines.First(l => l.StartsWith(prefix, StringComparison.Ordinal)).Substring(prefix.Length);
}

static string PythonTuple(long[] dims) => dims.Length == 1 ? $"({dims[0]},)" : "(" + string.Join(", ", dims) + ")";

// numpy Generator.uniform(low_array, high_array, size): lower + (high - lower) * next_double per element, in float64.
static double[] BoundedDraw(NumSharp.Generator rng, double low, double high, int n)
{
    using var u = rng.random(n);
    return u.ToArray<double>().Select(x => low + (high - low) * x).ToArray();
}

// The heuristic of Gymnasium's lunar_lander.py, as ported by the LunarLanderEnvironment test.
static object Pid(LunarLanderEnv env, float[] s)
{
    float angle_targ = s[0] * 0.5f + s[2] * 1f;
    if (angle_targ > 0.4f) angle_targ = 0.4f;
    if (angle_targ < -0.4) angle_targ = -0.4f;
    float hover_targ = 0.55f * Math.Abs(s[0]);
    float angle_todo = (angle_targ - s[4]) * 0.5f - s[5] * 1f;
    float hover_todo = (hover_targ - s[1]) * 0.5f - s[3] * 0.5f;
    if (s[6] > 0f || s[7] > 0f)
    {
        angle_todo = 0f;
        hover_todo = -s[3] * 0.5f;
    }
    if (env.ContinuousMode)
    {
        NDArray a = np.array(new float[] { hover_todo * 20f - 1f, -angle_todo * 20f });
        return np.clip(a, -1f, 1f);
    }
    var action = LunarLanderDiscreteActions.Nothing;
    if (hover_todo > Math.Abs(angle_todo) && hover_todo > 0.05f) action = LunarLanderDiscreteActions.FireMainEngine;
    else if (angle_todo < -0.05f) action = LunarLanderDiscreteActions.FireRightThruster;
    else if (angle_todo > 0.05f) action = LunarLanderDiscreteActions.FireLeftThruster;
    return (int)action;
}

/// <summary>
///     gymnasium/envs/classic_control/cartpole.py:164 @ 9e04324 transcribed in float64 with Math.Cos/Math.Sin: the projection
///     of the P4 re-port. Reward follows the default (sutton_barto_reward=False) branch.
/// </summary>
sealed class CartPoleFloat64
{
    const double gravity = 9.8, masscart = 1.0, masspole = 0.1, total_mass = masspole + masscart, length = 0.5;
    const double polemass_length = masspole * length, force_mag = 10.0, tau = 0.02;
    const double theta_threshold_radians = 12 * 2 * Math.PI / 360, x_threshold = 2.4;
    readonly double[] _state;
    int? _stepsBeyondTerminated;

    public CartPoleFloat64(double[] state) => _state = (double[])state.Clone();

    public (double[] State, double Reward, bool Terminated) Step(int action)
    {
        double x = _state[0], x_dot = _state[1], theta = _state[2], theta_dot = _state[3];
        double force = action == 1 ? force_mag : -force_mag;
        double costheta = Math.Cos(theta), sintheta = Math.Sin(theta);
        double temp = (force + polemass_length * (theta_dot * theta_dot) * sintheta) / total_mass;
        double thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * (costheta * costheta) / total_mass));
        double xacc = temp - polemass_length * thetaacc * costheta / total_mass;
        x = x + tau * x_dot;
        x_dot = x_dot + tau * xacc;
        theta = theta + tau * theta_dot;
        theta_dot = theta_dot + tau * thetaacc;
        _state[0] = x; _state[1] = x_dot; _state[2] = theta; _state[3] = theta_dot;
        bool terminated = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;
        double reward;
        if (!terminated) reward = 1.0;
        else if (_stepsBeyondTerminated is null) { _stepsBeyondTerminated = 0; reward = 1.0; }
        else { _stepsBeyondTerminated++; reward = 0.0; }
        return ((double[])_state.Clone(), reward, terminated);
    }
}
