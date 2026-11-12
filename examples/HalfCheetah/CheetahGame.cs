using System;
using System.Numerics;
using NumSharp;
using Gym.MuJoCo;

namespace MujocoPoc.Play;

/// <summary>
///     HalfCheetah-v5's env rules over the managed core: Gymnasium's reset noise, five physics steps per action, the
///     reward <c>x_velocity - 0.1*sum(action²)</c> with NEP 50's float32 control cost, and the 17-number observation
///     (qpos without x, then qvel). The physics checker proved these rules bit-exact against Gymnasium.
/// </summary>
/// <remarks>
///     An env step can also be taken in parts (<see cref="BeginStep"/>, <see cref="PhysicsStep"/> × frame_skip,
///     <see cref="EndStep"/>), which is the same step a renderer can draw between physics steps. Not thread-safe; give each
///     thread its own game (the model may be shared, the core never writes it). Dispose a game when done: it owns its
///     core's arrays (the simulation state), while the model stays the caller's.
/// </remarks>
public sealed class CheetahGame : IDisposable {
    /// <summary>The length of a Gymnasium episode (its TimeLimit).</summary>
    public const int EpisodeSteps = 1000;

    /// <summary>The number of actuators: back thigh, shin, foot, then front thigh, shin, foot.</summary>
    public const int ActionSize = 6;

    /// <summary>The observation size: qpos[1:] (8) and qvel (9).</summary>
    public const int ObservationSize = 17;

    /// <summary>The core being stepped; its arrays are the live simulation state.</summary>
    public readonly MjData Core;

    /// <summary>The model; borrowed from the caller, who may share it between games and releases it.</summary>
    [NDBorrowed]
    public readonly MjModel Model;

    /// <summary>The action of the current env step, float32 as Gymnasium's Box hands it to MuJoCo.</summary>
    public readonly float[] Action = new float[ActionSize];

    private readonly double[] observation = new double[ObservationSize];
    private int substep;
    private double xBefore;

    /// <summary>Env steps taken since the reset.</summary>
    public int StepIndex { get; private set; }

    /// <summary>The sum of the rewards since the reset, in Gymnasium's float64 order.</summary>
    public double EpisodeReturn { get; private set; }

    /// <summary>The last env step's reward.</summary>
    public double LastReward { get; private set; }

    /// <summary>The last env step's forward velocity of the root, <c>(x_after - x_before)/dt</c>.</summary>
    public double LastVelocity { get; private set; }

    /// <summary>The seed of the current episode, or -1 when it was reset from a <see cref="Random"/>.</summary>
    public int Seed { get; private set; } = -1;

    /// <summary>HalfCheetah-v5's physics steps per env step (Gymnasium's <c>frame_skip</c> default), an env setting.</summary>
    public const int FrameSkip = 5;

    /// <summary>The env time step: the physics time step times frame_skip (0.05 s).</summary>
    public double Dt => Model.timestep * FrameSkip;

    /// <summary>Creates a game at the model's reference configuration (call a <c>Reset</c> before stepping).</summary>
    /// <param name="model">The compiled HalfCheetah model.</param>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null.</exception>
    /// <exception cref="ArgumentException">The model's sizes are not HalfCheetah's.</exception>
    public CheetahGame(MjModel model) {
        Model = model ?? throw new ArgumentNullException(nameof(model));
        if (model.nu != ActionSize || model.nq + model.nv - 1 != ObservationSize)
            throw new ArgumentException($"expected HalfCheetah's sizes (nu {ActionSize}, nq + nv - 1 = {ObservationSize})", nameof(model));
        Core = new MjData(model);
    }

    /// <summary>
    ///     Gymnasium's <c>reset(seed=seed)</c>: <c>qpos = init_qpos + uniform(-0.1, 0.1)</c>, <c>qvel = init_qvel +
    ///     0.1*standard_normal()</c>, drawn from <c>Generator(PCG64(SeedSequence(seed)))</c> through NumSharp, so the episode
    ///     starts exactly where Gymnasium's does for that seed.
    /// </summary>
    /// <param name="seed">A non-negative seed.</param>
    /// <returns>The first observation (a buffer the next call overwrites).</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="seed"/> is negative.</exception>
    public double[] Reset(int seed) {
        if (seed < 0)
            throw new ArgumentOutOfRangeException(nameof(seed), seed, "seeds are non-negative, as Gymnasium requires");
        var rng = new Generator(new PCG64(new SeedSequence(new BigInteger(seed))));
        using NDArray uniform = rng.uniform(-0.1, 0.1, new Shape(Model.nq));
        using NDArray normal = rng.standard_normal(new Shape(Model.nv));
        double[] u = uniform.ToArray<double>(), z = normal.ToArray<double>();
        Start(u, z);
        Seed = seed;
        return Observation();
    }

    /// <summary>
    ///     A reset with the same distributions drawn from a <see cref="Random"/>: cheaper, for training, where no episode
    ///     needs to match Gymnasium's stream.
    /// </summary>
    /// <param name="random">The source of the noise.</param>
    /// <returns>The first observation (a buffer the next call overwrites).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="random"/> is null.</exception>
    public double[] Reset(Random random) {
        ArgumentNullException.ThrowIfNull(random);
        var u = new double[Model.nq];
        var z = new double[Model.nv];
        for (int i = 0; i < u.Length; i++)
            u[i] = -0.1 + 0.2 * random.NextDouble();
        for (int i = 0; i < z.Length; i++)
            z[i] = Gaussian(random);
        Start(u, z);
        Seed = -1;
        return Observation();
    }

    /// <summary>Puts the core at the reset state built from the drawn noise, as MuJoCo's reset then Gymnasium's set_state.</summary>
    /// <param name="u">The position noise, nq values in [-0.1, 0.1).</param>
    /// <param name="z">The standard normal velocity draws, nv values.</param>
    private void Start(double[] u, double[] z) {
        for (int i = 0; i < Model.nq; i++)
            Core.qpos[i] = Model.qpos0[i] + u[i];
        for (int i = 0; i < Model.nv; i++)
            Core.qvel[i] = 0.0 + 0.1 * z[i];
        // mj_resetData clears the warm start, the controls and the clock; set_state's mj_forward changes none of them
        Array.Clear(Core.qacc_warmstart);
        Array.Clear(Core.ctrl);
        Core.time = 0;
        Array.Clear(Action);
        StepIndex = 0;
        EpisodeReturn = 0;
        LastReward = 0;
        LastVelocity = 0;
        substep = 0;
        // derived quantities (frames, contacts) for drawing the start, as set_state's mj_forward computes them
        Core.Forward();
    }

    /// <summary>The observation: qpos without the root's x, then qvel.</summary>
    /// <returns>A buffer the next call overwrites.</returns>
    public double[] Observation() {
        Array.Copy(Core.qpos, 1, observation, 0, Model.nq - 1);
        Array.Copy(Core.qvel, 0, observation, Model.nq - 1, Model.nv);
        return observation;
    }

    /// <summary>One whole env step: the action held for frame_skip physics steps.</summary>
    /// <param name="action">Six values; Gymnasium's Box is float32, and MuJoCo clamps each to [-1, 1] (the cost does not).</param>
    /// <returns>The step's reward.</returns>
    /// <exception cref="ArgumentException"><paramref name="action"/> does not have six values.</exception>
    public double Step(ReadOnlySpan<float> action) {
        BeginStep(action);
        for (int k = 0; k < FrameSkip; k++)
            PhysicsStep();
        return EndStep();
    }

    /// <summary>Starts an env step: records the root's x and sets the controls (<c>data.ctrl[:] = action</c>).</summary>
    /// <param name="action">Six float32 values.</param>
    /// <exception cref="ArgumentException"><paramref name="action"/> does not have six values.</exception>
    public void BeginStep(ReadOnlySpan<float> action) {
        if (action.Length != ActionSize)
            throw new ArgumentException($"an action has {ActionSize} values", nameof(action));
        action.CopyTo(Action);
        xBefore = Core.qpos[0];
        for (int i = 0; i < ActionSize; i++)
            Core.ctrl[i] = Action[i];
        substep = 0;
    }

    /// <summary>One <c>mj_step</c> of the env step in progress.</summary>
    /// <exception cref="InvalidOperationException">All frame_skip physics steps of this env step were already taken.</exception>
    public void PhysicsStep() {
        if (substep >= FrameSkip)
            throw new InvalidOperationException("this env step already took its frame_skip physics steps; call EndStep");
        Core.Step();
        substep++;
    }

    /// <summary>
    ///     Ends the env step: HalfCheetah-v5's reward, <c>1.0*x_velocity - 0.1*sum(action²)</c>, where the control cost is
    ///     float32 (NumPy adds the float32 squares from 0, in order), and the episode's running return.
    /// </summary>
    /// <returns>The reward.</returns>
    /// <exception cref="InvalidOperationException">The env step has not taken its frame_skip physics steps.</exception>
    public double EndStep() {
        if (substep != FrameSkip)
            throw new InvalidOperationException($"an env step takes {FrameSkip} physics steps, {substep} were taken");
        double xVelocity = (Core.qpos[0] - xBefore) / Dt;
        float squares = 0;
        foreach (float a in Action)
            squares += a * a;
        float ctrlCost = 0.1f * (0f + squares);
        LastVelocity = xVelocity;
        LastReward = 1.0 * xVelocity - ctrlCost;
        EpisodeReturn += LastReward;
        StepIndex++;
        return LastReward;
    }

    /// <summary>A standard normal draw by the Box-Muller transform.</summary>
    /// <param name="random">The source.</param>
    /// <returns>The draw.</returns>
    internal static double Gaussian(Random random) {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Releases the core's arrays, the simulation state this game owns; the model stays the caller's.</summary>
    public void Dispose() => Core.Dispose();
}
