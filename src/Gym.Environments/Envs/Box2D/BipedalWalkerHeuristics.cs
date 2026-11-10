#nullable enable
using System;
using NumSharp;

namespace Gym.Environments.Envs.Box2D {
    /// <summary>
    ///     Gymnasium's <c>BipedalWalkerHeuristics</c>: the hand-written gait that walks <see cref="BipedalWalkerEnv"/>, a state
    ///     machine over three phases (stand on one leg while the other swings up, put the other leg down, push off) that
    ///     alternates the legs and steers each joint toward a target angle with a proportional-derivative term, plus a hull
    ///     tilt correction and vertical damping. It walks the normal terrain slowly and falls in the hardcore one.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Stateful: one instance per episode, fed every observation in order (<see cref="StepHeuristic"/>). The arithmetic is
    ///     NumPy 2's on a float32 observation (NEP 50): every product and difference with the observation is float32, the
    ///     comparisons with Python's constants are float32 comparisons, and <see cref="SupportingKneeAngle"/> is a Python
    ///     float or, after a push-off took it from the observation, an <c>np.float32</c>, which changes the precision of its
    ///     next increments. Gym.NET tracks that kind, so its actions are Gymnasium's bit for bit.
    ///     </para>
    ///     <para>
    ///     One difference of aliasing, not of values: Python's <c>step_heuristic</c> returns its own attribute array, which
    ///     its next call overwrites in place before replacing it, so a caller keeping an old action sees it change. Gym.NET
    ///     returns a new array each call and keeps its own copy in <see cref="A"/>.
    ///     </para>
    ///     <para>
    ///     Not thread-safe (it is a mutable state machine). Reference:
    ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L783">gymnasium/envs/box2d/bipedal_walker.py</see>.
    ///     </para>
    /// </remarks>
    public sealed class BipedalWalkerHeuristics {
        /// <summary>Gymnasium's <c>STAY_ON_ONE_LEG</c>: the moving leg swings up and forward while the supporting knee straightens.</summary>
        public const int StayOnOneLeg = 1;

        /// <summary>Gymnasium's <c>PUT_OTHER_DOWN</c>: the moving leg reaches down until its foot touches the ground.</summary>
        public const int PutOtherDown = 2;

        /// <summary>Gymnasium's <c>PUSH_OFF</c>: the supporting knee bends to push the hull forward; then the legs swap roles.</summary>
        public const int PushOff = 3;

        /// <summary>Gymnasium's <c>SPEED</c>: the horizontal speed (observation units) above which the walker would fall forward.</summary>
        public const double Speed = 0.29;

        /// <summary>Gymnasium's <c>SUPPORT_KNEE_ANGLE</c>: the supporting knee's target and cap.</summary>
        public const double SupportKneeAngle = +0.1;

        /// <summary>The supporting knee's target, with the Python kind it has (a float, or a float32 taken from an observation).</summary>
        private NpScalar _supportingKneeAngle = NpScalar.PyFloat(SupportKneeAngle);

        /// <summary>The last action, before clipping and halving on the first call (Python's class attribute <c>a</c> starts at zeros).</summary>
        private readonly double[] _a = new double[4];

        /// <summary>Gymnasium's <c>state</c>: the gait's phase, <see cref="StayOnOneLeg"/>, <see cref="PutOtherDown"/> or <see cref="PushOff"/>.</summary>
        public int State { get; set; } = StayOnOneLeg;

        /// <summary>Gymnasium's <c>moving_leg</c>: the leg that swings (0 left, 1 right).</summary>
        public int MovingLeg { get; set; }

        /// <summary>Gymnasium's <c>supporting_leg</c>: the leg that carries the hull, always <c>1 - MovingLeg</c> after a swap.</summary>
        public int SupportingLeg { get; set; } = 1;

        /// <summary>
        ///     Gymnasium's <c>supporting_knee_angle</c>: the supporting knee's target, raised by 0.03 or 0.06 per step up to
        ///     <see cref="SupportKneeAngle"/>, reset from the moving knee at each push-off. Setting it makes it a Python float.
        /// </summary>
        public double SupportingKneeAngle {
            get => _supportingKneeAngle.Value;
            set => _supportingKneeAngle = NpScalar.PyFloat(value);
        }

        /// <summary>Gymnasium's <c>a</c>: the last action returned (zeros before the first call); a new copy on each read.</summary>
        public double[] A => (double[])_a.Clone();

        /// <summary>
        ///     Gymnasium's <c>step_heuristic(s)</c>: advances the gait's phase from an observation and returns the four motor
        ///     commands, each joint driven toward its phase's target angle, with the hull's tilt and vertical speed corrected,
        ///     halved and clipped to [-1, 1].
        /// </summary>
        /// <param name="s">
        ///     An observation of <see cref="BipedalWalkerEnv"/>: 24 values, float32 as the env returns them (a float64 one is read
        ///     in float64, as NumPy would; other dtypes are read as their values in float64).
        /// </param>
        /// <returns>A new float64 array of four commands (hip, knee, hip, knee), the caller's to dispose.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="s"/> is <see langword="null"/>.</exception>
        /// <exception cref="ArgumentException">
        ///     Thrown for an observation of fewer than 14 values (the gait reads up to <c>s[13]</c>, the right foot's contact),
        ///     before any state changes; Python would fail midway with NumPy's <c>IndexError</c>.
        /// </exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/box2d/bipedal_walker.py#L793">gymnasium/envs/box2d/bipedal_walker.py</see>.</remarks>
        public NDArray StepHeuristic(NDArray s) {
            if (s is null)
                throw new ArgumentNullException(nameof(s));
            // NpScalar.Element reads without bounds checks, so a short observation is refused before it is read.
            if (s.size < 14)
                throw new ArgumentException($"A BipedalWalker observation has 24 values; {s.size} were given.", nameof(s));
            NpScalar S(int i) => NpScalar.Element(s, i);
            NpScalar F(double value) => NpScalar.PyFloat(value);

            int movingSBase = 4 + 5 * MovingLeg;
            int supportingSBase = 4 + 5 * SupportingLeg;

            var hipTarg = new NpScalar?[2]; // -0.8 .. +1.1
            var kneeTarg = new NpScalar?[2]; // -0.6 .. +0.9
            NpScalar[] hipTodo = { F(0.0), F(0.0) };
            NpScalar[] kneeTodo = { F(0.0), F(0.0) };

            if (State == StayOnOneLeg) {
                hipTarg[MovingLeg] = F(1.1);
                kneeTarg[MovingLeg] = F(-0.6);
                _supportingKneeAngle += F(0.03);
                if (S(2) > F(Speed))
                    _supportingKneeAngle += F(0.03);
                _supportingKneeAngle = NpScalar.PyMin(_supportingKneeAngle, F(SupportKneeAngle));
                kneeTarg[SupportingLeg] = _supportingKneeAngle;
                if (S(supportingSBase + 0) < F(0.10)) // supporting leg is behind
                    State = PutOtherDown;
            }

            if (State == PutOtherDown) {
                hipTarg[MovingLeg] = F(+0.1);
                kneeTarg[MovingLeg] = F(SupportKneeAngle);
                kneeTarg[SupportingLeg] = _supportingKneeAngle;
                if (Truthy(S(movingSBase + 4))) {
                    State = PushOff;
                    _supportingKneeAngle = NpScalar.PyMin(S(movingSBase + 2), F(SupportKneeAngle));
                }
            }

            if (State == PushOff) {
                kneeTarg[MovingLeg] = _supportingKneeAngle;
                kneeTarg[SupportingLeg] = F(+1.0);
                if (S(supportingSBase + 2) > F(0.88) || S(2) > F(1.2 * Speed)) {
                    State = StayOnOneLeg;
                    MovingLeg = 1 - MovingLeg;
                    SupportingLeg = 1 - MovingLeg;
                }
            }

            // `if hip_targ[0]:` is Python truthiness: None and a zero target both skip the term.
            if (Truthy(hipTarg[0]))
                hipTodo[0] = F(0.9) * (hipTarg[0]!.Value - S(4)) - F(0.25) * S(5);
            if (Truthy(hipTarg[1]))
                hipTodo[1] = F(0.9) * (hipTarg[1]!.Value - S(9)) - F(0.25) * S(10);
            if (Truthy(kneeTarg[0]))
                kneeTodo[0] = F(4.0) * (kneeTarg[0]!.Value - S(6)) - F(0.25) * S(7);
            if (Truthy(kneeTarg[1]))
                kneeTodo[1] = F(4.0) * (kneeTarg[1]!.Value - S(11)) - F(0.25) * S(12);

            hipTodo[0] -= F(0.9) * (NpScalar.PyInt(0) - S(0)) - F(1.5) * S(1); // PID to keep head strait
            hipTodo[1] -= F(0.9) * (NpScalar.PyInt(0) - S(0)) - F(1.5) * S(1);
            kneeTodo[0] -= F(15.0) * S(3); // vertical speed, to damp oscillations
            kneeTodo[1] -= F(15.0) * S(3);

            // self.a[i] = ...: stored into a float64 array, then np.clip(0.5 * self.a, -1.0, 1.0) in float64 (NaN kept).
            _a[0] = hipTodo[0].Value;
            _a[1] = kneeTodo[0].Value;
            _a[2] = hipTodo[1].Value;
            _a[3] = kneeTodo[1].Value;
            for (int i = 0; i < _a.Length; i++) {
                double half = 0.5 * _a[i];
                _a[i] = double.IsNaN(half) ? half : Math.Min(Math.Max(half, -1.0), 1.0);
            }

            return np.array((double[])_a.Clone());
        }

        /// <summary>Python's truth value of an optional number: <c>None</c> and zero are false, NaN is true.</summary>
        /// <param name="value">The value, or <see langword="null"/> for <c>None</c>.</param>
        /// <returns>Whether Python's <c>if</c> takes it.</returns>
        private static bool Truthy(NpScalar? value) => value is { } v && v.Value != 0;
    }
}
