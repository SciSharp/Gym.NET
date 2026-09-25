using System;
using System.Globalization;

namespace Gym.Rendering {
    /// <summary>
    ///     The part of <c>pygame.math.Vector2</c> Gymnasium's renderers use: two float64 components and
    ///     <see cref="RotateRad"/>, whose treatment of angles within <see cref="Epsilon"/> of a quarter turn decides where
    ///     CartPole's pole is drawn.
    /// </summary>
    /// <param name="X">The x component.</param>
    /// <param name="Y">The y component.</param>
    public readonly record struct Vector2(double X, double Y) {
        /// <summary>
        ///     pygame's default comparison tolerance for vectors (<c>Vector2().epsilon</c>, 1e-6). Rotation uses it to snap
        ///     angles near 0°, 90°, 180° and 270° to exact quarter turns.
        /// </summary>
        public const double Epsilon = 1e-6;

        /// <summary><c>v[0]</c> or <c>v[1]</c>, the components by index as Python unpacks them.</summary>
        /// <param name="index">0 for x, 1 for y.</param>
        /// <exception cref="IndexOutOfRangeException">Thrown for any other index, as pygame raises <c>IndexError</c>.</exception>
        public double this[int index] => index switch {
            0 => X,
            1 => Y,
            _ => throw new IndexOutOfRangeException("Index out of range."),
        };

        /// <summary>
        ///     <c>v.rotate_rad(angle)</c>: the vector rotated counter-clockwise by <paramref name="angle"/> radians, computed as
        ///     pygame-ce 2.5.8 computes it.
        /// </summary>
        /// <param name="angle">The angle in radians; NaN or an infinity gives a NaN vector.</param>
        /// <returns>The rotated vector.</returns>
        /// <exception cref="InvalidOperationException">
        ///     Never thrown for a reduced angle; it stands where pygame raises <c>RuntimeError</c> for an impossible quarter
        ///     turn, so a broken reduction fails loudly instead of rotating wrongly.
        /// </exception>
        /// <remarks>
        ///     <para>
        ///     The angle is first reduced into [0, 2π) with the exact floating-point remainder by <c>2π</c> (the double nearest
        ///     2π), adding <c>2π</c> to a negative remainder. If the reduced angle lies within <see cref="Epsilon"/> of a
        ///     multiple of π/2 (tested as <c>remainder(angle + ε, π/2) &lt; 2ε</c>), the rotation is an exact quarter turn
        ///     chosen by <c>(int)((angle + ε) / (π/2))</c>: 0 and 4 keep the vector, 1 gives <c>(-y, x)</c>, 2 gives
        ///     <c>(-x, -y)</c>, 3 gives <c>(y, -x)</c>. Otherwise it is <c>(cos·x - sin·y, sin·x + cos·y)</c> with the sine and
        ///     cosine of the reduced angle, each product rounded separately (no fused multiply-add).
        ///     </para>
        ///     <para>
        ///     The snapping is visible: CartPole's pole drawn at θ = 1e-7 is exactly upright, not rotated by 1e-7. The sine and
        ///     cosine are .NET's <see cref="Math.Sin"/> and <see cref="Math.Cos"/>, which on the reference platform (Windows
        ///     x64) come from the same C runtime pygame's are compiled against.
        ///     </para>
        /// </remarks>
        public Vector2 RotateRad(double angle) {
            const double twoPi = 2.0 * Math.PI;
            const double halfPi = Math.PI / 2.0;
            angle = CRemainder(angle, twoPi);
            if (angle < 0)
                angle += twoPi;

            if ((angle + Epsilon) % halfPi < 2 * Epsilon) {
                switch ((int)((angle + Epsilon) / halfPi)) {
                    case 0:
                    case 4:
                        return new Vector2(X, Y);
                    case 1:
                        return new Vector2(-Y, X);
                    case 2:
                        return new Vector2(-X, -Y);
                    case 3:
                        return new Vector2(Y, -X);
                    default:
                        // Unreachable for a reduced angle; pygame raises RuntimeError asking for a bug report here.
                        throw new InvalidOperationException($"Vector2.RotateRad reached an impossible quarter turn for angle {angle.ToString("R", CultureInfo.InvariantCulture)}.");
                }
            }

            double sin = Math.Sin(angle);
            double cos = Math.Cos(angle);
            return new Vector2(cos * X - sin * Y, sin * X + cos * Y);
        }

        /// <summary>
        ///     C's <c>fmod(x, y)</c> as the reference platform's C runtime (UCRT) computes it, including the NaN it returns for
        ///     a non-finite dividend, which decides the sign bit of every coordinate rotated by an infinite or NaN angle.
        /// </summary>
        /// <param name="x">The dividend.</param>
        /// <param name="y">The divisor, finite and non-zero here.</param>
        /// <returns>The exact truncated remainder, with the sign of <paramref name="x"/>.</returns>
        /// <remarks>
        ///     C# <c>%</c> gives the same exact remainder for finite operands on every runtime, but not the same NaN: .NET 8's
        ///     helper returns its own sign-set NaN for an infinite or NaN dividend (measured: <c>rotate_rad(inf)</c> then
        ///     differs from pygame in the sign bit), while UCRT returns a NaN operand unchanged (payload and all, signalling
        ///     included) and <c>±inf</c> as a quiet NaN with the infinity's sign.
        /// </remarks>
        private static double CRemainder(double x, double y) {
            if (double.IsNaN(x))
                return x;
            if (double.IsInfinity(x))
                return BitConverter.Int64BitsToDouble(x > 0 ? 0x7FF8000000000000L : unchecked((long)0xFFF8000000000000UL));
            return x % y;
        }
    }
}
