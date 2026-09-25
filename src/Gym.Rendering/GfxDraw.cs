using System;
using System.Collections.Generic;
using System.Numerics;
using Gym.Rendering.SdlGfx;
using Gym.Utils;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>
    ///     Port of <c>pygame.gfxdraw</c> (pygame-ce 2.5.8): the primitives Gymnasium's envs draw with, including pygame's
    ///     conversion of the Python arguments, so the same inputs give the same pixels or the same error.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Polygon vertices are floats, as Gymnasium computes them. pygame converts each coordinate with Python's
    ///     <c>int()</c> (truncation toward zero; NaN raises <see cref="ValueError"/>, an infinity
    ///     <see cref="OverflowException"/>), then to a C <c>long</c> (32 bits on the reference platform, Windows x64; larger
    ///     values raise <c>Python int too large to convert to C long</c>) and finally truncates it to Sint16 without a check:
    ///     a vertex at x = 40000 is drawn at x = -25536.
    ///     </para>
    ///     <para>
    ///     The integer arguments of <see cref="AACircle"/>, <see cref="FilledCircle"/>, <see cref="HLine"/> and
    ///     <see cref="VLine"/> are Python <c>int</c>s parsed with the <c>h</c> format: beyond a C long they raise the same
    ///     message, and beyond Sint16 <c>signed short integer is greater than maximum</c> (or <c>less than minimum</c>). Each
    ///     argument is checked in order, so the first bad one names the error, as in Python.
    ///     </para>
    ///     <para>
    ///     Python's <c>OverflowError</c> is <see cref="OverflowException"/>, the type NumSharp throws for NumPy's; its
    ///     <c>ValueError</c> is <see cref="ValueError"/>. On Linux and macOS a C long has 64 bits, so pygame there reports the
    ///     Sint16 range message for values between 2^31 and 2^63; Gym.NET follows the reference platform everywhere.
    ///     </para>
    /// </remarks>
    public static class GfxDraw {
        /// <summary>
        ///     <c>gfxdraw.aapolygon(surface, points, color)</c>: the anti-aliased outline of the polygon, each edge drawn
        ///     without its last pixel.
        /// </summary>
        /// <param name="surface">The surface.</param>
        /// <param name="points">The vertices, at least 3, as <c>(x, y)</c> floats.</param>
        /// <param name="color">The colour; its alpha weights the blend.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 3 points (<c>points must contain more than 2 points</c>) or a NaN coordinate.</exception>
        /// <exception cref="OverflowException">Thrown for an infinite coordinate or one beyond a C long.</exception>
        public static void AAPolygon(Surface surface, IReadOnlyList<(double X, double Y)> points, Color color) {
            (short[] vx, short[] vy) = PolygonArguments(surface, points);
            Check(SdlGfxPrimitives.AapolygonColor(surface, vx, vy, color.Packed), "aapolygon");
        }

        /// <summary><c>gfxdraw.filled_polygon(surface, points, color)</c>: the polygon's interior, scanline by scanline.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="points">The vertices, at least 3, as <c>(x, y)</c> floats.</param>
        /// <param name="color">The colour; below 255 its alpha blends each filled pixel.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> or <paramref name="points"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 3 points or a NaN coordinate.</exception>
        /// <exception cref="OverflowException">Thrown for an infinite coordinate or one beyond a C long.</exception>
        public static void FilledPolygon(Surface surface, IReadOnlyList<(double X, double Y)> points, Color color) {
            (short[] vx, short[] vy) = PolygonArguments(surface, points);
            Check(SdlGfxPrimitives.FilledPolygonColor(surface, vx, vy, color.Packed), "filled_polygon");
        }

        /// <summary><c>gfxdraw.aacircle(surface, x, y, r, color)</c>: the anti-aliased outline of a circle.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="x">The centre's column, a Python int.</param>
        /// <param name="y">The centre's row.</param>
        /// <param name="r">The radius.</param>
        /// <param name="color">The colour.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="OverflowException">Thrown for an argument outside Sint16 (with pygame's message for it).</exception>
        /// <exception cref="SdlError">Thrown for a negative radius, which SDL_gfx rejects.</exception>
        public static void AACircle(Surface surface, long x, long y, long r, Color color) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            short cx = ShortArgument(x), cy = ShortArgument(y), radius = ShortArgument(r);
            Check(SdlGfxPrimitives.AaellipseColor(surface, cx, cy, radius, radius, color.Packed), "aacircle");
        }

        /// <summary><c>gfxdraw.filled_circle(surface, x, y, r, color)</c>: a filled circle, row by row.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="x">The centre's column, a Python int.</param>
        /// <param name="y">The centre's row.</param>
        /// <param name="r">The radius.</param>
        /// <param name="color">The colour.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="OverflowException">Thrown for an argument outside Sint16.</exception>
        /// <exception cref="SdlError">Thrown for a negative radius.</exception>
        public static void FilledCircle(Surface surface, long x, long y, long r, Color color) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            short cx = ShortArgument(x), cy = ShortArgument(y), radius = ShortArgument(r);
            Check(SdlGfxPrimitives.FilledCircleColor(surface, cx, cy, radius, color.Packed), "filled_circle");
        }

        /// <summary><c>gfxdraw.hline(surface, x1, x2, y, color)</c>: a horizontal line, both ends included, clipped to the surface.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="x1">One end's column, a Python int.</param>
        /// <param name="x2">The other end's column.</param>
        /// <param name="y">The row.</param>
        /// <param name="color">The colour.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="OverflowException">Thrown for an argument outside Sint16.</exception>
        public static void HLine(Surface surface, long x1, long x2, long y, Color color) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            short a = ShortArgument(x1), b = ShortArgument(x2), row = ShortArgument(y);
            Check(SdlGfxPrimitives.HlineColor(surface, a, b, row, color.Packed), "hline");
        }

        /// <summary><c>gfxdraw.vline(surface, x, y1, y2, color)</c>: a vertical line, both ends included, clipped to the surface.</summary>
        /// <param name="surface">The surface.</param>
        /// <param name="x">The column, a Python int.</param>
        /// <param name="y1">One end's row.</param>
        /// <param name="y2">The other end's row.</param>
        /// <param name="color">The colour.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="surface"/> is <see langword="null"/>.</exception>
        /// <exception cref="OverflowException">Thrown for an argument outside Sint16.</exception>
        public static void VLine(Surface surface, long x, long y1, long y2, Color color) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            short column = ShortArgument(x), a = ShortArgument(y1), b = ShortArgument(y2);
            Check(SdlGfxPrimitives.VlineColor(surface, column, a, b, color.Packed), "vline");
        }

        /// <summary>
        ///     The <c>h</c> format of <c>PyArg_ParseTuple</c> applied to a Python int: <c>PyLong_AsLong</c> first (a C long,
        ///     32 bits on the reference platform), then the Sint16 range check.
        /// </summary>
        /// <param name="value">The Python int.</param>
        /// <returns>The Sint16.</returns>
        /// <exception cref="OverflowException">Thrown with pygame's message for a value outside the C long or the Sint16 range.</exception>
        internal static short ShortArgument(long value) {
            if (value < int.MinValue || value > int.MaxValue)
                throw new OverflowException("Python int too large to convert to C long");
            if (value > short.MaxValue)
                throw new OverflowException("signed short integer is greater than maximum");
            if (value < short.MinValue)
                throw new OverflowException("signed short integer is less than minimum");
            return (short)value;
        }

        /// <summary>
        ///     pygame's <c>Sint16FromObj</c> for a float coordinate: <c>int()</c>, then <c>PyLong_AsLong</c>, then an unchecked
        ///     truncation to Sint16 (so large coordinates wrap around).
        /// </summary>
        /// <param name="value">The coordinate.</param>
        /// <returns>The Sint16 SDL_gfx draws at.</returns>
        /// <exception cref="ValueError">Thrown for NaN.</exception>
        /// <exception cref="OverflowException">Thrown for an infinity or a value beyond a C long.</exception>
        internal static short CoordinateArgument(double value) {
            BigInteger integer = PyBuiltins.Int(value);
            if (integer < int.MinValue || integer > int.MaxValue)
                throw new OverflowException("Python int too large to convert to C long");
            return unchecked((short)(int)integer);
        }

        /// <summary>
        ///     Converts polygon vertices as <c>_gfx_aapolygoncolor</c> and <c>_gfx_filledpolygoncolor</c> do: the count check
        ///     first, then every vertex's x and y in order, so the first bad coordinate raises.
        /// </summary>
        /// <param name="surface">The surface, checked for <see langword="null"/>.</param>
        /// <param name="points">The vertices.</param>
        /// <returns>The Sint16 columns and rows.</returns>
        /// <exception cref="ArgumentNullException">Thrown when an argument is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for fewer than 3 points or a NaN coordinate.</exception>
        /// <exception cref="OverflowException">Thrown for an infinite coordinate or one beyond a C long.</exception>
        private static (short[] X, short[] Y) PolygonArguments(Surface surface, IReadOnlyList<(double X, double Y)> points) {
            if (surface is null)
                throw new ArgumentNullException(nameof(surface));
            if (points is null)
                throw new ArgumentNullException(nameof(points));
            if (points.Count < 3)
                throw new ValueError("points must contain more than 2 points");

            var vx = new short[points.Count];
            var vy = new short[points.Count];
            for (int i = 0; i < points.Count; i++) {
                vx[i] = CoordinateArgument(points[i].X);
                vy[i] = CoordinateArgument(points[i].Y);
            }

            return (vx, vy);
        }

        /// <summary>Turns SDL_gfx's -1 into pygame's <c>pygame.error</c>, as the gfxdraw wrappers do.</summary>
        /// <param name="result">The primitive's return value.</param>
        /// <param name="primitive">The gfxdraw function, for the message.</param>
        /// <exception cref="SdlError">Thrown when <paramref name="result"/> is -1.</exception>
        private static void Check(int result, string primitive) {
            if (result == -1)
                throw new SdlError($"SDL_gfx rejected the arguments of gfxdraw.{primitive}.");
        }
    }

    /// <summary>
    ///     Port of <c>pygame.error</c> as the gfxdraw functions raise it when an SDL_gfx primitive returns -1 (a negative
    ///     radius). pygame's message is <c>SDL_GetError()</c>, whatever SDL error happened to be set last, so it cannot be
    ///     reproduced; this message names the primitive instead.
    /// </summary>
    public sealed class SdlError : Exception {
        /// <summary>An SDL failure with a message.</summary>
        /// <param name="message">What failed.</param>
        public SdlError(string message) : base(message) { }
    }
}
