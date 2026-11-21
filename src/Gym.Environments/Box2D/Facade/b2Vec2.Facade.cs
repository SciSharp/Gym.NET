/*
* pybox2d -- http://pybox2d.googlecode.com
*
* Copyright (c) 2010 Ken Lauer / sirkne at gmail dot com
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
// Altered source version (Gym.NET, 2026-11-10): the b2Vec2, b2Rot and b2Transform extensions of pybox2d 2.3.10's
// Box2D/Box2D_math.i and its sequence typemap (Box2D_typemaps.i), translated to C#: indexing, the tuple conversion,
// scalar operators taking Python floats (rounded to float32 first, as SWIG's conversion does), cross/dot, repr.
using System;
using System.Globalization;
using NumSharp;

namespace Gym.Box2D {
    public partial struct b2Vec2 {
        /// <summary>
        ///     A coordinate by index, as pybox2d's <c>__getitem__</c>/<c>__setitem__</c>: 0 is <see cref="x"/>, 1 is
        ///     <see cref="y"/>.
        /// </summary>
        /// <param name="i">0 or 1.</param>
        /// <returns>The coordinate.</returns>
        /// <exception cref="IndexError">Thrown for any other index, with pybox2d's message <c>Index must be in (0,1)</c>.</exception>
        public float this[int i] {
            readonly get => i switch {
                0 => x,
                1 => y,
                _ => throw new IndexError("Index must be in (0,1)"),
            };
            set {
                if (i == 0)
                    x = value;
                else if (i == 1)
                    y = value;
                else
                    throw new IndexError("Index must be in (0,1)");
            }
        }

        /// <summary>The length (pybox2d's <c>length</c> property).</summary>
        public readonly float length => Length();

        /// <summary>The squared length (pybox2d's <c>lengthSquared</c>).</summary>
        public readonly float lengthSquared => LengthSquared();

        /// <summary>Whether both coordinates are finite (pybox2d's <c>valid</c>).</summary>
        public readonly bool valid => IsValid();

        /// <summary>The skew vector <c>(-y, x)</c> (pybox2d's <c>skew</c>).</summary>
        public readonly b2Vec2 skew => Skew();

        /// <summary>The coordinates as a tuple of Python floats (pybox2d's <c>tuple</c>); each float32 widens exactly.</summary>
        public readonly (double x, double y) tuple => (x, y);

        /// <summary>The 2-D cross product with another vector, in float32 (pybox2d's <c>cross</c>).</summary>
        /// <param name="other">The other vector.</param>
        /// <returns><c>x * other.y - y * other.x</c>.</returns>
        public readonly float cross(b2Vec2 other) => x * other.y - y * other.x;

        /// <summary>The cross product with a scalar, a vector (pybox2d's <c>cross(float)</c>).</summary>
        /// <param name="s">The scalar, rounded to float32 as pybox2d's argument conversion does.</param>
        /// <returns><c>(s * y, -s * x)</c>.</returns>
        /// <exception cref="OverflowError">Thrown when <paramref name="s"/> lies beyond the float32 range.</exception>
        public readonly b2Vec2 cross(double s) {
            float f = Box2DConvert.ToFloat32(s, "b2Vec2_cross", 2);
            return new b2Vec2(f * y, -f * x);
        }

        /// <summary>The dot product with another vector, in float32 (pybox2d's <c>dot</c>).</summary>
        /// <param name="other">The other vector.</param>
        /// <returns><c>x * other.x + y * other.y</c>.</returns>
        public readonly float dot(b2Vec2 other) => x * other.x + y * other.y;

        /// <summary>A copy (pybox2d's <c>copy</c>); C# structs copy on assignment, so this exists for readability.</summary>
        /// <returns>The same coordinates.</returns>
        public readonly b2Vec2 copy() => new b2Vec2(x, y);

        /// <summary>Unpacks the coordinates, as Python's <c>x, y = vec</c> iterates a <c>b2Vec2</c>.</summary>
        /// <param name="x">Receives <see cref="x"/>.</param>
        /// <param name="y">Receives <see cref="y"/>.</param>
        public readonly void Deconstruct(out float x, out float y) {
            x = this.x;
            y = this.y;
        }

        /// <summary>
        ///     A Python sequence of two numbers becoming a vector, as pybox2d's typemap converts a tuple argument: each element
        ///     is rounded to float32.
        /// </summary>
        /// <param name="value">The coordinates as Python floats.</param>
        /// <returns>The vector.</returns>
        /// <exception cref="TypeError">
        ///     Thrown for an element beyond the float32 range (an infinity included), with pybox2d's message
        ///     <c>Converting from sequence to b2Vec2, expected int/float arguments index 0</c> (or index 1).
        /// </exception>
        public static implicit operator b2Vec2((double x, double y) value) {
            // The typemap converts element by element and fails on the first bad one, naming its index.
            if (!Box2DConvert.FitsFloat32(value.x))
                throw new TypeError("Converting from sequence to b2Vec2, expected int/float arguments index 0");
            if (!Box2DConvert.FitsFloat32(value.y))
                throw new TypeError("Converting from sequence to b2Vec2, expected int/float arguments index 1");
            return new b2Vec2((float)value.x, (float)value.y);
        }

        /// <summary>Scales by a Python float, as pybox2d's <c>__mul__</c>: the scalar rounds to float32, then float32 products.</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The scale.</param>
        /// <returns><c>(a.x * (float)s, a.y * (float)s)</c>.</returns>
        /// <exception cref="OverflowError">Thrown when <paramref name="s"/> lies beyond the float32 range.</exception>
        public static b2Vec2 operator *(b2Vec2 a, double s) {
            float f = Box2DConvert.ToFloat32(s, "b2Vec2___mul__", 2);
            return new b2Vec2(a.x * f, a.y * f);
        }

        /// <summary>Scales by a Python float from the left, as pybox2d's <c>__rmul__</c> (same bits as <c>a * s</c>).</summary>
        /// <param name="s">The scale.</param>
        /// <param name="a">The vector.</param>
        /// <returns><c>(a.x * (float)s, a.y * (float)s)</c>.</returns>
        /// <exception cref="OverflowError">Thrown when <paramref name="s"/> lies beyond the float32 range.</exception>
        public static b2Vec2 operator *(double s, b2Vec2 a) {
            float f = Box2DConvert.ToFloat32(s, "b2Vec2___rmul__", 2);
            return new b2Vec2(a.x * f, a.y * f);
        }

        /// <summary>Divides by a Python float, as pybox2d's <c>__truediv__</c>: the divisor rounds to float32 first.</summary>
        /// <param name="a">The vector.</param>
        /// <param name="s">The divisor.</param>
        /// <returns><c>(a.x / (float)s, a.y / (float)s)</c>.</returns>
        /// <exception cref="OverflowError">Thrown when <paramref name="s"/> lies beyond the float32 range.</exception>
        public static b2Vec2 operator /(b2Vec2 a, double s) {
            float f = Box2DConvert.ToFloat32(s, "b2Vec2___truediv__", 2);
            return new b2Vec2(a.x / f, a.y / f);
        }

        /// <summary>pybox2d's <c>repr</c>: <c>b2Vec2(%g,%g)</c>.</summary>
        /// <returns>The text.</returns>
        public override readonly string ToString() => $"b2Vec2({Box2DConvert.FormatG(x)},{Box2DConvert.FormatG(y)})";
    }

    public partial struct b2Rot {
        /// <summary>The angle in radians, <c>atan2f(s, c)</c> on read and <c>sinf</c>/<c>cosf</c> on write (pybox2d's <c>angle</c>).</summary>
        public float angle {
            readonly get => GetAngle();
            set => Set(value);
        }

        /// <summary>Rotates a vector, as pybox2d's <c>b2Rot.__mul__</c>.</summary>
        /// <param name="q">The rotation.</param>
        /// <param name="v">The vector.</param>
        /// <returns><c>(q.c * v.x - q.s * v.y, q.s * v.x + q.c * v.y)</c>.</returns>
        public static b2Vec2 operator *(b2Rot q, b2Vec2 v) => b2Math.b2Mul(q, v);
    }

    public partial struct b2Transform {
        /// <summary>The translation (pybox2d's <c>position</c>, Box2D's <c>p</c>).</summary>
        public b2Vec2 position {
            readonly get => p;
            set => p = value;
        }

        /// <summary>The rotation's angle in radians (pybox2d's <c>angle</c>).</summary>
        public float angle {
            readonly get => q.GetAngle();
            set => q.Set(value);
        }

        /// <summary>
        ///     Maps a local point into the frame, as pybox2d's <c>b2Transform.__mul__</c> computes it:
        ///     <c>(q.c * v.x - q.s * v.y) + p.x</c> and <c>(q.s * v.x + q.c * v.y) + p.y</c> in float32. Gymnasium's
        ///     renderers place every fixture vertex with it.
        /// </summary>
        /// <param name="t">The transform.</param>
        /// <param name="v">The local point.</param>
        /// <returns>The world point.</returns>
        public static b2Vec2 operator *(b2Transform t, b2Vec2 v) {
            float x = (t.q.c * v.x - t.q.s * v.y) + t.p.x;
            float y = (t.q.s * v.x + t.q.c * v.y) + t.p.y;
            return new b2Vec2(x, y);
        }
    }

    /// <summary>
    ///     The argument conversions of pybox2d's SWIG layer: a Python float passed where C++ takes a <c>float32</c> rounds
    ///     to the nearest float32, and values beyond its range are refused rather than becoming infinities.
    /// </summary>
    /// <remarks>
    ///     SWIG's <c>SWIG_AsVal_float</c> refuses <c>x &lt; -FLT_MAX || x &gt; FLT_MAX</c> (so the infinities) and lets NaN
    ///     through, which then propagates as NaN.
    /// </remarks>
    public static class Box2DConvert {
        /// <summary>Whether a Python float converts to float32 without SWIG's overflow refusal (NaN does).</summary>
        /// <param name="value">The value.</param>
        /// <returns>False for values beyond ±FLT_MAX, the infinities included.</returns>
        public static bool FitsFloat32(double value) => !(value < -float.MaxValue || value > float.MaxValue);

        /// <summary>Converts a Python float to float32 as SWIG does for a <c>float32</c> parameter.</summary>
        /// <param name="value">The value.</param>
        /// <param name="method">The SWIG wrapper name the error names (<c>b2Vec2___mul__</c>).</param>
        /// <param name="argument">The 1-based argument position the error names.</param>
        /// <returns>The nearest float32.</returns>
        /// <exception cref="OverflowError">
        ///     Thrown, as Python's <c>OverflowError</c>, for a value beyond the float32 range, with SWIG's message
        ///     <c>in method '…', argument N of type 'float32'</c>.
        /// </exception>
        public static float ToFloat32(double value, string method, int argument) {
            if (!FitsFloat32(value))
                throw new OverflowError($"in method '{method}', argument {argument} of type 'float32'");
            return (float)value;
        }

        /// <summary>
        ///     Sets one keyword argument of a definition or shape constructor as pybox2d's <c>_init_kwargs</c> does: an
        ///     error the assignment raises is raised again as the same type, with the message
        ///     <c>Failed on kwargs for {className}.{key}: {message}</c> and the original as its inner exception (Python's
        ///     implicit chaining).
        /// </summary>
        /// <remarks>
        ///     Only the facade's own error types are rewrapped (<see cref="AssertionError"/>, <see cref="ValueError"/>,
        ///     <see cref="TypeError"/>, <see cref="OverflowError"/>); anything else propagates untouched. Footgun: a
        ///     vector coordinate is converted where the caller builds the <see cref="b2Vec2"/>, before the constructor
        ///     runs, so its <see cref="TypeError"/> carries no prefix, unlike pybox2d's (a deviation C#'s argument
        ///     evaluation forces).
        /// </remarks>
        /// <param name="className">pybox2d's class name (<c>b2BodyDef</c>, <c>b2PolygonShape</c>).</param>
        /// <param name="key">The keyword (<c>angle</c>, <c>vertices</c>).</param>
        /// <param name="assign">The assignment.</param>
        /// <exception cref="AssertionError">Thrown, prefixed, when the assignment fails a Box2D assertion.</exception>
        /// <exception cref="ValueError">Thrown, prefixed, when the assignment refuses a value.</exception>
        /// <exception cref="TypeError">Thrown, prefixed, when the assignment refuses a type.</exception>
        /// <exception cref="OverflowError">Thrown, prefixed, for a scalar beyond the float32 range.</exception>
        internal static void Kwarg(string className, string key, Action assign) =>
            Rewrapped(assign, $"Failed on kwargs for {className}.{key}: ");

        /// <summary>
        ///     Sets one keyword argument of a body's fixture shortcut (<c>CreatePolygonFixture</c>, <c>CreateEdgeFixture</c>,
        ///     <c>CreateCircleFixture</c>) as pybox2d's <c>__CreateShapeFixture</c> does: an error is raised again as the
        ///     same type, with the message <c>Failed on kwargs, class="{className}" key="{key}": {message}</c>.
        /// </summary>
        /// <remarks>The shortcut's own format, not <see cref="Kwarg"/>'s: pybox2d words the two differently.</remarks>
        /// <param name="className">The class the keyword went to: the shape's (<c>b2CircleShape</c>) or <c>b2FixtureDef</c>.</param>
        /// <param name="key">The keyword.</param>
        /// <param name="assign">The assignment.</param>
        /// <exception cref="AssertionError">Thrown, prefixed, when the assignment fails a Box2D assertion.</exception>
        /// <exception cref="ValueError">Thrown, prefixed, when the assignment refuses a value.</exception>
        /// <exception cref="TypeError">Thrown, prefixed, when the assignment refuses a type.</exception>
        /// <exception cref="OverflowError">Thrown, prefixed, for a scalar beyond the float32 range.</exception>
        internal static void ShapeFixtureKwarg(string className, string key, Action assign) =>
            Rewrapped(assign, $"Failed on kwargs, class=\"{className}\" key=\"{key}\": ");

        /// <summary>Runs an assignment and raises its failure again as the same type, its message after a prefix.</summary>
        /// <param name="assign">The assignment.</param>
        /// <param name="prefix">The text before the original message.</param>
        /// <exception cref="AssertionError">Thrown, prefixed, when the assignment fails a Box2D assertion.</exception>
        /// <exception cref="ValueError">Thrown, prefixed, when the assignment refuses a value.</exception>
        /// <exception cref="TypeError">Thrown, prefixed, when the assignment refuses a type.</exception>
        /// <exception cref="OverflowError">Thrown, prefixed, for a scalar beyond the float32 range.</exception>
        private static void Rewrapped(Action assign, string prefix) {
            try {
                assign();
            } catch (AssertionError ex) {
                throw new AssertionError(prefix + ex.Message, ex);
            } catch (ValueError ex) {
                // A facade ValueError never names a parameter, so its Message is the bare text pybox2d formats.
                throw new ValueError(prefix + ex.Message, ex);
            } catch (TypeError ex) {
                throw new TypeError(prefix + ex.Message, ex);
            } catch (OverflowError ex) {
                throw new OverflowError(prefix + ex.Message, ex);
            }
        }

        /// <summary>Formats a float as C's and Python's <c>%g</c>: 6 significant digits, trailing zeros dropped, exponent as <c>e+XX</c>.</summary>
        /// <param name="value">The value.</param>
        /// <returns>The text.</returns>
        public static string FormatG(double value) {
            if (double.IsNaN(value))
                return "nan";
            if (double.IsInfinity(value))
                return value > 0 ? "inf" : "-inf";
            if (value == 0)
                return BitConverter.DoubleToInt64Bits(value) < 0 ? "-0" : "0";
            // %g picks scientific notation when the exponent is below -4 or at least the precision (6).
            string e = value.ToString("e5", CultureInfo.InvariantCulture);
            int exponent = int.Parse(e.Substring(e.IndexOf('e') + 1), CultureInfo.InvariantCulture);
            if (exponent < -4 || exponent >= 6) {
                string mantissa = e.Substring(0, e.IndexOf('e'));
                if (mantissa.Contains('.'))
                    mantissa = mantissa.TrimEnd('0').TrimEnd('.');
                return $"{mantissa}e{(exponent < 0 ? '-' : '+')}{Math.Abs(exponent):00}";
            }

            string fixedText = value.ToString("F" + Math.Max(0, 5 - exponent).ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
            if (fixedText.Contains('.'))
                fixedText = fixedText.TrimEnd('0').TrimEnd('.');
            return fixedText;
        }
    }
}
