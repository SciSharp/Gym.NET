#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Numerics;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     Typed reads of an info dict (DR-78): <c>info.Get&lt;double&gt;("x_position")</c> reads a value as the .NET type the
    ///     caller wants, converting what Gymnasium's envs store (NumPy scalars, which arrive as 0-d arrays, and Python
    ///     numbers) where a plain cast would throw.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     An extension rather than a member of <see cref="Info"/>, so it reads any string-keyed dict the same way: an
    ///     <see cref="Info"/>, a <see cref="Metadata"/>, a typed info record's dict (<see cref="EnvInfo"/>). Python's dict has
    ///     no such method; it stands for <c>float(info["x_position"])</c>.
    ///     </para>
    ///     <para>
    ///     The conversions keep every value: a float32 or int scalar read as <see langword="double"/> widens exactly, an
    ///     integer read as a narrower integer is range-checked (<see cref="OverflowException"/>), and a float is never read
    ///     as an integer unless it is a whole number. Anything else, a non-0-d array read as a number included, is refused
    ///     with <see cref="InvalidCastException"/>.
    ///     </para>
    /// </remarks>
    public static class InfoExtensions {
        /// <summary>
        ///     The value under <paramref name="key"/> read as <typeparamref name="T"/>, or <c>default(T)</c> when the key is
        ///     missing: <c>info.Get&lt;double&gt;("x_position")</c>, <c>info.Get&lt;NDArray&gt;("action_mask")</c>,
        ///     <c>info.Get&lt;double?&gt;("x_velocity")</c> (<see langword="null"/> on a reset's info, which lacks it).
        /// </summary>
        /// <typeparam name="T">
        ///     The type to read as: the stored value's own type, a numeric type for a number or a NumPy scalar (a 0-d array),
        ///     <see langword="bool"/> for a bool or a NumPy bool, <see cref="NDArray"/> for an array or a number (a new 0-d
        ///     array then), or any of these as <see cref="Nullable{T}"/>.
        /// </typeparam>
        /// <param name="info">The dict.</param>
        /// <param name="key">The key.</param>
        /// <returns>
        ///     The value as <typeparamref name="T"/>, or <c>default(T)</c> when the key is missing. Footgun: a missing key and a
        ///     stored default read the same (<c>0.0</c> for <see langword="double"/>); read a <see cref="Nullable{T}"/>, or check
        ///     <c>ContainsKey</c>, to tell them apart. An array read as <see cref="NDArray"/> is the info's own (it is disposed
        ///     with the step result that holds it); a number read as <see cref="NDArray"/> is a new array the caller owns.
        /// </returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="info"/> or <paramref name="key"/> is <see langword="null"/>.</exception>
        /// <exception cref="InvalidCastException">
        ///     Thrown when the value cannot be read as <typeparamref name="T"/>: another kind of value, a float with a fraction
        ///     read as an integer, an array of more than one element read as a number, or <c>None</c> read as a non-nullable
        ///     value type.
        /// </exception>
        /// <exception cref="OverflowException">Thrown when an integer does not fit <typeparamref name="T"/>.</exception>
        public static T? Get<T>(this IReadOnlyDictionary<string, object?> info, string key) {
            if (info is null)
                throw new ArgumentNullException(nameof(info));
            if (key is null)
                throw new ArgumentNullException(nameof(key));
            return info.TryGetValue(key, out object? value) ? ConvertTo<T>(value, key) : default;
        }

        /// <summary>
        ///     A stored value as <typeparamref name="T"/> (see <see cref="Get{T}"/>'s rules): what the typed info records
        ///     (<see cref="EnvInfo"/>) read their members with.
        /// </summary>
        /// <typeparam name="T">The type to read as.</typeparam>
        /// <param name="value">The stored value.</param>
        /// <param name="key">The key it was stored under, for the messages.</param>
        /// <returns>The value as <typeparamref name="T"/>.</returns>
        /// <exception cref="InvalidCastException">Thrown when the value cannot be read as <typeparamref name="T"/>.</exception>
        /// <exception cref="OverflowException">Thrown when an integer does not fit <typeparamref name="T"/>.</exception>
        internal static T? ConvertTo<T>(object? value, string key) {
            Type target = typeof(T);
            Type? underlying = Nullable.GetUnderlyingType(target);
            Type effective = underlying ?? target;
            if (value is null) {
                // Python's None: a reference type or a Nullable<T> holds it, a plain value type cannot.
                if (!target.IsValueType || underlying is not null)
                    return default;
                throw new InvalidCastException($"info[{PyBuiltins.Repr(key)}] is None, which {effective.Name} cannot hold; read it as {effective.Name}? instead");
            }

            if (value is T same)
                return same;

            if (effective == typeof(NDArray)) {
                // A number read as an array: NumPy's np.asarray(x), a new 0-d array of the number's .NET type.
                if (IsNumber(value) || value is bool)
                    return (T)(object)NDArray.Scalar(value);
                throw Refuse(value, key, effective);
            }

            // A NumPy scalar (a 0-d array) reads as its element, as float(np.float64(1.5)) does; a larger array is no number.
            object scalar = value;
            if (value is NDArray array) {
                if (array.ndim != 0)
                    throw new InvalidCastException($"info[{PyBuiltins.Repr(key)}] is an array of shape ({string.Join(", ", array.shape)}), not a number; read it as NDArray");
                scalar = array.item();
            }

            if (effective.IsInstanceOfType(scalar))
                return (T)scalar;
            if (effective == typeof(bool))
                throw Refuse(value, key, effective);
            if (IsNumber(scalar) && IsNumericType(effective))
                return (T)ConvertNumber(scalar, effective, key);
            throw Refuse(value, key, effective);
        }

        /// <summary>A number converted to a numeric type without losing it: integers range-checked, floats to integers only when whole.</summary>
        /// <param name="number">The number, any .NET numeric type.</param>
        /// <param name="target">The numeric type.</param>
        /// <param name="key">The key, for the messages.</param>
        /// <returns>The converted number, boxed as <paramref name="target"/>.</returns>
        /// <exception cref="InvalidCastException">Thrown for a float with a fraction (or not finite) read as an integer.</exception>
        /// <exception cref="OverflowException">Thrown when an integer does not fit the target.</exception>
        private static object ConvertNumber(object number, Type target, string key) {
            bool floating = number is double or float or Half or decimal;
            bool integralTarget = !(target == typeof(double) || target == typeof(float) || target == typeof(Half) || target == typeof(decimal));
            if (floating && integralTarget) {
                double d = number is Half h ? (double)h : Convert.ToDouble(number, CultureInfo.InvariantCulture);
                // A whole float reads as an integer (Python's int(2.0)); a fraction would be silently dropped, so it is refused.
                if (double.IsNaN(d) || double.IsInfinity(d) || Math.Truncate(d) != d)
                    throw new InvalidCastException($"info[{PyBuiltins.Repr(key)}] is {PyBuiltins.Repr(number)}, which {target.Name} cannot hold without losing its fraction");
            }

            if (number is BigInteger big)
                number = target == typeof(double) ? (double)big : target == typeof(float) ? (float)big : (object)checked((long)big);
            if (number is Half half)
                number = (double)half;
            // Convert.ChangeType range-checks integers (OverflowException) and widens floats exactly.
            return target == typeof(Half) ? (Half)Convert.ToDouble(number, CultureInfo.InvariantCulture) : Convert.ChangeType(number, target, CultureInfo.InvariantCulture);
        }

        /// <summary>The refusal of a value that has no reading as the requested type.</summary>
        /// <param name="value">The stored value.</param>
        /// <param name="key">The key.</param>
        /// <param name="target">The requested type.</param>
        /// <returns>The exception to throw.</returns>
        private static InvalidCastException Refuse(object value, string key, Type target) =>
            new InvalidCastException($"info[{PyBuiltins.Repr(key)}] is {PyBuiltins.TypeRepr(value)}, which cannot be read as {target.Name}");

        /// <summary>Whether a value is a .NET number (a Python int or float), not a <see langword="bool"/>.</summary>
        /// <param name="value">The value.</param>
        /// <returns><see langword="true"/> for the integer and floating types, <see cref="BigInteger"/> and <see langword="decimal"/>.</returns>
        private static bool IsNumber(object value) =>
            value is sbyte or byte or short or ushort or int or uint or long or ulong or BigInteger or float or double or Half or decimal;

        /// <summary>Whether a type is one of .NET's numeric types.</summary>
        /// <param name="type">The type.</param>
        /// <returns><see langword="true"/> for the integer and floating types and <see langword="decimal"/>.</returns>
        private static bool IsNumericType(Type type) =>
            type == typeof(sbyte) || type == typeof(byte) || type == typeof(short) || type == typeof(ushort) || type == typeof(int) || type == typeof(uint)
            || type == typeof(long) || type == typeof(ulong) || type == typeof(float) || type == typeof(double) || type == typeof(Half) || type == typeof(decimal);
    }
}
