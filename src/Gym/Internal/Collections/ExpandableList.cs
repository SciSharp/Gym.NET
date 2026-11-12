// Gym.Internal is 0.2's code, [Obsolete] since 1.0 and removed in 2.0 with Gym.Legacy (DR-56): it keeps no nullable
// context (DR-54), rather than 89 annotations for code on its way out.
#nullable disable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace Gym.Collections {
    /// <summary>
    ///     ExpandableList is used for representing a single value that should be expanded to a expandable <see cref="List{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of the one value, and of the list it expands to.</typeparam>
    /// <remarks>
    ///     Obsolete with the legacy <see cref="Dict"/> (DR-56) and removed in 2.0: use <see cref="List{T}"/>. Footgun:
    ///     <see cref="Expand()"/> hides the base class's, so through an <see cref="ExpandableList"/> reference the base
    ///     version runs and returns a list of the value's run-time type instead.
    /// </remarks>
    [DebuggerStepThrough]
    [DebuggerDisplay("{" + nameof(obj) + "}")]
    [Obsolete("Part of the legacy Dict's merging: use List<T>. Removed in 2.0 with Gym.Legacy.")]
    public class ExpandableList<T> : ExpandableList {
        /// <summary>A new growable list holding the one value, typed <typeparamref name="T"/>.</summary>
        /// <returns>A new <see cref="List{T}"/> of one item.</returns>
        /// <exception cref="InvalidCastException">When the value is not a <typeparamref name="T"/> (only possible through the base constructor).</exception>
        public new List<T> Expand() {
            return new List<T>(1) {(T) obj};
        }

        /// <summary>Wraps one value.</summary>
        /// <param name="obj">The value; <see langword="null"/> is allowed.</param>
        public ExpandableList(T obj) : base(obj) { }
    }

    /// <summary>
    ///     A single value that a merge turns into a real list (the legacy deep merge's marker for "append to me").
    /// </summary>
    /// <remarks>Obsolete with the legacy <see cref="Dict"/> (DR-56) and removed in 2.0.</remarks>
    [Obsolete("Part of the legacy Dict's merging: use List<T>. Removed in 2.0 with Gym.Legacy.")]
    public interface IExpandableList {
        /// <summary>A new growable list holding the value.</summary>
        /// <returns>The list, of the value's run-time type when the value is not <see langword="null"/>.</returns>
        IList Expand();
    }

    /// <summary>
    ///     ExpandableList is used for representing a single value that should be expanded to a expandable <see cref="List{T}"/>.
    /// </summary>
    /// <remarks>
    ///     A read-only, fixed-size <see cref="IList"/> of exactly one item: every mutating member throws
    ///     <see cref="NotSupportedException"/>. Obsolete with the legacy <see cref="Dict"/> (DR-56) and removed in 2.0: use
    ///     <see cref="List{T}"/>. Footgun: <see cref="IsSynchronized"/> says <see langword="true"/> while
    ///     <see cref="SyncRoot"/> is <see langword="null"/>, so <c>lock (list.SyncRoot)</c> throws.
    /// </remarks>
    [DebuggerStepThrough]
    [DebuggerDisplay("{" + nameof(obj) + "}")]
    [Obsolete("Part of the legacy Dict's merging: use List<T>. Removed in 2.0 with Gym.Legacy.")]
    public class ExpandableList : IList, IEquatable<ExpandableList>, IExpandableList {
        /// <summary>The one value; <see langword="null"/> is allowed.</summary>
        protected readonly object obj;

        /// <summary>A new growable list holding the one value, typed <typeparamref name="T"/>.</summary>
        /// <typeparam name="T">The list's element type.</typeparam>
        /// <returns>A new <see cref="List{T}"/> of one item.</returns>
        /// <exception cref="InvalidCastException">When the value is not a <typeparamref name="T"/>.</exception>
        public virtual List<T> Expand<T>() {
            return new List<T>(1) {(T) obj};
        }

        /// <summary>
        ///     A new growable list holding the value, its element type the value's run-time type (made by reflection), or a
        ///     <see cref="List{T}"/> of <see cref="object"/> for a <see langword="null"/> value.
        /// </summary>
        /// <returns>The list of one item.</returns>
        public IList Expand() {
            if (obj == null)
                return new List<object>(1) {obj};
            var ret = (IList) Activator.CreateInstance(typeof(List<>).MakeGenericType(obj.GetType()));
            ret.Add(obj);

            return ret;
        }

        /// <summary>
        ///     A one-item array of the value's run-time type (an <see cref="object"/> array for a <see langword="null"/>
        ///     value). Arrays are fixed size: the merge cannot append to it.
        /// </summary>
        /// <returns>The array.</returns>
        public IList ExpandToArray() {
            if (obj == null)
                return new object[1] {obj};
            var arr = Array.CreateInstance(obj.GetType(), 1);
            arr.SetValue(obj, 0);
            return arr;
        }

        /// <summary>Wraps one value.</summary>
        /// <param name="obj">The value; <see langword="null"/> is allowed.</param>
        public ExpandableList(object obj) {
            this.obj = obj;
        }

        /// <summary>Enumerates the one value.</summary>
        /// <returns>An enumerator yielding the value once.</returns>
        public IEnumerator GetEnumerator() {
            return _yield(obj).GetEnumerator();
        }

        /// <summary>The value as a one-item sequence.</summary>
        /// <param name="obj">The value.</param>
        /// <returns>The sequence.</returns>
        private static IEnumerable _yield(object obj) {
            yield return obj;
        }

        /// <summary>Writes the value into <paramref name="array"/> at <paramref name="index"/>.</summary>
        /// <param name="array">The destination.</param>
        /// <param name="index">The position to write.</param>
        /// <exception cref="NullReferenceException">When <paramref name="array"/> is <see langword="null"/>.</exception>
        /// <exception cref="IndexOutOfRangeException">When <paramref name="index"/> is outside <paramref name="array"/>.</exception>
        /// <exception cref="InvalidCastException">When the value does not fit the array's element type.</exception>
        public void CopyTo(Array array, int index) {
            array.SetValue(obj, index);
        }

        /// <summary>Always 1.</summary>
        public int Count {
            get { return 1; }
        }

        /// <summary>Always <see langword="null"/>, though <see cref="IsSynchronized"/> says <see langword="true"/>: do not lock on it.</summary>
        public object SyncRoot {
            get { return null; }
        }

        /// <summary>Always <see langword="true"/>: the list never changes, so reading it from several threads is safe.</summary>
        public bool IsSynchronized {
            get { return true; }
        }

        /// <summary>Always throws: the list is read-only.</summary>
        /// <param name="value">Ignored.</param>
        /// <returns>Never returns.</returns>
        /// <exception cref="NotSupportedException">Always.</exception>
        public int Add(object value) {
            throw new NotSupportedException();
        }

        /// <summary>Whether <paramref name="value"/> equals the one value (<see cref="object.Equals(object, object)"/>).</summary>
        /// <param name="value">The value to look for.</param>
        /// <returns><see langword="true"/> when it equals the list's value.</returns>
        public bool Contains(object value) {
            return Equals(value, obj);
        }

        /// <summary>Always throws: the list is read-only.</summary>
        /// <exception cref="NotSupportedException">Always.</exception>
        public void Clear() {
            throw new NotSupportedException();
        }

        /// <summary>0 when <paramref name="value"/> equals the one value, otherwise -1.</summary>
        /// <param name="value">The value to look for.</param>
        /// <returns>0 or -1.</returns>
        public int IndexOf(object value) {
            return Contains(value) ? 0 : -1;
        }

        /// <summary>Always throws: the list is read-only.</summary>
        /// <param name="index">Ignored.</param>
        /// <param name="value">Ignored.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        public void Insert(int index, object value) {
            throw new NotSupportedException();
        }

        /// <summary>Always throws: the list is read-only.</summary>
        /// <param name="value">Ignored.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        public void Remove(object value) {
            throw new NotSupportedException();
        }

        /// <summary>Always throws: the list is read-only.</summary>
        /// <param name="index">Ignored.</param>
        /// <exception cref="NotSupportedException">Always.</exception>
        public void RemoveAt(int index) {
            throw new NotSupportedException();
        }

        /// <summary>The one value at index 0; setting always throws.</summary>
        /// <param name="index">Must be 0.</param>
        /// <returns>The value.</returns>
        /// <exception cref="ArgumentOutOfRangeException">When reading an index other than 0.</exception>
        /// <exception cref="NotSupportedException">When setting: the list is read-only.</exception>
        public object this[int index] {
            get {
                if (index != 0)
                    throw new ArgumentOutOfRangeException();
                return obj;
            }
            set => throw new NotSupportedException();
        }

        /// <summary>Always <see langword="true"/>.</summary>
        public bool IsReadOnly => true;

        /// <summary>Always <see langword="true"/>: the merge expands it instead of appending to it.</summary>
        public bool IsFixedSize => true;

        /// <summary>The value's text, or <c>null</c> for a <see langword="null"/> value.</summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            return obj?.ToString() ?? "null";
        }

        /// <summary>Whether <paramref name="other"/> wraps an equal value.</summary>
        /// <param name="other">The list to compare with.</param>
        /// <returns><see langword="true"/> when both wrap equal values (or are the same instance).</returns>
        public bool Equals(ExpandableList other) {
            if (ReferenceEquals(null, other))
                return false;
            if (ReferenceEquals(this, other))
                return true;
            return Equals(obj, other.obj);
        }

        /// <summary>Whether <paramref name="obj"/> is an <see cref="ExpandableList"/> of the exact same type wrapping an equal value.</summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns><see langword="true"/> when equal; an <see cref="ExpandableList{T}"/> never equals a plain one.</returns>
        public override bool Equals(object obj) {
            if (ReferenceEquals(null, obj))
                return false;
            if (ReferenceEquals(this, obj))
                return true;
            if (obj.GetType() != this.GetType())
                return false;
            return Equals((ExpandableList) obj);
        }

        /// <summary>The value's hash, 0 for <see langword="null"/>.</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() {
            return (obj != null ? obj.GetHashCode() : 0);
        }

        /// <summary>Value equality, as <see cref="object.Equals(object, object)"/> over the two lists.</summary>
        /// <param name="left">The first list.</param>
        /// <param name="right">The second list.</param>
        /// <returns><see langword="true"/> when both are <see langword="null"/> or wrap equal values.</returns>
        public static bool operator ==(ExpandableList left, ExpandableList right) {
            return Equals(left, right);
        }

        /// <summary>The negation of <see cref="operator ==(ExpandableList, ExpandableList)"/>.</summary>
        /// <param name="left">The first list.</param>
        /// <param name="right">The second list.</param>
        /// <returns><see langword="true"/> when they differ.</returns>
        public static bool operator !=(ExpandableList left, ExpandableList right) {
            return !Equals(left, right);
        }
    }
}
