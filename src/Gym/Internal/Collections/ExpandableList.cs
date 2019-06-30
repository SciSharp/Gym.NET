using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;

namespace Gym.Collections {
    /// <summary>
    ///     ExpandableList is used for representing a single value that should be expanded to a expandable <see cref="List{T}"/>.
    /// </summary>
    [DebuggerStepThrough]
    [DebuggerDisplay("{" + nameof(obj) + "}")]
    public class ExpandableList<T> : ExpandableList {
        public new List<T> Expand() {
            return new List<T>(1) {(T) obj};
        }

        /// <inheritdoc />
        public ExpandableList(T obj) : base(obj) { }
    }

    public interface IExpandableList {
        IList Expand();
    }

    /// <summary>
    ///     ExpandableList is used for representing a single value that should be expanded to a expandable <see cref="List{T}"/>.
    /// </summary>
    [DebuggerStepThrough]
    [DebuggerDisplay("{" + nameof(obj) + "}")]
    public class ExpandableList : IList, IEquatable<ExpandableList>, IExpandableList {
        protected readonly object obj;

        public virtual List<T> Expand<T>() {
            return new List<T>(1) {(T) obj};
        }

        public IList Expand() {
            if (obj == null)
                return new List<object>(1) {obj};
            var ret = (IList) Activator.CreateInstance(typeof(List<>).MakeGenericType(obj.GetType()));
            ret.Add(obj);

            return ret;
        }

        public IList ExpandToArray() {
            if (obj == null)
                return new object[1] {obj};
            var arr = Array.CreateInstance(obj.GetType(), 1);
            arr.SetValue(obj, 0);
            return arr;
        }

        /// <inheritdoc />
        public ExpandableList(object obj) {
            this.obj = obj;
        }

        /// <inheritdoc />
        public IEnumerator GetEnumerator() {
            return _yield(obj).GetEnumerator();
        }

        private static IEnumerable _yield(object obj) {
            yield return obj;
        }

        /// <inheritdoc />
        public void CopyTo(Array array, int index) {
            array.SetValue(obj, index);
        }

        /// <inheritdoc />
        public int Count {
            get { return 1; }
        }

        /// <inheritdoc />
        public object SyncRoot {
            get { return null; }
        }

        /// <inheritdoc />
        public bool IsSynchronized {
            get { return true; }
        }

        /// <inheritdoc />
        public int Add(object value) {
            throw new NotSupportedException();
        }

        /// <inheritdoc />
        public bool Contains(object value) {
            return Equals(value, obj);
        }

        /// <inheritdoc />
        public void Clear() {
            throw new NotSupportedException();
        }

        /// <inheritdoc />
        public int IndexOf(object value) {
            return Contains(value) ? 0 : -1;
        }

        /// <inheritdoc />
        public void Insert(int index, object value) {
            throw new NotSupportedException();
        }

        /// <inheritdoc />
        public void Remove(object value) {
            throw new NotSupportedException();
        }

        /// <inheritdoc />
        public void RemoveAt(int index) {
            throw new NotSupportedException();
        }

        /// <inheritdoc />
        public object this[int index] {
            get {
                if (index != 0)
                    throw new ArgumentOutOfRangeException();
                return obj;
            }
            set => throw new NotSupportedException();
        }

        /// <inheritdoc />
        public bool IsReadOnly => true;

        /// <inheritdoc />
        public bool IsFixedSize => true;

        /// <inheritdoc />
        public override string ToString() {
            return obj?.ToString() ?? "null";
        }

        /// <inheritdoc />
        public bool Equals(ExpandableList other) {
            if (ReferenceEquals(null, other))
                return false;
            if (ReferenceEquals(this, other))
                return true;
            return Equals(obj, other.obj);
        }

        /// <inheritdoc />
        public override bool Equals(object obj) {
            if (ReferenceEquals(null, obj))
                return false;
            if (ReferenceEquals(this, obj))
                return true;
            if (obj.GetType() != this.GetType())
                return false;
            return Equals((ExpandableList) obj);
        }

        /// <inheritdoc />
        public override int GetHashCode() {
            return (obj != null ? obj.GetHashCode() : 0);
        }

        public static bool operator ==(ExpandableList left, ExpandableList right) {
            return Equals(left, right);
        }

        public static bool operator !=(ExpandableList left, ExpandableList right) {
            return !Equals(left, right);
        }
    }
}