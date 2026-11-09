#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     Python's <c>collections.deque</c> as the wrappers use it: a sequence that keeps the last <see cref="MaxLen"/> items,
    ///     dropping the oldest as new ones are appended (<c>RecordEpisodeStatistics</c>'s <c>return_queue</c>). Indexing and
    ///     <c>repr</c> follow Python: <c>queue[-1]</c> is the newest, and it prints <c>deque([1.0, 2.0], maxlen=100)</c>.
    /// </summary>
    /// <typeparam name="T">The item type.</typeparam>
    /// <remarks>
    ///     A ring buffer: appending is O(1) and never allocates once the deque is full. Not synchronized, like Python's
    ///     (whose single appends happen to be atomic under the GIL); one wrapper fills it from one thread.
    ///     Reference: <see href="https://docs.python.org/3.12/library/collections.html#collections.deque">collections.deque</see>.
    /// </remarks>
    public sealed class Deque<T> : IReadOnlyList<T> {
        /// <summary>The items, oldest at <see cref="_head"/>, wrapping around; grown on demand while there is no bound.</summary>
        private T[] _items;

        /// <summary>The index of the oldest item in <see cref="_items"/>.</summary>
        private int _head;

        /// <summary>A deque of at most <paramref name="maxLen"/> items, or unbounded for <see langword="null"/> (Python's <c>deque(maxlen=n)</c>).</summary>
        /// <param name="maxLen">The bound; 0 keeps nothing, as in Python.</param>
        /// <exception cref="ValueError">Thrown for a negative bound: <c>maxlen must be non-negative</c>.</exception>
        public Deque(int? maxLen = null) {
            if (maxLen < 0)
                throw new ValueError("maxlen must be non-negative");
            MaxLen = maxLen;
            _items = new T[maxLen is { } bound ? Math.Min(bound, 16) : 16];
        }

        /// <summary>The bound (<c>maxlen</c>), or <see langword="null"/> when unbounded.</summary>
        public int? MaxLen { get; }

        /// <summary>The number of items held, <c>len(d)</c>.</summary>
        public int Count { get; private set; }

        /// <summary>
        ///     Python's <c>d[i]</c>: the <paramref name="index"/>-th oldest item, counting from the newest for a negative index
        ///     (<c>-1</c> is the newest).
        /// </summary>
        /// <param name="index">The index, negative from the end.</param>
        /// <returns>The item.</returns>
        /// <exception cref="IndexError">Thrown outside the deque: <c>deque index out of range</c>.</exception>
        public T this[int index] {
            get {
                int position = index < 0 ? index + Count : index;
                if (position < 0 || position >= Count)
                    throw new IndexError("deque index out of range");
                return _items[(_head + position) % _items.Length];
            }
        }

        /// <summary>
        ///     Python's <c>d.append(item)</c>: adds the item as the newest; a full bounded deque drops its oldest item first, so
        ///     it always holds the last <see cref="MaxLen"/> appended.
        /// </summary>
        /// <param name="item">The item.</param>
        public void Append(T item) {
            if (MaxLen == 0)
                return;
            if (MaxLen is { } bound && Count == bound) {
                // Full: the newest overwrites the oldest, which the head then moves past.
                _items[_head] = item;
                _head = (_head + 1) % _items.Length;
                return;
            }

            if (Count == _items.Length)
                Grow();
            _items[(_head + Count) % _items.Length] = item;
            Count++;
        }

        /// <summary>Python's <c>d.clear()</c>: drops every item.</summary>
        public void Clear() {
            Array.Clear(_items, 0, _items.Length);
            _head = 0;
            Count = 0;
        }

        /// <summary>Enumerates the items, oldest first.</summary>
        /// <returns>The enumerator.</returns>
        public IEnumerator<T> GetEnumerator() {
            for (int i = 0; i < Count; i++)
                yield return _items[(_head + i) % _items.Length];
        }

        /// <inheritdoc cref="GetEnumerator"/>
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Python's <c>repr(d)</c>: <c>deque([1.0, 2.0], maxlen=100)</c>, or <c>deque([1.0, 2.0])</c> when unbounded.</summary>
        /// <returns>The text.</returns>
        public override string ToString() {
            string items = "[" + string.Join(", ", this.Select(item => PyBuiltins.Repr(item))) + "]";
            return MaxLen is { } bound ? $"deque({items}, maxlen={bound.ToString(CultureInfo.InvariantCulture)})" : $"deque({items})";
        }

        /// <summary>Doubles the storage of an unbounded (or not yet full) deque, unwrapping the items to start at 0.</summary>
        private void Grow() {
            int capacity = _items.Length * 2;
            if (MaxLen is { } bound)
                capacity = Math.Min(capacity, bound);
            var items = new T[capacity];
            for (int i = 0; i < Count; i++)
                items[i] = _items[(_head + i) % _items.Length];
            _items = items;
            _head = 0;
        }
    }
}
