#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using NumSharp;

namespace Gym.Utils {
    /// <summary>
    ///     The part of Python's <c>difflib</c> Gym's error messages need: <c>get_close_matches</c>, which picks the
    ///     <c>Did you mean: `CartPole`?</c> suggestion of the registry's errors. The suggestion is part of the message tests
    ///     assert on, so the matcher is CPython's algorithm step for step, not an edit distance that would often agree.
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/python/cpython/blob/v3.12.12/Lib/difflib.py#L666">Lib/difflib.py (get_close_matches)</see>.</remarks>
    internal static class Difflib {
        /// <summary>
        ///     The best "good enough" matches of <paramref name="word"/> among <paramref name="possibilities"/> (Python's
        ///     <c>difflib.get_close_matches</c>): those whose <see cref="SequenceMatcher.Ratio"/> reaches
        ///     <paramref name="cutoff"/>, best first; equal scores are ordered by the larger string first, as Python's
        ///     <c>heapq.nlargest</c> orders <c>(score, x)</c> pairs, so the result does not depend on the order of a set.
        /// </summary>
        /// <param name="word">The word to match.</param>
        /// <param name="possibilities">The candidates.</param>
        /// <param name="n">The most matches to return; must be positive.</param>
        /// <param name="cutoff">The lowest similarity accepted, in <c>[0, 1]</c>.</param>
        /// <returns>At most <paramref name="n"/> candidates, best first.</returns>
        /// <exception cref="ValueError">Thrown when <paramref name="n"/> is not positive (<c>n must be &gt; 0: 0</c>) or <paramref name="cutoff"/> is outside <c>[0, 1]</c>.</exception>
        public static IReadOnlyList<string> GetCloseMatches(string word, IEnumerable<string> possibilities, int n = 3, double cutoff = 0.6) {
            if (!(n > 0))
                throw new ValueError($"n must be > 0: {n.ToString(CultureInfo.InvariantCulture)}");
            if (!(0.0 <= cutoff && cutoff <= 1.0))
                throw new ValueError($"cutoff must be in [0.0, 1.0]: {PyBuiltins.Repr(cutoff)}");

            var result = new List<(double Score, string Text, int[] CodePoints)>();
            var matcher = new SequenceMatcher();
            matcher.SetSeq2(CodePoints(word));
            foreach (string x in possibilities) {
                int[] candidate = CodePoints(x);
                matcher.SetSeq1(candidate);
                // The cheap upper bounds first, as Python does; the order matters only for speed.
                if (matcher.RealQuickRatio() >= cutoff && matcher.QuickRatio() >= cutoff && matcher.Ratio() >= cutoff)
                    result.Add((matcher.Ratio(), x, candidate));
            }

            // heapq.nlargest(n, result): descending (score, x), a stable sort keeping the input order of equal pairs.
            return result
                .Select((entry, index) => (entry, index))
                .OrderByDescending(pair => pair.entry.Score)
                .ThenByDescending(pair => pair.entry.CodePoints, CodePointComparer.Instance)
                .ThenBy(pair => pair.index)
                .Take(n)
                .Select(pair => pair.entry.Text)
                .ToList();
        }

        /// <summary>A string as Python sees it, a sequence of code points (a surrogate pair is one element).</summary>
        /// <param name="text">The string.</param>
        /// <returns>The code points.</returns>
        internal static int[] CodePoints(string text) => text.EnumerateRunes().Select(rune => rune.Value).ToArray();

        /// <summary>Python's string order: code point by code point, a prefix before its extensions.</summary>
        private sealed class CodePointComparer : IComparer<int[]> {
            /// <summary>The single instance.</summary>
            public static readonly CodePointComparer Instance = new();

            /// <summary>Compares two code-point sequences lexicographically.</summary>
            /// <param name="x">The first sequence.</param>
            /// <param name="y">The second sequence.</param>
            /// <returns>Negative, zero or positive, as <paramref name="x"/> sorts before, with or after <paramref name="y"/>.</returns>
            public int Compare(int[]? x, int[]? y) {
                if (ReferenceEquals(x, y))
                    return 0;
                if (x is null)
                    return -1;
                if (y is null)
                    return 1;
                int length = Math.Min(x.Length, y.Length);
                for (int i = 0; i < length; i++) {
                    if (x[i] != y[i])
                        return x[i].CompareTo(y[i]);
                }

                return x.Length.CompareTo(y.Length);
            }
        }
    }

    /// <summary>
    ///     Python's <c>difflib.SequenceMatcher</c> over code points, without a junk function (<c>isjunk=None</c>) and with
    ///     <c>autojunk=True</c>: the ratios <see cref="Difflib.GetCloseMatches"/> filters and ranks by. The second sequence is
    ///     indexed once (<see cref="SetSeq2"/>) and compared with many first sequences (<see cref="SetSeq1"/>).
    /// </summary>
    /// <remarks>
    ///     Not thread-safe: it caches the matching blocks of the current pair. Reference: <see href="https://github.com/python/cpython/blob/v3.12.12/Lib/difflib.py#L44">Lib/difflib.py (SequenceMatcher)</see>.
    /// </remarks>
    internal sealed class SequenceMatcher {
        /// <summary>The first sequence (<c>a</c>).</summary>
        private int[] _a = Array.Empty<int>();

        /// <summary>The second sequence (<c>b</c>).</summary>
        private int[] _b = Array.Empty<int>();

        /// <summary>For each element of <c>b</c>, the ascending indices where it occurs, popular elements excluded (<c>b2j</c>).</summary>
        private Dictionary<int, List<int>> _b2j = new();

        /// <summary>The count of each element of <c>b</c> (<c>fullbcount</c>), built on the first <see cref="QuickRatio"/>.</summary>
        private Dictionary<int, int>? _fullBCount;

        /// <summary>The matching blocks of the current pair (<c>matching_blocks</c>), computed once.</summary>
        private List<(int A, int B, int Size)>? _matchingBlocks;

        /// <summary>Sets the first sequence (<c>set_seq1</c>); the index of the second is kept.</summary>
        /// <param name="a">The sequence.</param>
        public void SetSeq1(int[] a) {
            if (ReferenceEquals(a, _a))
                return;
            _a = a;
            _matchingBlocks = null;
        }

        /// <summary>Sets the second sequence (<c>set_seq2</c>) and indexes it.</summary>
        /// <param name="b">The sequence.</param>
        public void SetSeq2(int[] b) {
            if (ReferenceEquals(b, _b))
                return;
            _b = b;
            _matchingBlocks = null;
            _fullBCount = null;
            ChainB();
        }

        /// <summary>
        ///     Indexes <c>b</c> (<c>__chain_b</c>). With no junk function nothing is junk; with <c>autojunk</c>, an element of a
        ///     sequence of 200 or more that occurs more than 1% of the time (plus one) is "popular" and left out of the index,
        ///     which changes which matches are found, so it is kept exactly.
        /// </summary>
        private void ChainB() {
            var b2j = new Dictionary<int, List<int>>();
            for (int i = 0; i < _b.Length; i++) {
                if (!b2j.TryGetValue(_b[i], out List<int>? indices))
                    b2j[_b[i]] = indices = new List<int>();
                indices.Add(i);
            }

            int n = _b.Length;
            if (n >= 200) {
                int ntest = n / 100 + 1;
                foreach (int popular in b2j.Where(pair => pair.Value.Count > ntest).Select(pair => pair.Key).ToList())
                    b2j.Remove(popular);
            }

            _b2j = b2j;
        }

        /// <summary>
        ///     The longest matching block in <c>a[alo:ahi]</c> and <c>b[blo:bhi]</c> (<c>find_longest_match</c>): of the longest,
        ///     the one starting earliest in <c>a</c>, then earliest in <c>b</c>, extended over the popular elements the index
        ///     left out.
        /// </summary>
        /// <param name="alo">Start in <c>a</c>.</param>
        /// <param name="ahi">End in <c>a</c> (exclusive).</param>
        /// <param name="blo">Start in <c>b</c>.</param>
        /// <param name="bhi">End in <c>b</c> (exclusive).</param>
        /// <returns>The block: its start in <c>a</c>, its start in <c>b</c>, and its size (0 when nothing matches).</returns>
        private (int A, int B, int Size) FindLongestMatch(int alo, int ahi, int blo, int bhi) {
            int besti = alo, bestj = blo, bestsize = 0;
            // During an iteration of the loop, j2len[j] is the length of the longest match ending with a[i-1] and b[j].
            var j2len = new Dictionary<int, int>();
            for (int i = alo; i < ahi; i++) {
                var newj2len = new Dictionary<int, int>();
                if (_b2j.TryGetValue(_a[i], out List<int>? indices)) {
                    foreach (int j in indices) {
                        // a[i] matches b[j]
                        if (j < blo)
                            continue;
                        if (j >= bhi)
                            break;
                        int k = newj2len[j] = (j2len.TryGetValue(j - 1, out int previous) ? previous : 0) + 1;
                        if (k > bestsize) {
                            besti = i - k + 1;
                            bestj = j - k + 1;
                            bestsize = k;
                        }
                    }
                }

                j2len = newj2len;
            }

            // Extend the best by non-junk elements on each end: popular elements are not in b2j, so the loop above never
            // matched them. Nothing is junk here, so Python's two junk-extension loops never run.
            while (besti > alo && bestj > blo && _a[besti - 1] == _b[bestj - 1]) {
                besti--;
                bestj--;
                bestsize++;
            }

            while (besti + bestsize < ahi && bestj + bestsize < bhi && _a[besti + bestsize] == _b[bestj + bestsize])
                bestsize++;

            return (besti, bestj, bestsize);
        }

        /// <summary>
        ///     The matching blocks (<c>get_matching_blocks</c>): the longest match, then recursively those left and right of it,
        ///     sorted, adjacent blocks merged, and the <c>(len(a), len(b), 0)</c> sentinel last.
        /// </summary>
        /// <returns>The blocks.</returns>
        private List<(int A, int B, int Size)> GetMatchingBlocks() {
            if (_matchingBlocks is not null)
                return _matchingBlocks;
            int la = _a.Length, lb = _b.Length;

            // Python keeps a queue (popped from its end) instead of recursing, so deep inputs cannot overflow the stack.
            var queue = new List<(int Alo, int Ahi, int Blo, int Bhi)> { (0, la, 0, lb) };
            var matchingBlocks = new List<(int A, int B, int Size)>();
            while (queue.Count > 0) {
                (int alo, int ahi, int blo, int bhi) = queue[^1];
                queue.RemoveAt(queue.Count - 1);
                (int i, int j, int k) = FindLongestMatch(alo, ahi, blo, bhi);
                if (k != 0) {
                    matchingBlocks.Add((i, j, k));
                    if (alo < i && blo < j)
                        queue.Add((alo, i, blo, j));
                    if (i + k < ahi && j + k < bhi)
                        queue.Add((i + k, ahi, j + k, bhi));
                }
            }

            matchingBlocks.Sort();

            // Collapse adjacent blocks.
            int i1 = 0, j1 = 0, k1 = 0;
            var nonAdjacent = new List<(int A, int B, int Size)>();
            foreach ((int i2, int j2, int k2) in matchingBlocks) {
                if (i1 + k1 == i2 && j1 + k1 == j2) {
                    k1 += k2;
                } else {
                    if (k1 != 0)
                        nonAdjacent.Add((i1, j1, k1));
                    (i1, j1, k1) = (i2, j2, k2);
                }
            }

            if (k1 != 0)
                nonAdjacent.Add((i1, j1, k1));
            nonAdjacent.Add((la, lb, 0));
            _matchingBlocks = nonAdjacent;
            return nonAdjacent;
        }

        /// <summary>The similarity, <c>2 * matches / (len(a) + len(b))</c>, 1.0 for two empty sequences (<c>ratio</c>).</summary>
        /// <returns>The ratio in <c>[0, 1]</c>.</returns>
        public double Ratio() {
            int matches = GetMatchingBlocks().Sum(block => block.Size);
            return CalculateRatio(matches, _a.Length + _b.Length);
        }

        /// <summary>An upper bound on <see cref="Ratio"/> that ignores order: the multiset intersection (<c>quick_ratio</c>).</summary>
        /// <returns>The bound.</returns>
        public double QuickRatio() {
            if (_fullBCount is null) {
                _fullBCount = new Dictionary<int, int>();
                foreach (int element in _b)
                    _fullBCount[element] = (_fullBCount.TryGetValue(element, out int count) ? count : 0) + 1;
            }

            // avail[x] is the number of times x appears in b less the number of times it was seen in a so far.
            var avail = new Dictionary<int, int>();
            int matches = 0;
            foreach (int element in _a) {
                int numb = avail.TryGetValue(element, out int left) ? left : _fullBCount.TryGetValue(element, out int total) ? total : 0;
                avail[element] = numb - 1;
                if (numb > 0)
                    matches++;
            }

            return CalculateRatio(matches, _a.Length + _b.Length);
        }

        /// <summary>The cheapest upper bound on <see cref="Ratio"/>: the shorter length as the matches (<c>real_quick_ratio</c>).</summary>
        /// <returns>The bound.</returns>
        public double RealQuickRatio() => CalculateRatio(Math.Min(_a.Length, _b.Length), _a.Length + _b.Length);

        /// <summary>Python's <c>_calculate_ratio</c>: <c>2.0 * matches / length</c>, or 1.0 when both sequences are empty.</summary>
        /// <param name="matches">The matched elements.</param>
        /// <param name="length">The total length.</param>
        /// <returns>The ratio.</returns>
        private static double CalculateRatio(int matches, int length) => length != 0 ? 2.0 * matches / length : 1.0;
    }
}
