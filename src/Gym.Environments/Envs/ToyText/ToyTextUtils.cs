#nullable enable
using System;
using System.Collections.Generic;
using NumSharp;

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     Gymnasium's <c>gymnasium.envs.toy_text.utils</c> (<see cref="CategoricalSample"/>) and the character maps the
    ///     toy_text envs build their worlds from (<see cref="CharMap"/>).
    /// </summary>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/utils.py">gymnasium/envs/toy_text/utils.py</see>.</remarks>
    public static class ToyTextUtils {
        /// <summary>
        ///     Gymnasium's <c>categorical_sample(prob_n, np_random)</c>: one draw from <paramref name="npRandom"/>
        ///     (<c>np_random.random()</c>), then the first index whose running total of <paramref name="probN"/> exceeds it;
        ///     0 when none does (NumPy's <c>argmax</c> of an all-false array).
        /// </summary>
        /// <param name="probN">The probabilities, in order; they need not sum to 1 (a total below the draw gives 0).</param>
        /// <param name="npRandom">The generator; exactly one float64 is drawn from it, as in Gymnasium.</param>
        /// <returns>The sampled index.</returns>
        /// <exception cref="ArgumentNullException">Thrown when an argument is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for no probabilities, as NumPy's <c>argmax</c> of an empty sequence raises (after the draw).</exception>
        /// <remarks>
        ///     The running total is NumPy's <c>cumsum</c>: float64 additions in order, each rounded, so a total that should
        ///     be 1 can fall just short of it, which this reproduces. Reference:
        ///     <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/utils.py#L4">gymnasium/envs/toy_text/utils.py</see>.
        /// </remarks>
        public static long CategoricalSample(IReadOnlyList<double> probN, Generator npRandom) {
            if (probN is null)
                throw new ArgumentNullException(nameof(probN));
            if (npRandom is null)
                throw new ArgumentNullException(nameof(npRandom));

            // np.argmax(csprob_n > np_random.random()): the draw happens before argmax sees an empty comparison and raises,
            // so even a failing call moves the generator, as in Python.
            double draw;
            using (NDArray random = npRandom.random())
                draw = random.GetAtIndex<double>(0);
            if (probN.Count == 0)
                throw new ValueError("attempt to get argmax of an empty sequence");

            double total = 0.0;
            for (int i = 0; i < probN.Count; i++) {
                total += probN[i];
                if (total > draw)
                    return i;
            }

            return 0;
        }

        /// <summary>
        ///     NumPy's <c>np.asarray(rows, dtype="c")</c>: a <c>(len(rows), len(row))</c> array of one-byte strings (<c>S1</c>),
        ///     one character per element, which is how FrozenLake and Taxi hold their maps.
        /// </summary>
        /// <param name="rows">The map's rows, all of one length, in ASCII.</param>
        /// <returns>A new <c>S1</c> array; the caller owns it.</returns>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="rows"/> or a row is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">Thrown for rows of different lengths, with NumPy's message.</exception>
        /// <remarks>
        ///     It builds the two-dimensional array of one-character strings NumPy builds for <c>np.asarray(rows, "c")</c>,
        ///     through NumSharp's own <c>'c'</c> dtype: same shape, dtype (<c>|S1</c>) and bytes. It was written while
        ///     NumSharp 0.80.0's <c>np.array(string[], "c")</c> kept one element per string (open item N-2); NumSharp 0.81.0
        ///     splits the strings itself (N-2 closed, DR-76, guarded by <c>NumSharpGapTests</c>), and this stays the toy_text
        ///     envs' one place that builds and validates their maps.
        /// </remarks>
        public static NDArray CharMap(IReadOnlyList<string> rows) {
            if (rows is null)
                throw new ArgumentNullException(nameof(rows));
            int columns = rows.Count == 0 ? 0 : (rows[0] ?? throw new ArgumentNullException(nameof(rows), "A map row is null.")).Length;
            var grid = new string[rows.Count, columns];
            for (int row = 0; row < rows.Count; row++) {
                string line = rows[row] ?? throw new ArgumentNullException(nameof(rows), "A map row is null.");
                // NumPy refuses a ragged map while building the array, naming the first dimension that went ragged.
                if (line.Length != columns)
                    throw new ValueError("setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. "
                                         + $"The detected shape was ({rows.Count},) + inhomogeneous part.");
                for (int column = 0; column < columns; column++)
                    grid[row, column] = line[column].ToString();
            }

            return np.array(grid, np.dtype("c"));
        }
    }
}
