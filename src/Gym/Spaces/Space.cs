#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Utils;
using NumSharp;

namespace Gym.Spaces {
    /// <summary>
    ///     The untyped view of every space: the set an env's actions or observations belong to. It records their
    ///     <see cref="Shape"/> and <see cref="DType"/> (when its elements are arrays), tests membership (<see cref="Contains"/>)
    ///     and draws random members (<see cref="Sample"/>) from a generator of its own, so seeding a space never disturbs the
    ///     env's stream. Code that knows the element type uses <see cref="Space{T}"/>, which every space derives from.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A space owns the arrays it holds (a <see cref="Box"/>'s bounds) and releases them in <see cref="Dispose()"/>; an
    ///     env disposes its spaces when it is disposed (D19). Samples are new arrays the caller owns.
    ///     </para>
    ///     <para>
    ///     Not thread-safe: the generator is not synchronized, so sample a space from one thread at a time, as Gymnasium's
    ///     single-threaded envs do. Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/space.py#L18">gymnasium/spaces/space.py</see>.
    ///     </para>
    /// </remarks>
    public abstract class Space : IDisposable {
        /// <summary>The generator, created on first use (Gymnasium's <c>_np_random</c>); <see langword="null"/> until then.</summary>
        private Generator? _npRandom;

        /// <summary>Whether <see cref="Dispose()"/> already ran, so a second call is a no-op.</summary>
        private bool _disposed;

        /// <summary>
        ///     Records the shape and dtype and seeds the generator when a seed is given; only <see cref="Space{T}"/> derives
        ///     from this class.
        /// </summary>
        /// <param name="shape">The shape of every element, or <see langword="null"/> for spaces whose elements are not arrays (Python's <c>None</c>).</param>
        /// <param name="dtype">The element dtype, or <see langword="null"/> for spaces whose elements are not arrays.</param>
        /// <param name="seed">
        ///     <see langword="null"/> to create the generator lazily from OS entropy on first use; an integer to seed one now;
        ///     a generator to share.
        /// </param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        private protected Space(Shape? shape, DType? dtype, SpaceSeed? seed) {
            Shape = shape;
            DType = dtype;
            if (seed is { } given) {
                if (given.Generator != null)
                    _npRandom = given.Generator;
                else
                    // Through the virtual Seed, as Gymnasium's __init__ calls self.seed(seed): a composite seeds its subspaces.
                    SeedCore(given.Value);
            }
        }

        /// <summary>
        ///     The shape of every element (Gymnasium's <c>shape</c>), or <see langword="null"/> for spaces whose elements are not
        ///     arrays (<c>Tuple</c>, <c>Dict</c>, <c>Text</c>, ...).
        /// </summary>
        public Shape? Shape { get; }

        /// <summary>
        ///     The element type of every element, as NumSharp's <see cref="NumSharp.DType"/> (Gymnasium's <c>dtype</c>), or
        ///     <see langword="null"/> for spaces whose elements are not arrays. Compare it with <c>==</c>, which also accepts a
        ///     <see cref="Type"/>, an <see cref="NPTypeCode"/> or a NumPy dtype string.
        /// </summary>
        /// <remarks>
        ///     Beware <c>np.@byte</c>: it is int8 (NumPy's C <c>char</c>), not uint8; uint8 is <c>np.uint8</c>.
        /// </remarks>
        public DType? DType { get; }

        /// <summary>
        ///     The generator this space samples from (Gymnasium's <c>np_random</c>), created from OS entropy on first use when
        ///     none was seeded or shared. Assigning one shares it, as Gymnasium's <c>space._np_random = rng</c> does.
        /// </summary>
        /// <remarks>
        ///     Lazily created because seeding costs a <c>SeedSequence</c>, which a space that is never sampled should not pay;
        ///     reading it is therefore not free the first time, and it draws OS entropy.
        /// </remarks>
        public virtual Generator NpRandom {
            get {
                if (_npRandom is null)
                    SeedCore(null);
                // A composite's seed seeds its subspaces and may leave its own generator unset (Gymnasium's comment in
                // Space.np_random), so one is drawn from entropy here rather than returning null.
                return _npRandom ??= Seeding.NpRandom().Rng;
            }
            set => _npRandom = value ?? throw new ArgumentNullException(nameof(value));
        }

        /// <summary>
        ///     The generator if one exists, without creating it (Gymnasium's <c>_np_random</c>); for code that must not draw OS
        ///     entropy just to look, such as a parity check of an unseeded space.
        /// </summary>
        public virtual Generator? NpRandomOrNull => _npRandom;

        /// <summary>Whether this space flattens to a <see cref="Box"/> (Gymnasium's <c>is_np_flattenable</c>).</summary>
        public abstract bool IsNpFlattenable { get; }

        /// <summary>
        ///     Draws a random element, untyped: what <see cref="Space{T}.Sample"/> returns, boxed. A mask restricts which
        ///     elements are drawn and a probability mask weights them; each space documents the forms it accepts.
        /// </summary>
        /// <param name="mask">A space-specific mask (an int8 array for <see cref="Discrete"/>), or <see langword="null"/>.</param>
        /// <param name="probability">A space-specific probability mask (a float64 array for <see cref="Discrete"/>), or <see langword="null"/>.</param>
        /// <returns>The element; arrays are new and owned by the caller.</returns>
        public object? Sample(object? mask = null, object? probability = null) => SampleObject(mask, probability);

        /// <summary>The typed sample, boxed; implemented once by <see cref="Space{T}"/>.</summary>
        /// <param name="mask">The mask.</param>
        /// <param name="probability">The probability mask.</param>
        /// <returns>The element.</returns>
        private protected abstract object? SampleObject(object? mask, object? probability);

        /// <summary>
        ///     Reseeds the generator (Gymnasium's <c>seed</c>): with a value, a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>
        ///     that draws what Gymnasium's space draws after <c>space.seed(seed)</c>; without one, a fresh generator from OS
        ///     entropy. A generator the space shared is replaced, never reseeded, so the env's stream is left untouched.
        /// </summary>
        /// <param name="seed">Any non-negative integer (Gymnasium's seeds are unbounded Python ints), or <see langword="null"/>.</param>
        /// <returns>
        ///     The seed used: <paramref name="seed"/> itself, or the 128-bit entropy of an unseeded call, which reproduces it.
        ///     Composite spaces return one seed per subspace (a list, or a dict for <c>Dict</c>), as Gymnasium's do.
        /// </returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative, with Gymnasium's message.</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/space.py#L112">gymnasium/spaces/space.py</see>.</remarks>
        public object Seed(BigInteger? seed = null) => SeedCore(seed);

        /// <summary>
        ///     Seeds a composite space's subspaces one by one (Gymnasium's <c>seed([s0, s1, ...])</c>): one seed per subspace,
        ///     each an integer, <see langword="null"/>, or itself a list or dictionary for a composite subspace.
        /// </summary>
        /// <param name="seeds">The seeds; <c>OneOf</c>, <c>Sequence</c> and <c>Graph</c> take their own seed first.</param>
        /// <returns>The seeds used, one per subspace, as Gymnasium's tuple.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a space that is not a composite, with Gymnasium's message.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown by a composite for a list of the wrong length.</exception>
        public object Seed(IReadOnlyList<object?> seeds) => SeedListCore(seeds ?? throw new ArgumentNullException(nameof(seeds)));

        /// <summary>
        ///     Seeds a <c>Dict</c> space's subspaces by key (Gymnasium's <c>seed({"a": 1, ...})</c>); the keys must be the
        ///     space's.
        /// </summary>
        /// <param name="seeds">One seed per key.</param>
        /// <returns>The seeds used, by key.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a space that is not a <c>Dict</c>, with Gymnasium's message.</exception>
        /// <exception cref="NumSharp.ValueError">Thrown by a <c>Dict</c> for keys that are not its own.</exception>
        public object Seed(IReadOnlyDictionary<string, object?> seeds) => SeedDictCore(seeds ?? throw new ArgumentNullException(nameof(seeds)));

        /// <summary>
        ///     Seeds with a value of any form, as Python's dynamically typed <c>space.seed(seed)</c> receives it: an integer (any
        ///     width, or a <see cref="BigInteger"/>; a <see langword="bool"/> is Python's <c>int</c> too), <see langword="null"/>,
        ///     a list or tuple, or a string-keyed dictionary. Composite spaces use it for nested seeds.
        /// </summary>
        /// <param name="seed">The seed.</param>
        /// <returns>The seed(s) used.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed, or by a simple space for a form it does not take, with Gymnasium's message.</exception>
        /// <exception cref="NumSharp.TypeError">Thrown by a composite for a form it does not take (<see cref="SeedTypeError"/>).</exception>
        /// <remarks>
        ///     A 0-d array (a NumPy scalar such as <c>np.int64(5)</c>) is refused like any other non-<c>int</c>, because
        ///     Gymnasium's seeding checks <c>isinstance(seed, int)</c>, which a NumPy scalar fails; convert it first.
        /// </remarks>
        public object SeedWith(object? seed) {
            switch (seed) {
                case null:
                    return SeedCore(null);
                case BigInteger big:
                    return SeedCore(big);
                case IReadOnlyDictionary<string, object?> dictionary:
                    return SeedDictCore(dictionary);
                case System.Collections.IDictionary dictionary:
                    return SeedDictCore(dictionary.Keys.Cast<object>().ToDictionary(k => Convert.ToString(k, CultureInfo.InvariantCulture)!, k => dictionary[k]));
                case string:
                case NumSharp.NDArray:
                    // Iterable in .NET, but a str or an ndarray is neither a Python list nor a tuple to Gymnasium's seed.
                    throw SeedTypeError(seed);
                case System.Runtime.CompilerServices.ITuple tuple:
                    return SeedListCore(Tuple.TupleItems(tuple));
                case System.Collections.IEnumerable items:
                    return SeedListCore(items.Cast<object?>().ToList());
                default:
                    if (seed is bool flag) {
                        // A Python bool is an int, so it seeds like 0 or 1; Gymnasium's seeding.np_random reports
                        // SeedSequence(seed).entropy, the bool itself, so a simple space returns it rather than the integer.
                        object used = SeedCore(flag ? BigInteger.One : BigInteger.Zero);
                        return used is BigInteger ? flag : used;
                    }

                    // Python ints take the integer path; anything else is this space's own TypeError/Error.
                    if (seed is ulong or long or int or short or sbyte or byte or ushort or uint or Enum)
                        return SeedCore(PythonIntSeed(seed));
                    throw SeedTypeError(seed);
            }
        }

        /// <summary>
        ///     The list seeding behind <see cref="Seed(IReadOnlyList{object})"/>. The base refuses, as Gymnasium's
        ///     <c>seeding.np_random</c> refuses a list given to a simple space.
        /// </summary>
        /// <param name="seeds">The seeds.</param>
        /// <returns>The seeds used.</returns>
        /// <exception cref="Gym.Exceptions.Error">Always, for a simple space: <c>Seed must be a python integer, actual type: &lt;class 'list'&gt;</c>.</exception>
        /// <exception cref="NumSharp.TypeError">Always, for a composite that takes no list (a <c>Dict</c>), through <see cref="SeedTypeError"/>.</exception>
        protected virtual object SeedListCore(IReadOnlyList<object?> seeds) => throw SeedTypeError(seeds);

        /// <summary>
        ///     The dictionary seeding behind <see cref="Seed(IReadOnlyDictionary{string, object})"/>; the base refuses through
        ///     <see cref="SeedTypeError"/>, so only <c>Dict</c> overrides it.
        /// </summary>
        /// <param name="seeds">The seeds.</param>
        /// <returns>The seeds used.</returns>
        /// <exception cref="Gym.Exceptions.Error">Always, for a simple space.</exception>
        /// <exception cref="NumSharp.TypeError">Always, for a composite that is not a <c>Dict</c>.</exception>
        protected virtual object SeedDictCore(IReadOnlyDictionary<string, object?> seeds) => throw SeedTypeError(seeds);

        /// <summary>
        ///     The error for a seed of a form this space does not take, which the caller throws. The base gives what
        ///     Gymnasium's <c>seeding.np_random</c> raises for a simple space; each composite overrides it with the
        ///     <see cref="NumSharp.TypeError"/> its own <c>seed</c> raises (the texts differ per space).
        /// </summary>
        /// <param name="seed">The refused seed, for its Python type in the message.</param>
        /// <returns>The exception to throw; never <see langword="null"/>.</returns>
        protected virtual Exception SeedTypeError(object? seed) =>
            new Gym.Exceptions.Error($"Seed must be a python integer, actual type: {PyBuiltins.TypeRepr(seed)}");

        /// <summary>
        ///     Seeds only this space's own generator from a seed of any form, as a composite's <c>super().seed(seed[0])</c>
        ///     does in Gymnasium: the subspaces are left alone, and the seed must be <see langword="null"/> or a Python
        ///     <c>int</c> (<see cref="PythonIntSeed"/>).
        /// </summary>
        /// <param name="seed">The seed: <see langword="null"/> for entropy, or an integer.</param>
        /// <returns>The seed used: the integer, or the entropy of an unseeded call.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed or one that is not a Python <c>int</c>, with Gymnasium's messages.</exception>
        /// <remarks>
        ///     Not virtual on purpose: it must reach this class's generator even when called from a composite whose
        ///     <see cref="SeedCore"/> override seeds the subspaces.
        /// </remarks>
        private protected object SeedOwnGenerator(object? seed) {
            (Generator rng, BigInteger used) = Seeding.NpRandom(PythonIntSeed(seed));
            _npRandom = rng;
            return used;
        }

        /// <summary>
        ///     A seed as Gymnasium's <c>seeding.np_random</c> accepts it: <see langword="null"/>, or a Python <c>int</c> (any
        ///     .NET integer, an enum, a <see langword="bool"/> or a <see cref="BigInteger"/>). A 0-d array is a NumPy scalar,
        ///     which <c>isinstance(seed, int)</c> refuses, so it is refused here too.
        /// </summary>
        /// <param name="seed">The candidate.</param>
        /// <returns>The integer, or <see langword="null"/>.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for any other type: <c>Seed must be a python integer, actual type: …</c>.</exception>
        internal static BigInteger? PythonIntSeed(object? seed) {
            switch (seed) {
                case null:
                    return null;
                case BigInteger big:
                    return big;
                case bool flag:
                    return flag ? BigInteger.One : BigInteger.Zero;
                case ulong large:
                    return large;
                case Enum value:
                    // An enum over ulong may exceed long; Python's IntEnum is unbounded, so neither may overflow here.
                    return Type.GetTypeCode(Enum.GetUnderlyingType(value.GetType())) == TypeCode.UInt64
                        ? new BigInteger(Convert.ToUInt64(value, CultureInfo.InvariantCulture))
                        : new BigInteger(Convert.ToInt64(value, CultureInfo.InvariantCulture));
                case long or int or short or sbyte or byte or ushort or uint:
                    return Convert.ToInt64(seed, CultureInfo.InvariantCulture);
                default:
                    throw new Gym.Exceptions.Error($"Seed must be a python integer, actual type: {PyBuiltins.TypeRepr(seed)}");
            }
        }

        /// <summary>
        ///     A deep copy of this space (Python's <c>copy.deepcopy(space)</c>, which Gymnasium's batching relies on): the
        ///     same parameters, copies of every array, subspace and generator, so drawing from the copy never moves the original.
        /// </summary>
        /// <returns>The copy; the caller owns it.</returns>
        /// <remarks>
        ///     The default copies member by member (<c>MemberwiseClone</c>) and then replaces every array (a new buffer),
        ///     generator (the same state, independent), subspace (deep-copied) and collection of those in the copy's fields, so
        ///     a custom space gets a correct deep copy without code of its own. A custom field type it does not know is shared,
        ///     like any other object <c>PyCopy</c> cannot copy; override <see cref="DeepCopyCore"/> for such a space.
        /// </remarks>
        public Space DeepCopy() => DeepCopyCore();

        /// <summary>The copying behind <see cref="DeepCopy"/>; see its remarks.</summary>
        /// <returns>The copy.</returns>
        protected virtual Space DeepCopyCore() {
            var copy = (Space)MemberwiseClone();
            for (Type? type = GetType(); type != null && type != typeof(object); type = type.BaseType) {
                foreach (System.Reflection.FieldInfo field in type.GetFields(System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public |
                                                                            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.DeclaredOnly)) {
                    object? value = field.GetValue(this);
                    object? copied = CopyMember(value);
                    if (!ReferenceEquals(value, copied))
                        field.SetValue(copy, copied);
                }
            }

            copy._disposed = false;
            return copy;
        }

        /// <summary>A copy of one member for <see cref="DeepCopyCore"/>: arrays, generators, subspaces and collections of them.</summary>
        /// <param name="value">The member's value.</param>
        /// <returns>The copy, or the value itself when it is immutable or of a type the copy shares.</returns>
        private static object? CopyMember(object? value) {
            switch (value) {
                case null:
                case string:
                    return value;
                case NumSharp.NDArray array:
                    return array.copy();
                case Generator generator:
                    return CopyGenerator(generator);
                case Space space:
                    return space.DeepCopy();
                case Space[] spaces:
                    return spaces.Select(s => s.DeepCopy()).ToArray();
                case Gym.Envs.Info info:
                    return Gym.Utils.PyCopy.DeepCopy(info);
                case IReadOnlyDictionary<string, Space> subspaces: {
                    // Dict's subspaces: a fresh insertion-ordered dictionary of copies, the key order kept.
                    var copied = new Dictionary<string, Space>();
                    foreach (KeyValuePair<string, Space> pair in subspaces)
                        copied.Add(pair.Key, pair.Value.DeepCopy());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /// <summary>
        ///     A generator with the same state as <paramref name="generator"/> and no link to it (Python's
        ///     <c>deepcopy(np_random)</c>): both draw the same next values, then go their own ways.
        /// </summary>
        /// <param name="generator">The generator to copy.</param>
        /// <returns>The copy.</returns>
        /// <exception cref="NumSharp.ValueError">Thrown for a generator whose bit generator is not a PCG64, which Gym.NET never creates.</exception>
        public static Generator CopyGenerator(Generator generator) {
            if (generator is null)
                throw new ArgumentNullException(nameof(generator));
            var copy = new Generator(new PCG64(0));
            // The state is copied in; NumPy's deepcopy pickles the bit generator, whose state is the same values.
            copy.bit_generator.state = generator.bit_generator.state;
            return copy;
        }

        /// <summary>
        ///     The seeding behind <see cref="Seed(BigInteger?)"/>; composite spaces override it to seed their subspaces. Called
        ///     from the constructor for an integer seed, so an override must not rely on state its own constructor sets after
        ///     the base call (composites pass <see langword="null"/> to the base and seed in their own constructor for this
        ///     reason).
        /// </summary>
        /// <param name="seed">The seed, or <see langword="null"/> for entropy.</param>
        /// <returns>The seed(s) used.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown when <paramref name="seed"/> is negative.</exception>
        protected virtual object SeedCore(BigInteger? seed) {
            (Generator rng, BigInteger used) = Seeding.NpRandom(seed);
            _npRandom = rng;
            return used;
        }

        /// <summary>Replaces the generator with one the caller made, for composite spaces seeding themselves; never shares by accident.</summary>
        /// <param name="generator">The generator; <see langword="null"/> forgets it, so the next use draws entropy.</param>
        protected void SetNpRandom(Generator? generator) => _npRandom = generator;

        /// <summary>
        ///     Whether <paramref name="x"/> is an element of this space (Gymnasium's <c>contains</c> and <c>x in space</c>).
        ///     Values of the wrong kind return <see langword="false"/> rather than throwing, as in Gymnasium.
        /// </summary>
        /// <param name="x">The candidate.</param>
        /// <returns><see langword="true"/> for an element.</returns>
        public abstract bool Contains(object? x);

        /// <summary>
        ///     Converts a batch of samples to JSON-friendly data (Gymnasium's <c>to_jsonable</c>), untyped: usually one value per
        ///     sample (a list), but a composite may arrange the batch otherwise, as <c>Dict</c>'s dict of per-key lists does,
        ///     which is why the result is an <see cref="object"/>.
        /// </summary>
        /// <param name="samples">Samples of this space.</param>
        /// <returns>The JSON-friendly data: lists, numbers, strings and string-keyed dictionaries.</returns>
        /// <exception cref="InvalidCastException">Thrown when a sample is not of this space's element type.</exception>
        public object ToJsonable(IEnumerable<object?> samples) => ToJsonableObject(samples);

        /// <summary>Converts JSON-friendly data back to a batch of samples (Gymnasium's <c>from_jsonable</c>), untyped.</summary>
        /// <param name="jsonable">Data <see cref="ToJsonable"/> produced.</param>
        /// <returns>The samples, boxed.</returns>
        /// <exception cref="ArgumentException">Thrown when the data is not in the form this space's <c>to_jsonable</c> produces.</exception>
        public IList<object?> FromJsonable(object jsonable) => FromJsonableObject(jsonable);

        /// <summary>The typed conversion, boxed; implemented once by <see cref="Space{T}"/>.</summary>
        /// <param name="samples">The samples.</param>
        /// <returns>The JSON-friendly data.</returns>
        private protected abstract object ToJsonableObject(IEnumerable<object?> samples);

        /// <summary>The typed conversion, boxed; implemented once by <see cref="Space{T}"/>.</summary>
        /// <param name="jsonable">The JSON-friendly data.</param>
        /// <returns>The samples, boxed.</returns>
        private protected abstract IList<object?> FromJsonableObject(object jsonable);

        /// <summary>Releases the arrays this space owns; idempotent. The space must not be used afterwards.</summary>
        public void Dispose() {
            if (_disposed)
                return;
            _disposed = true;
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        /// <summary>Releases what a derived space owns (a <see cref="Box"/>'s bounds); called once, by <see cref="Dispose()"/>.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Dispose()"/>; no space has a finalizer.</param>
        protected virtual void Dispose(bool disposing) { }

        /// <summary>
        ///     NumPy's <c>ndarray.tolist()</c> with Python's scalar types: nested lists whose floats are <see langword="double"/>s
        ///     (a float32 or float16 element widened exactly, as NumPy's <c>item()</c> gives a Python <c>float</c>), integers and
        ///     booleans as stored. NumSharp's own <c>tolist</c> keeps each element's CLR type, which a float32 box's JSON would
        ///     otherwise expose as <see langword="float"/>.
        /// </summary>
        /// <param name="array">The array; only read.</param>
        /// <returns>A new nested <see cref="List{T}"/> (or the scalar of a 0-d array).</returns>
        internal static object? PythonList(NDArray array) => PythonScalars(array.tolist());

        /// <summary>Replaces the float32 and float16 leaves of a <c>tolist()</c> result with <see langword="double"/>s, in place.</summary>
        /// <param name="value">A list or a leaf.</param>
        /// <returns>The value, with its leaves widened.</returns>
        private static object? PythonScalars(object? value) {
            switch (value) {
                case float f:
                    return (double)f;
                case Half h:
                    return (double)h;
                case List<object> list:
                    for (int i = 0; i < list.Count; i++)
                        list[i] = PythonScalars(list[i])!;
                    return list;
                default:
                    return value;
            }
        }

        /// <summary>A shape as Python prints the tuple: <c>()</c>, <c>(2,)</c>, <c>(3, 4)</c>; <c>None</c> for no shape.</summary>
        /// <param name="shape">The shape.</param>
        /// <returns>The text.</returns>
        internal static string ShapeRepr(Shape? shape) {
            if (shape is not { } s)
                return "None";
            long[] dimensions = s.dimensions ?? Array.Empty<long>();
            if (dimensions.Length == 1)
                return $"({dimensions[0].ToString(CultureInfo.InvariantCulture)},)";
            return "(" + string.Join(", ", dimensions.Select(d => d.ToString(CultureInfo.InvariantCulture))) + ")";
        }
    }

    /// <summary>
    ///     A space whose elements are <typeparamref name="T"/>: <see cref="Box"/> and <see cref="Discrete"/> are
    ///     <c>Space&lt;NDArray&gt;</c> (a discrete element is a 0-d int64 array, NumPy's <c>np.int64</c>, DR-08). An env's
    ///     <c>ActionSpace</c> and <c>ObservationSpace</c> are typed this way (DR-07), and custom spaces derive from it.
    /// </summary>
    /// <typeparam name="T">The element type, what <see cref="Sample"/> returns.</typeparam>
    /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/spaces/space.py#L18">gymnasium/spaces/space.py</see>.</remarks>
    public abstract class Space<T> : Space {
        /// <summary>Records the shape and dtype and seeds the generator when a seed is given.</summary>
        /// <param name="shape">The shape of every element, or <see langword="null"/> when elements are not arrays.</param>
        /// <param name="dtype">The element dtype, or <see langword="null"/> when elements are not arrays.</param>
        /// <param name="seed">An integer seed, a generator to share, or <see langword="null"/> for a lazy entropy-seeded one.</param>
        /// <exception cref="Gym.Exceptions.Error">Thrown when an integer <paramref name="seed"/> is negative.</exception>
        protected Space(Shape? shape, DType? dtype, SpaceSeed? seed) : base(shape, dtype, seed) { }

        /// <summary>
        ///     Draws a random element (Gymnasium's <c>sample</c>). A mask restricts which elements are drawn and a probability
        ///     mask weights them; the two cannot be combined, and each space documents the forms it accepts.
        /// </summary>
        /// <param name="mask">A space-specific mask, or <see langword="null"/>.</param>
        /// <param name="probability">A space-specific probability mask, or <see langword="null"/>.</param>
        /// <returns>The element; arrays are new and owned by the caller.</returns>
        public new abstract T Sample(object? mask = null, object? probability = null);

        /// <summary>
        ///     Converts a batch of samples to JSON-friendly data (Gymnasium's <c>to_jsonable</c>); the default keeps them as they
        ///     are, in a list, as Gymnasium's base does.
        /// </summary>
        /// <param name="samples">The samples.</param>
        /// <returns>The JSON-friendly data; a list of one value per sample unless the space documents another form.</returns>
        public virtual object ToJsonable(IEnumerable<T> samples) => samples.Select(s => (object?)s).ToList();

        /// <summary>
        ///     Converts JSON-friendly data back to samples (Gymnasium's <c>from_jsonable</c>); the default casts each item. It
        ///     hides the untyped <see cref="Space.FromJsonable(object)"/>, which returns the same samples boxed.
        /// </summary>
        /// <param name="jsonable">The data, a list for the default.</param>
        /// <returns>The samples.</returns>
        /// <exception cref="ArgumentException">Thrown by the default when the data is not a list.</exception>
        /// <exception cref="InvalidCastException">Thrown by the default when an item is not a <typeparamref name="T"/>.</exception>
        public new virtual IList<T> FromJsonable(object jsonable) => JsonList(jsonable).Select(v => (T)v!).ToList();

        /// <inheritdoc />
        private protected sealed override object? SampleObject(object? mask, object? probability) => Sample(mask, probability);

        /// <inheritdoc />
        private protected sealed override object ToJsonableObject(IEnumerable<object?> samples) => ToJsonable(samples.Select(s => (T)s!));

        /// <inheritdoc />
        private protected sealed override IList<object?> FromJsonableObject(object jsonable) =>
            FromJsonable(jsonable).Select(s => (object?)s).ToList();

        /// <summary>JSON-friendly data as the list a space's <c>from_jsonable</c> iterates over.</summary>
        /// <param name="jsonable">The data.</param>
        /// <returns>Its items.</returns>
        /// <exception cref="ArgumentException">Thrown when the data is not a list.</exception>
        protected static IList<object?> JsonList(object jsonable) => jsonable switch {
            IList<object?> list => list,
            System.Collections.IEnumerable items when jsonable is not string => items.Cast<object?>().ToList(),
            _ => throw new ArgumentException($"Expected a list of JSON-friendly samples, got {jsonable?.GetType().Name ?? "null"}.", nameof(jsonable)),
        };
    }
}
