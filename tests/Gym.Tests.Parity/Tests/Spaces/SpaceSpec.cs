using System;
using System.Collections.Generic;
using System.Linq;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>
    ///     A space described once and built on both sides: Gymnasium's (through the <see cref="Python"/> tree) and Gym.NET's,
    ///     with the same parameters, so one list of specs drives every generic space cell (samples, generator states, seeds,
    ///     <c>repr</c>, JSON, batching).
    /// </summary>
    /// <remarks>
    ///     Subspaces are built without seeds, as a Gymnasium user writes <c>Tuple((Discrete(3), Box(0, 1)), seed=7)</c>: only
    ///     the outermost space takes the seed, and its own seeding reaches the rest.
    /// </remarks>
    internal abstract class SpaceSpec {
        /// <summary>Records the Python spelling.</summary>
        /// <param name="python">The constructor as a Gymnasium user writes it, without the seed (reports only).</param>
        protected SpaceSpec(string python) => PythonText = python;

        /// <summary>The constructor as a Gymnasium user writes it, without the seed; shown in reports and as the case name.</summary>
        public string PythonText { get; }

        /// <summary>Builds Gymnasium's space. Must run under the GIL.</summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>A new owned reference.</returns>
        public abstract PyObject BuildPython(long? seed);

        /// <summary>Builds Gym.NET's space.</summary>
        /// <param name="seed">The seed, or <see langword="null"/>.</param>
        /// <returns>The space; the caller disposes it.</returns>
        public abstract Space BuildDotNet(long? seed);

        /// <summary>
        ///     The direct subspaces, each with a path for reports and how to reach it on either side; composites override it.
        /// </summary>
        public virtual IReadOnlyList<SubspaceAccess> Children => Array.Empty<SubspaceAccess>();

        /// <summary>Every subspace below this one, depth first, with paths relative to it (<c>[0]</c>, <c>['a'].feature_space</c>).</summary>
        /// <returns>The accessors, composed from the root.</returns>
        public IEnumerable<SubspaceAccess> Descendants() {
            foreach (SubspaceAccess child in Children) {
                yield return child;
                foreach (SubspaceAccess grandchild in child.Spec.Descendants()) {
                    yield return new SubspaceAccess(
                        child.Path + grandchild.Path,
                        grandchild.Spec,
                        root => {
                            using PyObject middle = child.Python(root);
                            return grandchild.Python(middle);
                        },
                        root => grandchild.DotNet(child.DotNet(root)));
                }
            }
        }

        /// <inheritdoc />
        public override string ToString() => PythonText;

        /// <summary>Builds the Python subspaces of a composite as a Python tuple. Must run under the GIL.</summary>
        /// <param name="items">The specs.</param>
        /// <returns>A new owned tuple.</returns>
        protected static PyObject PythonTuple(IReadOnlyList<SpaceSpec> items) {
            var built = new PyObject[items.Count];
            try {
                for (int i = 0; i < built.Length; i++)
                    built[i] = items[i].BuildPython(null);
                return new PyTuple(built);
            } finally {
                // The tuple holds its own references.
                foreach (PyObject? item in built)
                    item?.Dispose();
            }
        }
    }

    /// <summary>How to reach one subspace from the root space on either side.</summary>
    /// <param name="Path">The Python spelling of the access, appended to the root's name in reports (<c>[1]</c>, <c>.feature_space</c>).</param>
    /// <param name="Spec">The subspace's spec.</param>
    /// <param name="Python">From the root's Python object to a new reference to the subspace; runs under the GIL.</param>
    /// <param name="DotNet">From the root's Gym.NET space to the subspace, which the root still owns.</param>
    internal sealed record SubspaceAccess(string Path, SpaceSpec Spec, Func<PyObject, PyObject> Python, Func<Space, Space> DotNet);

    /// <summary><c>Discrete(n, start=start)</c>.</summary>
    internal sealed class DiscreteSpec : SpaceSpec {
        /// <summary>The number of elements.</summary>
        private readonly long _n;

        /// <summary>The smallest element.</summary>
        private readonly long _start;

        /// <summary>Describes the space.</summary>
        /// <param name="n">The number of elements.</param>
        /// <param name="start">The smallest element.</param>
        public DiscreteSpec(long n, long start = 0) : base(start == 0 ? $"Discrete({n})" : $"Discrete({n}, start={start})") {
            _n = n;
            _start = start;
        }

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) => Python.gymnasium.spaces.Discrete(_n, seed, _start);

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new Discrete(_n, seed: seed, start: _start);
    }

    /// <summary><c>Box(low, high, shape, dtype)</c> with scalar bounds given as Python floats.</summary>
    internal sealed class BoxSpec : SpaceSpec {
        /// <summary>The lower bound.</summary>
        private readonly double _low;

        /// <summary>The upper bound.</summary>
        private readonly double _high;

        /// <summary>The shape.</summary>
        private readonly long[] _shape;

        /// <summary>The dtype's NumPy name.</summary>
        private readonly string _dtype;

        /// <summary>Describes the space.</summary>
        /// <param name="low">The lower bound of every entry.</param>
        /// <param name="high">The upper bound of every entry.</param>
        /// <param name="shape">The shape.</param>
        /// <param name="dtype">The dtype's NumPy name (<c>float32</c>).</param>
        public BoxSpec(double low, double high, long[] shape, string dtype = "float32")
            : base($"Box({Gym.Utils.PyBuiltins.FloatRepr(low)}, {Gym.Utils.PyBuiltins.FloatRepr(high)}, ({string.Join(", ", shape)}{(shape.Length == 1 ? "," : "")}), np.{dtype})") {
            _low = low;
            _high = high;
            _shape = shape;
            _dtype = dtype;
        }

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject dtype = Python.numpy.dtype(_dtype);
            return Python.gymnasium.spaces.Box(_low, _high, _shape, dtype, seed);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new Box(_low, _high, new Shape(_shape), np.dtype(_dtype), seed: seed);
    }

    /// <summary><c>MultiBinary(n)</c>, with <c>n</c> an int or a shape.</summary>
    internal sealed class MultiBinarySpec : SpaceSpec {
        /// <summary>The shape.</summary>
        private readonly long[] _n;

        /// <summary>Whether <c>n</c> is given as an int rather than a list.</summary>
        private readonly bool _scalar;

        /// <summary>Describes <c>MultiBinary(n)</c> for an int.</summary>
        /// <param name="n">The number of bits.</param>
        public MultiBinarySpec(long n) : base($"MultiBinary({n})") {
            _n = new[] { n };
            _scalar = true;
        }

        /// <summary>Describes <c>MultiBinary([d0, d1, ...])</c>.</summary>
        /// <param name="n">The shape.</param>
        public MultiBinarySpec(long[] n) : base($"MultiBinary([{string.Join(", ", n)}])") => _n = n;

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject n = _scalar ? PyConvert.ToPython(_n[0]) : PyConvert.ToPython(_n.Cast<object?>().ToList());
            return Python.gymnasium.spaces.MultiBinary(n, seed);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => _scalar ? new MultiBinary(_n[0], seed) : new MultiBinary(_n, seed);
    }

    /// <summary><c>MultiDiscrete(nvec, dtype=dtype, start=start)</c> with a 1-D or 2-D <c>nvec</c>.</summary>
    internal sealed class MultiDiscreteSpec : SpaceSpec {
        /// <summary>The counts, flat.</summary>
        private readonly long[] _nvec;

        /// <summary>The counts' shape.</summary>
        private readonly long[] _shape;

        /// <summary>The starts, flat, or <see langword="null"/>.</summary>
        private readonly long[]? _start;

        /// <summary>The dtype's NumPy name, or <see langword="null"/> for the default.</summary>
        private readonly string? _dtype;

        /// <summary>Describes the space.</summary>
        /// <param name="nvec">The counts, flat (C order).</param>
        /// <param name="shape">Their shape (<c>[n]</c> for a vector).</param>
        /// <param name="start">The starts, flat, or <see langword="null"/>.</param>
        /// <param name="dtype">The dtype's NumPy name, or <see langword="null"/> for Gymnasium's <c>np.int64</c>.</param>
        public MultiDiscreteSpec(long[] nvec, long[]? shape = null, long[]? start = null, string? dtype = null)
            : base($"MultiDiscrete(np.array([{string.Join(", ", nvec)}]).reshape({string.Join(", ", shape ?? new[] { (long)nvec.Length })})" +
                   (dtype is null ? "" : $", dtype=np.{dtype}") + (start is null ? "" : $", start=[{string.Join(", ", start)}]") + ")") {
            _nvec = nvec;
            _shape = shape ?? new[] { (long)nvec.Length };
            _start = start;
            _dtype = dtype;
        }

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using NDArray nvec = Counts(_nvec);
            using PyObject pythonNvec = nvec.ToNumpyCopy();
            using PyObject? dtype = _dtype is null ? null : Python.numpy.dtype(_dtype);
            if (_start is null)
                return Python.gymnasium.spaces.MultiDiscrete(pythonNvec, dtype, seed);
            using NDArray start = Counts(_start);
            using PyObject pythonStart = start.ToNumpyCopy();
            return Python.gymnasium.spaces.MultiDiscrete(pythonNvec, dtype, seed, pythonStart);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) {
            using NDArray nvec = Counts(_nvec);
            if (_start is null)
                return new MultiDiscrete(nvec, _dtype is null ? null : np.dtype(_dtype), seed);
            using NDArray start = Counts(_start);
            return new MultiDiscrete(nvec, _dtype is null ? null : np.dtype(_dtype), seed, start);
        }

        /// <summary>A flat list as an int64 array of the spec's shape.</summary>
        /// <param name="values">The values.</param>
        /// <returns>A new array.</returns>
        private NDArray Counts(long[] values) {
            using NDArray flat = np.array(values);
            return flat.reshape(new Shape(_shape)).copy();
        }
    }

    /// <summary><c>Tuple((s0, s1, ...))</c>.</summary>
    internal sealed class TupleSpec : SpaceSpec {
        /// <summary>The subspaces.</summary>
        private readonly SpaceSpec[] _items;

        /// <summary>Describes the space.</summary>
        /// <param name="items">The subspaces.</param>
        public TupleSpec(params SpaceSpec[] items) : base($"Tuple(({string.Join(", ", items.Select(i => i.PythonText))}{(items.Length == 1 ? "," : "")}))") => _items = items;

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject spaces = PythonTuple(_items);
            return Python.gymnasium.spaces.Tuple(spaces, seed);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new Tuple(_items.Select(i => i.BuildDotNet(null)).ToArray(), seed);

        /// <inheritdoc />
        public override IReadOnlyList<SubspaceAccess> Children => _items.Select((spec, i) => new SubspaceAccess(
            $"[{i}]", spec,
            root => {
                using var index = new PyInt(i);
                return Python.@operator.getitem(root, index);
            },
            root => ((Tuple)root)[i])).ToArray();
    }

    /// <summary><c>Dict({k0: s0, ...})</c> (a mapping) or <c>Dict([(k0, s0), ...])</c> (a sequence of pairs).</summary>
    internal sealed class DictSpec : SpaceSpec {
        /// <summary>The keys and subspaces, in the order written.</summary>
        private readonly (string Key, SpaceSpec Space)[] _items;

        /// <summary>Gymnasium's <c>sort_keys</c>.</summary>
        private readonly bool _sortKeys;

        /// <summary>Whether the subspaces are given as a mapping (sorted when <see cref="_sortKeys"/>) rather than a list of pairs.</summary>
        private readonly bool _mapping;

        /// <summary>Describes the space.</summary>
        /// <param name="items">The keys and subspaces, in the order written.</param>
        /// <param name="sortKeys">Gymnasium's <c>sort_keys</c>.</param>
        /// <param name="mapping">A mapping (<see langword="true"/>) or a list of pairs.</param>
        public DictSpec((string Key, SpaceSpec Space)[] items, bool sortKeys = true, bool mapping = true)
            : base("Dict(" + (mapping ? "{" + string.Join(", ", items.Select(i => $"'{i.Key}': {i.Space.PythonText}")) + "}"
                                      : "[" + string.Join(", ", items.Select(i => $"('{i.Key}', {i.Space.PythonText})")) + "]") +
                   (sortKeys ? "" : ", sort_keys=False") + ")") {
            _items = items;
            _sortKeys = sortKeys;
            _mapping = mapping;
        }

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            PyObject container = _mapping ? new PyDict() : new PyList();
            try {
                foreach ((string key, SpaceSpec space) in _items) {
                    using PyObject built = space.BuildPython(null);
                    if (container is PyDict dict) {
                        dict[key] = built;
                    } else {
                        using var name = new PyString(key);
                        using var pair = new PyTuple(new PyObject[] { name, built });
                        ((PyList)container).Append(pair);
                    }
                }

                return Python.gymnasium.spaces.Dict(container, seed, _sortKeys);
            } finally {
                container.Dispose();
            }
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) {
            if (!_mapping)
                return new Dict(_items.Select(i => new KeyValuePair<string, Space>(i.Key, i.Space.BuildDotNet(null))).ToList(), seed, _sortKeys);
            // Dictionary keeps insertion order while nothing is removed, so it stands for the Python dict literal.
            var spaces = new Dictionary<string, Space>();
            foreach ((string key, SpaceSpec space) in _items)
                spaces.Add(key, space.BuildDotNet(null));
            return new Dict(spaces, seed, _sortKeys);
        }

        /// <inheritdoc />
        public override IReadOnlyList<SubspaceAccess> Children => _items.Select(item => new SubspaceAccess(
            $"['{item.Key}']", item.Space,
            root => {
                using var key = new PyString(item.Key);
                return Python.@operator.getitem(root, key);
            },
            root => ((Dict)root)[item.Key])).ToArray();
    }

    /// <summary><c>Text(max_length, min_length=min_length, charset=charset)</c>.</summary>
    internal sealed class TextSpec : SpaceSpec {
        /// <summary>The longest text.</summary>
        private readonly long _max;

        /// <summary>The shortest text.</summary>
        private readonly long _min;

        /// <summary>The characters, or <see langword="null"/> for the alphanumeric default.</summary>
        private readonly string? _charset;

        /// <summary>Describes the space.</summary>
        /// <param name="max">The longest text.</param>
        /// <param name="min">The shortest text.</param>
        /// <param name="charset">The characters, as a <c>str</c>, or <see langword="null"/> for the default.</param>
        public TextSpec(long max, long min = 1, string? charset = null)
            : base($"Text({max}, min_length={min}" + (charset is null ? "" : $", charset='{charset}'") + ")") {
            _max = max;
            _min = min;
            _charset = charset;
        }

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject? charset = _charset is null ? null : PyConvert.ToPython(_charset);
            return Python.gymnasium.spaces.Text(_max, _min, charset, seed);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new Text(_max, _min, _charset, seed);
    }

    /// <summary><c>Sequence(space, stack=stack)</c>.</summary>
    internal sealed class SequenceSpec : SpaceSpec {
        /// <summary>The feature space.</summary>
        private readonly SpaceSpec _feature;

        /// <summary>Gymnasium's <c>stack</c>.</summary>
        private readonly bool _stack;

        /// <summary>Describes the space.</summary>
        /// <param name="feature">The feature space.</param>
        /// <param name="stack">Whether samples are stacked.</param>
        public SequenceSpec(SpaceSpec feature, bool stack = false) : base($"Sequence({feature.PythonText}{(stack ? ", stack=True" : "")})") {
            _feature = feature;
            _stack = stack;
        }

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject feature = _feature.BuildPython(null);
            return Python.gymnasium.spaces.Sequence(feature, seed, _stack);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new Sequence(_feature.BuildDotNet(null), seed, _stack);

        /// <inheritdoc />
        public override IReadOnlyList<SubspaceAccess> Children => new[] {
            new SubspaceAccess(".feature_space", _feature, root => Python.builtins.getattr(root, "feature_space"), root => ((Sequence)root).FeatureSpace),
        };
    }

    /// <summary><c>Graph(node_space, edge_space)</c>.</summary>
    internal sealed class GraphSpec : SpaceSpec {
        /// <summary>The node feature space.</summary>
        private readonly SpaceSpec _node;

        /// <summary>The edge feature space, or <see langword="null"/>.</summary>
        private readonly SpaceSpec? _edge;

        /// <summary>Describes the space.</summary>
        /// <param name="node">The node feature space.</param>
        /// <param name="edge">The edge feature space, or <see langword="null"/> for <c>None</c>.</param>
        public GraphSpec(SpaceSpec node, SpaceSpec? edge) : base($"Graph({node.PythonText}, {edge?.PythonText ?? "None"})") {
            _node = node;
            _edge = edge;
        }

        /// <summary>Whether the graph has an edge space.</summary>
        public bool HasEdges => _edge != null;

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject node = _node.BuildPython(null);
            using PyObject edge = _edge?.BuildPython(null) ?? PyConvert.ToPython(null);
            return Python.gymnasium.spaces.Graph(node, edge, seed);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new Graph(_node.BuildDotNet(null), _edge?.BuildDotNet(null), seed);

        /// <inheritdoc />
        public override IReadOnlyList<SubspaceAccess> Children {
            get {
                var children = new List<SubspaceAccess> {
                    new(".node_space", _node, root => Python.builtins.getattr(root, "node_space"), root => ((Graph)root).NodeSpace),
                };
                if (_edge != null)
                    children.Add(new SubspaceAccess(".edge_space", _edge, root => Python.builtins.getattr(root, "edge_space"), root => ((Graph)root).EdgeSpace!));
                return children;
            }
        }
    }

    /// <summary><c>OneOf((s0, s1, ...))</c>.</summary>
    internal sealed class OneOfSpec : SpaceSpec {
        /// <summary>The subspaces.</summary>
        private readonly SpaceSpec[] _items;

        /// <summary>Describes the space.</summary>
        /// <param name="items">The subspaces.</param>
        public OneOfSpec(params SpaceSpec[] items) : base($"OneOf(({string.Join(", ", items.Select(i => i.PythonText))}{(items.Length == 1 ? "," : "")}))") => _items = items;

        /// <inheritdoc />
        public override PyObject BuildPython(long? seed) {
            using PyObject spaces = PythonTuple(_items);
            return Python.gymnasium.spaces.OneOf(spaces, seed);
        }

        /// <inheritdoc />
        public override Space BuildDotNet(long? seed) => new OneOf(_items.Select(i => i.BuildDotNet(null)).ToArray(), seed);

        /// <inheritdoc />
        public override IReadOnlyList<SubspaceAccess> Children => _items.Select((spec, i) => new SubspaceAccess(
            $"[{i}]", spec,
            root => {
                using var index = new PyInt(i);
                return Python.@operator.getitem(root, index);
            },
            root => ((OneOf)root)[i])).ToArray();
    }
}
