using System.Collections.Generic;
using Gym.Spaces;
using Gym.Tests.Parity.Interop;
using Gym.Tests.Parity.Oracle;
using Python.Runtime;

namespace Gym.Tests.Parity.Tests.Spaces {
    /// <summary>The checks and conversions the space cells share.</summary>
    internal static class SpaceParity {
        /// <summary>
        ///     Compares the generator of the root space and of every subspace below it: whether one exists yet (Gymnasium's
        ///     <c>_np_random is None</c>, Gym.NET's <see cref="Space.NpRandomOrNull"/>), and its state when it does.
        /// </summary>
        /// <param name="spec">The root's spec, whose <see cref="SpaceSpec.Descendants"/> name the subspaces.</param>
        /// <param name="python">Gymnasium's root space.</param>
        /// <param name="dotnet">Gym.NET's root space.</param>
        /// <param name="context">What was done before, for reports.</param>
        /// <remarks>
        ///     Reads the private <c>_np_random</c> rather than <c>np_random</c>: the property creates a missing generator,
        ///     and a composite's (a <c>Tuple</c> seeded with a list never creates its own) reseeds every subspace from entropy
        ///     when it does, which would compare two unrelated entropy draws and disturb the streams under test.
        /// </remarks>
        internal static void AssertGeneratorStates(SpaceSpec spec, PyRef python, Space dotnet, string context) {
            ByteContract.AssertSame($"{context}: generator state", PythonState(() => Python.builtins.getattr(python.Object, "_np_random")), DotNetState(dotnet));
            foreach (SubspaceAccess access in spec.Descendants()) {
                ByteContract.AssertSame($"{context}: generator state of {access.Path}",
                    PythonState(() => {
                        using PyObject subspace = access.Python(python.Object);
                        return Python.builtins.getattr(subspace, "_np_random");
                    }),
                    DotNetState(access.DotNet(dotnet)));
            }
        }

        /// <summary>A Gymnasium space's <c>_np_random</c> described: <c>None</c>, or its <c>bit_generator.state</c>.</summary>
        /// <param name="generator">Reads the space's <c>_np_random</c> (a new reference); runs under the GIL.</param>
        /// <returns>The outcome.</returns>
        private static Outcome PythonState(System.Func<PyObject> generator) =>
            Outcome.CapturePython(() => {
                PyObject value = generator();
                if (value.IsNone())
                    return value;
                using (value) {
                    using PyObject bitGenerator = Python.numpy.random.Generator.bit_generator(value);
                    return Python.numpy.random.BitGenerator.state(bitGenerator);
                }
            });

        /// <summary>A Gym.NET space's generator described without creating it: <c>None</c>, or its state.</summary>
        /// <param name="space">The space.</param>
        /// <returns>The outcome.</returns>
        private static Outcome DotNetState(Space space) =>
            Outcome.Returned(space.NpRandomOrNull is { } generator ? RngFacts.Of(generator) : ValueFacts.NoneValue());

        /// <summary>
        ///     The Python value a Gymnasium user passes for a .NET mask, seed or candidate: an <see langword="object"/>?[] is a
        ///     <c>tuple</c> (Gym.NET's tuple form, which <see cref="PyConvert.ToPython"/> would make a list), a string-keyed
        ///     dictionary a <c>dict</c>, a <see cref="List{T}"/> a <c>list</c>, recursively; anything else as
        ///     <see cref="PyConvert.ToPython"/> converts it. Must run under the GIL.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>A new owned reference.</returns>
        internal static PyObject ToPython(object? value) {
            switch (value) {
                case object?[] items: {
                    var converted = new PyObject[items.Length];
                    try {
                        for (int i = 0; i < items.Length; i++)
                            converted[i] = ToPython(items[i]);
                        return new PyTuple(converted);
                    } finally {
                        // The tuple holds its own references.
                        foreach (PyObject? item in converted)
                            item?.Dispose();
                    }
                }
                case List<object?> list: {
                    var converted = new PyList();
                    foreach (object? item in list) {
                        using PyObject one = ToPython(item);
                        converted.Append(one);
                    }

                    return converted;
                }
                case IEnumerable<KeyValuePair<string, object?>> entries: {
                    var converted = new PyDict();
                    foreach ((string key, object? item) in entries) {
                        using PyObject one = ToPython(item);
                        converted[key] = one;
                    }

                    return converted;
                }
                default:
                    return PyConvert.ToPython(value);
            }
        }
    }
}
