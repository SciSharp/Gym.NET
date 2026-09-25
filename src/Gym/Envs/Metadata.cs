#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using Gym.Utils;
using NumSharp;

namespace Gym.Envs {
    /// <summary>
    ///     An env's <c>metadata</c> dict: an <see cref="Info"/> with the keys Gym reads spelled as they are in Python
    ///     (<c>"render_modes"</c>, <c>"render_fps"</c>), plus typed accessors for them. Iteration order, equality and
    ///     <c>repr</c> are a dict's, so <c>{'render_modes': ['human', 'rgb_array'], 'render_fps': 50}</c> prints and compares
    ///     as in Python.
    /// </summary>
    /// <remarks>
    ///     Store Python lists as <see cref="List{T}"/> and Python ints as <see langword="long"/>, the forms the rest of Gym.NET
    ///     and the parity suite read back as Python's; the typed accessors also accept any string sequence and any number.
    /// </remarks>
    public class Metadata : Info {
        /// <summary>The key of the supported render modes, a list of strings.</summary>
        public const string RenderModesKey = "render_modes";

        /// <summary>The key of the frame rate human rendering and video recording use.</summary>
        public const string RenderFpsKey = "render_fps";

        /// <summary>An empty metadata dict, Python's <c>{}</c> (an env's default is <c>{"render_modes": []}</c>, see <see cref="Env{TObs, TAct}"/>).</summary>
        public Metadata() { }

        /// <summary>A metadata dict holding <paramref name="entries"/> in their order.</summary>
        /// <param name="entries">The entries.</param>
        /// <exception cref="ArgumentNullException">Thrown when <paramref name="entries"/> or one of its keys is <see langword="null"/>.</exception>
        public Metadata(IEnumerable<KeyValuePair<string, object?>> entries) : base(entries) { }

        /// <summary>
        ///     <c>metadata["render_modes"]</c> as strings, or <see langword="null"/> when the key is missing or holds
        ///     <c>None</c>. Setting it stores a <see cref="List{T}"/> (Python's list) under the key, in place if the key exists.
        /// </summary>
        /// <exception cref="InvalidCastException">Thrown by the getter when the stored value is not a sequence of strings.</exception>
        public IReadOnlyList<string>? RenderModes {
            get {
                object? value = Get(RenderModesKey);
                switch (value) {
                    case null:
                        return null;
                    case IReadOnlyList<string> list:
                        return list;
                    case IEnumerable sequence: {
                        var modes = new List<string>();
                        foreach (object? item in sequence)
                            modes.Add(item as string ?? throw new InvalidCastException($"metadata['{RenderModesKey}'] holds {PyBuiltins.Repr(item)}, which is not a render mode string."));
                        return modes;
                    }
                    default:
                        throw new InvalidCastException($"metadata['{RenderModesKey}'] holds {PyBuiltins.Repr(value)} ({PyBuiltins.TypeRepr(value)}), not a list of render modes.");
                }
            }
            set => this[RenderModesKey] = value is null ? null : new List<string>(value);
        }

        /// <summary>
        ///     <c>metadata["render_fps"]</c> as a number, or <see langword="null"/> when the key is missing or holds
        ///     <c>None</c>. Gymnasium stores an int (<c>50</c>) or a float; both read as a <see langword="double"/> through
        ///     Python's <c>float()</c>, which is how <c>pygame.time.Clock.tick</c> takes it. Setting it stores the
        ///     <see langword="double"/>.
        /// </summary>
        /// <exception cref="TypeError">Thrown by the getter when the stored value is not a number (Python's <c>float()</c> message).</exception>
        /// <exception cref="ValueError">Thrown by the getter for a string that is not a float literal.</exception>
        public double? RenderFps {
            get {
                object? value = Get(RenderFpsKey);
                return value is null ? null : PyBuiltins.Float(value);
            }
            set => this[RenderFpsKey] = value;
        }

        /// <inheritdoc />
        protected override Info CreateEmpty() => new Metadata();
    }
}
