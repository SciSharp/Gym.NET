#nullable enable
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Observations;
using Gym.Utils;
using NumSharp;

namespace Gym.Wrappers {
    /// <summary>
    ///     Collects a frame after every reset and step, so <c>render()</c> returns the list of frames since the last call
    ///     (<see cref="RenderResult.Frames"/>) instead of the current one: the <c>*_list</c> render modes. <c>make</c> adds it
    ///     when <c>render_mode="rgb_array_list"</c> is asked of an env that renders <c>rgb_array</c> itself.
    /// </summary>
    /// <typeparam name="TObs">The observation type, passed through.</typeparam>
    /// <typeparam name="TAct">The action type, passed through.</typeparam>
    /// <remarks>
    ///     <para>
    ///     Every reset and step renders the env once more, which costs a frame's drawing each. The render mode reads as the
    ///     env's with <c>_list</c> appended, and the metadata is the env's (a deep copy) with that mode added.
    ///     </para>
    ///     <para>
    ///     Ownership (D19): with <see cref="PopFrames"/> (the default) <see cref="Render"/> hands the collected frames over and
    ///     the caller owns them; without it the frames stay the wrapper's, returned again by every render, and the caller must
    ///     not dispose them. Frames still held are released when the wrapper is disposed.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/wrappers/rendering.py#L34">gymnasium/wrappers/rendering.py</see>.</para>
    /// </remarks>
    public class RenderCollection<TObs, TAct> : Wrapper<TObs, TAct>, IRecordConstructorArgs {
        /// <summary>Wraps <paramref name="env"/>, whose render mode must be a single-frame one.</summary>
        /// <param name="env">The env to collect frames of.</param>
        /// <param name="popFrames">Clear the collection whenever <see cref="Render"/> returns it.</param>
        /// <param name="resetClean">Clear the collection on every reset.</param>
        /// <exception cref="TypeError">Thrown when <paramref name="env"/> is <see langword="null"/>.</exception>
        /// <exception cref="ValueError">
        ///     Thrown when the env has no render mode (<c>Expected env.render_mode to be not None</c>) or already collects
        ///     (<c>Expected env.render_mode to not end with '_list', got 'rgb_array_list'</c>).
        /// </exception>
        public RenderCollection(Env<TObs, TAct> env, bool popFrames = true, bool resetClean = true) : base(env) {
            SavedKwargs = RecordConstructorArgs.Record(new Info { ["pop_frames"] = popFrames, ["reset_clean"] = resetClean });

            string renderMode = env.RenderMode ?? throw new ValueError("Expected env.render_mode to be not None");
            if (renderMode.EndsWith("_list", StringComparison.Ordinal))
                throw new ValueError($"Expected env.render_mode to not end with '_list', got '{renderMode}'");

            FrameList = new List<RenderResult>();
            PopFrames = popFrames;
            ResetClean = resetClean;

            Metadata metadata = (Metadata)env.Metadata.DeepCopy();
            AddRenderMode(metadata, $"{renderMode}_list");
            WrapperMetadata = metadata;
        }

        /// <summary>The recorded constructor arguments, <c>{'pop_frames': True, 'reset_clean': True}</c> (Python's <c>_saved_kwargs</c>).</summary>
        [PyName("_saved_kwargs")]
        public Info SavedKwargs { get; }

        /// <summary>The frames collected since the last render or reset (<c>frame_list</c>), oldest first, owned by the wrapper.</summary>
        public List<RenderResult> FrameList { get; private set; }

        /// <summary>Whether <see cref="Render"/> clears the collection (<c>pop_frames</c>).</summary>
        public bool PopFrames { get; set; }

        /// <summary>Whether a reset clears the collection (<c>reset_clean</c>).</summary>
        public bool ResetClean { get; set; }

        /// <summary>The env's render mode with <c>_list</c> appended (<c>rgb_array_list</c>).</summary>
        /// <exception cref="AttributeError">Thrown when assigned, as for every wrapper.</exception>
        public override string? RenderMode {
            get => $"{Env.RenderMode}_list";
            protected set => base.RenderMode = value;
        }

        /// <summary>Steps the env, then collects a frame.</summary>
        /// <param name="action">The action, passed on.</param>
        /// <returns>The env's transition.</returns>
        public override StepResult<TObs> Step(TAct action) {
            StepResult<TObs> output = base.Step(action);
            FrameList.Add(base.Render());
            return output;
        }

        /// <summary>Resets the env, clears the collection when <see cref="ResetClean"/>, then collects the first frame.</summary>
        /// <param name="seed">The seed, passed on.</param>
        /// <param name="options">The options, passed on.</param>
        /// <returns>The env's first observation and info.</returns>
        /// <exception cref="Gym.Exceptions.Error">Thrown for a negative seed.</exception>
        public override ResetResult<TObs> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            ResetResult<TObs> output = base.Reset(seed, options);
            if (ResetClean)
                ReleaseFrames();
            FrameList.Add(base.Render());
            return output;
        }

        /// <summary>
        ///     The frames collected so far, as one <see cref="RenderResult.Frames"/>; with <see cref="PopFrames"/> the
        ///     collection then starts over and the caller owns the returned frames.
        /// </summary>
        /// <returns>The frames.</returns>
        public override RenderResult Render() {
            var frames = new RenderResult.Frames(FrameList);
            if (PopFrames)
                // The frames now belong to the caller; a fresh list keeps the wrapper from releasing them.
                FrameList = new List<RenderResult>();
            return frames;
        }

        /// <summary>Releases the frames still collected, then closes and disposes the env as every wrapper does.</summary>
        /// <param name="disposing"><see langword="true"/> from <see cref="Env{TObs, TAct}.Dispose()"/>.</param>
        protected override void Dispose(bool disposing) {
            if (disposing)
                ReleaseFrames();
            base.Dispose(disposing);
        }

        /// <summary>Clears the collection, releasing the frames it holds (Python's <c>self.frame_list = []</c>).</summary>
        private void ReleaseFrames() {
            foreach (RenderResult frame in FrameList)
                frame.Dispose();
            FrameList = new List<RenderResult>();
        }

        /// <summary>
        ///     Appends a render mode to <c>metadata["render_modes"]</c> unless it is there (Python's <c>append</c> on the
        ///     copied list). A list that cannot grow (a typed array, Python's list to Gym.NET) is replaced by a longer one.
        /// </summary>
        /// <param name="metadata">The wrapper's own metadata copy.</param>
        /// <param name="mode">The mode to add.</param>
        /// <exception cref="KeyError">Thrown when the metadata has no <c>render_modes</c>, as Python's <c>metadata["render_modes"]</c> raises.</exception>
        /// <exception cref="AttributeError">Thrown when <c>render_modes</c> is not a list (Python's <c>append</c> on another type).</exception>
        private static void AddRenderMode(Metadata metadata, string mode) {
            object? modes = metadata[Metadata.RenderModesKey];
            if (modes is IEnumerable sequence and not string && sequence.Cast<object?>().Any(item => item is string text && text == mode))
                return;
            switch (modes) {
                case IList { IsFixedSize: false, IsReadOnly: false } list:
                    list.Add(mode);
                    break;
                case IEnumerable items and not string:
                    metadata[Metadata.RenderModesKey] = items.Cast<object?>().Select(item => item?.ToString() ?? "None").Append(mode).ToList();
                    break;
                default:
                    throw new AttributeError($"'{PyBuiltins.TypeName(modes)}' object has no attribute 'append'");
            }
        }
    }
}
