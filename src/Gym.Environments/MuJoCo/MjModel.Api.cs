using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Gym.Utils;
using NumSharp;

namespace Gym.MuJoCo {
    public sealed partial class MjModel : IDisposable {
        /// <summary>
        ///     The file names of Gymnasium's 14 bundled models, the only models Gym.NET has (DR-63), in
        ///     <c>gymnasium/envs/mujoco/assets</c>'s order.
        /// </summary>
        internal static readonly string[] BundledNames = [
            "ant.xml", "half_cheetah.xml", "hopper.xml", "humanoid.xml", "humanoidstandup.xml", "inverted_double_pendulum.xml",
            "inverted_pendulum.xml", "point.xml", "pusher.xml", "pusher_v5.xml", "reacher.xml", "swimmer.xml", "walker2d.xml",
            "walker2d_v5.xml",
        ];

        /// <summary>The embedded models' JSON by file name, read once per process; each model gets its own arrays.</summary>
        private static readonly ConcurrentDictionary<string, byte[]> BundledJson = new(StringComparer.Ordinal);

        /// <summary>The writable views handed out (each a root the public members view again), released by <see cref="Dispose"/>.</summary>
        private NDArray? actuatorCtrlrangeRoot, bodyMassRoot, geomRgbaRoot;

        /// <summary>
        ///     Loads one of Gymnasium's bundled models by its file name (<c>mujoco.MjModel.from_xml_path</c> as Gymnasium's
        ///     <c>MujocoEnv</c> calls it): <c>"ant.xml"</c>, <c>"walker2d_v5.xml"</c> and the other twelve, each compiled
        ///     once by MuJoCo 4.0.0 and embedded (DR-63). Every call returns a new model with its own arrays, as Python's
        ///     does, so writing one model's colours or options never reaches another env's.
        /// </summary>
        /// <param name="filename">A bundled file name, with no folder (<c>"hopper.xml"</c>).</param>
        /// <returns>The model.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="filename"/> is null.</exception>
        /// <exception cref="NotSupportedException">
        ///     Any other name or path: Gym.NET has no MJCF compiler, so a custom model cannot be loaded; the message lists the
        ///     names it has. Gymnasium would compile the file (or raise <c>OSError</c> for a missing one).
        /// </exception>
        /// <remarks>
        ///     Footgun: the XML files themselves are not in the package, only their compiled form, so a path to a copy of
        ///     Gymnasium's own <c>ant.xml</c> is refused like any other path; pass the bare name.
        /// </remarks>
        [PyName("from_xml_path")]
        public static MjModel FromXmlPath(string filename) {
            ArgumentNullException.ThrowIfNull(filename);
            if (Array.IndexOf(BundledNames, filename) < 0)
                throw new NotSupportedException(
                    $"Gym.NET embeds Gymnasium's {BundledNames.Length} compiled MuJoCo models and has no MJCF compiler (DR-63): "
                    + $"'{filename}' is not one of {string.Join(", ", BundledNames)}.");
            byte[] json = BundledJson.GetOrAdd(filename, ReadBundled);
            return Load(json);
        }

        /// <summary>Reads an embedded model's JSON (<c>assets/mujoco/&lt;name&gt;.json</c>).</summary>
        /// <param name="filename">The XML file name the model was compiled from.</param>
        /// <returns>The UTF-8 JSON.</returns>
        /// <exception cref="FileNotFoundException">The assembly does not embed it (a build that left the assets out).</exception>
        private static byte[] ReadBundled(string filename) {
            string resource = "assets/mujoco/" + Path.GetFileNameWithoutExtension(filename) + ".json";
            using Stream stream = typeof(MjModel).Assembly.GetManifestResourceStream(resource)
                                  ?? throw new FileNotFoundException($"Gym.Environments embeds no model '{resource}'.", resource);
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            return buffer.ToArray();
        }

        /// <summary>Number of position coordinates (<c>nq</c>): a free joint has 7 for its 6 degrees of freedom.</summary>
        [PyName("nq")]
        public int Nq => nq;

        /// <summary>Number of degrees of freedom (<c>nv</c>), the length of <see cref="MjData.Qvel"/>.</summary>
        [PyName("nv")]
        public int Nv => nv;

        /// <summary>Number of actuators (<c>nu</c>), the length of <see cref="MjData.Ctrl"/> and of an env's action.</summary>
        [PyName("nu")]
        public int Nu => nu;

        /// <summary>Number of activation states (<c>na</c>): 0 for every bundled model, which has no stateful actuator.</summary>
        [PyName("na")]
        public int Na => na;

        /// <summary>Number of bodies (<c>nbody</c>), the world body included.</summary>
        [PyName("nbody")]
        public int Nbody => nbody;

        /// <summary>Number of joints (<c>njnt</c>).</summary>
        [PyName("njnt")]
        public int Njnt => njnt;

        /// <summary>Number of geoms (<c>ngeom</c>).</summary>
        [PyName("ngeom")]
        public int Ngeom => ngeom;

        /// <summary>Number of tendons (<c>ntendon</c>): Humanoid's and HumanoidStandup's two fixed tendons, 0 elsewhere.</summary>
        [PyName("ntendon")]
        public int Ntendon => ntendon;

        /// <summary>
        ///     Number of nodes of the collision midphase's bounding-volume hierarchies (<c>nbvh</c>), as MuJoCo compiled them;
        ///     Gymnasium's tests use it to tell the <c>xml_file</c> variants apart. The engine itself does not read it.
        /// </summary>
        [PyName("nbvh")]
        public int Nbvh => nbvh;

        /// <summary>Number of cameras (<c>ncam</c>) the model defines (Humanoid's two, Ant's one, none elsewhere).</summary>
        [PyName("ncam")]
        public int Ncam => ncam;

        /// <summary>The physics options (<c>opt</c>): the time step Gymnasium reads.</summary>
        [PyName("opt")]
        public MjOption Opt => new(this);

        /// <summary>The model statistics (<c>stat</c>): the extent the free camera's distance starts from.</summary>
        [PyName("stat")]
        public MjStatistic Stat => new(this);

        /// <summary>The visualization options (<c>vis</c>): the offscreen buffer's size Gymnasium's envs grow.</summary>
        [PyName("vis")]
        public MjVisual Vis => new(this);

        /// <summary>
        ///     The actuators' control ranges (<c>actuator_ctrlrange</c>), shape <c>(nu, 2)</c>, float64: the bounds of an env's
        ///     action space after their cast to float32. A writable view of the model's own array: the caller owns the view
        ///     (dispose it), and writes reach the model, as in Python.
        /// </summary>
        [PyName("actuator_ctrlrange")]
        public NDArray ActuatorCtrlrange => View(ref actuatorCtrlrangeRoot, actuator_ctrlrange, nu, 2);

        /// <summary>
        ///     The bodies' masses (<c>body_mass</c>), shape <c>(nbody,)</c>, float64, as Humanoid's <c>mass_center</c> weighs
        ///     them. A writable view: writing a mass changes what the env computes from it, but not the derived quantities
        ///     MuJoCo's compiler set (subtree masses, inertias), exactly as writing it in Python does without
        ///     <c>mj_setConst</c>.
        /// </summary>
        [PyName("body_mass")]
        public NDArray BodyMass => View(ref bodyMassRoot, body_mass, nbody, 0);

        /// <summary>
        ///     The geoms' colours (<c>geom_rgba</c>), shape <c>(ngeom, 4)</c>, float32 as in MuJoCo, red, green, blue and alpha
        ///     in [0, 1]. A writable view; Gymnasium's renderer fades the geoms by writing their alpha.
        /// </summary>
        [PyName("geom_rgba")]
        public NDArray GeomRgba => View(ref geomRgbaRoot, geom_rgba, ngeom, 4);

        /// <summary>
        ///     A new writable view of one of the model's arrays: the root (the pinned array, shaped) is made once, and each
        ///     call returns a view of it, which the caller owns.
        /// </summary>
        /// <typeparam name="T">The element type (double, or float for colours).</typeparam>
        /// <param name="root">The cached root, created on first use.</param>
        /// <param name="array">The model's array.</param>
        /// <param name="rows">The first dimension.</param>
        /// <param name="cols">The second dimension, or 0 for a one-dimensional view.</param>
        /// <returns>A view sharing the array's memory.</returns>
        /// <exception cref="ObjectDisposedException">The model was disposed.</exception>
        private NDArray View<T>(ref NDArray? root, T[] array, int rows, int cols) where T : unmanaged {
            ObjectDisposedException.ThrowIf(disposed, this);
            root ??= MjViews.Root(array, rows, cols);
            return root.view();
        }

        /// <summary>Whether <see cref="Dispose"/> has run.</summary>
        private bool disposed;

        /// <summary>
        ///     Releases the views' roots, which pin the model's arrays; views the caller still holds keep working until
        ///     disposed. The model's own numbers stay usable by the engine; only the public views end.
        /// </summary>
        public void Dispose() {
            if (disposed)
                return;
            disposed = true;
            actuatorCtrlrangeRoot?.Dispose();
            bodyMassRoot?.Dispose();
            geomRgbaRoot?.Dispose();
        }

        /// <summary>
        ///     The index of a named element of a kind (<c>mj_name2id</c>'s lookup), by MuJoCo's kind abbreviation.
        /// </summary>
        /// <param name="kind">The kind (<c>body</c>, <c>jnt</c>, <c>geom</c>, <c>site</c>, <c>cam</c>, <c>light</c>,
        ///     <c>actuator</c>, <c>tendon</c>, <c>mat</c>, <c>tex</c>).</param>
        /// <param name="name">The name; the empty name matches nothing, as in MuJoCo.</param>
        /// <returns>The index, or -1.</returns>
        internal int NameToId(string kind, string name) {
            if (name.Length == 0 || !names.TryGetValue(kind, out string[]? list))
                return -1;
            return Array.IndexOf(list, name);
        }

        /// <summary>The names of one kind, in index order (<c>""</c> for an unnamed element), or none.</summary>
        /// <param name="kind">MuJoCo's kind abbreviation.</param>
        /// <returns>The names.</returns>
        internal IReadOnlyList<string> NamesOf(string kind) => names.TryGetValue(kind, out string[]? list) ? list : Array.Empty<string>();
    }

    /// <summary>
    ///     A model's physics options (<c>mujoco.MjOption</c>, <c>model.opt</c>): what Gymnasium's code reads of them. A
    ///     handle on the model, so it always reads the model's current values.
    /// </summary>
    [PyName("MjOption")]
    public sealed class MjOption {
        /// <summary>The model, borrowed: this is a handle on its fields and owns nothing.</summary>
        [NDBorrowed]
        private readonly MjModel model;

        /// <summary>Wraps a model's options.</summary>
        /// <param name="model">The model.</param>
        internal MjOption(MjModel model) => this.model = model;

        /// <summary>
        ///     The simulation time step in seconds (<c>timestep</c>); an env's <c>dt</c> is this times its frame skip.
        ///     Writable, as Python's is: the next physics step uses the new value.
        /// </summary>
        /// <remarks>
        ///     Footgun: an env's frame rate (<c>metadata["render_fps"]</c>) was derived from the step at construction and is not
        ///     updated, as in Gymnasium.
        /// </remarks>
        [PyName("timestep")]
        public double Timestep {
            get => model.timestep;
            set => model.timestep = value;
        }
    }

    /// <summary>
    ///     A model's statistics (<c>mujoco.MjStatistic</c>, <c>model.stat</c>), as MuJoCo's compiler computed them: what
    ///     Gymnasium's code reads of them.
    /// </summary>
    [PyName("MjStatistic")]
    public sealed class MjStatistic {
        /// <summary>The model, borrowed: this is a handle on its fields and owns nothing.</summary>
        [NDBorrowed]
        private readonly MjModel model;

        /// <summary>Wraps a model's statistics.</summary>
        /// <param name="model">The model.</param>
        internal MjStatistic(MjModel model) => this.model = model;

        /// <summary>The model's spatial extent (<c>extent</c>), the free camera's default distance in Gymnasium's renderer.</summary>
        [PyName("extent")]
        public double Extent => model.extent;
    }

    /// <summary>
    ///     A model's visualization options (<c>mujoco.MjVisual</c>, <c>model.vis</c>): what Gymnasium's code reads and
    ///     writes of them.
    /// </summary>
    [PyName("MjVisual")]
    public sealed class MjVisual {
        /// <summary>The model, borrowed: this is a handle on its fields and owns nothing.</summary>
        [NDBorrowed]
        private readonly MjModel model;

        /// <summary>Wraps a model's visualization options.</summary>
        /// <param name="model">The model.</param>
        internal MjVisual(MjModel model) => this.model = model;

        /// <summary>The global options (<c>global_</c>, Python's spelling of C's <c>global</c>).</summary>
        [PyName("global_")]
        public MjVisualGlobal Global => new(model);
    }

    /// <summary>
    ///     A model's global visualization options (<c>model.vis.global_</c>): the offscreen buffer's size, which
    ///     Gymnasium's <c>MujocoEnv</c> grows to its render size and never shrinks.
    /// </summary>
    [PyName("Global")]
    public sealed class MjVisualGlobal {
        /// <summary>The model, borrowed: this is a handle on its fields and owns nothing.</summary>
        [NDBorrowed]
        private readonly MjModel model;

        /// <summary>Wraps a model's global visualization options.</summary>
        /// <param name="model">The model.</param>
        internal MjVisualGlobal(MjModel model) => this.model = model;

        /// <summary>The offscreen buffer's width in pixels (<c>offwidth</c>); writable, as in Python.</summary>
        [PyName("offwidth")]
        public int Offwidth {
            get => model.offwidth;
            set => model.offwidth = value;
        }

        /// <summary>The offscreen buffer's height in pixels (<c>offheight</c>); writable, as in Python.</summary>
        [PyName("offheight")]
        public int Offheight {
            get => model.offheight;
            set => model.offheight = value;
        }
    }

    /// <summary>
    ///     The views <see cref="MjModel"/> and <see cref="MjData"/> hand out: NumSharp arrays over the engine's own managed
    ///     arrays, pinned, so a write through a view reaches the simulation, as Python's views of <c>mjModel</c> and
    ///     <c>mjData</c> do.
    /// </summary>
    /// <remarks>
    ///     NumSharp counts references to a buffer: the root made here holds one, every view of it another, and the pin is
    ///     released when the last is disposed. The owner keeps the root until its own <c>Dispose</c>, so a caller disposing
    ///     its view never unpins the engine's array.
    /// </remarks>
    internal static class MjViews {
        /// <summary>A root over an engine array: the pinned array, reshaped to two dimensions when asked.</summary>
        /// <typeparam name="T">The element type.</typeparam>
        /// <param name="array">The engine's array (pinned for as long as the root or a view of it lives).</param>
        /// <param name="rows">The first dimension.</param>
        /// <param name="cols">The second dimension, or 0 to keep one dimension of <paramref name="rows"/>.</param>
        /// <returns>The root.</returns>
        /// <exception cref="ArgumentException">The array's length is not <paramref name="rows"/> × <paramref name="cols"/>.</exception>
        public static NDArray Root<T>(T[] array, int rows, int cols) where T : unmanaged {
            long length = cols == 0 ? rows : (long)rows * cols;
            // every array a view exposes is sized exactly by the model (the scratch-sized contact arrays never are)
            if (array.Length != length)
                throw new ArgumentException($"an engine array of {array.Length} values is not {length}", nameof(array));
            NDArray flat = np.array(array, copy: false);
            if (cols == 0)
                return flat;
            // the reshaped view holds its own reference to the pinned buffer, so the flat wrapper can go
            using (flat)
                return flat.reshape(rows, cols);
        }
    }
}
