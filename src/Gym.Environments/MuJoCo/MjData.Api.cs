using System;
using System.Linq;
using Gym.Utils;
using NumSharp;

namespace Gym.MuJoCo {
    public sealed partial class MjData : IDisposable {
        /// <summary>The activation states (<c>act</c>): none, since no bundled model has a stateful actuator.</summary>
        private readonly double[] act = [];

        /// <summary>The roots of the views handed out (each pins one array), released by <see cref="Dispose"/>.</summary>
        private NDArray? qposRoot, qvelRoot, ctrlRoot, actRoot, cfrcExtRoot, cvelRoot, cinertRoot, qfrcActuatorRoot,
                         qfrcConstraintRoot, tenLengthRoot, tenVelocityRoot, siteXposRoot, xiposRoot, geomXposRoot, xposRoot,
                         solverNiterRoot;

        /// <summary>Whether <see cref="Dispose"/> has run.</summary>
        private bool disposed;

        /// <summary>The model this data was made for, which <see cref="Mujoco"/>'s functions require to be passed back.</summary>
        internal MjModel Model => m;

        /// <summary>
        ///     The joint positions (<c>qpos</c>), shape <c>(nq,)</c>, float64: a free joint's 3 coordinates and unit
        ///     quaternion, then one value per hinge or slide. A writable view of the simulation's own array, as Python's is:
        ///     <c>Data.Qpos[2] = 1.0</c> moves the model at the next <see cref="Mujoco.MjForward"/>. The caller owns the view
        ///     and should dispose it; disposing it never affects the simulation.
        /// </summary>
        [PyName("qpos")]
        public NDArray Qpos => View(ref qposRoot, qpos, m.nq, 0);

        /// <summary>The joint velocities (<c>qvel</c>), shape <c>(nv,)</c>, float64; a writable view, as <see cref="Qpos"/>.</summary>
        [PyName("qvel")]
        public NDArray Qvel => View(ref qvelRoot, qvel, nv, 0);

        /// <summary>
        ///     The controls (<c>ctrl</c>), shape <c>(nu,)</c>, float64: what the actuators apply at the next step, clamped to
        ///     their control ranges by the engine (not by the write). A writable view, as <see cref="Qpos"/>.
        /// </summary>
        [PyName("ctrl")]
        public NDArray Ctrl => View(ref ctrlRoot, ctrl, nu, 0);

        /// <summary>
        ///     The actuator activations (<c>act</c>), shape <c>(na,)</c>: always empty, since no bundled model has a stateful
        ///     actuator (Gymnasium's <c>set_state</c> clears it anyway).
        /// </summary>
        [PyName("act")]
        public NDArray Act => View(ref actRoot, act, 0, 0);

        /// <summary>The simulation time in seconds (<c>time</c>), advanced by each step's time step; writable, as in Python.</summary>
        [PyName("time")]
        public double Time {
            get => time;
            set => time = value;
        }

        /// <summary>
        ///     The external contact forces on each body (<c>cfrc_ext</c>), shape <c>(nbody, 6)</c>, float64, torque then force
        ///     in the body's center-of-mass frame. Footgun: only <see cref="Mujoco.MjRnePostConstraint"/> computes it (Gymnasium's
        ///     envs call it after each step's physics), so after a reset it holds zeros.
        /// </summary>
        [PyName("cfrc_ext")]
        public NDArray CfrcExt => View(ref cfrcExtRoot, cfrc_ext, nbody, 6);

        /// <summary>The bodies' velocities in their center-of-mass frames (<c>cvel</c>), shape <c>(nbody, 6)</c>, angular then linear.</summary>
        [PyName("cvel")]
        public NDArray Cvel => View(ref cvelRoot, cvel, nbody, 6);

        /// <summary>The bodies' inertias in their center-of-mass frames (<c>cinert</c>), shape <c>(nbody, 10)</c>, as MuJoCo packs them.</summary>
        [PyName("cinert")]
        public NDArray Cinert => View(ref cinertRoot, cinert, nbody, 10);

        /// <summary>The actuator forces in joint space (<c>qfrc_actuator</c>), shape <c>(nv,)</c>.</summary>
        [PyName("qfrc_actuator")]
        public NDArray QfrcActuator => View(ref qfrcActuatorRoot, qfrc_actuator, nv, 0);

        /// <summary>The constraint forces in joint space (<c>qfrc_constraint</c>), shape <c>(nv,)</c>.</summary>
        [PyName("qfrc_constraint")]
        public NDArray QfrcConstraint => View(ref qfrcConstraintRoot, qfrc_constraint, nv, 0);

        /// <summary>The tendon lengths (<c>ten_length</c>), shape <c>(ntendon,)</c>: Humanoid's two, empty elsewhere.</summary>
        [PyName("ten_length")]
        public NDArray TenLength => View(ref tenLengthRoot, ten_length, m.ntendon, 0);

        /// <summary>The tendon velocities (<c>ten_velocity</c>), shape <c>(ntendon,)</c>.</summary>
        [PyName("ten_velocity")]
        public NDArray TenVelocity => View(ref tenVelocityRoot, ten_velocity, m.ntendon, 0);

        /// <summary>The sites' positions in the world (<c>site_xpos</c>), shape <c>(nsite, 3)</c>: InvertedDoublePendulum's tip.</summary>
        [PyName("site_xpos")]
        public NDArray SiteXpos => View(ref siteXposRoot, site_xpos, m.nsite, 3);

        /// <summary>The bodies' centers of mass in the world (<c>xipos</c>), shape <c>(nbody, 3)</c>, as Humanoid's <c>mass_center</c> weighs them.</summary>
        [PyName("xipos")]
        public NDArray Xipos => View(ref xiposRoot, xipos, nbody, 3);

        /// <summary>The geoms' positions in the world (<c>geom_xpos</c>), shape <c>(ngeom, 3)</c>; the free camera looks at their median.</summary>
        [PyName("geom_xpos")]
        public NDArray GeomXpos => View(ref geomXposRoot, geom_xpos, m.ngeom, 3);

        /// <summary>
        ///     The solver's iterations per constraint island at the last step (<c>solver_niter</c>), shape <c>(20,)</c>, int32:
        ///     MuJoCo's <c>mjNISLAND</c> entries, all 0 when there were no constraints.
        /// </summary>
        [PyName("solver_niter")]
        public NDArray SolverNiter => View(ref solverNiterRoot, solver_niter_island, solver_niter_island.Length, 0);

        /// <summary>
        ///     The body named <paramref name="name"/>'s quantities (<c>data.body(name)</c>): its <see cref="MjDataBody.Xpos"/>,
        ///     which <c>MujocoEnv.get_body_com</c> returns.
        /// </summary>
        /// <param name="name">The body's name in the model.</param>
        /// <returns>A view of the body.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="name"/> is null.</exception>
        /// <exception cref="KeyError">No body has that name, with MuJoCo's message listing the valid names in sorted order.</exception>
        [PyName("body")]
        public MjDataBody Body(string name) {
            ArgumentNullException.ThrowIfNull(name);
            int id = name.Length == 0 ? -1 : Array.IndexOf(m.body_names, name);
            if (id < 0) {
                // MuJoCo's text holds Python's repr of the sorted list (['left_foot', ..., 'world']), and a KeyError's text is
                // the repr of its argument, as Gym's other KeyErrors give it
                string valid = string.Join(", ", m.body_names.OrderBy(n => n, StringComparer.Ordinal).Select(PyBuiltins.Repr));
                throw new KeyError(PyBuiltins.Repr($"Invalid name {PyBuiltins.Repr(name)}. Valid names: [{valid}]"));
            }
            return new MjDataBody(this, id);
        }

        /// <summary>The body with index <paramref name="id"/>'s quantities (<c>data.body(id)</c>).</summary>
        /// <param name="id">The body's index, 0 (the world) to <c>nbody - 1</c>.</param>
        /// <returns>A view of the body.</returns>
        /// <exception cref="IndexError">The index is out of range, with MuJoCo's message.</exception>
        [PyName("body")]
        public MjDataBody Body(int id) {
            if (id < 0 || id >= nbody)
                throw new IndexError($"Invalid index {id}. Valid indices from 0 to {nbody - 1}");
            return new MjDataBody(this, id);
        }

        /// <summary>One body's position in the world, a view of row <paramref name="id"/> of the body positions.</summary>
        /// <param name="id">The body.</param>
        /// <returns>The writable view, shape <c>(3,)</c>, which the caller owns.</returns>
        /// <exception cref="ObjectDisposedException">The data was disposed.</exception>
        internal NDArray BodyXpos(int id) {
            ObjectDisposedException.ThrowIf(disposed, this);
            xposRoot ??= MjViews.Root(xpos, nbody, 3);
            return xposRoot[id];
        }

        /// <summary>A new writable view of one of the data's arrays, over a root made once.</summary>
        /// <typeparam name="T">The element type (double, or int for the iteration counts).</typeparam>
        /// <param name="root">The cached root, created on first use.</param>
        /// <param name="array">The data's array.</param>
        /// <param name="rows">The first dimension.</param>
        /// <param name="cols">The second dimension, or 0 for a one-dimensional view.</param>
        /// <returns>A view sharing the array's memory, which the caller owns.</returns>
        /// <exception cref="ObjectDisposedException">The data was disposed.</exception>
        private NDArray View<T>(ref NDArray? root, T[] array, int rows, int cols) where T : unmanaged {
            ObjectDisposedException.ThrowIf(disposed, this);
            root ??= MjViews.Root(array, rows, cols);
            return root.view();
        }

        /// <summary>
        ///     Releases the views' roots, which pin the data's arrays; views a caller still holds keep working until disposed.
        ///     Gym.NET's MuJoCo envs dispose their data when closed.
        /// </summary>
        public void Dispose() {
            if (disposed)
                return;
            disposed = true;
            qposRoot?.Dispose();
            qvelRoot?.Dispose();
            ctrlRoot?.Dispose();
            actRoot?.Dispose();
            cfrcExtRoot?.Dispose();
            cvelRoot?.Dispose();
            cinertRoot?.Dispose();
            qfrcActuatorRoot?.Dispose();
            qfrcConstraintRoot?.Dispose();
            tenLengthRoot?.Dispose();
            tenVelocityRoot?.Dispose();
            siteXposRoot?.Dispose();
            xiposRoot?.Dispose();
            geomXposRoot?.Dispose();
            xposRoot?.Dispose();
            solverNiterRoot?.Dispose();
        }
    }

    /// <summary>
    ///     One body's quantities in a simulation (<c>data.body(name)</c>, Python's <c>_MjDataBodyViews</c>): what
    ///     Gymnasium's code reads of it.
    /// </summary>
    [PyName("_MjDataBodyViews")]
    public sealed class MjDataBody {
        /// <summary>The data the body belongs to, borrowed: the body view is a handle and owns nothing.</summary>
        [NDBorrowed]
        private readonly MjData data;

        /// <summary>Wraps a body of a simulation.</summary>
        /// <param name="data">The data.</param>
        /// <param name="id">The body's index.</param>
        internal MjDataBody(MjData data, int id) {
            this.data = data;
            Id = id;
        }

        /// <summary>The body's index (<c>id</c>).</summary>
        [PyName("id")]
        public int Id { get; }

        /// <summary>The body's name (<c>name</c>).</summary>
        [PyName("name")]
        public string Name => data.Model.body_names[Id];

        /// <summary>
        ///     The body frame's position in the world (<c>xpos</c>), shape <c>(3,)</c>, float64: a writable view of the
        ///     simulation's array, which the caller owns. It is what <c>get_body_com</c> returns, the frame's origin rather than
        ///     the center of mass (<see cref="MjData.Xipos"/>), whatever the name says.
        /// </summary>
        [PyName("xpos")]
        public NDArray Xpos => data.BodyXpos(Id);
    }
}
