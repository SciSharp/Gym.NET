using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace Gym.MuJoCo {
    /// <summary>
    ///     A compiled MuJoCo model, what <c>mujoco.MjModel</c> is to Gymnasium's MuJoCo envs: one of Gymnasium's 14 bundled
    ///     models as MuJoCo 4.0.0 compiled it, embedded in Gym.Environments (DR-63), so every number is MuJoCo's own and
    ///     Gym.NET needs no MJCF compiler. It holds what the eleven envs need (HalfCheetah, Hopper, Walker2d, Ant, Humanoid,
    ///     HumanoidStandup, InvertedPendulum, InvertedDoublePendulum, Reacher, Swimmer, Pusher) and Gymnasium's custom-env test
    ///     model, point.xml: kinematic trees of hinge and slide joints, each under an optional free root; a ground plane with
    ///     spheres, capsules, cylinders and point.xml's box (tested against the plane, refused if it would touch); motors on
    ///     joints; Euler or RK4; pyramidal and frictionless contacts; the dense Newton or PGS solver; inert fixed tendons;
    ///     sites; and the inertia-box fluid model.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Its public members are what Gymnasium's envs, utils and tests read of <c>mujoco.MjModel</c>, under .NET names,
    ///         each <c>[PyName]</c>d to Python's (DR-68); the internal fields keep MuJoCo's names and layouts (row-major,
    ///         <c>3*i</c>, <c>4*i</c>, <c>9*i</c> strides), which <see cref="MjData"/>'s engine reads.
    ///     </para>
    ///     <para>
    ///         <see cref="Load(byte[])"/> refuses any model that uses a feature the engine does not reproduce (ball joints,
    ///         springs on free joints, active tendons, equality constraints, frictionloss, colliding geoms of other types,
    ///         elliptic cones, the ellipsoid fluid model, stateful actuators, PGS over several kinematic trees, other
    ///         integrators or solvers), so a model that loads is one the engine steps exactly, as MuJoCo 4.0.0 does on
    ///         win-x64 (DR-64). The bundled models use none of them.
    ///     </para>
    /// </remarks>
    public sealed partial class MjModel {
        // sizes
        /// <summary>Number of position coordinates (a free joint has 7 for its 6 dofs).</summary>
        internal int nq;
        /// <summary>Number of degrees of freedom.</summary>
        internal int nv;
        /// <summary>Number of actuators (and controls).</summary>
        internal int nu;
        /// <summary>Number of bodies, the world included.</summary>
        internal int nbody;
        /// <summary>Number of joints.</summary>
        internal int njnt;
        /// <summary>Number of geoms.</summary>
        internal int ngeom;
        /// <summary>Number of stored entries of the sparse inertia matrix.</summary>
        internal int nC;
        /// <summary>Number of sites (frames the env code reads, such as InvertedDoublePendulum's tip).</summary>
        internal int nsite;
        /// <summary>Number of kinematic trees (subtrees hanging from the world); constraint islands are unions of them.</summary>
        internal int ntree;

        // bodies
        /// <summary>Parent body of each body.</summary>
        internal int[] body_parentid = [];
        /// <summary>Root body of each body's tree (the subtree whose center of mass anchors the motion dofs).</summary>
        internal int[] body_rootid = [];
        /// <summary>The body each body is welded to (itself for a body with joints).</summary>
        internal int[] body_weldid = [];
        /// <summary>Joints per body.</summary>
        internal int[] body_jntnum = [];
        /// <summary>First joint of each body.</summary>
        internal int[] body_jntadr = [];
        /// <summary>Dofs per body.</summary>
        internal int[] body_dofnum = [];
        /// <summary>First dof of each body.</summary>
        internal int[] body_dofadr = [];
        /// <summary>First geom of each body.</summary>
        internal int[] body_geomadr = [];
        /// <summary>Geoms per body.</summary>
        internal int[] body_geomnum = [];
        /// <summary>How each body's inertial frame relates to its frame (MuJoCo's <c>mjtSameFrame</c>).</summary>
        internal int[] body_sameframe = [];
        /// <summary>Body-level contype (the OR of its geoms').</summary>
        internal int[] body_contype = [];
        /// <summary>Body-level conaffinity (the OR of its geoms').</summary>
        internal int[] body_conaffinity = [];
        /// <summary>Body positions relative to their parents.</summary>
        internal double[] body_pos = [];
        /// <summary>Body orientations relative to their parents.</summary>
        internal double[] body_quat = [];
        /// <summary>Inertial frame positions in the body frames.</summary>
        internal double[] body_ipos = [];
        /// <summary>Inertial frame orientations in the body frames.</summary>
        internal double[] body_iquat = [];
        /// <summary>Body masses.</summary>
        internal double[] body_mass = [];
        /// <summary>Masses of the subtrees rooted at each body.</summary>
        internal double[] body_subtreemass = [];
        /// <summary>Principal inertias.</summary>
        internal double[] body_inertia = [];
        /// <summary>Inverse translational and rotational weights at the reference pose (2 per body).</summary>
        internal double[] body_invweight0 = [];

        // joints
        /// <summary>Joint types (2 slide, 3 hinge).</summary>
        internal int[] jnt_type = [];
        /// <summary>Address of each joint in qpos.</summary>
        internal int[] jnt_qposadr = [];
        /// <summary>Address of each joint in qvel.</summary>
        internal int[] jnt_dofadr = [];
        /// <summary>Whether each joint has a range limit.</summary>
        internal bool[] jnt_limited = [];
        /// <summary>Joint anchors in the body frames.</summary>
        internal double[] jnt_pos = [];
        /// <summary>Joint axes in the body frames.</summary>
        internal double[] jnt_axis = [];
        /// <summary>Joint stiffness (linear spring coefficient).</summary>
        internal double[] jnt_stiffness = [];
        /// <summary>Joint stiffness polynomial coefficients (2 per joint).</summary>
        internal double[] jnt_stiffnesspoly = [];
        /// <summary>Joint ranges (2 per joint).</summary>
        internal double[] jnt_range = [];
        /// <summary>Joint limit margins.</summary>
        internal double[] jnt_margin = [];
        /// <summary>Joint limit solref (2 per joint).</summary>
        internal double[] jnt_solref = [];
        /// <summary>Joint limit solimp (5 per joint).</summary>
        internal double[] jnt_solimp = [];

        // dofs
        /// <summary>Body of each dof.</summary>
        internal int[] dof_bodyid = [];
        /// <summary>Joint of each dof.</summary>
        internal int[] dof_jntid = [];
        /// <summary>Parent dof of each dof (-1 at a root).</summary>
        internal int[] dof_parentid = [];
        /// <summary>Dof damping.</summary>
        internal double[] dof_damping = [];
        /// <summary>Dof damping polynomial coefficients (2 per dof).</summary>
        internal double[] dof_dampingpoly = [];
        /// <summary>Dof armature inertia.</summary>
        internal double[] dof_armature = [];
        /// <summary>Inverse dof weights at the reference pose.</summary>
        internal double[] dof_invweight0 = [];
        /// <summary>
        ///     MuJoCo's simple dofs: nonzero for a dof of a body whose mass-matrix rows are diagonal and constant (sliders at a
        ///     root, such as Reacher's target), whose diagonal is then <see cref="dof_M0"/> rather than computed.
        /// </summary>
        internal int[] dof_simplenum = [];
        /// <summary>Each dof's diagonal of the mass matrix at the reference pose (used for simple dofs).</summary>
        internal double[] dof_M0 = [];
        /// <summary>The kinematic tree of each dof.</summary>
        internal int[] dof_treeid = [];
        /// <summary>The kinematic tree of each body (-1 for the world and bodies welded to it).</summary>
        internal int[] body_treeid = [];
        /// <summary>Dofs per kinematic tree.</summary>
        internal int[] tree_dofnum = [];

        // geoms
        /// <summary>Geom types (0 plane, 3 capsule).</summary>
        internal int[] geom_type = [];
        /// <summary>Geom contype bitmasks.</summary>
        internal int[] geom_contype = [];
        /// <summary>Geom conaffinity bitmasks.</summary>
        internal int[] geom_conaffinity = [];
        /// <summary>Geom contact dimensionality.</summary>
        internal int[] geom_condim = [];
        /// <summary>Body of each geom.</summary>
        internal int[] geom_bodyid = [];
        /// <summary>How each geom's frame relates to its body's frames.</summary>
        internal int[] geom_sameframe = [];
        /// <summary>Geom contact priorities.</summary>
        internal int[] geom_priority = [];
        /// <summary>Geom sizes (3 per geom; capsule: radius, half-length).</summary>
        internal double[] geom_size = [];
        /// <summary>Bounding sphere radii (0 for a plane).</summary>
        internal double[] geom_rbound = [];
        /// <summary>Geom positions in the body frames.</summary>
        internal double[] geom_pos = [];
        /// <summary>Geom orientations in the body frames.</summary>
        internal double[] geom_quat = [];
        /// <summary>Geom friction (3 per geom).</summary>
        internal double[] geom_friction = [];
        /// <summary>Geom solver mixing weights.</summary>
        internal double[] geom_solmix = [];
        /// <summary>Geom contact solref (2 per geom).</summary>
        internal double[] geom_solref = [];
        /// <summary>Geom contact solimp (5 per geom).</summary>
        internal double[] geom_solimp = [];
        /// <summary>Geom contact margins.</summary>
        internal double[] geom_margin = [];
        /// <summary>Geom contact gaps.</summary>
        internal double[] geom_gap = [];

        // actuators
        /// <summary>Transmission target joint of each actuator (2 per actuator, the second unused).</summary>
        internal int[] actuator_trnid = [];
        /// <summary>Whether each control is clamped to its range.</summary>
        internal bool[] actuator_ctrllimited = [];
        /// <summary>Control ranges (2 per actuator).</summary>
        internal double[] actuator_ctrlrange = [];
        /// <summary>Transmission gears (6 per actuator; joints use the first).</summary>
        internal double[] actuator_gear = [];
        /// <summary>Gain parameters (10 per actuator; a motor uses the first).</summary>
        internal double[] actuator_gainprm = [];

        // sites (frames fixed in bodies; empty when the model has none)
        /// <summary>Body of each site.</summary>
        internal int[] site_bodyid = [];
        /// <summary>Site positions in the body frames.</summary>
        internal double[] site_pos = [];
        /// <summary>Site orientations in the body frames.</summary>
        internal double[] site_quat = [];
        /// <summary>How each site's frame relates to its body's frames.</summary>
        internal int[] site_sameframe = [];

        /// <summary>
        ///     Body names, for env rules that read a body by name (Reacher's fingertip, Pusher's object); empty when the dump
        ///     predates them.
        /// </summary>
        internal string[] body_names = [];

        /// <summary>
        ///     Element names by MuJoCo's kind abbreviation (<c>body</c>, <c>jnt</c>, <c>geom</c>, <c>site</c>, <c>cam</c>,
        ///     <c>light</c>, <c>actuator</c>, <c>tendon</c>, <c>mat</c>, <c>tex</c>), each in index order with <c>""</c> for
        ///     an unnamed element; only <c>body</c> from a trace dump, which records no other names.
        /// </summary>
        internal Dictionary<string, string[]> names = new();

        // counts and data only Gymnasium's code and the renderer read (absent from trace dumps, which leave them 0 or empty)
        /// <summary>Number of activation states (<c>na</c>): 0 in every model the core loads (no stateful actuators).</summary>
        internal int na;
        /// <summary>Number of bounding-volume nodes of the collision midphase (<c>nbvh</c>), which Gymnasium's tests read.</summary>
        internal int nbvh;
        /// <summary>Number of cameras.</summary>
        internal int ncam;
        /// <summary>The model's spatial extent (<c>stat.extent</c>), the free camera's default distance.</summary>
        internal double extent;
        /// <summary>The offscreen buffer's width (<c>vis.global_.offwidth</c>); Gymnasium's envs grow it to the render width.</summary>
        internal int offwidth;
        /// <summary>The offscreen buffer's height (<c>vis.global_.offheight</c>).</summary>
        internal int offheight;
        /// <summary>Geom colours, 4 per geom (red, green, blue, alpha in [0, 1]), float32 as MuJoCo stores them.</summary>
        internal float[] geom_rgba = [];

        // reference configuration
        /// <summary>The reference configuration the joint values are measured from.</summary>
        internal double[] qpos0 = [];
        /// <summary>The spring rest configuration.</summary>
        internal double[] qpos_spring = [];

        // sparse inertia structure
        /// <summary>Nonzeros per row of the lower-triangular inertia storage (the dof's ancestor chain).</summary>
        internal int[] M_rownnz = [];
        /// <summary>Row start addresses of the inertia storage.</summary>
        internal int[] M_rowadr = [];
        /// <summary>Column indices of the inertia storage, diagonal last in each row.</summary>
        internal int[] M_colind = [];

        // options
        /// <summary>The time step.</summary>
        internal double timestep;
        /// <summary>Gravity.</summary>
        internal double[] gravity = new double[3];
        /// <summary>The solver's termination tolerance.</summary>
        internal double tolerance;
        /// <summary>The line search tolerance, relative to <see cref="tolerance"/>.</summary>
        internal double ls_tolerance;
        /// <summary>The ratio of frictional to normal constraint impedance.</summary>
        internal double impratio;
        /// <summary>Maximum solver iterations.</summary>
        internal int iterations;
        /// <summary>Maximum line search iterations.</summary>
        internal int ls_iterations;
        /// <summary>The integrator: 0 Euler (HalfCheetah), 1 RK4 (Hopper, Humanoid).</summary>
        internal int integrator;
        /// <summary>The constraint solver: 2 Newton (HalfCheetah, Hopper), 0 PGS (Humanoid).</summary>
        internal int solver;
        /// <summary>The model's mean inertia, which scales PGS's improvement test (<c>stat.meaninertia</c>).</summary>
        internal double meaninertia;
        /// <summary>The fluid's density (Swimmer's 4000; 0 disables the drag and lift terms).</summary>
        internal double density;
        /// <summary>The fluid's viscosity (Swimmer's 0.1; 0 disables the viscous terms).</summary>
        internal double viscosity;
        /// <summary>The fluid's velocity in the world frame (3 values; 0 in every Gymnasium model, still transformed).</summary>
        internal double[] wind = new double[3];
        /// <summary>The convex collider's iteration budget for GJK and EPA (<c>opt.ccd_iterations</c>).</summary>
        internal int ccd_iterations;
        /// <summary>The convex collider's tolerance (<c>opt.ccd_tolerance</c>).</summary>
        internal double ccd_tolerance;

        // fixed tendons (Humanoid's two; inert in the dynamics, reported in the env's info)
        /// <summary>The number of tendons.</summary>
        internal int ntendon;
        /// <summary>First wrap object of each tendon.</summary>
        internal int[] tendon_adr = [];
        /// <summary>Wrap objects per tendon.</summary>
        internal int[] tendon_num = [];
        /// <summary>The type of each wrap object (1: a joint of a fixed tendon).</summary>
        internal int[] wrap_type = [];
        /// <summary>The joint of each wrap object.</summary>
        internal int[] wrap_objid = [];
        /// <summary>The coefficient of each wrap object.</summary>
        internal double[] wrap_prm = [];
        /// <summary>Nonzeros per tendon Jacobian row.</summary>
        internal int[] ten_J_rownnz = [];
        /// <summary>Tendon Jacobian row addresses.</summary>
        internal int[] ten_J_rowadr = [];
        /// <summary>Tendon Jacobian column indices (dofs).</summary>
        internal int[] ten_J_colind = [];

        /// <summary>
        ///     Reads a model file (an embedded model of DR-63, or a trace folder's model.json) and checks that the model
        ///     stays inside what the core reproduces.
        /// </summary>
        /// <param name="path">The JSON file.</param>
        /// <returns>The model.</returns>
        /// <exception cref="IOException">The file cannot be read.</exception>
        /// <exception cref="JsonException">The file is not the expected JSON.</exception>
        /// <exception cref="KeyNotFoundException">A field the core needs is missing from the file.</exception>
        /// <exception cref="NotSupportedException">The model uses a feature the core does not reproduce; the message names it.</exception>
        internal static MjModel Load(string path) => Load(File.ReadAllBytes(path));

        /// <summary>
        ///     Reads a model from its JSON bytes and checks that it stays inside what the core reproduces. Two layouts are
        ///     read: the embedded models' (<c>tools/parity/mujoco/record_models.py</c>: the arrays under <c>arrays</c>,
        ///     options as plain numbers, names per kind, plus the counts and data only Gymnasium's code and the renderer
        ///     read), and oracle.py's trace dumps (the arrays at the root, every option a one-element list, body names
        ///     only), so the trace checker verifies the same loader the envs use.
        /// </summary>
        /// <param name="json">The UTF-8 JSON.</param>
        /// <returns>The model.</returns>
        /// <exception cref="JsonException">The bytes are not the expected JSON.</exception>
        /// <exception cref="KeyNotFoundException">A field the core needs is missing.</exception>
        /// <exception cref="InvalidOperationException">A field has the wrong JSON kind (a list where a number is expected).</exception>
        /// <exception cref="NotSupportedException">The model uses a feature the core does not reproduce; the message names it.</exception>
        internal static MjModel Load(byte[] json) {
            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;
            // the embedded layout nests the arrays; a trace dump keeps them at the root
            JsonElement r = root.TryGetProperty("arrays", out JsonElement nested) ? nested : root;
            double[] D(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
            int[] I(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.ValueKind == JsonValueKind.True ? 1 : e.ValueKind == JsonValueKind.False ? 0 : e.GetInt32()).ToArray();
            bool[] B(string name) => I(name).Select(v => v != 0).ToArray();
            JsonElement sizes = root.GetProperty("sizes"), opt = root.GetProperty("opt");
            // a trace dump wraps every option in a list (scalars as one element); the embedded layout writes the number
            double Opt(string name) {
                JsonElement e = opt.GetProperty(name);
                return e.ValueKind == JsonValueKind.Array ? e[0].GetDouble() : e.GetDouble();
            }

            var m = new MjModel {
                nq = sizes.GetProperty("nq").GetInt32(), nv = sizes.GetProperty("nv").GetInt32(),
                nu = sizes.GetProperty("nu").GetInt32(), nbody = sizes.GetProperty("nbody").GetInt32(),
                njnt = sizes.GetProperty("njnt").GetInt32(), ngeom = sizes.GetProperty("ngeom").GetInt32(),
                nC = sizes.GetProperty("nC").GetInt32(),
                body_parentid = I("body_parentid"), body_rootid = I("body_rootid"), body_weldid = I("body_weldid"),
                body_jntnum = I("body_jntnum"), body_jntadr = I("body_jntadr"), body_dofnum = I("body_dofnum"),
                body_dofadr = I("body_dofadr"), body_geomadr = I("body_geomadr"), body_geomnum = I("body_geomnum"),
                body_sameframe = I("body_sameframe"), body_contype = I("body_contype"), body_conaffinity = I("body_conaffinity"),
                body_pos = D("body_pos"), body_quat = D("body_quat"), body_ipos = D("body_ipos"), body_iquat = D("body_iquat"),
                body_mass = D("body_mass"), body_subtreemass = D("body_subtreemass"), body_inertia = D("body_inertia"),
                body_invweight0 = D("body_invweight0"),
                jnt_type = I("jnt_type"), jnt_qposadr = I("jnt_qposadr"), jnt_dofadr = I("jnt_dofadr"), jnt_limited = B("jnt_limited"),
                jnt_pos = D("jnt_pos"), jnt_axis = D("jnt_axis"), jnt_stiffness = D("jnt_stiffness"),
                jnt_stiffnesspoly = D("jnt_stiffnesspoly"), jnt_range = D("jnt_range"), jnt_margin = D("jnt_margin"),
                jnt_solref = D("jnt_solref"), jnt_solimp = D("jnt_solimp"),
                dof_bodyid = I("dof_bodyid"), dof_jntid = I("dof_jntid"), dof_parentid = I("dof_parentid"),
                dof_damping = D("dof_damping"), dof_dampingpoly = D("dof_dampingpoly"), dof_armature = D("dof_armature"),
                dof_invweight0 = D("dof_invweight0"), dof_simplenum = I("dof_simplenum"), dof_M0 = D("dof_M0"),
                dof_treeid = I("dof_treeid"), body_treeid = I("body_treeid"), tree_dofnum = I("tree_dofnum"),
                geom_type = I("geom_type"), geom_contype = I("geom_contype"), geom_conaffinity = I("geom_conaffinity"),
                geom_condim = I("geom_condim"), geom_bodyid = I("geom_bodyid"), geom_sameframe = I("geom_sameframe"),
                geom_priority = I("geom_priority"), geom_size = D("geom_size"), geom_rbound = D("geom_rbound"),
                geom_pos = D("geom_pos"), geom_quat = D("geom_quat"), geom_friction = D("geom_friction"),
                geom_solmix = D("geom_solmix"), geom_solref = D("geom_solref"), geom_solimp = D("geom_solimp"),
                geom_margin = D("geom_margin"), geom_gap = D("geom_gap"),
                actuator_trnid = I("actuator_trnid"), actuator_ctrllimited = B("actuator_ctrllimited"),
                actuator_ctrlrange = D("actuator_ctrlrange"), actuator_gear = D("actuator_gear"),
                actuator_gainprm = D("actuator_gainprm"),
                qpos0 = D("qpos0"), qpos_spring = D("qpos_spring"),
                M_rownnz = I("M_rownnz"), M_rowadr = I("M_rowadr"), M_colind = I("M_colind"),
                timestep = Opt("timestep"), gravity = opt.GetProperty("gravity").EnumerateArray().Select(e => e.GetDouble()).ToArray(),
                tolerance = Opt("tolerance"), ls_tolerance = Opt("ls_tolerance"), impratio = Opt("impratio"),
                iterations = (int)Opt("iterations"), ls_iterations = (int)Opt("ls_iterations"),
                integrator = (int)Opt("integrator"), solver = (int)Opt("solver"),
                meaninertia = root.GetProperty("stat").GetProperty("meaninertia").GetDouble(),
                density = Opt("density"), viscosity = Opt("viscosity"),
                wind = opt.GetProperty("wind").EnumerateArray().Select(e => e.GetDouble()).ToArray(),
                ccd_iterations = (int)Opt("ccd_iterations"), ccd_tolerance = Opt("ccd_tolerance"),
            };
            m.ntree = m.tree_dofnum.Length;
            // the tendon arrays are read only when the model has tendons, so a dump without them (HalfCheetah's) still loads
            m.ntendon = sizes.TryGetProperty("ntendon", out JsonElement nten) ? nten.GetInt32() : 0;
            if (m.ntendon > 0) {
                m.tendon_adr = I("tendon_adr");
                m.tendon_num = I("tendon_num");
                m.wrap_type = I("wrap_type");
                m.wrap_objid = I("wrap_objid");
                m.wrap_prm = D("wrap_prm");
                m.ten_J_rownnz = I("ten_J_rownnz");
                m.ten_J_rowadr = I("ten_J_rowadr");
                m.ten_J_colind = I("ten_J_colind");
            }
            // sites and names, both absent from dumps older than the remaining envs' (the play project's HalfCheetah has none)
            m.nsite = sizes.TryGetProperty("nsite", out JsonElement nsite) ? nsite.GetInt32() : 0;
            if (m.nsite > 0) {
                m.site_bodyid = I("site_bodyid");
                m.site_pos = D("site_pos");
                m.site_quat = D("site_quat");
                m.site_sameframe = I("site_sameframe");
            }
            // names: per kind in the embedded layout, bodies only in a trace dump
            if (root.TryGetProperty("names", out JsonElement kinds)) {
                foreach (JsonProperty kind in kinds.EnumerateObject())
                    m.names[kind.Name] = kind.Value.EnumerateArray().Select(e => e.GetString() ?? "").ToArray();
                m.body_names = m.names.TryGetValue("body", out string[]? bodies) ? bodies : [];
            } else if (r.TryGetProperty("body_names", out JsonElement bodyNames)) {
                m.body_names = bodyNames.EnumerateArray().Select(e => e.GetString() ?? "").ToArray();
                m.names["body"] = m.body_names;
            }
            // what only Gymnasium's code and the renderer read: present in the embedded layout, left at 0 by a trace dump
            int Size(string name) => sizes.TryGetProperty(name, out JsonElement e) ? e.GetInt32() : 0;
            m.na = Size("na");
            m.nbvh = Size("nbvh");
            m.ncam = Size("ncam");
            if (root.GetProperty("stat").TryGetProperty("extent", out JsonElement extent))
                m.extent = extent.GetDouble();
            if (root.TryGetProperty("vis", out JsonElement vis) && vis.TryGetProperty("global_", out JsonElement global)) {
                m.offwidth = global.GetProperty("offwidth").GetInt32();
                m.offheight = global.GetProperty("offheight").GetInt32();
            }
            // the recorded colours are float32 values written as doubles, so the cast back is exact
            if (r.TryGetProperty("geom_rgba", out _))
                m.geom_rgba = D("geom_rgba").Select(v => (float)v).ToArray();
            // the renderer's fields (DR-62): cameras, lights, materials, textures; empty for a trace dump
            m.LoadVisual(root, r, sizes);
            m.Validate(r, sizes, opt);
            return m;
        }

        /// <summary>
        ///     Refuses every feature the core does not reproduce, so that a loaded model is stepped exactly or not at all.
        /// </summary>
        /// <param name="r">The JSON element holding the arrays, for the fields only the checks read.</param>
        /// <param name="sizes">The model sizes.</param>
        /// <param name="opt">The model options (each a number, or a one-element list in a trace dump).</param>
        /// <exception cref="NotSupportedException">The first unsupported feature found.</exception>
        private void Validate(JsonElement r, JsonElement sizes, JsonElement opt) {
            void Require(bool ok, string what) {
                if (!ok)
                    throw new NotSupportedException("the core: " + what);
            }
            int Size(string name) => sizes.TryGetProperty(name, out JsonElement e) ? e.GetInt32() : 0;
            double Opt(string name) {
                JsonElement e = opt.GetProperty(name);
                return e.ValueKind == JsonValueKind.Array ? e[0].GetDouble() : e.GetDouble();
            }
            int[] I(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.ValueKind == JsonValueKind.True ? 1 : e.ValueKind == JsonValueKind.False ? 0 : e.GetInt32()).ToArray();
            double[] D(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();

            foreach (string s in new[] { "nexclude", "npair", "neq", "nflex", "nplugin", "na", "nmocap", "nsensor" })
                Require(Size(s) == 0, $"{s} must be 0");
            Require(nv < 60, "the dense solver path (jacobian=auto with nv < 60)");
            Require(integrator == 0 || integrator == 1, "the Euler or RK4 integrator");
            Require(Opt("cone") == 0, "pyramidal friction cone only");
            Require(solver == 2 || solver == 0, "the Newton or PGS solver");
            // PGS runs per island in MuJoCo; with one tree its single island holds every row and dof, which the core's
            // monolithic sweeps reproduce (Newton has the island-local solve)
            Require(solver != 0 || ntree == 1, "PGS on a single kinematic tree");
            Require(Opt("jacobian") == 2 || Opt("jacobian") == 0, "dense Jacobian only");
            Require(Opt("disableflags") == 0 && Opt("enableflags") == 0, "default flags only");
            Require(Opt("noslip_iterations") == 0, "no noslip");
            // fluid forces: the inertia-box model only (a geom's first fluid parameter selects the ellipsoid model instead)
            if (density != 0 || viscosity != 0)
                Require(D("geom_fluid").Where((_, k) => k % 12 == 0).All(v => v == 0), "fluid forces of the inertia-box model only");
            // joints: slides and hinges, and free joints as the only joint of a body, without a spring (the core's springs are
            // the scalar ones of slides and hinges)
            for (int j = 0; j < njnt; j++) {
                Require(jnt_type[j] == 0 || jnt_type[j] == 2 || jnt_type[j] == 3, $"joint {j}: free, hinge or slide only");
                if (jnt_type[j] == 0) {
                    int b = Array.IndexOf(body_jntadr, j);
                    Require(b > 0 && body_jntnum[b] == 1, $"joint {j}: a free joint must be its body's only joint");
                    Require(jnt_stiffness[j] == 0 && jnt_stiffnesspoly[2 * j] == 0 && jnt_stiffnesspoly[2 * j + 1] == 0,
                            $"joint {j}: no spring on a free joint");
                    Require(!jnt_limited[j], $"joint {j}: no limit on a free joint");
                }
            }
            for (int b = 0; b < nbody; b++) {
                int dofs = 0;
                for (int j = body_jntadr[b]; j < body_jntadr[b] + body_jntnum[b]; j++)
                    dofs += jnt_type[j] == 0 ? 6 : 1;
                Require(dofs == body_dofnum[b], $"body {b}: its dofs are its joints'");
            }
            Require(D("dof_frictionloss").All(v => v == 0), "no dof frictionloss");
            Require(I("jnt_actuatorid").All(v => v == -1), "no actuator-inherited damping or armature");
            Require(D("geom_adhesion").All(v => v == 0), "no adhesion");
            Require(I("body_mocapid").All(v => v == -1), "no mocap bodies");
            // geoms: those that can collide are planes on the world body, or spheres, capsules, cylinders and boxes on bodies,
            // whose every pair has a collider here (a box's only against a plane, point.xml's arrow, which refuses at the step
            // where it would touch). A geom no geom of another body can touch (no compatible bitmask) never reaches a
            // collider, so it may be of any type and on any body, the world included (InvertedPendulum's rail, Reacher's
            // arena); the core still computes its frame.
            for (int g = 0; g < ngeom; g++) {
                int gg = g;
                bool collides = Enumerable.Range(0, ngeom).Any(h => geom_bodyid[h] != geom_bodyid[gg]
                    && ((geom_contype[gg] & geom_conaffinity[h]) != 0 || (geom_contype[h] & geom_conaffinity[gg]) != 0));
                if (!collides)
                    continue;
                bool plane = geom_type[g] == 0 && geom_bodyid[g] == 0;
                Require(plane || ((geom_type[g] == 2 || geom_type[g] == 3 || geom_type[g] == 5 || geom_type[g] == 6) && geom_bodyid[g] > 0),
                        $"geom {g}: a plane on the world body, or a sphere, capsule, cylinder or box on a body");
                Require(geom_condim[g] == 1 || geom_condim[g] == 3, $"geom {g}: condim 1 or 3");
            }
            // every pair of geoms that may touch needs a collider the core reproduces: plane against sphere, capsule,
            // cylinder or box (the box's test only); sphere-sphere, sphere-capsule, capsule-capsule; capsule-cylinder through
            // the convex collider (a superset of the real candidates, which the body filters thin further)
            for (int g = 0; g < ngeom; g++)
                for (int h = g + 1; h < ngeom; h++) {
                    if (geom_bodyid[g] == geom_bodyid[h]
                        || ((geom_contype[g] & geom_conaffinity[h]) == 0 && (geom_contype[h] & geom_conaffinity[g]) == 0))
                        continue;
                    int lo = Math.Min(geom_type[g], geom_type[h]), hi = Math.Max(geom_type[g], geom_type[h]);
                    bool ok = (lo == 0 && (hi == 2 || hi == 3 || hi == 5 || hi == 6)) || (lo == 2 && (hi == 2 || hi == 3)) || (lo == 3 && (hi == 3 || hi == 5));
                    Require(ok, $"geoms {g} and {h}: a collider for geom types {lo} and {hi}");
                    // a capsule-cylinder pair needs no margin on MuJoCo 4: its convex collider clips the capsule's edge
                    // against the cylinder's disk with or without one (Ccd.Multicontact), where MuJoCo 3 perturbed instead
                }
            // tendons: fixed, and inert in the dynamics (no limit, spring, damper, armature, friction loss or actuator), so the
            // core only reports their lengths and velocities
            if (ntendon > 0) {
                Require(wrap_type.All(t => t == 1), "fixed tendons only (joint wraps)");
                Require(I("tendon_limited").All(v => v == 0), "no tendon limits");
                Require(D("tendon_stiffness").All(v => v == 0) && D("tendon_stiffnesspoly").All(v => v == 0), "no tendon springs");
                Require(D("tendon_damping").All(v => v == 0) && D("tendon_dampingpoly").All(v => v == 0), "no tendon dampers");
                Require(D("tendon_armature").All(v => v == 0), "no tendon armature");
                Require(D("tendon_frictionloss").All(v => v == 0), "no tendon friction loss");
                if (r.TryGetProperty("tendon_actuatorid", out _))
                    Require(I("tendon_actuatorid").All(v => v == -1), "no actuator on a tendon");
            }
            int[] trntype = I("actuator_trntype"), dyntype = I("actuator_dyntype"), gaintype = I("actuator_gaintype"),
                  biastype = I("actuator_biastype");
            bool[] forcelimited = I("actuator_forcelimited").Select(v => v != 0).ToArray();
            for (int i = 0; i < nu; i++) {
                Require(trntype[i] == 0 && dyntype[i] == 0 && gaintype[i] == 0 && biastype[i] == 0,
                        $"actuator {i}: a motor (joint transmission, no dynamics, fixed gain, no bias)");
                Require(!forcelimited[i], $"actuator {i}: no force limit");
                Require(jnt_type[actuator_trnid[2 * i]] != 0, $"actuator {i}: on a hinge or slide joint");
            }
            Require(I("jnt_actfrclimited").All(v => v == 0), "no joint actuator force limits");
        }
    }
}
