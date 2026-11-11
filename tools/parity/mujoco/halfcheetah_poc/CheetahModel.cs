using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace MujocoPoc;

/// <summary>
///     A compiled MuJoCo model reduced to what a planar walker needs: a kinematic tree of hinge and slide joints, a ground
///     plane and capsules, motors on joints, Euler integration, pyramidal contacts and the dense Newton solver. Read from
///     the JSON dump of a model MuJoCo compiled (oracle.py's model.json), so the numbers are MuJoCo's own; no MJCF compiler.
/// </summary>
/// <remarks>
///     <see cref="Load"/> refuses any model that uses a feature the core does not reproduce (free or ball joints, tendons,
///     equality constraints, frictionloss, other geom types, contact pairs between bodies, stateful actuators, other
///     integrators or solvers), so a model that loads is one the core steps exactly. Arrays keep MuJoCo's names and
///     layouts (row-major, <c>3*i</c>, <c>4*i</c>, <c>9*i</c> strides).
/// </remarks>
public sealed class CheetahModel {
    // sizes
    /// <summary>Number of position coordinates (equals <see cref="nv"/>: hinge and slide joints only).</summary>
    public int nq;
    /// <summary>Number of degrees of freedom.</summary>
    public int nv;
    /// <summary>Number of actuators (and controls).</summary>
    public int nu;
    /// <summary>Number of bodies, the world included.</summary>
    public int nbody;
    /// <summary>Number of joints.</summary>
    public int njnt;
    /// <summary>Number of geoms.</summary>
    public int ngeom;
    /// <summary>Number of stored entries of the sparse inertia matrix.</summary>
    public int nC;

    // bodies
    /// <summary>Parent body of each body.</summary>
    public int[] body_parentid = [];
    /// <summary>Root body of each body's tree (the subtree whose center of mass anchors the motion dofs).</summary>
    public int[] body_rootid = [];
    /// <summary>The body each body is welded to (itself for a body with joints).</summary>
    public int[] body_weldid = [];
    /// <summary>Joints per body.</summary>
    public int[] body_jntnum = [];
    /// <summary>First joint of each body.</summary>
    public int[] body_jntadr = [];
    /// <summary>Dofs per body.</summary>
    public int[] body_dofnum = [];
    /// <summary>First dof of each body.</summary>
    public int[] body_dofadr = [];
    /// <summary>First geom of each body.</summary>
    public int[] body_geomadr = [];
    /// <summary>Geoms per body.</summary>
    public int[] body_geomnum = [];
    /// <summary>How each body's inertial frame relates to its frame (MuJoCo's <c>mjtSameFrame</c>).</summary>
    public int[] body_sameframe = [];
    /// <summary>Body-level contype (the OR of its geoms').</summary>
    public int[] body_contype = [];
    /// <summary>Body-level conaffinity (the OR of its geoms').</summary>
    public int[] body_conaffinity = [];
    /// <summary>Body positions relative to their parents.</summary>
    public double[] body_pos = [];
    /// <summary>Body orientations relative to their parents.</summary>
    public double[] body_quat = [];
    /// <summary>Inertial frame positions in the body frames.</summary>
    public double[] body_ipos = [];
    /// <summary>Inertial frame orientations in the body frames.</summary>
    public double[] body_iquat = [];
    /// <summary>Body masses.</summary>
    public double[] body_mass = [];
    /// <summary>Masses of the subtrees rooted at each body.</summary>
    public double[] body_subtreemass = [];
    /// <summary>Principal inertias.</summary>
    public double[] body_inertia = [];
    /// <summary>Inverse translational and rotational weights at the reference pose (2 per body).</summary>
    public double[] body_invweight0 = [];

    // joints
    /// <summary>Joint types (2 slide, 3 hinge).</summary>
    public int[] jnt_type = [];
    /// <summary>Address of each joint in qpos.</summary>
    public int[] jnt_qposadr = [];
    /// <summary>Address of each joint in qvel.</summary>
    public int[] jnt_dofadr = [];
    /// <summary>Whether each joint has a range limit.</summary>
    public bool[] jnt_limited = [];
    /// <summary>Joint anchors in the body frames.</summary>
    public double[] jnt_pos = [];
    /// <summary>Joint axes in the body frames.</summary>
    public double[] jnt_axis = [];
    /// <summary>Joint stiffness (linear spring coefficient).</summary>
    public double[] jnt_stiffness = [];
    /// <summary>Joint stiffness polynomial coefficients (2 per joint).</summary>
    public double[] jnt_stiffnesspoly = [];
    /// <summary>Joint ranges (2 per joint).</summary>
    public double[] jnt_range = [];
    /// <summary>Joint limit margins.</summary>
    public double[] jnt_margin = [];
    /// <summary>Joint limit solref (2 per joint).</summary>
    public double[] jnt_solref = [];
    /// <summary>Joint limit solimp (5 per joint).</summary>
    public double[] jnt_solimp = [];

    // dofs
    /// <summary>Body of each dof.</summary>
    public int[] dof_bodyid = [];
    /// <summary>Joint of each dof.</summary>
    public int[] dof_jntid = [];
    /// <summary>Parent dof of each dof (-1 at a root).</summary>
    public int[] dof_parentid = [];
    /// <summary>Dof damping.</summary>
    public double[] dof_damping = [];
    /// <summary>Dof damping polynomial coefficients (2 per dof).</summary>
    public double[] dof_dampingpoly = [];
    /// <summary>Dof armature inertia.</summary>
    public double[] dof_armature = [];
    /// <summary>Inverse dof weights at the reference pose.</summary>
    public double[] dof_invweight0 = [];

    // geoms
    /// <summary>Geom types (0 plane, 3 capsule).</summary>
    public int[] geom_type = [];
    /// <summary>Geom contype bitmasks.</summary>
    public int[] geom_contype = [];
    /// <summary>Geom conaffinity bitmasks.</summary>
    public int[] geom_conaffinity = [];
    /// <summary>Geom contact dimensionality.</summary>
    public int[] geom_condim = [];
    /// <summary>Body of each geom.</summary>
    public int[] geom_bodyid = [];
    /// <summary>How each geom's frame relates to its body's frames.</summary>
    public int[] geom_sameframe = [];
    /// <summary>Geom contact priorities.</summary>
    public int[] geom_priority = [];
    /// <summary>Geom sizes (3 per geom; capsule: radius, half-length).</summary>
    public double[] geom_size = [];
    /// <summary>Bounding sphere radii (0 for a plane).</summary>
    public double[] geom_rbound = [];
    /// <summary>Geom positions in the body frames.</summary>
    public double[] geom_pos = [];
    /// <summary>Geom orientations in the body frames.</summary>
    public double[] geom_quat = [];
    /// <summary>Geom friction (3 per geom).</summary>
    public double[] geom_friction = [];
    /// <summary>Geom solver mixing weights.</summary>
    public double[] geom_solmix = [];
    /// <summary>Geom contact solref (2 per geom).</summary>
    public double[] geom_solref = [];
    /// <summary>Geom contact solimp (5 per geom).</summary>
    public double[] geom_solimp = [];
    /// <summary>Geom contact margins.</summary>
    public double[] geom_margin = [];
    /// <summary>Geom contact gaps.</summary>
    public double[] geom_gap = [];

    // actuators
    /// <summary>Transmission target joint of each actuator (2 per actuator, the second unused).</summary>
    public int[] actuator_trnid = [];
    /// <summary>Whether each control is clamped to its range.</summary>
    public bool[] actuator_ctrllimited = [];
    /// <summary>Control ranges (2 per actuator).</summary>
    public double[] actuator_ctrlrange = [];
    /// <summary>Transmission gears (6 per actuator; joints use the first).</summary>
    public double[] actuator_gear = [];
    /// <summary>Gain parameters (10 per actuator; a motor uses the first).</summary>
    public double[] actuator_gainprm = [];

    // reference configuration
    /// <summary>The reference configuration the joint values are measured from.</summary>
    public double[] qpos0 = [];
    /// <summary>The spring rest configuration.</summary>
    public double[] qpos_spring = [];

    // sparse inertia structure
    /// <summary>Nonzeros per row of the lower-triangular inertia storage (the dof's ancestor chain).</summary>
    public int[] M_rownnz = [];
    /// <summary>Row start addresses of the inertia storage.</summary>
    public int[] M_rowadr = [];
    /// <summary>Column indices of the inertia storage, diagonal last in each row.</summary>
    public int[] M_colind = [];

    // options
    /// <summary>The time step.</summary>
    public double timestep;
    /// <summary>Gravity.</summary>
    public double[] gravity = new double[3];
    /// <summary>The solver's termination tolerance.</summary>
    public double tolerance;
    /// <summary>The line search tolerance, relative to <see cref="tolerance"/>.</summary>
    public double ls_tolerance;
    /// <summary>The ratio of frictional to normal constraint impedance.</summary>
    public double impratio;
    /// <summary>Maximum solver iterations.</summary>
    public int iterations;
    /// <summary>Maximum line search iterations.</summary>
    public int ls_iterations;
    /// <summary>The number of physics steps per env step (Gymnasium's frame_skip).</summary>
    public int frame_skip;

    /// <summary>
    ///     Reads model.json and checks that the model stays inside what the core reproduces.
    /// </summary>
    /// <param name="path">The JSON dump written by oracle.py.</param>
    /// <returns>The model.</returns>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="JsonException">The file is not the expected JSON.</exception>
    /// <exception cref="KeyNotFoundException">A field the core needs is missing from the dump.</exception>
    /// <exception cref="NotSupportedException">The model uses a feature the core does not reproduce; the message names it.</exception>
    public static CheetahModel Load(string path) {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllBytes(path));
        JsonElement r = doc.RootElement;
        double[] D(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();
        int[] I(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.ValueKind == JsonValueKind.True ? 1 : e.ValueKind == JsonValueKind.False ? 0 : e.GetInt32()).ToArray();
        bool[] B(string name) => I(name).Select(v => v != 0).ToArray();
        JsonElement sizes = r.GetProperty("sizes"), opt = r.GetProperty("opt");
        double Opt(string name) => opt.GetProperty(name)[0].GetDouble();

        var m = new CheetahModel {
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
            dof_invweight0 = D("dof_invweight0"),
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
            frame_skip = r.GetProperty("frame_skip").GetInt32(),
        };
        m.Validate(r, sizes, opt);
        return m;
    }

    /// <summary>
    ///     Refuses every feature the core does not reproduce, so that a loaded model is stepped exactly or not at all.
    /// </summary>
    /// <param name="r">The JSON root, for the fields only the checks read.</param>
    /// <param name="sizes">The model sizes.</param>
    /// <param name="opt">The model options.</param>
    /// <exception cref="NotSupportedException">The first unsupported feature found.</exception>
    private void Validate(JsonElement r, JsonElement sizes, JsonElement opt) {
        void Require(bool ok, string what) {
            if (!ok)
                throw new NotSupportedException("HalfCheetah core: " + what);
        }
        int Size(string name) => sizes.TryGetProperty(name, out JsonElement e) ? e.GetInt32() : 0;
        double Opt(string name) => opt.GetProperty(name)[0].GetDouble();
        int[] I(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.ValueKind == JsonValueKind.True ? 1 : e.ValueKind == JsonValueKind.False ? 0 : e.GetInt32()).ToArray();
        double[] D(string name) => r.GetProperty(name).EnumerateArray().Select(e => e.GetDouble()).ToArray();

        foreach (string s in new[] { "nexclude", "npair", "neq", "ntendon", "nflex", "nplugin", "na", "nmocap", "nsensor" })
            Require(Size(s) == 0, $"{s} must be 0");
        Require(nq == nv, "only hinge and slide joints (nq == nv)");
        Require(nv < 60, "the dense solver path (jacobian=auto with nv < 60)");
        Require(Opt("integrator") == 0, "Euler integrator only");
        Require(Opt("cone") == 0, "pyramidal friction cone only");
        Require(Opt("solver") == 2, "Newton solver only");
        Require(Opt("jacobian") == 2 || Opt("jacobian") == 0, "dense Jacobian only");
        Require(Opt("disableflags") == 0 && Opt("enableflags") == 0, "default flags only");
        Require(Opt("noslip_iterations") == 0, "no noslip");
        Require(Opt("density") == 0 && Opt("viscosity") == 0, "no fluid forces");
        for (int j = 0; j < njnt; j++)
            Require(jnt_type[j] == 2 || jnt_type[j] == 3, $"joint {j}: hinge or slide only");
        for (int b = 0; b < nbody; b++)
            Require(body_jntnum[b] == body_dofnum[b], $"body {b}: one dof per joint");
        Require(I("dof_simplenum").All(v => v == 0), "no simple dofs");
        Require(D("dof_frictionloss").All(v => v == 0), "no dof frictionloss");
        Require(I("jnt_actuatorid").All(v => v == -1), "no actuator-inherited damping or armature");
        Require(D("geom_adhesion").All(v => v == 0), "no adhesion");
        Require(I("body_mocapid").All(v => v == -1), "no mocap bodies");
        for (int g = 0; g < ngeom; g++) {
            bool plane = geom_type[g] == 0 && geom_bodyid[g] == 0;
            Require(plane || geom_type[g] == 3, $"geom {g}: a plane on the world body or a capsule");
            Require(geom_condim[g] == 3, $"geom {g}: condim 3");
        }
        Require(body_geomnum[0] == 1 && geom_type[body_geomadr[0]] == 0, "exactly one geom, a plane, on the world body");
        // Contacts between two moving bodies would need capsule-capsule collision, which the core leaves out: the body
        // bitmasks must already exclude every such pair (as HalfCheetah's conaffinity 0 does).
        for (int b1 = 1; b1 < nbody; b1++)
            for (int b2 = b1 + 1; b2 < nbody; b2++)
                Require((body_contype[b1] & body_conaffinity[b2]) == 0 && (body_contype[b2] & body_conaffinity[b1]) == 0,
                        $"bodies {b1} and {b2} could collide");
        int[] trntype = I("actuator_trntype"), dyntype = I("actuator_dyntype"), gaintype = I("actuator_gaintype"),
              biastype = I("actuator_biastype");
        bool[] forcelimited = I("actuator_forcelimited").Select(v => v != 0).ToArray();
        for (int i = 0; i < nu; i++) {
            Require(trntype[i] == 0 && dyntype[i] == 0 && gaintype[i] == 0 && biastype[i] == 0,
                    $"actuator {i}: a motor (joint transmission, no dynamics, fixed gain, no bias)");
            Require(!forcelimited[i], $"actuator {i}: no force limit");
        }
        Require(I("jnt_actfrclimited").All(v => v == 0), "no joint actuator force limits");
    }
}
