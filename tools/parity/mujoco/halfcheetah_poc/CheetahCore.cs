using System;

namespace MujocoPoc;

/// <summary>
///     A from-scratch physics core for Gymnasium's MuJoCo envs (all eleven, the 21 ids that run on MuJoCo 3), written to
///     step exactly as MuJoCo 3.15.0's <c>mj_step</c> does on win-x64: the same quantities in the same floating-point
///     order, so every array it publishes (state, contacts, constraint rows, solver output) equals MuJoCo's bit for bit.
/// </summary>
/// <remarks>
///     <para>
///         It is not a translation of MuJoCo's sources: it computes only what such models need (hinge and slide joints
///         under an optional free root in each kinematic tree, a ground plane with spheres, capsules and cylinders, the
///         native convex collider's GJK and EPA for a capsule against a cylinder with its perturbed extra contacts, joint
///         limits, pyramidal and frictionless contacts, motors, constraint islands with the dense Newton solve per island
///         or PGS on one tree, Euler with implicit joint damping or RK4, fixed tendons' lengths, sites, the inertia-box
///         fluid forces, the external contact forces of <c>mj_rnePostConstraint</c>) and drops MuJoCo's generality
///         (sparse paths, sleeping, flex, active tendons, sensors, callbacks, meshes and boxes, the convex collider's
///         polygon clipping). <see cref="CheetahModel.Load"/> refuses models outside that scope. The class keeps its
///         first name because HalfCheetah's play project compiles it as it is.
///     </para>
///     <para>
///         Arrays carry MuJoCo's names and layouts so they can be compared with a recorded <c>mjData</c>. Not thread-safe:
///         one core owns its arrays, and <see cref="Step"/> reuses scratch storage.
///     </para>
///     <para>
///         Verified by the checker in this folder against oracle.py's traces (docs/plans/MUJOCO_ASSESSMENT.md §11 and
///         §13), and searched for the branches random actions reach (§13.4). Branches no trace has executed are written
///         from MuJoCo's source but unproven. Some no model here can reach: geom priorities, direct (negative) or
///         mixed-sign solref, an impedance of power 1, a body of zero subtree mass, the undamped Euler step, a tilted or
///         lifted cylinder (Pusher's object only slides), a body welded to the world in the fluid, and union-find chains
///         of three trees. Others none of about 90 million searched physics steps reached: the auto-reset of a bad
///         state or bad controls, coincident sphere centers, two parallel capsules touching at both ends, a contact in the
///         gap zone, PGS's cost-rise revert, Newton's
///         fallbacks (a zero step, an exhausted or stuck line search, a lost rank in the factor update), the convex
///         collider's degenerate exits (EPA from a segment, a degenerate simplex, EPA's failure exits), and the kernels'
///         degenerate cases (tiny pivots, zero-length vectors and quaternions, a zero rotation angle).
///     </para>
///     <para>
///         Reference: MuJoCo 3.15.0's <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_forward.c">engine_forward.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_smooth.c">engine_core_smooth.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_util.c">engine_core_util.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_collision_driver.c">engine_collision_driver.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_collision_primitive.c">engine_collision_primitive.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_collision_convex.c">engine_collision_convex.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_constraint.c">engine_core_constraint.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_island.c">engine_island.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_solver.c">engine_solver.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_passive.c">engine_passive.c</see> and
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_support.c">engine_support.c</see>
///         (Apache-2.0); each member names the MuJoCo function whose arithmetic it reproduces. GJK and EPA are in
///         <see cref="Ccd"/>.
///     </para>
/// </remarks>
public sealed class CheetahCore {
    private const int Free = 0;
    private const int Slide = 2;
    private const int Hinge = 3;
    private const int Plane = 0;
    private const int Sphere = 2;
    private const int Capsule = 3;
    private const int Cylinder = 5;
    private const int SolverPgs = 0;
    private const int IntegratorRk4 = 1;
    /// <summary>The constraint type of a joint-limit row (MuJoCo's <c>mjCNSTR_LIMIT_JOINT</c>).</summary>
    public const int LimitJoint = 3;
    /// <summary>The constraint type of a frictionless contact row (MuJoCo's <c>mjCNSTR_CONTACT_FRICTIONLESS</c>).</summary>
    public const int ContactFrictionless = 5;
    /// <summary>The constraint type of a pyramidal contact row (MuJoCo's <c>mjCNSTR_CONTACT_PYRAMIDAL</c>).</summary>
    public const int ContactPyramidal = 6;
    /// <summary>A constraint row that exerts no force (<c>mjCNSTRSTATE_SATISFIED</c>).</summary>
    public const int Satisfied = 0;
    /// <summary>A constraint row in its quadratic zone (<c>mjCNSTRSTATE_QUADRATIC</c>).</summary>
    public const int Quadratic = 1;
    private const double MinMu = 1e-5;
    private const double MinImp = 0.0001;
    private const double MaxImp = 0.9999;
    private const double MaxVal = 1e10;

    private readonly CheetahModel m;
    private readonly int nv, nbody, nu;

    // ---- state (what a step reads and advances)
    /// <summary>Joint positions.</summary>
    public readonly double[] qpos;
    /// <summary>Joint velocities.</summary>
    public readonly double[] qvel;
    /// <summary>Controls, read by the next <see cref="Step"/>.</summary>
    public readonly double[] ctrl;
    /// <summary>The previous step's constrained acceleration, the solver's warm start.</summary>
    public readonly double[] qacc_warmstart;
    /// <summary>Simulation time.</summary>
    public double time;

    // ---- position stage
    /// <summary>Body frame positions.</summary>
    public readonly double[] xpos;
    /// <summary>Body frame orientations.</summary>
    public readonly double[] xquat;
    /// <summary>Body frame rotation matrices.</summary>
    public readonly double[] xmat;
    /// <summary>Body inertial frame positions.</summary>
    public readonly double[] xipos;
    /// <summary>Body inertial frame rotation matrices.</summary>
    public readonly double[] ximat;
    /// <summary>Joint anchors in the world frame.</summary>
    public readonly double[] xanchor;
    /// <summary>Joint axes in the world frame.</summary>
    public readonly double[] xaxis;
    /// <summary>Geom positions.</summary>
    public readonly double[] geom_xpos;
    /// <summary>Geom rotation matrices.</summary>
    public readonly double[] geom_xmat;
    /// <summary>Site positions (InvertedDoublePendulum's reward reads its tip's).</summary>
    public readonly double[] site_xpos;
    /// <summary>Site rotation matrices.</summary>
    public readonly double[] site_xmat;
    /// <summary>Subtree centers of mass.</summary>
    public readonly double[] subtree_com;
    /// <summary>Body inertias about the subtree centers of mass, in the world frame.</summary>
    public readonly double[] cinert;
    /// <summary>The motion axis of each dof about the subtree center of mass.</summary>
    public readonly double[] cdof;
    /// <summary>Composite rigid body inertias.</summary>
    public readonly double[] crb;
    /// <summary>The joint-space inertia matrix, in MuJoCo's sparse lower-triangular storage.</summary>
    public readonly double[] M;
    /// <summary>The <c>L'*D*L</c> factor of <see cref="M"/>.</summary>
    public readonly double[] qLD;
    /// <summary>The inverse diagonal of the factor.</summary>
    public readonly double[] qLDiagInv;
    /// <summary>Actuator moment arms (the joint gear).</summary>
    public readonly double[] actuator_moment;
    /// <summary>Actuator lengths (joint position times gear).</summary>
    public readonly double[] actuator_length;

    // ---- contacts
    /// <summary>The number of contacts, excluded ones included.</summary>
    public int ncon;
    /// <summary>Signed contact distances (negative: penetration).</summary>
    public readonly double[] con_dist;
    /// <summary>Contact points.</summary>
    public readonly double[] con_pos;
    /// <summary>Contact frames: normal, then the two tangents.</summary>
    public readonly double[] con_frame;
    /// <summary>Distances below which each contact is active.</summary>
    public readonly double[] con_includemargin;
    /// <summary>Contact friction, 5 per contact.</summary>
    public readonly double[] con_friction;
    /// <summary>Contact solref, 2 per contact.</summary>
    public readonly double[] con_solref;
    /// <summary>Contact solimp, 5 per contact.</summary>
    public readonly double[] con_solimp;
    /// <summary>The friction of each contact's regularized cone.</summary>
    public readonly double[] con_mu;
    /// <summary>First geom of each contact (the plane).</summary>
    public readonly int[] con_geom1;
    /// <summary>Second geom of each contact.</summary>
    public readonly int[] con_geom2;
    /// <summary>Contact dimensionality.</summary>
    public readonly int[] con_dim;
    /// <summary>Whether each contact is excluded from the constraints (non-zero) and why.</summary>
    public readonly int[] con_exclude;
    /// <summary>First constraint row of each contact, or -1.</summary>
    public readonly int[] con_efc_address;

    // ---- constraints
    /// <summary>The number of constraint rows.</summary>
    public int nefc;
    /// <summary>The constraint Jacobian, <see cref="nefc"/> rows of nv.</summary>
    public readonly double[] efc_J;
    /// <summary>Constraint types.</summary>
    public readonly int[] efc_type;
    /// <summary>The joint or contact of each row.</summary>
    public readonly int[] efc_id;
    /// <summary>Constraint positions (limit distance, contact distance).</summary>
    public readonly double[] efc_pos;
    /// <summary>Constraint margins.</summary>
    public readonly double[] efc_margin;
    /// <summary>Constraint friction loss (always 0 here).</summary>
    public readonly double[] efc_frictionloss;
    /// <summary>Approximate inverse inertia per row, then adjusted to the regularization.</summary>
    public readonly double[] efc_diagA;
    /// <summary>Stiffness, damping, impedance and its derivative, 4 per row.</summary>
    public readonly double[] efc_KBIP;
    /// <summary>Constraint weights, <c>1/R</c>.</summary>
    public readonly double[] efc_D;
    /// <summary>Constraint regularizers.</summary>
    public readonly double[] efc_R;
    /// <summary>Constraint velocities.</summary>
    public readonly double[] efc_vel;
    /// <summary>Reference accelerations.</summary>
    public readonly double[] efc_aref;
    /// <summary>The unconstrained residual <c>J*qacc_smooth - aref</c>.</summary>
    public readonly double[] efc_b;
    /// <summary>Constraint forces from the solver.</summary>
    public readonly double[] efc_force;
    /// <summary>
    ///     Constraint states as MuJoCo publishes them. Under Newton, from the warm start's evaluation at
    ///     <see cref="qacc_smooth"/>, since the island solver keeps its own states and does not copy them back; under PGS,
    ///     the solver's last sweep, which works on these states directly.
    /// </summary>
    public readonly int[] efc_state;
    /// <summary>
    ///     The dual matrix <c>J*M^-1*J' + R</c>, <see cref="nefc"/> by <see cref="nefc"/>, built for the PGS solver only
    ///     (empty otherwise).
    /// </summary>
    public readonly double[] efc_AR;
    /// <summary>
    ///     The number of constraint islands: groups of kinematic trees joined by constraint rows (<c>mj_island</c>), each
    ///     solved on its own; a tree without rows is in none.
    /// </summary>
    public int nisland;
    /// <summary>Solver iterations of the last solve (Newton iterations, or PGS sweeps).</summary>
    public int solver_niter;

    // ---- tendons and external forces
    /// <summary>Tendon lengths (fixed tendons: their coefficients times the joint positions).</summary>
    public readonly double[] ten_length;
    /// <summary>Tendon Jacobian values in the model's sparse layout (<c>ten_J_rowadr</c>, <c>ten_J_colind</c>).</summary>
    public readonly double[] ten_J;
    /// <summary>Tendon velocities.</summary>
    public readonly double[] ten_velocity;
    /// <summary>
    ///     External forces on each body about its subtree centre of mass (6 per body, torque then force), set by
    ///     <see cref="RnePostConstraint"/> only: like MuJoCo's, it is not part of <see cref="Step"/>.
    /// </summary>
    public readonly double[] cfrc_ext;

    // ---- velocity, actuation, acceleration
    /// <summary>Body spatial velocities about the subtree centers of mass.</summary>
    public readonly double[] cvel;
    /// <summary>Time derivatives of <see cref="cdof"/>.</summary>
    public readonly double[] cdof_dot;
    /// <summary>Joint spring forces.</summary>
    public readonly double[] qfrc_spring;
    /// <summary>Joint damping forces.</summary>
    public readonly double[] qfrc_damper;
    /// <summary>Fluid forces (Swimmer's inertia-box drag, lift and viscosity; zero without a fluid).</summary>
    public readonly double[] qfrc_fluid;
    /// <summary>Passive forces (springs plus dampers, then fluid forces).</summary>
    public readonly double[] qfrc_passive;
    /// <summary>Coriolis, centrifugal and gravity forces.</summary>
    public readonly double[] qfrc_bias;
    /// <summary>Actuator forces.</summary>
    public readonly double[] actuator_force;
    /// <summary>Actuator forces in joint space.</summary>
    public readonly double[] qfrc_actuator;
    /// <summary>All forces but the constraints.</summary>
    public readonly double[] qfrc_smooth;
    /// <summary>The unconstrained acceleration.</summary>
    public readonly double[] qacc_smooth;
    /// <summary>The constrained acceleration.</summary>
    public readonly double[] qacc;
    /// <summary>Constraint forces in joint space.</summary>
    public readonly double[] qfrc_constraint;
    /// <summary>The Euler step's <c>M + h*diag(damping)</c>, factorized.</summary>
    public readonly double[] qH;
    /// <summary>The inverse diagonal of <see cref="qH"/>'s factor.</summary>
    public readonly double[] qHDiagInv;

    // ---- scratch
    private readonly double[] cacc, cfrc, jac1, jac2, jacdif, jacrot, jar, da, Mda, qaccEuler;
    // the Newton solver's context (MuJoCo's mjPrimalContext of the one island)
    private readonly double[] Jaref, Jv, Ma, Mv, grad, Mgrad, search, quad, Hdiag, cholupd, L;
    private readonly int[] sState, oldState;
    private double scale, cost, quadGauss1, quadGauss2;
    private int lsIter;
    // the PGS solver: Y = J*M^-1/2 and its transpose, the inverse diagonal of AR, Nesterov's iterates, the sweep order
    private readonly double[] efcY, efcYT, sqrtInvD, arInv, forcePrev, forceMomentum, arForce;
    private readonly int[] blockStart;
    private Pcg32 pgsRng;
    // RK4: the stage states (qpos then qvel), the stage accelerations, the weighted increment
    private readonly double[] rkX, rkF, rkdX;
    // the collision candidates in MuJoCo's order: body pairs by (lower id, higher id), then their geom pairs as the
    // midphase visits them; geom-level filters that depend on the pose run at each step
    private readonly int[] candA, candB;
    // the fluid model's Jacobians and projected force (mj_applyFT's stack arrays)
    private readonly double[] fluidJacp, fluidJacr, fluidQforce;
    // the convex collider's two geoms, query status and polytope workspace (reused by every query)
    private readonly CcdObject ccdObj1 = new(), ccdObj2 = new();
    private readonly CcdStatus ccdStatus = new();
    private readonly CcdPolytope ccdPoly = new();
    // constraint islands (mj_island): the union-find over trees, each row's tree, and the maps that gather an island's
    // dofs and rows in model order (unconstrained dofs follow the island dofs in mapIdof2dof)
    private readonly int[] treeParent, treeIsland, efcTree, mapDof2idof, mapIdof2dof, mapIefc2efc;
    private readonly int[] islandNv, islandIdofadr, islandNefc, islandIefcadr, islandFill;
    private int nidof;
    // the island being solved (MuJoCo's island-local mjPrimalContext): its sizes and gathered copies
    private int snv, snefc;
    private readonly double[] iM, iQLD, iQLDiagInv, iJ, iD, iAref, iForce, iQacc, iQaccSmooth, iQfrcSmooth, iQfrcConstraint;
    private readonly int[] iMrownnz, iMrowadr, iMcolind;

    /// <summary>The RK4 tableau's A (3x3, row i for stage i+1), MuJoCo's <c>RK4_A</c>.</summary>
    private static readonly double[] Rk4A = [0.5, 0, 0, 0, 0.5, 0, 0, 0, 1];
    /// <summary>The RK4 tableau's weights, MuJoCo's <c>RK4_B</c>: 1/6, 1/3, 1/3, 1/6 as the compiler folds them.</summary>
    private static readonly double[] Rk4B = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

    /// <summary>A line-search point: step length, cost change, first and second derivative.</summary>
    private struct Pnt {
        /// <summary>
        ///     The step length along the search direction (<c>Alpha</c>), the cost change from the start (<c>Cost</c>), and
        ///     the cost's first (<c>D0</c>) and second (<c>D1</c>) derivatives with respect to the step length.
        /// </summary>
        public double Alpha, Cost, D0, D1;
    }

    /// <summary>Allocates the core's arrays for a model; the state starts at the reference configuration, at rest.</summary>
    /// <param name="model">The model, as returned by <see cref="CheetahModel.Load"/>.</param>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null.</exception>
    public CheetahCore(CheetahModel model) {
        m = model ?? throw new ArgumentNullException(nameof(model));
        nv = m.nv;
        nbody = m.nbody;
        nu = m.nu;
        int njnt = m.njnt, ngeom = m.ngeom;
        (candA, candB) = CollisionCandidates(m);
        // Capacities from the candidates: plane-cylinder makes four contacts at most, capsule-cylinder (the convex
        // collider) one plus four from MultiCCD's perturbations, a pair with a capsule two, others one; four pyramid rows
        // per condim-3 contact, one row per frictionless contact; two sides per limited joint.
        int maxcon = 0, maxefc = 0;
        for (int c = 0; c < candA.Length; c++) {
            int ga = candA[c], gb = candB[c];
            int ta = m.geom_type[ga], tb = m.geom_type[gb];
            int contacts = ta == Cylinder || tb == Cylinder ? (ta == Plane || tb == Plane ? 4 : 5)
                         : ta == Capsule || tb == Capsule ? 2 : 1;
            int dim = m.geom_priority[ga] != m.geom_priority[gb]
                ? m.geom_condim[m.geom_priority[ga] > m.geom_priority[gb] ? ga : gb]
                : Math.Max(m.geom_condim[ga], m.geom_condim[gb]);
            maxcon += contacts;
            maxefc += contacts * (dim == 1 ? 1 : 2 * (dim - 1));
        }
        for (int j = 0; j < njnt; j++)
            maxefc += m.jnt_limited[j] ? 2 : 0;
        maxcon = Math.Max(maxcon, 1);
        maxefc = Math.Max(maxefc, 1);
        qpos = (double[])m.qpos0.Clone();
        qvel = new double[nv];
        ctrl = new double[nu];
        qacc_warmstart = new double[nv];
        xpos = new double[3 * nbody];
        xquat = new double[4 * nbody];
        xmat = new double[9 * nbody];
        xipos = new double[3 * nbody];
        ximat = new double[9 * nbody];
        xanchor = new double[3 * njnt];
        xaxis = new double[3 * njnt];
        geom_xpos = new double[3 * ngeom];
        geom_xmat = new double[9 * ngeom];
        site_xpos = new double[3 * m.nsite];
        site_xmat = new double[9 * m.nsite];
        subtree_com = new double[3 * nbody];
        cinert = new double[10 * nbody];
        cdof = new double[6 * nv];
        crb = new double[10 * nbody];
        M = new double[m.nC];
        qLD = new double[m.nC];
        qLDiagInv = new double[nv];
        actuator_moment = new double[nu];
        actuator_length = new double[nu];
        con_dist = new double[maxcon];
        con_pos = new double[3 * maxcon];
        con_frame = new double[9 * maxcon];
        con_includemargin = new double[maxcon];
        con_friction = new double[5 * maxcon];
        con_solref = new double[2 * maxcon];
        con_solimp = new double[5 * maxcon];
        con_mu = new double[maxcon];
        con_geom1 = new int[maxcon];
        con_geom2 = new int[maxcon];
        con_dim = new int[maxcon];
        con_exclude = new int[maxcon];
        con_efc_address = new int[maxcon];
        efc_J = new double[maxefc * nv];
        efc_type = new int[maxefc];
        efc_id = new int[maxefc];
        efc_pos = new double[maxefc];
        efc_margin = new double[maxefc];
        efc_frictionloss = new double[maxefc];
        efc_diagA = new double[maxefc];
        efc_KBIP = new double[4 * maxefc];
        efc_D = new double[maxefc];
        efc_R = new double[maxefc];
        efc_vel = new double[maxefc];
        efc_aref = new double[maxefc];
        efc_b = new double[maxefc];
        efc_force = new double[maxefc];
        efc_state = new int[maxefc];
        cvel = new double[6 * nbody];
        cdof_dot = new double[6 * nv];
        qfrc_spring = new double[nv];
        qfrc_damper = new double[nv];
        qfrc_passive = new double[nv];
        qfrc_fluid = new double[nv];
        fluidJacp = new double[3 * nv];
        fluidJacr = new double[3 * nv];
        fluidQforce = new double[nv];
        treeParent = new int[m.ntree];
        treeIsland = new int[m.ntree];
        efcTree = new int[maxefc];
        mapDof2idof = new int[nv];
        mapIdof2dof = new int[nv];
        mapIefc2efc = new int[maxefc];
        islandNv = new int[m.ntree + 1];
        islandIdofadr = new int[m.ntree + 1];
        islandNefc = new int[m.ntree + 1];
        islandIefcadr = new int[m.ntree + 1];
        islandFill = new int[m.ntree + 1];
        iM = new double[m.nC];
        iQLD = new double[m.nC];
        iMcolind = new int[m.nC];
        iMrownnz = new int[nv];
        iMrowadr = new int[nv];
        iQLDiagInv = new double[nv];
        iJ = new double[maxefc * nv];
        iD = new double[maxefc];
        iAref = new double[maxefc];
        iForce = new double[maxefc];
        iQacc = new double[nv];
        iQaccSmooth = new double[nv];
        iQfrcSmooth = new double[nv];
        iQfrcConstraint = new double[nv];
        qfrc_bias = new double[nv];
        actuator_force = new double[nu];
        qfrc_actuator = new double[nv];
        qfrc_smooth = new double[nv];
        qacc_smooth = new double[nv];
        qacc = new double[nv];
        qfrc_constraint = new double[nv];
        qH = new double[m.nC];
        qHDiagInv = new double[nv];
        cacc = new double[6 * nbody];
        cfrc = new double[6 * nbody];
        jac1 = new double[3 * nv];
        jac2 = new double[3 * nv];
        jacdif = new double[6 * nv];
        jacrot = new double[3 * nv];
        jar = new double[maxefc];
        da = new double[nv];
        Mda = new double[nv];
        qaccEuler = new double[nv];
        Jaref = new double[maxefc];
        Jv = new double[maxefc];
        Ma = new double[nv];
        Mv = new double[nv];
        grad = new double[nv];
        Mgrad = new double[nv];
        search = new double[nv];
        quad = new double[3 * maxefc];
        Hdiag = new double[maxefc];
        cholupd = new double[nv];
        L = new double[nv * nv];
        sState = new int[maxefc];
        oldState = new int[maxefc];
        ten_length = new double[m.ntendon];
        ten_J = new double[m.ntendon > 0 ? m.ten_J_rowadr[m.ntendon - 1] + m.ten_J_rownnz[m.ntendon - 1] : 0];
        ten_velocity = new double[m.ntendon];
        cfrc_ext = new double[6 * nbody];
        bool pgs = m.solver == SolverPgs;
        // PGS needs the dense nefc x nefc dual matrix, the Newton solver none
        efc_AR = pgs ? new double[maxefc * maxefc] : [];
        efcY = pgs ? new double[maxefc * nv] : [];
        efcYT = pgs ? new double[maxefc * nv] : [];
        sqrtInvD = new double[nv];
        arInv = pgs ? new double[maxefc] : [];
        forcePrev = pgs ? new double[maxefc] : [];
        forceMomentum = pgs ? new double[maxefc] : [];
        arForce = pgs ? new double[maxefc] : [];
        blockStart = pgs ? new int[maxefc] : [];
        rkX = new double[4 * (m.nq + nv)];
        rkF = new double[4 * nv];
        rkdX = new double[2 * nv];
    }

    /// <summary>
    ///     The geom pairs that can collide, in MuJoCo's order (<c>mj_broadphase</c>, <c>mj_collision</c>,
    ///     <c>mj_collideTree</c>): body pairs sorted by (lower id, higher id), keeping those MuJoCo keeps (both bodies
    ///     collidable, not welded together, not both dof-less, not parent and child unless one is the world, compatible
    ///     bitmasks), then each pair's geoms: the single geom of each, or all their geom pairs in the sorted order the
    ///     midphase leaves them in. Static filters (geom bitmasks, a collider for the two types) are applied here; the
    ///     bounding-sphere filter, which depends on the pose, runs at each step.
    /// </summary>
    /// <param name="m">The model.</param>
    /// <returns>The first and second geom of each candidate, the first of the lower geom type (of either body).</returns>
    /// <remarks>
    ///     MuJoCo's sweep-and-prune and the midphase's box tests only prune pairs that cannot touch, so walking every body
    ///     pair yields the same contacts in the same order. Within a pair of bodies the order is
    ///     <see cref="GeomPairOrder"/>'s sort, which needs no bounding volume hierarchy from the dump.
    /// </remarks>
    private static (int[] A, int[] B) CollisionCandidates(CheetahModel m) {
        var a = new System.Collections.Generic.List<int>();
        var b = new System.Collections.Generic.List<int>();
        bool CanCollide(int body) => m.body_contype[body] != 0 || m.body_conaffinity[body] != 0;
        int GeomTypeMask(int body, bool affinity) {
            int mask = 0;
            for (int g = m.body_geomadr[body]; g < m.body_geomadr[body] + m.body_geomnum[body]; g++)
                mask |= affinity ? m.geom_conaffinity[g] : m.geom_contype[g];
            return mask;
        }
        bool GeomPairOk(int g1, int g2) {
            // filterBitmask, then a collider must exist for the type pair (plane-plane has none)
            if ((m.geom_contype[g1] & m.geom_conaffinity[g2]) == 0 && (m.geom_contype[g2] & m.geom_conaffinity[g1]) == 0)
                return false;
            return !(m.geom_type[g1] == Plane && m.geom_type[g2] == Plane);
        }
        for (int b1 = 0; b1 < m.nbody; b1++) {
            for (int b2 = b1 + 1; b2 < m.nbody; b2++) {
                if (!CanCollide(b1) || !CanCollide(b2))
                    continue;
                if (b1 == 0 && m.body_geomnum[0] == 0)
                    continue;
                // filterBodyPair
                int weld1 = m.body_weldid[b1], weld2 = m.body_weldid[b2];
                int parentWeld1 = m.body_weldid[m.body_parentid[weld1]], parentWeld2 = m.body_weldid[m.body_parentid[weld2]];
                if (weld1 == weld2 || (m.body_dofnum[weld1] == 0 && m.body_dofnum[weld2] == 0))
                    continue;
                if (weld1 != 0 && weld2 != 0 && (weld1 == parentWeld2 || weld2 == parentWeld1))
                    continue;
                // add_pair's compatibility from the geoms' masks, then mj_collision's canCollide2 on the body masks
                if ((GeomTypeMask(b1, false) & GeomTypeMask(b2, true)) == 0 && (GeomTypeMask(b2, false) & GeomTypeMask(b1, true)) == 0)
                    continue;
                if ((m.body_contype[b1] & m.body_conaffinity[b2]) == 0 && (m.body_contype[b2] & m.body_conaffinity[b1]) == 0)
                    continue;
                foreach ((int g1, int g2) in GeomPairOrder(m, b1, b2))
                    if (GeomPairOk(g1, g2)) {
                        a.Add(g1);
                        b.Add(g2);
                    }
            }
        }
        return (a.ToArray(), b.ToArray());
    }

    /// <summary>
    ///     The geom pairs of two bodies in the order MuJoCo's narrow phase takes them: the one pair when each body has a
    ///     single geom, else every pair of their geoms sorted by first then second geom, as <c>mj_collideTree</c> sorts
    ///     the pairs its traversal finds (<c>pairSort</c>). Each pair has its lower-type geom first
    ///     (<c>pushGeomGeom</c>'s swap), and the sort keys on the swapped pair.
    /// </summary>
    /// <param name="m">The model.</param>
    /// <param name="b1">The lower body.</param>
    /// <param name="b2">The higher body.</param>
    /// <returns>The pairs, each (lower-type geom, other geom), in narrow-phase order.</returns>
    /// <remarks>
    ///     The traversal (it descends the hierarchy whose stored box has the larger surface) decides only which pairs
    ///     survive its box tests, never their order: after the sort the order depends on geom ids and types alone. Its box
    ///     tests are conservative, so a pair it would prune makes no contact here either. Because the key is the swapped
    ///     pair, a sphere of the higher body sorts before the capsules of the lower one (Humanoid's hands against its
    ///     torso), which a plain ascending walk over the two bodies gets wrong. MuJoCo walks the pairs unsorted only with
    ///     the midphase disabled, which <see cref="CheetahModel.Load"/> refuses (default flags only).
    /// </remarks>
    private static System.Collections.Generic.List<(int, int)> GeomPairOrder(CheetahModel m, int b1, int b2) {
        var pairs = new System.Collections.Generic.List<(int, int)>();
        for (int g1 = m.body_geomadr[b1]; g1 < m.body_geomadr[b1] + m.body_geomnum[b1]; g1++)
            for (int g2 = m.body_geomadr[b2]; g2 < m.body_geomadr[b2] + m.body_geomnum[b2]; g2++)
                pairs.Add(m.geom_type[g1] > m.geom_type[g2] ? (g2, g1) : (g1, g2));
        // two single-geom bodies skip the midphase and its sort; one pair needs no order anyway
        if (m.body_geomnum[b1] > 1 || m.body_geomnum[b2] > 1)
            pairs.Sort();
        return pairs;
    }

    /// <summary>
    ///     Advances the simulation by one time step, as <c>mj_step</c>: state checks, the forward dynamics at the current
    ///     state, then the model's integrator: semi-implicit Euler (HalfCheetah) or RK4 (Hopper, Humanoid), which updates
    ///     <see cref="qvel"/>, <see cref="qpos"/> and <see cref="time"/>.
    /// </summary>
    /// <remarks>
    ///     A NaN or a value beyond 1e10 in qpos, qvel or the solved qacc resets the state to the reference configuration at
    ///     rest (MuJoCo's auto-reset), controls included; the warning MuJoCo records is not kept. After an RK4 step the
    ///     published arrays (frames, contacts, constraint rows, <see cref="qacc"/>) are those of its last stage, as in
    ///     MuJoCo's <c>mjData</c>.
    /// </remarks>
    public void Step() {
        if (AnyBad(qpos, m.nq) || AnyBad(qvel, nv))
            ResetData();
        Forward();
        if (AnyBad(qacc, nv)) {
            ResetData();
            Forward();
        }
        if (m.integrator == IntegratorRk4)
            RungeKutta();
        else
            Euler();
    }

    /// <summary>Whether any of the first <paramref name="n"/> values is NaN or beyond ±1e10 (<c>mju_isBad</c>).</summary>
    /// <param name="v">The values.</param>
    /// <param name="n">How many to check.</param>
    /// <returns><see langword="true"/> when a value is bad.</returns>
    private static bool AnyBad(double[] v, int n) {
        for (int i = 0; i < n; i++)
            if (double.IsNaN(v[i]) || v[i] > MaxVal || v[i] < -MaxVal)
                return true;
        return false;
    }

    /// <summary>The part of <c>mj_resetData</c> this core's state holds: reference configuration, at rest, time 0.</summary>
    private void ResetData() {
        Array.Copy(m.qpos0, qpos, m.nq);
        Array.Clear(qvel);
        Array.Clear(ctrl);
        Array.Clear(qacc_warmstart);
        Array.Clear(qacc);
        time = 0;
    }

    /// <summary>
    ///     The forward dynamics at the current state (<c>mj_forward</c>): positions, velocities, actuation, the
    ///     unconstrained acceleration and the constraint solve. Leaves the state untouched.
    /// </summary>
    public void Forward() {
        // position stage
        Kinematics();
        ComPos();
        Tendons();
        MakeM();
        Array.Copy(M, qLD, m.nC);
        Kernels.FactorI(qLD, qLDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        Collision();
        MakeConstraint();
        Islands();
        // mj_projectConstraint: the dual solver's matrix, built in the position stage
        if (m.solver == SolverPgs && nefc > 0)
            MakeDualMatrix();
        for (int i = 0; i < nu; i++) {
            int j = m.actuator_trnid[2 * i];
            actuator_length[i] = qpos[m.jnt_qposadr[j]] * m.actuator_gear[6 * i];
            actuator_moment[i] = m.actuator_gear[6 * i];
        }
        // velocity stage: tendon velocities (a sparse product per tendon), then the bodies
        for (int t = 0; t < m.ntendon; t++) {
            int adr = m.ten_J_rowadr[t];
            ten_velocity[t] = Kernels.DotSparse(ten_J.AsSpan(adr), qvel, m.ten_J_rownnz[t], m.ten_J_colind.AsSpan(adr));
        }
        ComVel();
        Passive();
        Kernels.MulMatVec(efc_vel, efc_J, qvel, nefc, nv);
        for (int i = 0; i < nefc; i++) {
            // aref = -B*vel - K*I*(pos - margin + shift*vel); shift is 0 outside the discrete integrator, and its
            // product is still added because 0*vel can be -0
            ReadOnlySpan<double> kbip = efc_KBIP.AsSpan(4 * i, 4);
            efc_aref[i] = -kbip[1] * efc_vel[i] - kbip[0] * kbip[2] * (efc_pos[i] - efc_margin[i] + 0.0 * efc_vel[i]);
        }
        Rne();
        // actuation, acceleration, constraints
        Actuation();
        Kernels.Sub(qfrc_smooth, qfrc_passive, qfrc_bias, nv);
        // qfrc_applied is all zero, yet MuJoCo adds it: x + 0 turns a -0 into +0
        for (int i = 0; i < nv; i++)
            qfrc_smooth[i] += 0.0;
        Kernels.AddTo(qfrc_smooth, qfrc_actuator, nv);
        Array.Copy(qfrc_smooth, qacc_smooth, nv);
        Kernels.SolveLD(qacc_smooth, qLD, qLDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        FwdConstraint();
    }

    // ================================================================================== kinematics

    /// <summary>
    ///     Body, inertial, geom and site frames from the joint positions (<c>mj_kinematics</c>): each body composes its
    ///     parent's frame with its fixed offset, then its joints in order (slides translate along the rotated axis, hinges
    ///     rotate about the anchor), and its quaternion is renormalized. A free joint instead sets the body's frame from its
    ///     7 coordinates, the quaternion normalized (and normalized again with the others).
    /// </summary>
    private void Kinematics() {
        xpos[0] = xpos[1] = xpos[2] = 0;
        xquat[0] = 1;
        xquat[1] = xquat[2] = xquat[3] = 0;
        xipos[0] = xipos[1] = xipos[2] = 0;
        Array.Clear(xmat, 0, 9);
        Array.Clear(ximat, 0, 9);
        xmat[0] = xmat[4] = xmat[8] = 1;
        ximat[0] = ximat[4] = ximat[8] = 1;
        Span<double> xp = stackalloc double[3], xq = stackalloc double[4], qloc = stackalloc double[4], vec = stackalloc double[3];
        for (int i = 1; i < nbody; i++) {
            int jfree = m.body_jntadr[i];
            if (m.body_jntnum[i] == 1 && m.jnt_type[jfree] == Free) {
                int qadr = m.jnt_qposadr[jfree];
                qpos.AsSpan(qadr, 3).CopyTo(xp);
                qpos.AsSpan(qadr + 3, 4).CopyTo(xq);
                Spatial.Normalize4(xq);
                xp.CopyTo(xanchor.AsSpan(3 * jfree));
                m.jnt_axis.AsSpan(3 * jfree, 3).CopyTo(xaxis.AsSpan(3 * jfree));
                Spatial.Normalize4(xq);
                xq.CopyTo(xquat.AsSpan(4 * i));
                xp.CopyTo(xpos.AsSpan(3 * i));
                Spatial.Quat2Mat(xmat.AsSpan(9 * i), xq);
                continue;
            }
            int pid = m.body_parentid[i];
            if (pid != 0) {
                Spatial.MulMatVec3(xp, xmat.AsSpan(9 * pid), m.body_pos.AsSpan(3 * i));
                xp[0] += xpos[3 * pid];
                xp[1] += xpos[3 * pid + 1];
                xp[2] += xpos[3 * pid + 2];
                Spatial.MulQuat(xq, xquat.AsSpan(4 * pid), m.body_quat.AsSpan(4 * i));
            } else {
                m.body_pos.AsSpan(3 * i, 3).CopyTo(xp);
                m.body_quat.AsSpan(4 * i, 4).CopyTo(xq);
            }
            int jend = m.body_jntadr[i] + m.body_jntnum[i];
            for (int j = m.body_jntadr[i]; j < jend; j++) {
                int qadr = m.jnt_qposadr[j];
                Span<double> axis = xaxis.AsSpan(3 * j, 3), anchor = xanchor.AsSpan(3 * j, 3);
                Spatial.RotVecQuat(axis, m.jnt_axis.AsSpan(3 * j), xq);
                Spatial.RotVecQuat(anchor, m.jnt_pos.AsSpan(3 * j), xq);
                anchor[0] += xp[0];
                anchor[1] += xp[1];
                anchor[2] += xp[2];
                if (m.jnt_type[j] == Slide) {
                    double s = qpos[qadr] - m.qpos0[qadr];
                    xp[0] += axis[0] * s;
                    xp[1] += axis[1] * s;
                    xp[2] += axis[2] * s;
                } else {
                    // the local joint axis, not the rotated one, builds the rotation; the anchor then corrects xpos
                    Spatial.AxisAngle2Quat(qloc, m.jnt_axis.AsSpan(3 * j), qpos[qadr] - m.qpos0[qadr]);
                    Spatial.MulQuat(xq, xq, qloc);
                    Spatial.RotVecQuat(vec, m.jnt_pos.AsSpan(3 * j), xq);
                    xp[0] = anchor[0] - vec[0];
                    xp[1] = anchor[1] - vec[1];
                    xp[2] = anchor[2] - vec[2];
                }
            }
            Spatial.Normalize4(xq);
            xq.CopyTo(xquat.AsSpan(4 * i));
            xp.CopyTo(xpos.AsSpan(3 * i));
            Spatial.Quat2Mat(xmat.AsSpan(9 * i), xq);
        }
        for (int i = 1; i < nbody; i++)
            Local2Global(xipos.AsSpan(3 * i), ximat.AsSpan(9 * i), m.body_ipos.AsSpan(3 * i), m.body_iquat.AsSpan(4 * i), i, m.body_sameframe[i]);
        for (int b = 0; b < nbody; b++) {
            int end = m.body_geomadr[b] + m.body_geomnum[b];
            for (int g = m.body_geomadr[b]; g < end; g++)
                Local2Global(geom_xpos.AsSpan(3 * g), geom_xmat.AsSpan(9 * g), m.geom_pos.AsSpan(3 * g), m.geom_quat.AsSpan(4 * g), m.geom_bodyid[g], m.geom_sameframe[g]);
        }
        // sites in index order, composed like geoms (mj_kinematics2)
        for (int s = 0; s < m.nsite; s++)
            Local2Global(site_xpos.AsSpan(3 * s), site_xmat.AsSpan(9 * s), m.site_pos.AsSpan(3 * s), m.site_quat.AsSpan(4 * s), m.site_bodyid[s], m.site_sameframe[s]);
    }

    /// <summary>
    ///     Maps a frame fixed in a body to the world (<c>mj_local2Global</c>); <paramref name="sameframe"/> lets MuJoCo copy
    ///     the body or inertial frame instead of composing (1 body, 2 inertia, 3 body rotation, 4 inertia rotation).
    /// </summary>
    /// <param name="pos">Receives the world position.</param>
    /// <param name="mat">Receives the world rotation matrix.</param>
    /// <param name="lpos">The local position.</param>
    /// <param name="lquat">The local orientation.</param>
    /// <param name="body">The body.</param>
    /// <param name="sameframe">MuJoCo's <c>mjtSameFrame</c> of the frame.</param>
    private void Local2Global(Span<double> pos, Span<double> mat, ReadOnlySpan<double> lpos, ReadOnlySpan<double> lquat, int body, int sameframe) {
        switch (sameframe) {
            case 1:
                xpos.AsSpan(3 * body, 3).CopyTo(pos);
                break;
            case 2:
                xipos.AsSpan(3 * body, 3).CopyTo(pos);
                break;
            default:
                Spatial.MulMatVec3(pos, xmat.AsSpan(9 * body), lpos);
                pos[0] += xpos[3 * body];
                pos[1] += xpos[3 * body + 1];
                pos[2] += xpos[3 * body + 2];
                break;
        }
        switch (sameframe) {
            case 0:
                Span<double> q = stackalloc double[4];
                Spatial.MulQuat(q, xquat.AsSpan(4 * body), lquat);
                Spatial.Quat2Mat(mat, q);
                break;
            case 1:
            case 3:
                xmat.AsSpan(9 * body, 9).CopyTo(mat);
                break;
            default:
                ximat.AsSpan(9 * body, 9).CopyTo(mat);
                break;
        }
    }

    /// <summary>
    ///     Subtree centers of mass, body inertias about them and the motion axes of the dofs (<c>mj_comPos</c>).
    /// </summary>
    private void ComPos() {
        for (int i = 0; i < nbody; i++)
            for (int k = 0; k < 3; k++)
                subtree_com[3 * i + k] = xipos[3 * i + k] * m.body_mass[i];
        for (int i = nbody - 1; i > 0; i--) {
            int p = m.body_parentid[i];
            subtree_com[3 * p] += subtree_com[3 * i];
            subtree_com[3 * p + 1] += subtree_com[3 * i + 1];
            subtree_com[3 * p + 2] += subtree_com[3 * i + 2];
        }
        for (int i = 0; i < nbody; i++) {
            if (m.body_subtreemass[i] < Kernels.MinVal) {
                xipos.AsSpan(3 * i, 3).CopyTo(subtree_com.AsSpan(3 * i));
            } else {
                // MuJoCo multiplies by the reciprocal, which rounds differently from dividing
                double inv = 1.0 / m.body_subtreemass[i];
                for (int k = 0; k < 3; k++)
                    subtree_com[3 * i + k] *= inv;
            }
        }
        Array.Clear(cinert, 0, 10);
        Span<double> offset = stackalloc double[3];
        for (int i = 1; i < nbody; i++) {
            int root = m.body_rootid[i];
            for (int k = 0; k < 3; k++)
                offset[k] = xipos[3 * i + k] - subtree_com[3 * root + k];
            Spatial.InertCom(cinert.AsSpan(10 * i), m.body_inertia.AsSpan(3 * i), ximat.AsSpan(9 * i), offset, m.body_mass[i]);
        }
        Span<double> ax = stackalloc double[3];
        for (int i = 1; i < nbody; i++) {
            int root = m.body_rootid[i], jend = m.body_jntadr[i] + m.body_jntnum[i];
            for (int j = m.body_jntadr[i]; j < jend; j++) {
                Span<double> dof = cdof.AsSpan(6 * m.jnt_dofadr[j], 6);
                for (int k = 0; k < 3; k++)
                    offset[k] = subtree_com[3 * root + k] - xanchor[3 * j + k];
                if (m.jnt_type[j] == Free) {
                    // three translations along the world axes, then three rotations about the body frame's axes
                    // (columns of xmat), each crossed with the offset like a hinge
                    Span<double> free = cdof.AsSpan(6 * m.jnt_dofadr[j], 36);
                    free.Slice(0, 18).Clear();
                    free[3] = free[3 + 7] = free[3 + 14] = 1;
                    for (int k = 0; k < 3; k++) {
                        ax[0] = xmat[9 * i + k];
                        ax[1] = xmat[9 * i + k + 3];
                        ax[2] = xmat[9 * i + k + 6];
                        ax.CopyTo(free.Slice(18 + 6 * k));
                        Spatial.Cross(free.Slice(18 + 6 * k + 3), ax, offset);
                    }
                } else if (m.jnt_type[j] == Slide) {
                    dof[0] = dof[1] = dof[2] = 0;
                    xaxis.AsSpan(3 * j, 3).CopyTo(dof.Slice(3));
                } else {
                    xaxis.AsSpan(3 * j, 3).CopyTo(dof);
                    Spatial.Cross(dof.Slice(3), xaxis.AsSpan(3 * j, 3), offset);
                }
            }
        }
    }

    /// <summary>
    ///     The joint-space inertia by the composite rigid body algorithm (<c>mj_crb</c>): composite inertias accumulated
    ///     from the leaves, then for each dof its armature plus the projections onto its ancestor chain.
    /// </summary>
    private void MakeM() {
        Array.Copy(cinert, crb, 10 * nbody);
        for (int i = nbody - 1; i >= 0; i--) {
            int p = m.body_parentid[i];
            if (p > 0)
                Kernels.AddTo(crb.AsSpan(10 * p), crb.AsSpan(10 * i), 10);
        }
        Array.Clear(M);
        Span<double> buf = stackalloc double[6];
        for (int i = 0; i < nv; i++) {
            // a simple dof's row holds only its constant diagonal, set as mj_crb does (not computed, so its bits are M0's)
            if (m.dof_simplenum[i] != 0) {
                M[m.M_rowadr[i]] = m.dof_M0[i];
                continue;
            }
            int adr = m.M_rowadr[i] + m.M_rownnz[i] - 1;
            // the armature MuJoCo adds from actuators is 0 for every joint here (no inherited armature), still added
            M[adr] = m.dof_armature[i] + 0.0;
            Spatial.MulInertVec(buf, crb.AsSpan(10 * m.dof_bodyid[i]), cdof.AsSpan(6 * i));
            for (int j = i; j >= 0; j = m.dof_parentid[j])
                M[adr--] += Spatial.Dot6(cdof.AsSpan(6 * j), buf);
        }
    }

    // ================================================================================== collision

    /// <summary>
    ///     Contacts in MuJoCo's order (<c>mj_collision</c>): the candidate geom pairs of <see cref="CollisionCandidates"/>,
    ///     each filtered by the bounding spheres (or the plane-to-sphere distance) at the current pose, then the pair's
    ///     collider, which takes the lower geom type first (<c>pushGeomGeom</c>), and its contacts' parameters.
    /// </summary>
    /// <remarks>
    ///     A body with several geoms goes through MuJoCo's bounding-volume midphase (<c>mj_collideTree</c>), which sorts
    ///     the pairs it finds by (first geom, second geom) after putting the lower geom type first; that order is
    ///     <see cref="GeomPairOrder"/>'s. A depth-first order taken from the hierarchies instead broke HalfCheetah's flipped
    ///     episodes at step 38 (torso and head on the ground), by one ulp.
    /// </remarks>
    /// <exception cref="NotSupportedException">
    ///     A pair of geom types without a collider here, or a convex pair on a path the core leaves out (no margin,
    ///     polygon clipping, other geom types); <see cref="CheetahModel.Load"/> refuses such models first.
    /// </exception>
    /// <exception cref="InvalidOperationException">A degenerate contact normal (MuJoCo's error, which should not occur).</exception>
    private void Collision() {
        ncon = 0;
        for (int c = 0; c < candA.Length; c++) {
            // the candidates already have the lower geom type first (pushGeomGeom's swap, made in GeomPairOrder); the
            // sphere filter and the summed margins are symmetric, so the swap made before them changes no bit
            int g1 = candA[c], g2 = candB[c];
            double margin = m.geom_margin[g1] + m.geom_margin[g2];
            double gap = m.geom_gap[g1] + m.geom_gap[g2];
            if (FilterSphere(g1, g2, margin + gap))
                continue;
            int first = ncon;
            switch (m.geom_type[g1], m.geom_type[g2]) {
                case (Plane, Capsule):
                    PlaneCapsule(g1, g2, margin + gap);
                    break;
                case (Plane, Sphere):
                    PlaneSphere(g1, geom_xpos.AsSpan(3 * g2, 3), m.geom_size[3 * g2], margin + gap);
                    break;
                case (Sphere, Sphere):
                    SphereSphere(geom_xpos.AsSpan(3 * g1, 3), geom_xmat.AsSpan(9 * g1, 9), m.geom_size[3 * g1],
                                 geom_xpos.AsSpan(3 * g2, 3), geom_xmat.AsSpan(9 * g2, 9), m.geom_size[3 * g2], margin + gap);
                    break;
                case (Sphere, Capsule):
                    SphereCapsule(g1, g2, margin + gap);
                    break;
                case (Capsule, Capsule):
                    CapsuleCapsule(g1, g2, margin + gap);
                    break;
                case (Plane, Cylinder):
                    PlaneCylinder(g1, g2, margin + gap);
                    break;
                case (Capsule, Cylinder):
                    Convex(g1, g2, margin + gap);
                    break;
                default:
                    // the loader refuses every other pair of types that can touch
                    throw new NotSupportedException($"the core: no collider for geom types {m.geom_type[g1]} and {m.geom_type[g2]}");
            }
            if (ncon > first)
                ContactParams(first, ncon, g1, g2, margin);
        }
    }

    /// <summary>
    ///     The bounding-sphere filter at the current pose (<c>mj_filterSphere</c>): two spheres farther apart than their
    ///     radii and the margin, or a geom's sphere farther from a plane than its radius and the margin, cannot touch.
    /// </summary>
    /// <param name="g1">The first geom.</param>
    /// <param name="g2">The second geom.</param>
    /// <param name="margin">The detection margin (margin plus gap).</param>
    /// <returns><see langword="true"/> when the pair is dropped.</returns>
    private bool FilterSphere(int g1, int g2, double margin) {
        double r1 = m.geom_rbound[g1], r2 = m.geom_rbound[g2];
        if (r1 > 0 && r2 > 0) {
            double d0 = geom_xpos[3 * g1] - geom_xpos[3 * g2];
            double d1 = geom_xpos[3 * g1 + 1] - geom_xpos[3 * g2 + 1];
            double d2 = geom_xpos[3 * g1 + 2] - geom_xpos[3 * g2 + 2];
            double bound = r1 + r2 + margin;
            return d0 * d0 + d1 * d1 + d2 * d2 > bound * bound;
        }
        if (m.geom_type[g1] == Plane && r2 > 0 && PlaneDist(g1, g2) > margin + r2)
            return true;
        if (m.geom_type[g2] == Plane && r1 > 0 && PlaneDist(g2, g1) > margin + r1)
            return true;
        return false;
    }

    /// <summary>The signed distance of a geom's center above a plane (<c>planeGeomDist</c>).</summary>
    /// <param name="plane">The plane geom.</param>
    /// <param name="g">The other geom.</param>
    /// <returns>The distance along the plane normal.</returns>
    private double PlaneDist(int plane, int g) {
        Span<double> dif = stackalloc double[3], normal = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            dif[k] = geom_xpos[3 * g + k] - geom_xpos[3 * plane + k];
            normal[k] = geom_xmat[9 * plane + 3 * k + 2];
        }
        return Spatial.Dot3(dif, normal);
    }

    /// <summary>
    ///     Appends the contacts of a capsule with a plane (<c>mjc_PlaneCapsule</c>): each end sphere of the capsule's
    ///     segment within <paramref name="margin"/> of the plane gives one contact, with the capsule axis as tangent.
    /// </summary>
    /// <param name="plane">The plane geom.</param>
    /// <param name="g">The capsule geom.</param>
    /// <param name="margin">The detection margin (margin plus gap).</param>
    private void PlaneCapsule(int plane, int g, double margin) {
        ReadOnlySpan<double> mat2 = geom_xmat.AsSpan(9 * g, 9), pos2 = geom_xpos.AsSpan(3 * g, 3);
        Span<double> axis = stackalloc double[3] { mat2[2], mat2[5], mat2[8] };
        double half = m.geom_size[3 * g + 1];
        Span<double> segment = stackalloc double[3] { half * axis[0], half * axis[1], half * axis[2] };
        Span<double> end = stackalloc double[3];
        for (int side = 0; side < 2; side++) {
            for (int k = 0; k < 3; k++)
                end[k] = side == 0 ? pos2[k] + segment[k] : pos2[k] - segment[k];
            if (PlaneSphere(plane, end, m.geom_size[3 * g], margin)) {
                axis.CopyTo(con_frame.AsSpan(9 * (ncon - 1) + 3));
            }
        }
    }

    /// <summary>
    ///     Appends the contact of a sphere with a plane when the sphere is within <paramref name="margin"/>
    ///     (<c>mjraw_PlaneSphere</c>): distance <c>(c - p)·n - r</c>, point halfway into the overlap.
    /// </summary>
    /// <param name="plane">The plane geom.</param>
    /// <param name="center">The sphere center.</param>
    /// <param name="radius">The sphere radius.</param>
    /// <param name="margin">The detection margin.</param>
    /// <returns>Whether a contact was appended (its tangent is left zero for the caller).</returns>
    private bool PlaneSphere(int plane, ReadOnlySpan<double> center, double radius, double margin) {
        Span<double> normal = stackalloc double[3], tmp = stackalloc double[3];
        for (int k = 0; k < 3; k++) {
            normal[k] = geom_xmat[9 * plane + 3 * k + 2];
            tmp[k] = center[k] - geom_xpos[3 * plane + k];
        }
        double cdist = Spatial.Dot3(tmp, normal);
        if (cdist > margin + radius)
            return false;
        int c = ncon++;
        double dist = cdist - radius;
        con_dist[c] = dist;
        double s = -dist / 2 - radius;
        for (int k = 0; k < 3; k++) {
            con_pos[3 * c + k] = center[k] + normal[k] * s;
            con_frame[9 * c + k] = normal[k];
            con_frame[9 * c + 3 + k] = 0;
        }
        return true;
    }

    /// <summary>
    ///     Appends the contact of two spheres within <paramref name="margin"/> (<c>mjraw_SphereSphere</c>): distance
    ///     between centers minus both radii, normal from the first center to the second (from the two z axes' cross product
    ///     when the centers coincide), point halfway into the overlap from the first sphere's surface.
    /// </summary>
    /// <param name="pos1">The first center.</param>
    /// <param name="mat1">The first geom's rotation (its z axis serves the coincident case).</param>
    /// <param name="r1">The first radius.</param>
    /// <param name="pos2">The second center.</param>
    /// <param name="mat2">The second geom's rotation.</param>
    /// <param name="r2">The second radius.</param>
    /// <param name="margin">The detection margin.</param>
    /// <returns>The number of contacts appended (0 or 1); the tangent is left zero.</returns>
    private int SphereSphere(ReadOnlySpan<double> pos1, ReadOnlySpan<double> mat1, double r1,
                             ReadOnlySpan<double> pos2, ReadOnlySpan<double> mat2, double r2, double margin) {
        double d0 = pos1[0] - pos2[0], d1 = pos1[1] - pos2[1], d2 = pos1[2] - pos2[2];
        double cdistSqr = d0 * d0 + d1 * d1 + d2 * d2;
        double minDist = margin + r1 + r2;
        if (cdistSqr > minDist * minDist)
            return 0;
        int c = ncon++;
        double dist = Math.Sqrt(cdistSqr) - r1 - r2;
        con_dist[c] = dist;
        Span<double> normal = con_frame.AsSpan(9 * c, 3);
        normal[0] = pos2[0] - pos1[0];
        normal[1] = pos2[1] - pos1[1];
        normal[2] = pos2[2] - pos1[2];
        if (Spatial.Normalize3(normal) < Kernels.MinVal) {
            Span<double> axis1 = stackalloc double[3] { mat1[2], mat1[5], mat1[8] };
            Span<double> axis2 = stackalloc double[3] { mat2[2], mat2[5], mat2[8] };
            Spatial.Cross(normal, axis1, axis2);
            Spatial.Normalize3(normal);
        }
        double s = r1 + dist / 2;
        for (int k = 0; k < 3; k++) {
            con_pos[3 * c + k] = normal[k] * s;
            con_pos[3 * c + k] += pos1[k];
            con_frame[9 * c + 3 + k] = 0;
        }
        return 1;
    }

    /// <summary>
    ///     Appends the contact of a sphere and a capsule (<c>mjraw_SphereCapsule</c>): the sphere against the point of the
    ///     capsule's segment nearest its center, as two spheres.
    /// </summary>
    /// <param name="g1">The sphere geom.</param>
    /// <param name="g2">The capsule geom.</param>
    /// <param name="margin">The detection margin.</param>
    private void SphereCapsule(int g1, int g2, double margin) {
        ReadOnlySpan<double> pos1 = geom_xpos.AsSpan(3 * g1, 3), pos2 = geom_xpos.AsSpan(3 * g2, 3);
        ReadOnlySpan<double> mat2 = geom_xmat.AsSpan(9 * g2, 9);
        double len = m.geom_size[3 * g2 + 1];
        Span<double> axis = stackalloc double[3] { mat2[2], mat2[5], mat2[8] };
        Span<double> vec = stackalloc double[3] { pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2] };
        double x = Clip(Spatial.Dot3(axis, vec), -len, len);
        for (int k = 0; k < 3; k++) {
            vec[k] = axis[k] * x;
            vec[k] += pos2[k];
        }
        SphereSphere(pos1, geom_xmat.AsSpan(9 * g1, 9), m.geom_size[3 * g1], vec, mat2, m.geom_size[3 * g2], margin);
    }

    /// <summary>
    ///     Appends the contacts of two capsules (<c>mjraw_CapsuleCapsule</c>): the closest points of the two segments,
    ///     clipped to their ends, as two spheres; for (nearly) parallel axes, up to two contacts from the segments' ends.
    /// </summary>
    /// <param name="g1">The first capsule.</param>
    /// <param name="g2">The second capsule.</param>
    /// <param name="margin">The detection margin.</param>
    private void CapsuleCapsule(int g1, int g2, double margin) {
        ReadOnlySpan<double> pos1 = geom_xpos.AsSpan(3 * g1, 3), pos2 = geom_xpos.AsSpan(3 * g2, 3);
        ReadOnlySpan<double> mat1 = geom_xmat.AsSpan(9 * g1, 9), mat2 = geom_xmat.AsSpan(9 * g2, 9);
        double r1 = m.geom_size[3 * g1], h1 = m.geom_size[3 * g1 + 1];
        double r2 = m.geom_size[3 * g2], h2 = m.geom_size[3 * g2 + 1];
        Span<double> axis1 = stackalloc double[3] { mat1[2] * h1, mat1[5] * h1, mat1[8] * h1 };
        Span<double> axis2 = stackalloc double[3] { mat2[2] * h2, mat2[5] * h2, mat2[8] * h2 };
        Span<double> dif = stackalloc double[3] { pos1[0] - pos2[0], pos1[1] - pos2[1], pos1[2] - pos2[2] };
        double ma = Spatial.Dot3(axis1, axis1);
        double mb = -Spatial.Dot3(axis1, axis2);
        double mc = Spatial.Dot3(axis2, axis2);
        double u = -Spatial.Dot3(axis1, dif);
        double v = Spatial.Dot3(axis2, dif);
        double det = ma * mc - mb * mb;
        Span<double> vec1 = stackalloc double[3], vec2 = stackalloc double[3];
        // the points pos + axis*x of each segment, scaled then offset as MuJoCo does
        void Point(Span<double> res, ReadOnlySpan<double> pos, ReadOnlySpan<double> axis, double x) {
            for (int k = 0; k < 3; k++) {
                res[k] = axis[k] * x;
                res[k] += pos[k];
            }
        }
        if (Math.Abs(det) >= Kernels.MinVal) {
            double x1 = (mc * u - mb * v) / det;
            double x2 = (ma * v - mb * u) / det;
            if (x1 > 1) {
                x1 = 1;
                x2 = (v - mb) / mc;
            } else if (x1 < -1) {
                x1 = -1;
                x2 = (v + mb) / mc;
            }
            if (x2 > 1) {
                x2 = 1;
                x1 = Clip((u - mb) / ma, -1, 1);
            } else if (x2 < -1) {
                x2 = -1;
                x1 = Clip((u + mb) / ma, -1, 1);
            }
            Point(vec1, pos1, axis1, x1);
            Point(vec2, pos2, axis2, x2);
            SphereSphere(vec1, mat1, r1, vec2, mat2, r2, margin);
            return;
        }
        // parallel axes: the first segment's ends against the second, then the second's ends against the first
        for (int k = 0; k < 3; k++)
            vec1[k] = pos1[k] + axis1[k];
        Point(vec2, pos2, axis2, Clip((v - mb) / mc, -1, 1));
        int n = SphereSphere(vec1, mat1, r1, vec2, mat2, r2, margin);
        for (int k = 0; k < 3; k++)
            vec1[k] = pos1[k] - axis1[k];
        Point(vec2, pos2, axis2, Clip((v + mb) / mc, -1, 1));
        n += SphereSphere(vec1, mat1, r1, vec2, mat2, r2, margin);
        if (n >= 2)
            return;
        for (int k = 0; k < 3; k++)
            vec2[k] = pos2[k] + axis2[k];
        Point(vec1, pos1, axis1, Clip((u - mb) / ma, -1, 1));
        n += SphereSphere(vec1, mat1, r1, vec2, mat2, r2, margin);
        if (n >= 2)
            return;
        for (int k = 0; k < 3; k++)
            vec2[k] = pos2[k] - axis2[k];
        Point(vec1, pos1, axis1, Clip((u + mb) / ma, -1, 1));
        SphereSphere(vec1, mat1, r1, vec2, mat2, r2, margin);
    }

    /// <summary>
    ///     Appends the contacts of a cylinder with a plane (<c>mjc_PlaneCylinder</c>): the rim point nearest the plane at
    ///     the end of the axis that points toward it, then the other end's, then, when the disk tilts toward the plane, two
    ///     more rim points a third of the way around; up to four, all with the plane's normal.
    /// </summary>
    /// <param name="plane">The plane geom.</param>
    /// <param name="g">The cylinder geom.</param>
    /// <param name="margin">The detection margin (margin plus gap).</param>
    private void PlaneCylinder(int plane, int g, double margin) {
        ReadOnlySpan<double> pos1 = geom_xpos.AsSpan(3 * plane, 3), pos2 = geom_xpos.AsSpan(3 * g, 3);
        ReadOnlySpan<double> mat1 = geom_xmat.AsSpan(9 * plane, 9), mat2 = geom_xmat.AsSpan(9 * g, 9);
        double radius = m.geom_size[3 * g], half = m.geom_size[3 * g + 1];
        Span<double> normal = stackalloc double[3] { mat1[2], mat1[5], mat1[8] };
        Span<double> axis = stackalloc double[3] { mat2[2], mat2[5], mat2[8] };
        // project the axis on the normal, turning it toward the plane
        double prjaxis = Spatial.Dot3(normal, axis);
        if (prjaxis > 0) {
            for (int k = 0; k < 3; k++)
                axis[k] = axis[k] * -1;
            prjaxis = -prjaxis;
        }
        Span<double> vec = stackalloc double[3] { pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2] };
        double dist0 = Spatial.Dot3(vec, normal);
        // the rim direction nearest the plane: -normal without its component along the axis, scaled to the radius
        for (int k = 0; k < 3; k++)
            vec[k] = axis[k] * prjaxis;
        for (int k = 0; k < 3; k++)
            vec[k] -= normal[k];
        double lenSqr = Spatial.Dot3(vec, vec);
        if (lenSqr >= Kernels.MinVal * Kernels.MinVal) {
            double scl = radius / Math.Sqrt(lenSqr);
            vec[0] *= scl;
            vec[1] *= scl;
            vec[2] *= scl;
        } else {
            // the disk parallel to the plane: the cylinder's x axis
            vec[0] = mat2[0] * radius;
            vec[1] = mat2[3] * radius;
            vec[2] = mat2[6] * radius;
        }
        double prjvec = Spatial.Dot3(vec, normal);
        for (int k = 0; k < 3; k++)
            axis[k] = axis[k] * half;
        prjaxis *= half;
        // the near end's rim point: none at all when it is beyond the margin
        if (!(dist0 + prjaxis + prjvec <= margin))
            return;
        AddPlaneCylinderContact(dist0 + prjaxis + prjvec, pos2, vec, 1, axis, 1, vec, 0, normal);
        if (dist0 - prjaxis + prjvec <= margin)
            AddPlaneCylinderContact(dist0 - prjaxis + prjvec, pos2, vec, 1, axis, -1, vec, 0, normal);
        double prjvec1 = -prjvec * 0.5;
        if (dist0 + prjaxis + prjvec1 <= margin) {
            // the sideways rim vector, at sqrt(3)/2 of the radius
            Span<double> vec1 = stackalloc double[3];
            Spatial.Cross(vec1, vec, axis);
            Spatial.Normalize3(vec1);
            double s3 = radius * Math.Sqrt(3.0) / 2;
            for (int k = 0; k < 3; k++)
                vec1[k] = vec1[k] * s3;
            double d = dist0 + prjaxis + prjvec1;
            AddPlaneCylinderContact(d, pos2, vec1, 1, axis, 1, vec, -0.5, normal);
            AddPlaneCylinderContact(d, pos2, vec1, -1, axis, 1, vec, -0.5, normal);
        }
    }

    /// <summary>
    ///     Appends one plane-cylinder contact at <c>pos2 ± rim ± axis + rimScale*vec - (dist/2)*normal</c>, in MuJoCo's
    ///     order of additions (<c>mji_add3</c> or <c>mji_sub3</c>, <c>mji_addTo3</c> or <c>mji_subFrom3</c>, then
    ///     <c>mji_addToScl3</c>).
    /// </summary>
    /// <param name="dist">The contact distance.</param>
    /// <param name="pos2">The cylinder's centre.</param>
    /// <param name="rim">The rim vector.</param>
    /// <param name="rimSign">+1 to add the rim vector, -1 to subtract it.</param>
    /// <param name="axis">The half-axis toward the plane.</param>
    /// <param name="axisSign">+1 to add the half-axis, -1 to subtract it.</param>
    /// <param name="vec">The nearest rim direction, added scaled for the side points.</param>
    /// <param name="vecScale">Its scale (0: not added).</param>
    /// <param name="normal">The plane's normal.</param>
    private void AddPlaneCylinderContact(double dist, ReadOnlySpan<double> pos2, ReadOnlySpan<double> rim, int rimSign,
                                         ReadOnlySpan<double> axis, int axisSign, ReadOnlySpan<double> vec, double vecScale,
                                         ReadOnlySpan<double> normal) {
        int c = ncon++;
        con_dist[c] = dist;
        Span<double> pos = con_pos.AsSpan(3 * c, 3);
        for (int k = 0; k < 3; k++)
            pos[k] = rimSign > 0 ? pos2[k] + rim[k] : pos2[k] - rim[k];
        for (int k = 0; k < 3; k++)
            pos[k] = axisSign > 0 ? pos[k] + axis[k] : pos[k] - axis[k];
        if (vecScale != 0)
            for (int k = 0; k < 3; k++)
                pos[k] += vec[k] * vecScale;
        double scl = -dist * 0.5;
        for (int k = 0; k < 3; k++)
            pos[k] += normal[k] * scl;
        for (int k = 0; k < 3; k++) {
            con_frame[9 * c + k] = normal[k];
            con_frame[9 * c + 3 + k] = 0;
        }
    }

    /// <summary>
    ///     Appends the contacts of two convex geoms by MuJoCo's native convex collider (<c>mjc_Convex</c>): one query of
    ///     <see cref="Ccd.Run"/>; when it finds one contact, MultiCCD rotates the two geoms by ±0.001 rad about each of
    ///     the contact frame's tangents, around the contact point and in opposite senses, and keeps each new contact
    ///     farther than 0.001 of the smaller bounding radius from all the others, with the first contact's distance.
    /// </summary>
    /// <param name="g1">The first geom (the capsule).</param>
    /// <param name="g2">The second geom (the cylinder).</param>
    /// <param name="margin">The detection margin (margin plus gap).</param>
    /// <exception cref="NotSupportedException">A zero margin (MuJoCo's single-pass multi-contact), or a convex path not reproduced.</exception>
    private void Convex(int g1, int g2, double margin) {
        int first = ncon;
        ccdObj1.Init(m.geom_type[g1], m.geom_size.AsSpan(3 * g1, 3), geom_xpos.AsSpan(3 * g1, 3), geom_xmat.AsSpan(9 * g1, 9), margin);
        ccdObj2.Init(m.geom_type[g2], m.geom_size.AsSpan(3 * g2, 3), geom_xpos.AsSpan(3 * g2, 3), geom_xmat.AsSpan(9 * g2, 9), margin);
        // maxContacts: a margin makes the single pass return one contact; without one, edge against face would want two
        if (!(margin > 0))
            throw new NotSupportedException("the convex collider: a pair without margin (single-pass multi-contact)");
        int n = Penetration(first, margin);
        if (n == 1) {
            Span<double> frame = stackalloc double[9], quat = stackalloc double[4], rot = stackalloc double[9], invrot = stackalloc double[9];
            con_frame.AsSpan(9 * first, 3).CopyTo(frame);
            frame.Slice(3).Clear();
            MakeFrame(frame);
            double rb1 = m.geom_rbound[g1], rb2 = m.geom_rbound[g2];
            double tolerance = 1e-3 * (rb1 <= rb2 ? rb1 : rb2);
            for (int axisId = 0; axisId < 2; axisId++) {
                for (int angleId = 0; angleId < 2; angleId++) {
                    double angle = angleId == 0 ? -1e-3 : 1e-3;
                    Spatial.AxisAngle2Quat(quat, frame.Slice(3 + 3 * axisId, 3), angle);
                    Spatial.Quat2Mat(rot, quat);
                    RotateFrame(con_pos.AsSpan(3 * first, 3), rot, ccdObj1.Mat, ccdObj1.Pos);
                    Kernels.Transpose(invrot, rot, 3, 3);
                    RotateFrame(con_pos.AsSpan(3 * first, 3), invrot, ccdObj2.Mat, ccdObj2.Pos);
                    // the candidate is written after the kept contacts and kept only when distinct
                    if (Penetration(first + n, margin) > 0 && IsDistinctContact(first, n + 1, tolerance)) {
                        con_dist[first + n] = con_dist[first];
                        n++;
                    }
                    geom_xpos.AsSpan(3 * g1, 3).CopyTo(ccdObj1.Pos);
                    geom_xmat.AsSpan(9 * g1, 9).CopyTo(ccdObj1.Mat);
                    geom_xpos.AsSpan(3 * g2, 3).CopyTo(ccdObj2.Pos);
                    geom_xmat.AsSpan(9 * g2, 9).CopyTo(ccdObj2.Mat);
                }
            }
        }
        ncon = first + n;
    }

    /// <summary>
    ///     One convex query turned into contacts (<c>mjc_penetration</c>, nativeccd): each witness pair of a negative
    ///     distance gives a contact at the pair's midpoint, with distance <c>margin + witness distance</c> and normal from
    ///     the second witness to the first; the tangent is left zero.
    /// </summary>
    /// <param name="at">The index of the first contact written.</param>
    /// <param name="margin">The detection margin.</param>
    /// <returns>The contacts written (0 when the geoms do not penetrate).</returns>
    private int Penetration(int at, double margin) {
        double dist = Ccd.Run(ccdStatus, ccdPoly, ccdObj1, ccdObj2, m.ccd_iterations, m.ccd_tolerance, 1);
        if (!(dist < 0))
            return 0;
        int nwitness = ccdStatus.Nx;
        for (int i = 0; i < nwitness; i++) {
            int c = at + i;
            con_dist[c] = margin + ccdStatus.Dist[i];
            for (int k = 0; k < 3; k++)
                con_pos[3 * c + k] = ccdStatus.X1[3 * i + k] + ccdStatus.X2[3 * i + k];
            for (int k = 0; k < 3; k++)
                con_pos[3 * c + k] = con_pos[3 * c + k] * 0.5;
            for (int k = 0; k < 3; k++)
                con_frame[9 * c + k] = ccdStatus.X1[3 * i + k] - ccdStatus.X2[3 * i + k];
            Spatial.Normalize3(con_frame.AsSpan(9 * c, 3));
            con_frame.AsSpan(9 * c + 3, 3).Clear();
        }
        return nwitness;
    }

    /// <summary>
    ///     Rotates a geom's frame about a point (<c>mju_rotateFrame</c>): its matrix becomes <c>rot*mat</c>, and its
    ///     position moves by the point's displacement under the rotation.
    /// </summary>
    /// <param name="origin">The point rotated about.</param>
    /// <param name="rot">The rotation.</param>
    /// <param name="xmat">The frame's matrix, updated.</param>
    /// <param name="xpos">The frame's position, updated.</param>
    private static void RotateFrame(ReadOnlySpan<double> origin, ReadOnlySpan<double> rot, Span<double> xmat, Span<double> xpos) {
        Span<double> mat = stackalloc double[9], vec = stackalloc double[3], rel = stackalloc double[3];
        Spatial.MulMatMat3(mat, rot, xmat);
        mat.CopyTo(xmat);
        for (int k = 0; k < 3; k++)
            rel[k] = origin[k] - xpos[k];
        Spatial.MulMatVec3(vec, rot, rel);
        for (int k = 0; k < 3; k++)
            vec[k] -= rel[k];
        for (int k = 0; k < 3; k++)
            xpos[k] -= vec[k];
    }

    /// <summary>
    ///     Whether the last of a pair's contacts lies farther than the tolerance from every earlier one
    ///     (<c>mjc_isDistinctContact</c>, by <c>mju_dist3</c>).
    /// </summary>
    /// <param name="first">The pair's first contact.</param>
    /// <param name="count">The contacts including the candidate (the last).</param>
    /// <param name="tolerance">The distance below which two contacts are the same.</param>
    /// <returns>Whether the candidate is new.</returns>
    private bool IsDistinctContact(int first, int count, double tolerance) {
        int last = first + count - 1;
        for (int i = first; i < last; i++) {
            double d0 = con_pos[3 * i] - con_pos[3 * last], d1 = con_pos[3 * i + 1] - con_pos[3 * last + 1];
            double d2 = con_pos[3 * i + 2] - con_pos[3 * last + 2];
            if (Math.Sqrt(d0 * d0 + d1 * d1 + d2 * d2) <= tolerance)
                return false;
        }
        return true;
    }

    /// <summary><c>mju_clip</c>: <paramref name="lo"/> below it, <paramref name="hi"/> above it, else the value.</summary>
    /// <param name="x">The value.</param>
    /// <param name="lo">The lower bound.</param>
    /// <param name="hi">The upper bound.</param>
    /// <returns>The clipped value.</returns>
    private static double Clip(double x, double lo, double hi) => x < lo ? lo : x > hi ? hi : x;

    /// <summary>
    ///     Sets the parameters of a pair's contacts (<c>mj_contactParam</c>, <c>mj_setContact</c>): priority or mixed
    ///     solref and solimp, the larger friction unpacked to 5 numbers, exclusion beyond the margin, and the completed
    ///     orthonormal frame.
    /// </summary>
    /// <param name="first">The pair's first contact.</param>
    /// <param name="last">One past its last contact.</param>
    /// <param name="g1">The plane geom.</param>
    /// <param name="g2">The capsule geom.</param>
    /// <param name="margin">The pair's margin (without gap), the distance below which a contact is active.</param>
    /// <exception cref="InvalidOperationException">A contact normal is degenerate (MuJoCo's error, which should not occur).</exception>
    private void ContactParams(int first, int last, int g1, int g2, double margin) {
        Span<double> solref = stackalloc double[2], solimp = stackalloc double[5], fri = stackalloc double[3];
        int condim;
        int p1 = m.geom_priority[g1], p2 = m.geom_priority[g2];
        if (p1 != p2) {
            int g = p1 > p2 ? g1 : g2;
            condim = m.geom_condim[g];
            m.geom_solref.AsSpan(2 * g, 2).CopyTo(solref);
            m.geom_solimp.AsSpan(5 * g, 5).CopyTo(solimp);
            m.geom_friction.AsSpan(3 * g, 3).CopyTo(fri);
        } else {
            condim = Math.Max(m.geom_condim[g1], m.geom_condim[g2]);
            double s1 = m.geom_solmix[g1], s2 = m.geom_solmix[g2];
            double mix = s1 >= Kernels.MinVal && s2 >= Kernels.MinVal ? s1 / (s1 + s2)
                       : s1 < Kernels.MinVal && s2 < Kernels.MinVal ? 0.5
                       : s1 < Kernels.MinVal ? 0.0 : 1.0;
            if (m.geom_solref[2 * g1] > 0 && m.geom_solref[2 * g2] > 0) {
                for (int i = 0; i < 2; i++)
                    solref[i] = mix * m.geom_solref[2 * g1 + i] + (1 - mix) * m.geom_solref[2 * g2 + i];
            } else {
                for (int i = 0; i < 2; i++)
                    solref[i] = Kernels.Min(m.geom_solref[2 * g1 + i], m.geom_solref[2 * g2 + i]);
            }
            for (int i = 0; i < 5; i++)
                solimp[i] = mix * m.geom_solimp[5 * g1 + i] + (1 - mix) * m.geom_solimp[5 * g2 + i];
            for (int i = 0; i < 3; i++)
                fri[i] = Kernels.Max(m.geom_friction[3 * g1 + i], m.geom_friction[3 * g2 + i]);
        }
        for (int c = first; c < last; c++) {
            con_geom1[c] = g1;
            con_geom2[c] = g2;
            con_dim[c] = condim;
            con_includemargin[c] = margin;
            solref.CopyTo(con_solref.AsSpan(2 * c));
            solimp.CopyTo(con_solimp.AsSpan(5 * c));
            Span<double> f = con_friction.AsSpan(5 * c, 5);
            f[0] = Kernels.Max(MinMu, fri[0]);
            f[1] = Kernels.Max(MinMu, fri[0]);
            f[2] = Kernels.Max(MinMu, fri[1]);
            f[3] = Kernels.Max(MinMu, fri[2]);
            f[4] = Kernels.Max(MinMu, fri[2]);
            con_exclude[c] = con_dist[c] >= margin ? 1 : 0;
            con_efc_address[c] = -1;
            con_mu[c] = 0;
            MakeFrame(con_frame.AsSpan(9 * c, 9));
        }
    }

    /// <summary>
    ///     Completes a contact frame from its normal and optional tangent (<c>mju_makeFrame</c>): normalize the normal,
    ///     pick a tangent if none, orthogonalize and normalize it, and take the cross product as the third axis.
    /// </summary>
    /// <param name="frame">The 9-number frame, normal first.</param>
    /// <exception cref="InvalidOperationException">The normal is shorter than 0.5.</exception>
    private static void MakeFrame(Span<double> frame) {
        if (Spatial.Normalize3(frame) < 0.5)
            throw new InvalidOperationException("xaxis of contact frame undefined");
        Span<double> y = frame.Slice(3, 3);
        if (Spatial.Dot3(y, y) < 0.25) {
            y.Clear();
            if (frame[1] < 0.5 && frame[1] > -0.5)
                frame[4] = 1;
            else
                frame[5] = 1;
        }
        double d = Spatial.Dot3(frame, y);
        Span<double> tmp = stackalloc double[3] { frame[0] * d, frame[1] * d, frame[2] * d };
        y[0] -= tmp[0];
        y[1] -= tmp[1];
        y[2] -= tmp[2];
        Spatial.Normalize3(y);
        Span<double> z = stackalloc double[3];
        Spatial.Cross(z, frame, y);
        z.CopyTo(frame.Slice(6));
    }

    // ================================================================================== constraints

    /// <summary>
    ///     Builds the constraint rows (<c>mj_makeConstraint</c>): joint limits that are violated within their margin, then
    ///     per active contact one frictionless row (condim 1) or four pyramid edges (condim 3); then their approximate
    ///     inverse inertia and impedance.
    /// </summary>
    private void MakeConstraint() {
        nefc = 0;
        for (int j = 0; j < m.njnt; j++) {
            if (!m.jnt_limited[j])
                continue;
            double value = qpos[m.jnt_qposadr[j]];
            for (int side = -1; side <= 1; side += 2) {
                double dist = side * (m.jnt_range[2 * j + (side + 1) / 2] - value);
                if (dist < m.jnt_margin[j]) {
                    Span<double> row = efc_J.AsSpan(nefc * nv, nv);
                    row.Clear();
                    row[m.jnt_dofadr[j]] = -(double)side;
                    AddRow(dist, m.jnt_margin[j], LimitJoint, j);
                }
            }
        }
        for (int c = 0; c < ncon; c++) {
            if (con_exclude[c] != 0)
                continue;
            con_efc_address[c] = nefc;
            int b1 = m.geom_bodyid[con_geom1[c]], b2 = m.geom_bodyid[con_geom2[c]];
            if (m.body_weldid[b1] == m.body_weldid[b2]) {
                con_efc_address[c] = -1;
                con_exclude[c] = 3;
                continue;
            }
            ReadOnlySpan<double> pos = con_pos.AsSpan(3 * c, 3);
            Jac(jac1, pos, b1);
            Jac(jac2, pos, b2);
            Kernels.Sub(jacdif, jac2, jac1, 3 * nv);
            int dim = con_dim[c];
            // a frictionless contact rotates only the normal row of the Jacobian difference
            Kernels.MulMatMat(jacrot, con_frame.AsSpan(9 * c, 9), jacdif, dim > 1 ? 3 : 1, 3, nv);
            if (dim == 1) {
                jacrot.AsSpan(0, nv).CopyTo(efc_J.AsSpan(nefc * nv, nv));
                AddRow(con_dist[c], con_includemargin[c], ContactFrictionless, c);
                continue;
            }
            for (int k = 1; k < dim; k++) {
                double f = con_friction[5 * c + k - 1];
                // the pair of opposing pyramid edges: normal row plus and minus friction times tangent row k
                Kernels.AddScl(efc_J.AsSpan(nefc * nv, nv), jacrot, jacrot.AsSpan(k * nv), f, nv);
                AddRow(con_dist[c], con_includemargin[c], ContactPyramidal, c);
                Kernels.AddScl(efc_J.AsSpan(nefc * nv, nv), jacrot, jacrot.AsSpan(k * nv), -f, nv);
                AddRow(con_dist[c], con_includemargin[c], ContactPyramidal, c);
            }
        }
        if (nefc == 0)
            return;
        DiagApprox();
        MakeImpedance();
    }

    /// <summary>Records the scalars of the row whose Jacobian was just written at index <see cref="nefc"/>.</summary>
    /// <param name="pos">Its position.</param>
    /// <param name="margin">Its margin.</param>
    /// <param name="type">Its constraint type.</param>
    /// <param name="id">Its joint or contact.</param>
    private void AddRow(double pos, double margin, int type, int id) {
        efc_pos[nefc] = pos;
        efc_margin[nefc] = margin;
        efc_frictionloss[nefc] = 0;
        efc_type[nefc] = type;
        efc_id[nefc] = id;
        nefc++;
    }

    /// <summary>
    ///     The translational Jacobian of a world point attached to a body (<c>mj_jac</c>, dense): each dof on the body's
    ///     ancestor chain contributes its linear axis plus its angular axis crossed with the offset from the subtree center
    ///     of mass.
    /// </summary>
    /// <param name="jacp">Receives the 3-by-nv Jacobian.</param>
    /// <param name="point">The world point.</param>
    /// <param name="body">The body.</param>
    private void Jac(Span<double> jacp, ReadOnlySpan<double> point, int body) {
        jacp.Slice(0, 3 * nv).Clear();
        int root = m.body_rootid[body];
        Span<double> offset = stackalloc double[3], tmp = stackalloc double[3];
        for (int k = 0; k < 3; k++)
            offset[k] = point[k] - subtree_com[3 * root + k];
        body = m.body_weldid[body];
        if (m.body_dofnum[body] == 0)
            return;
        for (int i = m.body_dofadr[body] + m.body_dofnum[body] - 1; i >= 0; i = m.dof_parentid[i]) {
            ReadOnlySpan<double> dof = cdof.AsSpan(6 * i, 6);
            Spatial.Cross(tmp, dof, offset);
            jacp[i] = dof[3] + tmp[0];
            jacp[i + nv] = dof[4] + tmp[1];
            jacp[i + 2 * nv] = dof[5] + tmp[2];
        }
    }

    /// <summary>
    ///     The approximate inverse inertia of each row (<c>mj_diagApprox</c>): a limit takes its dof's reference weight; a
    ///     frictionless contact the two bodies' translational weight; a pyramidal contact's edges that weight scaled by
    ///     <c>1 + friction²</c>.
    /// </summary>
    private void DiagApprox() {
        for (int i = 0; i < nefc; i++) {
            int id = efc_id[i];
            if (efc_type[i] == LimitJoint) {
                efc_diagA[i] = m.dof_invweight0[m.jnt_dofadr[id]];
                continue;
            }
            // both sides' weights, each times its body weight 1, added in side order from 0
            int b1 = m.geom_bodyid[con_geom1[id]], b2 = m.geom_bodyid[con_geom2[id]];
            double tran = 0, rot = 0;
            tran += m.body_invweight0[2 * b1] * 1.0;
            rot += m.body_invweight0[2 * b1 + 1] * 1.0;
            tran += m.body_invweight0[2 * b2] * 1.0;
            rot += m.body_invweight0[2 * b2 + 1] * 1.0;
            if (efc_type[i] == ContactFrictionless) {
                efc_diagA[i] = tran;
                continue;
            }
            int dim = con_dim[id];
            for (int j = 0; j < dim - 1; j++) {
                double fri = con_friction[5 * id + j];
                efc_diagA[i + 2 * j] = efc_diagA[i + 2 * j + 1] = tran + fri * fri * (j < 2 ? tran : rot);
            }
            i += 2 * dim - 3;
        }
    }

    /// <summary>
    ///     Impedance, stiffness, damping and regularization of every row (<c>mj_makeImpedance</c>), then the pyramid's
    ///     common regularizer matching the elliptic cone, <c>D = 1/R</c>, and <c>diagA</c> re-derived from R.
    /// </summary>
    private void MakeImpedance() {
        Span<double> solref = stackalloc double[2], solimp = stackalloc double[5];
        for (int i = 0; i < nefc; i++) {
            int id = efc_id[i], dim = 1;
            if (efc_type[i] == LimitJoint) {
                m.jnt_solref.AsSpan(2 * id, 2).CopyTo(solref);
                m.jnt_solimp.AsSpan(5 * id, 5).CopyTo(solimp);
            } else {
                con_solref.AsSpan(2 * id, 2).CopyTo(solref);
                con_solimp.AsSpan(5 * id, 5).CopyTo(solimp);
                // getposdim: a pyramid's rows share one impedance, a frictionless contact has its one row
                dim = efc_type[i] == ContactPyramidal ? 2 * (con_dim[id] - 1) : 1;
            }
            if ((solref[0] > 0) ^ (solref[1] > 0)) {
                solref[0] = 0.02;
                solref[1] = 1;
            }
            // integrator safety: a time constant shorter than two steps is lengthened
            if (solref[0] > 0)
                solref[0] = Kernels.Max(solref[0], 2 * m.timestep);
            solimp[0] = Kernels.Min(MaxImp, Kernels.Max(MinImp, solimp[0]));
            solimp[1] = Kernels.Min(MaxImp, Kernels.Max(MinImp, solimp[1]));
            solimp[2] = Kernels.Max(0, solimp[2]);
            solimp[3] = Kernels.Min(MaxImp, Kernels.Max(MinImp, solimp[3]));
            solimp[4] = Kernels.Max(1, solimp[4]);
            Impedance(solimp, efc_pos[i], efc_margin[i], out double imp, out double impP);
            for (int j = 0; j < dim; j++) {
                efc_R[i + j] = Kernels.Max(Kernels.MinVal, (1 - imp) * efc_diagA[i + j] / imp);
                Span<double> kbip = efc_KBIP.AsSpan(4 * (i + j), 4);
                kbip[0] = solref[0] > 0
                    ? 1 / Kernels.Max(Kernels.MinVal, solimp[1] * solimp[1] * solref[0] * solref[0] * solref[1] * solref[1])
                    : -solref[0] / Kernels.Max(Kernels.MinVal, solimp[1] * solimp[1]);
                kbip[1] = solref[1] > 0
                    ? 2 / Kernels.Max(Kernels.MinVal, solimp[1] * solref[0])
                    : -solref[1] / Kernels.Max(Kernels.MinVal, solimp[1]);
                kbip[2] = imp;
                kbip[3] = impP;
            }
            i += dim - 1;
        }
        for (int i = 0; i < nefc; i++) {
            if (efc_type[i] != ContactPyramidal)
                continue;
            int id = efc_id[i], dim = con_dim[id];
            efc_R[i + 1] = efc_R[i] / Kernels.Max(Kernels.MinVal, m.impratio);
            con_mu[id] = con_friction[5 * id] * Math.Sqrt(efc_R[i + 1] / efc_R[i]);
            double rpy = 2 * con_mu[id] * con_mu[id] * efc_R[i];
            for (int j = 0; j < 2 * (dim - 1); j++)
                efc_R[i + j] = rpy;
            i += 2 * (dim - 1) - 1;
        }
        for (int i = 0; i < nefc; i++)
            efc_D[i] = 1 / efc_R[i];
        for (int i = 0; i < nefc; i++)
            efc_diagA[i] = efc_R[i] * efc_KBIP[4 * i + 2] / (1 - efc_KBIP[4 * i + 2]);
    }

    /// <summary>
    ///     A row's impedance and its derivative from its violation (<c>getimpedance</c>): flat, saturated, or MuJoCo's
    ///     two-piece polynomial between <c>solimp[0]</c> and <c>solimp[1]</c> over the width <c>solimp[2]</c>.
    /// </summary>
    /// <param name="solimp">The clamped solimp.</param>
    /// <param name="pos">The row position.</param>
    /// <param name="margin">The row margin.</param>
    /// <param name="imp">The impedance.</param>
    /// <param name="impP">Its derivative with respect to the position.</param>
    private static void Impedance(ReadOnlySpan<double> solimp, double pos, double margin, out double imp, out double impP) {
        if (solimp[0] == solimp[1] || solimp[2] <= Kernels.MinVal) {
            imp = 0.5 * (solimp[0] + solimp[1]);
            impP = 0;
            return;
        }
        double x = (pos - margin) / solimp[2], sgn = 1;
        if (x < 0) {
            x = -x;
            sgn = -1;
        }
        if (x >= 1 || x <= 0) {
            imp = x >= 1 ? solimp[1] : solimp[0];
            impP = 0;
            return;
        }
        double y, yP;
        if (solimp[4] == 1) {
            y = x;
            yP = 1;
        } else if (x <= solimp[3]) {
            double a = 1 / Power(solimp[3], solimp[4] - 1);
            y = a * Power(x, solimp[4]);
            yP = solimp[4] * a * Power(x, solimp[4] - 1);
        } else {
            double b = 1 / Power(1 - solimp[3], solimp[4] - 1);
            y = 1 - b * Power(1 - x, solimp[4]);
            yP = solimp[4] * b * Power(1 - x, solimp[4] - 1);
        }
        imp = solimp[0] + y * (solimp[1] - solimp[0]);
        impP = yP * sgn * (solimp[1] - solimp[0]) / solimp[2];
    }

    /// <summary><c>a^b</c> with MuJoCo's exact shortcuts for the powers 1 and 2.</summary>
    /// <param name="a">The base.</param>
    /// <param name="b">The exponent.</param>
    /// <returns>The power.</returns>
    private static double Power(double a, double b) => b == 1 ? a : b == 2 ? a * a : Math.Pow(a, b);

    // ================================================================================== velocity, forces

    /// <summary>
    ///     Body spatial velocities and the dof axis derivatives (<c>mj_comVel</c>): each body starts from its parent's
    ///     velocity and adds its dofs one by one, each dof's derivative taken against the velocity before it. A free
    ///     joint adds its three translations at once (their derivatives zero), then derives its three rotations against
    ///     that velocity and adds them at once.
    /// </summary>
    private void ComVel() {
        Array.Clear(cvel, 0, 6);
        Span<double> cv = stackalloc double[6], tmp = stackalloc double[6];
        for (int i = 1; i < nbody; i++) {
            cvel.AsSpan(6 * m.body_parentid[i], 6).CopyTo(cv);
            int bda = m.body_dofadr[i];
            for (int j = 0; j < m.body_dofnum[i]; j++) {
                int d = bda + j;
                if (m.jnt_type[m.dof_jntid[d]] == Free) {
                    cdof_dot.AsSpan(6 * d, 18).Clear();
                    Kernels.MulMatTVec(tmp, cdof.AsSpan(6 * d), qvel.AsSpan(d), 3, 6);
                    Kernels.AddTo(cv, tmp, 6);
                    j += 3;
                    d = bda + j;
                    for (int k = 0; k < 3; k++)
                        Spatial.CrossMotion(cdof_dot.AsSpan(6 * (d + k)), cv, cdof.AsSpan(6 * (d + k)));
                    Kernels.MulMatTVec(tmp, cdof.AsSpan(6 * d), qvel.AsSpan(d), 3, 6);
                    Kernels.AddTo(cv, tmp, 6);
                    j += 2;
                    continue;
                }
                Spatial.CrossMotion(cdof_dot.AsSpan(6 * d), cv, cdof.AsSpan(6 * d));
                Kernels.Scl(tmp, cdof.AsSpan(6 * d), qvel[d], 6);
                Kernels.AddTo(cv, tmp, 6);
            }
            cv.CopyTo(cvel.AsSpan(6 * i));
        }
    }

    /// <summary>
    ///     Passive forces (<c>mj_passive</c>): joint springs and dof dampers summed into <see cref="qfrc_passive"/>, then the
    ///     fluid forces added when the model has a fluid (gravity compensation, contact and adhesion forces do not apply).
    /// </summary>
    private void Passive() {
        Array.Clear(qfrc_spring);
        Array.Clear(qfrc_damper);
        for (int b = 0; b < nbody; b++) {
            int jend = m.body_jntadr[b] + m.body_jntnum[b];
            for (int j = m.body_jntadr[b]; j < jend; j++) {
                double k = m.jnt_stiffness[j];
                ReadOnlySpan<double> spoly = m.jnt_stiffnesspoly.AsSpan(2 * j, 2);
                if (k == 0 && spoly[0] == 0 && spoly[1] == 0)
                    continue;
                int padr = m.jnt_qposadr[j];
                double x = qpos[padr] - m.qpos_spring[padr];
                qfrc_spring[m.jnt_dofadr[j]] = -x * PolyForce(k, spoly, x, false);
            }
        }
        for (int i = 0; i < nv; i++) {
            ReadOnlySpan<double> poly = m.dof_dampingpoly.AsSpan(2 * i, 2);
            // + 0: the damping actuators would add, none here
            double damping = m.dof_damping[i] + 0.0;
            if (damping != 0 || poly[0] != 0 || poly[1] != 0) {
                double v = qvel[i];
                qfrc_damper[i] = -v * PolyForce(damping, poly, v, true);
            }
        }
        Kernels.Add(qfrc_passive, qfrc_spring, qfrc_damper, nv);
        // mj_fluid returns early without a fluid, and only then is qfrc_fluid added
        if (m.viscosity != 0 || m.density != 0) {
            Fluid();
            Kernels.AddTo(qfrc_passive, qfrc_fluid, nv);
        }
    }

    /// <summary>
    ///     Fluid forces by the inertia-box model (<c>mj_fluid</c>, <c>mj_inertiaBoxFluidModel</c>): each body with mass is
    ///     the box of its principal inertia, moving at its local velocity through the fluid. Viscosity adds Stokes-like terms
    ///     on the equivalent sphere; density adds quadratic drag on the box's faces and its rotation; the force and torque,
    ///     rotated to the world, reach the dofs through the body's Jacobians at its centre of mass (<c>mj_applyFT</c>).
    /// </summary>
    /// <remarks>
    ///     The wind is transformed to the body frame even when it is zero, as MuJoCo does: the transform's zeros carry
    ///     signs, and subtracting a -0 turns a -0 velocity into +0. Every product follows MuJoCo's left-to-right order.
    /// </remarks>
    private void Fluid() {
        Array.Clear(qfrc_fluid);
        Span<double> lvel = stackalloc double[6], wind = stackalloc double[6], lwind = stackalloc double[6];
        Span<double> lfrc = stackalloc double[6], bfrc = stackalloc double[6], box = stackalloc double[3];
        for (int i = 0; i < nbody; i++) {
            if (m.body_mass[i] < Kernels.MinVal)
                continue;
            ReadOnlySpan<double> inertia = m.body_inertia.AsSpan(3 * i, 3);
            double mass = m.body_mass[i];
            // mju_max keeps its first argument on a tie
            box[0] = Math.Sqrt(MaxFirst(Kernels.MinVal, inertia[1] + inertia[2] - inertia[0]) / mass * 6.0);
            box[1] = Math.Sqrt(MaxFirst(Kernels.MinVal, inertia[0] + inertia[2] - inertia[1]) / mass * 6.0);
            box[2] = Math.Sqrt(MaxFirst(Kernels.MinVal, inertia[0] + inertia[1] - inertia[2]) / mass * 6.0);
            ObjectVelocityLocal(lvel, i);
            wind.Clear();
            m.wind.AsSpan(0, 3).CopyTo(wind.Slice(3));
            TransformSpatialMotion(lwind, wind, xipos.AsSpan(3 * i, 3), subtree_com.AsSpan(3 * m.body_rootid[i], 3), ximat.AsSpan(9 * i, 9));
            for (int k = 3; k < 6; k++)
                lvel[k] -= lwind[k];
            lfrc.Clear();
            if (m.viscosity > 0) {
                double diam = (box[0] + box[1] + box[2]) / 3.0;
                double angular = -Math.PI * diam * diam * diam * m.viscosity;
                double linear = -3.0 * Math.PI * diam * m.viscosity;
                for (int k = 0; k < 3; k++) {
                    lfrc[k] = lvel[k] * angular;
                    lfrc[3 + k] = lvel[3 + k] * linear;
                }
            }
            if (m.density > 0) {
                double rho = m.density;
                lfrc[3] -= 0.5 * rho * box[1] * box[2] * Math.Abs(lvel[3]) * lvel[3];
                lfrc[4] -= 0.5 * rho * box[0] * box[2] * Math.Abs(lvel[4]) * lvel[4];
                lfrc[5] -= 0.5 * rho * box[0] * box[1] * Math.Abs(lvel[5]) * lvel[5];
                lfrc[0] -= rho * box[0] * (box[1] * box[1] * box[1] * box[1] + box[2] * box[2] * box[2] * box[2]) * Math.Abs(lvel[0]) * lvel[0] / 64.0;
                lfrc[1] -= rho * box[1] * (box[0] * box[0] * box[0] * box[0] + box[2] * box[2] * box[2] * box[2]) * Math.Abs(lvel[1]) * lvel[1] / 64.0;
                lfrc[2] -= rho * box[2] * (box[0] * box[0] * box[0] * box[0] + box[1] * box[1] * box[1] * box[1]) * Math.Abs(lvel[2]) * lvel[2] / 64.0;
            }
            Spatial.MulMatVec3(bfrc, ximat.AsSpan(9 * i, 9), lfrc);
            Spatial.MulMatVec3(bfrc.Slice(3), ximat.AsSpan(9 * i, 9), lfrc.Slice(3));
            ApplyFT(bfrc.Slice(3, 3), bfrc.Slice(0, 3), xipos.AsSpan(3 * i, 3), i, qfrc_fluid);
        }
    }

    /// <summary><c>mju_max</c>: the first argument when it is not below the second.</summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The larger, <paramref name="a"/> on a tie.</returns>
    private static double MaxFirst(double a, double b) => a >= b ? a : b;

    /// <summary>
    ///     A body's 6-D velocity at its centre of mass in its inertial frame (<c>mj_objectVelocity</c> for a body, local):
    ///     its <see cref="cvel"/> moved from the root's subtree centre of mass and rotated; zero for a body welded to the
    ///     world.
    /// </summary>
    /// <param name="res">Receives the angular then linear velocity.</param>
    /// <param name="body">The body.</param>
    private void ObjectVelocityLocal(Span<double> res, int body) {
        if (m.body_dofnum[m.body_weldid[body]] == 0) {
            res.Slice(0, 6).Clear();
            return;
        }
        TransformSpatialMotion(res, cvel.AsSpan(6 * body, 6), xipos.AsSpan(3 * body, 3),
                               subtree_com.AsSpan(3 * m.body_rootid[body], 3), ximat.AsSpan(9 * body, 9));
    }

    /// <summary>
    ///     Moves a motion vector (angular, linear) to a new point and rotates it into a frame (<c>mju_transformSpatial</c>
    ///     with <c>flg_force=0</c>): the linear part loses <c>(newpos - oldpos) x angular</c>, then both parts are
    ///     multiplied by the transpose of <paramref name="rot"/>.
    /// </summary>
    /// <param name="res">Receives the transformed vector; must not alias <paramref name="vec"/>.</param>
    /// <param name="vec">The motion vector.</param>
    /// <param name="newpos">The new reference point.</param>
    /// <param name="oldpos">The old reference point.</param>
    /// <param name="rot">The rotation from the new frame to the old (row-major).</param>
    private static void TransformSpatialMotion(Span<double> res, ReadOnlySpan<double> vec, ReadOnlySpan<double> newpos,
                                               ReadOnlySpan<double> oldpos, ReadOnlySpan<double> rot) {
        Span<double> tran = stackalloc double[6], dif = stackalloc double[3], cros = stackalloc double[3];
        vec.Slice(0, 6).CopyTo(tran);
        for (int k = 0; k < 3; k++)
            dif[k] = newpos[k] - oldpos[k];
        Spatial.Cross(cros, dif, vec);
        for (int k = 0; k < 3; k++)
            tran[3 + k] = vec[3 + k] - cros[k];
        Spatial.MulMatTVec3(res, rot, tran);
        Spatial.MulMatTVec3(res.Slice(3), rot, tran.Slice(3));
    }

    /// <summary>
    ///     Adds a force and a torque applied at a point of a body to generalized forces (<c>mj_applyFT</c>, dense): the
    ///     transposed translational Jacobian times the force, then the transposed rotational Jacobian times the torque, each
    ///     added in turn.
    /// </summary>
    /// <param name="force">The world-frame force.</param>
    /// <param name="torque">The world-frame torque.</param>
    /// <param name="point">The application point.</param>
    /// <param name="body">The body.</param>
    /// <param name="target">The generalized force vector added to.</param>
    private void ApplyFT(ReadOnlySpan<double> force, ReadOnlySpan<double> torque, ReadOnlySpan<double> point, int body, Span<double> target) {
        Jac(fluidJacp, point, body);
        JacRot(fluidJacr, body);
        Kernels.MulMatTVec(fluidQforce, fluidJacp, force, 3, nv);
        Kernels.AddTo(target, fluidQforce, nv);
        Kernels.MulMatTVec(fluidQforce, fluidJacr, torque, 3, nv);
        Kernels.AddTo(target, fluidQforce, nv);
    }

    /// <summary>
    ///     The rotational Jacobian of a body (<c>mj_jac</c>'s <c>jacr</c>): the angular part of each dof's motion axis on
    ///     the body's chain, zero elsewhere.
    /// </summary>
    /// <param name="jacr">Receives the 3 x nv Jacobian.</param>
    /// <param name="body">The body.</param>
    private void JacRot(Span<double> jacr, int body) {
        jacr.Slice(0, 3 * nv).Clear();
        body = m.body_weldid[body];
        if (m.body_dofnum[body] == 0)
            return;
        for (int i = m.body_dofadr[body] + m.body_dofnum[body] - 1; i >= 0; i = m.dof_parentid[i]) {
            jacr[i] = cdof[6 * i];
            jacr[i + nv] = cdof[6 * i + 1];
            jacr[i + 2 * nv] = cdof[6 * i + 2];
        }
    }

    /// <summary>A linear coefficient plus polynomial terms (<c>mju_polyForce</c>), of |x| for odd (damping) laws.</summary>
    /// <param name="linear">The linear coefficient.</param>
    /// <param name="poly">The higher-order coefficients.</param>
    /// <param name="x">The displacement or velocity.</param>
    /// <param name="odd">Whether to use |x|.</param>
    /// <returns>The effective coefficient.</returns>
    private static double PolyForce(double linear, ReadOnlySpan<double> poly, double x, bool odd) {
        x = odd ? Math.Abs(x) : x;
        double res = linear, xpow = 1;
        for (int i = 0; i < poly.Length; i++) {
            xpow *= x;
            res += poly[i] * xpow;
        }
        return res;
    }

    /// <summary>The derivative of <c>x*PolyForce(x)</c> (<c>mjd_xPolyForce</c>), the Euler step's damping slope.</summary>
    /// <param name="linear">The linear coefficient.</param>
    /// <param name="poly">The higher-order coefficients.</param>
    /// <param name="x">The velocity.</param>
    /// <returns>The slope.</returns>
    private static double XPolyForce(double linear, ReadOnlySpan<double> poly, double x) {
        x = Math.Abs(x);
        double res = linear, xpow = 1;
        for (int i = 0; i < poly.Length; i++) {
            xpow *= x;
            res += (i + 2) * poly[i] * xpow;
        }
        return res;
    }

    /// <summary>
    ///     The bias forces by recursive Newton-Euler without acceleration (<c>mj_rne</c> with <c>flg_acc=0</c>): gravity as
    ///     the world's acceleration, body forces from inertia and velocity, accumulated to the root and projected on the dofs.
    /// </summary>
    private void Rne() {
        Array.Clear(cacc, 0, 6);
        for (int k = 0; k < 3; k++)
            cacc[3 + k] = m.gravity[k] * -1;
        Span<double> tmp = stackalloc double[6], tmp1 = stackalloc double[6];
        for (int i = 1; i < nbody; i++) {
            int bda = m.body_dofadr[i], n = m.body_dofnum[i];
            // mju_mulDofVec: one dof scales its axis (no zero test); several sum their scaled axes, skipping zero velocities
            if (n == 1) {
                Kernels.Scl(tmp, cdof_dot.AsSpan(6 * bda), qvel[bda], 6);
            } else if (n <= 0) {
                tmp.Clear();
            } else {
                Kernels.MulMatTVec(tmp, cdof_dot.AsSpan(6 * bda), qvel.AsSpan(bda), n, 6);
            }
            Kernels.Add(cacc.AsSpan(6 * i), cacc.AsSpan(6 * m.body_parentid[i]), tmp, 6);
            Spatial.MulInertVec(cfrc.AsSpan(6 * i), cinert.AsSpan(10 * i), cacc.AsSpan(6 * i));
            Spatial.MulInertVec(tmp, cinert.AsSpan(10 * i), cvel.AsSpan(6 * i));
            Spatial.CrossForce(tmp1, cvel.AsSpan(6 * i), tmp);
            Kernels.AddTo(cfrc.AsSpan(6 * i), tmp1, 6);
        }
        Array.Clear(cfrc, 0, 6);
        for (int i = nbody - 1; i > 0; i--) {
            int j = m.body_parentid[i];
            if (j != 0)
                Kernels.AddTo(cfrc.AsSpan(6 * j), cfrc.AsSpan(6 * i), 6);
        }
        for (int v = 0; v < nv; v++)
            qfrc_bias[v] = Spatial.Dot6(cdof.AsSpan(6 * v), cfrc.AsSpan(6 * m.dof_bodyid[v]));
    }

    /// <summary>
    ///     Motor forces (<c>mj_fwdActuation</c> for motors): controls clamped to their ranges (all zeroed if any is bad),
    ///     force = gain * control + 0 bias, mapped to the joints through the gear.
    /// </summary>
    private void Actuation() {
        Span<double> c = stackalloc double[nu];
        ctrl.AsSpan(0, nu).CopyTo(c);
        for (int i = 0; i < nu; i++)
            if (m.actuator_ctrllimited[i]) {
                double lo = m.actuator_ctrlrange[2 * i], hi = m.actuator_ctrlrange[2 * i + 1];
                c[i] = c[i] < lo ? lo : c[i] > hi ? hi : c[i];
            }
        for (int i = 0; i < nu; i++)
            if (double.IsNaN(c[i]) || c[i] > MaxVal || c[i] < -MaxVal) {
                c.Clear();
                break;
            }
        for (int i = 0; i < nu; i++) {
            actuator_force[i] = m.actuator_gainprm[10 * i] * c[i];
            actuator_force[i] += 0.0;
        }
        Array.Clear(qfrc_actuator);
        for (int i = 0; i < nu; i++) {
            double scl = actuator_force[i];
            if (scl == 0)
                continue;
            qfrc_actuator[m.jnt_dofadr[m.actuator_trnid[2 * i]]] += actuator_moment[i] * scl;
        }
    }

    // ================================================================================== constraint solver

    /// <summary>
    ///     The constrained acceleration (<c>mj_fwdConstraint</c>): the unconstrained residual, the warm start over every
    ///     row and dof, then the model's solver. Newton: the warm start chosen by primal cost, dofs outside islands reset
    ///     to their unconstrained acceleration, then each island's primal Newton iterations on gathered copies. PGS (one
    ///     tree, so one island holding everything): the warm start's forces kept unless their dual cost is positive, the
    ///     sweeps, then the acceleration from the forces (<c>dualFinish</c>).
    /// </summary>
    private void FwdConstraint() {
        Array.Clear(qfrc_constraint);
        if (nefc == 0) {
            Array.Copy(qacc_smooth, qacc, nv);
            solver_niter = 0;
            return;
        }
        Kernels.MulMatVec(efc_b, efc_J, qacc_smooth, nefc, nv);
        Kernels.SubFrom(efc_b, efc_aref, nefc);
        // warm start: the previous step's acceleration unless the unconstrained one is cheaper
        Array.Copy(qacc_warmstart, qacc, nv);
        Kernels.MulMatVec(jar, efc_J, qacc_warmstart, nefc, nv);
        Kernels.SubFrom(jar, efc_aref, nefc);
        double costWarm = ConstraintUpdate(jar, efc_force, efc_state, efc_D, nefc);
        Kernels.MulMatTVec(qfrc_constraint, efc_J, efc_force, nefc, nv);
        if (m.solver == SolverPgs) {
            // the dual cost of the warm start's forces, f'*b + 0.5*f'*AR*f: positive means worse than no force
            double pgsWarm = Kernels.Dot(efc_force, efc_b, nefc);
            Kernels.MulMatVec(arForce, efc_AR, efc_force, nefc, nefc);
            pgsWarm += 0.5 * Kernels.Dot(efc_force, arForce, nefc);
            if (pgsWarm > 0) {
                Array.Clear(efc_force, 0, nefc);
                Array.Clear(qfrc_constraint);
            }
            solver_niter = 0;
            SolvePgs();
            // dualFinish: qfrc_constraint = J'*f, qacc = M^-1*qfrc_constraint + qacc_smooth
            Kernels.MulMatTVec(qfrc_constraint, efc_J, efc_force, nefc, nv);
            Array.Copy(qfrc_constraint, qacc, nv);
            Kernels.SolveLD(qacc, qLD, qLDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
            Kernels.AddTo(qacc, qacc_smooth, nv);
            return;
        }
        Kernels.Sub(da, qacc_warmstart, qacc_smooth, nv);
        Kernels.MulSymVecSparse(Mda, M, da, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        costWarm += 0.5 * Kernels.Dot(da, Mda, nv);
        // this second evaluation is what MuJoCo leaves in efc_state (and, until the solve, efc_force)
        double costSmooth = ConstraintUpdate(efc_b, efc_force, efc_state, efc_D, nefc);
        Kernels.MulMatTVec(qfrc_constraint, efc_J, efc_force, nefc, nv);
        if (costWarm > costSmooth)
            Array.Copy(qacc_smooth, qacc, nv);
        // a dof no island holds keeps its unconstrained acceleration (the end of MuJoCo's warmstart)
        for (int i = nidof; i < nv; i++) {
            int dof = mapIdof2dof[i];
            qacc[dof] = qacc_smooth[dof];
        }
        // each island alone, on gathered copies; the reported iterations are the first island's (solver_niter[0])
        solver_niter = 0;
        for (int island = 0; island < nisland; island++) {
            GatherIsland(island);
            int iter = SolveNewton();
            if (island == 0)
                solver_niter += iter;
            ScatterIsland(island);
        }
    }

    // ================================================================================== islands

    /// <summary>
    ///     Constraint islands (<c>mj_island</c>): a union-find over kinematic trees joins the trees each constraint touches
    ///     (a joint limit its joint's tree, a contact the trees of its two geoms, the world none); islands are numbered by
    ///     their smallest tree, an island's dofs and rows keep model order, and the dofs of trees without rows come last.
    /// </summary>
    /// <remarks>
    ///     Consecutive rows of one constraint share its tree. With one tree holding every dof and row, the maps are the
    ///     identity and the island solve does the monolithic arithmetic exactly.
    /// </remarks>
    /// <exception cref="InvalidOperationException">A constraint between two static bodies (MuJoCo's error; cannot occur here).</exception>
    private void Islands() {
        nisland = 0;
        nidof = 0;
        if (nefc == 0)
            return;
        Array.Fill(treeParent, -1);
        for (int i = 0; i < nefc; i++) {
            if (i > 0 && efc_type[i] == efc_type[i - 1] && efc_id[i] == efc_id[i - 1]) {
                efcTree[i] = efcTree[i - 1];
                continue;
            }
            int tree1, tree2;
            if (efc_type[i] == LimitJoint) {
                tree1 = m.dof_treeid[m.jnt_dofadr[efc_id[i]]];
                tree2 = -2;
            } else {
                tree1 = m.body_treeid[m.geom_bodyid[con_geom1[efc_id[i]]]];
                tree2 = m.body_treeid[m.geom_bodyid[con_geom2[efc_id[i]]]];
            }
            efcTree[i] = tree1 >= 0 ? tree1 : tree2;
            if (efcTree[i] < 0)
                throw new InvalidOperationException($"constraint {i} is between two static bodies");
            DsuMerge(tree1, tree2 == -2 ? -1 : tree2);
        }
        // mj_dsuAssign: ascending trees; a root opens an island, a member joins its (already compressed) root's
        for (int t = 0; t < m.ntree; t++) {
            if (treeParent[t] == -1) {
                treeIsland[t] = -1;
                continue;
            }
            if (treeParent[t] == t) {
                treeIsland[t] = nisland++;
            } else {
                treeParent[t] = treeParent[treeParent[t]];
                treeIsland[t] = treeIsland[treeParent[t]];
            }
            nidof += m.tree_dofnum[t];
        }
        // dofs: counts, addresses, then the maps in dof order (the unconstrained ones after nidof)
        Array.Clear(islandNv, 0, nisland);
        for (int i = 0; i < nv; i++) {
            int isl = treeIsland[m.dof_treeid[i]];
            if (isl >= 0)
                islandNv[isl]++;
        }
        islandIdofadr[0] = 0;
        for (int i = 1; i < nisland; i++)
            islandIdofadr[i] = islandIdofadr[i - 1] + islandNv[i - 1];
        Array.Clear(islandFill, 0, nisland + 1);
        for (int dof = 0; dof < nv; dof++) {
            int isl = treeIsland[m.dof_treeid[dof]];
            int idof = isl >= 0 ? islandIdofadr[isl] + islandFill[isl]++ : nidof + islandFill[nisland]++;
            mapDof2idof[dof] = idof;
            mapIdof2dof[idof] = dof;
        }
        // rows: counts, addresses, then the map in row order
        Array.Clear(islandNefc, 0, nisland);
        for (int i = 0; i < nefc; i++)
            islandNefc[treeIsland[efcTree[i]]]++;
        islandIefcadr[0] = 0;
        for (int i = 1; i < nisland; i++)
            islandIefcadr[i] = islandIefcadr[i - 1] + islandNefc[i - 1];
        Array.Clear(islandFill, 0, nisland);
        for (int c = 0; c < nefc; c++) {
            int isl = treeIsland[efcTree[c]];
            mapIefc2efc[islandIefcadr[isl] + islandFill[isl]++] = c;
        }
    }

    /// <summary>
    ///     Joins the trees of a constraint (<c>mj_dsuMerge</c>): -1 stands for the world, so a constraint with it activates
    ///     the other tree alone; two roots are linked under the smaller.
    /// </summary>
    /// <param name="tree1">The first tree, or -1.</param>
    /// <param name="tree2">The second tree, or -1.</param>
    private void DsuMerge(int tree1, int tree2) {
        if (tree1 == -1)
            tree1 = tree2;
        if (tree2 == -1)
            tree2 = tree1;
        if (treeParent[tree1] == -1)
            treeParent[tree1] = tree1;
        if (treeParent[tree2] == -1)
            treeParent[tree2] = tree2;
        // MuJoCo compares the parents, not the roots, before finding the roots (a shortcut that only skips work)
        if (treeParent[tree1] == treeParent[tree2])
            return;
        int root1 = DsuRoot(tree1), root2 = DsuRoot(tree2);
        if (root1 < root2)
            treeParent[root2] = root1;
        else if (root2 < root1)
            treeParent[root1] = root2;
    }

    /// <summary>The root of a tree's set, compressing the path to it (<c>mj_dsuRoot</c>).</summary>
    /// <param name="tree">The tree.</param>
    /// <returns>The root tree.</returns>
    private int DsuRoot(int tree) {
        int root = tree;
        while (treeParent[root] != root)
            root = treeParent[root];
        while (treeParent[tree] != tree) {
            int next = treeParent[tree];
            treeParent[tree] = root;
            tree = next;
        }
        return root;
    }

    /// <summary>
    ///     Gathers an island's problem into the solve context (<c>fwdConstraint</c>'s gathers, <c>PrimalAllocate</c>'s
    ///     blocks): its dof vectors, its rows' force, reference and impedance, the island blocks of <see cref="M"/> and of
    ///     its factor (block-diagonal by tree, so the factor's block is the block's factor) with columns renumbered
    ///     within the island, and the dense Jacobian block of its rows and dofs.
    /// </summary>
    /// <param name="island">The island.</param>
    private void GatherIsland(int island) {
        snv = islandNv[island];
        snefc = islandNefc[island];
        int idofadr = islandIdofadr[island], iefcadr = islandIefcadr[island];
        for (int i = 0; i < snv; i++) {
            int dof = mapIdof2dof[idofadr + i];
            iQfrcSmooth[i] = qfrc_smooth[dof];
            iQfrcConstraint[i] = qfrc_constraint[dof];
            iQaccSmooth[i] = qacc_smooth[dof];
            iQacc[i] = qacc[dof];
            iQLDiagInv[i] = qLDiagInv[dof];
        }
        for (int r = 0; r < snefc; r++) {
            int c = mapIefc2efc[iefcadr + r];
            iForce[r] = efc_force[c];
            iAref[r] = efc_aref[c];
            iD[r] = efc_D[c];
        }
        // mju_blockSparse: each island row keeps all its stored entries (its ancestors share its tree)
        for (int r = 0; r < snv; r++) {
            int k = mapIdof2dof[idofadr + r], nnz = m.M_rownnz[k], madr = m.M_rowadr[k];
            int adr = r == 0 ? 0 : iMrowadr[r - 1] + iMrownnz[r - 1];
            iMrownnz[r] = nnz;
            iMrowadr[r] = adr;
            for (int j = 0; j < nnz; j++) {
                iMcolind[adr + j] = mapDof2idof[m.M_colind[madr + j]] - idofadr;
                iQLD[adr + j] = qLD[madr + j];
                iM[adr + j] = M[madr + j];
            }
        }
        // mju_block: rows in island order, columns the island's dofs
        for (int r = 0; r < snefc; r++) {
            int c = mapIefc2efc[iefcadr + r];
            for (int j = 0; j < snv; j++)
                iJ[r * snv + j] = efc_J[c * nv + mapIdof2dof[idofadr + j]];
        }
    }

    /// <summary>
    ///     Writes an island's solution back (<c>fwdConstraint</c>'s scatters): its accelerations and constraint forces in
    ///     joint space, and its rows' forces. The solver's row states stay its own: <see cref="efc_state"/> keeps the
    ///     warm start's.
    /// </summary>
    /// <param name="island">The island.</param>
    private void ScatterIsland(int island) {
        int idofadr = islandIdofadr[island], iefcadr = islandIefcadr[island];
        for (int i = 0; i < snv; i++) {
            int dof = mapIdof2dof[idofadr + i];
            qacc[dof] = iQacc[i];
            qfrc_constraint[dof] = iQfrcConstraint[i];
        }
        for (int r = 0; r < snefc; r++)
            efc_force[mapIefc2efc[iefcadr + r]] = iForce[r];
    }

    /// <summary>
    ///     MuJoCo's projected Gauss-Seidel on the one island (<c>solPGS</c> for limit, frictionless and pyramidal rows):
    ///     per sweep a Nesterov extrapolation of the forces (restarted when the correction opposes it), then every row in
    ///     a shuffled order, its force moved to the minimum of the dual along it and clamped at zero, kept only if the
    ///     dual cost does not rise by more than 1e-10; the sweeps end when the scaled improvement falls below the
    ///     tolerance or after <c>iterations</c> sweeps.
    /// </summary>
    /// <remarks>
    ///     The shuffle restarts its generator in every solve, so the sweep orders repeat from step to step. The
    ///     improvement is scaled by the model's mean inertia, not the island's (unlike Newton). <see cref="efc_state"/>
    ///     follows the sweeps.
    /// </remarks>
    private void SolvePgs() {
        double pgsScale = 1 / (m.meaninertia * Math.Max(1, nv));
        for (int c = 0; c < nefc; c++)
            arInv[c] = 1 / efc_AR[c * (nefc + 1)];
        Array.Copy(efc_force, forcePrev, nefc);
        DualState();
        // every row is a block of its own: limits, frictionless contacts and pyramid edges have one dimension
        for (int c = 0; c < nefc; c++)
            blockStart[c] = c;
        pgsRng.Restart();
        int iter = 0, nesterovK = 0;
        while (iter < m.iterations) {
            double beta = 0;
            if (iter > 0)
                beta = (double)(nesterovK - 1) / (double)(nesterovK + 2);
            if (beta > 0) {
                for (int c = 0; c < nefc; c++) {
                    double saved = efc_force[c];
                    efc_force[c] += beta * (efc_force[c] - forcePrev[c]);
                    forcePrev[c] = saved;
                }
                // the cone of a limit or pyramid row is the half-line: project by clamping at zero
                for (int c = 0; c < nefc; c++)
                    if (efc_force[c] < 0)
                        efc_force[c] = 0;
            } else {
                Array.Copy(efc_force, forcePrev, nefc);
            }
            Array.Copy(efc_force, forceMomentum, nefc);
            double improvement = 0;
            pgsRng.Shuffle(blockStart, nefc);
            for (int bi = 0; bi < nefc; bi++) {
                int c = blockStart[bi];
                double res = efc_b[c] + Kernels.Dot(efc_AR.AsSpan(c * nefc), efc_force, nefc);
                double old = efc_force[c];
                efc_force[c] -= res * arInv[c];
                if (efc_force[c] < 0)
                    efc_force[c] = 0;
                // costChange with the diagonal as 1/(1/AR_cc), as MuJoCo recomputes it
                double athis = 1 / arInv[c];
                double delta = efc_force[c] - old;
                double change = 0.5 * delta * delta * athis + delta * res;
                if (change > 1e-10) {
                    efc_force[c] = old;
                    change = 0;
                }
                improvement -= change;
            }
            DualState();
            improvement *= pgsScale;
            // O'Donoghue-Candes gradient restart: the correction against the extrapolation
            bool restart = false;
            if (iter > 0) {
                double dot = 0;
                for (int c = 0; c < nefc; c++) {
                    double correction = efc_force[c] - forceMomentum[c];
                    double extrapolation = forceMomentum[c] - forcePrev[c];
                    dot += correction * extrapolation;
                }
                restart = dot < 0;
            }
            nesterovK = restart ? 0 : nesterovK + 1;
            iter++;
            if (improvement < m.tolerance)
                break;
        }
        solver_niter += iter;
    }

    /// <summary>The dual states of the rows (<c>dualState</c>): a positive force is quadratic, otherwise satisfied.</summary>
    private void DualState() {
        for (int c = 0; c < nefc; c++)
            efc_state[c] = efc_force[c] <= 0 ? Satisfied : Quadratic;
    }

    /// <summary>
    ///     The dual matrix <c>AR = Y*Y' + R</c> with <c>Y = J*M^-1/2</c> (<c>mj_projectConstraint</c>, dense): each row of
    ///     <c>J</c> is back-substituted through the factor's <c>L'</c> from the last dof down, then scaled by
    ///     <c>1/sqrt(D)</c> (<c>mj_solveM2_impl</c>); <c>Y*Y'</c> is MuJoCo's unweighted <c>mju_sqrMatTD</c> of
    ///     <c>Y'</c>, and <c>R</c> is added to the diagonal.
    /// </summary>
    private void MakeDualMatrix() {
        for (int i = 0; i < nv; i++)
            sqrtInvD[i] = 1 / Math.Sqrt(qLD[m.M_rowadr[i] + m.M_rownnz[i] - 1]);
        Array.Copy(efc_J, efcY, nefc * nv);
        for (int i = nv - 1; i > 0; i--) {
            int start = m.M_rowadr[i], end = start + m.M_rownnz[i] - 1;
            for (int r = 0; r < nefc; r++) {
                double xi = efcY[r * nv + i];
                if (xi == 0)
                    continue;
                for (int adr = start; adr < end; adr++)
                    efcY[r * nv + m.M_colind[adr]] -= qLD[adr] * xi;
            }
        }
        for (int i = 0; i < nv; i++)
            for (int r = 0; r < nefc; r++)
                efcY[r * nv + i] *= sqrtInvD[i];
        Kernels.Transpose(efcYT, efcY, nefc, nv);
        Kernels.SqrMatT(efc_AR, efcYT, nv, nefc);
        for (int r = 0; r < nefc; r++)
            efc_AR[r * (nefc + 1)] += efc_R[r];
    }

    /// <summary>
    ///     Forces, states and cost of the rows at residual <paramref name="r"/> (<c>mj_constraintUpdate_impl</c> for
    ///     inequality rows): a negative residual is quadratic with force <c>-D*r</c>, a non-negative one is satisfied.
    /// </summary>
    /// <param name="r">The residuals <c>J*qacc - aref</c>.</param>
    /// <param name="force">Receives the forces.</param>
    /// <param name="state">Receives the states.</param>
    /// <param name="D">The rows' impedance (all rows for the warm start, an island's for its solve).</param>
    /// <param name="n">The number of rows.</param>
    /// <returns>The constraint cost.</returns>
    private static double ConstraintUpdate(ReadOnlySpan<double> r, Span<double> force, Span<int> state, ReadOnlySpan<double> D, int n) {
        for (int i = 0; i < n; i++)
            force[i] = -D[i] * r[i];
        double s = 0;
        for (int i = 0; i < n; i++) {
            if (r[i] >= 0) {
                force[i] = 0;
                state[i] = Satisfied;
            } else {
                s += 0.5 * D[i] * r[i] * r[i];
                state[i] = Quadratic;
            }
        }
        return s;
    }

    /// <summary>
    ///     MuJoCo's primal Newton solver on the gathered island (<c>mj_solPrimal</c>): a convergence certificate first,
    ///     then Newton directions from the Cholesky factor of <c>M + J'*D*J</c>, exact line searches on the
    ///     piecewise-quadratic cost, and rank-one factor updates as rows change state. Every size and product is the
    ///     island's: its dot products, its Hessian, and its scale (the inverse of its summed inertia diagonal).
    /// </summary>
    /// <returns>The iterations taken (MuJoCo's <c>solver_niter</c> of the island).</returns>
    private int SolveNewton() {
        double tol = m.tolerance;
        int iter = 0;
        Kernels.MulSymVecSparse(Ma, iM, iQacc, snv, iMrownnz, iMrowadr, iMcolind);
        Kernels.MulMatVec(Jaref, iJ, iQacc, snefc, snv);
        Kernels.SubFrom(Jaref, iAref, snefc);
        PrimalUpdateConstraint();
        PrimalUpdateGrad();
        // island scale: the inverse of the island's summed inertia diagonal (not the model's mean inertia)
        double inertia = 0;
        for (int i = 0; i < snv; i++)
            inertia += iM[iMrowadr[i] + iMrownnz[i] - 1];
        scale = 1 / inertia;
        Array.Copy(grad, Mgrad, snv);
        Kernels.SolveLD(Mgrad, iQLD, iQLDiagInv, snv, iMrownnz, iMrowadr, iMcolind);
        bool gap = Kernels.Max(0, 0.5 * scale * Kernels.Dot(grad, Mgrad, snv)) < tol;
        bool gradient = scale * Kernels.Norm(grad, snv) < tol;
        bool done = gap && gradient;
        if (!done) {
            MakeHessian();
            Kernels.CholFactor(L, snv, Kernels.MinVal);
            Kernels.CholSolve(Mgrad, L, grad, snv);
            done = gradient && Kernels.Max(0, 0.5 * scale * Kernels.Dot(grad, Mgrad, snv)) < tol;
        }
        if (!done)
            Kernels.Scl(search, Mgrad, -1, snv);
        while (!done && iter < m.iterations) {
            double alpha = PrimalSearch(tol * m.ls_tolerance, m.ls_iterations, out double lsImprovement);
            if (alpha == 0)
                break;
            Kernels.AddToScl(iQacc, search, alpha, snv);
            Kernels.AddToScl(Ma, Mv, alpha, snv);
            Kernels.AddToScl(Jaref, Jv, alpha, snefc);
            Array.Copy(sState, oldState, snefc);
            PrimalUpdateConstraint();
            HessianIncremental();
            PrimalUpdateGrad();
            Kernels.CholSolve(Mgrad, L, grad, snv);
            double improvement = scale * lsImprovement;
            double gradNorm = scale * Kernels.Norm(grad, snv);
            double decrement = Kernels.Max(0, 0.5 * scale * Kernels.Dot(grad, Mgrad, snv));
            iter++;
            if ((improvement > 0 && improvement < tol) || gradNorm < tol || decrement < tol)
                break;
            Kernels.Scl(search, Mgrad, -1, snv);
        }
        return iter;
    }

    /// <summary>
    ///     Re-evaluates the island's rows at the current <see cref="Jaref"/>: forces (into the island's copy, which
    ///     <see cref="ScatterIsland"/> copies back), the solver's own states, the island's joint-space constraint forces, and the cost with the
    ///     Gauss term <c>0.5*(Ma - qfrc_smooth)·(qacc - qacc_smooth)</c>.
    /// </summary>
    private void PrimalUpdateConstraint() {
        cost = ConstraintUpdate(Jaref, iForce, sState, iD, snefc);
        Kernels.MulMatTVec(iQfrcConstraint, iJ, iForce, snefc, snv);
        double gauss = 0;
        for (int i = 0; i < snv; i++)
            gauss += 0.5 * (Ma[i] - iQfrcSmooth[i]) * (iQacc[i] - iQaccSmooth[i]);
        cost += gauss;
    }

    /// <summary>The cost gradient <c>M*qacc - qfrc_smooth - qfrc_constraint</c>.</summary>
    private void PrimalUpdateGrad() {
        for (int i = 0; i < snv; i++)
            grad[i] = Ma[i] - iQfrcSmooth[i] - iQfrcConstraint[i];
    }

    /// <summary>Builds the lower triangle of <c>H = J'*D_active*J + M</c> into <see cref="L"/> (not yet factorized).</summary>
    private void MakeHessian() {
        for (int i = 0; i < snefc; i++)
            Hdiag[i] = sState[i] == Quadratic ? iD[i] : 0;
        Kernels.SqrMatTDLower(L, iJ, Hdiag, snefc, snv);
        Kernels.AddToSymSparseLower(L, iM, snv, iMrownnz, iMrowadr, iMcolind);
    }

    /// <summary>
    ///     Updates the Hessian factor for rows that entered or left the quadratic zone by rank-one updates of
    ///     <c>J_i*sqrt(D_i)</c>; a lost rank rebuilds and refactorizes the Hessian from scratch, as MuJoCo does.
    /// </summary>
    private void HessianIncremental() {
        for (int i = 0; i < snefc; i++) {
            int flag = -1;
            if (oldState[i] != Quadratic && sState[i] == Quadratic)
                flag = 1;
            else if (oldState[i] == Quadratic && sState[i] != Quadratic)
                flag = 0;
            if (flag == -1)
                continue;
            Kernels.Scl(cholupd, iJ.AsSpan(i * snv), Math.Sqrt(iD[i]), snv);
            if (Kernels.CholUpdate(L, cholupd, snv, flag == 1) < snv) {
                MakeHessian();
                Kernels.CholFactor(L, snv, Kernels.MinVal);
                return;
            }
        }
    }

    /// <summary>
    ///     The exact line search along <see cref="search"/> (<c>PrimalSearch</c>): Newton steps on the 1-D piecewise
    ///     quadratic until the slope is within tolerance, bracketing and bisecting when they overshoot.
    /// </summary>
    /// <param name="tolerance">The gradient tolerance before scaling (<c>tolerance*ls_tolerance</c>).</param>
    /// <param name="lsIterations">The evaluation budget.</param>
    /// <param name="improvement">The cost decrease at the returned step.</param>
    /// <returns>The step length; 0 when no improvement was found.</returns>
    private double PrimalSearch(double tolerance, double lsIterations, out double improvement) {
        lsIter = 0;
        improvement = 0;
        double snorm = Kernels.Norm(search, snv);
        if (snorm < Kernels.MinVal)
            return 0;
        double gtol = tolerance * snorm / scale;
        Kernels.MulSymVecSparse(Mv, iM, search, snv, iMrownnz, iMrowadr, iMcolind);
        Kernels.MulMatVec(Jv, iJ, search, snefc, snv);
        PrimalPrepare();
        Pnt p0 = new Pnt { Alpha = 0 };
        Eval(ref p0);
        Pnt p1 = new Pnt { Alpha = p0.Alpha - p0.D0 / p0.D1 };
        Eval(ref p1);
        if (Math.Abs(p1.D0) < gtol && (p1.Alpha == 0 || p1.Cost < 0)) {
            improvement = -p1.Cost;
            return p1.Alpha;
        }
        int dir = p1.D0 < 0 ? +1 : -1;
        Pnt p2 = p0;
        while (p1.D0 * dir <= -gtol && lsIter < lsIterations) {
            p2 = p1;
            p1.Alpha -= p1.D0 / p1.D1;
            Eval(ref p1);
            if (Math.Abs(p1.D0) < gtol && p1.Cost < 0) {
                improvement = -p1.Cost;
                return p1.Alpha;
            }
        }
        if (lsIter >= lsIterations) {
            improvement = -p1.Cost;
            return p1.Alpha;
        }
        Pnt p2next = p1;
        Pnt p1next = new Pnt { Alpha = p1.Alpha - p1.D0 / p1.D1 };
        Eval(ref p1next);
        Span<Pnt> candidates = stackalloc Pnt[3];
        while (lsIter < lsIterations) {
            Pnt pmid = new Pnt { Alpha = 0.5 * (p1.Alpha + p2.Alpha) };
            Eval(ref pmid);
            // the candidates are copies taken before either bracket moves, as in MuJoCo
            candidates[0] = p1next;
            candidates[1] = p2next;
            candidates[2] = pmid;
            double bestcost = 0;
            int bestind = -1;
            for (int i = 0; i < 3; i++)
                if (Math.Abs(candidates[i].D0) < gtol && (bestind == -1 || candidates[i].Cost < bestcost)) {
                    bestcost = candidates[i].Cost;
                    bestind = i;
                }
            if (bestind >= 0) {
                improvement = -candidates[bestind].Cost;
                return candidates[bestind].Alpha;
            }
            bool b1 = UpdateBracket(ref p1, candidates, ref p1next);
            bool b2 = UpdateBracket(ref p2, candidates, ref p2next);
            if (!b1 && !b2) {
                improvement = -pmid.Cost;
                return pmid.Alpha;
            }
        }
        if (p1.Cost <= p2.Cost && p1.Cost < 0) {
            improvement = -p1.Cost;
            return p1.Alpha;
        }
        if (p2.Cost <= p1.Cost && p2.Cost < 0) {
            improvement = -p2.Cost;
            return p2.Alpha;
        }
        return 0;
    }

    /// <summary>
    ///     Moves a bracket end to the candidate with the same slope sign and a smaller slope magnitude, then evaluates the
    ///     Newton point from it (<c>updateBracket</c>).
    /// </summary>
    /// <param name="p">The bracket end.</param>
    /// <param name="candidates">The three candidate points.</param>
    /// <param name="pnext">Receives the next Newton point when the end moved.</param>
    /// <returns>Whether the end moved.</returns>
    private bool UpdateBracket(ref Pnt p, ReadOnlySpan<Pnt> candidates, ref Pnt pnext) {
        bool moved = false;
        for (int i = 0; i < 3; i++) {
            if (p.D0 < 0 && candidates[i].D0 < 0 && p.D0 < candidates[i].D0) {
                p = candidates[i];
                moved = true;
            } else if (p.D0 > 0 && candidates[i].D0 > 0 && p.D0 > candidates[i].D0) {
                p = candidates[i];
                moved = true;
            }
        }
        if (moved) {
            pnext.Alpha = p.Alpha - p.D0 / p.D1;
            Eval(ref pnext);
        }
        return moved;
    }

    /// <summary>
    ///     The 1-D quadratics of the line search (<c>PrimalPrepare</c>): the Gauss term's slope and curvature along the
    ///     search direction, and per row <c>0.5*D*r²</c>, <c>D*r*Jv</c>, <c>0.5*D*Jv²</c>.
    /// </summary>
    private void PrimalPrepare() {
        quadGauss1 = Kernels.Dot(search, Ma, snv) - Kernels.Dot(iQfrcSmooth, search, snv);
        quadGauss2 = 0.5 * Kernels.Dot(search, Mv, snv);
        for (int i = 0; i < snefc; i++) {
            double d = iD[i], dj0 = d * Jaref[i];
            quad[3 * i] = Jaref[i] * dj0;
            quad[3 * i + 1] = Jv[i] * dj0;
            quad[3 * i + 2] = Jv[i] * d * Jv[i];
            quad[3 * i] *= 0.5;
            quad[3 * i + 2] *= 0.5;
        }
    }

    /// <summary>
    ///     The cost change and its first two derivatives at step <c>p.Alpha</c> (<c>PrimalEval</c>): rows active at the
    ///     step add their quadratic, rows active at 0 but not at the step subtract their starting cost.
    /// </summary>
    /// <param name="p">The point; its cost and derivatives are filled.</param>
    private void Eval(ref Pnt p) {
        double c = 0, alpha = p.Alpha, d0 = 0, d1 = 0;
        double q0 = 0, q1 = quadGauss1, q2 = quadGauss2;
        for (int i = 0; i < snefc; i++) {
            double start = Jaref[i];
            double x = start + alpha * Jv[i];
            double cost0 = start < 0 ? quad[3 * i] : 0;
            if (x < 0) {
                q0 += quad[3 * i] - cost0;
                q1 += quad[3 * i + 1];
                q2 += quad[3 * i + 2];
            } else {
                c -= cost0;
            }
        }
        c += alpha * alpha * q2 + alpha * q1 + q0;
        d0 += 2 * alpha * q2 + q1;
        d1 += 2 * q2;
        // MuJoCo warns ("not convex") and clamps; with an SPD mass matrix it does not happen
        if (d1 <= 0)
            d1 = Kernels.MinVal;
        p.Cost = c;
        p.D0 = d0;
        p.D1 = d1;
        lsIter++;
    }

    // ================================================================================== integration

    /// <summary>
    ///     The Euler step (<c>mj_EulerSkip</c>): with dof damping, <c>(M + h*diag(damping'))*a = qfrc_smooth +
    ///     qfrc_constraint</c> is solved for the velocity update (implicit in damping), else <see cref="qacc"/> is used;
    ///     velocities then positions advance, time moves on, and <see cref="qacc"/> becomes the next warm start.
    /// </summary>
    private void Euler() {
        bool damped = false;
        for (int i = 0; i < nv && !damped; i++)
            damped = m.dof_damping[i] > 0 || m.dof_dampingpoly[2 * i] != 0 || m.dof_dampingpoly[2 * i + 1] != 0;
        if (!damped) {
            Array.Copy(qacc, qaccEuler, nv);
        } else {
            Array.Copy(M, qH, m.nC);
            for (int i = 0; i < nv; i++) {
                double damping = m.dof_damping[i] + 0.0;
                double slope = XPolyForce(damping, m.dof_dampingpoly.AsSpan(2 * i, 2), qvel[i]);
                qH[m.M_rowadr[i] + m.M_rownnz[i] - 1] += m.timestep * slope;
            }
            Kernels.FactorI(qH, qHDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
            Kernels.Add(qaccEuler, qfrc_smooth, qfrc_constraint, nv);
            Kernels.SolveLD(qaccEuler, qH, qHDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        }
        Kernels.AddToScl(qvel, qaccEuler, m.timestep, nv);
        IntegratePos(qpos, qvel, m.timestep);
        time += m.timestep;
        Array.Copy(qacc, qacc_warmstart, nv);
    }

    /// <summary>
    ///     Integrates positions over <paramref name="dt"/> with the velocities <paramref name="vel"/>
    ///     (<c>mj_integratePos</c>): hinges and slides add <c>dt*vel</c>; a free joint adds <c>dt*vel</c> to its position
    ///     and rotates its quaternion by <c>vel*dt</c> (<c>mju_quatIntegrate</c>).
    /// </summary>
    /// <param name="pos">The positions, updated in place.</param>
    /// <param name="vel">The velocities.</param>
    /// <param name="dt">The time step.</param>
    private void IntegratePos(Span<double> pos, ReadOnlySpan<double> vel, double dt) {
        for (int b = 1; b < nbody; b++) {
            int jend = m.body_jntadr[b] + m.body_jntnum[b];
            for (int j = m.body_jntadr[b]; j < jend; j++) {
                int padr = m.jnt_qposadr[j], vadr = m.jnt_dofadr[j];
                if (m.jnt_type[j] == Free) {
                    for (int k = 0; k < 3; k++)
                        pos[padr + k] += dt * vel[vadr + k];
                    QuatIntegrate(pos.Slice(padr + 3, 4), vel.Slice(vadr + 3, 3), dt);
                } else {
                    pos[padr] += dt * vel[vadr];
                }
            }
        }
    }

    /// <summary>
    ///     Rotates a quaternion by an angular velocity over a time scale (<c>mju_quatIntegrate</c>): the rotation of angle
    ///     <c>scale*|vel|</c> about <c>vel</c>, applied after normalizing the quaternion.
    /// </summary>
    /// <param name="quat">The quaternion, updated in place.</param>
    /// <param name="vel">The angular velocity.</param>
    /// <param name="scale">The time scale.</param>
    private static void QuatIntegrate(Span<double> quat, ReadOnlySpan<double> vel, double scale) {
        Span<double> axis = stackalloc double[3] { vel[0], vel[1], vel[2] }, qrot = stackalloc double[4];
        double angle = scale * Spatial.Normalize3(axis);
        Spatial.AxisAngle2Quat(qrot, axis, angle);
        Spatial.Normalize4(quat);
        Spatial.MulQuat(quat, quat, qrot);
    }

    /// <summary>
    ///     The RK4 step (<c>mj_RungeKutta</c> with MuJoCo's tableau): three more forward passes at the intermediate states
    ///     <c>X0 + h*sum(A_ij*(v_j, a_j))</c> (positions integrated, velocities added) and times, then the state advanced
    ///     from <c>X0</c> by the weighted increment with the B weights; <see cref="qacc"/>, which the last pass solved,
    ///     becomes the next warm start.
    /// </summary>
    /// <remarks>
    ///     The increments are accumulated with MuJoCo's <c>addToScl</c>, zero weights included (their products can turn a
    ///     -0 into +0). Every published array afterwards is the last pass's, at the third intermediate state.
    /// </remarks>
    private void RungeKutta() {
        int nq = m.nq, stride = nq + nv;
        double h = m.timestep, time0 = time;
        Span<double> c = stackalloc double[3], t = stackalloc double[3];
        for (int i = 1; i < 4; i++) {
            c[i - 1] = 0;
            for (int j = 0; j < i; j++)
                c[i - 1] += Rk4A[(i - 1) * 3 + j];
            t[i - 1] = time + c[i - 1] * h;
        }
        Span<double> x0 = rkX.AsSpan(0, stride);
        qpos.AsSpan(0, nq).CopyTo(x0);
        qvel.AsSpan(0, nv).CopyTo(x0.Slice(nq));
        qacc.AsSpan(0, nv).CopyTo(rkF);
        for (int i = 1; i < 4; i++) {
            Span<double> dX = rkdX;
            dX.Clear();
            for (int j = 0; j < i; j++) {
                Kernels.AddToScl(dX, rkX.AsSpan(j * stride + nq, nv), Rk4A[(i - 1) * 3 + j], nv);
                Kernels.AddToScl(dX.Slice(nv), rkF.AsSpan(j * nv, nv), Rk4A[(i - 1) * 3 + j], nv);
            }
            Span<double> xi = rkX.AsSpan(i * stride, stride);
            x0.CopyTo(xi);
            IntegratePos(xi.Slice(0, nq), dX.Slice(0, nv), h);
            Kernels.AddToScl(xi.Slice(nq), dX.Slice(nv), h, nv);
            xi.Slice(0, nq).CopyTo(qpos);
            xi.Slice(nq, nv).CopyTo(qvel);
            time = t[i - 1];
            Forward();
            qacc.AsSpan(0, nv).CopyTo(rkF.AsSpan(i * nv));
        }
        rkdX.AsSpan().Clear();
        for (int j = 0; j < 4; j++) {
            Kernels.AddToScl(rkdX, rkX.AsSpan(j * stride + nq, nv), Rk4B[j], nv);
            Kernels.AddToScl(rkdX.AsSpan(nv), rkF.AsSpan(j * nv, nv), Rk4B[j], nv);
        }
        time = time0;
        x0.Slice(0, nq).CopyTo(qpos);
        x0.Slice(nq, nv).CopyTo(qvel);
        // mj_advance: velocities with the weighted acceleration, positions with the weighted velocity
        Kernels.AddToScl(qvel, rkdX.AsSpan(nv, nv), h, nv);
        IntegratePos(qpos, rkdX.AsSpan(0, nv), h);
        time += h;
        Array.Copy(qacc, qacc_warmstart, nv);
    }

    /// <summary>
    ///     Fixed tendons' lengths and Jacobians (<c>mj_tendon</c>): each length is the sum of its coefficients times its
    ///     joints' positions, added in wrap order from zero; each Jacobian holds the coefficients at the joints' dofs.
    /// </summary>
    private void Tendons() {
        for (int t = 0; t < m.ntendon; t++) {
            ten_length[t] = 0;
            int rowadr = m.ten_J_rowadr[t], rownnz = m.ten_J_rownnz[t];
            ten_J.AsSpan(rowadr, rownnz).Clear();
            int adr = m.tendon_adr[t];
            for (int j = 0; j < m.tendon_num[t]; j++) {
                int k = m.wrap_objid[adr + j];
                ten_length[t] += m.wrap_prm[adr + j] * qpos[m.jnt_qposadr[k]];
                // mju_combineSparseInc: J = 1*J + coef*1 at the joint's dof
                int dof = m.jnt_dofadr[k];
                for (int p = 0; p < rownnz; p++)
                    if (m.ten_J_colind[rowadr + p] == dof)
                        ten_J[rowadr + p] = 1 * ten_J[rowadr + p] + m.wrap_prm[adr + j] * 1;
            }
        }
    }

    /// <summary>
    ///     The external forces of <c>mj_rnePostConstraint</c> into <see cref="cfrc_ext"/>: each active contact's force
    ///     (the pyramid decoded, or the frictionless normal) rotated to the world, moved to the subtree centre of mass of
    ///     each body, minus on the first geom's body and plus on the second's; the world gets nothing.
    /// </summary>
    /// <remarks>
    ///     Call after <see cref="Step"/>, as Gymnasium's <c>do_simulation</c> calls <c>mj_rnePostConstraint</c>. MuJoCo's
    ///     body accelerations and internal forces (<c>cacc</c>, <c>cfrc_int</c>), which no env here reads, are not
    ///     computed; fixed tendons add nothing.
    /// </remarks>
    public void RnePostConstraint() {
        Array.Clear(cfrc_ext);
        Span<double> lfrc = stackalloc double[6], frc = stackalloc double[6], com = stackalloc double[6];
        for (int c = 0; c < ncon; c++) {
            int adr = con_efc_address[c];
            if (adr < 0)
                continue;
            lfrc.Clear();
            int dim = con_dim[c];
            if (dim == 1) {
                lfrc[0] = efc_force[adr];
            } else {
                // mju_decodePyramid: the normal is the edges' sum, each tangent its pair's difference times friction
                lfrc[0] = 0;
                for (int i = 0; i < 2 * (dim - 1); i++)
                    lfrc[0] += efc_force[adr + i];
                for (int i = 0; i < dim - 1; i++)
                    lfrc[i + 1] = (efc_force[adr + 2 * i] - efc_force[adr + 2 * i + 1]) * con_friction[5 * c + i];
            }
            lfrc[0] -= 0.0;
            // torque from the local torque, force from the local force, each through frame' (mju_mulMatTVec3)
            ReadOnlySpan<double> f = con_frame.AsSpan(9 * c, 9);
            for (int k = 0; k < 3; k++) {
                frc[k] = f[k] * lfrc[3] + f[3 + k] * lfrc[4] + f[6 + k] * lfrc[5];
                frc[3 + k] = f[k] * lfrc[0] + f[3 + k] * lfrc[1] + f[6 + k] * lfrc[2];
            }
            AddExternal(m.geom_bodyid[con_geom1[c]], con_pos.AsSpan(3 * c, 3), frc, -1, com);
            AddExternal(m.geom_bodyid[con_geom2[c]], con_pos.AsSpan(3 * c, 3), frc, 1, com);
        }
    }

    /// <summary>
    ///     Adds a world wrench applied at a point to a body's <see cref="cfrc_ext"/> (<c>addExternal</c>): the torque
    ///     moved to the body's subtree centre of mass (<c>mju_transformSpatial</c> for a force), times the scale.
    /// </summary>
    /// <param name="body">The body (the world takes nothing).</param>
    /// <param name="point">The application point.</param>
    /// <param name="frc">The wrench, torque then force.</param>
    /// <param name="scl">The scale, -1 or 1.</param>
    /// <param name="com">Scratch for the moved wrench.</param>
    private void AddExternal(int body, ReadOnlySpan<double> point, ReadOnlySpan<double> frc, double scl, Span<double> com) {
        if (body == 0)
            return;
        int root = m.body_rootid[body];
        Span<double> dif = stackalloc double[3], cros = stackalloc double[3];
        for (int k = 0; k < 3; k++)
            dif[k] = subtree_com[3 * root + k] - point[k];
        Spatial.Cross(cros, dif, frc.Slice(3));
        for (int k = 0; k < 3; k++) {
            com[k] = frc[k] - cros[k];
            com[3 + k] = frc[3 + k];
        }
        Kernels.AddToScl(cfrc_ext.AsSpan(6 * body), com, scl, 6);
    }
}
