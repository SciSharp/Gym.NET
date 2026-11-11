using System;

namespace MujocoPoc;

/// <summary>
///     A from-scratch physics core for planar walkers like HalfCheetah-v5, written to step exactly as MuJoCo 3.15.0's
///     <c>mj_step</c> does on win-x64: the same quantities in the same floating-point order, so every array it publishes
///     (state, contacts, constraint rows, solver output) equals MuJoCo's bit for bit.
/// </summary>
/// <remarks>
///     <para>
///         It is not a translation of MuJoCo's sources: it computes only what such a model needs (hinge and slide joints,
///         a ground plane against capsules, joint limits, pyramidal contacts, motors, the dense Newton solver, Euler with
///         implicit joint damping) and drops MuJoCo's generality (sparse paths, sleeping, islands of many trees, flex,
///         tendons, sensors, callbacks). <see cref="CheetahModel.Load"/> refuses models outside that scope.
///     </para>
///     <para>
///         Arrays carry MuJoCo's names and layouts so they can be compared with a recorded <c>mjData</c>. Not thread-safe:
///         one core owns its arrays, and <see cref="Step"/> reuses scratch storage.
///     </para>
///     <para>
///         Reference: MuJoCo 3.15.0's <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_forward.c">engine_forward.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_smooth.c">engine_core_smooth.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_collision_driver.c">engine_collision_driver.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_collision_primitive.c">engine_collision_primitive.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_core_constraint.c">engine_core_constraint.c</see>,
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_solver.c">engine_solver.c</see> and
///         <see href="https://github.com/google-deepmind/mujoco/blob/3.15.0/src/engine/engine_passive.c">engine_passive.c</see>
///         (Apache-2.0); each member names the MuJoCo function whose arithmetic it reproduces.
///     </para>
/// </remarks>
public sealed class CheetahCore {
    private const int Slide = 2;
    private const int Hinge = 3;
    private const int Plane = 0;
    /// <summary>The constraint type of a joint-limit row (MuJoCo's <c>mjCNSTR_LIMIT_JOINT</c>).</summary>
    public const int LimitJoint = 3;
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
    ///     Constraint states as MuJoCo publishes them: from the warm start's evaluation at <see cref="qacc_smooth"/>, since
    ///     the island solver keeps its own states and does not copy them back.
    /// </summary>
    public readonly int[] efc_state;
    /// <summary>The number of constraint islands (0 or 1: one kinematic tree).</summary>
    public int nisland;
    /// <summary>Newton iterations of the last solve.</summary>
    public int solver_niter;

    // ---- velocity, actuation, acceleration
    /// <summary>Body spatial velocities about the subtree centers of mass.</summary>
    public readonly double[] cvel;
    /// <summary>Time derivatives of <see cref="cdof"/>.</summary>
    public readonly double[] cdof_dot;
    /// <summary>Joint spring forces.</summary>
    public readonly double[] qfrc_spring;
    /// <summary>Joint damping forces.</summary>
    public readonly double[] qfrc_damper;
    /// <summary>Passive forces (springs plus dampers).</summary>
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

    /// <summary>A line-search point: step length, cost change, first and second derivative.</summary>
    private struct Pnt {
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
        // Two contacts per plane-capsule pair at most; four pyramid rows per condim-3 contact; two sides per limit.
        int maxcon = 2 * ngeom, maxefc = 2 * njnt + 4 * maxcon;
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
    }

    /// <summary>
    ///     Advances the simulation by one time step, as <c>mj_step</c> with the Euler integrator: state checks, the forward
    ///     dynamics at the current state, then the semi-implicit Euler update of <see cref="qvel"/>, <see cref="qpos"/> and
    ///     <see cref="time"/>.
    /// </summary>
    /// <remarks>
    ///     A NaN or a value beyond 1e10 in qpos, qvel or the solved qacc resets the state to the reference configuration at
    ///     rest (MuJoCo's auto-reset), controls included; the warning MuJoCo records is not kept.
    /// </remarks>
    public void Step() {
        if (AnyBad(qpos, m.nq) || AnyBad(qvel, nv))
            ResetData();
        Forward();
        if (AnyBad(qacc, nv)) {
            ResetData();
            Forward();
        }
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
        MakeM();
        Array.Copy(M, qLD, m.nC);
        Kernels.FactorI(qLD, qLDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        Collision();
        MakeConstraint();
        // one kinematic tree: any constraint row puts all of it in the single island
        nisland = nefc > 0 ? 1 : 0;
        for (int i = 0; i < nu; i++) {
            int j = m.actuator_trnid[2 * i];
            actuator_length[i] = qpos[m.jnt_qposadr[j]] * m.actuator_gear[6 * i];
            actuator_moment[i] = m.actuator_gear[6 * i];
        }
        // velocity stage
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
    ///     Body, inertial and geom frames from the joint positions (<c>mj_kinematics</c>): each body composes its parent's
    ///     frame with its fixed offset, then its joints in order (slides translate along the rotated axis, hinges rotate
    ///     about the anchor), and its quaternion is renormalized.
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
        for (int i = 1; i < nbody; i++) {
            int root = m.body_rootid[i], jend = m.body_jntadr[i] + m.body_jntnum[i];
            for (int j = m.body_jntadr[i]; j < jend; j++) {
                Span<double> dof = cdof.AsSpan(6 * m.jnt_dofadr[j], 6);
                for (int k = 0; k < 3; k++)
                    offset[k] = subtree_com[3 * root + k] - xanchor[3 * j + k];
                if (m.jnt_type[j] == Slide) {
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
    ///     Plane-capsule contacts in MuJoCo's order (<c>mj_collision</c>): the world body's plane against each body's geoms,
    ///     bodies ascending (the broadphase's world pairs sorted by signature), geoms ascending, each pair filtered by the
    ///     contact bitmasks and the plane-to-bounding-sphere distance, then up to two contacts, one per capsule end.
    /// </summary>
    /// <remarks>
    ///     A body with several geoms goes through MuJoCo's bounding-volume midphase, whose visiting order this core assumes
    ///     to be ascending; the HalfCheetah traces never have two such geoms in contact at once, so it is unverified.
    /// </remarks>
    private void Collision() {
        ncon = 0;
        int plane = m.body_geomadr[0];
        for (int b = 1; b < nbody; b++) {
            int weld = m.body_weldid[b];
            // world pairs the broadphase drops: welded to the world, both dof-less, or body masks that cannot meet
            if (weld == 0 || m.body_dofnum[weld] == 0)
                continue;
            if ((m.body_contype[0] & m.body_conaffinity[b]) == 0 && (m.body_contype[b] & m.body_conaffinity[0]) == 0)
                continue;
            int end = m.body_geomadr[b] + m.body_geomnum[b];
            for (int g = m.body_geomadr[b]; g < end; g++) {
                if ((m.geom_contype[plane] & m.geom_conaffinity[g]) == 0 && (m.geom_contype[g] & m.geom_conaffinity[plane]) == 0)
                    continue;
                double margin = m.geom_margin[plane] + m.geom_margin[g];
                double gap = m.geom_gap[plane] + m.geom_gap[g];
                if (m.geom_rbound[g] > 0 && PlaneDist(plane, g) > margin + gap + m.geom_rbound[g])
                    continue;
                int first = ncon;
                PlaneCapsule(plane, g, margin + gap);
                if (ncon > first)
                    ContactParams(first, ncon, plane, g, margin);
            }
        }
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
    ///     four pyramid edges per active contact; then their approximate inverse inertia and impedance.
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
            Kernels.MulMatMat(jacrot, con_frame.AsSpan(9 * c, 9), jacdif, 3, 3, nv);
            for (int k = 1; k < con_dim[c]; k++) {
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
    ///     pyramidal contact's edges take the two bodies' translational weight scaled by <c>1 + friction²</c>.
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
                dim = 2 * (con_dim[id] - 1);
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
    ///     velocity and adds its dofs one by one, each dof's derivative taken against the velocity before it.
    /// </summary>
    private void ComVel() {
        Array.Clear(cvel, 0, 6);
        Span<double> cv = stackalloc double[6], tmp = stackalloc double[6];
        for (int i = 1; i < nbody; i++) {
            cvel.AsSpan(6 * m.body_parentid[i], 6).CopyTo(cv);
            int bda = m.body_dofadr[i];
            for (int j = 0; j < m.body_dofnum[i]; j++) {
                int d = bda + j;
                Spatial.CrossMotion(cdof_dot.AsSpan(6 * d), cv, cdof.AsSpan(6 * d));
                Kernels.Scl(tmp, cdof.AsSpan(6 * d), qvel[d], 6);
                Kernels.AddTo(cv, tmp, 6);
            }
            cv.CopyTo(cvel.AsSpan(6 * i));
        }
    }

    /// <summary>Joint springs and dof dampers (<c>mj_passive</c>'s spring-damper part; nothing else applies here).</summary>
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
    ///     The constrained acceleration (<c>mj_fwdConstraint</c> with the Newton solver on the single island): the
    ///     unconstrained residual, the warm start chosen by cost, then the primal Newton iterations.
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
        double costWarm = ConstraintUpdate(jar, efc_force, efc_state);
        Kernels.MulMatTVec(qfrc_constraint, efc_J, efc_force, nefc, nv);
        Kernels.Sub(da, qacc_warmstart, qacc_smooth, nv);
        Kernels.MulSymVecSparse(Mda, M, da, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        costWarm += 0.5 * Kernels.Dot(da, Mda, nv);
        // this second evaluation is what MuJoCo leaves in efc_state (and, until the solve, efc_force)
        double costSmooth = ConstraintUpdate(efc_b, efc_force, efc_state);
        Kernels.MulMatTVec(qfrc_constraint, efc_J, efc_force, nefc, nv);
        if (costWarm > costSmooth)
            Array.Copy(qacc_smooth, qacc, nv);
        solver_niter = 0;
        SolveNewton();
    }

    /// <summary>
    ///     Forces, states and cost of the rows at residual <paramref name="r"/> (<c>mj_constraintUpdate_impl</c> for
    ///     inequality rows): a negative residual is quadratic with force <c>-D*r</c>, a non-negative one is satisfied.
    /// </summary>
    /// <param name="r">The residuals <c>J*qacc - aref</c>.</param>
    /// <param name="force">Receives the forces.</param>
    /// <param name="state">Receives the states.</param>
    /// <returns>The constraint cost.</returns>
    private double ConstraintUpdate(ReadOnlySpan<double> r, Span<double> force, Span<int> state) {
        for (int i = 0; i < nefc; i++)
            force[i] = -efc_D[i] * r[i];
        double s = 0;
        for (int i = 0; i < nefc; i++) {
            if (r[i] >= 0) {
                force[i] = 0;
                state[i] = Satisfied;
            } else {
                s += 0.5 * efc_D[i] * r[i] * r[i];
                state[i] = Quadratic;
            }
        }
        return s;
    }

    /// <summary>
    ///     MuJoCo's primal Newton solver on the one island (<c>mj_solPrimal</c>): a convergence certificate first, then
    ///     Newton directions from the Cholesky factor of <c>M + J'*D*J</c>, exact line searches on the piecewise-quadratic
    ///     cost, and rank-one factor updates as rows change state.
    /// </summary>
    private void SolveNewton() {
        double tol = m.tolerance;
        int iter = 0;
        Kernels.MulSymVecSparse(Ma, M, qacc, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        Kernels.MulMatVec(Jaref, efc_J, qacc, nefc, nv);
        Kernels.SubFrom(Jaref, efc_aref, nefc);
        PrimalUpdateConstraint();
        PrimalUpdateGrad();
        // island scale: the inverse of the island's summed inertia diagonal (not the model's mean inertia)
        double inertia = 0;
        for (int i = 0; i < nv; i++)
            inertia += M[m.M_rowadr[i] + m.M_rownnz[i] - 1];
        scale = 1 / inertia;
        Array.Copy(grad, Mgrad, nv);
        Kernels.SolveLD(Mgrad, qLD, qLDiagInv, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        bool gap = Kernels.Max(0, 0.5 * scale * Kernels.Dot(grad, Mgrad, nv)) < tol;
        bool gradient = scale * Kernels.Norm(grad, nv) < tol;
        bool done = gap && gradient;
        if (!done) {
            MakeHessian();
            Kernels.CholFactor(L, nv, Kernels.MinVal);
            Kernels.CholSolve(Mgrad, L, grad, nv);
            done = gradient && Kernels.Max(0, 0.5 * scale * Kernels.Dot(grad, Mgrad, nv)) < tol;
        }
        if (!done)
            Kernels.Scl(search, Mgrad, -1, nv);
        while (!done && iter < m.iterations) {
            double alpha = PrimalSearch(tol * m.ls_tolerance, m.ls_iterations, out double lsImprovement);
            if (alpha == 0)
                break;
            Kernels.AddToScl(qacc, search, alpha, nv);
            Kernels.AddToScl(Ma, Mv, alpha, nv);
            Kernels.AddToScl(Jaref, Jv, alpha, nefc);
            Array.Copy(sState, oldState, nefc);
            PrimalUpdateConstraint();
            HessianIncremental();
            PrimalUpdateGrad();
            Kernels.CholSolve(Mgrad, L, grad, nv);
            double improvement = scale * lsImprovement;
            double gradNorm = scale * Kernels.Norm(grad, nv);
            double decrement = Kernels.Max(0, 0.5 * scale * Kernels.Dot(grad, Mgrad, nv));
            iter++;
            if ((improvement > 0 && improvement < tol) || gradNorm < tol || decrement < tol)
                break;
            Kernels.Scl(search, Mgrad, -1, nv);
        }
        solver_niter += iter;
    }

    /// <summary>
    ///     Re-evaluates the rows at the current <see cref="Jaref"/>: forces (into <see cref="efc_force"/>, which MuJoCo
    ///     copies back from the island), the solver's own states, <see cref="qfrc_constraint"/>, and the cost with the
    ///     Gauss term <c>0.5*(Ma - qfrc_smooth)·(qacc - qacc_smooth)</c>.
    /// </summary>
    private void PrimalUpdateConstraint() {
        cost = ConstraintUpdate(Jaref, efc_force, sState);
        Kernels.MulMatTVec(qfrc_constraint, efc_J, efc_force, nefc, nv);
        double gauss = 0;
        for (int i = 0; i < nv; i++)
            gauss += 0.5 * (Ma[i] - qfrc_smooth[i]) * (qacc[i] - qacc_smooth[i]);
        cost += gauss;
    }

    /// <summary>The cost gradient <c>M*qacc - qfrc_smooth - qfrc_constraint</c>.</summary>
    private void PrimalUpdateGrad() {
        for (int i = 0; i < nv; i++)
            grad[i] = Ma[i] - qfrc_smooth[i] - qfrc_constraint[i];
    }

    /// <summary>Builds the lower triangle of <c>H = J'*D_active*J + M</c> into <see cref="L"/> (not yet factorized).</summary>
    private void MakeHessian() {
        for (int i = 0; i < nefc; i++)
            Hdiag[i] = sState[i] == Quadratic ? efc_D[i] : 0;
        Kernels.SqrMatTDLower(L, efc_J, Hdiag, nefc, nv);
        Kernels.AddToSymSparseLower(L, M, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
    }

    /// <summary>
    ///     Updates the Hessian factor for rows that entered or left the quadratic zone by rank-one updates of
    ///     <c>J_i*sqrt(D_i)</c>; a lost rank rebuilds and refactorizes the Hessian from scratch, as MuJoCo does.
    /// </summary>
    private void HessianIncremental() {
        for (int i = 0; i < nefc; i++) {
            int flag = -1;
            if (oldState[i] != Quadratic && sState[i] == Quadratic)
                flag = 1;
            else if (oldState[i] == Quadratic && sState[i] != Quadratic)
                flag = 0;
            if (flag == -1)
                continue;
            Kernels.Scl(cholupd, efc_J.AsSpan(i * nv), Math.Sqrt(efc_D[i]), nv);
            if (Kernels.CholUpdate(L, cholupd, nv, flag == 1) < nv) {
                MakeHessian();
                Kernels.CholFactor(L, nv, Kernels.MinVal);
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
        double snorm = Kernels.Norm(search, nv);
        if (snorm < Kernels.MinVal)
            return 0;
        double gtol = tolerance * snorm / scale;
        Kernels.MulSymVecSparse(Mv, M, search, nv, m.M_rownnz, m.M_rowadr, m.M_colind);
        Kernels.MulMatVec(Jv, efc_J, search, nefc, nv);
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
        quadGauss1 = Kernels.Dot(search, Ma, nv) - Kernels.Dot(qfrc_smooth, search, nv);
        quadGauss2 = 0.5 * Kernels.Dot(search, Mv, nv);
        for (int i = 0; i < nefc; i++) {
            double d = efc_D[i], dj0 = d * Jaref[i];
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
        for (int i = 0; i < nefc; i++) {
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
        for (int b = 1; b < nbody; b++) {
            int jend = m.body_jntadr[b] + m.body_jntnum[b];
            for (int j = m.body_jntadr[b]; j < jend; j++)
                qpos[m.jnt_qposadr[j]] += m.timestep * qvel[m.jnt_dofadr[j]];
        }
        time += m.timestep;
        Array.Copy(qacc, qacc_warmstart, nv);
    }
}
