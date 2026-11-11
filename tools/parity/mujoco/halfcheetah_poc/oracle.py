"""HalfCheetah-v5 oracle for the from-scratch core: records MuJoCo 3.15.0's exact state after every mj_step of a seeded
episode, and Gymnasium's env results, as binary files the C# proof of concept compares against bit for bit.

Writes into the output folder:
    model.json  every compiled model array the core reads, as exact doubles (Python's repr round-trips), plus the
                options, the sizes and frame_skip.
    run.json    the episode's seed, length and start ({"seed", "steps", "flip"}).
    trace.bin   records of an int32 count and that many little-endian float64. Header: qpos, qvel, qacc_warmstart and
                [time] at the start. Then per env step the action, and per mj_step (frame_skip of them) the state block
                (STATE below), [time, ncon, nefc, solver_niter, nisland], the contacts (dist, geom pairs, pos, frame),
                efc_J, the efc arrays (EFC below) and the position/velocity-stage arrays (DEBUG below).
    env.bin     the observation after reset, then per env step the observation and [reward, terminated, truncated,
                x_position, x_velocity, reward_forward, reward_ctrl], from a second env stepped through Gymnasium.

Args (command line):
    --out DIR     output folder (required; a 1,000-step episode writes about 40 MB, so keep it outside the repo).
    --seed N      reset and action-space seed (default 42).
    --steps N     env steps (default 1000).
    --flip        turn the cheetah upside down after the reset (qpos[2] += pi), so its torso and head land on the
                  ground; both envs get the same start.

Raises:
    AssertionError: the manual stepping stops matching Gymnasium's own step (checked on the first steps).
"""
import argparse
import json
import os
import struct
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "refs", "Gymnasium"))
import gymnasium as gym  # noqa: E402  (imported from the pinned submodule, as the parity suite does)
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--flip", action="store_true")
args = ap.parse_args()
OUT, SEED, STEPS = args.out, args.seed, args.steps
os.makedirs(OUT, exist_ok=True)

env = gym.make("HalfCheetah-v5")
u = env.unwrapped
m, d = u.model, u.data

# ---- the compiled model, every array the core reads
fields = {}
for name in ["body_parentid", "body_rootid", "body_weldid", "body_jntnum", "body_jntadr", "body_dofnum", "body_dofadr",
             "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_subtreemass", "body_inertia",
             "body_invweight0", "body_sameframe", "body_simple", "body_gravcomp",
             "jnt_type", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid", "jnt_limited", "jnt_pos", "jnt_axis",
             "jnt_stiffness", "jnt_range", "jnt_margin", "jnt_solref", "jnt_solimp",
             "dof_bodyid", "dof_jntid", "dof_parentid", "dof_Madr", "dof_simplenum", "dof_damping", "dof_armature",
             "dof_frictionloss", "dof_dampingpoly", "dof_M0", "dof_invweight0", "dof_solref", "dof_solimp",
             "geom_type", "geom_contype", "geom_conaffinity", "geom_condim", "geom_bodyid", "geom_sameframe",
             "geom_priority", "geom_size", "geom_rbound", "geom_aabb", "geom_pos", "geom_quat", "geom_friction",
             "geom_solmix", "geom_solref", "geom_solimp", "geom_margin", "geom_gap",
             "actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_trnid",
             "actuator_ctrllimited", "actuator_forcelimited", "actuator_ctrlrange", "actuator_forcerange",
             "actuator_gear", "actuator_gainprm", "actuator_biasprm",
             "qpos0", "qpos_spring", "M_rownnz", "M_rowadr", "M_colind", "mapM2M", "mapM2D", "D_rownnz", "D_rowadr",
             "D_colind", "D_diag", "B_rownnz", "B_rowadr", "B_colind", "dof_treeid", "body_treeid", "tree_bodyadr",
             "tree_bodynum", "tree_dofadr", "tree_dofnum",
             "jnt_actuatorid", "jnt_stiffnesspoly", "jnt_actfrclimited", "actuator_damping", "actuator_armature",
             "actuator_dampingpoly", "actuator_actadr", "actuator_actnum", "actuator_ctrladr", "actuator_outadr",
             "actuator_delay", "actuator_ctrlnum", "geom_adhesion", "body_geomadr", "body_geomnum", "body_mocapid",
             "body_contype", "body_conaffinity"]:
    if hasattr(m, name):
        fields[name] = np.asarray(getattr(m, name)).ravel().tolist()
o = m.opt
fields["opt"] = {k: (np.asarray(getattr(o, k)).ravel().tolist()) for k in
                 ["timestep", "gravity", "tolerance", "ls_tolerance", "noslip_tolerance", "impratio", "o_margin",
                  "o_solref", "o_solimp", "o_friction", "integrator", "cone", "jacobian", "solver", "iterations",
                  "ls_iterations", "noslip_iterations", "disableflags", "enableflags", "density", "viscosity", "wind",
                  "magnetic", "ccd_tolerance", "ccd_iterations", "sdf_iterations", "sdf_initpoints"] if hasattr(o, k)}
fields["stat"] = {"meaninertia": float(m.stat.meaninertia), "meanmass": float(m.stat.meanmass), "extent": float(m.stat.extent)}
fields["sizes"] = {k: int(getattr(m, k)) for k in ["nq", "nv", "nu", "nbody", "njnt", "ngeom", "nM", "nC", "nD", "nB",
                                                   "nexclude", "npair", "neq", "ntendon", "nflex", "nsite", "nplugin",
                                                   "na", "nmocap", "nsensor"] if hasattr(m, k)}
fields["frame_skip"] = int(u.frame_skip)
fields["mujoco"] = mujoco.__version__
with open(os.path.join(OUT, "model.json"), "w") as f:
    json.dump(fields, f, indent=0)
with open(os.path.join(OUT, "run.json"), "w") as f:
    json.dump({"seed": SEED, "steps": STEPS, "flip": args.flip}, f)


def flip(e):
    """Turns an env's cheetah upside down in place and recomputes its derived quantities.

    Args:
        e: the unwrapped MujocoEnv.
    """
    e.data.qpos[2] += np.pi
    mujoco.mj_forward(e.model, e.data)


# ---- a seeded episode, stepped as Gymnasium's do_simulation does, recording MuJoCo's state after every mj_step
obs, info = env.reset(seed=SEED)
if args.flip:
    flip(u)
env.action_space.seed(SEED)
actions = [env.action_space.sample() for _ in range(STEPS)]
init = (d.qpos.copy(), d.qvel.copy(), d.qacc_warmstart.copy(), float(d.time))

STATE = ["qpos", "qvel", "qacc", "qacc_warmstart", "qacc_smooth", "qfrc_constraint", "qfrc_passive", "qfrc_bias",
         "qfrc_actuator", "actuator_force"]
EFC = ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss", "efc_diagA", "efc_KBIP", "efc_D", "efc_R",
       "efc_vel", "efc_aref", "efc_b", "efc_force", "efc_state"]
DEBUG = ["xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "subtree_com", "cinert", "cdof", "crb",
         "M", "qLD", "qLDiagInv", "cvel", "cdof_dot"]
with open(os.path.join(OUT, "trace.bin"), "wb") as trace, open(os.path.join(OUT, "env.bin"), "wb") as envf:
    def put(f, arr):
        """Writes one record: the element count, then the values as little-endian float64.

        Args:
            f: the open binary file.
            arr: anything numpy converts to a float64 array.
        """
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).ravel()
        f.write(struct.pack("<i", a.size))
        f.write(a.tobytes())

    for a in init[:3]:
        put(trace, a)
    put(trace, [init[3]])
    # replay the same actions on a second env through Gymnasium's own step, for the env-level results
    env2 = gym.make("HalfCheetah-v5")
    obs2, _ = env2.reset(seed=SEED)
    if args.flip:
        flip(env2.unwrapped)
        obs2 = env2.unwrapped._get_obs()
    put(envf, obs2)
    for t, action in enumerate(actions):
        put(trace, action)
        d.ctrl[:] = action
        for k in range(u.frame_skip):
            mujoco.mj_step(m, d)
            for name in STATE:
                put(trace, getattr(d, name))
            put(trace, [d.time, d.ncon, d.nefc, d.solver_niter[0] if d.nisland else 0, d.nisland])
            put(trace, [d.contact.dist[i] for i in range(d.ncon)])
            put(trace, [d.contact.geom[i][j] for i in range(d.ncon) for j in range(2)])
            put(trace, [d.contact.pos[i][j] for i in range(d.ncon) for j in range(3)])
            put(trace, [d.contact.frame[i][j] for i in range(d.ncon) for j in range(9)])
            put(trace, d.efc_J[: d.nefc * m.nv] if d.nefc else [])
            for name in EFC:
                arr = getattr(d, name)
                put(trace, np.asarray(arr)[: d.nefc * (4 if name == "efc_KBIP" else 1)] if d.nefc else [])
            for name in DEBUG:
                put(trace, getattr(d, name))
        mujoco.mj_rnePostConstraint(m, d)
        obs2, r2, term2, trunc2, info2 = env2.step(action)
        put(envf, obs2)
        put(envf, [r2, float(term2), float(trunc2), info2["x_position"], info2["x_velocity"], info2["reward_forward"],
                   info2["reward_ctrl"]])
        if t < 3:
            # sanity: the manual stepping is Gymnasium's
            assert np.array_equal(d.qpos, env2.unwrapped.data.qpos) and np.array_equal(d.qvel, env2.unwrapped.data.qvel)
print(f"mujoco {mujoco.__version__}: seed {SEED}{' flipped' if args.flip else ''}, {STEPS} env steps, "
      f"{STEPS * u.frame_skip} mj_steps -> {OUT}")
