"""Oracle for the from-scratch core (Gymnasium's 21 MuJoCo ids that run on MuJoCo 3 and 4, and any other MuJoCo env
physics only): records the installed MuJoCo's exact state (the core follows 4.0.0, DR-64) after every mj_step of seeded episodes, and Gymnasium's env results, as
binary files the C# checker compares against bit for bit.

Writes into the output folder:
    model.json  every compiled model array the cores read, as exact doubles (Python's repr round-trips), plus the
                options, the statistics, the sizes, frame_skip and the body names.
    run.json    {"env", "seeds", "steps" (env steps per episode), "flip", "terminate", "frame_skip", "debug_episodes",
                "info_keys" ([key, size] in the order env.bin stores them), "max_episode_steps" (the TimeLimit)}.
    trace.bin   records of an int32 count and that many little-endian float64. First [3, n_episodes] (the format and
                the episode count). Per episode: [n_steps, has_debug]; qpos, qvel, qacc_warmstart and [time] at the
                start; then per env step the action, and per mj_step (frame_skip of them) the state block (STATE below),
                [time, ncon, nefc, solver_niter, nisland], the contacts (dist, geom pairs, pos, frame), efc_J, the efc
                arrays (EFC below), and when has_debug the dual matrix efc_AR (dual solvers, else empty) and the
                position/velocity-stage arrays (DEBUG below); after the env step's mj_rnePostConstraint, POST below
                (the external forces, tendons, sites and body positions the envs read). Format 3 added sites and body
                positions to POST; the checker reads format 3 only.
    env.bin     per episode the observation after reset, then per env step the observation and [reward, terminated,
                truncated, the info values in info_keys order], from a second env stepped through Gymnasium.

Args (command line):
    --out DIR         output folder (required; HalfCheetah writes about 40 MB per 1,000 env steps, Humanoid more: keep it
                      outside the repo).
    --env ID          any Gymnasium MuJoCo id (default HalfCheetah-v5), or Point-v0: Gymnasium's custom-env test PointEnv on
                      point.xml, which this script registers. The checker has env rules (EnvRules.cs) for the 21 ids that run
                      on MuJoCo 3 and 4 (every v4 and v5 but Pusher-v4, which refuses MuJoCo 3 and later), and checks any
                      other id physics only.
    --seed N          the first episode's reset and action-space seed (default 42); episode k uses seed + k.
    --episodes N      how many episodes (default 1).
    --steps N         the most env steps per episode (default 1000; Gymnasium's TimeLimit truncates at 1,000, Reacher's
                      at 50 and Pusher's at 100).
    --flip            turn the model upside down after the reset (qpos[2] += pi: the planar walkers, whose qpos[2] is the
                      root's pitch hinge), so its body lands on the ground; both envs get the same start.
    --no-terminate    pass terminate_when_unhealthy=False (Hopper, Walker2d, Ant, Humanoid), so episodes run past a fall.
    --debug-episodes N  record efc_AR and the DEBUG arrays for the first N episodes only (default: all), to bound the size.

Raises:
    AssertionError: the manual stepping stops matching Gymnasium's own step (checked on each episode's first steps).
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
ap.add_argument("--env", default="HalfCheetah-v5")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--episodes", type=int, default=1)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--flip", action="store_true")
ap.add_argument("--no-terminate", action="store_true")
ap.add_argument("--debug-episodes", type=int, default=-1)
args = ap.parse_args()
OUT = args.out
os.makedirs(OUT, exist_ok=True)
KWARGS = {"terminate_when_unhealthy": False} if args.no_terminate else {}
DEBUG_EPISODES = args.episodes if args.debug_episodes < 0 else args.debug_episodes


def register_point_env():
    """Registers Point-v0: the PointEnv of Gymnasium's custom-env test (tests/envs/mujoco/test_mujoco_custom_env.py) on
    point.xml, the one bundled model no Gymnasium id uses, so a trace can hold it (the checker runs it physics only: it has
    no env rules). Its reset is the reference state without noise, its reward the root's x displacement, its info empty.
    """
    from gymnasium.envs.mujoco import MujocoEnv
    from gymnasium.spaces import Box

    class PointEnv(MujocoEnv):
        """Gymnasium's test PointEnv, without EzPickle."""

        metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"]}

        def __init__(self, xml_file="point.xml", frame_skip=1, **kwargs):
            """Loads the model and sets the spaces and the frame rate.

            Args:
                xml_file: the model.
                frame_skip: physics steps per env step.
                **kwargs: MujocoEnv's keyword arguments.
            """
            MujocoEnv.__init__(self, xml_file, frame_skip=frame_skip, observation_space=None, default_camera_config={},
                               **kwargs)
            self.metadata = {"render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple"],
                             "render_fps": int(np.round(1.0 / self.dt))}
            obs_size = self.data.qpos.size + self.data.qvel.size
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)

        def step(self, action):
            """One env step.

            Args:
                action: the controls.
            Returns:
                The observation, the x displacement as reward, False, False and an empty info.
            """
            x_position_before = self.data.qpos[0]
            self.do_simulation(action, self.frame_skip)
            x_position_after = self.data.qpos[0]
            return self._get_obs(), x_position_after - x_position_before, False, False, {}

        def _get_obs(self):
            """The positions then the velocities.

            Returns:
                The observation.
            """
            return np.concatenate((self.data.qpos.flat.copy(), self.data.qvel.flat.copy()))

        def reset_model(self):
            """The reference state, no noise.

            Returns:
                The first observation.
            """
            self.set_state(self.init_qpos, self.init_qvel)
            return self._get_obs()

    gym.register("Point-v0", entry_point=PointEnv, max_episode_steps=1000)


if args.env == "Point-v0":
    register_point_env()
env = gym.make(args.env, **KWARGS)
u = env.unwrapped
m, d = u.model, u.data

# ---- the compiled model, every array the cores read
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
             "body_contype", "body_conaffinity",
             # tendons (Humanoid's two fixed ones)
             "tendon_adr", "tendon_num", "tendon_limited", "tendon_actfrclimited", "tendon_range",
             "tendon_stiffness", "tendon_stiffnesspoly", "tendon_damping", "tendon_dampingpoly", "tendon_armature",
             "tendon_frictionloss", "tendon_lengthspring", "tendon_margin", "tendon_actuatorid",
             "wrap_type", "wrap_objid", "wrap_prm", "ten_J_rownnz", "ten_J_rowadr", "ten_J_colind",
             # sites (InvertedDoublePendulum's tip) and the geoms' fluid parameters (Swimmer's inertia-box model)
             "site_bodyid", "site_pos", "site_quat", "site_sameframe", "geom_fluid"]:
    if hasattr(m, name):
        fields[name] = np.asarray(getattr(m, name)).ravel().tolist()
# body names, for the envs that read bodies by name (Reacher's fingertip, Pusher's object, Ant's torso)
fields["body_names"] = [m.body(i).name for i in range(m.nbody)]
o = m.opt
fields["opt"] = {k: (np.asarray(getattr(o, k)).ravel().tolist()) for k in
                 ["timestep", "gravity", "tolerance", "ls_tolerance", "noslip_tolerance", "impratio", "o_margin",
                  "o_solref", "o_solimp", "o_friction", "integrator", "cone", "jacobian", "solver", "iterations",
                  "ls_iterations", "noslip_iterations", "disableflags", "enableflags", "density", "viscosity", "wind",
                  "magnetic", "ccd_tolerance", "ccd_iterations", "sdf_iterations", "sdf_initpoints"] if hasattr(o, k)}
fields["stat"] = {"meaninertia": float(m.stat.meaninertia), "meanmass": float(m.stat.meanmass), "extent": float(m.stat.extent)}
fields["sizes"] = {k: int(getattr(m, k)) for k in ["nq", "nv", "nu", "nbody", "njnt", "ngeom", "nM", "nC", "nD", "nB",
                                                   "nexclude", "npair", "neq", "ntendon", "nwrap", "nJten", "nflex",
                                                   "nsite", "nplugin", "na", "nmocap", "nsensor"]
                   if hasattr(m, k)}
fields["frame_skip"] = int(u.frame_skip)
fields["mujoco"] = mujoco.__version__
with open(os.path.join(OUT, "model.json"), "w") as f:
    json.dump(fields, f, indent=0)


def flip(e):
    """Turns an env's model upside down in place and recomputes its derived quantities.

    Args:
        e: the unwrapped MujocoEnv.
    """
    e.data.qpos[2] += np.pi
    mujoco.mj_forward(e.model, e.data)


def info_vector(info, keys):
    """Flattens an info dict in a fixed key order.

    Args:
        info: the step's info dict.
        keys: the [key, size] pairs.
    Returns:
        The values as one float64 list.
    """
    return [float(v) for k, _ in keys for v in np.ravel(np.asarray(info[k], dtype=np.float64))]


STATE = ["qpos", "qvel", "qacc", "qacc_warmstart", "qacc_smooth", "qfrc_constraint", "qfrc_passive", "qfrc_bias",
         "qfrc_actuator", "actuator_force"]
EFC = ["efc_type", "efc_id", "efc_pos", "efc_margin", "efc_frictionloss", "efc_diagA", "efc_KBIP", "efc_D", "efc_R",
       "efc_vel", "efc_aref", "efc_b", "efc_force", "efc_state"]
DEBUG = ["xpos", "xquat", "xmat", "xipos", "ximat", "geom_xpos", "geom_xmat", "subtree_com", "cinert", "cdof", "crb",
         "M", "qLD", "qLDiagInv", "cvel", "cdof_dot"]
POST = ["cfrc_ext", "ten_length", "ten_velocity", "site_xpos", "site_xmat", "xpos"]
DUAL = m.opt.solver == mujoco.mjtSolver.mjSOL_PGS or m.opt.noslip_iterations > 0

seeds, lengths, info_keys = [], [], None
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

    put(trace, [3, args.episodes])
    env2 = gym.make(args.env, **KWARGS)
    for e in range(args.episodes):
        seed = args.seed + e
        seeds.append(seed)
        debug = e < DEBUG_EPISODES
        env.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        if args.flip:
            flip(u)
            flip(env2.unwrapped)
            obs2 = env2.unwrapped._get_obs()
        env.action_space.seed(seed)
        # the episode is recorded in memory first: its length is only known at its end
        records, envrecs = [], [obs2]
        records.append(d.qpos.copy())
        records.append(d.qvel.copy())
        records.append(d.qacc_warmstart.copy())
        records.append([float(d.time)])
        t = 0
        while t < args.steps:
            action = env.action_space.sample()
            records.append(action)
            d.ctrl[:] = action
            for k in range(u.frame_skip):
                mujoco.mj_step(m, d)
                for name in STATE:
                    records.append(getattr(d, name).copy())
                records.append([d.time, d.ncon, d.nefc, d.solver_niter[0] if d.nisland else 0, d.nisland])
                records.append([d.contact.dist[i] for i in range(d.ncon)])
                records.append([d.contact.geom[i][j] for i in range(d.ncon) for j in range(2)])
                records.append([d.contact.pos[i][j] for i in range(d.ncon) for j in range(3)])
                records.append([d.contact.frame[i][j] for i in range(d.ncon) for j in range(9)])
                records.append(d.efc_J[: d.nefc * m.nv].copy() if d.nefc else [])
                for name in EFC:
                    arr = getattr(d, name)
                    records.append(np.asarray(arr)[: d.nefc * (4 if name == "efc_KBIP" else 1)].copy() if d.nefc else [])
                if debug:
                    records.append(np.asarray(d.efc_AR)[: d.nefc * d.nefc].copy() if (DUAL and d.nefc) else [])
                    for name in DEBUG:
                        records.append(getattr(d, name).copy())
            mujoco.mj_rnePostConstraint(m, d)
            for name in POST:
                records.append(getattr(d, name).copy())
            obs2, r2, term2, trunc2, info2 = env2.step(action)
            if info_keys is None:
                info_keys = [[k, int(np.size(v))] for k, v in info2.items()]
            envrecs.append(obs2)
            envrecs.append([r2, float(term2), float(trunc2)] + info_vector(info2, info_keys))
            if t < 3:
                # sanity: the manual stepping is Gymnasium's
                assert np.array_equal(d.qpos, env2.unwrapped.data.qpos) and np.array_equal(d.qvel, env2.unwrapped.data.qvel)
            t += 1
            if term2 or trunc2:
                break
        lengths.append(t)
        put(trace, [t, 1.0 if debug else 0.0])
        for r in records:
            put(trace, r)
        for r in envrecs:
            put(envf, r)

with open(os.path.join(OUT, "run.json"), "w") as f:
    json.dump({"env": args.env, "seeds": seeds, "steps": lengths, "flip": args.flip, "terminate": not args.no_terminate,
               "frame_skip": int(u.frame_skip), "debug_episodes": DEBUG_EPISODES, "info_keys": info_keys,
               "max_episode_steps": env.spec.max_episode_steps}, f)
print(f"mujoco {mujoco.__version__}: {args.env} seeds {seeds[0]}..{seeds[-1]}{' flipped' if args.flip else ''}"
      f"{' no-terminate' if args.no_terminate else ''}: {len(lengths)} episodes, {sum(lengths)} env steps "
      f"(lengths min {min(lengths)} max {max(lengths)}), {sum(lengths) * u.frame_skip} mj_steps -> {OUT}")
