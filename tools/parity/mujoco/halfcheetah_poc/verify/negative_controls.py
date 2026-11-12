"""The negative controls of MUJOCO_ASSESSMENT.md §13.5 and §14: each copies the POC project, applies one plausible
"almost right" change to the pieces §13 added or MuJoCo 4.0.0 changed (many of the latter put back what 3.15.0 did),
builds it in Release and runs the checker on the traces that should tell, recorded with mujoco 4.0.0. A control passes
when its verdict matches the expected one: "detected" (the checker exits 1) or "same bits" (exit 0), the latter for
changes that provably cannot move a recorded bit on these models (the reason is in the list).

Usage:
    python negative_controls.py <poc folder> <scratch folder> <trace set folder> [control names...]
The trace set is record_traces.py's output (traces.txt's folder names); the scratch folder receives one project copy per
control.

Args (positional): the POC folder, the scratch folder for the copies, the trace set folder; optionally control names to
run only those.
Returns (exit code): 0 when every control run gave its expected verdict, 1 otherwise.
Raises:
    FileNotFoundError: the POC folder or a trace folder is missing.
"""
import os
import shutil
import subprocess
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
# the engine's sources, which the checker compiles (src/Gym.Environments/MuJoCo, namespace Gym.MuJoCo)
ENGINE = os.path.join(REPO, "src", "Gym.Environments", "MuJoCo")
ENGINE_FILES = {"MjData.cs", "Kernels.cs", "Ccd.cs", "MjModel.cs"}


def shift(text):
    """Moves a control's text into the engine's namespace block, four spaces deeper than the proof of concept had it.

    Args:
        text: the text as written for the file-scoped namespace: each line that starts a line gets four more spaces; a
            first line that begins mid-line (no leading space) stays as it is.
    Returns:
        The shifted text.
    """
    lines = text.split("\n")
    out = []
    for i, line in enumerate(lines):
        starts_line = i > 0 or line.startswith(" ")
        out.append(("    " + line) if starts_line and line else line)
    return "\n".join(out)

PUS = ["pus5_a", "pus5_b", "pus5_c", "pus5_d", "pus5_e", "pus5_f", "pus5_g"]
PUS_P3 = ["pus5_c", "pus5_e", "pus5_s654", "pus5_s1559", "pus5_s2455", "pus5_s20432", "pus5_s40228", "pus5_s60390"]
REA = ["rea5", "rea4"]
SWI = ["swi5", "swi4"]
# the RK4 envs, those with a free joint, and Hopper's episodes with two parallel capsules (§13.4's targeted seeds)
RK4 = ["hop5", "wal5", "ant5", "hum5", "idp5", "rea5", "swi5"]
FREE = ["ant5", "ant4", "hum5", "hst5"]
# Hopper lying down: on MuJoCo 4.0.0 its capsules fall within mjc_segWeights' 1e-8 parallel band in seeds 0 and 1 of the
# broad set (§14's search: 909 of 1,600 no-terminate episodes, never exactly parallel); 109 and 280 are 3.15.0's seeds
PARALLEL = ["hop5_nt", "hop5_nt_s109", "hop5_nt_s280"]
# the Pusher seeds whose multi-contact pass clips the capsule's edge (§14's search; the broad set has none)
PUS_MC = ["pus5_s94", "pus5_s418", "pus5_s654", "pus5_s2851", "pus5_s3016"]

# (name, file, exact old text, new text, traces, expected verdict, what it shows)
CONTROLS = [
    ("island-scale-over-all-dofs", "MjData.cs",
     "        for (int i = 0; i < snv; i++)\n            inertia += iM[iMrowadr[i] + iMrownnz[i] - 1];",
     "        for (int i = 0; i < nv; i++)\n            inertia += M[m.M_rowadr[i] + m.M_rownnz[i] - 1];",
     PUS + REA, "detected", "Newton's scale is the island's own inertia, not the model's"),
    ("unconstrained-dofs-keep-warmstart", "MjData.cs",
     "        for (int i = nidof; i < nv; i++) {\n            int dof = mapIdof2dof[i];\n            qacc[dof] = qacc_smooth[dof];\n        }\n",
     "", PUS + REA, "same bits",
     "the branch search: in the 7.3 million Pusher steps (80,000 episodes) and 0.64 million Reacher steps (40,000) that keep "
     "the warm start while a tree is in no island, every free dof's warm start already equals qacc_smooth (the goal and the "
     "target never move); when Pusher's arm is free with a different warm start, the smooth start wins and resets it anyway"),
    ("plane-cylinder-without-side-points", "MjData.cs",
     "            AddPlaneCylinderContact(d, pos2, vec1, 1, axis, 1, vec, -0.5, normal);\n"
     "            AddPlaneCylinderContact(d, pos2, vec1, -1, axis, 1, vec, -0.5, normal);\n",
     "", PUS, "detected", "the upright cylinder's three-point base"),
    ("convex-one-contact-and-multiccd-as-in-3.15", "MjData.cs",
     "        int maxContacts = MaxContacts(m.geom_type[g1], m.geom_type[g2]);",
     "        int maxContacts = 1;", PUS, "detected",
     "4.0's single pass asks a capsule's edge against a face for two contacts and runs no MultiCCD"),
    ("convex-shallow-multicontact-skipped", "Ccd.cs",
     "                } else if (maxContacts > 1 && status.Dist[0] < 0 && obj1.NSupp > 0) {",
     "                } else if (false) {", PUS + PUS_MC, "detected", "the multi-contact pass after a shallow contact"),
    ("convex-deep-multicontact-skipped", "Ccd.cs",
     "                if (maxContacts > 1 && face >= 0)\n                    status.Nx = Multicontact(",
     "                if (false)\n                    status.Nx = Multicontact(", PUS + PUS_MC, "detected",
     "the multi-contact pass after EPA's contact"),
    ("gjk-s2d-weights-over-the-minor-as-in-3.15", "Ccd.cs",
     "                double mDet = c31 + c32 + c33;\n                lambda[0] = c31 / mDet;",
     "                double mDet = mMax;\n                lambda[0] = c31 / mDet;",
     PUS, "detected", "4.0's S2D divides by the sum of the cofactors, not by the largest minor"),
    ("gjk-s2d-tie-taken", "Ccd.cs",
     "            double d = Dot3(p, p);\n            if (d < dmin) {\n                lambda[0] = l1[0];\n                lambda[1] = 0;",
     "            double d = Dot3(p, p);\n            if (d <= dmin) {\n                lambda[0] = l1[0];\n                lambda[1] = 0;",
     PUS, "same bits", "exact ties are common (59,516 of 425,129 edge comparisons in 20,000 Pusher episodes), but every "
     "one is at the two edges' shared vertex, where both edges give the same weights, so the rule cannot change a result"),
    ("epa-polytope3-normal-from-the-other-edge", "Ccd.cs",
     "            diff2[k] = V(pt, v3i)[k] - V(pt, v1i)[k];",
     "            diff2[k] = V(pt, v3i)[k] - V(pt, v2i)[k];",
     PUS_P3, "detected", "polytope3's normal reaches the contact (the targeted traces)"),
    ("epa-polytope3-supports-swapped", "Ccd.cs",
     "        int v5i = EpaSupport(pt, obj1, obj2, nNeg, nNorm);\n        int v4i = EpaSupport(pt, obj1, obj2, n, nNorm);",
     "        int v4i = EpaSupport(pt, obj1, obj2, n, nNorm);\n        int v5i = EpaSupport(pt, obj1, obj2, nNeg, nNorm);",
     PUS_P3, "same bits", "the order only renumbers two vertices; the support points and EPA's faces are the same"),
    ("fluid-torque-added-before-force", "MjData.cs",
     "        Kernels.MulMatTVec(fluidQforce, fluidJacp, force, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);\n"
     "        Kernels.MulMatTVec(fluidQforce, fluidJacr, torque, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);",
     "        Kernels.MulMatTVec(fluidQforce, fluidJacr, torque, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);\n"
     "        Kernels.MulMatTVec(fluidQforce, fluidJacp, force, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);",
     SWI, "detected", "mj_applyFT's order: the force's term, then the torque's"),
    ("fluid-face-drag-regrouped", "MjData.cs",
     "                lfrc[3] -= 0.5 * rho * box[1] * box[2] * Math.Abs(lvel[3]) * lvel[3];",
     "                lfrc[3] -= 0.5 * rho * box[1] * box[2] * (Math.Abs(lvel[3]) * lvel[3]);",
     SWI, "same bits", "the regrouped drag differs in its last bit in 32% of evaluations (Swimmer-v5's first 5 episodes), "
     "but it is small against the viscous force it is subtracted from, and lfrc[3] never changes"),
    ("fluid-wind-not-transformed", "MjData.cs",
     "            TransformSpatialMotion(lwind, wind, xipos.AsSpan(3 * i, 3), subtree_com.AsSpan(3 * m.body_rootid[i], 3), ximat.AsSpan(9 * i, 9));",
     "            lwind.Clear();", SWI, "same bits",
     "the zero wind's transform only signs zeros; Swimmer's local velocities are exactly zero in 3% of components (it is "
     "planar), but subtracting the transformed wind never changes a bit in 28.8 million components (200 episodes)"),
    ("simple-dofs-diagonal-computed", "MjData.cs",
     "            if (m.dof_simplenum[i] != 0) {",
     "            if (m.dof_simplenum[i] != 0 && false) {", PUS + REA, "same bits",
     "an axis-aligned slide on a leaf body: the composite inertia gives armature + mass exactly, the bits dof_M0 holds "
     "(checked in 561,200 evaluations over 200 Pusher and 200 Reacher episodes), and its couplings to other dofs are 0"),
    ("idp-squares-by-multiplication", "EnvRules.cs",
     "double distPenalty = 0.01 * Math.Pow(x, 2) + Math.Pow(y - 2, 2);",
     "double distPenalty = 0.01 * (x * x) + (y - 2) * (y - 2);", ["idp5", "idp4"], "detected",
     "NumPy's float64 ** 2 is C's pow, not a product"),
    ("reacher-v4-priced-after-step", "EnvRules.cs",
     "        if (v4) {\n            rewardDist = v4Dist;\n            rewardCtrl = v4Ctrl;\n        } else {",
     "        if (false) {\n            rewardDist = v4Dist;\n            rewardCtrl = v4Ctrl;\n        } else {", ["rea4"], "detected",
     "Reacher-v4 prices the state before the step"),
    ("ant-control-cost-left-to-right", "EnvRules.cs",
     "        if (n < 8) {\n            float r = 0;",
     "        if (n <= 8) {\n            float r = 0;", ["ant5", "ant4"], "detected",
     "NumPy's pairwise float32 sum from 8 entries (Ant's 8 actions)"),
    ("pusher-object-first-draw-kept", "EnvRules.cs",
     "        } while (!(Norm2(c0 - 0, c1 - 0) > 0.17));",
     "        } while (false);", PUS, "detected", "the object's rejection sampling at reset"),
    ("pusher-reward-summed-in-another-order", "EnvRules.cs",
     "        double reward = rewardDist + rewardCtrl + rewardNear;",
     "        double reward = rewardDist + rewardNear + rewardCtrl;", PUS, "detected", "the reward's summation order"),
    # MuJoCo 4.0.0's changes, most of them put back as 3.15.0 had them
    ("rk4-free-joints-uncorrected-as-in-3.15", "MjData.cs",
     "            RkCorrectVel(xi.Slice(nq, nv), dX.Slice(0, nv), h);\n", "", FREE, "detected",
     "4.0's RKMK correction of a free joint's angular velocity at each stage"),
    ("rk4-qacc-of-the-last-stage-as-in-3.15", "MjData.cs",
     "        Array.Copy(rkdX, nv, qacc, 0, nv);\n", "", RK4, "detected",
     "4.0 publishes the step's weighted acceleration as qacc"),
    ("rk4-warmstart-from-the-first-stage", "MjData.cs",
     "        Array.Copy(rkF, 3 * nv, qacc_warmstart, 0, nv);",
     "        Array.Copy(rkF, 0, qacc_warmstart, 0, nv);", RK4, "detected",
     "4.0 warm-starts the next step from the last stage's solve"),
    ("quat2mat-diagonal-left-to-right-as-in-3.15", "Kernels.cs",
     "        res[0] = (q00 - q22) + (q11 - q33);",
     "        res[0] = q00 + q11 - q22 - q33;", ["ant5", "hc5"], "detected",
     "4.0's AVX mju_quat2Mat groups the diagonal's squares"),
    ("mulquat-left-to-right-as-in-3.15", "Kernels.cs",
     "        double r0 = (qa[0] * qb[0] - qa[1] * qb[1]) + (-qa[2] * qb[2] - qa[3] * qb[3]);",
     "        double r0 = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3];", ["ant5", "hc5"], "detected",
     "4.0's AVX mju_mulQuat sums its products in pairs"),
    ("normalize4-left-to-right-as-in-3.15", "Kernels.cs",
     "Math.Sqrt((q[0] * q[0] + q[1] * q[1]) + (q[2] * q[2] + q[3] * q[3]))",
     "Math.Sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])", FREE, "detected",
     "4.0's mju_normalize4 adds its squares in pairs"),
    ("cholsolve-backward-by-rows-as-in-3.15", "Kernels.cs",
     "        for (int i = n - 1; i >= 0; i--) {\n            res[i] /= mat[i * (n + 1)];\n            if (i != 0 && res[i] != 0) {\n"
     "                double scl = -res[i];\n                for (int k = 0; k < i; k++)\n                    res[k] += mat[i * n + k] * scl;\n"
     "            }\n        }",
     "        for (int i = n - 1; i >= 0; i--) {\n            for (int k = i + 1; k < n; k++)\n                res[i] -= mat[k * n + i] * res[k];\n"
     "            res[i] /= mat[i * (n + 1)];\n        }",
     ["hop5", "wal5", "ant5", "pus5_a"], "detected", "4.0's dense mju_cholSolve substitutes backward by columns"),
    ("pgs-scale-from-the-mean-inertia-as-in-3.15", "MjData.cs",
     "        double pgsScale = 1 / MaxFirst(Kernels.MinVal, trace);",
     "        double pgsScale = 1 / MaxFirst(Kernels.MinVal, m.meaninertia * nv);", ["hum5", "hst5"], "detected",
     "4.0's PGS scales its improvement by the island's inertia trace"),
    ("capsules-parallel-one-pair", "MjData.cs",
     "        ClampSeg(w, 2, 1, (dot5 + dot3) / dot2, s0, s1);\n        return 2;",
     "        ClampSeg(w, 2, 1, (dot5 + dot3) / dot2, s0, s1);\n        return 1;", PARALLEL, "detected",
     "mjc_segWeights' second pair for parallel capsules"),
    ("capsules-parallel-only-when-exact", "MjData.cs",
     "        if (det > sin2 * dot1 * dot2) {",
     "        if (det > 0) {", PARALLEL, "detected",
     "mjc_segWeights' 1e-8 band: Hopper's capsules fall inside it without being exactly parallel"),
    ("multicontact-cylinder-margin-not-inflated", "Ccd.cs",
     "            Inflate(nx, x1, x2, dist, margin1, margin2, normal);\n", "", PUS_MC, "detected",
     "the clipped contacts are inflated by the cylinder's half margin"),
    ("polygon-clip-edge-keeps-its-first-two-points", "Ccd.cs",
     "            double dd = 0;",
     "            double dd = double.MaxValue;", PUS_MC, "detected",
     "a clipped edge keeps the two points farthest apart"),
    ("polygon-clip-margin-not-pruned", "Ccd.cs",
     "            if (d <= margin) {",
     "            if (true) {", PUS_MC, "detected",
     "a clipped point farther from the disk than the margin is dropped"),
]


def run_control(poc, work, traces, control):
    """Builds one control's copy and runs the checker on its traces.

    Args:
        poc: the POC folder.
        work: the scratch folder.
        traces: the trace set folder.
        control: one CONTROLS entry.
    Returns:
        (verdict, detail): "detected", "same bits", "not applied" or "build failed", and the checker's totals and first
        difference (or the reason).
    """
    name, file, old, new, names = control[:5]
    dst = os.path.join(work, name)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "Engine"))
    for f in os.listdir(poc):
        if f.endswith(".cs") or f.endswith(".csproj"):
            shutil.copy(os.path.join(poc, f), dst)
    # the engine is the product's source; the copy patches its own copy of it
    for f in os.listdir(ENGINE):
        if f.endswith(".cs"):
            shutil.copy(os.path.join(ENGINE, f), os.path.join(dst, "Engine"))
    engine_file = file in ENGINE_FILES
    path = os.path.join(dst, "Engine", file) if engine_file else os.path.join(dst, file)
    if engine_file:
        old, new = shift(old), shift(new)
    with open(path, encoding="utf-8", newline="") as fh:
        src = fh.read().replace("\r\n", "\n")
    if src.count(old) != 1:
        return "not applied", f"the old text occurs {src.count(old)} times"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(src.replace(old, new))
    b = subprocess.run(["dotnet", "build", "-c", "Release", "-nologo", "-v", "q", f"-p:GymRoot={REPO}{os.sep}",
                        f"-p:MuJoCoSource={os.path.join(dst, 'Engine')}"], cwd=dst, capture_output=True, text=True)
    if b.returncode != 0:
        return "build failed", b.stdout[-800:]
    exe = os.path.join(dst, "bin", "Release", "net10.0", "HalfCheetahPoc.exe")
    r = subprocess.run([exe] + [os.path.join(traces, t) for t in names], capture_output=True, text=True)
    lines = r.stdout.splitlines()
    firsts = [f"{l.split(':')[0]}: {l.split('first difference: ')[1]}" for l in lines if "first difference: " in l]
    total = next((l for l in lines if l.startswith("total:")), r.stdout[-300:])
    return ("detected" if r.returncode == 1 else "same bits"), f"{total}; first: {firsts[0] if firsts else '-'}"


def main():
    """Runs the controls and prints one line per control with its verdict against the expected one.

    Returns:
        The exit code: 0 when every control gave its expected verdict.
    """
    poc, work, traces = sys.argv[1], sys.argv[2], sys.argv[3]
    only = set(sys.argv[4:])
    bad = 0
    for control in CONTROLS:
        name, expected, why = control[0], control[5], control[6]
        if only and name not in only:
            continue
        verdict, detail = run_control(poc, work, traces, control)
        ok = verdict == expected
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name}: {verdict} (expected {expected}: {why}); {detail}")
        sys.stdout.flush()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
