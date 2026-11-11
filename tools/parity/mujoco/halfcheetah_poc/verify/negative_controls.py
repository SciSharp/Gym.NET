"""The negative controls of MUJOCO_ASSESSMENT.md §13.5: each copies the POC project, applies one plausible "almost right"
change to the pieces §13 added, builds it in Release and runs the checker on the traces that should tell. A control
passes when its verdict matches the expected one: "detected" (the checker exits 1) or "same bits" (exit 0), the latter
for changes that provably cannot move a recorded bit on these models (the reason is in the list).

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

PUS = ["pus5_a", "pus5_b", "pus5_c", "pus5_d", "pus5_e", "pus5_f", "pus5_g"]
PUS_P3 = ["pus5_c", "pus5_e", "pus5_s654", "pus5_s1559", "pus5_s2455", "pus5_s20432", "pus5_s40228", "pus5_s60390"]
REA = ["rea5", "rea4"]
SWI = ["swi5", "swi4"]

# (name, file, exact old text, new text, traces, expected verdict, what it shows)
CONTROLS = [
    ("island-scale-over-all-dofs", "CheetahCore.cs",
     "        for (int i = 0; i < snv; i++)\n            inertia += iM[iMrowadr[i] + iMrownnz[i] - 1];",
     "        for (int i = 0; i < nv; i++)\n            inertia += M[m.M_rowadr[i] + m.M_rownnz[i] - 1];",
     PUS + REA, "detected", "Newton's scale is the island's own inertia, not the model's"),
    ("unconstrained-dofs-keep-warmstart", "CheetahCore.cs",
     "        for (int i = nidof; i < nv; i++) {\n            int dof = mapIdof2dof[i];\n            qacc[dof] = qacc_smooth[dof];\n        }\n",
     "", PUS + REA, "same bits",
     "the branch search: in the 7.3 million Pusher steps (80,000 episodes) and 0.64 million Reacher steps (40,000) that keep "
     "the warm start while a tree is in no island, every free dof's warm start already equals qacc_smooth (the goal and the "
     "target never move); when Pusher's arm is free with a different warm start, the smooth start wins and resets it anyway"),
    ("plane-cylinder-without-side-points", "CheetahCore.cs",
     "            AddPlaneCylinderContact(d, pos2, vec1, 1, axis, 1, vec, -0.5, normal);\n"
     "            AddPlaneCylinderContact(d, pos2, vec1, -1, axis, 1, vec, -0.5, normal);\n",
     "", PUS, "detected", "the upright cylinder's three-point base"),
    ("convex-without-multiccd", "CheetahCore.cs",
     "        int n = Penetration(first, margin);\n        if (n == 1) {",
     "        int n = Penetration(first, margin);\n        if (n == -1) {", PUS, "detected", "MultiCCD's perturbed contacts"),
    ("gjk-s2d-m14-summed-in-another-order", "Kernels.cs",
     "        double m14 = s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1] + s1[1] * s2[2] - s1[2] * s2[1];",
     "        double m14 = s1[1] * s2[2] - s1[2] * s2[1] + s2[1] * s3[2] - s2[2] * s3[1] - s1[1] * s3[2] + s1[2] * s3[1];",
     PUS, "detected", "the signed-volume subalgorithm's cofactors, term by term"),
    ("gjk-s2d-tie-taken", "Kernels.cs",
     "            double d = Dot3(p, p);\n            if (d < dmin) {\n                lambda[0] = l1[0];\n                lambda[1] = 0;",
     "            double d = Dot3(p, p);\n            if (d <= dmin) {\n                lambda[0] = l1[0];\n                lambda[1] = 0;",
     PUS, "same bits", "exact ties are common (59,516 of 425,129 edge comparisons in 20,000 Pusher episodes), but every "
     "one is at the two edges' shared vertex, where both edges give the same weights, so the rule cannot change a result"),
    ("epa-polytope3-normal-from-the-other-edge", "Kernels.cs",
     "            diff1[k] = sx[VertexSize + k] - sx[k];\n            diff2[k] = sx[2 * VertexSize + k] - sx[k];",
     "            diff1[k] = sx[VertexSize + k] - sx[k];\n            diff2[k] = sx[2 * VertexSize + k] - sx[VertexSize + k];",
     PUS_P3, "detected", "polytope3's normal reaches the contact, in seeds 654 and 1559 (the targeted traces)"),
    ("epa-polytope3-supports-swapped", "Kernels.cs",
     "        int v5i = EpaSupport(pt, obj1, obj2, nNeg, nNorm);\n        int v4i = EpaSupport(pt, obj1, obj2, n, nNorm);",
     "        int v4i = EpaSupport(pt, obj1, obj2, n, nNorm);\n        int v5i = EpaSupport(pt, obj1, obj2, nNeg, nNorm);",
     PUS_P3, "same bits", "the order only renumbers two vertices; the support points and EPA's faces are the same"),
    ("fluid-torque-added-before-force", "CheetahCore.cs",
     "        Kernels.MulMatTVec(fluidQforce, fluidJacp, force, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);\n"
     "        Kernels.MulMatTVec(fluidQforce, fluidJacr, torque, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);",
     "        Kernels.MulMatTVec(fluidQforce, fluidJacr, torque, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);\n"
     "        Kernels.MulMatTVec(fluidQforce, fluidJacp, force, 3, nv);\n        Kernels.AddTo(target, fluidQforce, nv);",
     SWI, "detected", "mj_applyFT's order: the force's term, then the torque's"),
    ("fluid-face-drag-regrouped", "CheetahCore.cs",
     "                lfrc[3] -= 0.5 * rho * box[1] * box[2] * Math.Abs(lvel[3]) * lvel[3];",
     "                lfrc[3] -= 0.5 * rho * box[1] * box[2] * (Math.Abs(lvel[3]) * lvel[3]);",
     SWI, "same bits", "the regrouped drag differs in its last bit in 32% of evaluations (Swimmer-v5's first 5 episodes), "
     "but it is small against the viscous force it is subtracted from, and lfrc[3] never changes"),
    ("fluid-wind-not-transformed", "CheetahCore.cs",
     "            TransformSpatialMotion(lwind, wind, xipos.AsSpan(3 * i, 3), subtree_com.AsSpan(3 * m.body_rootid[i], 3), ximat.AsSpan(9 * i, 9));",
     "            lwind.Clear();", SWI, "same bits",
     "the zero wind's transform only signs zeros; Swimmer's local velocities are exactly zero in 3% of components (it is "
     "planar), but subtracting the transformed wind never changes a bit in 28.8 million components (200 episodes)"),
    ("simple-dofs-diagonal-computed", "CheetahCore.cs",
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
    os.makedirs(dst)
    for f in os.listdir(poc):
        if f.endswith(".cs") or f.endswith(".csproj"):
            shutil.copy(os.path.join(poc, f), dst)
    path = os.path.join(dst, file)
    with open(path, encoding="utf-8", newline="") as fh:
        src = fh.read()
    if src.count(old) != 1:
        return "not applied", f"the old text occurs {src.count(old)} times"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(src.replace(old, new))
    b = subprocess.run(["dotnet", "build", "-c", "Release", "-nologo", "-v", "q"], cwd=dst, capture_output=True, text=True)
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
