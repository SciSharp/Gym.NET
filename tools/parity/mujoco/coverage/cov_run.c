// Replays an episode plan (gen_plans.py) on a coverage-instrumented MuJoCo, the way Gymnasium's MujocoEnv steps:
// mj_resetData, the reset state, mj_forward (set_state); then per env step ctrl = action, mj_step x frame_skip, and
// mj_rnePostConstraint when the plan asks for it. With --setup-only it loads the model, makes the data and exits:
// subtracting that run's line counts from a full run's leaves exactly what the episodes executed.
//
// Usage: cov_run <model.mjb> <plan> [--setup-only]
// Exit codes: 0 done; 1 bad arguments, an unreadable model or plan, or a plan that does not match the model.
#include <mujoco/mujoco.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// reads exactly n bytes into p, or ends the run: a short read means the plan was cut off
static void readall(FILE* f, void* p, size_t n) {
  if (fread(p, 1, n, f) != n) {
    fprintf(stderr, "truncated plan\n");
    exit(1);
  }
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: cov_run <model.mjb> <plan> [--setup-only]\n");
    return 1;
  }
  // an MJB is MuJoCo's own compiled model: loading it runs no MJCF compiler code
  mjModel* m = mj_loadModel(argv[1], NULL);
  if (!m) {
    fprintf(stderr, "cannot load %s\n", argv[1]);
    return 1;
  }
  mjData* d = mj_makeData(m);
  if (argc > 3 && !strcmp(argv[3], "--setup-only")) {
    mj_deleteData(d);
    mj_deleteModel(m);
    return 0;
  }
  FILE* f = fopen(argv[2], "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", argv[2]);
    return 1;
  }
  int hdr[6];
  readall(f, hdr, sizeof(hdr));
  int nq = hdr[0], nv = hdr[1], nu = hdr[2], frame_skip = hdr[3], rnepost = hdr[4], neps = hdr[5];
  if (nq != m->nq || nv != m->nv || nu != m->nu) {
    fprintf(stderr, "plan does not match the model\n");
    return 1;
  }
  long physics = 0;
  for (int e = 0; e < neps; e++) {
    int nsteps;
    readall(f, &nsteps, sizeof(int));
    double* q0 = malloc(sizeof(double) * nq);
    double* v0 = malloc(sizeof(double) * nv);
    double* acts = malloc(sizeof(double) * (size_t)nsteps * nu);
    if (!q0 || !v0 || !acts) {
      fprintf(stderr, "out of memory\n");
      return 1;
    }
    readall(f, q0, sizeof(double) * nq);
    readall(f, v0, sizeof(double) * nv);
    readall(f, acts, sizeof(double) * (size_t)nsteps * nu);
    // Gymnasium's reset: mj_resetData, then set_state (qpos, qvel, mj_forward)
    mj_resetData(m, d);
    memcpy(d->qpos, q0, sizeof(double) * nq);
    memcpy(d->qvel, v0, sizeof(double) * nv);
    mj_forward(m, d);
    for (int t = 0; t < nsteps; t++) {
      memcpy(d->ctrl, acts + (size_t)t * nu, sizeof(double) * nu);
      for (int k = 0; k < frame_skip; k++) {
        mj_step(m, d);
        physics++;
      }
      if (rnepost) {
        mj_rnePostConstraint(m, d);
      }
    }
    free(q0);
    free(v0);
    free(acts);
  }
  fclose(f);
  printf("%d episodes, %ld physics steps\n", neps, physics);
  mj_deleteData(d);
  mj_deleteModel(m);
  return 0;
}
