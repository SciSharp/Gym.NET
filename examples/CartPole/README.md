# CartPole: learning to balance with the cross-entropy method

Gym.NET's example: it learns CartPole-v1 with no machine-learning library, using only Gym.NET's 1.0 API, and then plays
the learned policy.

```bash
dotnet run --project examples/CartPole -c Release              # train, then play one episode headless
dotnet run --project examples/CartPole -c Release -- --render  # play the final episode in a window (Avalonia)
```

Output of a run (the same on every run, since every draw is seeded):

```
generation  0: population   15.6, elite   56.3, learned policy   60.5
generation  1: population   60.4, elite  174.9, learned policy   73.5
generation  2: population   98.0, elite  418.5, learned policy  195.0
generation  3: population  160.4, elite  500.0, learned policy  334.0
generation  4: population  172.6, elite  500.0, learned policy  436.4
generation  5: population  265.5, elite  500.0, learned policy  438.3
generation  6: population  263.5, elite  500.0, learned policy  500.0
Solved: policy [0.15, 1.03, 2.24, 1.75, 0.00] played a new episode for a return of 500 (threshold 475).
```

## What it does

A policy is five numbers: a weight for each observation entry (cart position, cart velocity, pole angle, pole angular
velocity) and a bias. It pushes the cart right when the weighted sum is positive, left otherwise.

The cross-entropy method searches for those numbers:

1. Draw 100 policies from a Gaussian (mean 0, standard deviation 1 at the start).
2. Play one episode with each. A vector env plays all 100 at once: `Gymnasium.MakeVec("CartPole-v1", 100)` builds
   `CartPoleVectorEnv`, CartPole's own vectorized env, which steps every cart in one call.
3. Refit the Gaussian to the 10 best policies, and add a little noise so the search keeps exploring.
4. Judge the Gaussian's mean, the learned policy, on 100 episodes from starting states no generation was selected on.
   It is solved when they average at least 475, the reward threshold CartPole-v1 is registered with
   (`Gymnasium.Spec("CartPole-v1").RewardThreshold`).

At the end, `Gymnasium.Make("CartPole-v1")` builds the env through the registry, with `make`'s wrappers
(`TimeLimit` of 500 steps, `OrderEnforcing`, `PassiveEnvChecker`), and the learned policy plays one more episode.
With `--render`, that env is built with `render_mode = "human"`, and the frames go to the window that
`Gymnasium.HumanViewerFactory = AvaloniaEnvViewer.Factory` provides.

## The API it shows

| Call | What it returns |
|---|---|
| `Gymnasium.Make<NDArray, NDArray>("CartPole-v1", kwargs)` | An `Env<NDArray, NDArray>` with `make`'s wrappers; `kwargs` are constructor arguments by Python name (`new { render_mode = "human" }`) |
| `env.Reset(seed: s)` | `ResetResult<NDArray>`: `(Observation, Info)`; a seed reseeds the env's generator |
| `env.Step(action)` | `StepResult<NDArray>`: `(Observation, Reward, Terminated, Truncated, Info)`; the episode ends on either flag |
| `Gymnasium.MakeVec<NDArray, NDArray>("CartPole-v1", n)` | A `VectorEnv<NDArray, NDArray>` of `n` carts; `Step` takes `n` actions and returns `VectorStepResult<NDArray>` of batched arrays |
| `Seeding.NpRandom(seed)` | A NumPy-exact PCG64 `Generator`, as `gymnasium.utils.seeding.np_random` |

Arrays a step returns belong to the caller: the example disposes each one when the next replaces it.

## Using it outside this repository

The project references Gym.NET's source projects so it always builds against the current tree (`Gym.NET.sln` builds
it). In your own project, reference the packages instead:

```bash
dotnet add package Gym.NET.Environments        # Gym.NET, the environments, their registration
dotnet add package Gym.NET.Rendering.Avalonia  # only for --render
```

and drop the `Import` of `Gym.NET.Environments.targets` from the project file: the package brings it, and it registers
the built-in environment ids before `Main` runs.
