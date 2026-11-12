# FrozenLake8x8: value iteration

The 8×8 map of FrozenLake: a longer way across the same slippery ice, with more holes.

```bash
dotnet run --project examples/FrozenLake8x8 -c Release                   # one episode, headless
dotnet run --project examples/FrozenLake8x8 -c Release -- --render       # the same episode in a window
dotnet run --project examples/FrozenLake8x8 -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/FrozenLake8x8 -c Release -- --episodes 5   # seeds 2-6, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
FrozenLake8x8-v1, seed 2: 63 steps, return 1.0, reached the goal
```

## The agent

Value iteration over the env's own transition table, `P`: for every state and action it lists the possible next states
with their probabilities and rewards. The agent sweeps the states, updating each one's value to the best expected reward
plus 0.99 times the next state's value, until nothing moves by more than 1e-10, then always takes the action with the
best value. From seed 2 it reaches the goal in 63 steps.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Arrow keys | Move left, down, right or up (the ice slips sideways two times in three) |

## The API it shows

- `((FrozenLakeEnv)env.Unwrapped).P`: the transition table, Gymnasium's `P[state][action]` list of `(probability,
  next_state, reward, terminated)`, as `ToyTextTransition` records
- `Play.Run(..., waitOnPlayer: true)`: a turn-based game, one step per key press
- `env.Reset(seed: s)` and `env.Step(action)`: their results own the arrays they hold (D19), so the example releases
  each one once the agent has acted on it

## Outside this repository

The project references Gym.NET's source projects so it always builds against the current tree (`Gym.NET.sln` builds it).
In your own project, reference the packages instead:

```bash
dotnet add package Gym.NET.Environments        # Gym.NET, the environments, their registration
dotnet add package Gym.NET.Rendering.Avalonia  # only for --render and --play
```

and drop the `Import` of `Gym.NET.Environments.targets` from the project file: the package brings it, and it registers
the built-in environment ids before `Main` runs.
