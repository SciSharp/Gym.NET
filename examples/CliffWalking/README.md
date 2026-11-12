# CliffWalking: value iteration

A walker crosses a grid from the bottom-left corner to the bottom-right one, along a cliff. Every step costs 1, and a
step off the cliff costs 100 and sends the walker back to the start.

```bash
dotnet run --project examples/CliffWalking -c Release                   # one episode, headless
dotnet run --project examples/CliffWalking -c Release -- --render       # the same episode in a window
dotnet run --project examples/CliffWalking -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/CliffWalking -c Release -- --episodes 5   # seeds 0-4, and their mean
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
CliffWalking-v1, seed 0: 13 steps, return -13.0, reached the goal
```

## The agent

Value iteration over the env's own transition table, `P`: for every state and action it lists the possible next states
with their probabilities and rewards. The agent sweeps the states, updating each one's value to the best expected reward
plus 0.99 times the next state's value, until nothing moves by more than 1e-10, then always takes the action with the
best value. Here every move is certain, and the best path is the shortest one that stays on the grid: up one row, along
the cliff's edge, and down to the goal, 13 steps for a return of -13.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Arrow keys | Move up, right, down or left |

## The API it shows

- `((CliffWalkingEnv)env.Unwrapped).P`: the transition table, Gymnasium's `P[state][action]` list of `(probability,
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
