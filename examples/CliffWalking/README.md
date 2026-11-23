# CliffWalking: value iteration

A walker crosses a grid from the bottom-left corner to the bottom-right one, along a cliff. Every step costs 1, and a
step off the cliff costs 100 and sends the walker back to the start.

```bash
dotnet run --project examples/CliffWalking -c Release                   # one episode, headless
dotnet run --project examples/CliffWalking -c Release -- --render       # the same episode in a window
dotnet run --project examples/CliffWalking -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/CliffWalking -c Release -- --episodes 5   # seeds 0-4, and their mean
dotnet run --project examples/CliffWalking -c Release -- --record docs/environments/toy_text/media   # the docs page's media
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

## Recording the docs page's media

`--record <folder>` plays the seed-0 episode once more (whatever `--seed` says) and writes the media of Cliff Walking's
page on the docs site (DR-112) into the folder, `docs/environments/toy_text/media` for the site:

| File | What it is |
|---|---|
| `CliffWalking-v1.episode.mp4` | The whole episode, one frame per step at the env's 4 frames a second (14 frames: the reset and 13 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `CliffWalking-v1.episode.json` | Every step: the cell the agent acted on (its row, column and state number), its move and the reward (-100 for a fall); the cell after the last step as `final` |
| `CliffWalking-v1.board.png` | The grid as the reset shows it, which the page's anatomy draws the path over (DR-131) |
| `CliffWalking-v1.figure.png`, `.figure.json` | The frame after 6 steps, the walker halfway along the cliff's edge on cell 29, which the page annotates, and what its labels point at: the grid, the walker's cell, each move's aimed cell, landing cell and reward, the start, the goal and the cliff, all read from the env's transition table |

```
CliffWalking-v1, seed 0: 13 steps, return -13.0, recorded into docs/environments/toy_text/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

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
