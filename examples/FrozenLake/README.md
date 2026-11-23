# FrozenLake: value iteration

An elf crosses a frozen lake from the top-left corner to the goal at the bottom right, between holes. The ice is
slippery: a move goes the intended way only a third of the time.

```bash
dotnet run --project examples/FrozenLake -c Release                   # one episode, headless
dotnet run --project examples/FrozenLake -c Release -- --render       # the same episode in a window
dotnet run --project examples/FrozenLake -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/FrozenLake -c Release -- --episodes 5   # seeds 1-5, and their mean
dotnet run --project examples/FrozenLake -c Release -- --record docs/environments/toy_text/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
FrozenLake-v1, seed 1: 41 steps, return 1.0, reached the goal
```

## The agent

Value iteration over the env's own transition table, `P`: for every state and action it lists the possible next states
with their probabilities and rewards. The agent sweeps the states, updating each one's value to the best expected reward
plus 0.99 times the next state's value, until nothing moves by more than 1e-10, then always takes the action with the
best value. On the slippery lake the best action often looks wrong: it edges along a hole's far side, so that a slip
cannot drop the elf in. From seed 1 it reaches the goal in 41 steps; it does not always succeed, because no policy can
on this map.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Arrow keys | Move left, down, right or up (the ice slips sideways two times in three) |

## Recording the docs page's media

`--record <folder>` plays the seed-1 episode once more (whatever `--seed` says) and writes the media of Frozen Lake's
page on the docs site (DR-112) into the folder, `docs/environments/toy_text/media` for the site:

| File | What it is |
|---|---|
| `FrozenLake-v1.episode.mp4` | The whole episode, one frame per step at the env's 4 frames a second (42 frames: the reset and 41 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `FrozenLake-v1.episode.json` | Every step: the cell the agent acted on (its row, column and state number), its move, whether the ice slipped (the next cell is not the one the move aims at) and the reward; the cell after the last step as `final` |
| `FrozenLake-v1.board.png` | The lake as the reset shows it, which the page's anatomy draws the path over (DR-131) |
| `FrozenLake-v1.figure.png`, `.figure.json` | The frame after 40 steps, the elf on cell 14 aiming down into the edge, which the page annotates, and what its labels point at: the grid, the elf's cell, the cell its move aims at and the two a slip reaches, the holes, the start and the goal |

```
FrozenLake-v1, seed 1: 41 steps, return 1.0, recorded into docs/environments/toy_text/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

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
