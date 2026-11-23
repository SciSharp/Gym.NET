# Taxi: value iteration

A taxi in a 5×5 grid with walls has to pick a passenger up at one of four stands and drop them at another.

```bash
dotnet run --project examples/Taxi -c Release                   # one episode, headless
dotnet run --project examples/Taxi -c Release -- --render       # the same episode in a window
dotnet run --project examples/Taxi -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/Taxi -c Release -- --episodes 5   # seeds 12-16, and their mean
dotnet run --project examples/Taxi -c Release -- --record docs/environments/toy_text/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Taxi-v4, seed 12: 16 steps, return 5.0, delivered the passenger
```

## The agent

Value iteration over the env's own transition table, `P`: for every state and action it lists the possible next states
with their probabilities and rewards. The agent sweeps the states, updating each one's value to the best expected reward
plus 0.99 times the next state's value, until nothing moves by more than 1e-10, then always takes the action with the
best value. With 500 states and every move certain, the plan is the shortest delivery from every state: from seed 12 it
takes 16 steps, for a return of 5 (20 for the delivery, minus 1 per step).

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| Arrow keys | Drive: down is south, up north, right east, left west |
| P | Pick the passenger up |
| D | Drop the passenger off |

## Recording the docs page's media

`--record <folder>` plays the seed-12 episode once more (whatever `--seed` says) and writes the media of Taxi's page on
the docs site (DR-112) into the folder, `docs/environments/toy_text/media` for the site:

| File | What it is |
|---|---|
| `Taxi-v4.episode.mp4` | The whole episode, one frame per step at the env's 4 frames a second (17 frames: the reset and 16 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `Taxi-v4.episode.json` | Every step: the state the agent acted on, decoded (the taxi's row and column, the passenger's place, the destination) and as its number, whether the passenger is aboard, the action and the reward; the state after the last step as `final` |
| `Taxi-v4.board.png` | The town as the reset shows it, the passenger on their stand and the destination framed, which the page's anatomy draws the taxi's path over (DR-131) |
| `Taxi-v4.figure.png`, `.figure.json` | The frame after 6 steps, the taxi on the passenger's stand about to pick them up, which the page annotates, and what its labels point at: the grid's origin and cells, the state and its decoding, the action, the action mask, the stands and the walls, read from the env's map |

```
Taxi-v4, seed 12: 16 steps, return 5.0, recorded into docs/environments/toy_text/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `((TaxiEnv)env.Unwrapped).P`: the transition table, Gymnasium's `P[state][action]` list of `(probability, next_state,
  reward, terminated)`, as `ToyTextTransition` records
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
