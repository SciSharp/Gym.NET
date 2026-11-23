# Blackjack: basic strategy

Blackjack against Gymnasium's dealer, who draws until 17. The observation is the player's sum, the dealer's showing card
and whether the player holds a usable ace.

```bash
dotnet run --project examples/Blackjack -c Release                   # one episode, headless
dotnet run --project examples/Blackjack -c Release -- --render       # the same episode in a window
dotnet run --project examples/Blackjack -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/Blackjack -c Release -- --episodes 5   # seeds 40-44, and their mean
dotnet run --project examples/Blackjack -c Release -- --record docs/environments/toy_text/media   # the docs page's media
```

Output, the same on every run since the reset is seeded and the agent has no randomness of its own (measured on an x64
CPU with FMA: [the numbers](../README.md#the-numbers) says why that matters):

```
Blackjack-v1, seed 40: 1 step, return 1.0, won
```

## The agent

Basic strategy for these rules: hit a soft total (one with an ace counted as 11) below 18, and a soft 18 against a 9, a
10 or an ace; hit a hard total below 12; hit 12 to 16 against a 7 or better or an ace; hit 12 against a 2 or a 3;
otherwise stick. One episode is one hand, so `--episodes 1000` plays a thousand from consecutive seeds and prints the
mean return: -0.019 per hand from seed 40, and -0.048 over 10,000 hands. Basic strategy still loses slightly, as it does
in a casino.

## Watching and playing

`--render` opens the env's window and plays the episode in it, at the env's frame rate.

`--play` hands the env to you through Gymnasium's `play` utility (`Play.Run`). It resets with the seed whenever an
episode ends; Escape or the window's close button quits.

| Key | Does |
|---|---|
| H | Hit |
| S | Stick |

## Recording the docs page's media

`--record <folder>` plays the hand Blackjack's page shows (DR-128), seed 1,574,758 whatever `--seed` says, and writes
the page's media on the docs site (DR-112) into the folder, `docs/environments/toy_text/media` for the site. The
showreel's seed 40 deals a one-step win, a single frame; seed 1,574,758 is the longest hand this strategy wins on
exactly 21 among seeds 0 to 1,999,999: two 2s against the dealer's ace, six hits through a soft 15 and a soft 17 that
turns hard, and a stick on 21 against the dealer's soft 17.

| File | What it is |
|---|---|
| `Blackjack-v1.episode.mp4` | The whole hand, one frame per step at the env's 4 frames a second (8 frames: the deal and 7 steps), H.264 through Gymnasium's `RecordVideo` and `FfmpegVideoWriter` |
| `Blackjack-v1.episode.json` | Every step: the observation acted on (the player's sum, the dealer's face-up card, the usable ace), the action and the reward; the observation after the stick as `final` |
| `Blackjack-v1.figure.png`, `.figure.json` | The frame after 2 steps, the player's soft 17 against the dealer's ace, which the page annotates, and what its labels point at: the observation, the hands behind it (which the table never draws), the face-up card's name and Gymnasium's table layout |

```
Blackjack-v1, seed 1574758: 7 steps, return 1.0, recorded into docs/environments/toy_text/media
```

ffmpeg must be on PATH to record; every other mode runs without it.

## The API it shows

- `Gymnasium.Make<object?[], NDArray>("Blackjack-v1", kwargs)`: Blackjack's observation is Gym.NET's tuple of three 0-d
  arrays (DR-13); `BlackjackObservation` reads it by name
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
