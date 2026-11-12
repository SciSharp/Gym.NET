# Blackjack: basic strategy

Blackjack against Gymnasium's dealer, who draws until 17. The observation is the player's sum, the dealer's showing card
and whether the player holds a usable ace.

```bash
dotnet run --project examples/Blackjack -c Release                   # one episode, headless
dotnet run --project examples/Blackjack -c Release -- --render       # the same episode in a window
dotnet run --project examples/Blackjack -c Release -- --play         # you play it, from the keyboard
dotnet run --project examples/Blackjack -c Release -- --episodes 5   # seeds 40-44, and their mean
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
