# tests/Golden: recorded Gymnasium trajectories

Every file here is a recording of a live parity cell that passed: Gymnasium (`refs/Gymnasium`, embedded through
pythonnet) and Gym.NET ran the same case in lockstep, every call and frame matched byte for byte, and only then was
Gymnasium's side written (decision DR-04, completion plan step P0r.3). The replay cells check Gym.NET against these
files **without Python**, so a machine without the reference stack (the Linux CI job, a laptop) still verifies whole
trajectories and frames.

| What | Where |
|---|---|
| The cases | `tests/Gym.Tests.Parity/Golden/GoldenCatalog.cs` (one `GoldenCase` per golden) |
| Live + record | `GoldenLiveTests` (needs the reference stack of `tools/parity/manifest.json`) |
| Replay | `GoldenReplayTests`, category `Golden` (needs nothing but .NET) |
| How it works | `tests/Gym.Tests.Parity/CLAUDE.md`, "Recording and replay" and "A golden case" |

## Commands

```bash
# Replay every golden (no Python needed)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=Golden"

# Record or re-record from the live cells (Windows reference stack); an unchanged golden is left untouched
GYMNET_PARITY_RECORD=1 dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj -f net10.0 --filter "FullyQualifiedName~GoldenLiveTests"
```

Re-record after changing a case's parameters and after every Gymnasium re-pin: the replay refuses a golden whose
header records other parameters or another Gymnasium commit than the manifest pins, and prints this command.

## Layout

```
tests/Golden/<suite>/<case>.jsonl.gz      the trajectory: gzip-compressed JSON lines
tests/Golden/<suite>/<case>/frame-NNN.png the frames, in trajectory order (rgb_array pixels, lossless)
```

`<suite>` is Gymnasium's env package (`classic_control`), `<case>` the case's name (`CartPole-v1.policy.seed0`).

## Format (`gym.net-golden/1`)

UTF-8 JSON lines, `\n` endings, no indentation, properties in a fixed order, so recording the same trajectory twice
gives the same text. Line 1 is the header; every further line is a call or a frame, in the order they happened.

**Header**: `format`, `suite`, `case`, `env_id`, `kwargs` (`[name, value]` pairs), `seed`, `actions` (the source's
description), `replay_samples` (whether replay re-samples the action space), `max_steps`, `frame_every`,
`state_attribute`, `via_make` (present and `true` only for a case recorded through `make`'s wrapper stack rather than
the unwrapped env, such as CartPole-v0's 200-step limit; absent otherwise), `gymnasium_commit`, `platform` (the runtime
it was recorded on), `stack` (the versions of the manifest's keys at recording time: the per-file manifest of decision
D15).

**Call**: `call` (`reset(seed=0)`, `step #3: step(1)`), `step`, `action` (`null` for the reset), then either
`result` (`observation`, for a step also `reward`, `terminated`, `truncated`, then `info`) or `raised` (`type` as
Python names it, `message`), then `state` and `rng`: the state attribute and the generator state after the call, when
the live cell compared them (`null` otherwise).

**Frame**: `frame` (label), `step`, `png` (the PNG's path relative to the suite directory).

**Values** are the parity suite's side-neutral descriptions (`ValueFacts`): `{"kind", "type", ...}` with `kind` one of
`none`, `bool` (`value`), `int` (`dtype`, `value` as decimal text, since Python ints are unbounded), `float` (`dtype`,
`hex`: the raw little-endian bytes, so NaN payloads and signed zeros survive), `array` (`dtype`, `shape`, `hex`: C-order
bytes), `str` (`value`), `dict` (`entries`: `[key, value]` pairs, in iteration order), `seq` (`items`), `other`
(`repr`). `type` is the recording side's type name, kept for reports and never compared.

## Platform

A case that rests on the platform's libm (CartPole's `cos`/`sin`) is strict on the manifest's reference platform
(`win-x64`). Elsewhere its replay reports a divergence as Inconclusive, with the full report, instead of failing (port
plan §14.9).
