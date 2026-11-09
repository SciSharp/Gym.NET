# Development and testing

[Documentation index](../README.md)

How to build Gym.NET, run its two test suites, and add to them. The rules for contributors' code are in the
repository's [CLAUDE.md](../../CLAUDE.md); the parity suite's own guide is
[tests/Gym.Tests.Parity/CLAUDE.md](../../tests/Gym.Tests.Parity/CLAUDE.md).

## Prerequisites

- The .NET 8 and .NET 10 SDKs (the libraries target `net8.0` and `net10.0`; the unit tests and the Windows Forms viewer
  target `net8.0-windows` and `net10.0-windows`).
- The two reference submodules, initialized without `--recursive` (that would also clone NumSharp's own submodules,
  which the build doesn't need):

  ```sh
  git submodule update --init refs/Gymnasium refs/NumSharp
  ```

  `refs/Gymnasium` is Gymnasium v1.4.0 (`8ba8c3a6`), the behavior Gym.NET is held to. `refs/NumSharp` is NumSharp
  v0.80.0's source, used only by the opt-in `-p:NumSharpSource=Submodule` build; the default build uses the NumSharp
  0.80.0 package.
- For the parity suite only: Python 3.12 with NumPy 2.4.2, pygame-ce 2.5.8 and Box2D 2.3.10, the stack pinned in
  `tools/parity/manifest.json`. `pwsh tools/parity/setup_venv.ps1` (or `bash tools/parity/setup_venv.sh`) creates it in
  `tools/parity/.venv`, where the suite finds it.
- Set `AVALONIA_TELEMETRY_OPTOUT=1`: building the Avalonia projects otherwise sends Avalonia's build telemetry.

## Build

```sh
dotnet build Gym.NET.sln -c Debug
dotnet build Gym.NET.sln -c Release
```

The solution holds the five packaged libraries (`src/Gym`, `src/Gym.Rendering`, `src/Gym.Environments`,
`src/Gym.Rendering.Avalonia`, `src/Gym.Rendering.WinForm`), the source generator (`src/Gym.SourceGenerators`), the test
projects and `examples/CartPole`. Building the libraries also packs them (`GeneratePackageOnBuild`).

## The unit suite: tests/Gym.Tests

MSTest, headless, no Python. It holds Gymnasium's own pytest files rewritten in C# under `tests/Gym.Tests/Gymnasium`
(one file per Python file, `test_core.py` → `TestCore.cs`), plus Gym.NET's own tests.

```sh
# The gate: everything except the viewer tests, which open windows, and the known gaps
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName!~AvaloniaEnv&FullyQualifiedName!~WinFormEnv&TestCategory!=OpenBugs"

# One framework, one class, one test
dotnet test tests/Gym.Tests/Gym.Tests.csproj -f net10.0-windows --filter "FullyQualifiedName~ToyTextTests"
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~TestMake.TestMakeWithEnvSpecLevels"
```

`-f net10.0` (without `-windows`) matches no target framework of the unit project and silently runs nothing.

## The parity suite: tests/Gym.Tests.Parity

The suite embeds CPython through pythonnet, imports Gymnasium from `refs/Gymnasium`, and runs both libraries side by
side with the same seeds and actions. Each cell compares dtypes, shapes and bytes of arrays, IEEE-754 bits of floats,
info keys in order, exception types and messages, the PCG64 generator state, and `rgb_array` frames pixel for pixel.
There is no tolerance.

```sh
# The gate
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory!=OpenBugs"

# The known gaps, each printing its first divergence from Gymnasium
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=OpenBugs" --logger "console;verbosity=detailed"

# The recorded trajectories, replayed without Python (what the Linux CI job runs)
dotnet test tests/Gym.Tests.Parity/Gym.Tests.Parity.csproj --filter "TestCategory=Golden"
```

- **Goldens** (`tests/Golden`): trajectories and frames the live cells verified, recorded with
  `GYMNET_PARITY_RECORD=1` and replayed without Python, so machines without the stack still check the envs.
- **The reverse bridge** (`tests/Gym.Tests.Parity/Bridge`): Gym.NET's envs exposed to Python as Gymnasium envs, so
  Gymnasium's own `check_env` and `check_environments_match` run on them.
- **Failed frames** write `python.png`, `dotnet.png` and `diff.png` to the test results (or `GYMNET_PARITY_ARTIFACTS`).

## Known gaps: [OpenBugs]

A test Gym.NET does not pass yet carries `[OpenBugs("id")]`, NumSharp's convention: the gated runs exclude it, and a
plain `dotnet test` runs it and fails. The id names the reason: `N-1`…`N-3` are NumSharp gaps
(docs/plans/COMPLETION_PLAN.md §7), `P6`/`P7`/`P8` are phases still to come, `B-xx` the defect register of the port plan.
The expectation is always Gymnasium's answer, never today's output: when the gap closes, the test passes, and the
attribute is removed so it becomes part of the gate. A deliberate, documented deviation is not an open bug; it is a
gated test that pins the deviation.

## The name inventory

`tests/Gym.Tests/Gymnasium/inventory.expected-missing.txt` lists every public Gymnasium name Gym.NET has no counterpart
for yet, each with its reason. A test fails when a listed name appears (remove the line) or an unlisted one is missing,
so the list only shrinks.

## Writing code here

- Every member is documented with what it does, what it costs and its footguns: each parameter, return value and
  exception. A member based on Gymnasium's source links it as a pinned permalink in `<remarks>`, for example
  `Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/core.py#L28">gymnasium/core.py</see>.`
  The docs describe Gym.NET's own members in its own voice.
- Exceptions: Python's built-ins come from NumSharp (`ValueError`, `TypeError`, `KeyError`, `AssertionError`),
  Gymnasium's from `Gym.Exceptions`; messages are Gymnasium's, verbatim.
- Rewards are `double` everywhere.
- Arrays are disposed by whoever owns them. NumSharp's analyzer reports an array created and dropped (`NDW012`);
  `Gym.Rendering` and `Gym.Tests.Parity` treat warnings as errors, so there it fails the build.
- A behavior change comes with a live parity cell that compares it with Gymnasium, and a new comparison rule in the
  harness with a self-test that plants a defect and checks it is caught (`Tests/Harness/HarnessSelfTests.cs`).

## CI

`.github/workflows/ci.yml` builds the solution in Debug and Release on Windows, runs the unit gate, sets up the parity
stack and runs the parity gate; a Linux job builds the parity suite and replays the goldens. Until 2.0 it runs on the
working branch and is informational.
