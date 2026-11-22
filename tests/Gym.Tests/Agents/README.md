# The expert agents

Published agents for Gym.NET's envs (DR-103, 2026-11-22): the eleven MuJoCo v5 envs played by the experts the Farama
Foundation, Gymnasium's maintainers, trained on them to record the Minari datasets, and BipedalWalkerHardcore-v3 by
RL Baselines3 Zoo's TQC agent. The maintainer: "All of our envs must be good or state of the art", and "i want all sota
weights to be used and served in the tests project or wherever fit".

This folder is their home. `ExpertAgentTests` measures every one in Gym.NET, the showreel plays them (it links this
folder's files and `ExpertPolicy.cs`), and the examples carry copies (`examples/<Name>/policy.json`, HalfCheetah's
`expert.json`).

| Id | Agent | Source (Hugging Face, pinned revision) | Network | Model card | Gym.NET, seeds 0-9 |
|---|---|---|---|---|---|
| Ant-v5 | SAC | [farama-minari/Ant-v5-SAC-expert](https://huggingface.co/farama-minari/Ant-v5-SAC-expert) `e2eb33e` | 256, 256 | 5,846.28 ± 138.52 | 5,661.2 |
| HalfCheetah-v5 | TQC | [farama-minari/HalfCheetah-v5-TQC-expert](https://huggingface.co/farama-minari/HalfCheetah-v5-TQC-expert) `995505a` | 256, 256 | 17,641.80 ± 61.88 | 17,629.0 |
| Hopper-v5 | SAC | [farama-minari/Hopper-v5-SAC-expert](https://huggingface.co/farama-minari/Hopper-v5-SAC-expert) `bff2655` | 256, 256 | 4,098.17 ± 247.70 | 4,143.5 |
| Humanoid-v5 | TQC | [farama-minari/Humanoid-v5-TQC-expert](https://huggingface.co/farama-minari/Humanoid-v5-TQC-expert) `e5a86ff` | 256, 256 | 10,370.61 ± 1,542.02 | 10,650.7 |
| HumanoidStandup-v5 | SAC | [farama-minari/HumanoidStandup-v5-SAC-expert](https://huggingface.co/farama-minari/HumanoidStandup-v5-SAC-expert) `a8a03ed` | 256, 256 | 395,485.91 ± 33,309.92 | 400,097.0 |
| InvertedDoublePendulum-v5 | SAC | [farama-minari/InvertedDoublePendulum-v5-SAC-expert](https://huggingface.co/farama-minari/InvertedDoublePendulum-v5-SAC-expert) `8e4db31` | 256, 256 | 9,356.02 ± 0.06 | 9,356.0 |
| InvertedPendulum-v5 | SAC | [farama-minari/InvertedPendulum-v5-SAC-expert](https://huggingface.co/farama-minari/InvertedPendulum-v5-SAC-expert) `d67981f` | 256, 256 | 1,000.00 ± 0.00 | 1,000.0 |
| Pusher-v5 | SAC | [farama-minari/Pusher-v5-SAC-expert](https://huggingface.co/farama-minari/Pusher-v5-SAC-expert) `3e10360` | 256, 256 | -22.05 ± 2.69 | -28.0 |
| Reacher-v5 | SAC | [farama-minari/Reacher-v5-SAC-expert](https://huggingface.co/farama-minari/Reacher-v5-SAC-expert) `ad8fadf` | 256, 256 | -3.28 ± 1.38 | -3.4 |
| Swimmer-v5 | PPO | [farama-minari/Swimmer-v5-PPO-expert](https://huggingface.co/farama-minari/Swimmer-v5-PPO-expert) `11ae7c5` | 64, 64 (tanh) | 363.70 ± 1.81 | 364.0 |
| Walker2d-v5 | SAC | [farama-minari/Walker2d-v5-SAC-expert](https://huggingface.co/farama-minari/Walker2d-v5-SAC-expert) `2b0ab40` | 256, 256 | 6,956.61 ± 15.95 | 6,949.0 |
| BipedalWalkerHardcore-v3 | TQC | [sb3/tqc-BipedalWalkerHardcore-v3](https://huggingface.co/sb3/tqc-BipedalWalkerHardcore-v3) `4eb282a` | 400, 300 | 208.05 ± 121.38 | 243.8 (six of ten finish) |

The model cards' numbers are the reported means (the Farama experts were trained 20 to 100 million steps each, by
`scripts/mujoco/train.py` of
[minari-dataset-generation-scripts](https://github.com/Farama-Foundation/minari-dataset-generation-scripts)); the last
column is Gym.NET's own engine, measured on 2026-11-22 (net10.0, the reference i9). Where a repository holds two models,
the one in the table scored better in Gym.NET: the other Walker2d fell on nine seeds of ten. HumanoidStandup's expert was
trained seeing positions and velocities only, so its file's `make_kwargs` turns off the four other observation groups.

## The licence

The model repositories state no licence; the code that trained them is MIT (Farama's generation scripts, RL Baselines3
Zoo). The weights are committed here with this attribution, on the maintainer's decision ("Commit, attributed",
2026-11-22, DR-103), and every file's `source` names its repository, revision and trainer.

## The format

Each `<id>.json` is `gym.net-agent/1`, written by `tools/agents/import_agents.py` from the pinned revision: the actor
network's dense layers (base64 of little-endian float32, each weight row-major `outputs x inputs` as PyTorch's
`nn.Linear` stores it) with the activation after each; `output`, how the last layer becomes the action (`tanh-rescale`
for SAC and TQC, `clip` for PPO, as Stable-Baselines3's deterministic `predict`); the action bounds; `make_kwargs`;
`source`; and `checks`, four observations with the actions PyTorch computes for them. `ExpertPolicy.cs` reads it and
computes in float32 as PyTorch does, each sum taken in float64; `ExpertAgentTests.ReaderMatchesPyTorch` holds it to
those checks (measured: within 6e-7 of each action's range).

```bash
python tools/agents/import_agents.py          # rewrites the files from the pinned revisions (needs PyTorch)
python tools/agents/import_agents.py --pin    # prints the repositories' current revisions, to re-pin on purpose
dotnet test tests/Gym.Tests/Gym.Tests.csproj --filter "FullyQualifiedName~ExpertAgentTests"
```
