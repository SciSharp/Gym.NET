"""Imports the expert agents Gym.NET plays and tests with into tests/Gym.Tests/Agents (design round 34, DR-109).

Each agent is a published Stable-Baselines3 model for one of Gym.NET's ids: Farama's own experts for the eleven MuJoCo
v5 envs (the agents Farama trained to record the Minari datasets) and RL Zoo's TQC agent for BipedalWalkerHardcore-v3.
The script downloads each model's ``policy.pth`` at a pinned Hugging Face revision, keeps only the deterministic actor
(the network that maps an observation to an action; the critics and optimizers are training state), and writes it as
``<id>.json`` in the format ``tests/Gym.Tests/Agents/ExpertPolicy.cs`` reads:

* ``layers``: the actor's dense layers in order, each ``{"inputs", "outputs", "activation", "weight", "bias"}``, the
  weight row-major ``outputs x inputs`` as PyTorch's ``nn.Linear`` stores it, both base64 of little-endian float32;
* ``output``: how the last layer's values become the action: ``tanh-rescale`` (SAC and TQC: ``low + (tanh(y) + 1) / 2
  * (high - low)``) or ``clip`` (PPO: ``y`` clipped to the bounds), as Stable-Baselines3's deterministic ``predict``
  does;
* ``make_kwargs``: the constructor arguments the model was trained with that change what it observes (HumanoidStandup's
  expert sees only positions and velocities);
* ``checks``: actions PyTorch computes for four fixed observations, so the C# reader proves it computes the same;
* ``source``: the repository, the pinned revision, the algorithm, the reported mean return, and the licence situation.

Run with a Python that has PyTorch (any recent CPU build): ``python tools/agents/import_agents.py``. Writes the files
again from the pinned revisions; change a revision only on purpose, then rerun the agent tests, which measure each
agent in Gym.NET.
"""
import argparse
import base64
import io
import json
import os
import struct
import sys
import urllib.request

import numpy as np
import torch

#: Where the agents land, relative to the repository root.
DEFAULT_OUT = os.path.join("tests", "Gym.Tests", "Agents")

#: Farama's training code for the MuJoCo experts (MIT): ``scripts/mujoco/train.py`` and ``make_env.py``.
FARAMA_CODE = "https://github.com/Farama-Foundation/minari-dataset-generation-scripts (MIT)"

#: RL Baselines3 Zoo, which trained and published the ``sb3`` models (MIT).
ZOO_CODE = "https://github.com/DLR-RM/rl-baselines3-zoo (MIT)"

#: The observation flags Farama trained HumanoidStandup's expert with (``make_env.py``): positions and velocities only.
STANDUP_KWARGS = {
    "include_cinert_in_observation": False,
    "include_cvel_in_observation": False,
    "include_qfrc_actuator_in_observation": False,
    "include_cfrc_ext_in_observation": False,
}

#: One entry per agent: (id, Hugging Face repository, pinned revision, model folder in it, algorithm, make kwargs,
#: trainer, training code, the model card's reported mean return).
AGENTS = [
    ("Ant-v5", "farama-minari/Ant-v5-SAC-expert", None, "ant-v5-sac-expert", "SAC", {}),
    ("HalfCheetah-v5", "farama-minari/HalfCheetah-v5-TQC-expert", None, "halfcheetah-v5-TQC-expert", "TQC", {}),
    ("Hopper-v5", "farama-minari/Hopper-v5-SAC-expert", None, "hopper-v5-SAC-expert", "SAC", {}),
    ("Humanoid-v5", "farama-minari/Humanoid-v5-TQC-expert", None, "humanoid-v5-TQC-expert", "TQC", {}),
    ("HumanoidStandup-v5", "farama-minari/HumanoidStandup-v5-SAC-expert", None, "humanoidstandup-v5-SAC-expert", "SAC", STANDUP_KWARGS),
    ("InvertedDoublePendulum-v5", "farama-minari/InvertedDoublePendulum-v5-SAC-expert", None, "inverteddoublependulum-v5-sac-expert", "SAC", {}),
    ("InvertedPendulum-v5", "farama-minari/InvertedPendulum-v5-SAC-expert", None, "invertedpendulum-v5-sac-expert", "SAC", {}),
    ("Pusher-v5", "farama-minari/Pusher-v5-SAC-expert", None, "pusher-v5-SAC-expert", "SAC", {}),
    ("Reacher-v5", "farama-minari/Reacher-v5-SAC-expert", None, "reacher-v5-SAC-expert", "SAC", {}),
    ("Swimmer-v5", "farama-minari/Swimmer-v5-PPO-expert", None, "swimmer-v5-PPO-expert", "PPO", {}),
    ("Walker2d-v5", "farama-minari/Walker2d-v5-SAC-expert", None, "walker2d-v5-SAC-expert", "SAC", {}),
    ("BipedalWalkerHardcore-v3", "sb3/tqc-BipedalWalkerHardcore-v3", None, "tqc-BipedalWalkerHardcore-v3", "TQC", {}),
]

#: The pinned revisions (commit hashes of the Hugging Face repositories), printed by ``--pin`` on 2026-11-22 and kept
#: here so an import is reproducible; an entry's own revision (third field above) overrides its pin.
PINS = {
    "farama-minari/Ant-v5-SAC-expert": "e2eb33e2b09c51a82716e1f106d1ff5ea19a88b5",
    "farama-minari/HalfCheetah-v5-TQC-expert": "995505ac89e1a96bec541dfe9ec8bd86b7e46fda",
    "farama-minari/Hopper-v5-SAC-expert": "bff2655df9e6bfcafa9edb1aa88cf1541f9ee3af",
    "farama-minari/Humanoid-v5-TQC-expert": "e5a86ffdb70e6f4750f39c0464ac026a8437001a",
    "farama-minari/HumanoidStandup-v5-SAC-expert": "a8a03edfc28ef8713f73b3d25445eedd4bbb776d",
    "farama-minari/InvertedDoublePendulum-v5-SAC-expert": "8e4db31851f6572d16c79f835687b666fdccb9c8",
    "farama-minari/InvertedPendulum-v5-SAC-expert": "d67981f53073e6d22908e8a3c793931533ca0bcd",
    "farama-minari/Pusher-v5-SAC-expert": "3e103602f3fba3500ee6f21d762c0ff3e94892ff",
    "farama-minari/Reacher-v5-SAC-expert": "ad8fadfbe3ab40ce9c1bae5b33641041056b3090",
    "farama-minari/Swimmer-v5-PPO-expert": "11ae7c53a3e3156e5399eea8855c64329a107d56",
    "farama-minari/Walker2d-v5-SAC-expert": "2b0ab400beed313a86d84bd82a6e120aa89a61e3",
    "sb3/tqc-BipedalWalkerHardcore-v3": "4eb282af916aa82a00fbdfb33959c669a3cd8b5a",
}


def fetch(url):
    """Downloads a URL's bytes.

    Args:
        url: The address.

    Returns:
        The response body.

    Raises:
        urllib.error.URLError: The download failed (no network, a missing file).
    """
    request = urllib.request.Request(url, headers={"User-Agent": "gym.net-agent-import"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def latest_revision(repo):
    """The repository's current commit, for pinning.

    Args:
        repo: The Hugging Face repository, ``owner/name``.

    Returns:
        The commit hash.
    """
    return json.loads(fetch(f"https://huggingface.co/api/models/{repo}"))["sha"]


def reported_return(repo, revision):
    """The mean return the model card reports, as its text (``"10370.61 +/- 1542.02"``), or None.

    Args:
        repo: The Hugging Face repository.
        revision: The pinned commit.

    Returns:
        The reported value, or None when the card states none.
    """
    try:
        card = fetch(f"https://huggingface.co/{repo}/raw/{revision}/README.md").decode("utf-8")
    except Exception:  # noqa: BLE001 - a card is optional; its absence only leaves the field empty
        return None
    for line in card.splitlines():
        line = line.strip()
        if line.startswith("value:"):
            return line.split(":", 1)[1].strip()
    return None


def bounds(value):
    """An action bound as Stable-Baselines3's saved ``data`` holds it: numpy's printed array (``"[-0.4 -0.4 ...]"``,
    wrapped over lines when long), or already a list.

    Unpickling the space's ``:serialized:`` form instead would need gymnasium installed; the printed form is exact for
    these bounds (short decimals such as -1, -0.4, -2, which float32 prints in full).

    Args:
        value: The saved ``low`` or ``high``.

    Returns:
        The bounds as floats.

    Raises:
        ValueError: A value is not a number.
    """
    if isinstance(value, str):
        return [float(v) for v in value.replace("[", " ").replace("]", " ").split()]
    return [float(v) for v in value]


def b64(array):
    """Base64 of a float32 array's little-endian bytes, C order.

    Args:
        array: The array (any float dtype; converted).

    Returns:
        The text.
    """
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def actor_layers(state, algorithm):
    """The deterministic actor's dense layers from a Stable-Baselines3 ``policy.pth`` state dict.

    SAC and TQC (sb3-contrib) share the actor ``latent_pi`` (Linear, ReLU, ...) and the mean head ``mu``; PPO's
    ``ActorCriticPolicy`` has ``mlp_extractor.policy_net`` (Linear, Tanh, ...) and the head ``action_net``. The
    activations are the libraries' defaults, which the models did not override (their ``policy_kwargs`` set none).

    Args:
        state: The state dict.
        algorithm: ``SAC``, ``TQC`` or ``PPO``.

    Returns:
        A list of (weight, bias, activation) with numpy arrays, input layer first.

    Raises:
        ValueError: The state dict does not have the expected keys.
    """
    if algorithm in ("SAC", "TQC"):
        prefix, head, activation = "actor.latent_pi.", "actor.mu.", "relu"
    elif algorithm == "PPO":
        prefix, head, activation = "mlp_extractor.policy_net.", "action_net.", "tanh"
    else:
        raise ValueError(f"unknown algorithm {algorithm}")
    hidden = sorted({int(k[len(prefix):].split(".")[0]) for k in state if k.startswith(prefix) and k.endswith(".weight")})
    if not hidden or head + "weight" not in state:
        raise ValueError(f"no {prefix}* or {head}weight in the state dict: {sorted(state)[:8]}...")
    layers = [(state[f"{prefix}{i}.weight"].numpy(), state[f"{prefix}{i}.bias"].numpy(), activation) for i in hidden]
    layers.append((state[head + "weight"].numpy(), state[head + "bias"].numpy(), "none"))
    return layers


def forward(layers, observation, output, low, high):
    """The deterministic action in PyTorch, as Stable-Baselines3's ``predict(deterministic=True)`` computes it.

    Args:
        layers: The actor's layers from :func:`actor_layers`.
        observation: One observation (float64 or float32).
        output: ``tanh-rescale`` or ``clip``.
        low: The action space's lower bounds.
        high: The action space's upper bounds.

    Returns:
        The action as float32.
    """
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(observation, dtype=np.float32))
        for weight, bias, activation in layers:
            x = torch.nn.functional.linear(x, torch.as_tensor(weight), torch.as_tensor(bias))
            if activation == "relu":
                x = torch.relu(x)
            elif activation == "tanh":
                x = torch.tanh(x)
        y = x.numpy().astype(np.float32)
    low32, high32 = np.asarray(low, np.float32), np.asarray(high, np.float32)
    if output == "tanh-rescale":
        return (low32 + np.float32(0.5) * (np.tanh(y) + np.float32(1.0)) * (high32 - low32)).astype(np.float32)
    return np.clip(y, low32, high32).astype(np.float32)


def check_observations(size):
    """Four fixed observations for the reader's check: zeros, two seeded normals (wide and narrow), a seeded uniform.

    Args:
        size: The observation length.

    Returns:
        The four observations as float32 lists.
    """
    rng = np.random.default_rng(20261122)
    return [np.zeros(size, np.float32), rng.normal(0, 1, size).astype(np.float32), rng.normal(0, 0.3, size).astype(np.float32),
            rng.uniform(-1, 1, size).astype(np.float32)]


def import_agent(entry, out_folder, pins):
    """Downloads one agent at its pinned revision and writes ``<id>.json``.

    Args:
        entry: One :data:`AGENTS` row.
        out_folder: The destination folder.
        pins: Revision per repository (``--pin`` fills it with the current heads).

    Returns:
        A one-line summary.

    Raises:
        ValueError: The model's layout or spaces are not what the reader supports.
    """
    env_id, repo, revision, folder, algorithm, make_kwargs = entry
    revision = revision or pins[repo]
    base = f"https://huggingface.co/{repo}/resolve/{revision}/{folder}"
    state = torch.load(io.BytesIO(fetch(base + "/policy.pth")), map_location="cpu", weights_only=True)
    data = json.loads(fetch(base + "/data"))
    action_space = data["action_space"]
    low = bounds(action_space["low"])
    high = bounds(action_space["high"])
    observation_size = int(data["observation_space"]["_shape"][0])
    layers = actor_layers(state, algorithm)
    if layers[0][0].shape[1] != observation_size or layers[-1][0].shape[0] != len(low):
        raise ValueError(f"{env_id}: layers {layers[0][0].shape} ... {layers[-1][0].shape} do not match "
                         f"{observation_size} observations and {len(low)} actions")
    output = "clip" if algorithm == "PPO" else "tanh-rescale"
    checks = [{"observation": [float(v) for v in o], "action": [float(v) for v in forward(layers, o, output, low, high)]}
              for o in check_observations(observation_size)]
    farama = repo.startswith("farama-minari/")
    document = {
        "format": "gym.net-agent/1",
        "id": env_id,
        "make_kwargs": make_kwargs,
        "observation_size": observation_size,
        "action_size": len(low),
        "action_low": low,
        "action_high": high,
        "output": output,
        "layers": [{"inputs": int(w.shape[1]), "outputs": int(w.shape[0]), "activation": a, "weight": b64(w), "bias": b64(b)}
                   for w, b, a in layers],
        "source": {
            "repository": f"https://huggingface.co/{repo}",
            "revision": revision,
            "file": f"{folder}/policy.pth",
            "algorithm": algorithm,
            "library": "stable-baselines3" + (" + sb3-contrib" if algorithm == "TQC" else ""),
            "trained_by": "the Farama Foundation, to record the Minari datasets" if farama else "RL Baselines3 Zoo",
            "training_code": FARAMA_CODE if farama else ZOO_CODE,
            "reported_mean_return": reported_return(repo, revision),
            "licence": "No licence is stated on the model repository; its training code is MIT. Committed with "
                       "attribution on the maintainer's decision (DR-109).",
        },
        "checks": checks,
    }
    path = os.path.join(out_folder, f"{env_id}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(document, f, indent=1)
        f.write("\n")
    params = sum(w.size + b.size for w, b, _ in layers)
    return f"{env_id}: {algorithm} {[l[0].shape[0] for l in layers]}, {params} parameters, {os.path.getsize(path)} bytes"


def main():
    """Imports every agent, or pins the current revisions with ``--pin``.

    Returns:
        The exit code: 0 on success, 1 when a repository has no pinned revision.

    Raises:
        urllib.error.URLError: A download failed.
        ValueError: A model's layout does not match its spaces (see :func:`import_agent`).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="destination folder (default %(default)s)")
    parser.add_argument("--pin", action="store_true", help="print the repositories' current revisions for PINS and exit")
    args = parser.parse_args()
    if args.pin:
        for _, repo, *_ in AGENTS:
            print(f'    "{repo}": "{latest_revision(repo)}",')
        return 0
    missing = [repo for _, repo, revision, *_ in AGENTS if revision is None and repo not in PINS]
    if missing:
        print("unpinned repositories (run --pin and fill PINS): " + ", ".join(missing), file=sys.stderr)
        return 1
    os.makedirs(args.out, exist_ok=True)
    for entry in AGENTS:
        print(import_agent(entry, args.out, PINS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
