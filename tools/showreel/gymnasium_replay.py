"""Replays a Gym.NET showreel episode in Gymnasium and records what Gymnasium renders and returns.

The showreel's parity segment shows Gymnasium and Gym.NET side by side. The C# tool (tools/showreel) records the actions
its policy took; this script replays them in Gymnasium on the parity suite's reference stack (Python 3.12, NumPy 2.4.2,
pygame-ce 2.5.8, Box2D 2.3.10, with refs/Gymnasium on PYTHONPATH), and the tool then compares every frame, observation,
reward and flag with Gym.NET's, failing on any difference.

Output: a little-endian binary file: b"GYMR", int32 version (1), int32 records, int32 height, int32 width, int32
observation bytes; then one record for the reset and one per step: the RGB frame (height * width * 3 bytes), the
observation's bytes in C order, the reward as float64, terminated and truncated as one byte each.
"""
import argparse
import json
import struct
import sys

import numpy as np
import gymnasium as gym


def main() -> int:
    """Reads the actions, replays them in Gymnasium and writes the records.

    Returns:
        The process exit code, 0 on success.

    Raises:
        OSError: The actions file cannot be read or the output cannot be written.
        gymnasium.error.Error: The env id is not registered, or Gymnasium rejects an action.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--actions", required=True, help="the JSON file the showreel tool wrote")
    parser.add_argument("--out", required=True, help="the binary file to write")
    args = parser.parse_args()
    with open(args.actions, encoding="utf-8") as f:
        spec = json.load(f)
    if spec["kind"] == "discrete":
        actions = [int(a) for a in spec["actions"]]
    else:
        # Box actions travel as float32 bit patterns, so no decimal round trip can change a bit.
        actions = list(np.array(spec["bits"], dtype=np.uint32).view(np.float32))
    env = gym.make(spec["env"], render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=spec["seed"])
        frame = env.render()
        height, width, _ = frame.shape
        obs_bytes = np.ascontiguousarray(obs).tobytes()
        with open(args.out, "wb") as out:
            out.write(b"GYMR")
            out.write(struct.pack("<iiiii", 1, len(actions) + 1, height, width, len(obs_bytes)))
            write_record(out, frame, obs, 0.0, False, False)
            for action in actions:
                obs, reward, terminated, truncated, _ = env.step(action)
                write_record(out, env.render(), obs, reward, terminated, truncated)
    finally:
        env.close()
    return 0


def write_record(out, frame, obs, reward, terminated, truncated):
    """Appends one record: frame bytes, observation bytes, the float64 reward, the two flags.

    Args:
        out: The binary file being written.
        frame: The RGB frame render() returned, uint8 of shape (height, width, 3).
        obs: The observation; its bytes are written in C order and its dtype is the env's own.
        reward: The step's reward (0.0 for the reset record), written as float64.
        terminated: The step's terminated flag (False for the reset record).
        truncated: The step's truncated flag (False for the reset record).

    Raises:
        OSError: The file cannot be written.
    """
    out.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    out.write(np.ascontiguousarray(obs).tobytes())
    out.write(struct.pack("<d??", float(reward), bool(terminated), bool(truncated)))


if __name__ == "__main__":
    sys.exit(main())
