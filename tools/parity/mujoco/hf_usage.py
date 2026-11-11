"""Which MuJoCo envs are used most: the number of models on the Hugging Face Hub whose id names each env in Gym's id form
(`Hopper-v4`), a public proxy for usage (docs/plans/MUJOCO_ASSESSMENT.md §9). Looser forms ("humanoid", "ant") also catch
robotics and unrelated models, so they are not counted.

Usage (needs network access to huggingface.co):
    python tools/parity/mujoco/hf_usage.py
"""
import json
import re
import sys
import urllib.request

ENVS = ["HalfCheetah", "Hopper", "Walker2d", "Ant", "Humanoid", "HumanoidStandup", "Swimmer", "InvertedPendulum",
        "InvertedDoublePendulum", "Reacher", "Pusher"]


def search(term, cap=20000):
    """The ids of the Hub's models matching a search term, following its pagination.

    Args:
        term: The search text.
        cap: The most ids to fetch.

    Returns:
        list: The model ids.

    Raises:
        urllib.error.URLError: The Hub cannot be reached.
    """
    url = f"https://huggingface.co/api/models?search={term}&limit=1000"
    ids = []
    while url and len(ids) < cap:
        with urllib.request.urlopen(url, timeout=60) as response:
            ids += [m["id"] for m in json.load(response)]
            link = response.headers.get("Link") or ""
        following = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = following.group(1) if following else None
    return ids


def main():
    """Prints each env's model count, most used first.

    Returns:
        int: 0.

    Raises:
        urllib.error.URLError: The Hub cannot be reached.
    """
    rows = []
    for env in ENVS:
        pattern = re.compile(rf"(?i)(?<![a-z]){env}-v\d")
        named = [i for i in search(env) if pattern.search(i.split("/")[-1])]
        # "Humanoid-v" and "InvertedPendulum-v" do not match their longer siblings' ids, which carry "Standup"/"Double".
        rows.append((len(named), env))
    for count, env in sorted(rows, reverse=True):
        print(f"{count:6}  {env}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
