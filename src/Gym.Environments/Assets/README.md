# Env assets

The images Gym.NET's envs render with, embedded in `Gym.Environments` as resources (decision DR-19): copied once from
the pinned Gymnasium submodule, never built from it, so the package carries its own copies.

| File | Source | Used by |
|---|---|---|
| `classic_control/clockwise.png` | `refs/Gymnasium/gymnasium/envs/classic_control/assets/clockwise.png` | Pendulum's torque arrow |

Source commit: Gymnasium v1.4.0, `8ba8c3a63bb4e000b7f33923f43d292acac07be1`. The files are byte-identical to it
(SHA-256 `97b50edad865fb564c7634cc26fb5712880a16870f9f8b89b52d89f0374ac9a9` for `clockwise.png`).

License: Gymnasium's MIT License (Copyright (c) 2016 OpenAI, Copyright (c) 2022 Farama Foundation), decision D14. The
license text ships with Gymnasium; keep this table and the hashes in step when a re-pin re-copies the files.

Each file is embedded under the resource name `assets/<path>` (forward slashes on every OS), read through
`EnvAssets.Load` and decoded with `ImageIO.Load(Stream)`.
