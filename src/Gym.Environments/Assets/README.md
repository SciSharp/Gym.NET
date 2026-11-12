# Env assets

The images Gym.NET's envs render with, embedded in `Gym.Environments` as resources (decision DR-19): copied once from
the pinned Gymnasium submodule, never built from it, so the package carries its own copies. Only the images Gymnasium's
code loads are copied (the unused `elf_*.png` and `stool.png` of toy_text are not).

Source commit: Gymnasium v1.4.0, `8ba8c3a63bb4e000b7f33923f43d292acac07be1`. Every file is byte-identical to its source,
as the SHA-256 column records.

License: Gymnasium's MIT License (Copyright (c) 2016 OpenAI, Copyright (c) 2022 Farama Foundation), decision D14; the
toy_text art is Farama's own, under the same license. The license text ships with Gymnasium; keep these tables and the
hashes in step when a re-pin re-copies the files.

Each file is embedded under the resource name `assets/<path>` (forward slashes on every OS), read through
`EnvAssets.Load` and decoded with `ImageIO.Load(Stream)`.

Two kinds of file are not copies but recordings: `fonts/freesansbold-42.json`, CarRacing's score glyphs, recorded from
pygame-ce (see [fonts](#fonts) below), and `mujoco/*.json` with `mujoco/textures/*.png`, Gymnasium's MuJoCo models as
the pinned MuJoCo compiles them (see [mujoco](#mujoco) below). Their tables record the source each was made from.

## classic_control

| File | Source | SHA-256 | Used by |
|---|---|---|---|
| `classic_control/clockwise.png` | `refs/Gymnasium/gymnasium/envs/classic_control/assets/clockwise.png` | `97b50edad865fb564c7634cc26fb5712880a16870f9f8b89b52d89f0374ac9a9` | Pendulum's torque arrow |

## toy_text

FrozenLake's tiles and overlays (`toy_text/*.png`), CliffWalking's (`toy_text/cliff_walking`), Taxi's
(`toy_text/taxi`) and Blackjack's cards, digits and labels (`toy_text/blackjack`).

| File | Source | SHA-256 |
|---|---|---|
| `toy_text/tile_ice.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/tile_ice.png` | `559be147c8d49e197c6ccda13a11af3e69c1575d2d019e5ddbfc1c49dfb27357` |
| `toy_text/tile_hole.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/tile_hole.png` | `aa2c92406eede3ffddddf9480c14c5483d307043b2fedbda48f80a493f03cdc9` |
| `toy_text/tile_ice_crack_a.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/tile_ice_crack_a.png` | `4c666f529eabf630326b66affbbdd25619b26a7f1c9db73163268c0c2b12976f` |
| `toy_text/tile_ice_crack_b.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/tile_ice_crack_b.png` | `f9b9e9da61be03dbbaee490b71f2cf462a125457a85524c43baa7c501b05078a` |
| `toy_text/agent.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/agent.png` | `1c7603264f0f99e569a75f51f0b8850842422b18524a50beffc5b533c6fa4dcb` |
| `toy_text/goal_flag.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/goal_flag.png` | `15de4a33ae7f61f7f89bf9cffe96412bba45659c4c1e0e3b2f0a791bde26f3ab` |
| `toy_text/letter_s.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/letter_s.png` | `cd22b19e4c1bbe803d9dd80ec97c7a97ff52a98d27f75de540ef7bcc6028f400` |
| `toy_text/letter_g.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/letter_g.png` | `46b9b4899e197f2bd887cfbeee5acda96af65a230c15e58d776531ad4662f6e1` |
| `toy_text/cliff_walking/agent.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/agent.png` | `1c7603264f0f99e569a75f51f0b8850842422b18524a50beffc5b533c6fa4dcb` |
| `toy_text/cliff_walking/goal_flag.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/goal_flag.png` | `15de4a33ae7f61f7f89bf9cffe96412bba45659c4c1e0e3b2f0a791bde26f3ab` |
| `toy_text/cliff_walking/letter_g.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/letter_g.png` | `46b9b4899e197f2bd887cfbeee5acda96af65a230c15e58d776531ad4662f6e1` |
| `toy_text/cliff_walking/letter_s.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/letter_s.png` | `cd22b19e4c1bbe803d9dd80ec97c7a97ff52a98d27f75de540ef7bcc6028f400` |
| `toy_text/cliff_walking/tile_cliff_left.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/tile_cliff_left.png` | `94e26627d572b1bbeea8dfbead4c78b4d0691574a4915fb039c54ab1e54c1f50` |
| `toy_text/cliff_walking/tile_cliff_mid_a.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/tile_cliff_mid_a.png` | `b187b222d6bf37fe0d84d75fbbee49f9f6535bbb1477155bc53f85e2e4b4be57` |
| `toy_text/cliff_walking/tile_cliff_mid_b.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/tile_cliff_mid_b.png` | `61cb9c58066c50c836e1c1d6f9d1a3dd00ce8f0b7e171bc2c5fdfd8d29748564` |
| `toy_text/cliff_walking/tile_cliff_right.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/tile_cliff_right.png` | `978d8b672a0c1b61fd37d80a9bfa3f976b2216570ea85e06cdd124eb4b9e661e` |
| `toy_text/cliff_walking/tile_ground.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/cliff_walking/tile_ground.png` | `8666022d226483ccd6f8795b30034f13c448207e6f431ef11c30521cbd2d0470` |
| `toy_text/taxi/background.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/background.png` | `612671f71e299872b11748f3b7556fd6a9737ad5dbaf0fb940c8d27385e4b5a0` |
| `toy_text/taxi/house.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/house.png` | `d4019fdaaa30825e1197120cd26b86ba4df28e4af3c41f8cf47d9f001f71981a` |
| `toy_text/taxi/marker_destination.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/marker_destination.png` | `9b576678895bfb4bbda15146edc74939bcbbb11f0b573fe32dee2401b127a123` |
| `toy_text/taxi/passenger.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/passenger.png` | `a0fabf3438aa5b31efb3f18258392807ab192c82248b9dedf3f2c899ddbdbe5a` |
| `toy_text/taxi/taxi_h.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/taxi_h.png` | `de8046cbd185e0a4cf8eccfecba728d10e0e2e506877c9278f1b5d7118b9a4fc` |
| `toy_text/taxi/taxi_h_full.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/taxi_h_full.png` | `74d73f6c454786ba5f22f6c05439104453351a1187a248a93a2238ea8e73281e` |
| `toy_text/taxi/taxi_v.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/taxi_v.png` | `7c96136a32ab47e48b770bc3a88b314c2dbb24a10e2817d65f26e2b3edf27f14` |
| `toy_text/taxi/taxi_v_full.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/taxi/taxi_v_full.png` | `9b63362e0817b271ae50e6591e17318a3f75a025d8224e6c997628bbac232886` |
| `toy_text/blackjack/card_2C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_2C.png` | `5a1e03a1aca34e4ea8384d9c925d9e47a97dd5de7f6c64c353700d29b8a7fe3d` |
| `toy_text/blackjack/card_2D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_2D.png` | `9ae28c98c37bf91c25c65005970e7feebb8cc2091a479aad7ba13d1bd1716de5` |
| `toy_text/blackjack/card_2H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_2H.png` | `61a60476fd73a2793c45713469d2f4a0d2024b14e1e497486ed60ecd467143c6` |
| `toy_text/blackjack/card_2S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_2S.png` | `8ce5129c5f637bc38860050f88290d4ad5063e30c07f9a91672d513ef97561b4` |
| `toy_text/blackjack/card_3C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_3C.png` | `5a7660131ffc561b35dc2d37e1368b1a4faa309b1b1b494aa9e201a9fdb624e7` |
| `toy_text/blackjack/card_3D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_3D.png` | `354fc97fbe4544c9052291d6bec79bfff17d7417a37fa14645a3736c5f89f6a7` |
| `toy_text/blackjack/card_3H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_3H.png` | `4e0cd947a5105f42617c80818fb83a940276fe32d7f900034f89f5d880f37230` |
| `toy_text/blackjack/card_3S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_3S.png` | `14796f05462f42a5f6b3a301c2e1244e89388e7ed0572b19eb9039ca48d773b1` |
| `toy_text/blackjack/card_4C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_4C.png` | `2ce6eabe1c87ef894e87528e61b0d5b7a58a87065c2d37985101417e9f6a9b8c` |
| `toy_text/blackjack/card_4D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_4D.png` | `603b50ca19bd8370d0a71cbef54d4ea347552537dd768424f0820783412f35e5` |
| `toy_text/blackjack/card_4H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_4H.png` | `c62480bfb630106db4e2681d6ba322f92117855c42c06efb04976a85531866d7` |
| `toy_text/blackjack/card_4S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_4S.png` | `2fc3e22894de0bca69fe078f128696111900fa0b3d06bb1939ff7d5379bb8d5c` |
| `toy_text/blackjack/card_5C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_5C.png` | `a958e6eb68f2d1b36abcbc23b928274706a8f477e7e6495c48078cd3b366b93e` |
| `toy_text/blackjack/card_5D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_5D.png` | `2b5cd083b939cc213e66615db476ee3002dd5eb20e979c555960548f35f9d8c4` |
| `toy_text/blackjack/card_5H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_5H.png` | `e4044823d896480e2b7488402e8e45cf60bd628c26de8671cca5585a271e9617` |
| `toy_text/blackjack/card_5S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_5S.png` | `d2e95d7918924756c4d1707fc1be85f2ed4d80588d70f421abfebb4f2f2c4693` |
| `toy_text/blackjack/card_6C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_6C.png` | `7921942d0031c2ad07b3d74694a30221be8204e8e585585ab06f93a909357f3e` |
| `toy_text/blackjack/card_6D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_6D.png` | `18b48ea1228e2314c421ac6f08c741b18b43c8d514949effe6f9177d48117ec6` |
| `toy_text/blackjack/card_6H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_6H.png` | `d59686413e69296cddea9e4ccbf97865ad742a417e9dda036a1de2febb34ee3a` |
| `toy_text/blackjack/card_6S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_6S.png` | `3aff0e40bdf9e6ceb625fc5d2b65ec366267f5b1f2b23392ebf43293fed76434` |
| `toy_text/blackjack/card_7C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_7C.png` | `ca64ab0f1a0467bac691f06bf6e695b3c2b3c6a2c676eb7cb476ca04979dffcf` |
| `toy_text/blackjack/card_7D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_7D.png` | `0ec5df09cb2ad40737362bab475ba8cb4797074c5f4abadaaef1a5d0923ff5e2` |
| `toy_text/blackjack/card_7H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_7H.png` | `85593479b334377d20bba2db695c0246aebaa35b7ec6a957b89ce1506c271426` |
| `toy_text/blackjack/card_7S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_7S.png` | `b55b7dde29635a260b4b2f49010a4e5c8173c31cccf7d2fca2e76e065d5cbe19` |
| `toy_text/blackjack/card_8C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_8C.png` | `9a573646737bda812903e239c41e7df4d79f31d2d812fa4ab24bc80a2663fdb7` |
| `toy_text/blackjack/card_8D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_8D.png` | `cd524c1bcca2a42bdd87f90109f8a9e33e606916f380c879824bb73ad7508a36` |
| `toy_text/blackjack/card_8H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_8H.png` | `a6c02f8deae57a6804c88e4112bb3e0e8318a42ea2c3c554c6e8d7392394d6ce` |
| `toy_text/blackjack/card_8S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_8S.png` | `67d73db6ded6eef9b28aa27d8b0d28d072b2bf88d13dd91eab4a9688b2c3fc98` |
| `toy_text/blackjack/card_9C.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_9C.png` | `215a06a0709f027448900b4a0f05ef4144da28daa1a2cdaea56576853df9bfd9` |
| `toy_text/blackjack/card_9D.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_9D.png` | `d453abb6fa9c482433adccd1a01ab2781d424101530657e9375ba373a9442ba9` |
| `toy_text/blackjack/card_9H.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_9H.png` | `63f9bf3222080b7f2a8294912d02c68437e079dfb07d0b749efd9f7837998387` |
| `toy_text/blackjack/card_9S.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_9S.png` | `53bd8ed358c60d14281301f63cf52f9575c19638fa592cef6c63e5281f530be7` |
| `toy_text/blackjack/card_AC.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_AC.png` | `83dcedb1fa172cd79b788dc027948825ead0e360587a9ebb2ee1dd86cb447652` |
| `toy_text/blackjack/card_AD.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_AD.png` | `b8cc88d71fbcf79542d2df9e3b2961ebb6b0826c626054297e4374dc62f14df8` |
| `toy_text/blackjack/card_AH.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_AH.png` | `73dd41181cf9fa46dfd4f12b9dece161a4529259a1b28e1ee9bbbc0a470a08ef` |
| `toy_text/blackjack/card_AS.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_AS.png` | `54a0862989e5bcc62234e62b5599c0ac9cceaa9847ad3a7168c7b7d2fc64e091` |
| `toy_text/blackjack/card_JC.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_JC.png` | `9e129a92c3c311b67d113505881c4185f0ec91d9c23f6b10aa6f3e6725d8f09c` |
| `toy_text/blackjack/card_JD.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_JD.png` | `961cc01f539b5b8a8b2394530268716e6189f639eb8f00f11941d8876d2bb798` |
| `toy_text/blackjack/card_JH.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_JH.png` | `598c652589c482e67d69ca4e28991a8c8912564ee0543d654fce5e9d8aad0198` |
| `toy_text/blackjack/card_JS.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_JS.png` | `34698b365262aa635df0a5417bcb98fe0fdaa9526f8041b2d194c677be4cd662` |
| `toy_text/blackjack/card_KC.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_KC.png` | `5d3aaab4df850ad248ea9b69319198f709d70c57f17a84b2d4c6e42cf3dffe0d` |
| `toy_text/blackjack/card_KD.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_KD.png` | `bd1829cac6dd50a49fb4ee442b8a6b4f21cce2c44e20b26ca7a3c0897e8566db` |
| `toy_text/blackjack/card_KH.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_KH.png` | `c5655b16ec24e25b2fd96b42c75128ec36e7eb9e84647d88e7f350b81fa94482` |
| `toy_text/blackjack/card_KS.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_KS.png` | `e2a196bbcd428cfa8e0ed4cb297cfcc75b1cedc054382ddf23a412bb5fac8c6d` |
| `toy_text/blackjack/card_QC.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_QC.png` | `dfe0556046e401ae58c5d30ac5feafea52395bc9d89a899c581b80ee3d329558` |
| `toy_text/blackjack/card_QD.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_QD.png` | `64eaea92ccbcafc6573a2fc0eda8b92538c7c16960bbf163b2bc2f39e659b3bc` |
| `toy_text/blackjack/card_QH.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_QH.png` | `acd5e3a441577762291ebc1677e48a6b3e90308235ff396635edd85c3bd06da3` |
| `toy_text/blackjack/card_QS.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_QS.png` | `c283d6efea9898b625f3ad69d08bb0e12104e2fbe8dbcea86727e15144cb7a20` |
| `toy_text/blackjack/card_back.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/card_back.png` | `559ed908184bad0d0ff330f0d5cd6ac05d91b452a3e8927dfe3aac1f63678dd3` |
| `toy_text/blackjack/digit_big_0.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_0.png` | `66054d8e6eeb9b3ceb64c201adef132b12e4ff081081dbbedbfc26a5f6ebda71` |
| `toy_text/blackjack/digit_big_1.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_1.png` | `eb5fcdddda13075082ca6488c7da4e993da3f15e6edea382dfe8a61e62fe877d` |
| `toy_text/blackjack/digit_big_2.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_2.png` | `abcfa129a439f241a56a3eef660478bed76bc9aa6ce7983e73478c9a088babee` |
| `toy_text/blackjack/digit_big_3.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_3.png` | `544b729dc5aa4f2c42c88f036c48b9a50d3ba8e88fe526e579638c36a18130f0` |
| `toy_text/blackjack/digit_big_4.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_4.png` | `d9ff39a50a1fee3e75f36370f9b75c7ee46a87e84e95f4474924d8d6a9bb1f49` |
| `toy_text/blackjack/digit_big_5.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_5.png` | `09767c8e23a17bcf66501003429a7532fe993cf0d45c52881f75523ca6a7f84c` |
| `toy_text/blackjack/digit_big_6.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_6.png` | `4140e387b888b696f76e093e95e90830c7f1b6d7e57f31fbae73e88d863f85aa` |
| `toy_text/blackjack/digit_big_7.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_7.png` | `e2c5ab296b3c1e99ac7169ba90868da929f787e0140247f69c4c37eaaa3f6b52` |
| `toy_text/blackjack/digit_big_8.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_8.png` | `c25ad1a0b0f29426da48b7e193da1985693c2209db5329d7d8a120f4021521b5` |
| `toy_text/blackjack/digit_big_9.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_big_9.png` | `5adfc2b4bd9013b32942b29df99a9d45ef463366e739f2884477ae6d4516f9e2` |
| `toy_text/blackjack/digit_small_0.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_0.png` | `dc1ee7e512126e4031a950c987ecbb3a2dff234b0042f0273db93e45c6169ef9` |
| `toy_text/blackjack/digit_small_1.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_1.png` | `7da4c882995db14e20d48a2ab9d19a31000f7cf38bc5d4657820e78e3b7a2a5e` |
| `toy_text/blackjack/digit_small_2.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_2.png` | `771e81521cfa3184efd33ad3d52580ee09cb86f523afb2b0303e0e3cc64eda02` |
| `toy_text/blackjack/digit_small_3.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_3.png` | `b49d85e52906dca5ee997d5ffd80fe21167fb2606d505d7d8f713dd7f90c2430` |
| `toy_text/blackjack/digit_small_4.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_4.png` | `b2e7c65908faa20a2b4820f2c740e5c0910b1712b1b9b2267f5507f33fac3f11` |
| `toy_text/blackjack/digit_small_5.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_5.png` | `cd22b19e4c1bbe803d9dd80ec97c7a97ff52a98d27f75de540ef7bcc6028f400` |
| `toy_text/blackjack/digit_small_6.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_6.png` | `ba506a036eb11aed9c95368cf632276225e2a4fabbe4e989db9469d3ca54de93` |
| `toy_text/blackjack/digit_small_7.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_7.png` | `5f3248e20d31958938ad1c6ee46548187634a4bfe9f06e46347af90c3f499807` |
| `toy_text/blackjack/digit_small_8.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_8.png` | `9e7f05c7a9b02698378d9ab7305a3b5ca4d1369d5de88132b5d5c7bc4692dab9` |
| `toy_text/blackjack/digit_small_9.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/digit_small_9.png` | `dfaf4cb1c2d339127e18e84915cc5cb68a6debe7caf797cf9e36d321652bdbb7` |
| `toy_text/blackjack/label_dealer.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/label_dealer.png` | `3caf50e1d24e93ed6e2d7b289be33c622b9c806fee25a9769d5708b15c00aa94` |
| `toy_text/blackjack/label_player.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/label_player.png` | `572060fc4835852e5b131b2d6d05a3518fe357e765ae743d332f04ee8a938277` |
| `toy_text/blackjack/label_usable_ace.png` | `refs/Gymnasium/gymnasium/envs/toy_text/img/blackjack/label_usable_ace.png` | `4db00a4cfabeb98d68d0890762e1eb0c8d82abaea367aac9c363bec07a64b2bf` |

## fonts

CarRacing's score (`f"{self.reward:04.0f}"` in pygame's default font at size 42) as glyph data, embedded on the
maintainer's decision of 2026-11-11 (DR-48) and read by `ScoreFont`, which lays it out as SDL_ttf 2.24 does. It holds
the 15 characters the score can show (the digits, `-`, and the letters of `nan` and `inf`): each glyph's box, bearings,
hinted 26.6 advance and 8-bit coverage; the kerning of their 109 kerned pairs; the one ligature among them (`fi`), which
`ScoreFont` refuses; the font's ascent, height and line skip; and the source font's name and SHA-256.

| File | Source | Source SHA-256 | SHA-256 |
|---|---|---|---|
| `fonts/freesansbold-42.json` | pygame-ce 2.5.8's `freesansbold.ttf` (GNU FreeFont's FreeSans Bold, 2002), recorded by `tools/parity/record_score_font.py` | `bf9251269f11e4b35582a99376096deee3d0c4973a459cbd97b0bebc01f440ad` | `5b00872b898cbbebe2dad843c850a937a931e80bc45b45ceb42480606125b072` |

The parity cell `CarRacingLiveParityTests.GymNet_ScoreFont_EmbeddedDataIsTheLiveRecording` compares the file with a live
recording, so a pygame-ce re-pin that changes the font fails there. Re-record with the parity stack's pygame-ce:
`python tools/parity/record_score_font.py src/Gym.Environments/Assets/fonts/freesansbold-42.json`.

## mujoco

Gymnasium's 14 bundled MuJoCo models as MuJoCo 4.0.0 compiles them (DR-63, DR-64), embedded on the maintainer's
decision of 2026-11-12. Gym.NET never compiles MJCF: `MjModel.FromXmlPath("ant.xml")` reads `mujoco/ant.json`, so
every number the engine and the renderer read is MuJoCo's own, bit for bit, as CarRacing's glyphs are pygame's own. A
model file (format 1) holds the compiler's sizes, `opt`, `stat` and `vis`, the element names per kind, the textures,
and the arrays by MuJoCo's names: the physics fields the engine reads or validates and every `geom_*`, `site_*`,
`mat_*`, `tex_*` (but `tex_data`), `light_*` and `cam_*` array. Doubles are written by Python's `repr`, which .NET's
parser reads back to the same bits. `tools/parity/mujoco/record_models.py` documents every field.

| File | Source | Source SHA-256 | SHA-256 | Used by |
|---|---|---|---|---|
| `mujoco/ant.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/ant.xml` | `30cf9aa568dd66ce11a8f07191f656ddec47c6b405e86dd5fa73a8f02c47ceae` | `b6f27587d53d474146789bcb583912aa7211b8b1455dcd7729665ad3e204a374` | Ant-v4, Ant-v5 |
| `mujoco/half_cheetah.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/half_cheetah.xml` | `e39468f09810d648e1ae55d09677d02696087e7de5c32bdc3d31fffa69a6d0ff` | `3519aa3fe2f51fe5e8b32f090efe8ecbfa4ef34e234722d1d03f14e3f3a8515d` | HalfCheetah-v4, HalfCheetah-v5 |
| `mujoco/hopper.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/hopper.xml` | `53a729aa38026cf543c13f60934606ccd222c6d5b0a3b99172cfa2a8668fe294` | `cbb37a35b4ce2f5f985f044eefdc54cd08b16f1885e210e26ab2c14523020b4d` | Hopper-v4, Hopper-v5 |
| `mujoco/humanoid.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/humanoid.xml` | `28ddce0cefce55d9d7f1a54eb4a1f229fda1d4630828a95871e8ae620e367e7f` | `b18f6387562958cdf917a76d03947475c0f63f2503f2ceb900e2e9eb43074c1b` | Humanoid-v4, Humanoid-v5 |
| `mujoco/humanoidstandup.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/humanoidstandup.xml` | `c135edbf02a7b59a8371ba244e8c27a720aa4d8cf502f68157598e0b1106cb33` | `ee484fcb0e2138cdd91e7e7b03e2e817430b0dd6d86ce44281b1ef3bc7ae3a3b` | HumanoidStandup-v4, HumanoidStandup-v5 |
| `mujoco/inverted_double_pendulum.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/inverted_double_pendulum.xml` | `66fd208bbeb5b7a912d7bf2636398a6aeb9fce577c484e7565ca4f7a4c5fb250` | `d94aaaef1c687f779432fa3f090ce812e4a95f94224d5cbfee2d48b9bd1909f3` | InvertedDoublePendulum-v4, InvertedDoublePendulum-v5 |
| `mujoco/inverted_pendulum.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/inverted_pendulum.xml` | `4456cc21fff17f87bf7760eee8753a454fdc272a50f46e5c7e4669e4e240d07c` | `101104dc9f2ccbb1da8afa389b14a3f9afb46b5b2101289eb3cdfa5adbd8d566` | InvertedPendulum-v4, InvertedPendulum-v5 |
| `mujoco/point.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/point.xml` | `9ff7bb872106c2db7925ae918476a5d1a388e39412aee1c554becfbe49eed846` | `2ad3a58f3986f9c31643414b8c5bd4a8abeda31822410880338508b1dd4b2b2a` | no registered id: Gymnasium's custom-env test (`PointEnv`), mirrored in `TestMujocoCustomEnv` |
| `mujoco/pusher.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/pusher.xml` | `91b2983c6ad315636af04a167f458a87e05b97da82a0c050eaf95871b664ac4a` | `3f1ebc90693dead8349b4bb003956ff0dede5c064104f859e1a79f5158d0de18` | Pusher-v5 with `xml_file="pusher.xml"` (Pusher-v4 is refused, as Gymnasium refuses it on MuJoCo 3 and later) |
| `mujoco/pusher_v5.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/pusher_v5.xml` | `937f3487d73e16b22256d4d3ac3ed03db59cc17e631f4633c29fdac8833a0b50` | `c900ab72f8640d25c2a281d71e8287733e0c0c1340dbbb88c82c6700f7dab3b3` | Pusher-v5 |
| `mujoco/reacher.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/reacher.xml` | `8d3cd344fd96db12f939716e47a84bb67102a349f6e3c0db519c1125bf76fdbc` | `30b17d01a1bc788008a78b6460813bc8f5f913d663174dde626b7131897229da` | Reacher-v4, Reacher-v5 |
| `mujoco/swimmer.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/swimmer.xml` | `ef9a07a5967b1affdc0ccca423dfa4e26227f72287d081af013fb2ecdc145249` | `35e042bf432a2f44093a18776b92bf50e2eff685ce48ecf353f40d037ba142ca` | Swimmer-v4, Swimmer-v5 |
| `mujoco/walker2d.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/walker2d.xml` | `dde87cd83e4e69e139aa7d1782321e7f62c846eebabfb41c9300fbf96eeaaf85` | `5450228f749e3ee93298910ca085e1666ff5ab8f242fb2297f2366771b3631e7` | Walker2d-v4 |
| `mujoco/walker2d_v5.json` | `refs/Gymnasium/gymnasium/envs/mujoco/assets/walker2d_v5.xml` | `bdc043313cae4b718c97f929175e27213666af5b997eedc953479fdb6c9daeb7` | `cd2d8601a387e63960e1c45551b87eab01ceb4f8ebadb9c8f8b931de35cb8eea` | Walker2d-v5 |

The textures are MuJoCo's builtin textures (`builtin="gradient"`, `"flat"` with its marks and noise, and `"checker"`
in the XMLs), generated by the compiler, not image files of Gymnasium's: each is stored once as a lossless 8-bit RGB
PNG named by the first 16 hex digits of the SHA-256 of its raw pixels (MuJoCo's `tex_data` slice), which the model
file records in full beside the PNG's path. The inverted pendulums, Pusher and Reacher have none.

| File | Texture | Size | Pixels' SHA-256 (in the model files) | SHA-256 | Used by |
|---|---|---|---|---|---|
| `mujoco/textures/ce75c84c42cbeced.png` | skybox (gradient) | 100 × 600 | `ce75c84c42cbecedd913a5a95b7b57561944563ccb91ecb32ef56b1bb46fb343` | `8616f21254a81531e2ee71b89f536fed2a112e7a85fbdebeed3efce6972e9929` | ant, half_cheetah, point, swimmer |
| `mujoco/textures/4148cd0be74ddcde.png` | skybox (gradient) | 100 × 600 | `4148cd0be74ddcdec7e4f62298778b284763ea3c7e8cbf15989034c26521fb98` | `679786935bb2d95362fc2902d6f12c56c5f4ccda5fc30183a9015059e3bce673` | hopper, humanoid, humanoidstandup, walker2d, walker2d_v5 |
| `mujoco/textures/639956a14614a17d.png` | `texgeom` (cube) | 127 × 762 | `639956a14614a17deea04796f5191be9227b1eeebe793f39cf30a440ad3048a8` | `b845f69a95d6c643b440c3f358106bb9b2835710c775a6f8fb0052e1afac3d82` | the nine models with a skybox |
| `mujoco/textures/6baa02e8cd04a4ae.png` | `texplane` (2D checker) | 100 × 100 | `6baa02e8cd04a4aeccb0dbd48bbfc4e851e1ece471bfb516e1a68e9289f537fb` | `bfd5afd735b25c1ed33c6074fafcd4a49a79dc7970a128addd0d3865000dda41` | the nine models with a skybox |

Re-record after a re-pin of MuJoCo or Gymnasium, with a Python holding the pinned `mujoco` and NumPy:
`python -I tools/parity/mujoco/record_models.py`; the files change only where the compiler's output changed, and
`--check` compares the committed files with a fresh recording instead (exit 1 on any difference). When they were
embedded, the files were compared with the models the trace checker's 50 MuJoCo 4.0.0 traces
(`tools/parity/mujoco/halfcheetah_poc`) ran on, field by field: 9,450 fields of the 13 models the traces use (all but
`pusher`, which Pusher-v5 takes only through `xml_file`), all equal.
