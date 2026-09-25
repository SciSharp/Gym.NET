# Third-party notices

Gym.NET is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)). Parts of it are ports of third-party software,
listed below with the notices their licenses require. The port plan's §4.3
([docs/plans/GYMNASIUM_PORT_PLAN.md](docs/plans/GYMNASIUM_PORT_PLAN.md)) keeps this list in step with what is ported.

---

## Gymnasium (MIT)

Gym.NET's environments, spaces and utilities are ports of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium),
pinned as the `refs/Gymnasium` submodule. Ported so far: `gymnasium/envs/classic_control/cartpole.py` and `utils.py`
(`src/Gym.Environments/Envs/Classic`), `gymnasium/logger.py` and `gymnasium/utils/colorize.py` (`src/Gym/Logger.cs`),
`gymnasium/utils/seeding.py` (`src/Gym/Utils/Seeding.cs`), `gymnasium/error.py` (`src/Gym/Exceptions`), and parts of
`gymnasium/spaces` (`src/Gym/Spaces`).

```
The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## SDL_gfx 2.0.26 (zlib)

`src/Gym.Rendering/SdlGfx/SdlGfxPrimitives.cs` is a C# translation of part of `SDL_gfxPrimitives.c` from
[SDL_gfx](https://www.ferzkopp.net/wordpress/2016/01/02/sdl_gfx-sdl2_gfx/) 2.0.26, the library behind `pygame.gfxdraw`.
It is an altered version, marked as such in the file: translated to C#, reduced to the primitives Gymnasium's envs use,
specialized to 32-bit surfaces without an alpha mask, and aligned with the copy pygame-ce 2.5.8 bundles.

```
SDL_gfxPrimitives.c: graphics primitives for SDL surfaces

Copyright (C) 2001-2012  Andreas Schiffler

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.

2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.

3. This notice may not be removed or altered from any source
distribution.

Andreas Schiffler -- aschiffler at ferzkopp dot net
```

---

## pygame-ce (behaviour reference only)

`src/Gym.Rendering` reproduces the observable behaviour of [pygame-ce](https://github.com/pygame-community/pygame-ce)
2.5.8 (LGPL-2.1): the argument conversion of `pygame.gfxdraw`, `Surface.fill`/`blit`, `transform.flip`, `surfarray`, and
`math.Vector2.rotate_rad`. None of pygame-ce's own source code is copied: its behaviour was established by reading its
source and documentation and by probing it, the implementations are Gym.NET's own, and the result is verified pixel for
pixel and bit for bit by the live parity suite (`tests/Gym.Tests.Parity/Tests/Rendering`). Porting pygame's own `draw`
module, if ever needed, is decision D4 of the port plan.
