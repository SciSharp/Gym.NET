# Third-party notices

Gym.NET is licensed under the Apache License 2.0 (see [LICENSE](LICENSE)), every package of it (the maintainer's
decision DR-49 of [docs/plans/DESIGN_REVIEW.md](docs/plans/DESIGN_REVIEW.md), 2026-11-11). Parts of Gym.NET follow
third-party source code or carry third-party files and data; they are listed below with their sources' notices. The port plan's §4.3
([docs/plans/GYMNASIUM_PORT_PLAN.md](docs/plans/GYMNASIUM_PORT_PLAN.md)) keeps this list in step with the code.

---

## Gymnasium (MIT)

Gym.NET reproduces [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)'s behavior, pinned at v1.4.0
(`8ba8c3a63bb4e000b7f33923f43d292acac07be1`) as the `refs/Gymnasium` submodule; each member based on Gymnasium's source
links the file it follows in its documentation. As of 2.0 that covers:

- the core API, errors, logger and seeding (`gymnasium/core.py`, `error.py`, `logger.py`, `utils/seeding.py`):
  `src/Gym/Envs`, `src/Gym/Exceptions`, `src/Gym/Logger.cs`, `src/Gym/Utils`;
- the spaces and their utilities (`gymnasium/spaces`): `src/Gym/Spaces`;
- registration and `make`/`make_vec` (`gymnasium/envs/registration.py`): `src/Gym/Envs/Registration`, `src/Gym/Gymnasium.cs`;
- the env checkers and step-API helpers (`gymnasium/utils/env_checker.py`, `passive_env_checker.py`, `env_match.py`,
  `step_api_compatibility.py`): `src/Gym/Utils`;
- the wrappers 2.0 includes (`gymnasium/wrappers`): `src/Gym/Wrappers`;
- the vector API (`gymnasium/vector`): `src/Gym/Vector`;
- the classic_control and toy_text environments (`gymnasium/envs/classic_control`, `gymnasium/envs/toy_text`):
  `src/Gym.Environments/Envs/Classic`, `src/Gym.Environments/Envs/ToyText`;
- the images those environments draw, copied unchanged from Gymnasium v1.4.0 and embedded in Gym.NET.Environments
  (`src/Gym.Environments/Assets`, whose README lists every file with its source path and SHA-256). The toy_text art is
  the Farama Foundation's own, under the same license.

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

## Box2D 2.3.0 and pybox2d 2.3.10 (zlib)

`src/Gym.Box2D` (the `Gym.Box2D` assembly, shipped inside Gym.NET.Environments) is a C# translation of the
[Box2D](https://box2d.org) 2.3.0 C++ sources that [pybox2d](https://github.com/pybox2d/pybox2d) 2.3.10 bundles
(`Box2D/Common`, `Collision`, `Dynamics`: the parts Gymnasium's box2d environments reach), and of pybox2d's SWIG
extensions (`Box2D/*.i`) for the Python-shaped surface those environments call. Every file is an altered version, marked
as such under the notice it keeps: translated to C#, reduced to the shapes and joints Gymnasium uses (circle, edge,
polygon; revolute), with Box2D's allocators, timers, GJK and TOI call statistics and debug drawing removed and its
assertions raising Python's `AssertionError`, as pybox2d compiles them. Each file keeps its own copyright line; the notice below is the one
they share, and pybox2d's package notice follows it.

```
Copyright (c) 2006-2012 Erin Catto http://www.box2d.org

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
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
3. This notice may not be removed or altered from any source distribution.
```

```
This software is covered under the zlib license.

C++ version Copyright (c) 2006-2011 Erin Catto http://www.box2d.org
Python version Copyright (c) 2008-2011 Ken Lauer / sirkne at gmail dot com

Implemented using the pybox2d SWIG interface for Box2D (pybox2d.googlecode.com)

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
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
3. This notice may not be removed or altered from any source distribution.
```

---

## SDL_ttf 2.24.0 (zlib)

`src/Gym.Environments/Envs/Box2D/ScoreFont.cs` is a C# translation of the single-line path of `SDL_ttf.c` from
[SDL_ttf](https://github.com/libsdl-org/SDL_ttf) 2.24.0, the library behind `pygame.font`: the HarfBuzz layout of
`TTF_Size_Internal`, the row height of `TTF_Render_Wrapped_Internal` and the shaded blit of `Render_Line`, which lay out
CarRacing's score. It is an altered version, marked as such in the file: translated to C# and driven by recorded glyph
data instead of FreeType and HarfBuzz, refusing the texts HarfBuzz would shape into a ligature. The glyph data has its
own section below.

```
SDL_ttf:  A companion library to SDL for working with TrueType (tm) fonts
Copyright (C) 2001-2025 Sam Lantinga <slouken@libsdl.org>

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
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
3. This notice may not be removed or altered from any source distribution.
```

---

## FreeSans Bold (GNU FreeFont): CarRacing's score glyphs

`src/Gym.Environments/Assets/fonts/freesansbold-42.json`, embedded in Gym.NET.Environments, holds glyph data recorded
from pygame-ce 2.5.8's default font, `freesansbold.ttf` (SHA-256
`bf9251269f11e4b35582a99376096deee3d0c4973a459cbd97b0bebc01f440ad`): the size-42 rasterizations (8-bit coverage,
bearings, hinted advances) of the 15 characters CarRacing's score can show, the kerning of their pairs and their one
ligature, recorded by `tools/parity/record_score_font.py` (decisions DR-27 and DR-48 of
[docs/plans/DESIGN_REVIEW.md](docs/plans/DESIGN_REVIEW.md)). The font file itself is not included. The font is GNU
FreeFont's FreeSans Bold, whose name table reads:

```
Copyright 2002 Free Software Foundation.
$Revision: 0.4 $
The use of this font is granted subject to GNU General Public License.
```

---

## CPython 3.12 (PSF License Version 2)

Three files in `src/Gym/Utils` follow CPython v3.12.12's source so that Gym.NET's error messages and printed values
match Python's character for character:

- `Difflib.cs` follows `Lib/difflib.py`: `SequenceMatcher` (its matching-block search, junk heuristic and ratio) and
  `get_close_matches`, which Gymnasium's registry uses for its "Did you mean" suggestions. Changes: translated to C#,
  reduced to what `get_close_matches` needs, over strings only.
- `PyJson.cs` follows `Lib/json/encoder.py` and `Lib/json/decoder.py`: the text `json.dumps` writes and `json.loads`
  reads for `EnvSpec.to_json`/`from_json`. Changes: written in C# over Gym.NET's value types, reduced to the value kinds
  an `EnvSpec` holds.
- `PyMath.cs` follows `Objects/floatobject.c` and `Python/bltinmodule.c`: Python's `//`, `%`, `divmod` and `round` on
  floats and integers. Changes: written in C# over .NET's numeric types.

Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation; All Rights Reserved.

```
PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2
--------------------------------------------

1. This LICENSE AGREEMENT is between the Python Software Foundation
("PSF"), and the Individual or Organization ("Licensee") accessing and
otherwise using this software ("Python") in source or binary form and
its associated documentation.

2. Subject to the terms and conditions of this License Agreement, PSF hereby
grants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,
analyze, test, perform and/or display publicly, prepare derivative works,
distribute, and otherwise use Python alone or in any derivative version,
provided, however, that PSF's License Agreement and PSF's notice of copyright,
i.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023 Python Software Foundation;
All Rights Reserved" are retained in Python alone or in any derivative version
prepared by Licensee.

3. In the event Licensee prepares a derivative work that is based on
or incorporates Python or any part thereof, and wants to make
the derivative work available to others as provided herein, then
Licensee hereby agrees to include in any such work a brief summary of
the changes made to Python.

4. PSF is making Python available to Licensee on an "AS IS"
basis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND
DISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS
FOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT
INFRINGE ANY THIRD PARTY RIGHTS.

5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON
FOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS
A RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,
OR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.

6. This License Agreement will automatically terminate upon a material
breach of its terms and conditions.

7. Nothing in this License Agreement shall be deemed to create any
relationship of agency, partnership, or joint venture between PSF and
Licensee.  This License Agreement does not grant permission to use PSF
trademarks or trade name in a trademark sense to endorse or promote
products or services of Licensee, or any third party.

8. By copying, installing or otherwise using Python, Licensee
agrees to be bound by the terms and conditions of this License
Agreement.
```

---

## OpenCV 4.13.0 (Apache-2.0)

`src/Gym/Utils/Cv2.cs` is a C# translation of part of `modules/imgproc/src/resize.cpp` from
[OpenCV](https://github.com/opencv/opencv) 4.13.0 (decision DR-23), the library behind `cv2.resize`, which Gymnasium's
`ResizeObservation` calls. It is an altered version, marked as such in the file: translated to C#, reduced to the uint8
and float32 area (`INTER_AREA`) and bilinear (`INTER_LINEAR`) paths, single-threaded, without OpenCV's IPP and OpenCL
code; the SIMD kernels' order of summation is kept where it changes float results (the float32 2×2 area mean). Intel
IPP's float32 bilinear, which the wheels use by default, is not translated (it is not published). OpenCV 4.5 and later are licensed under the Apache License 2.0,
the license of Gym.NET itself (see [LICENSE](LICENSE)); the file `resize.cpp` carries this notice:

```
                          License Agreement
               For Open Source Computer Vision Library

Copyright (C) 2000-2008, 2017, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistribution's of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistribution's in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * The name of the copyright holders may not be used to endorse or promote products
    derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
```

The translation is verified bit for bit against live `cv2` 4.13.0 by the parity suite
(`tests/Gym.Tests.Parity/Tests/Utils/Cv2ResizeLiveParityTests.cs`; float32 `INTER_LINEAR` with IPP switched off).

---

## pygame-ce 2.5.8: `draw.aaline` and `draw.aalines` (LGPL-2.0-or-later)

`src/Gym.Rendering/Draw.Aaline.cs` is a C# translation of the functions `aaline`, `aalines`, `draw_aaline`,
`get_antialiased_color` and `set_and_check_rect` of `src_c/draw.c` from
[pygame-ce](https://github.com/pygame-community/pygame-ce) 2.5.8 (decision DR-37 of
[docs/plans/DESIGN_REVIEW.md](docs/plans/DESIGN_REVIEW.md)). It is an altered version, marked as such in the file with
the date of the change (2026-11-10): translated from C to C#, reduced to width 1 on 32-bit surfaces without per-pixel
alpha, without pygame's drawn-area bookkeeping and deprecated `blend` argument, and refusing NaN and infinite
coordinates.

Its source's notice, which the file keeps, follows. The `Gym.NET.Rendering` package that compiles it is licensed
Apache-2.0, as every Gym.NET package is (DR-49, which replaced DR-37's `Apache-2.0 AND LGPL-2.0-or-later` on
2026-11-11). The complete source is this repository.

```
pygame-ce - Python Game Library
Copyright (C) 2000-2001  Pete Shinners

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Library General Public
License as published by the Free Software Foundation; either
version 2 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Library General Public License for more details.

You should have received a copy of the GNU Library General Public
License along with this library; if not, write to the Free
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
```

The translation is verified pixel for pixel against live pygame-ce by the parity suite
(`tests/Gym.Tests.Parity/Tests/Rendering/PygameDrawLiveParityTests.cs`, and MountainCar's frames in
`tests/Gym.Tests.Parity/Tests/Envs`).

---

## pygame-ce (behaviour reference only)

Beyond the translation above, `src/Gym.Rendering` reproduces the observable behaviour of
[pygame-ce](https://github.com/pygame-community/pygame-ce) 2.5.8 (LGPL-2.1). None of pygame-ce's own source code is
copied there, and the implementations are Gym.NET's own:

- the argument conversion of `pygame.gfxdraw`, `Surface.fill`/`blit`, `transform.flip`, `surfarray` and
  `math.Vector2.rotate_rad`, established by reading pygame-ce's source and documentation and by probing it;
- `draw.line`, `draw.rect`, per-pixel alpha and its blit, `transform.scale`, `transform.smoothscale` and
  `image.load`/`frombytes`, derived from pygame-ce's output alone, with none of its source read (decision D4 of the
  completion plan).

Everything is verified pixel for pixel and bit for bit against live pygame-ce by the parity suite
(`tests/Gym.Tests.Parity/Tests/Rendering`).
