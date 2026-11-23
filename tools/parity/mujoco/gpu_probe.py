"""Measures the GPU behaviours Gym.NET's MuJoCo renderer reproduces, in MuJoCo's own GL context, and scores the models.

MuJoCo's classic renderer (the frames Gymnasium returns) draws with OpenGL's fixed-function pipeline, and the GPU decides
details the OpenGL specification leaves open. Gym.NET's renderer (src/Gym.Environments/Envs/MuJoCo/Rendering) reproduces
the reference GPU's choices, measured with this script on 2026-11-12 (NVIDIA GeForce RTX 4060 Ti, driver 616.56):

    store       a colour reaches the 8-bit buffer truncated to 12 bits, then rounded with an exact half going down
                (GlContext.ToUnorm8)
    blend       the source colour and alpha are stored to 8 bits first, then blended exactly in integers, rounded
                (GlContext.WriteSample)
    resolve     the 4-sample resolve blit averages the samples' levels rounding a quarter up from three, a half down,
                but the halves just below 16, 32, ..., 112 up (MujocoFrame.ResolveFour; every one of the 1,021 sums,
                measured 2026-11-21)
    matrices    the driver's stack: glMultMatrixf and glTranslatef sum pairwise, (a0 b0 + a1 b1) + (a2 b2 + a3 b3);
                glFrustum divides double numerators by float32-rounded denominators; glOrtho is double, rounded once
                (GlMath.MultiplyDriver, GlMath.Frustum, GlMath.Ortho)
    vertex      the fixed-function position is fma(m12, w, fma(m8, z, fma(m4, y, m0 x))) over P * MV
                (GlMath.TransformGpu)
    rcp         the viewport stage's 1/w is faithful but not correctly rounded: 13% of mantissas one ulp off, measured
                for all 2^23 into Assets/mujoco/gpu/rcp.bin (GlReciprocal.Of); z / w is RN(z * rcp(w))
    snap        a vertex's window position is ((x rcp(w) + 1) 0.5) size in float32, snapped to 1/256 of a pixel with
                ties to even (GlContext.Triangle), located through a screen-linear varying to 1e-4 of a pixel
    ties        a sample exactly on an edge belongs to the triangle whose left or bottom edge it is, in OpenGL's y-up
                window (GlContext.LeftOrBottom; measured 2026-11-22)
    centroid    an edge pixel's colour is evaluated at its first covered sample, texture coordinates at the pixel centre
                (GlContext.RasterRows)
    split       GL_QUADS draws (0, 1, 2) and (0, 2, 3)
    lighting    per-vertex lighting is OpenGL's equation in float
    bilinear    texture weights are rounded to 1/256 (GlTexture.Bilinear)
    lod_table   the level of detail is the exponent plus a 256-entry table of the mantissa truncated to 8 bits, in 1/256
                steps (GlTexture.LodSteps, GlTexture.Log2Steps)
    lod_norm    a derivative's length is max + 0.3442 min of its texel components, truncated to 9 fractional bits at the
                larger one's exponent (GlContext.Length; fitted, not exact)
    lod_quad    one level of detail per 2 x 2 quad, from the differences at its lower-left pixel (GlContext.Shade)
    mipmaps     glGenerateMipmap's chain against the area box with ties down (GlTexture.Downsample)

Each probe prints how many measured values its model reproduces, so a new GPU or driver shows at once which of Gym.NET's
models no longer hold. The MuJoCo frame cells (DR-71) are pinned to the reference GPU for this reason; after re-pinning
them to another machine, run this first and compare.

Usage (with the parity stack's interpreter, which has mujoco 4.0.0 and PyOpenGL):
    python -I tools/parity/mujoco/gpu_probe.py [--out <dir>]

Args (command line):
    --out: where the dense level-of-detail table is written as log2_table.json; nothing is written without it.
    --rcp-table: where the viewport reciprocal's table is written (Assets/mujoco/gpu/rcp.bin); without it, the
        embedded table is checked against this GPU.

Returns (exit code): 0 when every exact model reproduces every measured value (lod_norm and mipmaps report their known,
fitted shortfalls without failing); 1 otherwise.
"""
import argparse
import json
import math
import os
import re
import sys

import mujoco
import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

# the framebuffer the probes draw into: 512 x 256 quads of 2 x 2 pixels
W, H = 1024, 512
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SAMPLES = [(0.375, 0.125), (0.875, 0.375), (0.125, 0.625), (0.625, 0.875)]
FETCH = """
#version 330 compatibility
uniform sampler2DMS ms;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    gl_FragColor = texelFetch(ms, ivec2(p.x / 4, p.y), p.x % 4);
}
"""
# writes sample s of each pixel from a lookup texture holding four samples per pixel side by side
WRITE_SAMPLE = """
#version 330 compatibility
uniform sampler2D lut;
uniform int s;
void main() { ivec2 p = ivec2(gl_FragCoord.xy); gl_FragColor = texelFetch(lut, ivec2(p.x * 4 + s, p.y), 0); }
"""
# the fixed-function position, which GLSL defines ftransform() to reproduce exactly
FTRANSFORM = """
#version 120
void main() { gl_Position = ftransform(); }
"""


def store12(v: float) -> int:
    """Gym.NET's model of the GPU's colour store (GlContext.ToUnorm8).

    Args:
        v: the channel.
    Returns:
        The stored byte.
    """
    if not v > 0:
        return 0
    c12 = 4096 if v >= 1 else int(math.floor(v * 4096))
    return (c12 * 255 + 2047) >> 12


class Context:
    """MuJoCo's GL context with a single-sample RGBA8 framebuffer object and MuJoCo's fixed-function defaults."""

    def __init__(self) -> None:
        """Creates the context and the framebuffer.

        Raises:
            RuntimeError: when the framebuffer object is incomplete.
        """
        self.gl = mujoco.GLContext(64, 64)
        self.gl.make_current()
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        rb = GL.glGenRenderbuffers(1)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_RGBA8, W, H)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb)
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("incomplete framebuffer")
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        self.reset()

    def reset(self, width: int = W, height: int = H) -> None:
        """Binds the framebuffer with a window-space orthographic projection and plain fixed-function state.

        Args:
            width: the viewport's width.
            height: the viewport's height.
        """
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, width, 0, height, -1, 1)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        for cap in (GL.GL_DEPTH_TEST, GL.GL_LIGHTING, GL.GL_DITHER, GL.GL_BLEND, GL.GL_CULL_FACE, GL.GL_TEXTURE_2D):
            GL.glDisable(cap)
        GL.glTexEnvi(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_MODULATE)
        GL.glColor4f(1, 1, 1, 1)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

    @staticmethod
    def read(width: int = W, height: int = H) -> np.ndarray:
        """Reads the bound framebuffer.

        Args:
            width: columns to read.
            height: rows to read.
        Returns:
            height x width x 4 int array, row 0 at the bottom.
        """
        data = GL.glReadPixels(0, 0, width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE)
        return np.frombuffer(data, np.uint8).reshape(height, width, 4).astype(int)


def quad(x0: float, y0: float, x1: float, y1: float, st=None) -> None:
    """Draws one window-space quad, optionally with texture coordinates at its corners.

    Args:
        x0: left.
        y0: bottom.
        x1: right.
        y1: top.
        st: four (s, t) pairs for (x0, y0), (x1, y0), (x1, y1), (x0, y1), or None.
    """
    GL.glBegin(GL.GL_QUADS)
    for k, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
        if st is not None:
            GL.glTexCoord2f(*st[k])
        GL.glVertex2f(x, y)
    GL.glEnd()


def texture(levels: list, nearest: bool = False) -> int:
    """Uploads a 2D texture with explicit levels and MuJoCo's parameters.

    Args:
        levels: H x W x 3 uint8 arrays, level 0 first.
        nearest: GL_NEAREST filters instead of MuJoCo's (a probe of where coordinates are evaluated).
    Returns:
        The texture name.
    """
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST if nearest else GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST if nearest else GL.GL_LINEAR_MIPMAP_LINEAR)
    for k, a in enumerate(levels):
        h, w, _ = a.shape
        GL.glTexImage2D(GL.GL_TEXTURE_2D, k, GL.GL_RGB, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(a).tobytes())
    return tex


def level_texture() -> None:
    """A 256 x 256 texture whose odd levels are R 255 / B 0 and even levels R 0 / B 255: (R, B) gives the blend."""
    levels = []
    for k in range(9):
        n = 256 >> k
        a = np.zeros((n, n, 3), np.uint8)
        a[..., 0] = 255 if k % 2 else 0
        a[..., 2] = 0 if k % 2 else 255
        levels.append(a)
    GL.glEnable(GL.GL_TEXTURE_2D)
    texture(levels)


def blend_index(r: int, b: int, odd: bool):
    """The blend toward the next level, in 1/256 steps, that stores (R, B) under the 12-bit model.

    Args:
        r: the stored red.
        b: the stored blue.
        odd: whether the lower level is odd (R 255).
    Returns:
        The index 0..256, or None when no index stores this pair.
    """
    for k in range(257):
        f = k / 256
        if odd and (store12(1 - f), store12(f)) == (r, b):
            return k
        if not odd and (store12(f), store12(1 - f)) == (r, b):
            return k
    return None


def report(name: str, ok: int, total: int, exact: bool, note: str = "") -> bool:
    """Prints one probe's score.

    Args:
        name: the probe.
        ok: values the model reproduced.
        total: values measured.
        exact: whether the model claims every value (a fitted model only reports).
        note: what the numbers mean.
    Returns:
        False when an exact model missed a value.
    """
    verdict = "ok" if ok == total else ("MISSES" if exact else "fitted")
    print(f"{name:10s} {ok:7d}/{total:<7d} {verdict:7s} {note}")
    return ok == total or not exact


def probe_store(ctx: Context) -> bool:
    """Flat colours (i + d) / 255 around every half step: the stored byte against the 12-bit model.

    Args:
        ctx: the context.
    Returns:
        Whether the model held for every colour.
    """
    ctx.reset()
    vals = []
    n = 0
    for i in range(0, 255, 3):
        for d in np.arange(0.40, 0.601, 0.005):
            v = float(np.float32((i + d) / 255.0))
            GL.glColor4f(v, v, v, 1)
            qx, qy = 2 * (n % 512), 2 * (n // 512)
            quad(qx, qy, qx + 2, qy + 2)
            vals.append((v, qx, qy))
            n += 1
    img = ctx.read()
    ok = sum(int(img[qy, qx, 0] == store12(v)) for v, qx, qy in vals)
    return report("store", ok, len(vals), True, "flat colours at every half step, ties included")


def probe_blend(ctx: Context) -> bool:
    """Random sources over random destinations, SRC_ALPHA/ONE_MINUS_SRC_ALPHA and ONE/ONE, against the integer model.

    Args:
        ctx: the context.
    Returns:
        Whether the model held for every pixel.
    """
    rng = np.random.default_rng(11)
    n = 20000
    dst = rng.integers(0, 256, n)
    src = rng.uniform(0, 1, n).astype(np.float32)
    alpha = rng.uniform(0, 1, n).astype(np.float32)
    good = True
    for mode, (sf, df) in {"alpha": (GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA), "add": (GL.GL_ONE, GL.GL_ONE)}.items():
        ctx.reset()
        scale = 0.3 if mode == "add" else 1.0
        for i in range(n):
            qx, qy = 2 * (i % 512), 2 * (i // 512)
            GL.glColor4f(dst[i] / 255.0, dst[i] / 255.0, dst[i] / 255.0, 1)
            quad(qx, qy, qx + 2, qy + 2)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(sf, df)
        for i in range(n):
            qx, qy = 2 * (i % 512), 2 * (i // 512)
            s = float(src[i]) * scale
            GL.glColor4f(s, s, s, float(alpha[i]))
            quad(qx, qy, qx + 2, qy + 2)
        GL.glDisable(GL.GL_BLEND)
        img = ctx.read()
        ok = 0
        for i in range(n):
            s8, a8 = store12(float(src[i]) * scale), store12(float(alpha[i]))
            fs, fd = (a8, 255 - a8) if mode == "alpha" else (255, 255)
            want = min(255, (s8 * fs + int(dst[i]) * fd + 127) // 255)
            ok += int(img[2 * (i // 512), 2 * (i % 512), 0] == want)
        good &= report(f"blend.{mode}", ok, n, True, "source stored to 8 bits, then an exact integer blend")
    return good


def ms_texture():
    """A 4-sample RGBA8 texture and its framebuffer.

    Returns:
        (texture, framebuffer, the sample positions the driver reports).
    """
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, tex)
    GL.glTexImage2DMultisample(GL.GL_TEXTURE_2D_MULTISAMPLE, 4, GL.GL_RGBA8, 256, 128, GL.GL_TRUE)
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D_MULTISAMPLE, tex, 0)
    positions = [tuple(float(c) for c in GL.glGetMultisamplefv(GL.GL_SAMPLE_POSITION, i)) for i in range(4)]
    return tex, fbo, positions


def fetch_samples(ctx: Context, tex: int, prog) -> np.ndarray:
    """Every sample of the 256 x 128 multisample texture, through a texelFetch shader into the single-sample buffer.

    Args:
        ctx: the context.
        tex: the multisample texture.
        prog: the fetch program.
    Returns:
        128 x 256 x 4 x 4 int array (row, column, sample, channel).
    """
    ctx.reset(4 * 256, 128)
    GL.glUseProgram(prog)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D_MULTISAMPLE, tex)
    GL.glUniform1i(GL.glGetUniformLocation(prog, "ms"), 0)
    quad(0, 0, 4 * 256, 128)
    GL.glUseProgram(0)
    return Context.read(4 * 256, 128).reshape(128, 256, 4, 4)


def probe_multisample(ctx: Context) -> bool:
    """Sample positions, the centroid rule for colours and texture coordinates, and the resolve blit.

    Args:
        ctx: the context.
    Returns:
        Whether every exact model held.
    """
    tex, fbo, positions = ms_texture()
    good = report("positions", int(positions == SAMPLES), 1, True, f"{positions}")
    prog = shaders.compileProgram(shaders.compileShader(FETCH, GL.GL_FRAGMENT_SHADER))
    rng = np.random.default_rng(5)
    colour_ok = colour_n = tex_ok = tex_n = 0
    ramp = np.zeros((256, 256, 3), np.uint8)
    ramp[..., 0] = np.arange(256)[None, :]
    ramp[..., 1] = np.arange(256)[:, None]
    texture([ramp], nearest=True)
    for trial in range(16):
        cx, cy = 128 + rng.uniform(-20, 20), 64 + rng.uniform(-10, 10)
        ang = rng.uniform(0, 2 * math.pi)
        # vertices on the rasterizer's 1/256 grid, so the model's coverage and weights are the GPU's own
        tri = [(round((cx + 120 * math.cos(ang + k * 2.1)) * 256) / 256, round((cy + 60 * math.sin(ang + k * 2.1)) * 256) / 256)
               for k in range(3)]
        cvals = rng.uniform(0, 1, 3)
        textured = trial % 2 == 1
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glViewport(0, 0, 256, 128)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(0, 256, 0, 128, -1, 1)
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        if textured:
            GL.glEnable(GL.GL_TEXTURE_2D)
            GL.glColor4f(1, 1, 1, 1)
        GL.glBegin(GL.GL_TRIANGLES)
        for (x, y), c in zip(tri, cvals):
            if textured:
                GL.glTexCoord2f(x / 64.0 + 0.0005, y / 64.0 + 0.0005)
            else:
                GL.glColor4f(float(c), 0, 0, 1)
            GL.glVertex2f(x, y)
        GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        smp = fetch_samples(ctx, tex, prog)
        (x0, y0), (x1, y1), (x2, y2) = tri
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

        def bary(px, py):
            l1 = ((px - x0) * (y2 - y0) - (x2 - x0) * (py - y0)) / det
            l2 = ((x1 - x0) * (py - y0) - (px - x0) * (y1 - y0)) / det
            return 1 - l1 - l2, l1, l2

        for py in range(128):
            for px in range(256):
                alpha = smp[py, px, :, 3]
                cov = [int(a > 0) for a in alpha]
                if sum(cov) in (0, 4):
                    continue
                first = SAMPLES[cov.index(1)]
                values = {tuple(smp[py, px, s, :3]) for s in range(4) if cov[s]}
                if len(values) != 1:
                    continue
                got = values.pop()
                if textured:
                    # nearest texel of s = x/64 at the pixel centre: the coordinates are not centroid-sampled
                    er = int(math.floor(((px + 0.5) / 64.0 + 0.0005) * 256)) % 256
                    eg = int(math.floor(((py + 0.5) / 64.0 + 0.0005) * 256)) % 256
                    tex_ok += int((got[0], got[1]) == (er, eg))
                    tex_n += 1
                else:
                    l0, l1, l2 = bary(px + first[0], py + first[1])
                    colour_ok += int(got[0] == store12(l0 * cvals[0] + l1 * cvals[1] + l2 * cvals[2]))
                    colour_n += 1
    good &= report("centroid", colour_ok, colour_n, True, "edge pixels' colour at their first covered sample")
    good &= report("texcoords", tex_ok, tex_n, True, "edge pixels' texture coordinates at the pixel centre")

    # the resolve blit over overlapping random triangles, against the samples it averaged
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glViewport(0, 0, 256, 128)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glOrtho(0, 256, 0, 128, -1, 1)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glBegin(GL.GL_TRIANGLES)
    for _ in range(400):
        c = rng.uniform(0, 1, 3)
        cx, cy = rng.uniform(0, 256), rng.uniform(0, 128)
        for _k in range(3):
            GL.glColor4f(float(c[0]), float(c[1]), float(c[2]), 1)
            GL.glVertex2f(cx + rng.uniform(-30, 30), cy + rng.uniform(-30, 30))
    GL.glEnd()
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, ctx.fbo)
    GL.glBlitFramebuffer(0, 0, 256, 128, 0, 0, 256, 128, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, ctx.fbo)
    resolved = Context.read(256, 128)[..., :3]
    smp = fetch_samples(ctx, tex, prog)[..., :3]
    total = smp.sum(axis=2)
    mixed = (smp.max(axis=2) != smp.min(axis=2)).any(axis=2)
    ok = (resolve4(total) == resolved).all(axis=2)
    report("resolve", int(ok[mixed].sum()), int(mixed.sum()), True,
           "pixels whose samples differ, against MujocoFrame.ResolveFour")
    return good & report("resolve.eq", int(ok[~mixed].sum()), int((~mixed).sum()), True, "pixels whose samples agree")
def resolve4(total):
    """Gym.NET's model of the 4-sample resolve (MujocoFrame.ResolveFour), by the sum of the four levels.

    Args:
        total: the sums, any integer array.
    Returns:
        The resolved levels: (sum + 1) >> 2, with the exact halves just below 16, 32, ..., 112 rounded up.
    """
    level = (total + 1) >> 2
    up = ((total & 3) == 2) & ((total >> 2) < 127) & (((total >> 2) & 15) == 15)
    return level + up


def probe_resolve_sums(ctx: Context) -> bool:
    """Every possible sum of four sample levels through the resolve blit: samples written one at a time through the
    sample mask from a lookup texture, the multisample texture resolved by glBlitFramebuffer as mjr_readPixels does.

    Args:
        ctx: the context.
    Returns:
        Whether MujocoFrame.ResolveFour reproduced every resolved level.
    """
    rng = np.random.default_rng(11)
    w, h = 256, 128
    vals = rng.integers(0, 256, size=(h, w, 4, 4))
    near = np.clip(rng.integers(0, 255, size=(h, w, 1, 4)) + rng.integers(0, 2, size=(h, w, 4, 4)), 0, 255)
    vals = np.where((rng.integers(0, 2, size=(h, w)) == 1)[..., None, None], near, vals)
    vals[..., 3] = 255
    lut = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, lut)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w * 4, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                    np.ascontiguousarray(vals.reshape(h, w * 4, 4).astype(np.uint8)).tobytes())
    tex, fbo, _ = ms_texture()
    write = shaders.compileProgram(shaders.compileShader(WRITE_SAMPLE, GL.GL_FRAGMENT_SHADER))
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glViewport(0, 0, w, h)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glOrtho(0, w, 0, h, -1, 1)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
    GL.glUseProgram(write)
    GL.glActiveTexture(GL.GL_TEXTURE0)
    GL.glBindTexture(GL.GL_TEXTURE_2D, lut)
    GL.glUniform1i(GL.glGetUniformLocation(write, "lut"), 0)
    GL.glEnable(GL.GL_SAMPLE_MASK)
    for s in range(4):
        GL.glSampleMaski(0, 1 << s)
        GL.glUniform1i(GL.glGetUniformLocation(write, "s"), s)
        quad(0, 0, w, h)
    GL.glDisable(GL.GL_SAMPLE_MASK)
    GL.glUseProgram(0)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo)
    GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, ctx.fbo)
    GL.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, ctx.fbo)
    resolved = Context.read(w, h)[..., :3]
    total = vals[..., :3].sum(axis=2)
    ok = resolve4(total) == resolved
    sums = len(np.unique(total))
    return report("resolve4", int(ok.sum()), ok.size, True, f"every channel, {sums} distinct sums of the 1,021 possible")


def pair_product(a, b):
    """GlMath.MultiplyDriver: the driver's 4x4 product, each element's four products summed pairwise in float32.

    Args:
        a: the current matrix, column-major float32[16].
        b: the matrix it is multiplied by.
    Returns:
        The product, float32[16].
    """
    f32 = np.float32
    out = np.zeros(16, np.float32)
    for c in range(4):
        for r in range(4):
            p = [f32(a[k * 4 + r] * b[c * 4 + k]) for k in range(4)]
            out[c * 4 + r] = f32(f32(p[0] + p[1]) + f32(p[2] + p[3]))
    return out


def probe_matrix_stack() -> bool:
    """The driver's matrix stack read back with glGetFloatv after random glTranslatef, glMultMatrixf, glFrustum and
    glOrtho calls, against GlMath.MultiplyDriver's pairwise sums and GlMath.Frustum's and GlMath.Ortho's elements.

    Returns:
        Whether every element matched.
    """
    rng = np.random.default_rng(1)
    f32 = np.float32
    ok = total = 0
    GL.glMatrixMode(GL.GL_MODELVIEW)
    for _ in range(100):
        a, b = rng.uniform(-3, 3, 16).astype(np.float32), rng.uniform(-3, 3, 16).astype(np.float32)
        GL.glLoadMatrixf(a)
        GL.glMultMatrixf(b)
        ok += int((pair_product(a, b) == np.array(GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX), np.float32).reshape(16)).sum())
        v = rng.uniform(-5, 5, 3).astype(np.float32)
        t = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, v[0], v[1], v[2], 1], np.float32)
        GL.glLoadMatrixf(a)
        GL.glTranslatef(float(v[0]), float(v[1]), float(v[2]))
        ok += int((pair_product(a, t) == np.array(GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX), np.float32).reshape(16)).sum())
        total += 32
    ts = np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -0.5, 0, 0, 0, 0.5, 1], np.float32)
    GL.glMatrixMode(GL.GL_PROJECTION)
    for k in range(200):
        n, f = sorted(rng.uniform(0.001, 60, 2))
        if k % 2 == 0:
            w, h = rng.uniform(0.001, 0.2, 2)
            l, r, b, t = -w, w, -h, h
            rl, tb, fn = float(f32(r - l)), float(f32(t - b)), float(f32(f - n))
            m = np.array([2 * n / rl, 0, 0, 0, 0, 2 * n / tb, 0, 0, (r + l) / rl, (t + b) / tb, -(f + n) / fn, -1,
                          0, 0, -2 * f * n / fn, 0], np.float64).astype(np.float32)
        else:
            c = rng.uniform(0.5, 20)
            l, r, b, t = -c, c, -c, c
            m = np.array([2 / (r - l), 0, 0, 0, 0, 2 / (t - b), 0, 0, 0, 0, -2 / (f - n), 0,
                          -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n), 1], np.float64).astype(np.float32)
        GL.glLoadIdentity()
        GL.glTranslatef(0.0, 0.0, 0.5)
        GL.glScalef(1.0, 1.0, -0.5)
        (GL.glFrustum if k % 2 == 0 else GL.glOrtho)(float(l), float(r), float(b), float(t), float(n), float(f))
        ok += int((pair_product(ts, m) == np.array(GL.glGetFloatv(GL.GL_PROJECTION_MATRIX), np.float32).reshape(16)).sum())
        total += 16
    GL.glLoadIdentity()
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    return report("matrices", ok, total, True, "glMultMatrixf/glTranslatef pairwise, glFrustum/glOrtho elements")


def probe_vertex() -> bool:
    """The fixed-function position (GLSL's ftransform(), captured by transform feedback) against GlMath.TransformGpu
    over the projection times the modelview.

    Returns:
        Whether every component matched.
    """
    import ctypes
    rng = np.random.default_rng(2)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, shaders.compileShader(FTRANSFORM, GL.GL_VERTEX_SHADER))
    names = (ctypes.c_char_p * 1)(b"gl_Position")
    GL.glTransformFeedbackVaryings(prog, 1, ctypes.cast(names, ctypes.POINTER(ctypes.POINTER(GL.GLchar))), GL.GL_INTERLEAVED_ATTRIBS)
    GL.glLinkProgram(prog)
    GL.glEnable(GL.GL_RASTERIZER_DISCARD)
    buf = GL.glGenBuffers(1)
    f32 = np.float32
    ok = total = 0
    for _ in range(100):
        p = np.zeros(16, np.float32)
        p[0], p[5] = rng.uniform(1, 10, 2)
        p[8], p[9] = rng.uniform(-0.5, 0.5, 2)
        p[10], p[11], p[14] = rng.uniform(0.4, 0.6), -1, rng.uniform(0.001, 0.1)
        mv = rng.uniform(-2, 2, 16).astype(np.float32)
        mv[3] = mv[7] = mv[11] = 0
        mv[15] = 1
        v = rng.uniform(-3, 3, (16, 3)).astype(np.float32)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(p)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadMatrixf(mv)
        GL.glBindBuffer(GL.GL_TRANSFORM_FEEDBACK_BUFFER, buf)
        GL.glBufferData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 16 * 16, None, GL.GL_STATIC_READ)
        GL.glBindBufferBase(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, buf)
        GL.glUseProgram(prog)
        GL.glBeginTransformFeedback(GL.GL_POINTS)
        GL.glBegin(GL.GL_POINTS)
        for x, y, z in v:
            GL.glVertex3f(float(x), float(y), float(z))
        GL.glEnd()
        GL.glEndTransformFeedback()
        GL.glUseProgram(0)
        got = np.frombuffer(GL.glGetBufferSubData(GL.GL_TRANSFORM_FEEDBACK_BUFFER, 0, 256), np.float32).reshape(16, 4)
        mvp = pair_product(p, mv)
        for (x, y, z), g in zip(v, got):
            for r in range(4):
                # fma(m12, 1, fma(m8, z, fma(m4, y, m0 x))): each fused step is exact in float64, then rounded once
                acc = f32(mvp[r] * x)
                acc = f32(float(mvp[4 + r]) * float(y) + float(acc))
                acc = f32(float(mvp[8 + r]) * float(z) + float(acc))
                acc = f32(float(mvp[12 + r]) + float(acc))
                ok += int(acc == g[r])
                total += 1
    GL.glDisable(GL.GL_RASTERIZER_DISCARD)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    return report("vertex", ok, total, True, "ftransform() = fused x, y, z, w chain over P*MV")


def measure_rcp() -> np.ndarray:
    """The viewport stage's reciprocal 1/w for every mantissa: points at (x, y, 1, w) store rcp(w) as their depth.

    Returns:
        The 2^23 reciprocals of w = 1 + k * 2^-23, float32.
    """
    import ctypes
    w_px, h_px = 4096, 2048
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    rb = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH32F_STENCIL8, w_px, h_px)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_STENCIL_ATTACHMENT, GL.GL_RENDERBUFFER, rb)
    GL.glDrawBuffer(GL.GL_NONE)
    GL.glReadBuffer(GL.GL_NONE)
    GL.glClipControl(GL.GL_LOWER_LEFT, GL.GL_ZERO_TO_ONE)
    GL.glViewport(0, 0, w_px, h_px)
    GL.glPointSize(1.0)
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glLoadIdentity()
    GL.glClearDepth(0)
    GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
    GL.glEnable(GL.GL_DEPTH_TEST)
    GL.glDepthFunc(GL.GL_ALWAYS)
    k = np.arange(w_px * h_px, dtype=np.int64)
    w = (1.0 + k / 2.0 ** 23).astype(np.float32)
    verts = np.empty((w_px * h_px, 4), np.float32)
    verts[:, 0] = (((k % w_px) + 0.5) / w_px * 2 - 1) * w
    verts[:, 1] = (((k // w_px) + 0.5) / h_px * 2 - 1) * w
    verts[:, 2] = 1.0
    verts[:, 3] = w
    vbo = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
    GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
    GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
    GL.glVertexPointer(4, GL.GL_FLOAT, 0, ctypes.c_void_p(0))
    GL.glDrawArrays(GL.GL_POINTS, 0, w_px * h_px)
    GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
    depth = np.frombuffer(GL.glReadPixels(0, 0, w_px, h_px, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT), np.float32).reshape(-1).copy()
    GL.glClipControl(GL.GL_LOWER_LEFT, GL.GL_NEGATIVE_ONE_TO_ONE)
    GL.glDisable(GL.GL_DEPTH_TEST)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
    return depth


def probe_rcp(table_path) -> bool:
    """The viewport stage's reciprocal against the embedded table (Assets/mujoco/gpu/rcp.bin), and, with table_path,
    writes that table: per mantissa of w, the hardware's 1/w minus the correctly rounded one in ulps (-1, 0 or 1), two
    bits each (code = delta + 1, four to a byte, the first in the low bits), raw-deflated after an 8-byte header.

    Args:
        table_path: where to write the table, or None.
    Returns:
        Whether the table (written or embedded) reproduced every reciprocal.
    """
    import zlib
    hw = measure_rcp()
    k = np.arange(2 ** 23)
    rn = (np.float32(1) / (1.0 + k / 2.0 ** 23).astype(np.float32)).astype(np.float32)
    delta = hw.view(np.int32).astype(np.int64) - rn.view(np.int32).astype(np.int64)
    if not np.all(np.abs(delta) <= 1):
        return report("rcp", 0, 1, True, "a reciprocal more than one ulp from the rounded one: the table cannot hold it")
    code = (delta + 1).astype(np.uint8)
    packed = (code[0::4] | (code[1::4] << 2) | (code[2::4] << 4) | (code[3::4] << 6)).astype(np.uint8)
    if table_path:
        comp = zlib.compressobj(9, zlib.DEFLATED, -15)
        blob = b"GYMRCP1\0" + comp.compress(packed.tobytes()) + comp.flush()
        with open(table_path, "wb") as f:
            f.write(blob)
    path = table_path or os.path.join(REPO, "src", "Gym.Environments", "Assets", "mujoco", "gpu", "rcp.bin")
    with open(path, "rb") as f:
        stored = np.frombuffer(zlib.decompress(f.read()[8:], -15), np.uint8)
    ok = int(np.sum(stored == packed)) * 4 if stored.size == packed.size else 0
    return report("rcp", ok, 2 ** 23, True, f"1/w per mantissa, {int((delta != 0).sum())} one ulp off the rounded value")


def rcp_table() -> np.ndarray:
    """The embedded viewport reciprocal table (Assets/mujoco/gpu/rcp.bin) unpacked: per mantissa, the hardware's 1/w
    minus the correctly rounded one, in ulps.

    Returns:
        2^23 int64 deltas (-1, 0 or 1).
    Raises:
        OSError: when the table is missing.
    """
    import zlib
    with open(os.path.join(REPO, "src", "Gym.Environments", "Assets", "mujoco", "gpu", "rcp.bin"), "rb") as f:
        packed = np.frombuffer(zlib.decompress(f.read()[8:], -15), np.uint8)
    codes = np.empty(packed.size * 4, np.int64)
    for k in range(4):
        codes[k::4] = (packed >> (2 * k)) & 3
    return codes - 1


def rcp_gpu(w: np.float32, deltas: np.ndarray) -> np.float32:
    """Gym.NET's model of the viewport stage's 1/w (GlReciprocal.Of) for a positive normal float32.

    Args:
        w: the clip w.
        deltas: rcp_table()'s deltas.
    Returns:
        The rounded reciprocal of w's mantissa moved by the table's delta, scaled back by w's exponent.
    """
    bits = int(np.array([w], np.float32).view(np.int32)[0])
    e = ((bits >> 23) & 0xFF) - 127
    m = np.array([(bits & 0x7FFFFF) | (127 << 23)], np.int32).view(np.float32)[0]
    r = np.array([np.float32(1) / m], np.float32).view(np.int32)[0] + deltas[bits & 0x7FFFFF]
    return np.float32(float(np.array([r], np.int32).view(np.float32)[0]) * 2.0 ** (-e))


SNAP_VS = """
#version 330 compatibility
noperspective out float lam;
void main() { gl_Position = gl_Vertex; lam = gl_MultiTexCoord0.x; }
"""
SNAP_FS = """
#version 330 compatibility
noperspective in float lam;
out vec4 color;
void main() { color = vec4(lam, 0, 0, 1); }
"""


def probe_snap() -> bool:
    """Where the rasterizer puts a vertex given in clip coordinates, against GlContext.Triangle's window transform.

    For each trial a vertex V gets random clip coordinates (w from 0.05 to 60); two triangles (V, P1, P2) and
    (V, P3, P4), their far vertices at exact window positions on the 1/256 grid, are drawn with a noperspective varying
    that is 1 at V and 0 at the others, stored as float32. The varying is a plane over each triangle's pixels that
    reaches 1 on a line through V, so fitting both planes and intersecting the two lines locates the GPU's snapped V to
    about 1e-4 of a pixel, well inside the grid's 1/256.

    Returns:
        Whether the model, ((x * rcp(w) + 1) * 0.5) * size in float32 rounded to 1/256 with ties to even, put every
        located vertex on the GPU's grid point.
    """
    f32 = np.float32
    sw, sh = 1024, 512
    deltas = rcp_table()
    rng = np.random.default_rng(7)
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    rb = GL.glGenRenderbuffers(1)
    GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, rb)
    GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_R32F, sw, sh)
    GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, rb)
    GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
    GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
    # MuJoCo's clip control: the window depth is z / w, and the transform below is the one its frames go through
    GL.glClipControl(GL.GL_LOWER_LEFT, GL.GL_ZERO_TO_ONE)
    GL.glViewport(0, 0, sw, sh)
    for cap in (GL.GL_LIGHTING, GL.GL_BLEND, GL.GL_CULL_FACE, GL.GL_DITHER, GL.GL_DEPTH_TEST, GL.GL_MULTISAMPLE):
        GL.glDisable(cap)
    prog = shaders.compileProgram(shaders.compileShader(SNAP_VS, GL.GL_VERTEX_SHADER),
                                  shaders.compileShader(SNAP_FS, GL.GL_FRAGMENT_SHADER))
    ok = total = 0
    worst = 0.0
    for _ in range(100):
        wv = f32(np.exp(rng.uniform(np.log(0.05), np.log(60))))
        xw, yw = rng.uniform(300, 700), rng.uniform(150, 350)
        xc, yc, zc = f32((xw / sw * 2 - 1) * wv), f32((yw / sh * 2 - 1) * wv), f32(wv * 0.5)
        base = rng.uniform(0, 2 * np.pi)
        far = []
        for a in (base, base + 0.9, base + 2.2, base + 3.1):
            r = rng.uniform(120, 160)
            far.append((np.round((xw + r * np.cos(a)) * 256) / 256, np.round((yw + r * np.sin(a)) * 256) / 256))
        lines = []
        for pa, pb in ((far[0], far[1]), (far[2], far[3])):
            GL.glClearColor(-1, 0, 0, 0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            GL.glUseProgram(prog)
            GL.glBegin(GL.GL_TRIANGLES)
            GL.glTexCoord1f(1.0)
            GL.glVertex4f(float(xc), float(yc), float(zc), float(wv))
            for px, py in (pa, pb):
                GL.glTexCoord1f(0.0)
                GL.glVertex4f(float(f32(px / sw * 2 - 1)), float(f32(py / sh * 2 - 1)), 0.5, 1.0)
            GL.glEnd()
            GL.glUseProgram(0)
            img = np.frombuffer(GL.glReadPixels(0, 0, sw, sh, GL.GL_RED, GL.GL_FLOAT), np.float32).reshape(sh, sw)
            ys, xs = np.nonzero(img >= 0)
            A = np.c_[xs + 0.5, ys + 0.5, np.ones(xs.size)]
            coef, *_ = np.linalg.lstsq(A, img[ys, xs].astype(np.float64), rcond=None)
            lines.append(coef)
        (a1, b1, c1), (a2, b2, c2) = lines
        vx, vy = np.linalg.solve(np.array([[a1, b1], [a2, b2]]), np.array([1 - c1, 1 - c2]))
        worst = max(worst, abs(vx * 256 - round(vx * 256)), abs(vy * 256 - round(vy * 256)))
        inv = rcp_gpu(wv, deltas)
        for c, size, got in ((xc, sw, vx), (yc, sh, vy)):
            win = f32(f32(f32(f32(c * inv) + f32(1)) * f32(0.5)) * f32(size))
            ok += int(int(np.round(float(win) * 256)) == round(got * 256))
            total += 1
    GL.glClipControl(GL.GL_LOWER_LEFT, GL.GL_NEGATIVE_ONE_TO_ONE)
    return report("snap", ok, total, True,
                  f"window x and y of vertices at w 0.05 to 60 (located within {worst:.4f} of a 1/256 step)")


def probe_ties(ctx: Context) -> bool:
    """Which triangle owns a pixel centre lying exactly on the edge two triangles share, against
    GlContext.LeftOrBottom: horizontal, vertical and sloped edges through rows of pixel centres, both windings.

    Args:
        ctx: the context (single-sampled, so the sample is the pixel centre; window coordinates are exact through its
            orthographic projection).
    Returns:
        Whether every centre on a shared edge went to the triangle the rule names.
    """
    rng = np.random.default_rng(8)
    ok = total = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (2, 1), (1, 2), (-2, 1),
                  (1, -2), (3, -1), (-1, -3)]
    for dx, dy in directions:
        for ccw in (True, False):
            steps = 40 // max(abs(dx), abs(dy))
            ax, ay = 200.5 + rng.integers(-20, 20), 200.5 + rng.integers(-20, 20)
            bx, by = ax + steps * dx, ay + steps * dy
            # the two third vertices, one on each side of the edge from A to B, 30 pixels away along its normal
            nl = math.hypot(dx, dy)
            left = (ax + (bx - ax) / 2 - 30 * dy / nl, ay + (by - ay) / 2 + 30 * dx / nl)
            right = (ax + (bx - ax) / 2 + 30 * dy / nl, ay + (by - ay) / 2 - 30 * dx / nl)
            for third in (left, right):
                tri = [(ax, ay), (bx, by), third]
                area = (bx - ax) * (third[1] - ay) - (third[0] - ax) * (by - ay)
                if (area > 0) != ccw:
                    tri = [tri[0], tri[2], tri[1]]
                ctx.reset()
                GL.glBegin(GL.GL_TRIANGLES)
                for x, y in tri:
                    GL.glVertex2f(float(x), float(y))
                GL.glEnd()
                img = ctx.read()
                # the shared edge oriented as the counter-clockwise triangle walks it
                ex0, ey0, ex1, ey1 = (ax, ay, bx, by) if area > 0 else (bx, by, ax, ay)
                owns = ey1 < ey0 or (ey1 == ey0 and ex1 > ex0)
                for j in range(1, steps):
                    px, py = int(ax + j * dx), int(ay + j * dy)
                    ok += int((img[py, px, 0] > 0) == owns)
                    total += 1
    ctx.reset()
    return report("ties", ok, total, True, "pixel centres on shared edges, 14 directions, both windings")



def probe_split(ctx: Context) -> bool:
    """A quad with one bright corner: which diagonal GL_QUADS splits on.

    Args:
        ctx: the context.
    Returns:
        Whether every pixel is the (0, 1, 2) + (0, 2, 3) split's.
    """
    ctx.reset()
    verts = [(10, 10, 0.0), (74, 10, 0.0), (74, 74, 1.0), (10, 74, 0.0)]
    GL.glBegin(GL.GL_QUADS)
    for x, y, c in verts:
        GL.glColor4f(c, 0, 0, 1)
        GL.glVertex2f(x, y)
    GL.glEnd()
    img = ctx.read()
    ok = n = 0
    for py in range(12, 72):
        for px in range(12, 72):
            x, y = px + 0.5, py + 0.5
            # (0, 2, 3) holds y > x; there the red rises along the diagonal from (10, 10)
            lower = y <= x
            a, b, c = (verts[0], verts[1], verts[2]) if lower else (verts[0], verts[2], verts[3])
            det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
            l1 = ((x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (y - a[1])) / det
            l2 = ((b[0] - a[0]) * (y - a[1]) - (x - a[0]) * (b[1] - a[1])) / det
            ok += int(img[py, px, 0] == store12((1 - l1 - l2) * a[2] + l1 * b[2] + l2 * c[2]))
            n += 1
    return report("split", ok, n, True, "GL_QUADS as (0, 1, 2) and (0, 2, 3)")


def probe_lighting(ctx: Context) -> bool:
    """Per-vertex lighting (local viewer, colour material, one directional light) against OpenGL's equation.

    Args:
        ctx: the context.
    Returns:
        Whether the equation held for (nearly) every vertex: the store's 12-bit truncation makes a few flips expected.
    """
    eps = 1e-4
    rng = np.random.default_rng(1)
    normals = []
    for _ in range(4000):
        v = rng.normal(size=3)
        v[2] = abs(v[2])
        normals.append(v / np.linalg.norm(v))
    cols = rng.uniform(0, 1, len(normals))
    ldir = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])
    amb, dif, spe, mspec, shin = 0.1, 0.4, 0.5, 0.5, 64.0
    ctx.reset()
    GL.glMatrixMode(GL.GL_PROJECTION)
    GL.glLoadIdentity()
    GL.glOrtho(0, W * eps, 0, H * eps, 0.5, 2)
    GL.glMatrixMode(GL.GL_MODELVIEW)
    GL.glEnable(GL.GL_LIGHTING)
    GL.glEnable(GL.GL_NORMALIZE)
    GL.glEnable(GL.GL_COLOR_MATERIAL)
    GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
    GL.glLightModelfv(GL.GL_LIGHT_MODEL_AMBIENT, [0, 0, 0, 1])
    GL.glLightModeli(GL.GL_LIGHT_MODEL_LOCAL_VIEWER, 1)
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_EMISSION, [0, 0, 0, 1])
    GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_SPECULAR, [mspec, mspec, mspec, 1])
    GL.glMaterialf(GL.GL_FRONT_AND_BACK, GL.GL_SHININESS, shin)
    GL.glEnable(GL.GL_LIGHT0)
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, [amb, amb, amb, 1])
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, [dif, dif, dif, 1])
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, [spe, spe, spe, 1])
    GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, list(ldir) + [0])
    for i, (nrm, col) in enumerate(zip(normals, cols)):
        qx, qy = 2 * (i % 512), 2 * (i // 512)
        GL.glColor4f(col, col, col, 1)
        GL.glNormal3f(*nrm)
        GL.glBegin(GL.GL_QUADS)
        for x, y in ((qx, qy), (qx + 2, qy), (qx + 2, qy + 2), (qx, qy + 2)):
            GL.glVertex3f(x * eps, y * eps, -1)
        GL.glEnd()
    GL.glDisable(GL.GL_LIGHTING)
    img = ctx.read()
    ok = 0
    for i, (nrm, col) in enumerate(zip(normals, cols)):
        qx, qy = 2 * (i % 512), 2 * (i // 512)
        p = np.array([(qx + 1) * eps, (qy + 1) * eps, -1.0])
        v = -p / np.linalg.norm(p)
        ndl = max(float(np.dot(nrm, ldir)), 0.0)
        spec = 0.0
        if ndl > 0:
            hv = (ldir + v) / np.linalg.norm(ldir + v)
            spec = max(float(np.dot(nrm, hv)), 0.0) ** shin * spe * mspec
        ok += int(img[qy, qx, 0] == store12(min(1.0, amb * col + ndl * dif * col + spec)))
    return report("lighting", ok, len(normals), False, "OpenGL's equation in float64; a few flip at the 12-bit store")


def probe_bilinear(ctx: Context) -> bool:
    """A two-texel ramp (0 | 255) magnified 1024 times: the weight's rounding.

    Args:
        ctx: the context.
    Returns:
        Whether every pixel is the 1/256-rounded weight's.
    """
    ctx.reset()
    GL.glEnable(GL.GL_TEXTURE_2D)
    ramp = np.array([[[0] * 3, [255] * 3], [[0] * 3, [255] * 3]], np.uint8)
    texture([ramp, np.full((1, 1, 3), 128, np.uint8)])
    quad(0, 0, W, 2, [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)])
    row = ctx.read()[0, :, 0]
    ok = sum(int(row[x] == store12(math.floor((x + 0.5) / W * 256 + 0.5) / 256)) for x in range(W))
    return report("bilinear", ok, W, True, "weights rounded to 1/256")


def gym_net_table() -> list:
    """The table Gym.NET embeds (GlTexture.Log2Steps), read from its source.

    Returns:
        256 integers.
    Raises:
        ValueError: when the source no longer holds a 256-entry Log2Steps.
    """
    path = os.path.join(REPO, "src", "Gym.Environments", "Envs", "MuJoCo", "Rendering", "GlTexture.cs")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"Log2Steps = \[(.*?)\];", text, re.S)
    if not m:
        raise ValueError("GlTexture.cs has no Log2Steps table")
    values = [int(v) for v in re.findall(r"\d+", m.group(1))]
    if len(values) != 256:
        raise ValueError(f"Log2Steps has {len(values)} entries")
    return values


def probe_lod_table(ctx: Context, out) -> tuple:
    """One octave of isotropic derivative lengths in 131,072 steps: the 256 thresholds of the blend index.

    Args:
        ctx: the context.
        out: a directory for log2_table.json, or None.
    Returns:
        (whether the measured table equals Gym.NET's and every threshold sits on a 1/256 mantissa, the table).
    """
    n = 131072
    tables = []
    for octave in (1, 4):
        ctx.reset()
        level_texture()
        mant = 1 + np.arange(n) / n
        GL.glBegin(GL.GL_QUADS)
        for i in range(n):
            k = float(mant[i]) * 2.0 ** octave / 256
            qx, qy = 2 * (i % 512), 2 * (i // 512)
            for (x, y), (s, t) in zip(((qx, qy), (qx + 2, qy), (qx + 2, qy + 2), (qx, qy + 2)),
                                      ((0, 0), (2 * k, 0), (2 * k, 2 * k), (0, 2 * k))):
                GL.glTexCoord2f(s, t)
                GL.glVertex2f(x, y)
        GL.glEnd()
        img = ctx.read()
        GL.glDisable(GL.GL_TEXTURE_2D)
        lookup = {}
        idx = np.zeros(n, int)
        for i in range(n):
            qx, qy = 2 * (i % 512), 2 * (i // 512)
            key = (int(img[qy, qx, 0]), int(img[qy, qx, 2]))
            if key not in lookup:
                lookup[key] = blend_index(key[0], key[1], octave % 2 == 1)
            idx[i] = -1 if lookup[key] is None else lookup[key]
        thr = [float(mant[np.nonzero(idx >= k)[0][0]]) for k in range(1, 256)]
        on_grid = all(abs(t * 256 - round(t * 256)) < 1e-9 for t in thr)
        table = [int(np.searchsorted(thr, 1 + j / 256, side="right")) for j in range(256)]
        tables.append((table, on_grid))
    table, on_grid = tables[0]
    same_octaves = tables[0][0] == tables[1][0]
    ours = gym_net_table()
    ok = sum(int(a == b) for a, b in zip(table, ours))
    if out:
        with open(os.path.join(out, "log2_table.json"), "w", encoding="utf-8") as f:
            json.dump(table, f)
    note = f"thresholds on 1/256 mantissas: {on_grid}; the same in two octaves: {same_octaves}"
    return report("lod_table", ok, 256, True, note) and on_grid and same_octaves, table


def gpu_lod(rho: float, table: list) -> int:
    """The level of detail in 1/256 steps from a length (GlTexture.LodSteps).

    Args:
        rho: the length in texels per pixel.
        table: the 256-entry table.
    Returns:
        The level in 1/256 steps.
    """
    m, e = math.frexp(rho)
    m, e = 2 * m, e - 1
    return 256 * e + table[int((m - 1) * 256)]


def gpu_length(du: float, dv: float) -> float:
    """A derivative's length as GlContext.Length takes it.

    Args:
        du: the u component in texels.
        dv: the v component in texels.
    Returns:
        The length.
    """
    hi, lo = max(abs(du), abs(dv)), min(abs(du), abs(dv))
    if hi == 0:
        return 0.0
    scale = 2.0 ** (9 - math.frexp(hi)[1] + 1)
    return (math.floor(hi * scale) + math.floor(0.3442 * math.floor(lo * scale))) / scale


def probe_lod_norm(ctx: Context, table: list) -> bool:
    """Derivatives rotated 0 to 90 degrees (each screen axis in turn): the length's approximation.

    Args:
        ctx: the context.
        table: the measured log2 table.
    Returns:
        True (a fitted model reports its score without failing).
    """
    ok = total = 0
    for which in ("x", "y"):
        ctx.reset()
        level_texture()
        thetas = np.arange(0, 90.001, 0.25)
        k = 2.0 ** 1.5
        for n, deg in enumerate(thetas):
            th = math.radians(deg)
            big, small = (k * math.cos(th) / 256, k * math.sin(th) / 256), (0.0, k * 0.125 / 256)
            dx, dy = (big, small) if which == "x" else ((small[1], 0.0), big)
            qx, qy = 2 * (n % 512), 2 * (n // 512)
            quad(qx, qy, qx + 2, qy + 2, [(0, 0), (2 * dx[0], 2 * dx[1]), (2 * dx[0] + 2 * dy[0], 2 * dx[1] + 2 * dy[1]),
                                          (2 * dy[0], 2 * dy[1])])
        img = ctx.read()
        GL.glDisable(GL.GL_TEXTURE_2D)
        for n, deg in enumerate(thetas):
            th = math.radians(deg)
            qx, qy = 2 * (n % 512), 2 * (n // 512)
            lx = gpu_length(k * math.cos(th), k * math.sin(th))
            ly = gpu_length(0.0, k * 0.125)
            lod = gpu_lod(max(lx, ly), table)
            odd = (lod >> 8) % 2 == 1
            got = blend_index(int(img[qy, qx, 0]), int(img[qy, qx, 2]), odd)
            ok += int(got == (lod & 255))
            total += 1
    return report("lod_norm", ok, total, False, "max + 0.3442 min at 9 bits: a fit, not the GPU's exact rule")


def probe_lod_quad(ctx: Context, table: list) -> bool:
    """A projective quad: one level per 2 x 2 quad, from the differences at its lower-left pixel.

    Args:
        ctx: the context.
        table: the measured log2 table.
    Returns:
        Whether every quad shows one level.
    """
    ctx.reset()
    level_texture()
    n_, sc = 64, 0.5
    verts = [(0, 0, 0, 0, 1), (n_, 0, sc, 0, 1), (n_, n_, sc * 4, sc * 4, 4), (0, n_, 0, sc * 4, 4)]
    GL.glBegin(GL.GL_TRIANGLES)
    for i in (0, 1, 2, 0, 2, 3):
        x, y, s, t, q = verts[i]
        GL.glTexCoord4f(s, t, 0, q)
        GL.glVertex2f(x, y)
    GL.glEnd()
    img = ctx.read()
    GL.glDisable(GL.GL_TEXTURE_2D)
    one = total = ok = n = 0
    for qy in range(0, n_, 2):
        for qx in range(0, n_, 2):
            if qy + 1 >= qx and qy <= qx + 1:
                continue  # quads on the diagonal mix the two triangles
            block = img[qy:qy + 2, qx:qx + 2, :3].reshape(4, 3)
            total += 1
            one += int((block == block[0]).all())
            upper = qy > qx
            a, b, c = (verts[0], verts[2], verts[3]) if upper else (verts[0], verts[1], verts[2])

            def uv(px, py):
                det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
                l1 = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / det
                l2 = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / det
                l0 = 1 - l1 - l2
                s, t, q = (l0 * a[k] + l1 * b[k] + l2 * c[k] for k in (2, 3, 4))
                return s / q * 256, t / q * 256

            u0, v0 = uv(qx + 0.5, qy + 0.5)
            u1, v1 = uv(qx + 1.5, qy + 0.5)
            u2, v2 = uv(qx + 0.5, qy + 1.5)
            rho = max(gpu_length(u1 - u0, v1 - v0), gpu_length(u2 - u0, v2 - v0))
            if rho <= 1:
                continue  # magnified quads show level 0 whatever the derivative
            lod = gpu_lod(rho, table)
            got = blend_index(int(img[qy, qx, 0]), int(img[qy, qx, 2]), (lod >> 8) % 2 == 1)
            ok += int(got is not None and abs(got - (lod & 255)) <= 1)
            n += 1
    good = report("lod_quad", one, total, True, "quads showing one level")
    report("lod_origin", ok, n, False, "the lower-left pixel's differences, within one 1/256 step")
    return good


def area_chain(level0: np.ndarray, sizes: list) -> list:
    """Gym.NET's mipmap chain (GlTexture.Downsample): area-weighted box, each level stored with the 12-bit rule.

    Args:
        level0: H x W grey levels.
        sizes: (h, w) of each smaller level.
    Returns:
        The levels after level 0.
    """
    out = []
    lv = level0.astype(np.float64)
    for h2, w2 in sizes:
        def weights(n_src, n_dst):
            m = np.zeros((n_dst, n_src))
            s = n_src / n_dst
            for i in range(n_dst):
                a, b = i * s, (i + 1) * s
                for k in range(int(math.floor(a)), min(n_src, int(math.ceil(b)))):
                    m[i, k] = min(b, k + 1) - max(a, k)
                m[i] /= m[i].sum()
            return m
        exact = weights(lv.shape[0], h2) @ lv @ weights(lv.shape[1], w2).T
        lv = np.vectorize(lambda v: store12(float(np.float32(v / 255.0))))(exact).astype(np.float64)
        out.append(lv)
    return out


def probe_mipmaps() -> bool:
    """glGenerateMipmap's chain of Humanoid-v5's floor checker, read back, against Gym.NET's reduction.

    Returns:
        True (the GPU's 25 -> 12 filter is not the area box; the shortfall is reported).
    """
    sys.path.insert(0, os.path.join(REPO, "refs", "Gymnasium"))
    import gymnasium as gym  # noqa: E402
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    env.reset(seed=0)
    env.render()
    viewer = env.unwrapped.mujoco_renderer._get_viewer("rgb_array")
    viewer.opengl_context.make_current()
    m = env.unwrapped.model
    tex = 2
    GL.glBindTexture(GL.GL_TEXTURE_2D, int(viewer.con.texture[tex]))
    levels = []
    while True:
        k = len(levels)
        w = int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, k, GL.GL_TEXTURE_WIDTH))
        h = int(GL.glGetTexLevelParameteriv(GL.GL_TEXTURE_2D, k, GL.GL_TEXTURE_HEIGHT))
        if w == 0 or h == 0:
            break
        data = GL.glGetTexImage(GL.GL_TEXTURE_2D, k, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        levels.append(np.frombuffer(data, np.uint8).reshape(h, w, 3)[..., 0].astype(np.float64))
    adr, tw, th = int(m.tex_adr[tex]), int(m.tex_width[tex]), int(m.tex_height[tex])
    base = m.tex_data[adr: adr + 3 * tw * th].reshape(th, tw, 3)[..., 0]
    ours = area_chain(levels[0], [lv.shape for lv in levels[1:]])
    differ = [int((a != b).sum()) for a, b in zip(ours, levels[1:])]
    env.close()
    ok = int(np.array_equal(base, levels[0]))
    report("mip_base", ok, 1, True, "level 0 is the model's texture")
    total = sum(lv.size for lv in levels[1:])
    report("mipmaps", total - sum(differ), total, False, f"texels differing per level from 1: {differ} (known: 0, 0, 10, 4, 2, 0)")
    return ok == 1


def main() -> int:
    """Runs every probe and prints the report.

    Returns:
        0 when every exact model held, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="a directory for log2_table.json")
    parser.add_argument("--rcp-table", default=None, help="writes the viewport reciprocal's table (Assets/mujoco/gpu/rcp.bin)")
    args = parser.parse_args()
    ctx = Context()
    print(f"GPU: {GL.glGetString(GL.GL_RENDERER).decode()} | OpenGL {GL.glGetString(GL.GL_VERSION).decode()}")
    print(f"MuJoCo {mujoco.__version__}")
    good = probe_store(ctx)
    good &= probe_blend(ctx)
    good &= probe_multisample(ctx)
    good &= probe_resolve_sums(ctx)
    good &= probe_split(ctx)
    good &= probe_lighting(ctx)
    good &= probe_bilinear(ctx)
    table_ok, table = probe_lod_table(ctx, args.out)
    good &= table_ok
    good &= probe_lod_norm(ctx, table)
    good &= probe_lod_quad(ctx, table)
    good &= probe_matrix_stack()
    good &= probe_vertex()
    good &= probe_rcp(args.rcp_table)
    good &= probe_snap()
    good &= probe_ties(ctx)
    # last: closing its Gymnasium env terminates GLFW, and every context with it
    good &= probe_mipmaps()
    print("every exact model held" if good else "some exact model missed: Gym.NET's frames will drift on this GPU")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
