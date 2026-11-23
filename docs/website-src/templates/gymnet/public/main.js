// Gym.NET DocFX template customization (https://dotnet.github.io/docfx/docs/template.html).
//
// The modern template imports this module's default export through its options() hook. It adds three behaviours:
//   1. pixel text: every <canvas class="gn-pixel" data-text="..."> is drawn in the showreel's 5x7 font;
//   2. the 8-bit "select your environment" gallery (on the landing page and the Environments page, one shared include,
//      website-src/includes/arcade.md), filled from images/envs/envs.json, which the
//      showreel tool (tools/showreel) writes together with the looping clips (the cartridges' 8-bit WebPs and the
//      "now playing" screen's full-size MP4s);
//   3. links docfx could not resolve (repository files outside the site, such as CLAUDE.md or LICENSE) are pointed at
//      GitHub, so pages copied from docs/ keep working links on the site.

/**
 * The classic 5x7 dot-matrix font for ASCII 32-95, five column bytes per glyph, bit 0 the top row. The same table as
 * tools/showreel/PixelFont.cs, so the page letters exactly like the video.
 * @type {number[]}
 */
const GLYPHS = [
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5F, 0x00, 0x00, 0x00, 0x07, 0x00, 0x07, 0x00, 0x14, 0x7F, 0x14, 0x7F, 0x14,
  0x24, 0x2A, 0x7F, 0x2A, 0x12, 0x23, 0x13, 0x08, 0x64, 0x62, 0x36, 0x49, 0x55, 0x22, 0x50, 0x00, 0x05, 0x03, 0x00, 0x00,
  0x00, 0x1C, 0x22, 0x41, 0x00, 0x00, 0x41, 0x22, 0x1C, 0x00, 0x2A, 0x1C, 0x7F, 0x1C, 0x2A, 0x08, 0x08, 0x3E, 0x08, 0x08,
  0x00, 0x50, 0x30, 0x00, 0x00, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x60, 0x60, 0x00, 0x00, 0x20, 0x10, 0x08, 0x04, 0x02,
  0x3E, 0x51, 0x49, 0x45, 0x3E, 0x00, 0x42, 0x7F, 0x40, 0x00, 0x42, 0x61, 0x51, 0x49, 0x46, 0x21, 0x41, 0x45, 0x4B, 0x31,
  0x18, 0x14, 0x12, 0x7F, 0x10, 0x27, 0x45, 0x45, 0x45, 0x39, 0x3C, 0x4A, 0x49, 0x49, 0x30, 0x01, 0x71, 0x09, 0x05, 0x03,
  0x36, 0x49, 0x49, 0x49, 0x36, 0x06, 0x49, 0x49, 0x29, 0x1E, 0x00, 0x36, 0x36, 0x00, 0x00, 0x00, 0x56, 0x36, 0x00, 0x00,
  0x08, 0x14, 0x22, 0x41, 0x00, 0x14, 0x14, 0x14, 0x14, 0x14, 0x00, 0x41, 0x22, 0x14, 0x08, 0x02, 0x01, 0x51, 0x09, 0x06,
  0x32, 0x49, 0x79, 0x41, 0x3E, 0x7E, 0x11, 0x11, 0x11, 0x7E, 0x7F, 0x49, 0x49, 0x49, 0x36, 0x3E, 0x41, 0x41, 0x41, 0x22,
  0x7F, 0x41, 0x41, 0x22, 0x1C, 0x7F, 0x49, 0x49, 0x49, 0x41, 0x7F, 0x09, 0x09, 0x09, 0x01, 0x3E, 0x41, 0x49, 0x49, 0x7A,
  0x7F, 0x08, 0x08, 0x08, 0x7F, 0x00, 0x41, 0x7F, 0x41, 0x00, 0x20, 0x40, 0x41, 0x3F, 0x01, 0x7F, 0x08, 0x14, 0x22, 0x41,
  0x7F, 0x40, 0x40, 0x40, 0x40, 0x7F, 0x02, 0x0C, 0x02, 0x7F, 0x7F, 0x04, 0x08, 0x10, 0x7F, 0x3E, 0x41, 0x41, 0x41, 0x3E,
  0x7F, 0x09, 0x09, 0x09, 0x06, 0x3E, 0x41, 0x51, 0x21, 0x5E, 0x7F, 0x09, 0x19, 0x29, 0x46, 0x46, 0x49, 0x49, 0x49, 0x31,
  0x01, 0x01, 0x7F, 0x01, 0x01, 0x3F, 0x40, 0x40, 0x40, 0x3F, 0x1F, 0x20, 0x40, 0x20, 0x1F, 0x3F, 0x40, 0x38, 0x40, 0x3F,
  0x63, 0x14, 0x08, 0x14, 0x63, 0x07, 0x08, 0x70, 0x08, 0x07, 0x61, 0x51, 0x49, 0x45, 0x43, 0x00, 0x7F, 0x41, 0x41, 0x00,
  0x02, 0x04, 0x08, 0x10, 0x20, 0x00, 0x41, 0x41, 0x7F, 0x00, 0x04, 0x02, 0x01, 0x02, 0x04, 0x40, 0x40, 0x40, 0x40, 0x40,
]

/** Character cell width in font pixels: five columns and one of spacing. @type {number} */
const CELL = 6

/** Glyph height in font pixels. @type {number} */
const ROWS = 7

/**
 * The five column bytes of a character; lower case draws as upper case, anything outside ASCII 32-95 as a space.
 * @param {string} ch One character.
 * @returns {number[]} The glyph's columns.
 */
function glyph(ch) {
  const c = ch.toUpperCase().charCodeAt(0)
  const i = c >= 32 && c <= 95 ? c - 32 : 0
  return GLYPHS.slice(i * 5, i * 5 + 5)
}

/**
 * Draws pixel text onto a canvas, resizing the canvas to the text (one canvas pixel per screen pixel at the given
 * scale), with an optional drop shadow one font pixel down and right. Lines split on "\n".
 * @param {HTMLCanvasElement} canvas The canvas; its previous content is cleared.
 * @param {string} text The text.
 * @param {number} scale Canvas pixels per font pixel (a whole number keeps the edges crisp).
 * @param {string} color A CSS colour for the letters.
 * @param {string} [shadow] A CSS colour for the shadow, or nothing for none.
 * @returns {void}
 */
function paintPixelText(canvas, text, scale, color, shadow) {
  const lines = text.split('\n')
  const longest = Math.max(1, ...lines.map((l) => l.length))
  const pad = shadow ? scale : 0
  const lineHeight = (ROWS + 3) * scale
  canvas.width = Math.max(1, (longest * CELL - 1) * scale + pad)
  canvas.height = Math.max(1, lines.length * lineHeight - 3 * scale + pad)
  const g = canvas.getContext('2d')
  g.clearRect(0, 0, canvas.width, canvas.height)
  /**
   * Fills every lit font pixel of every line once, offset by (dx, dy): called for the shadow, then for the letters.
   * @param {number} dx Horizontal offset in canvas pixels.
   * @param {number} dy Vertical offset in canvas pixels.
   * @param {string} fill The CSS colour.
   * @returns {void}
   */
  const run = (dx, dy, fill) => {
    g.fillStyle = fill
    lines.forEach((line, n) => {
      for (let i = 0; i < line.length; i++) {
        const cols = glyph(line[i])
        for (let col = 0; col < 5; col++) {
          for (let row = 0; row < ROWS; row++) {
            if (cols[col] & (1 << row)) {
              g.fillRect(dx + (i * CELL + col) * scale, dy + n * lineHeight + row * scale, scale, scale)
            }
          }
        }
      }
    })
  }
  if (shadow) {
    run(scale, scale, shadow)
  }
  run(0, 0, color)
  canvas.setAttribute('role', 'img')
  canvas.setAttribute('aria-label', text.replace(/\n/g, ' '))
}

/**
 * The text of a cartridge's label for the room it has: the id on one line when it fits at one canvas pixel per font
 * pixel, else two lines split at the capital nearest the id's middle (InvertedDouble / Pendulum-v5), as a printed
 * label wraps. A label never shrinks to fit instead: a canvas drawn narrower than its own pixels loses whole columns of
 * its 5x7 letters. Only the longest ids wrap, and only on narrow shelves (the Environments page's two columns).
 * @param {string} id The env's id.
 * @param {number} room The label's width in CSS pixels (the cartridge's content box).
 * @returns {string} The text to paint, with "\n" where it wraps; the id itself when it fits or has no capital to
 *   split at.
 */
function cartLabel(id, room) {
  if (id.length * CELL - 1 <= room) {
    return id
  }
  let split = -1
  for (let i = 1; i < id.length; i++) {
    // A capital after a lower-case letter or digit starts a word of the id; the one nearest the middle balances the lines.
    if (/[A-Z]/.test(id[i]) && /[a-z0-9]/.test(id[i - 1]) && (split < 0 || Math.abs(i - id.length / 2) < Math.abs(split - id.length / 2))) {
      split = i
    }
  }
  return split > 0 ? `${id.slice(0, split)}\n${id.slice(split)}` : id
}

/**
 * Paints every static pixel-text canvas on the page from its data attributes (data-text, data-scale, data-color,
 * data-shadow). Canvases without data-text are drawn by the gallery instead.
 * @returns {void}
 */
function paintStaticPixelText() {
  for (const canvas of document.querySelectorAll('canvas.gn-pixel[data-text]')) {
    paintPixelText(canvas, canvas.dataset.text, Number(canvas.dataset.scale || 2), canvas.dataset.color || '#fff', canvas.dataset.shadow)
  }
}

/**
 * The site's root URL, from the modern template's docfx:rel meta (the relative path from this page to the root).
 * @returns {URL} The root, ending in a slash.
 */
function siteRoot() {
  const rel = document.querySelector('meta[name="docfx:rel"]')?.getAttribute('content') || './'
  return new URL(rel, window.location.href)
}

/**
 * Builds the 8-bit gallery on the page that holds it (the landing page or the Environments page): one cartridge per
 * environment (its loop scaled up in chunky pixels, its id as the label, wrapped where the shelf is narrow), a
 * "now playing" screen with the chosen env's full-size loop, a HUD with the showreel episode's seed, steps and return,
 * and the make() call. It runs in attract mode (the cursor moves on by itself every few seconds) until the reader
 * hovers, clicks or uses the arrow keys. Does nothing on pages without the gallery, or when the manifest cannot be
 * fetched.
 * @returns {Promise<void>} Settles when the gallery is built (or skipped).
 */
async function installArcade() {
  const arcade = document.querySelector('.gn-arcade')
  if (!arcade) {
    return
  }
  const root = siteRoot()
  let envs
  try {
    const response = await fetch(new URL('images/envs/envs.json', root))
    if (!response.ok) {
      return
    }
    envs = await response.json()
  } catch {
    // Offline or opened from the file system: the noscript list and the video still describe the envs.
    return
  }
  const carts = arcade.querySelector('.gn-carts')
  const label = arcade.querySelector('.gn-now-label')
  const screenVideo = arcade.querySelector('.gn-screen-video')
  const screenImg = arcade.querySelector('.gn-screen-img')
  const hud = arcade.querySelector('.gn-hud')
  const blurb = arcade.querySelector('.gn-blurb')
  const snippet = arcade.querySelector('.gn-snippet code')
  const buttons = envs.map((env, i) => {
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'gn-cart'
    button.setAttribute('role', 'option')
    button.setAttribute('aria-selected', 'false')
    button.setAttribute('aria-label', env.id)
    const art = document.createElement('span')
    art.className = 'gn-cart-art'
    const img = document.createElement('img')
    img.src = new URL(env.loop, root).href
    img.alt = ''
    img.width = env.width
    img.height = env.height
    art.appendChild(img)
    const name = document.createElement('canvas')
    name.className = 'gn-pixel gn-cart-name'
    paintPixelText(name, env.id, 1, '#000')
    // The text painted now; fitLabels compares against it, so a label that fits is not painted twice.
    name.dataset.label = env.id
    button.append(art, name)
    button.addEventListener('click', () => select(i, true))
    button.addEventListener('mouseenter', () => select(i, true))
    carts.appendChild(button)
    return button
  })

  /**
   * Repaints each cartridge's label for the width its cartridge now has (cartLabel): one line where the id fits, two
   * where it would be squeezed. Painting only on a change keeps a resize from redrawing all 24 labels every frame.
   * @returns {void}
   */
  function fitLabels() {
    // Every cartridge has the same width and padding (one grid track each), so the first one's content box is all.
    const style = getComputedStyle(buttons[0])
    const room = buttons[0].clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
    buttons.forEach((button, i) => {
      const name = button.querySelector('.gn-cart-name')
      const text = cartLabel(envs[i].id, room)
      if (name.dataset.label !== text) {
        name.dataset.label = text
        paintPixelText(name, text, 1, '#000')
      }
    })
  }
  fitLabels()
  // The shelf's columns change with the window (and with the docs layout's sidebar), so the labels follow its width.
  new ResizeObserver(fitLabels).observe(carts)
  let current = -1
  let attract = true
  // H.264 in MP4 plays in every current browser except Chromium builds without the proprietary codecs. The codecs string
  // names the screen loops' own profile (High; level 3.0 is the highest any of them needs), so the answer is about these
  // files, not about MP4 in general.
  const playsScreenLoops = screenVideo.canPlayType('video/mp4; codecs="avc1.64001E"') !== ''
  // Set once the browser refuses to start a muted video by itself (iOS in Low Power Mode, some data savers): from then on
  // every env shows its cartridge loop, an image that animates anyway, never a still frame waiting for a tap.
  let autoplayRefused = false
  // Envs whose screen loop failed to load or decode (a missing file, a broken download); each falls back on its own.
  const brokenLoops = new Set()

  /**
   * Puts an env's loop on the "now playing" screen: its full-size H.264 loop, muted, looping and without controls, or,
   * where that cannot play, its cartridge's 8-bit loop stretched in chunky pixels, as the screen showed every env before
   * the full-size loops existed. Only one of the two elements is visible, and a hidden video is emptied so that it stops
   * downloading and decoding.
   * @param {{id: string, loop: string, screen?: string}} env The env's manifest entry; a manifest written before the
   *   screen loops existed has no screen field, and its envs show their cartridge loops.
   * @returns {void}
   */
  function showOnScreen(env) {
    const description = `${env.id}, played by the showreel's policy`
    if (env.screen && playsScreenLoops && !autoplayRefused && !brokenLoops.has(env.id)) {
      screenImg.hidden = true
      screenImg.removeAttribute('src')
      screenVideo.hidden = false
      screenVideo.setAttribute('aria-label', description)
      // The muted attribute only sets the default; autoplay policies read the property, so it is set as well.
      screenVideo.muted = true
      screenVideo.src = new URL(env.screen, root).href
      screenVideo.play().catch((error) => {
        // AbortError: a newer selection replaced this source before it started, as attract mode does every few seconds.
        // NotAllowedError: this browser will not autoplay here at all, so the screen switches to the animated images.
        if (error.name === 'NotAllowedError') {
          autoplayRefused = true
          if (envs[current] === env) {
            showOnScreen(env)
          }
        }
      })
    } else {
      screenVideo.hidden = true
      if (screenVideo.hasAttribute('src')) {
        screenVideo.removeAttribute('src')
        // load() with no source releases the old one: without it the hidden video keeps its download and decoder alive.
        screenVideo.load()
      }
      screenImg.hidden = false
      screenImg.alt = description
      screenImg.src = new URL(env.loop, root).href
    }
  }

  // A screen loop that fails to load or decode falls back to its cartridge loop. The event always belongs to the current
  // source: a new source aborts the old load and drops its queued media events, "error" among them.
  screenVideo.addEventListener('error', () => {
    const env = envs[current]
    if (env && screenVideo.hasAttribute('src')) {
      brokenLoops.add(env.id)
      showOnScreen(env)
    }
  })

  /**
   * Shows an env on the "now playing" screen and moves the cartridge cursor to it.
   * @param {number} i The env's index in the manifest.
   * @param {boolean} byReader Whether the reader chose it (which ends attract mode).
   * @returns {void}
   */
  function select(i, byReader) {
    if (byReader) {
      attract = false
    }
    if (i === current) {
      return
    }
    current = i
    const env = envs[i]
    buttons.forEach((b, n) => b.setAttribute('aria-selected', String(n === i)))
    paintPixelText(label, `NOW PLAYING: ${env.id}`, 2, '#3cbcfc')
    showOnScreen(env)
    const result = env.result ? `  ${env.result}` : ''
    paintPixelText(hud, `SEED ${env.seed}  STEPS ${env.steps}\nRETURN ${env.return}${result}`, 2, '#fcfcfc')
    blurb.textContent = `${env.blurb} Actions: ${env.actionSpace}.`
    // The manifest carries the exact call the loop was rendered with (the locomotion envs name their track camera);
    // the fallback keeps a manifest written before that field from showing nothing.
    snippet.textContent = env.make?.startsWith('using ')
      ? env.make
      : `using IEnv env = Gymnasium.Make("${env.id}", new { render_mode = "rgb_array" });`
  }

  // Arrow keys move the cursor through the grid while focus is inside the gallery; Home and End jump to the ends.
  arcade.addEventListener('keydown', (e) => {
    const columns = Math.max(1, Math.round(carts.clientWidth / (buttons[0].offsetWidth + 12)))
    const moves = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: columns, ArrowUp: -columns }
    let next = current
    if (e.key in moves) {
      next = Math.min(envs.length - 1, Math.max(0, current + moves[e.key]))
    } else if (e.key === 'Home') {
      next = 0
    } else if (e.key === 'End') {
      next = envs.length - 1
    } else {
      return
    }
    e.preventDefault()
    select(next, true)
    buttons[next].focus()
  })

  select(0, false)
  // Attract mode: like an arcade cabinet nobody is playing, the cursor walks the shelf by itself.
  const timer = window.setInterval(() => {
    if (!attract) {
      window.clearInterval(timer)
      return
    }
    select((current + 1) % envs.length, false)
  }, 4500)
}

/**
 * Points links that docfx left unresolved at GitHub. The docs pages are the repository's own docs/ files at the same
 * paths, so a relative link to a repository file the site does not publish (CLAUDE.md, LICENSE, a plan, a source file)
 * resolves to a site path that equals its repository path; that path is rewritten to the file on GitHub.
 * @returns {void}
 */
function rewriteRepoLinks() {
  const root = siteRoot()
  const unresolved = /(\.(md|cs|csproj|props|targets|py|ps1|sh|json|yml)|\/LICENSE)$/i
  for (const a of document.querySelectorAll('article a[href]')) {
    const href = a.getAttribute('href')
    if (!href || /^(https?:|mailto:|#)/i.test(href)) {
      continue
    }
    const target = new URL(href, window.location.href)
    if (!target.href.startsWith(root.href) || !unresolved.test(target.pathname)) {
      continue
    }
    const path = target.pathname.substring(root.pathname.length)
    a.href = `https://github.com/SciSharp/Gym.NET/blob/master/${path}${target.hash}`
  }
}

export default {
  iconLinks: [
    {
      icon: 'github',
      href: 'https://github.com/SciSharp/Gym.NET',
      title: 'GitHub'
    },
    {
      icon: 'box-seam',
      href: 'https://www.nuget.org/packages/Gym.NET.Environments',
      title: 'NuGet'
    }
  ],

  start: () => {
    paintStaticPixelText()
    installArcade()
    rewriteRepoLinks()
  }
}
