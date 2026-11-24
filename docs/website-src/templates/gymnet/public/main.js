// Gym.NET DocFX template customization (https://dotnet.github.io/docfx/docs/template.html).
//
// The modern template imports this module's default export through its options() hook. It adds five behaviours:
//   1. pixel text: every <canvas class="gn-pixel" data-text="..."> is drawn in the showreel's 5x7 font (the gallery's
//      lettering and the env pages' banners);
//   2. the 8-bit "select your environment" gallery (on the landing page and the Environments page, one shared include,
//      website-src/includes/arcade.md), filled from images/envs/envs.json, which the showreel tool (tools/showreel)
//      writes together with the looping clips (the cartridges' 8-bit WebPs and the "now playing" screen's full-size
//      MP4s), with each cartridge's game manual (DR-102); a hover previews a cartridge, a click locks it (DR-143);
//   3. links docfx could not resolve (repository files outside the site, such as CLAUDE.md or LICENSE) are pointed at
//      GitHub, so pages copied from docs/ keep working links on the site;
//   4. the env pages' episode anatomy (DR-115): a chart of the recorded episode, drawn from its step trace, with a
//      cursor and a readout that follow the playing video;
//   5. the sidebar remembers which sections the reader opened or closed and restores them on the next page, for 48
//      hours that slide forward with every visit (DR-138), the same code as NumSharp's site, so the SciSharp sites on
//      scisharp.github.io behave alike; toc.yml's `expanded` sets a section's state until the reader changes it.

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
 * The "NOW PLAYING" label's text for the room the panel has: on one line where it fits at the label's two canvas pixels
 * per font pixel, else with the id on a second line. Like a cartridge label (cartLabel), it wraps rather than shrink:
 * the two longest ids made the one-line label 454 and 442 px wide in a 440 px panel, and scaled down its letters lost
 * whole columns.
 * @param {string} id The env's id.
 * @param {number} room The panel's content width in CSS pixels.
 * @returns {string} The text to paint, with "\n" where it wraps.
 */
function nowPlayingText(id, room) {
  const line = `NOW PLAYING: ${id}`
  return (line.length * CELL - 1) * 2 <= room ? line : `NOW PLAYING:\n${id}`
}

/**
 * A game-manual line with its numbers held to what they measure: a range ("−1 to 1"), and a count or size and its
 * unit ("1,000 steps", "0.2 rad", "0.7 m"), are joined by no-break spaces, so a row never wraps between them (the
 * panel once broke "each from −1 to" from its "1"). Typography only: the words are the manifest's.
 * @param {string} text A manual line.
 * @returns {string} The line with those spaces made unbreakable.
 */
function keepTogether(text) {
  return text
    .replace(/(\d) (to [−+-]?\d)/g, '$1\u00a0$2')
    .replace(/\bto ([−+-]?\d)/g, 'to\u00a0$1')
    .replace(/(\d) (steps?|rad|m)\b/g, '$1\u00a0$2')
}

/**
 * The family line under "NOW PLAYING", in the gallery's words: Gymnasium's family names as the site writes them.
 * @param {string} family The manifest's family, Gymnasium's folder name (classic_control, box2d, toy_text, mujoco).
 * @returns {string} The line, e.g. "CLASSIC CONTROL"; an unknown family is shown with its underscores as spaces.
 */
function familyName(family) {
  return { classic_control: 'CLASSIC CONTROL', box2d: 'BOX2D', toy_text: 'TOY TEXT', mujoco: 'MUJOCO' }[family] ?? family.replace(/_/g, ' ').toUpperCase()
}

/**
 * The env pages the site has, from the Environments sidebar's table of contents (DocFX's docs/environments/toc.json).
 * READ THE MANUAL shows only for a cartridge whose page is listed there: the pages are written family by family, and
 * a button leading to a page not written yet would be a dead end. The sidebar lists exactly the pages that exist, so it
 * is the one list to trust.
 * @param {URL} root The site's root.
 * @returns {Promise<Set<string>>} Each listed page's absolute URL, without any #section; empty when the table cannot be
 *   read (offline, or a site built before the Environments section), which hides every manual button.
 */
async function writtenPages(root) {
  const pages = new Set()
  try {
    const toc = new URL('docs/environments/toc.json', root)
    const response = await fetch(toc)
    if (!response.ok) {
      return pages
    }
    /**
     * Adds every href of a toc level and the levels under it (a family's entries sit in its items).
     * @param {{href?: string, items?: object[]}[] | undefined} items The level's entries.
     * @returns {void}
     */
    const walk = (items) => {
      for (const item of items ?? []) {
        if (item.href) {
          // Entries are relative to the toc file's folder; a hash would only name a section of the same page.
          pages.add(new URL(item.href, toc).href.split('#')[0])
        }
        walk(item.items)
      }
    }
    walk((await response.json()).items)
  } catch {
    // Unreadable or not JSON: no page is known, so no button leads anywhere.
  }
  return pages
}

/**
 * Builds the 8-bit gallery on the page that holds it (the landing page or the Environments page): one cartridge per
 * environment (its loop scaled up in chunky pixels, its id as the label, wrapped where the shelf is narrow), and a
 * "now playing" panel with the chosen env's family, its full-size loop, a HUD with the showreel episode's seed, steps
 * and return, its game manual (DR-102: HOW TO PLAY in game words, READ THE MANUAL opening the env's page, links to its
 * example, Gymnasium's page and the API). It runs in attract mode (the cursor moves on by itself every few seconds)
 * until the reader hovers, clicks or uses the arrow keys; Enter opens the selected env's page. Hovering a cartridge
 * shows it; a click locks it (DR-143): while locked, hovering the others leaves NOW PLAYING alone, so a reader can pick a
 * cartridge mid-shelf and cross the rest to the panel or the sidebar. A click on the locked cartridge, or Esc, releases
 * it; a click on another moves the lock; the arrow keys, Home and End carry it along. Does nothing on pages without the
 * gallery, or when the manifest cannot be fetched.
 * @returns {Promise<void>} Settles when the gallery is built (or skipped).
 */
async function installArcade() {
  const arcade = document.querySelector('.gn-arcade')
  if (!arcade) {
    return
  }
  const root = siteRoot()
  // The page list is fetched beside the manifest, so it costs no extra wait; it never fails (an empty set at worst).
  const pagesLoaded = writtenPages(root)
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
  const pages = await pagesLoaded
  const carts = arcade.querySelector('.gn-carts')
  const now = arcade.querySelector('.gn-now')
  const label = arcade.querySelector('.gn-now-label')
  const family = arcade.querySelector('.gn-family')
  const screenVideo = arcade.querySelector('.gn-screen-video')
  const screenImg = arcade.querySelector('.gn-screen-img')
  const hud = arcade.querySelector('.gn-hud')
  const manualRows = arcade.querySelectorAll('.gn-manual-rows dd[data-manual]')
  const manualButton = arcade.querySelector('.gn-manual-button')
  // Quiet until the reader picks a cartridge (select turns it back to polite): attract mode is not news.
  now.setAttribute('aria-live', 'off')
  const linksLine = arcade.querySelector('.gn-links')
  const links = linksLine.querySelectorAll('a[data-link]')
  const enterHint = arcade.querySelector('.gn-enter')
  // The lock's hint at the end of the links line (DR-143); an include older than the lock has none.
  const lockHint = arcade.querySelector('.gn-lock-hint')

  /**
   * The address READ THE MANUAL opens for an env: its page (with the variant's section) when the site has that page,
   * else nothing.
   * @param {{docs?: string}} env The env's manifest entry; a manifest written before the manual existed has no docs.
   * @returns {string | null} The absolute URL, or null when there is no page to open.
   */
  function manualHref(env) {
    if (!env.docs) {
      return null
    }
    const page = new URL(env.docs, root)
    return pages.has(page.href.split('#')[0]) ? page.href : null
  }

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
    button.addEventListener('click', () => toggleLock(i))
    // A hover only previews: while a cartridge is locked, the pointer crossing the others changes nothing (DR-143).
    // On a touch screen a tap fires this before its click, so a tap on an unlocked shelf previews, then locks.
    button.addEventListener('mouseenter', () => {
      if (locked < 0) {
        select(i, true)
      }
    })
    carts.appendChild(button)
    return button
  })

  let current = -1

  /**
   * The cartridge the reader locked with a click (DR-143), or -1 while none is: a locked cartridge stays on NOW PLAYING
   * whatever the pointer crosses, until a click on it or Esc releases it.
   * @type {number}
   */
  let locked = -1

  /**
   * A click on a cartridge: it locks that env on the screen (showing it first, which ends attract mode as any reader
   * choice does), moves the lock there from another cartridge, or, on the locked cartridge itself, releases the lock.
   * @param {number} i The clicked cartridge's index in the manifest.
   * @returns {void}
   */
  function toggleLock(i) {
    if (locked === i) {
      setLock(-1)
      return
    }
    select(i, true)
    setLock(i)
  }

  /**
   * Sets the lock and its marks: the locked cartridge's settled cursor (`.gn-locked`: its frame stops blinking and
   * stays gold, its arrow turns gold, main.css) and the hint at the end of the links line, which says how to lock, or
   * how to let go.
   * @param {number} i The env to lock (always the selected one), or -1 to release the lock.
   * @returns {void}
   */
  function setLock(i) {
    locked = i
    buttons.forEach((b, n) => b.classList.toggle('gn-locked', n === i))
    if (lockHint) {
      lockHint.textContent = i < 0 ? ' · click a cartridge to keep it' : ' · click it again or Esc to let go'
    }
  }

  /**
   * Repaints each cartridge's label for the width its cartridge now has (cartLabel), and the NOW PLAYING label for the
   * panel's width (nowPlayingText): one line where the text fits, two where it would be squeezed. Painting only on a
   * change keeps a resize from redrawing all 24 labels every frame.
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
    paintNowPlaying()
  }

  /**
   * Paints the NOW PLAYING label for the selected env at the panel's current width, unless it already shows that text.
   * @returns {void}
   */
  function paintNowPlaying() {
    if (current < 0) {
      return
    }
    const text = nowPlayingText(envs[current].id, now.clientWidth)
    if (label.dataset.label !== text) {
      label.dataset.label = text
      paintPixelText(label, text, 2, '#3cbcfc')
    }
  }
  fitLabels()
  // The shelf's columns and the panel's width change with the window (and with the docs layout's sidebar), so the
  // labels follow the gallery body's width; selecting the first env below paints the panel's label the first time.
  new ResizeObserver(fitLabels).observe(arcade.querySelector('.gn-arcade-body'))
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
    // The blurb left the panel for the manual (DR-102); it stays the loop's description for screen readers.
    const description = env.blurb ? `${env.id}: ${env.blurb}` : `${env.id}, played by the showreel's policy`
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
   * Shows an env on the "now playing" panel and moves the cartridge cursor to it: the label, the family, the loop, the
   * HUD, the game manual's four lines, READ THE MANUAL (only when the env's page exists) and the links.
   * @param {number} i The env's index in the manifest.
   * @param {boolean} byReader Whether the reader chose it (which ends attract mode).
   * @returns {void}
   */
  function select(i, byReader) {
    if (byReader) {
      attract = false
      // The panel is a live region only once the reader drives it: in attract mode it changes every few seconds, and a
      // screen reader would read out a whole manual each time.
      now.setAttribute('aria-live', 'polite')
    }
    if (i === current) {
      return
    }
    current = i
    const env = envs[i]
    buttons.forEach((b, n) => b.setAttribute('aria-selected', String(n === i)))
    paintNowPlaying()
    paintPixelText(family, familyName(env.family ?? ''), 1, '#f8b800')
    showOnScreen(env)
    const result = env.result ? `  ${env.result}` : ''
    paintPixelText(hud, `SEED ${env.seed}  STEPS ${env.steps}\nRETURN ${env.return}${result}`, 2, '#fcfcfc')
    // The manual's lines are plain text from the manifest (textContent, never markup); a manifest written before the
    // manual existed leaves them empty.
    for (const row of manualRows) {
      row.textContent = keepTogether(env.manual?.[row.dataset.manual] ?? '')
    }
    const href = manualHref(env)
    manualButton.hidden = !href
    enterHint.hidden = !href
    if (href) {
      manualButton.href = href
      manualButton.setAttribute('aria-label', `Read the manual: ${env.id}`)
    } else {
      manualButton.removeAttribute('href')
    }
    // A manifest written before the links existed carries none of them: the whole line is hidden then, not left as
    // words that lead nowhere.
    linksLine.hidden = ![...links].some((link) => env[link.dataset.link])
    for (const link of links) {
      const target = env[link.dataset.link]
      link.hidden = !target
      if (target) {
        link.href = new URL(target, root).href
      }
    }
  }

  // Arrow keys move the cursor through the grid while focus is inside the gallery; Home and End jump to the ends;
  // Enter opens the selected env's page; Esc releases a lock.
  arcade.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      // Only a lock to release takes the key; otherwise Esc stays the page's (a dialog's, the search box's).
      if (locked >= 0) {
        e.preventDefault()
        setLock(-1)
      }
      return
    }
    if (e.key === 'Enter') {
      // A focused link (the example, Gymnasium's page, the API, READ THE MANUAL itself) follows its own address.
      if (e.target instanceof Element && e.target.closest('a')) {
        return
      }
      const href = current >= 0 ? manualHref(envs[current]) : null
      if (href) {
        // A cartridge is a button, whose default Enter is a click that only selects it again.
        e.preventDefault()
        window.location.href = href
      }
      return
    }
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
    // A key is as deliberate as a click: a lock moves along with the cursor instead of being left behind.
    if (locked >= 0) {
      setLock(next)
    }
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

/** The SVG namespace, for the anatomy charts' elements. @type {string} */
const SVG_NS = 'http://www.w3.org/2000/svg'

/**
 * How many anatomy boards this page has drawn: each board's arrowhead markers take ids from it, since a page can hold
 * two boards (FrozenLake's 4x4 and its 8x8 section) whose markers would otherwise collide. @type {number}
 */
let boardCount = 0

/**
 * Creates an SVG element with attributes and appends it to a parent.
 * @param {string} tag The element name.
 * @param {Record<string, string | number>} attrs Its attributes.
 * @param {Element} parent Where it goes.
 * @returns {SVGElement} The element.
 */
function svgElement(tag, attrs, parent) {
  const element = document.createElementNS(SVG_NS, tag)
  for (const [name, value] of Object.entries(attrs)) {
    element.setAttribute(name, String(value))
  }
  parent.appendChild(element)
  return element
}

/**
 * Formats a number for the anatomy readout: fixed digits, a true minus sign, and the units the spec names. A small
 * negative value that rounds to zero shows as 0, not −0: toFixed keeps the sign (−0.3 at 0 digits is "-0"), which the
 * walker's and the lander's angle readouts showed while their hulls stood level.
 * @param {number} value The value.
 * @param {number} digits Digits after the point.
 * @returns {string} The text.
 */
function formatReadout(value, digits) {
  const text = value.toFixed(digits)
  // "-0", "-0.0", "-0.00": every digit a zero, so the sign carries no information.
  return (/^-0(\.0*)?$/.test(text) ? text.slice(1) : text).replace('-', '−')
}

/**
 * Builds the env pages' episode anatomy (DR-115): the recorded episode beside a chart of every step, with a cursor
 * that follows the playing video and a live readout. Each `.gn-anatomy` element names the example's step trace
 * (`data-trace`, the `<id>.episode.json` that `--record` writes, DR-112), holds the video, and
 * carries its chart's description as JSON in a `script.gn-anatomy-spec`:
 * `bands`, stacked top to bottom, each either lines (`series` of trace columns, a `range`, `ticks`,
 * an optional `scale` such as radians to degrees) or a `strip` of one discrete column's values as coloured
 * ticks, or for a grid world (DR-131) one `board`: the env's frame (`image`, its pixel `size`, the top-left
 * `origin` and the `cell` size of its grid) with the episode's moves drawn between the cells two trace columns
 * name (`row`, `col`), each coloured by a discrete column (`path`: a slip in orange), and a ring on the cell the
 * video shows; and `readout`, the fields shown under the video. The board is drawn as its own picture above
 * the time bands, so the steps' cursor never crosses it.
 * The chart is drawn at the width it is shown and drawn again when that changes, so its 11 px text stays 11 px.
 * The trace's frame convention: video frame `f` shows the state after `f` steps, which is row `f`'s
 * observation (the one acted on at step `f`), or the trace's `final` state after the last step; the return
 * shown is the sum of the first `f` rewards. Pages without an anatomy section are untouched, and a trace that
 * cannot be fetched leaves the video playing without a chart.
 * @returns {Promise<void>} Settles when every chart on the page is built (or skipped).
 */
async function installAnatomy() {
  for (const box of document.querySelectorAll('.gn-anatomy')) {
    const video = box.querySelector('video')
    const chartBox = box.querySelector('.gn-anatomy-chart')
    const readout = box.querySelector('.gn-anatomy-readout')
    const legend = box.querySelector('.gn-anatomy-legend')
    let spec
    let trace
    try {
      spec = JSON.parse(box.querySelector('script.gn-anatomy-spec').textContent)
      const response = await fetch(new URL(box.dataset.trace, window.location.href))
      if (!response.ok) {
        continue
      }
      trace = await response.json()
    } catch {
      // A missing trace or a broken spec leaves the section as the video alone, never a half-drawn chart.
      continue
    }
    const column = Object.fromEntries(trace.columns.map((name, i) => [name, i]))
    const steps = trace.rows.length
    // The return after f steps, for every f from 0 to the episode's length.
    const returns = [0]
    for (const row of trace.rows) {
      returns.push(returns[returns.length - 1] + row[column.reward])
    }
    /**
     * The value of a trace column in the state video frame f shows: row f's, or after the last step the final state,
     * whose values follow the columns after "step" in order (the observation's columns).
     * @param {string} name The column.
     * @param {number} f The frame, 0 to steps.
     * @returns {number | undefined} The value; undefined for a column the final state does not have (the action).
     */
    const valueAt = (name, f) => {
      if (f < steps) {
        return trace.rows[f][column[name]]
      }
      const index = column[name] - 1
      return trace.final && index >= 0 && index < trace.final.length ? trace.final[index] : undefined
    }

    const padLeft = 52
    const padRight = 10
    let cursor = null
    let x = () => 0
    // The board (DR-131) is drawn apart from the bands that run along the steps; a page has one board at most.
    const boardBand = spec.bands.find((band) => band.board)
    const timeBands = spec.bands.filter((band) => !band.board)
    const boardId = boardBand ? boardCount++ : -1
    // The frame on screen, -1 before the first one: the board's ring goes back to it when a resize redraws the board.
    let shown = -1
    /** Puts the board's ring on the cell of frame f; replaced by each drawing of the board. @type {(f: number) => void} */
    let placeRing = () => {}
    /**
     * Draws the board at the chart's width: the frame scaled to the band's height (or narrower, to fit), every move
     * of the episode as an arrow from its cell to the next one's (a move that stays in place has no arrow; its strip
     * still shows it), each spread a little from the others so repeated passes stay visible, and the ring.
     * @param {number} width The chart's width in CSS pixels.
     * @returns {void}
     */
    const drawBoard = (width) => {
      const board = boardBand.board
      const [imageWidth, imageHeight] = board.size
      let drawnHeight = boardBand.height ?? 220
      let drawnWidth = (drawnHeight * imageWidth) / imageHeight
      if (drawnWidth > width - 8) {
        drawnWidth = width - 8
        drawnHeight = (drawnWidth * imageHeight) / imageWidth
      }
      const scale = drawnWidth / imageWidth
      const left = (width - drawnWidth) / 2
      const svg = svgElement('svg', { viewBox: `0 0 ${width} ${drawnHeight + 8}`, width, height: drawnHeight + 8, role: 'img', 'aria-label': board.name ?? 'The episode on the board' }, chartBox)
      svgElement('image', { href: new URL(board.image, window.location.href).href, x: left.toFixed(1), y: 4, width: drawnWidth.toFixed(1), height: drawnHeight.toFixed(1), preserveAspectRatio: 'none' }, svg)
      /** The centre of the cell frame f shows, in the board's drawn pixels. @param {number} f The frame. @returns {number[]} x and y. */
      const centre = (f) => [
        left + (board.origin[0] + (valueAt(board.col, f) + 0.5) * board.cell[0]) * scale,
        4 + (board.origin[1] + (valueAt(board.row, f) + 0.5) * board.cell[1]) * scale
      ]
      // One arrowhead per colour the path uses, each filled like its line.
      const defs = svgElement('defs', {}, svg)
      const markers = {}
      Object.entries(board.path.colors).forEach(([value, colour], n) => {
        const id = `gn-board-${boardId}-${n}`
        const marker = svgElement('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 4, markerHeight: 4, orient: 'auto' }, defs)
        svgElement('path', { d: 'M0,0 L10,5 L0,10 z', fill: colour }, marker)
        markers[value] = id
      })
      const spread = Math.max(1.5, 2.5 * scale)
      for (let i = 0; i < steps; i++) {
        const [ax, ay] = centre(i)
        const [bx, by] = centre(i + 1)
        if (Math.abs(ax - bx) + Math.abs(ay - by) < 0.5) {
          continue
        }
        const offset = ((i % 5) - 2) * spread
        const value = String(trace.rows[i][column[board.path.column]])
        svgElement('line', { x1: (ax + offset).toFixed(1), y1: (ay + offset).toFixed(1), x2: (bx + offset).toFixed(1), y2: (by + offset).toFixed(1), stroke: board.path.colors[value] ?? '#adb5bd', 'stroke-width': 3, 'stroke-opacity': 0.8, 'marker-end': markers[value] ? `url(#${markers[value]})` : '' }, svg)
      }
      // The ring: a white circle under a magenta one, readable on ice, on the dark holes and on the taxi's road.
      const radius = Math.max(6, 0.32 * Math.min(board.cell[0], board.cell[1]) * scale)
      const halo = svgElement('circle', { r: radius.toFixed(1), fill: 'none', stroke: '#fff', 'stroke-width': 3.5 }, svg)
      const ring = svgElement('circle', { r: radius.toFixed(1), fill: 'none', stroke: '#c2189b', 'stroke-width': 1.8 }, svg)
      placeRing = (f) => {
        const [cx, cy] = centre(f)
        for (const circle of [halo, ring]) {
          circle.setAttribute('cx', cx.toFixed(1))
          circle.setAttribute('cy', cy.toFixed(1))
        }
      }
      placeRing(Math.max(0, shown))
    }
    /** Draws the chart at the box's current width; the cursor is redrawn by the frame loop. @returns {void} */
    const draw = () => {
      const width = Math.max(240, Math.round(chartBox.clientWidth))
      chartBox.replaceChildren()
      if (boardBand) {
        drawBoard(width)
      }
      const heights = timeBands.map((band) => (band.strip ? 16 : band.height ?? 110))
      const height = heights.reduce((a, b) => a + b, 0) + (timeBands.length - 1) * 14 + 32
      const svg = svgElement('svg', { viewBox: `0 0 ${width} ${height}`, width, height, 'font-family': 'system-ui, sans-serif', 'font-size': 11, fill: '#adb5bd', role: 'img', 'aria-label': spec.label ?? 'The episode, step by step' }, chartBox)
      x = (step) => padLeft + (step / steps) * (width - padLeft - padRight)
      let top = 8
      timeBands.forEach((band, b) => {
        const h = heights[b]
        if (band.strip) {
          // One tick per step, coloured by the step's value of the column (an action: which way the push went).
          const colours = band.strip.colors
          for (let i = 0; i < steps; i++) {
            svgElement('rect', { x: x(i).toFixed(2), y: top, width: Math.max(1, (width - padLeft - padRight) / steps).toFixed(2), height: h - 4, fill: colours[String(trace.rows[i][column[band.strip.column]])] ?? '#5c636a' }, svg)
          }
          svgElement('text', { x: 4, y: top + h - 6 }, svg).textContent = band.label
        } else {
          const [lo, hi] = band.range
          const scale = band.scale ?? 1
          const y = (v) => top + (1 - (v * scale - lo) / (hi - lo)) * h
          for (const tick of band.ticks ?? []) {
            const ty = top + (1 - (tick - lo) / (hi - lo)) * h
            svgElement('line', { x1: padLeft, x2: width - padRight, y1: ty.toFixed(1), y2: ty.toFixed(1), stroke: tick === 0 ? '#495057' : '#343a40' }, svg)
            svgElement('text', { x: 4, y: (ty + 4).toFixed(1) }, svg).textContent = `${String(tick).replace('-', '\u2212')}${band.unit ?? ''}`
          }
          for (const series of band.series) {
            const points = trace.rows.map((row, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(row[column[series.column]]).toFixed(1)}`).join('')
            const attrs = { d: points, fill: 'none', stroke: series.color, 'stroke-width': series.width ?? 1.6 }
            if (series.dash) {
              attrs['stroke-dasharray'] = '4 3'
            }
            svgElement('path', attrs, svg)
          }
        }
        top += h + 14
      })
      // Step numbers along the bottom, the first anchored left and the last right so neither leaves the chart.
      const marks = 5
      for (let m = 0; m <= marks; m++) {
        const step = Math.round((steps * m) / marks)
        svgElement('text', { x: x(step).toFixed(1), y: height - 6, 'text-anchor': m === 0 ? 'start' : m === marks ? 'end' : 'middle' }, svg).textContent = String(step)
      }
      cursor = svgElement('line', { x1: x(0), x2: x(0), y1: 4, y2: height - 18, stroke: '#fff', 'stroke-width': 1.2 }, svg)
    }

    if (legend) {
      legend.replaceChildren(...spec.bands.flatMap((band) => (band.board ? [{ name: band.board.name, colors: band.board.path.colors }] : band.strip ? [band.strip] : band.series)).map((item) => {
        const entry = document.createElement('span')
        const swatch = document.createElement('i')
        // A strip's swatch shows its colours side by side; a line's, its colour.
        swatch.style.background = item.colors ? `linear-gradient(90deg, ${Object.values(item.colors).join(', ')})` : item.color
        entry.append(swatch, item.name)
        return entry
      }))
    }
    draw()
    // Redrawn at the new width; the frame loop below moves the new cursor on its next frame.
    new ResizeObserver(() => draw()).observe(chartBox)

    /**
     * The step a video time shows: frame f is stamped f / fps, and a hair of slack keeps a timestamp a rounding error
     * below its frame's from reading as the frame before.
     * @param {number} seconds The time in the video.
     * @returns {number} The frame, 0 to steps.
     */
    const frameAt = (seconds) => Math.min(steps, Math.max(0, Math.floor(seconds * trace.fps + 1e-6)))
    /**
     * Moves the cursor, the board's ring and the readout to frame f, when it is not the one they show already.
     * @param {number} f The frame on screen.
     * @returns {void}
     */
    const show = (f) => {
      if (f !== shown && cursor) {
        shown = f
        const at = Math.min(f, steps - 1)
        cursor.setAttribute('x1', x(at).toFixed(1))
        cursor.setAttribute('x2', x(at).toFixed(1))
        placeRing(f)
        if (readout) {
          readout.replaceChildren(...spec.readout.flatMap((field, n) => {
            const label = document.createTextNode(`${n ? ' · ' : ''}${field.label} `)
            const value = document.createElement('b')
            if (field.column === '@step') {
              value.textContent = String(f)
            } else if (field.column === '@return') {
              value.textContent = formatReadout(returns[f], field.digits ?? 1)
            } else {
              const raw = valueAt(field.column, f)
              value.textContent = raw === undefined ? '—' : field.map ? field.map[String(raw)] ?? String(raw) : formatReadout(raw * (field.scale ?? 1), field.digits ?? 2)
            }
            // The unit carries its own spacing, as the spec writes it: " m" after a length, "°" right after an angle.
            return field.unit ? [label, value, document.createTextNode(field.unit)] : [label, value]
          }))
        }
      }
    }
    if ('requestVideoFrameCallback' in video) {
      // The presented frame's own timestamp: currentTime runs ahead of the picture by up to a frame, which at the toy
      // texts' 4 fps put the ring a cell ahead of the elf.
      const onFrame = (now, metadata) => {
        show(frameAt(metadata.mediaTime))
        video.requestVideoFrameCallback(onFrame)
      }
      video.requestVideoFrameCallback(onFrame)
      show(frameAt(video.currentTime))
    } else {
      /** Follows currentTime once per animation frame, where the callback does not exist. @returns {void} */
      const follow = () => {
        show(frameAt(video.currentTime))
        window.requestAnimationFrame(follow)
      }
      window.requestAnimationFrame(follow)
    }
  }
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

/**
 * How long a remembered sidebar layout survives with no visits, in milliseconds (48 hours). Every read and write of
 * the record re-stamps it from "now", so the window slides: the state expires only after 48 hours of inactivity.
 * @type {number}
 */
const TOC_STATE_TTL_MS = 48 * 60 * 60 * 1000

/**
 * Storage-format version. Bump it to discard every reader's saved sidebar state at once when the record's shape
 * changes (a record of another version is dropped as if expired), so an old record is never misread as the new shape.
 * @type {number}
 */
const TOC_STATE_SCHEMA = 1

/**
 * Separator joining ancestor labels into a node key. U+203A (›) occurs in no sidebar title; a title that contained it
 * could map two distinct nodes onto one key (a cosmetic glitch, not a failure).
 * @type {string}
 */
const TOC_KEY_SEP = '›'

/**
 * Remembers the sidebar's expanded and collapsed sections in `localStorage` for a sliding 48-hour window, and
 * restores them on the next page (DR-138). The code is NumSharp's site's, so the SciSharp sites behave alike.
 *
 * Why it is not a one-liner: the modern template keeps each node's expanded flag in a private closure and re-renders
 * the whole `#toc` on every toggle (lit-html), so setting the `.expanded` class on restore would be wiped by the next
 * toggle. Instead it drives docfx's own model, clicking the toggle of each node whose remembered state differs from
 * what docfx rendered.
 *
 * Defaults: a section's `expanded` in toc.yml (the Environments sidebar's four families open, DR-138) holds until the
 * reader toggles that section; only toggled sections are stored. The current page's own branch is never collapsed by
 * a remembered choice, so the reader always sees where they are.
 *
 * Safe to call before `#toc` is filled (docfx fills it after fetching toc.json): a MutationObserver waits for the
 * first expander. Does nothing without a `#toc` or without a usable `localStorage` (private mode, file://).
 *
 * Reference: <https://github.com/SciSharp/NumSharp/blob/4deab50400eda8f853c076a6dd09fdf167cd3512/docs/website-src/templates/numsharp/public/main.js>
 * (NumSharp's `installTocStatePersistence`).
 * @returns {void}
 */
function installTocStatePersistence() {
  // Reading localStorage can itself throw (site data blocked), so probe it and bail rather than break the page.
  let store
  try {
    store = window.localStorage
  } catch {
    return
  }
  if (!store) {
    return
  }

  const toc = document.getElementById('toc')
  if (!toc) {
    return
  }

  // The record is scoped to this page's toc.json: each sidebar (Docs, Environments, API) keeps its own state, and
  // localStorage is per origin, not per path, so NumSharp's, OptunaSharp's and Gym.NET's sites, all on
  // scisharp.github.io, would otherwise share (and overwrite) one record.
  const tocRelMeta = document.querySelector('meta[name="docfx:tocrel"]')
  const tocRel = tocRelMeta ? tocRelMeta.content || '' : ''
  let scope
  try {
    scope = new URL(tocRel, window.location.href).href
  } catch {
    scope = tocRel
  }
  const storageKey = 'docfx:tocExpanded:' + scope

  // docfx overwrites toc.yml's `expanded` when it opens the active branch, so the defaults are read from toc.json
  // itself: a section that links to the current page then honours its configured default too.
  const defaultsReady = fetch(scope.replace(/\.html(?=$|[?#])/i, '.json'))
    .then(response => response.ok ? response.json() : null)
    .then(data => {
      const defaults = Object.create(null)
      function visit(items, ancestors = []) {
        for (const item of items || []) {
          const path = [...ancestors, item.name]
          if (typeof item.expanded === 'boolean') {
            defaults[path.join(TOC_KEY_SEP)] = item.expanded
          }
          visit(item.items, path)
        }
      }
      visit(data?.items)
      return defaults
    })
    .catch(() => Object.create(null))
  let defaultStates = Object.create(null)

  /**
   * Reads the remembered states, dropping an expired, malformed or other-version record (each read as "nothing
   * remembered").
   * @returns {{[nodeKey: string]: boolean}} node key to expanded flag; empty when nothing valid is stored.
   */
  function loadNodes() {
    const nodes = Object.create(null)
    let raw
    try {
      raw = store.getItem(storageKey)
    } catch {
      return nodes
    }
    if (!raw) {
      return nodes
    }
    let data
    try {
      data = JSON.parse(raw)
    } catch {
      // A corrupt record cannot be repaired; drop it so it cannot keep failing.
      removeRecord()
      return nodes
    }
    const expired = !data ||
      data.v !== TOC_STATE_SCHEMA ||
      !Number.isFinite(data.expiresAt) ||
      data.expiresAt <= Date.now()
    if (expired) {
      removeRecord()
      return nodes
    }
    if (data.nodes && typeof data.nodes === 'object' && !Array.isArray(data.nodes)) {
      for (const [key, value] of Object.entries(data.nodes)) {
        if (typeof value === 'boolean') nodes[key] = value
      }
    }
    return nodes
  }

  /**
   * Stores the states and re-stamps the 48-hour expiry from now, which is what slides the window forward on every
   * visit and toggle.
   * @param {{[nodeKey: string]: boolean}} nodeStates node key to expanded flag.
   * @returns {void}
   */
  function saveNodes(nodeStates) {
    const record = { v: TOC_STATE_SCHEMA, expiresAt: Date.now() + TOC_STATE_TTL_MS, nodes: nodeStates }
    try {
      store.setItem(storageKey, JSON.stringify(record))
    } catch {
      // A full or unavailable store is not fatal: the sidebar still works, it just forgets this time.
    }
  }

  /**
   * Deletes this sidebar's record, when it is expired or corrupt.
   * @returns {void}
   */
  function removeRecord() {
    try {
      store.removeItem(storageKey)
    } catch {
      // Nothing to do, and it must not break the page.
    }
  }

  /** The remembered states, refreshed before the restore and before each recorded toggle. */
  let nodeStates = loadNodes()
  if (Object.keys(nodeStates).length > 0) saveNodes(nodeStates)

  /**
   * The label of an `<li>`'s own node: its direct `<a>`, or the `name-only` span of a group without a link; never the
   * empty `.expand-stub` nor the nested `<ul>`. The `<wbr>` breaks docfx inserts carry no text.
   * @param {Element} li the sidebar's list item.
   * @returns {string} the node's own label, trimmed; empty when it has none.
   */
  function labelOf(li) {
    const anchor = li.querySelector(':scope > a')
    if (anchor) {
      return anchor.textContent.trim()
    }
    const nameSpan = li.querySelector(':scope > span.name-only')
    return nameSpan ? nameSpan.textContent.trim() : ''
  }

  /**
   * A node's identity: the chain of its ancestors' labels down to it. It survives docfx's re-render on every toggle
   * (the elements are new, their titles are not) and is the same on every page sharing the toc.json, which is what
   * lets a choice made on one page be restored on the next.
   * @param {Element} li the list item to identify.
   * @returns {string} the labels from the root down, joined by {@link TOC_KEY_SEP}.
   */
  function keyOf(li) {
    const parts = []
    for (let cur = li; cur && cur !== toc; cur = cur.parentElement) {
      if (cur.tagName === 'LI') {
        parts.unshift(labelOf(cur))
      }
    }
    return parts.join(TOC_KEY_SEP)
  }

  /**
   * True while the restore clicks, so the click listener records only the reader's own toggles.
   * @type {boolean}
   */
  let applyingRestore = false

  /**
   * Finds a section whose rendered state differs from what it should be. A remembered choice wins; otherwise only a
   * section that links to the current page itself takes its toc.yml default here (docfx already renders the other
   * defaults, and forces the current page's branch open). A remembered collapse is never applied to an ancestor of
   * the current page, so the page stays visible in the sidebar.
   * @returns {Element|null} the `<li>` to toggle, or null when the sidebar already matches.
   */
  function findMismatchedExpander() {
    const expanders = toc.querySelectorAll('li.expander')
    for (const li of expanders) {
      const key = keyOf(li)
      const anchor = li.querySelector(':scope > a')
      const url = anchor && anchor.getAttribute('href') !== '#'
        ? new URL(anchor.href, window.location.href)
        : null
      // docfx's equivalence for index pages and extensionless URLs.
      const normalize = path => path.replace(/\/index\.html$/gi, '/')
        .replace(/\.html$/gi, '').replace(/\/$/g, '').toLowerCase()
      const isCurrentPage = url && url.origin === window.location.origin &&
        normalize(url.pathname) === normalize(window.location.pathname)
      const remembered = Object.hasOwn(nodeStates, key)
      if (!remembered && (!isCurrentPage || !Object.hasOwn(defaultStates, key))) {
        continue
      }
      const wanted = remembered ? nodeStates[key] : defaultStates[key]
      // Only a true ancestor of the current page has to stay open to show it.
      if (wanted === false && li.classList.contains('active') && !isCurrentPage) {
        continue
      }
      const isExpanded = li.classList.contains('expanded')
      if (isExpanded !== wanted) {
        return li
      }
    }
    return null
  }

  /**
   * Brings the rendered sidebar to the remembered states by clicking each mismatched section's `.expand-stub`. Each
   * click re-renders synchronously and replaces the elements, so every pass queries again; a toggle changes no other
   * node's state, so each pass fixes one node, and the budget only guards against a pathological tree.
   * @returns {void}
   */
  function restore() {
    applyingRestore = true
    try {
      const budget = toc.querySelectorAll('li.expander').length * 2 + 8
      for (let pass = 0; pass < budget; pass++) {
        const li = findMismatchedExpander()
        if (!li) {
          break
        }
        const stub = li.querySelector(':scope > .expand-stub')
        if (!stub) {
          // No handle to drive docfx's model with: stop rather than spin.
          break
        }
        stub.click()
      }
    } finally {
      applyingRestore = false
    }
  }

  // The toggled node's identity is taken in the capture phase, before docfx replaces the clicked element; its new
  // state is read in the bubble phase, after docfx's own handler has re-rendered (a microtask queued in the capture
  // phase can run before that handler for real input).
  const toggles = new WeakMap()
  toc.addEventListener('click', event => {
    // The restore's clicks are untrusted, and so is any other synthetic click: neither is the reader's choice.
    if (applyingRestore || !event.isTrusted) {
      return
    }
    const target = event.target
    if (!(target instanceof Element)) {
      return
    }
    // A toggle is the caret or a group's own href='#' link; a link to a page navigates and is not recorded.
    const onStub = target.closest('.expand-stub')
    const anchor = target.closest('a')
    const onGroupToggle = anchor && anchor.getAttribute('href') === '#'
    if (!onStub && !onGroupToggle) {
      return
    }
    const li = target.closest('li.expander')
    if (!li) {
      return
    }
    toggles.set(event, keyOf(li))
  }, true)

  toc.addEventListener('click', event => {
    if (!toggles.has(event)) return
    const key = toggles.get(event)
    toggles.delete(event)
    for (const current of toc.querySelectorAll('li.expander')) {
      if (keyOf(current) !== key) continue
      // Merged into the latest record: another open tab may have changed another section since this page loaded.
      nodeStates = loadNodes()
      nodeStates[key] = current.classList.contains('expanded')
      saveNodes(nodeStates)
      break
    }
  })

  /**
   * True once this page load has restored, so the observer's later firings (and the restore's own mutations) do not
   * run it again.
   * @type {boolean}
   */
  let booted = false

  /**
   * Restores once the unfiltered sidebar has rendered. docfx keeps its title filter across reloads, and a filtered
   * sidebar hides branches that cannot be reconciled until the filter is cleared.
   * @returns {Promise<void>} settles when the restore has run, or at once when the sidebar is not ready.
   */
  async function boot() {
    if (booted) {
      return
    }
    const filter = toc.querySelector('input')
    if (filter && filter.value.trim()) return
    if (!toc.querySelector('li.expander')) {
      // Not filled yet: wait for the next mutation.
      return
    }
    booted = true
    // Stop observing before restoring, so the restore's own mutations do not call back.
    observer.disconnect()
    defaultStates = await defaultsReady
    // The reader may have started filtering while the defaults loaded.
    if (filter && filter.value.trim()) {
      booted = false
      observer.observe(toc, { childList: true, subtree: true })
      return
    }
    // Another tab may have changed the layout while a restored filter postponed this.
    nodeStates = loadNodes()
    restore()
  }

  // #toc fills after toc.json loads, so wait for its first expander; and try now, in case it is already there.
  const observer = new MutationObserver(boot)
  observer.observe(toc, { childList: true, subtree: true })
  boot()
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
    installAnatomy()
    rewriteRepoLinks()
    installTocStatePersistence()
  }
}
