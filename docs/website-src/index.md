---
_layout: landing
---

<div class="gn-home">
<section class="gn-hero" aria-label="The Gym.NET showreel">
<video class="gn-video" autoplay muted loop playsinline controls preload="metadata" poster="images/showreel-poster.jpg">
<source src="../media/gym-net-showreel.mp4" type="video/mp4">
<img src="../media/gym-net-hero.webp" alt="Gym.NET's environments playing in panels over a 16-bit dusk beach">
</video>
</section>
<section class="gn-intro">
<p class="gn-kicker">Reinforcement learning environments for .NET</p>
<h1 class="gn-title">Gym.NET</h1>
<p class="gn-lede">Farama Gymnasium's API and behavior in C#. Given the same seed and actions, every environment returns the same observations, rewards, flags and infos as Gymnasium's, bit for bit, and renders the same frames, pixel for pixel (MuJoCo's within one level). A test suite that runs Gymnasium itself inside the test process checks it, call by call.</p>
<div class="gn-actions">
<a class="gn-button gn-button-primary" href="../guides/getting_started.md">Get started</a>
<a class="gn-button" href="#select-your-environment">Environments</a>
<a class="gn-button" href="api/index.md">API reference</a>
<a class="gn-button" href="https://github.com/SciSharp/Gym.NET">GitHub</a>
</div>
<div class="gn-code">
<pre><code class="lang-bash">dotnet add package Gym.NET.Environments</code></pre>
<pre><code class="lang-csharp">using Gym;
using Gym.Envs;
using NumSharp;
// make builds the registered env inside Gymnasium's wrappers, as gymnasium.make does
using Env&lt;NDArray, NDArray&gt; env = Gymnasium.Make&lt;NDArray, NDArray&gt;("LunarLander-v3");
NDArray observation = env.Reset(seed: 42).Observation;
using NDArray action = env.ActionSpace.Sample();
var (next, reward, terminated, truncated, info) = env.Step(action);</code></pre>
</div>
</section>

[!INCLUDE [The environment gallery](includes/arcade.md)]

<section class="gn-docmap" aria-label="Documentation">
<a class="gn-card" href="../guides/getting_started.md"><span>Start here</span><strong>Getting started</strong><p>Making envs, the reset/step loop, seeding, spaces, render modes, wrappers, vector envs.</p></a>
<a class="gn-card" href="../guides/creating_custom_environments.md"><span>Build</span><strong>Custom environments</strong><p>Writing an <code>Env&lt;TObs, TAct&gt;</code>, registering it for <code>Gymnasium.Make</code>, checking it.</p></a>
<a class="gn-card" href="../guides/migrating_to_1.0.md"><span>Upgrade</span><strong>Migrating from 0.2</strong><p>What changed in each 0.2 call, and <code>Gym.Legacy</code> for code that cannot move yet.</p></a>
<a class="gn-card" href="../architecture.md"><span>Design</span><strong>Architecture</strong><p>The packages, the env contract, array ownership, and how parity with Gymnasium is proven.</p></a>
<a class="gn-card" href="../releases/RELEASE_1.0.0.md"><span>Release</span><strong>1.0.0 notes</strong><p>What 1.0 contains and leaves out, and every known difference from Gymnasium.</p></a>
<a class="gn-card" href="../guides/development_and_testing.md"><span>Contribute</span><strong>Development and testing</strong><p>Building, the unit suite, and the live parity suite against Gymnasium.</p></a>
</section>
</div>
