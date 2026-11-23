<!-- The 8-bit SELECT YOUR ENVIRONMENT gallery, shared by the landing page (website-src/index.md) and the Environments
page (environments/index.md) through DocFX's [!INCLUDE] (DR-106). It is only the empty frame: main.js fills the shelf
and the screen from images/envs/envs.json, which tools/showreel's gallery command writes. Keep it one HTML block with
no blank line inside, or markdown would end the block there and render the rest as text. -->
<section class="gn-arcade" id="select-your-environment" aria-label="Select your environment">
<div class="gn-tv">
<canvas class="gn-pixel gn-arcade-title" data-text="SELECT YOUR ENVIRONMENT" data-scale="3" data-color="#fcfcfc" data-shadow="#f83800"></canvas>
<div class="gn-arcade-body">
<div class="gn-carts" role="listbox" aria-label="Environments"></div>
<div class="gn-now" aria-live="polite">
<canvas class="gn-pixel gn-now-label" data-scale="2" data-color="#3cbcfc"></canvas>
<div class="gn-screen"><video class="gn-screen-video" muted loop playsinline autoplay disablepictureinpicture disableremoteplayback></video><img class="gn-screen-img" alt="" hidden></div>
<canvas class="gn-pixel gn-hud" data-scale="2" data-color="#fcfcfc"></canvas>
<p class="gn-blurb"></p>
<pre class="gn-snippet"><code class="lang-csharp"></code></pre>
</div>
</div>
<canvas class="gn-pixel gn-press-start" data-text="PRESS START" data-scale="2" data-color="#f8b800"></canvas>
</div>
<noscript><p>Twenty-four environments play here when JavaScript is on: CartPole-v1, MountainCar-v0, Pendulum-v1, Acrobot-v1, InvertedPendulum-v5, InvertedDoublePendulum-v5, Reacher-v5, Pusher-v5, LunarLander-v3, BipedalWalker-v3, BipedalWalkerHardcore-v3, CarRacing-v3, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5, HumanoidStandup-v5, Swimmer-v5, Blackjack-v1, FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1 and Taxi-v4.</p></noscript>
</section>
