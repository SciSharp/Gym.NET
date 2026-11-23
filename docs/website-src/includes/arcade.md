<!-- The 8-bit SELECT YOUR ENVIRONMENT gallery, shared by the landing page (website-src/index.md) and the Environments
page (environments/index.md) through DocFX's [!INCLUDE] (DR-106). It is only the empty frame: main.js fills the shelf,
the screen and the game manual (DR-102) from images/envs/envs.json, which tools/showreel's gallery command writes.
Keep it one HTML block with no blank line inside, or markdown would end the block there and render the rest as text. -->
<section class="gn-arcade" id="select-your-environment" aria-label="Select your environment">
<div class="gn-tv">
<canvas class="gn-pixel gn-arcade-title" data-text="SELECT YOUR ENVIRONMENT" data-scale="3" data-color="#fcfcfc" data-shadow="#f83800"></canvas>
<div class="gn-arcade-body">
<div class="gn-carts" role="listbox" aria-label="Environments"></div>
<div class="gn-now" aria-live="polite">
<canvas class="gn-pixel gn-now-label" data-scale="2" data-color="#3cbcfc"></canvas>
<canvas class="gn-pixel gn-family" data-scale="1" data-color="#f8b800"></canvas>
<div class="gn-screen"><video class="gn-screen-video" muted loop playsinline autoplay disablepictureinpicture disableremoteplayback></video><img class="gn-screen-img" alt="" hidden></div>
<canvas class="gn-pixel gn-hud" data-scale="2" data-color="#fcfcfc"></canvas>
<div class="gn-manual">
<canvas class="gn-pixel gn-manual-title" data-text="HOW TO PLAY" data-scale="2" data-color="#f8b800"></canvas>
<dl class="gn-manual-rows">
<div><dt><canvas class="gn-pixel" data-text="CONTROLS" data-scale="1" data-color="#3cbcfc"></canvas></dt><dd data-manual="controls"></dd></div>
<div><dt><canvas class="gn-pixel" data-text="YOU SEE" data-scale="1" data-color="#3cbcfc"></canvas></dt><dd data-manual="youSee"></dd></div>
<div><dt><canvas class="gn-pixel" data-text="SCORE" data-scale="1" data-color="#3cbcfc"></canvas></dt><dd data-manual="score"></dd></div>
<div><dt><canvas class="gn-pixel" data-text="GAME OVER" data-scale="1" data-color="#3cbcfc"></canvas></dt><dd data-manual="gameOver"></dd></div>
</dl>
</div>
<a class="gn-manual-button" hidden><canvas class="gn-pixel" data-text="&gt; READ THE MANUAL" data-scale="3" data-color="#000"></canvas></a>
<p class="gn-links">Or open <a data-link="example" target="_blank" rel="noopener">its example</a> · <a data-link="gymnasium" target="_blank" rel="noopener">Gymnasium's page</a> · <a data-link="api">the API</a><span class="gn-enter" hidden> · Enter opens the manual</span></p>
</div>
</div>
<canvas class="gn-pixel gn-press-start" data-text="PRESS START" data-scale="2" data-color="#f8b800"></canvas>
</div>
<noscript><p>Twenty-four environments play here when JavaScript is on: CartPole-v1, MountainCar-v0, Pendulum-v1, Acrobot-v1, InvertedPendulum-v5, InvertedDoublePendulum-v5, Reacher-v5, Pusher-v5, LunarLander-v3, BipedalWalker-v3, BipedalWalkerHardcore-v3, CarRacing-v3, HalfCheetah-v5, Hopper-v5, Walker2d-v5, Ant-v5, Humanoid-v5, HumanoidStandup-v5, Swimmer-v5, Blackjack-v1, FrozenLake-v1, FrozenLake8x8-v1, CliffWalking-v1 and Taxi-v4.</p></noscript>
</section>
