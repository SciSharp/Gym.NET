#nullable enable
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Numerics;
using Gym.Envs;
using Gym.Envs.Registration;
using Gym.Exceptions;
using Gym.Observations;
using Gym.Rendering;
using Gym.Spaces;
using Gym.Utils;
using NumSharp;
using Color = Gym.Rendering.Color;
using Tuple = Gym.Spaces.Tuple;

namespace Gym.Environments.Envs.ToyText {
    /// <summary>
    ///     Blackjack-v1: beat the dealer without going over 21. The player hits (1) or sticks (0); a hit that busts loses
    ///     (-1.0), sticking plays the dealer's hand out (drawing below 17) and scores the game (1.0 win, -1.0 loss, 0.0 draw).
    ///     <c>natural</c> pays 1.5 for a winning natural blackjack; <c>sab</c> (Sutton and Barto's rules, Blackjack-v1's
    ///     registered default, which overrides <c>natural</c>) makes a natural against a dealer without one an automatic win.
    ///     Byte for byte what Gymnasium's Blackjack computes: the cards drawn from the generator (an infinite deck,
    ///     <c>np_random.choice</c>), the face-up card's suit and face, observations, rewards and <c>rgb_array</c> pixels.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Observations are Gym.NET's tuple, <see langword="object"/>?[] (what the <c>Tuple(Discrete(32), Discrete(11),
    ///     Discrete(2))</c> observation space samples), of three new 0-d int64 arrays the caller owns:
    ///     <c>(player_sum, dealer_card, usable_ace)</c>. <see cref="BlackjackObservation"/> reads one by name (DR-13). Rewards
    ///     are Gymnasium's float64 values; the info is empty. A step asserts the action against the space first (Gymnasium's
    ///     bare <c>assert</c>), so only 0, 1, <c>True</c>, <c>False</c> and 0-d integer arrays of those values pass.
    ///     </para>
    ///     <para>
    ///     A reset draws the dealer's two cards, the player's two, the face-up card's suit and, for a ten, its face (J, Q or
    ///     K): five or six draws, in Gymnasium's order. Nothing stops a step after the game ended: the hand goes on, as in
    ///     Gymnasium. Rendering draws Gymnasium's pixel-art table (the digits are images, no font); human mode renders into
    ///     the window from <c>viewer_factory</c> or <c>Gymnasium.HumanViewerFactory</c> (DR-20) at 4 frames per second, and any
    ///     other mode returns the frame. Not thread-safe.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L124">gymnasium/envs/toy_text/blackjack.py</see>.</para>
    /// </remarks>
    [GymOptions]
    public partial class BlackjackEnv : Env<object?[], NDArray> {
        /// <summary>Gymnasium's <c>ART_PIXEL</c>: the art is authored at 4x, so every position is a multiple of 4.</summary>
        private const int ArtPixel = 4;

        /// <summary>Gymnasium's <c>SCREEN_SIZE</c>: the frame, 384x376 pixels.</summary>
        private static readonly (int Width, int Height) ScreenSize = (384, 376);

        /// <summary>Gymnasium's <c>FELT_COLOR</c>.</summary>
        private static readonly Color FeltColor = new(196, 216, 158);

        /// <summary>Gymnasium's <c>DEALER_LABEL_POS</c>, <c>DEALER_CARD_POS</c>, <c>HIDDEN_CARD_POS</c> and <c>PLAYER_LABEL_POS</c>.</summary>
        private static readonly (int X, int Y) DealerLabelPos = (40, 32), DealerCardPos = (96, 72), HiddenCardPos = (208, 72), PlayerLabelPos = (40, 208);

        /// <summary>Gymnasium's <c>PLAYER_SUM_TOP</c> and <c>USABLE_ACE_TOP</c>: the rows of the player's total and of the usable-ace label.</summary>
        private const int PlayerSumTop = 248, UsableAceTop = 328;

        /// <summary>Gymnasium's <c>LABEL_DIGIT_GAP</c>, <c>SMALL_DIGIT_GAP</c> and <c>BIG_DIGIT_GAP</c>, in pixels.</summary>
        private const int LabelDigitGap = 12, SmallDigitGap = 4, BigDigitGap = 12;

        /// <summary>The suits <c>reset</c> chooses the face-up card's from, in Gymnasium's order.</summary>
        private static readonly string[] Suits = { "C", "D", "H", "S" };

        /// <summary>The faces <c>reset</c> chooses a ten's from, in Gymnasium's order.</summary>
        private static readonly string[] Faces = { "J", "Q", "K" };

        /// <summary>The human-mode window, opened on the first human render.</summary>
        private readonly HumanWindow _window;

        /// <summary>Creates the env.</summary>
        /// <param name="renderMode">Gymnasium's <c>render_mode</c>: <see langword="null"/>, <c>"human"</c> or <c>"rgb_array"</c> (any other mode draws like <c>rgb_array</c>, as in Gymnasium).</param>
        /// <param name="natural">Gymnasium's <c>natural</c>: a winning natural blackjack pays 1.5 (ignored when <paramref name="sab"/> is set).</param>
        /// <param name="sab">Gymnasium's <c>sab</c>: Sutton and Barto's rules, where a natural against a dealer without one wins outright.</param>
        /// <param name="viewerFactory">Gym.NET's <c>viewer_factory</c> (DR-20) for human mode; <see langword="null"/> uses <c>Gymnasium.HumanViewerFactory</c>.</param>
        public BlackjackEnv(string? renderMode = null, bool natural = false, bool sab = false, IEnvironmentViewerFactoryDelegate? viewerFactory = null) {
            Metadata = ClassMetadata;
            ActionSpace = new Discrete(2);
            ObservationSpace = new Tuple(new Discrete(32), new Discrete(11), new Discrete(2));
            // Flag to payout 1.5 on a "natural" blackjack win, like casino rules
            Natural = natural;
            // Flag for full agreement with the (Sutton and Barto, 2018) definition. Overrides self.natural
            Sab = sab;
            RenderMode = renderMode;
            _window = new HumanWindow("Blackjack", viewerFactory);
        }

        /// <summary>
        ///     Gymnasium's class attribute <c>metadata</c>: <c>{"render_modes": ["human", "rgb_array"], "render_fps": 4}</c>. Every
        ///     env starts with a copy as its instance metadata.
        /// </summary>
        /// <remarks>A new dict on every read, so editing one env's metadata never reaches another (Python's is one shared dict).</remarks>
        public static new Metadata ClassMetadata => new() {
            [Metadata.RenderModesKey] = new List<string> { "human", "rgb_array" },
            [Metadata.RenderFpsKey] = 4L,
        };

        /// <summary>
        ///     Gymnasium's module-level <c>deck</c>: 1 (ace), 2 to 9, and four tens (10, jack, queen, king), drawn with
        ///     replacement, so the deck is infinite.
        /// </summary>
        public static IReadOnlyList<long> Deck { get; } = new long[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10 };

        /// <summary>Gymnasium's <c>natural</c>: whether a winning natural blackjack pays 1.5. Read at every stick.</summary>
        public bool Natural { get; set; }

        /// <summary>Gymnasium's <c>sab</c>: whether Sutton and Barto's rules apply (a natural wins outright). Read at every stick.</summary>
        public bool Sab { get; set; }

        /// <summary>
        ///     Gymnasium's <c>player</c>: the player's cards, <see langword="null"/> before the first reset (Python has no such
        ///     attribute yet). The env appends to this very list; assigning one injects a hand.
        /// </summary>
        public List<long>? Player { get; set; }

        /// <summary>
        ///     Gymnasium's <c>dealer</c>: the dealer's cards, the first one face up; <see langword="null"/> before the first reset.
        ///     The env appends to this very list; assigning one injects a hand.
        /// </summary>
        public List<long>? Dealer { get; set; }

        /// <summary>Gymnasium's <c>dealer_top_card_suit</c>: <c>C</c>, <c>D</c>, <c>H</c> or <c>S</c>, drawn by the reset; only the frame shows it.</summary>
        public string? DealerTopCardSuit { get; set; }

        /// <summary>Gymnasium's <c>dealer_top_card_value_str</c>: <c>A</c>, a digit, or the drawn <c>J</c>/<c>Q</c>/<c>K</c> of a ten; only the frame shows it.</summary>
        public string? DealerTopCardValueStr { get; set; }

        /// <summary>Gymnasium's <c>screen</c>: the frame every render draws on, <see langword="null"/> until the first one (Python has no such attribute yet).</summary>
        public Surface? Screen { get; private set; }

        /// <summary>Gymnasium's <c>clock</c>, <see langword="null"/> until the first render.</summary>
        public Clock? Clock { get; private set; }

        /// <summary>Gymnasium's <c>cmp</c>: <c>float(a &gt; b) - float(a &lt; b)</c>, the game's score of two totals.</summary>
        /// <param name="a">The player's score.</param>
        /// <param name="b">The dealer's score.</param>
        /// <returns>1.0, -1.0 or 0.0 (also for a NaN, which compares neither way).</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L74">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static double Cmp(double a, double b) => (a > b ? 1.0 : 0.0) - (a < b ? 1.0 : 0.0);

        /// <summary>Gymnasium's <c>draw_card</c>: <c>int(np_random.choice(deck))</c>, one draw from the generator.</summary>
        /// <param name="npRandom">The generator.</param>
        /// <returns>The card's value, 1 to 10.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L82">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static long DrawCard(Generator npRandom) {
            // choice over the deck as an array, as NumPy converts the list: the same bounded draw over 13 positions.
            using NDArray deck = np.array(Deck.ToArray());
            using NDArray card = npRandom.choice(deck);
            return card.GetAtIndex<long>(0);
        }

        /// <summary>Gymnasium's <c>draw_hand</c>: two cards, two draws.</summary>
        /// <param name="npRandom">The generator.</param>
        /// <returns>A new list of the two cards, in draw order.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L86">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static List<long> DrawHand(Generator npRandom) => new() { DrawCard(npRandom), DrawCard(npRandom) };

        /// <summary>Gymnasium's <c>usable_ace</c>: 1 when the hand holds an ace that can count as 11 without busting, else 0.</summary>
        /// <param name="hand">The cards.</param>
        /// <returns>1 or 0 (Python's <c>int</c> of the test).</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L90">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static long UsableAce(IReadOnlyList<long> hand) => hand.Contains(1) && hand.Sum() + 10 <= 21 ? 1 : 0;

        /// <summary>Gymnasium's <c>sum_hand</c>: the hand's total, a usable ace counted as 11.</summary>
        /// <param name="hand">The cards.</param>
        /// <returns>The total.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L94">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static long SumHand(IReadOnlyList<long> hand) => UsableAce(hand) != 0 ? hand.Sum() + 10 : hand.Sum();

        /// <summary>Gymnasium's <c>is_bust</c>: whether the hand's total is over 21.</summary>
        /// <param name="hand">The cards.</param>
        /// <returns><see langword="true"/> for a bust.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L100">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static bool IsBust(IReadOnlyList<long> hand) => SumHand(hand) > 21;

        /// <summary>Gymnasium's <c>score</c>: the hand's total, or 0 for a bust.</summary>
        /// <param name="hand">The cards.</param>
        /// <returns>The score.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L104">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static long Score(IReadOnlyList<long> hand) => IsBust(hand) ? 0 : SumHand(hand);

        /// <summary>Gymnasium's <c>is_natural</c>: <c>sorted(hand) == [1, 10]</c>, an ace and a ten as the only two cards.</summary>
        /// <param name="hand">The cards.</param>
        /// <returns><see langword="true"/> for a natural blackjack.</returns>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L108">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static bool IsNatural(IReadOnlyList<long> hand) => hand.Count == 2 && hand.Order().SequenceEqual(new long[] { 1, 10 });

        /// <summary>
        ///     Gymnasium's <c>draw_table</c>: the felt, the dealer's label and face-up value, the face-up card and the hidden one,
        ///     the player's label and total centred in big digits, and the usable-ace label when there is one.
        /// </summary>
        /// <param name="surface">The surface drawn on, at least 384x376 for the whole table.</param>
        /// <param name="playerSum">The player's total.</param>
        /// <param name="dealerCardValue">The dealer's face-up value (an ace is 1).</param>
        /// <param name="usableAce">Non-zero to show the usable-ace label.</param>
        /// <param name="dealerCard">The face-up card's image name, face then suit (<c>"KS"</c>, <c>"AH"</c>, <c>"7C"</c>).</param>
        /// <exception cref="System.IO.FileNotFoundException">
        ///     Thrown for a card no image exists for (a ten named <c>"10S"</c>), or a negative number's minus sign, where pygame
        ///     raises its own <c>FileNotFoundError</c>.
        /// </exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L24">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public static void DrawTable(Surface surface, long playerSum, long dealerCardValue, long usableAce, string dealerCard) {
            surface.Fill(FeltColor);

            Surface dealerLabel = Load("label_dealer");
            surface.Blit(dealerLabel, DealerLabelPos.X, DealerLabelPos.Y);
            BlitDigits(surface, dealerCardValue, "small", SmallDigitGap, DealerLabelPos.X + dealerLabel.Width + LabelDigitGap, DealerLabelPos.Y);
            surface.Blit(Load($"card_{dealerCard}"), DealerCardPos.X, DealerCardPos.Y);
            surface.Blit(Load("card_back"), HiddenCardPos.X, HiddenCardPos.Y);

            surface.Blit(Load("label_player"), PlayerLabelPos.X, PlayerLabelPos.Y);
            string digits = playerSum.ToString(CultureInfo.InvariantCulture);
            int digitWidth = Load("digit_big_0").Width;
            int sumWidth = digits.Length * digitWidth + (digits.Length - 1) * BigDigitGap;
            BlitDigits(surface, playerSum, "big", BigDigitGap, CentredX(sumWidth), PlayerSumTop);

            if (usableAce != 0) {
                Surface usableAceLabel = Load("label_usable_ace");
                surface.Blit(usableAceLabel, CentredX(usableAceLabel.Width), UsableAceTop);
            }
        }

        /// <summary>
        ///     Deals a game: reseeds the generator when a seed is given, draws the dealer's hand, then the player's, then the
        ///     face-up card's suit and, for a ten, its face. Renders in human mode.
        /// </summary>
        /// <param name="seed">The seed of a fresh <c>Generator(PCG64(SeedSequence(seed)))</c>, or <see langword="null"/> to keep the generator.</param>
        /// <param name="options">Ignored, as in Gymnasium.</param>
        /// <returns>The observation (three new 0-d int64 arrays the caller owns) and an empty info.</returns>
        /// <exception cref="Error">Thrown for a negative seed, with Gymnasium's message.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L279">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public override ResetResult<object?[]> Reset(BigInteger? seed = null, IReadOnlyDictionary<string, object?>? options = null) {
            SeedNpRandom(seed);
            Dealer = DrawHand(NpRandom);
            Player = DrawHand(NpRandom);
            long dealerCardValue = ObservationValues().DealerCard;

            // self.np_random.choice(suits): the same bounded draw as a choice over the four positions.
            DealerTopCardSuit = Suits[Choose(Suits.Length)];
            if (dealerCardValue == 1)
                DealerTopCardValueStr = "A";
            else if (dealerCardValue == 10)
                DealerTopCardValueStr = Faces[Choose(Faces.Length)];
            else
                DealerTopCardValueStr = dealerCardValue.ToString(CultureInfo.InvariantCulture);

            if (RenderMode == "human")
                Render();
            return new ResetResult<object?[]>(Observation(), new Info());
        }

        /// <summary>
        ///     Plays the action: a hit (true) draws a card for the player and ends the game on a bust (-1.0); a stick (false)
        ///     draws the dealer's cards while their total is under 17 and scores the game, with <see cref="Sab"/>'s automatic
        ///     natural win and <see cref="Natural"/>'s 1.5. Renders in human mode.
        /// </summary>
        /// <param name="action">0 (stick) or 1 (hit): a 0-d integer array, or through the untyped step a .NET integer or <see langword="bool"/>.</param>
        /// <returns>The observation (three new 0-d int64 arrays the caller owns), the reward, whether the game ended, <c>truncated=False</c>, and an empty info.</returns>
        /// <exception cref="AssertionError">Thrown for an action <c>Discrete(2)</c> does not contain (Gymnasium's bare <c>assert</c>).</exception>
        /// <exception cref="AttributeError">Thrown before the first reset: <c>'BlackjackEnv' object has no attribute 'player'</c> for a hit, <c>'dealer'</c> for a stick.</exception>
        /// <exception cref="IndexError">Thrown when an injected dealer's hand is empty (<c>list index out of range</c>).</exception>
        /// <exception cref="DependencyNotInstalled">Thrown in human mode by the first render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L242">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public override StepResult<object?[]> Step(NDArray action) {
            if (!ActionSpace.Contains(action))
                throw new AssertionError();
            bool terminated;
            double reward;
            // if action: a contained action is a 0-d integer array, true when non-zero.
            if (Discrete.TryToInt64(action, out long hit) && hit != 0) {
                // self.player.append(draw_card(...)): the attribute is looked up before the card is drawn.
                List<long> player = Player ?? throw MissingAttribute("player");
                player.Add(DrawCard(NpRandom));
                if (IsBust(player)) {
                    terminated = true;
                    reward = -1.0;
                } else {
                    terminated = false;
                    reward = 0.0;
                }
            } else {
                // stick: play out the dealers hand, and score
                terminated = true;
                List<long> dealer = Dealer ?? throw MissingAttribute("dealer");
                while (SumHand(dealer) < 17)
                    dealer.Add(DrawCard(NpRandom));
                List<long> player = Player ?? throw MissingAttribute("player");
                reward = Cmp(Score(player), Score(dealer));
                if (Sab && IsNatural(player) && !IsNatural(dealer))
                    // Player automatically wins. Rules consistent with S&B
                    reward = 1.0;
                else if (!Sab && Natural && IsNatural(player) && reward == 1.0)
                    // Natural gives extra points, but doesn't autowin. Legacy implementation
                    reward = 1.5;
            }

            if (RenderMode == "human")
                Render();
            // truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`.
            return new StepResult<object?[]>(Observation(), reward, terminated, false, new Info());
        }

        /// <summary>
        ///     Renders the table (any render mode but <c>human</c> returns the frame); without a render mode, warns as Gymnasium
        ///     does and returns nothing.
        /// </summary>
        /// <returns><c>human</c>: nothing. Any other mode: the <c>(376, 384, 3)</c> uint8 frame, a new array the caller owns.</returns>
        /// <exception cref="AssertionError">Thrown without a render mode when the env has no spec (<c>assert self.spec is not None</c>).</exception>
        /// <exception cref="AttributeError">Thrown before the first reset (no <c>player</c>), after the screen was created; or for hands injected before it (no face-up card's face and suit).</exception>
        /// <exception cref="IndexError">Thrown when an injected dealer's hand is empty.</exception>
        /// <exception cref="System.IO.FileNotFoundException">Thrown for a total or face-up value whose digits have no image (a negative number).</exception>
        /// <exception cref="KeyError">Thrown in human mode when <c>metadata["render_fps"]</c> was removed.</exception>
        /// <exception cref="DependencyNotInstalled">Thrown on the first human render when no viewer factory is set (DR-20).</exception>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L304">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public override RenderResult Render() {
            if (RenderMode is null) {
                EnvSpec spec = Spec ?? throw new AssertionError();
                Logger.Warn("You are calling render method without specifying any render mode. You can specify the render_mode at initialization, "
                            + $"e.g. gym.make(\"{spec.Id}\", render_mode=\"rgb_array\")");
                return RenderResult.None;
            }

            if (Screen is null) {
                if (RenderMode == "human")
                    // pygame.display.set_mode(SCREEN_SIZE)
                    _window.Ensure(ScreenSize.Width, ScreenSize.Height);
                Screen = new Surface(ScreenSize.Width, ScreenSize.Height);
            }

            Clock ??= new Clock();

            (long playerSum, long dealerCardValue, long usableAce) = ObservationValues();
            // self.dealer_top_card_value_str + self.dealer_top_card_suit: both set by every reset, so missing only for a hand
            // injected before the first one.
            string dealerCard = (DealerTopCardValueStr ?? throw MissingAttribute("dealer_top_card_value_str"))
                                + (DealerTopCardSuit ?? throw MissingAttribute("dealer_top_card_suit"));
            DrawTable(Screen, playerSum, dealerCardValue, usableAce, dealerCard);

            if (RenderMode == "human") {
                // pygame.event.pump(); pygame.display.update(); self.clock.tick(self.metadata["render_fps"])
                _window.Show(Screen);
                Clock.Tick(PyBuiltins.Float(Metadata[Metadata.RenderFpsKey]));
                return RenderResult.None;
            }

            return new RenderResult.Rgb(SurfArray.ToRgbArray(Screen));
        }

        /// <summary>Once a screen exists, shuts the human-mode window down (pygame's <c>display.quit()</c>); the screen is kept.</summary>
        /// <remarks>Reference: <see href="https://github.com/Farama-Foundation/Gymnasium/blob/8ba8c3a63bb4e000b7f33923f43d292acac07be1/gymnasium/envs/toy_text/blackjack.py#L349">gymnasium/envs/toy_text/blackjack.py</see>.</remarks>
        public override void Close() {
            if (Screen is not null)
                _window.Close();
        }

        /// <summary>
        ///     The untyped step's action conversion: Gymnasium's <c>assert self.action_space.contains(action)</c> on the action
        ///     as given (a Python <c>True</c> is contained, a 0-d bool array is not), then the base conversion.
        /// </summary>
        /// <param name="action">The action as given to the untyped step.</param>
        /// <returns>The action as a 0-d array: the one given, or a new int64 one the step releases.</returns>
        /// <exception cref="AssertionError">Thrown for anything <c>Discrete(2)</c> does not contain (the bare assertion).</exception>
        protected override NDArray ConvertAction(object? action) {
            if (!ActionSpace.Contains(action))
                throw new AssertionError();
            return base.ConvertAction(action);
        }

        /// <summary>
        ///     Gymnasium's <c>_get_obs</c> as values: <c>_hand_sum_and_usable_ace(self.player)</c>, then <c>self.dealer[0]</c>, in
        ///     Python's order (so before a reset the missing <c>player</c> is reported).
        /// </summary>
        /// <returns>The player's total, the dealer's face-up card, and the usable-ace flag.</returns>
        /// <exception cref="AttributeError">Thrown when <see cref="Player"/> or <see cref="Dealer"/> does not exist yet.</exception>
        /// <exception cref="IndexError">Thrown when the dealer's hand is empty.</exception>
        private (long PlayerSum, long DealerCard, long UsableAce) ObservationValues() {
            List<long> player = Player ?? throw MissingAttribute("player");
            // _hand_sum_and_usable_ace: one pass for the total and the flag.
            long sum = player.Sum();
            (long playerSum, long usable) = player.Contains(1) && sum + 10 <= 21 ? (sum + 10, 1L) : (sum, 0L);
            List<long> dealer = Dealer ?? throw MissingAttribute("dealer");
            if (dealer.Count == 0)
                throw new IndexError("list index out of range");
            return (playerSum, dealer[0], usable);
        }

        /// <summary>The observation: Gymnasium's <c>_get_obs</c> tuple as three new 0-d int64 arrays.</summary>
        /// <returns>The observation; the caller owns its arrays.</returns>
        /// <exception cref="AttributeError">Thrown when <see cref="Player"/> or <see cref="Dealer"/> does not exist yet.</exception>
        /// <exception cref="IndexError">Thrown when the dealer's hand is empty.</exception>
        private object?[] Observation() {
            (long playerSum, long dealerCard, long usableAce) = ObservationValues();
            return new object?[] { NDArray.Scalar<long>(playerSum), NDArray.Scalar<long>(dealerCard), NDArray.Scalar<long>(usableAce) };
        }

        /// <summary><c>self.np_random.choice(items)</c> as a position: the same bounded draw NumPy makes over a list of that length.</summary>
        /// <param name="count">The list's length.</param>
        /// <returns>The chosen position.</returns>
        private int Choose(int count) {
            // int64 positions, as NumPy holds a list's indices; only the length decides the draw.
            using NDArray positions = np.array(Enumerable.Range(0, count).Select(i => (long)i).ToArray());
            using NDArray chosen = NpRandom.choice(positions);
            return (int)chosen.GetAtIndex<long>(0);
        }

        /// <summary>The error Python raises for an attribute the env sets only at its first reset.</summary>
        /// <param name="name">The attribute.</param>
        /// <returns>The error to throw.</returns>
        private static AttributeError MissingAttribute(string name) => new($"'BlackjackEnv' object has no attribute '{name}'");

        /// <summary>Gymnasium's <c>load</c>: an image of <c>blackjack</c>, unscaled.</summary>
        /// <param name="name">The image's name without extension.</param>
        /// <returns>The shared, read-only surface (never drawn on).</returns>
        /// <exception cref="System.IO.FileNotFoundException">Thrown when no image has that name.</exception>
        private static Surface Load(string name) => EnvAssets.Load($"toy_text/blackjack/{name}.png");

        /// <summary>Gymnasium's <c>blit_digits</c>: the number's decimal digits as images, left to right, each followed by the gap.</summary>
        /// <param name="surface">The surface drawn on.</param>
        /// <param name="number">The number (<c>str(int(number))</c>).</param>
        /// <param name="size">The digit set, <c>small</c> or <c>big</c>.</param>
        /// <param name="gap">The pixels between digits.</param>
        /// <param name="x">The first digit's left edge.</param>
        /// <param name="y">The digits' top edge.</param>
        /// <exception cref="System.IO.FileNotFoundException">Thrown for a negative number's minus sign, which has no image.</exception>
        private static void BlitDigits(Surface surface, long number, string size, int gap, int x, int y) {
            foreach (char digit in number.ToString(CultureInfo.InvariantCulture)) {
                Surface img = Load($"digit_{size}_{digit}");
                surface.Blit(img, x, y);
                x += img.Width + gap;
            }
        }

        /// <summary>Gymnasium's <c>centred_x</c>: <c>(SCREEN_SIZE[0] - width) // 2 // ART_PIXEL * ART_PIXEL</c>, floor divisions as Python's.</summary>
        /// <param name="width">The width to centre.</param>
        /// <returns>The left edge, snapped down to the art's 4-pixel grid.</returns>
        private static int CentredX(int width) => (int)(PyMath.FloorDiv(PyMath.FloorDiv(ScreenSize.Width - width, 2), ArtPixel) * ArtPixel);
    }
}
