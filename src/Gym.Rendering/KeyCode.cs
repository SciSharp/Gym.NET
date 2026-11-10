namespace Gym.Rendering {
    /// <summary>
    ///     The keycodes pygame puts in a key event's <c>key</c> (SDL2's <c>SDL_Keycode</c>), by readable names:
    ///     <see cref="Left"/> is pygame's <c>K_LEFT</c>. Use them in a <c>keys_to_action</c> mapping for
    ///     <see cref="Gym.Utils.Play"/>, and in an <see cref="Gym.Environments.IInteractiveEnvViewer"/> that translates its
    ///     framework's keys.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A key that types a character has that character's code, unshifted (<see cref="A"/> is 97 for <c>a</c> and
    ///     <c>A</c> alike); every other key is its USB scancode with bit 30 set (<see cref="ScancodeMask"/>), so the arrows
    ///     are above a billion. pygame 1's codes are gone: MountainCar's <c>get_keys_to_action</c> still names 275 and 276,
    ///     pygame 1's arrows, which no pygame 2 window produces.
    ///     </para>
    ///     <para>
    ///     The values were read from live pygame-ce 2.5.8 (SDL 2.32.10). The viewers report the keyboard layout's key, as
    ///     pygame does (DR-47): the key labeled A is <see cref="A"/> on any layout; the punctuation keys are named as on a US
    ///     layout.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/constants.c">src_c/constants.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    public static class KeyCode {
        /// <summary>
        ///     The bit SDL sets on the code of a key that types no character (<c>SDLK_SCANCODE_MASK</c>): such a key's code is
        ///     <c>scancode | ScancodeMask</c>.
        /// </summary>
        public const int ScancodeMask = 1 << 30;

        /// <summary>Backspace, pygame's <c>K_BACKSPACE</c> (8).</summary>
        public const int Backspace = 8;

        /// <summary>Tab, pygame's <c>K_TAB</c> (9).</summary>
        public const int Tab = 9;

        /// <summary>Return (Enter on the main keyboard), pygame's <c>K_RETURN</c> (13).</summary>
        public const int Return = 13;

        /// <summary>Escape, which also ends <c>play</c>, pygame's <c>K_ESCAPE</c> (27).</summary>
        public const int Escape = 27;

        /// <summary>The space bar, pygame's <c>K_SPACE</c> (32).</summary>
        public const int Space = 32;

        /// <summary><c>!</c>, pygame's <c>K_EXCLAIM</c> (33).</summary>
        public const int Exclaim = 33;

        /// <summary><c>"</c>, pygame's <c>K_QUOTEDBL</c> (34).</summary>
        public const int QuoteDbl = 34;

        /// <summary><c>#</c>, pygame's <c>K_HASH</c> (35).</summary>
        public const int Hash = 35;

        /// <summary><c>$</c>, pygame's <c>K_DOLLAR</c> (36).</summary>
        public const int Dollar = 36;

        /// <summary><c>%</c>, pygame's <c>K_PERCENT</c> (37).</summary>
        public const int Percent = 37;

        /// <summary><c>&amp;</c>, pygame's <c>K_AMPERSAND</c> (38).</summary>
        public const int Ampersand = 38;

        /// <summary><c>'</c>, pygame's <c>K_QUOTE</c> (39).</summary>
        public const int Quote = 39;

        /// <summary><c>(</c>, pygame's <c>K_LEFTPAREN</c> (40).</summary>
        public const int LeftParen = 40;

        /// <summary><c>)</c>, pygame's <c>K_RIGHTPAREN</c> (41).</summary>
        public const int RightParen = 41;

        /// <summary><c>*</c>, pygame's <c>K_ASTERISK</c> (42).</summary>
        public const int Asterisk = 42;

        /// <summary><c>+</c>, pygame's <c>K_PLUS</c> (43).</summary>
        public const int Plus = 43;

        /// <summary><c>,</c>, pygame's <c>K_COMMA</c> (44).</summary>
        public const int Comma = 44;

        /// <summary><c>-</c>, pygame's <c>K_MINUS</c> (45).</summary>
        public const int Minus = 45;

        /// <summary><c>.</c>, pygame's <c>K_PERIOD</c> (46).</summary>
        public const int Period = 46;

        /// <summary><c>/</c>, pygame's <c>K_SLASH</c> (47).</summary>
        public const int Slash = 47;

        /// <summary>The digit <c>0</c> on the main keyboard, pygame's <c>K_0</c> (48).</summary>
        public const int Digit0 = 48;

        /// <summary>The digit <c>1</c> on the main keyboard, pygame's <c>K_1</c> (49).</summary>
        public const int Digit1 = 49;

        /// <summary>The digit <c>2</c> on the main keyboard, pygame's <c>K_2</c> (50).</summary>
        public const int Digit2 = 50;

        /// <summary>The digit <c>3</c> on the main keyboard, pygame's <c>K_3</c> (51).</summary>
        public const int Digit3 = 51;

        /// <summary>The digit <c>4</c> on the main keyboard, pygame's <c>K_4</c> (52).</summary>
        public const int Digit4 = 52;

        /// <summary>The digit <c>5</c> on the main keyboard, pygame's <c>K_5</c> (53).</summary>
        public const int Digit5 = 53;

        /// <summary>The digit <c>6</c> on the main keyboard, pygame's <c>K_6</c> (54).</summary>
        public const int Digit6 = 54;

        /// <summary>The digit <c>7</c> on the main keyboard, pygame's <c>K_7</c> (55).</summary>
        public const int Digit7 = 55;

        /// <summary>The digit <c>8</c> on the main keyboard, pygame's <c>K_8</c> (56).</summary>
        public const int Digit8 = 56;

        /// <summary>The digit <c>9</c> on the main keyboard, pygame's <c>K_9</c> (57).</summary>
        public const int Digit9 = 57;

        /// <summary><c>:</c>, pygame's <c>K_COLON</c> (58).</summary>
        public const int Colon = 58;

        /// <summary><c>;</c>, pygame's <c>K_SEMICOLON</c> (59).</summary>
        public const int Semicolon = 59;

        /// <summary><c>&lt;</c>, pygame's <c>K_LESS</c> (60).</summary>
        public const int Less = 60;

        /// <summary><c>=</c> (named so it does not hide <c>object.Equals</c>), pygame's <c>K_EQUALS</c> (61).</summary>
        public const int EqualsSign = 61;

        /// <summary><c>&gt;</c>, pygame's <c>K_GREATER</c> (62).</summary>
        public const int Greater = 62;

        /// <summary><c>?</c>, pygame's <c>K_QUESTION</c> (63).</summary>
        public const int Question = 63;

        /// <summary><c>@</c>, pygame's <c>K_AT</c> (64).</summary>
        public const int At = 64;

        /// <summary><c>[</c>, pygame's <c>K_LEFTBRACKET</c> (91).</summary>
        public const int LeftBracket = 91;

        /// <summary><c>\</c>, pygame's <c>K_BACKSLASH</c> (92).</summary>
        public const int Backslash = 92;

        /// <summary><c>]</c>, pygame's <c>K_RIGHTBRACKET</c> (93).</summary>
        public const int RightBracket = 93;

        /// <summary><c>^</c>, pygame's <c>K_CARET</c> (94).</summary>
        public const int Caret = 94;

        /// <summary><c>_</c>, pygame's <c>K_UNDERSCORE</c> (95).</summary>
        public const int Underscore = 95;

        /// <summary><c>`</c>, pygame's <c>K_BACKQUOTE</c> (96).</summary>
        public const int Backquote = 96;

        /// <summary>The letter key <c>a</c> (lower case whatever the shift state), pygame's <c>K_a</c> (97).</summary>
        public const int A = 97;

        /// <summary>The letter key <c>b</c> (lower case whatever the shift state), pygame's <c>K_b</c> (98).</summary>
        public const int B = 98;

        /// <summary>The letter key <c>c</c> (lower case whatever the shift state), pygame's <c>K_c</c> (99).</summary>
        public const int C = 99;

        /// <summary>The letter key <c>d</c> (lower case whatever the shift state), pygame's <c>K_d</c> (100).</summary>
        public const int D = 100;

        /// <summary>The letter key <c>e</c> (lower case whatever the shift state), pygame's <c>K_e</c> (101).</summary>
        public const int E = 101;

        /// <summary>The letter key <c>f</c> (lower case whatever the shift state), pygame's <c>K_f</c> (102).</summary>
        public const int F = 102;

        /// <summary>The letter key <c>g</c> (lower case whatever the shift state), pygame's <c>K_g</c> (103).</summary>
        public const int G = 103;

        /// <summary>The letter key <c>h</c> (lower case whatever the shift state), pygame's <c>K_h</c> (104).</summary>
        public const int H = 104;

        /// <summary>The letter key <c>i</c> (lower case whatever the shift state), pygame's <c>K_i</c> (105).</summary>
        public const int I = 105;

        /// <summary>The letter key <c>j</c> (lower case whatever the shift state), pygame's <c>K_j</c> (106).</summary>
        public const int J = 106;

        /// <summary>The letter key <c>k</c> (lower case whatever the shift state), pygame's <c>K_k</c> (107).</summary>
        public const int K = 107;

        /// <summary>The letter key <c>l</c> (lower case whatever the shift state), pygame's <c>K_l</c> (108).</summary>
        public const int L = 108;

        /// <summary>The letter key <c>m</c> (lower case whatever the shift state), pygame's <c>K_m</c> (109).</summary>
        public const int M = 109;

        /// <summary>The letter key <c>n</c> (lower case whatever the shift state), pygame's <c>K_n</c> (110).</summary>
        public const int N = 110;

        /// <summary>The letter key <c>o</c> (lower case whatever the shift state), pygame's <c>K_o</c> (111).</summary>
        public const int O = 111;

        /// <summary>The letter key <c>p</c> (lower case whatever the shift state), pygame's <c>K_p</c> (112).</summary>
        public const int P = 112;

        /// <summary>The letter key <c>q</c> (lower case whatever the shift state), pygame's <c>K_q</c> (113).</summary>
        public const int Q = 113;

        /// <summary>The letter key <c>r</c> (lower case whatever the shift state), pygame's <c>K_r</c> (114).</summary>
        public const int R = 114;

        /// <summary>The letter key <c>s</c> (lower case whatever the shift state), pygame's <c>K_s</c> (115).</summary>
        public const int S = 115;

        /// <summary>The letter key <c>t</c> (lower case whatever the shift state), pygame's <c>K_t</c> (116).</summary>
        public const int T = 116;

        /// <summary>The letter key <c>u</c> (lower case whatever the shift state), pygame's <c>K_u</c> (117).</summary>
        public const int U = 117;

        /// <summary>The letter key <c>v</c> (lower case whatever the shift state), pygame's <c>K_v</c> (118).</summary>
        public const int V = 118;

        /// <summary>The letter key <c>w</c> (lower case whatever the shift state), pygame's <c>K_w</c> (119).</summary>
        public const int W = 119;

        /// <summary>The letter key <c>x</c> (lower case whatever the shift state), pygame's <c>K_x</c> (120).</summary>
        public const int X = 120;

        /// <summary>The letter key <c>y</c> (lower case whatever the shift state), pygame's <c>K_y</c> (121).</summary>
        public const int Y = 121;

        /// <summary>The letter key <c>z</c> (lower case whatever the shift state), pygame's <c>K_z</c> (122).</summary>
        public const int Z = 122;

        /// <summary>Delete, pygame's <c>K_DELETE</c> (127).</summary>
        public const int Delete = 127;

        /// <summary>Caps Lock, pygame's <c>K_CAPSLOCK</c>: scancode 57 with bit 30 set (1073741881).</summary>
        public const int CapsLock = 1073741881;

        /// <summary>The function key F1, pygame's <c>K_F1</c>: scancode 58 with bit 30 set (1073741882).</summary>
        public const int F1 = 1073741882;

        /// <summary>The function key F2, pygame's <c>K_F2</c>: scancode 59 with bit 30 set (1073741883).</summary>
        public const int F2 = 1073741883;

        /// <summary>The function key F3, pygame's <c>K_F3</c>: scancode 60 with bit 30 set (1073741884).</summary>
        public const int F3 = 1073741884;

        /// <summary>The function key F4, pygame's <c>K_F4</c>: scancode 61 with bit 30 set (1073741885).</summary>
        public const int F4 = 1073741885;

        /// <summary>The function key F5, pygame's <c>K_F5</c>: scancode 62 with bit 30 set (1073741886).</summary>
        public const int F5 = 1073741886;

        /// <summary>The function key F6, pygame's <c>K_F6</c>: scancode 63 with bit 30 set (1073741887).</summary>
        public const int F6 = 1073741887;

        /// <summary>The function key F7, pygame's <c>K_F7</c>: scancode 64 with bit 30 set (1073741888).</summary>
        public const int F7 = 1073741888;

        /// <summary>The function key F8, pygame's <c>K_F8</c>: scancode 65 with bit 30 set (1073741889).</summary>
        public const int F8 = 1073741889;

        /// <summary>The function key F9, pygame's <c>K_F9</c>: scancode 66 with bit 30 set (1073741890).</summary>
        public const int F9 = 1073741890;

        /// <summary>The function key F10, pygame's <c>K_F10</c>: scancode 67 with bit 30 set (1073741891).</summary>
        public const int F10 = 1073741891;

        /// <summary>The function key F11, pygame's <c>K_F11</c>: scancode 68 with bit 30 set (1073741892).</summary>
        public const int F11 = 1073741892;

        /// <summary>The function key F12, pygame's <c>K_F12</c>: scancode 69 with bit 30 set (1073741893).</summary>
        public const int F12 = 1073741893;

        /// <summary>The function key F13, pygame's <c>K_F13</c>: scancode 104 with bit 30 set (1073741928).</summary>
        public const int F13 = 1073741928;

        /// <summary>The function key F14, pygame's <c>K_F14</c>: scancode 105 with bit 30 set (1073741929).</summary>
        public const int F14 = 1073741929;

        /// <summary>The function key F15, pygame's <c>K_F15</c>: scancode 106 with bit 30 set (1073741930).</summary>
        public const int F15 = 1073741930;

        /// <summary>Print Screen, pygame's <c>K_PRINTSCREEN</c>: scancode 70 with bit 30 set (1073741894).</summary>
        public const int PrintScreen = 1073741894;

        /// <summary>Scroll Lock, pygame's <c>K_SCROLLLOCK</c>: scancode 71 with bit 30 set (1073741895).</summary>
        public const int ScrollLock = 1073741895;

        /// <summary>Pause (Break), pygame's <c>K_PAUSE</c>: scancode 72 with bit 30 set (1073741896).</summary>
        public const int Pause = 1073741896;

        /// <summary>Insert, pygame's <c>K_INSERT</c>: scancode 73 with bit 30 set (1073741897).</summary>
        public const int Insert = 1073741897;

        /// <summary>Home, pygame's <c>K_HOME</c>: scancode 74 with bit 30 set (1073741898).</summary>
        public const int Home = 1073741898;

        /// <summary>Page Up, pygame's <c>K_PAGEUP</c>: scancode 75 with bit 30 set (1073741899).</summary>
        public const int PageUp = 1073741899;

        /// <summary>End, pygame's <c>K_END</c>: scancode 77 with bit 30 set (1073741901).</summary>
        public const int End = 1073741901;

        /// <summary>Page Down, pygame's <c>K_PAGEDOWN</c>: scancode 78 with bit 30 set (1073741902).</summary>
        public const int PageDown = 1073741902;

        /// <summary>The right arrow, pygame's <c>K_RIGHT</c>: scancode 79 with bit 30 set (1073741903).</summary>
        public const int Right = 1073741903;

        /// <summary>The left arrow, pygame's <c>K_LEFT</c>: scancode 80 with bit 30 set (1073741904).</summary>
        public const int Left = 1073741904;

        /// <summary>The down arrow, pygame's <c>K_DOWN</c>: scancode 81 with bit 30 set (1073741905).</summary>
        public const int Down = 1073741905;

        /// <summary>The up arrow, pygame's <c>K_UP</c>: scancode 82 with bit 30 set (1073741906).</summary>
        public const int Up = 1073741906;

        /// <summary>Num Lock, pygame's <c>K_NUMLOCK</c>: scancode 83 with bit 30 set (1073741907).</summary>
        public const int NumLock = 1073741907;

        /// <summary>The keypad's <c>/</c>, pygame's <c>K_KP_DIVIDE</c>: scancode 84 with bit 30 set (1073741908).</summary>
        public const int KeypadDivide = 1073741908;

        /// <summary>The keypad's <c>*</c>, pygame's <c>K_KP_MULTIPLY</c>: scancode 85 with bit 30 set (1073741909).</summary>
        public const int KeypadMultiply = 1073741909;

        /// <summary>The keypad's <c>-</c>, pygame's <c>K_KP_MINUS</c>: scancode 86 with bit 30 set (1073741910).</summary>
        public const int KeypadMinus = 1073741910;

        /// <summary>The keypad's <c>+</c>, pygame's <c>K_KP_PLUS</c>: scancode 87 with bit 30 set (1073741911).</summary>
        public const int KeypadPlus = 1073741911;

        /// <summary>The keypad's Enter, pygame's <c>K_KP_ENTER</c>: scancode 88 with bit 30 set (1073741912).</summary>
        public const int KeypadEnter = 1073741912;

        /// <summary>The keypad's <c>1</c> (with Num Lock on), pygame's <c>K_KP_1</c>: scancode 89 with bit 30 set (1073741913).</summary>
        public const int Keypad1 = 1073741913;

        /// <summary>The keypad's <c>2</c> (with Num Lock on), pygame's <c>K_KP_2</c>: scancode 90 with bit 30 set (1073741914).</summary>
        public const int Keypad2 = 1073741914;

        /// <summary>The keypad's <c>3</c> (with Num Lock on), pygame's <c>K_KP_3</c>: scancode 91 with bit 30 set (1073741915).</summary>
        public const int Keypad3 = 1073741915;

        /// <summary>The keypad's <c>4</c> (with Num Lock on), pygame's <c>K_KP_4</c>: scancode 92 with bit 30 set (1073741916).</summary>
        public const int Keypad4 = 1073741916;

        /// <summary>The keypad's <c>5</c> (with Num Lock on), pygame's <c>K_KP_5</c>: scancode 93 with bit 30 set (1073741917).</summary>
        public const int Keypad5 = 1073741917;

        /// <summary>The keypad's <c>6</c> (with Num Lock on), pygame's <c>K_KP_6</c>: scancode 94 with bit 30 set (1073741918).</summary>
        public const int Keypad6 = 1073741918;

        /// <summary>The keypad's <c>7</c> (with Num Lock on), pygame's <c>K_KP_7</c>: scancode 95 with bit 30 set (1073741919).</summary>
        public const int Keypad7 = 1073741919;

        /// <summary>The keypad's <c>8</c> (with Num Lock on), pygame's <c>K_KP_8</c>: scancode 96 with bit 30 set (1073741920).</summary>
        public const int Keypad8 = 1073741920;

        /// <summary>The keypad's <c>9</c> (with Num Lock on), pygame's <c>K_KP_9</c>: scancode 97 with bit 30 set (1073741921).</summary>
        public const int Keypad9 = 1073741921;

        /// <summary>The keypad's <c>0</c> (with Num Lock on), pygame's <c>K_KP_0</c>: scancode 98 with bit 30 set (1073741922).</summary>
        public const int Keypad0 = 1073741922;

        /// <summary>The keypad's <c>.</c>, pygame's <c>K_KP_PERIOD</c>: scancode 99 with bit 30 set (1073741923).</summary>
        public const int KeypadPeriod = 1073741923;

        /// <summary>The keypad's <c>=</c>, pygame's <c>K_KP_EQUALS</c>: scancode 103 with bit 30 set (1073741927).</summary>
        public const int KeypadEquals = 1073741927;

        /// <summary>The context-menu key, pygame's <c>K_MENU</c>: scancode 118 with bit 30 set (1073741942).</summary>
        public const int Menu = 1073741942;

        /// <summary>The left Ctrl, pygame's <c>K_LCTRL</c>: scancode 224 with bit 30 set (1073742048).</summary>
        public const int LeftCtrl = 1073742048;

        /// <summary>The left Shift, pygame's <c>K_LSHIFT</c>: scancode 225 with bit 30 set (1073742049).</summary>
        public const int LeftShift = 1073742049;

        /// <summary>The left Alt, pygame's <c>K_LALT</c>: scancode 226 with bit 30 set (1073742050).</summary>
        public const int LeftAlt = 1073742050;

        /// <summary>The left Windows (Command, Super) key, pygame's <c>K_LGUI</c>: scancode 227 with bit 30 set (1073742051).</summary>
        public const int LeftGui = 1073742051;

        /// <summary>The right Ctrl, pygame's <c>K_RCTRL</c>: scancode 228 with bit 30 set (1073742052).</summary>
        public const int RightCtrl = 1073742052;

        /// <summary>The right Shift, pygame's <c>K_RSHIFT</c>: scancode 229 with bit 30 set (1073742053).</summary>
        public const int RightShift = 1073742053;

        /// <summary>The right Alt (AltGr), pygame's <c>K_RALT</c>: scancode 230 with bit 30 set (1073742054).</summary>
        public const int RightAlt = 1073742054;

        /// <summary>The right Windows (Command, Super) key, pygame's <c>K_RGUI</c>: scancode 231 with bit 30 set (1073742055).</summary>
        public const int RightGui = 1073742055;
    }
}
