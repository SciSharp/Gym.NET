using System;
using System.Collections.Generic;
using Gym.Envs;
using Gym.Utils;
using NumSharp;

namespace Gym.Rendering {
    /// <summary>
    ///     pygame's event types (<c>pygame.QUIT</c>, <c>pygame.KEYDOWN</c>, ...): the ones a Gym.NET window posts and
    ///     <see cref="Gym.Utils.Play"/> reads, with pygame-ce 2.5.8's values, and <see cref="Name"/> for every type pygame
    ///     names.
    /// </summary>
    /// <remarks>
    ///     A type outside this list can still be posted (any int is accepted, as by pygame for user events); it prints by
    ///     <see cref="Name"/>, which knows every type pygame-ce 2.5.8 knows.
    ///     Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/event.c">src_c/event.c</see> (pygame-ce 2.5.8).
    /// </remarks>
    public static class EventType {
        /// <summary><c>NOEVENT</c> (0): what an empty queue's <c>poll</c> returns; an event of this type is falsy in pygame.</summary>
        public const int NoEvent = 0;

        /// <summary>
        ///     <c>QUIT</c> (256): the program should end. A window's close button posts it (after <see cref="WindowClose"/>);
        ///     <c>play</c> stops at it.
        /// </summary>
        public const int Quit = 256;

        /// <summary><c>KEYDOWN</c> (768): a key went down; its <c>key</c> attribute is the <see cref="KeyCode"/>.</summary>
        public const int KeyDown = 768;

        /// <summary><c>KEYUP</c> (769): a key went up; its <c>key</c> attribute is the <see cref="KeyCode"/>.</summary>
        public const int KeyUp = 769;

        /// <summary>
        ///     <c>VIDEORESIZE</c> (32769): pygame 1's resize event, still posted with <c>size</c>, <c>w</c> and <c>h</c> when the
        ///     user resizes a window; <c>play</c> ignores it and reads <see cref="WindowResized"/>.
        /// </summary>
        public const int VideoResize = 32769;

        /// <summary>
        ///     <c>WINDOWRESIZED</c> (32777): the user resized the window to <c>x</c> by <c>y</c>; by the time it is read, the
        ///     display surface already has that size. <c>play</c> rescales its frames to fit.
        /// </summary>
        public const int WindowResized = 32777;

        /// <summary><c>WINDOWSIZECHANGED</c> (32778): the window's size changed, for any reason; follows <see cref="WindowResized"/> after a user resize.</summary>
        public const int WindowSizeChanged = 32778;

        /// <summary><c>WINDOWCLOSE</c> (32786): the window's close button was pressed; <see cref="Quit"/> follows it.</summary>
        public const int WindowClose = 32786;

        /// <summary><c>USEREVENT</c> (32865): the first type free for a program's own events, up to <see cref="NumEvents"/> (excluded).</summary>
        public const int UserEvent = 32865;

        /// <summary><c>NUMEVENTS</c> (65535): one past the last user event type.</summary>
        public const int NumEvents = 65535;

        /// <summary>
        ///     pygame's <c>pygame.event.event_name(type)</c>: the name an event's <c>repr</c> shows, <c>KeyDown</c> for 768,
        ///     <c>UserEvent</c> for a type in [<see cref="UserEvent"/>, <see cref="NumEvents"/>), else <c>Unknown</c>.
        /// </summary>
        /// <param name="type">The event type.</param>
        /// <returns>The name.</returns>
        public static string Name(int type) {
            if (Names.TryGetValue(type, out string? name))
                return name;
            return type >= UserEvent && type < NumEvents ? "UserEvent" : "Unknown";
        }

        /// <summary>Every event type pygame-ce 2.5.8 names, read from <c>event_name</c> over the whole 16-bit range.</summary>
        private static readonly Dictionary<int, string> Names = new() {
            [0] = "NoEvent", [256] = "Quit", [257] = "AppTerminating", [258] = "AppLowMemory", [259] = "AppWillEnterBackground",
            [260] = "AppDidEnterBackground", [261] = "AppWillEnterForeground", [262] = "AppDidEnterForeground", [263] = "LocaleChanged",
            [513] = "SysWMEvent", [768] = "KeyDown", [769] = "KeyUp", [770] = "TextEditing", [771] = "TextInput", [772] = "KeyMapChanged",
            [1024] = "MouseMotion", [1025] = "MouseButtonDown", [1026] = "MouseButtonUp", [1027] = "MouseWheel",
            [1536] = "JoyAxisMotion", [1537] = "JoyBallMotion", [1538] = "JoyHatMotion", [1539] = "JoyButtonDown", [1540] = "JoyButtonUp",
            [1541] = "JoyDeviceAdded", [1542] = "JoyDeviceRemoved", [1616] = "ControllerAxisMotion", [1617] = "ControllerButtonDown",
            [1618] = "ControllerButtonUp", [1619] = "ControllerDeviceAdded", [1620] = "ControllerDeviceRemoved",
            [1621] = "ControllerDeviceMapped", [1622] = "ControllerTouchpadDown", [1623] = "ControllerTouchpadMotion",
            [1624] = "ControllerTouchpadUp", [1625] = "ControllerSensorUpdate", [1792] = "FingerDown", [1793] = "FingerUp",
            [1794] = "FingerMotion", [2050] = "MultiGesture", [2304] = "ClipboardUpdate", [4096] = "DropFile", [4097] = "DropText",
            [4098] = "DropBegin", [4099] = "DropComplete", [4352] = "AudioDeviceAdded", [4353] = "AudioDeviceRemoved",
            [8192] = "RenderTargetsReset", [8193] = "RenderDeviceReset", [32768] = "ActiveEvent", [32769] = "VideoResize",
            [32770] = "VideoExpose", [32771] = "MidiIn", [32772] = "MidiOut", [32773] = "WindowShown", [32774] = "WindowHidden",
            [32775] = "WindowExposed", [32776] = "WindowMoved", [32777] = "WindowResized", [32778] = "WindowSizeChanged",
            [32779] = "WindowMinimized", [32780] = "WindowMaximized", [32781] = "WindowRestored", [32782] = "WindowEnter",
            [32783] = "WindowLeave", [32784] = "WindowFocusGained", [32785] = "WindowFocusLost", [32786] = "WindowClose",
            [32787] = "WindowTakeFocus", [32788] = "WindowHitTest", [32789] = "WindowICCProfChanged", [32790] = "WindowDisplayChanged",
        };
    }

    /// <summary>
    ///     An event, <c>pygame.event.Event(type, dict)</c>: a type and a dict of attributes (<c>key</c> for a key event,
    ///     <c>x</c> and <c>y</c> for a resize), read by Python name through <see cref="PyAttributes"/> or <see cref="Dict"/>.
    ///     Post one to <see cref="EventQueue"/> to drive <see cref="Gym.Utils.Play"/> without a keyboard, as Gymnasium's
    ///     tests do.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     Like pygame's, an event is mutable (attributes can be added or replaced through <see cref="PyAttributes.SetAttr"/>
    ///     or <see cref="Dict"/>), equal to another of the same type and an equal dict, and printed as
    ///     <c>&lt;Event(768-KeyDown {'key': 97})&gt;</c>. Reading an attribute it lacks raises pygame's
    ///     <c>'pygame.event.Event' object has no attribute 'x'</c>. Its hash is its type only, since the dict can change.
    ///     </para>
    ///     <para>Reference: <see href="https://github.com/pygame-community/pygame-ce/blob/2.5.8/src_c/event.c">src_c/event.c</see> (pygame-ce 2.5.8).</para>
    /// </remarks>
    [PyName("pygame.event.Event")]
    public sealed class Event : IDynamicAttributes, IEquatable<Event> {
        /// <summary>An event of <paramref name="type"/> with the attributes of <paramref name="attributes"/>, pygame's <c>Event(type, dict)</c> or <c>Event(type, **kwargs)</c>.</summary>
        /// <param name="type">The event type, an <see cref="EventType"/> constant or a user type.</param>
        /// <param name="attributes">
        ///     The attributes by Python name: a dictionary, or an anonymous object (<c>new { key = 97 }</c>), read by
        ///     <see cref="PyKwargs.From"/>; <see langword="null"/> for none. The values are shared, not copied.
        /// </param>
        /// <exception cref="ValueError">Thrown when the attributes name <c>type</c>, with pygame's message: <c>redundant type field in event dict</c>.</exception>
        /// <exception cref="TypeError">Thrown when <paramref name="attributes"/> is no mapping, or has a key that is not a string.</exception>
        public Event(int type, object? attributes = null) {
            Info dict = PyKwargs.From(attributes);
            if (dict.ContainsKey("type"))
                throw new ValueError("redundant type field in event dict");
            Type = type;
            Dict = dict;
        }

        /// <summary>The event type (<c>event.type</c>); fixed at construction.</summary>
        public int Type { get; }

        /// <summary>The attributes (<c>event.dict</c>), in insertion order; changing it changes the event.</summary>
        public Info Dict { get; }

        /// <summary>The attributes, for <see cref="PyAttributes"/>: <c>event.key</c> reads <c>Dict["key"]</c>.</summary>
        IDictionary<string, object?> IDynamicAttributes.DynamicAttributes => Dict;

        /// <summary>Python's truth of an event: <see langword="false"/> only for <see cref="EventType.NoEvent"/>.</summary>
        public bool IsTruthy => Type != EventType.NoEvent;

        /// <summary>pygame's <c>==</c>: the same type and equal dicts (Python's <c>==</c> of the values).</summary>
        /// <param name="other">The other event.</param>
        /// <returns><see langword="true"/> when both are equal.</returns>
        public bool Equals(Event? other) => other is not null && Type == other.Type && PyEquality.AreEqual(Dict, other.Dict);

        /// <summary>pygame's <c>==</c> against any object: only an equal <see cref="Event"/> is equal.</summary>
        /// <param name="obj">The other object.</param>
        /// <returns><see langword="true"/> when <paramref name="obj"/> is an equal event.</returns>
        public override bool Equals(object? obj) => Equals(obj as Event);

        /// <summary>The type, the only part that cannot change after construction (pygame's events are unhashable).</summary>
        /// <returns>The hash.</returns>
        public override int GetHashCode() => Type;

        /// <summary>pygame's <c>repr</c>: <c>&lt;Event(768-KeyDown {'key': 97})&gt;</c>.</summary>
        /// <returns>The text.</returns>
        public override string ToString() => $"<Event({Type}-{EventType.Name(Type)} {PyBuiltins.Repr(Dict)})>";
    }
}
