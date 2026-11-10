#nullable enable
using System.Collections.Generic;
using System.Reflection;
using System.Windows.Forms;
using Avalonia.Input;
using Gym.Rendering;
using Gym.Rendering.Avalonia;
using Gym.Rendering.WinForm;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Gym.Tests.Rendering {
    /// <summary>
    ///     DR-47: both viewers map the keyboard layout's key (Avalonia's <see cref="Key"/>, Windows' <see cref="Keys"/>) to
    ///     SDL keycodes, as pygame reports keys. Read from each viewer's map by reflection, so no window opens: the maps are
    ///     built when the viewer type is first touched, and a wrong key name would throw there, at the first key a player
    ///     presses.
    /// </summary>
    [TestClass]
    public class ViewerKeyMapTests {
        /// <summary>Avalonia's map is keyed by the layout key (not <see cref="PhysicalKey"/>) and names letters, digits, arrows and the keypad as SDL does.</summary>
        /// <exception cref="AssertFailedException">Thrown when the map is keyed otherwise or a code differs from SDL's.</exception>
        [TestMethod]
        public void AvaloniaViewer_MapsLayoutKeysToSdlKeycodes() {
            var map = ReadMap<Key>(typeof(AvaloniaEnvViewer));
            Assert.AreEqual(KeyCode.A, map[Key.A]);
            Assert.AreEqual(97, map[Key.A], "the key labeled A is K_a");
            Assert.AreEqual(KeyCode.Z, map[Key.Z]);
            Assert.AreEqual(KeyCode.Digit1, map[Key.D1]);
            Assert.AreEqual(KeyCode.Left, map[Key.Left]);
            Assert.AreEqual(KeyCode.Escape, map[Key.Escape]);
            Assert.AreEqual(KeyCode.Return, map[Key.Enter]);
            Assert.AreEqual(KeyCode.Keypad1, map[Key.NumPad1]);
            Assert.AreEqual(KeyCode.Keypad0, map[Key.NumPad0]);
            Assert.AreEqual(KeyCode.F13, map[Key.F13]);
            Assert.AreEqual(KeyCode.Slash, map[Key.OemQuestion]);
        }

        /// <summary>WinForms' map is keyed by Windows' virtual key, which follows the layout for letters, with the same SDL codes.</summary>
        /// <exception cref="AssertFailedException">Thrown when a code differs from SDL's.</exception>
        [TestMethod]
        public void WinFormsViewer_MapsVirtualKeysToSdlKeycodes() {
            var map = ReadMap<Keys>(typeof(WinFormEnvViewer));
            Assert.AreEqual(97, map[Keys.A], "the key labeled A is K_a");
            Assert.AreEqual(KeyCode.Digit1, map[Keys.D1]);
            Assert.AreEqual(KeyCode.Left, map[Keys.Left]);
            Assert.AreEqual(KeyCode.LeftShift, map[Keys.ShiftKey], "Shift without its side is the left one");
        }

        /// <summary>A viewer's private static key map, which reading builds.</summary>
        /// <typeparam name="TKey">The map's key type.</typeparam>
        /// <param name="viewer">The viewer type.</param>
        /// <returns>The map.</returns>
        /// <exception cref="AssertFailedException">Thrown when the viewer has no such map of that key type.</exception>
        private static IReadOnlyDictionary<TKey, int> ReadMap<TKey>(System.Type viewer) where TKey : notnull {
            FieldInfo? field = viewer.GetField("SdlKeys", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.IsNotNull(field, $"{viewer.Name} has no SdlKeys map");
            var map = field!.GetValue(null) as Dictionary<TKey, int>;
            Assert.IsNotNull(map, $"{viewer.Name}'s map is not keyed by {typeof(TKey).Name}");
            return map!;
        }
    }
}
