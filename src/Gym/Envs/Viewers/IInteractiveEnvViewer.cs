using System.Collections.Generic;

// Beside IEnvViewer, in the core assembly and the Gym.Environments namespace its implementations already import (DR-20).
namespace Gym.Environments {
    /// <summary>What a window reported: a key pressed or released, its close button, or a new client size.</summary>
    public enum ViewerInputKind {
        /// <summary>A key went down; <see cref="ViewerInput.Key"/> holds its SDL keycode.</summary>
        KeyDown,

        /// <summary>A key went up; <see cref="ViewerInput.Key"/> holds its SDL keycode.</summary>
        KeyUp,

        /// <summary>The window's close button was pressed; the window stays open (see <see cref="IInteractiveEnvViewer.BeginInput"/>).</summary>
        Close,

        /// <summary>The window's client area changed size; <see cref="ViewerInput.Width"/> and <see cref="ViewerInput.Height"/> hold the new size.</summary>
        Resized,
    }

    /// <summary>
    ///     One input a window reported, in the window's own order. Keys are SDL keycodes (pygame's <c>event.key</c>): a
    ///     printable key is its unshifted character (<c>'a'</c> is 97), any other is its scancode with bit 30 set
    ///     (<c>K_LEFT</c> is 1073741904); <c>Gym.Rendering.KeyCode</c> names them.
    /// </summary>
    /// <param name="Kind">What happened.</param>
    /// <param name="Key">The SDL keycode for <see cref="ViewerInputKind.KeyDown"/> and <see cref="ViewerInputKind.KeyUp"/>, else 0.</param>
    /// <param name="Width">The new client width for <see cref="ViewerInputKind.Resized"/>, else 0.</param>
    /// <param name="Height">The new client height for <see cref="ViewerInputKind.Resized"/>, else 0.</param>
    public readonly record struct ViewerInput(ViewerInputKind Kind, int Key = 0, int Width = 0, int Height = 0) {
        /// <summary>A key press.</summary>
        /// <param name="key">The SDL keycode.</param>
        /// <returns>The input.</returns>
        public static ViewerInput KeyDown(int key) => new(ViewerInputKind.KeyDown, key);

        /// <summary>A key release.</summary>
        /// <param name="key">The SDL keycode.</param>
        /// <returns>The input.</returns>
        public static ViewerInput KeyUp(int key) => new(ViewerInputKind.KeyUp, key);

        /// <summary>A press of the window's close button.</summary>
        /// <returns>The input.</returns>
        public static ViewerInput Close() => new(ViewerInputKind.Close);

        /// <summary>A new client size.</summary>
        /// <param name="width">The new width in pixels.</param>
        /// <param name="height">The new height in pixels.</param>
        /// <returns>The input.</returns>
        public static ViewerInput Resized(int width, int height) => new(ViewerInputKind.Resized, Width: width, Height: height);
    }

    /// <summary>
    ///     A viewer whose window also reports the user's input, as SDL reports a pygame window's: keys, the close button and
    ///     resizes. <c>Gym.Rendering.Display</c> pumps it into pygame's event queue (<c>Gym.Rendering.EventQueue</c>), which
    ///     is how <c>Gym.Utils.Play</c> lets a person play an env from the keyboard.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     A viewer reports nothing until <see cref="BeginInput"/>, so the human-mode windows of envs keep their behaviour: in
    ///     particular their close button keeps closing them. After it, the viewer buffers its input (its UI thread adds,
    ///     the thread that plays takes) and the close button only reports <see cref="ViewerInputKind.Close"/>, as a pygame
    ///     window stays open after its close button until the program quits; <see cref="IEnvViewer.CloseEnvironment"/> still
    ///     closes it.
    ///     </para>
    ///     <para>
    ///     The built-in Avalonia and WinForms viewers implement it; <see cref="NullEnvViewer"/> does not (a headless run is
    ///     driven by events posted to the queue instead, as Gymnasium's own tests post them).
    ///     </para>
    /// </remarks>
    public interface IInteractiveEnvViewer : IEnvViewer {
        /// <summary>
        ///     Starts buffering input; from now on the close button reports <see cref="ViewerInputKind.Close"/> instead of
        ///     closing the window. Calling it again does nothing more.
        /// </summary>
        void BeginInput();

        /// <summary>Takes the input buffered since the last call, oldest first; safe to call from any thread.</summary>
        /// <returns>The inputs, empty when there are none; the caller owns the list.</returns>
        IReadOnlyList<ViewerInput> TakeInput();
    }
}
