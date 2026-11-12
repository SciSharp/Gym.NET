using System.Collections.Generic;

// Beside IEnvViewer, in the core assembly and the Gym.Environments namespace its implementations already import (DR-20).
namespace Gym.Environments {
    /// <summary>
    ///     What a window reported: a key pressed or released, its close button, a new client size, or the mouse (a button
    ///     pressed or released, the pointer moved, the wheel turned).
    /// </summary>
    public enum ViewerInputKind {
        /// <summary>A key went down; <see cref="ViewerInput.Key"/> holds its SDL keycode.</summary>
        KeyDown,

        /// <summary>A key went up; <see cref="ViewerInput.Key"/> holds its SDL keycode.</summary>
        KeyUp,

        /// <summary>The window's close button was pressed; the window stays open (see <see cref="IInteractiveEnvViewer.BeginInput"/>).</summary>
        Close,

        /// <summary>The window's client area changed size; <see cref="ViewerInput.Width"/> and <see cref="ViewerInput.Height"/> hold the new size.</summary>
        Resized,

        /// <summary>
        ///     A mouse button went down over the frame; <see cref="ViewerInput.Button"/> names it and <see cref="ViewerInput.X"/>,
        ///     <see cref="ViewerInput.Y"/> hold the pointer.
        /// </summary>
        MouseDown,

        /// <summary>A mouse button went up; <see cref="ViewerInput.Button"/> names it, <see cref="ViewerInput.X"/> and <see cref="ViewerInput.Y"/> hold the pointer.</summary>
        MouseUp,

        /// <summary>The pointer moved over the frame (or, a button held, anywhere); <see cref="ViewerInput.X"/> and <see cref="ViewerInput.Y"/> hold it.</summary>
        MouseMove,

        /// <summary>
        ///     The wheel turned; <see cref="ViewerInput.ScrollX"/> and <see cref="ViewerInput.ScrollY"/> hold the turn in notches,
        ///     <see cref="ViewerInput.X"/> and <see cref="ViewerInput.Y"/> the pointer.
        /// </summary>
        Scroll,
    }

    /// <summary>
    ///     One input a window reported, in the window's own order. Keys are SDL keycodes (pygame's <c>event.key</c>): a
    ///     printable key is its unshifted character (<c>'a'</c> is 97), any other is its scancode with bit 30 set
    ///     (<c>K_LEFT</c> is 1073741904); <c>Gym.Rendering.KeyCode</c> names them. The mouse is reported as GLFW reports it
    ///     to MuJoCo's viewer (DR-70): the pointer in the client area's units from its top-left corner, buttons numbered as
    ///     GLFW's (<see cref="LeftButton"/>, <see cref="RightButton"/>, <see cref="MiddleButton"/>), the wheel in notches with
    ///     a positive <see cref="ScrollY"/> away from the user.
    /// </summary>
    /// <param name="Kind">What happened.</param>
    /// <param name="Key">The SDL keycode for <see cref="ViewerInputKind.KeyDown"/> and <see cref="ViewerInputKind.KeyUp"/>, else 0.</param>
    /// <param name="Width">The new client width for <see cref="ViewerInputKind.Resized"/>, else 0.</param>
    /// <param name="Height">The new client height for <see cref="ViewerInputKind.Resized"/>, else 0.</param>
    /// <param name="X">The pointer's x for the mouse kinds, from the client area's left edge, else 0.</param>
    /// <param name="Y">The pointer's y for the mouse kinds, from the client area's top edge, else 0.</param>
    /// <param name="Button">The button for <see cref="ViewerInputKind.MouseDown"/> and <see cref="ViewerInputKind.MouseUp"/>, else 0.</param>
    /// <param name="ScrollX">The horizontal turn for <see cref="ViewerInputKind.Scroll"/>, in notches, else 0.</param>
    /// <param name="ScrollY">The vertical turn for <see cref="ViewerInputKind.Scroll"/>, in notches (positive away from the user), else 0.</param>
    public readonly record struct ViewerInput(ViewerInputKind Kind, int Key = 0, int Width = 0, int Height = 0, double X = 0, double Y = 0,
                                              int Button = 0, double ScrollX = 0, double ScrollY = 0) {
        /// <summary>The left mouse button (GLFW's <c>MOUSE_BUTTON_LEFT</c>).</summary>
        public const int LeftButton = 0;

        /// <summary>The right mouse button (GLFW's <c>MOUSE_BUTTON_RIGHT</c>).</summary>
        public const int RightButton = 1;

        /// <summary>The middle mouse button (GLFW's <c>MOUSE_BUTTON_MIDDLE</c>).</summary>
        public const int MiddleButton = 2;

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

        /// <summary>A mouse button press.</summary>
        /// <param name="button"><see cref="LeftButton"/>, <see cref="RightButton"/> or <see cref="MiddleButton"/>.</param>
        /// <param name="x">The pointer's x.</param>
        /// <param name="y">The pointer's y.</param>
        /// <returns>The input.</returns>
        public static ViewerInput MouseDown(int button, double x, double y) => new(ViewerInputKind.MouseDown, X: x, Y: y, Button: button);

        /// <summary>A mouse button release.</summary>
        /// <param name="button"><see cref="LeftButton"/>, <see cref="RightButton"/> or <see cref="MiddleButton"/>.</param>
        /// <param name="x">The pointer's x.</param>
        /// <param name="y">The pointer's y.</param>
        /// <returns>The input.</returns>
        public static ViewerInput MouseUp(int button, double x, double y) => new(ViewerInputKind.MouseUp, X: x, Y: y, Button: button);

        /// <summary>A pointer move.</summary>
        /// <param name="x">The pointer's new x.</param>
        /// <param name="y">The pointer's new y.</param>
        /// <returns>The input.</returns>
        public static ViewerInput MouseMove(double x, double y) => new(ViewerInputKind.MouseMove, X: x, Y: y);

        /// <summary>A turn of the wheel.</summary>
        /// <param name="scrollX">The horizontal turn in notches.</param>
        /// <param name="scrollY">The vertical turn in notches, positive away from the user.</param>
        /// <param name="x">The pointer's x.</param>
        /// <param name="y">The pointer's y.</param>
        /// <returns>The input.</returns>
        public static ViewerInput Scroll(double scrollX, double scrollY, double x, double y) =>
            new(ViewerInputKind.Scroll, X: x, Y: y, ScrollX: scrollX, ScrollY: scrollY);
    }

    /// <summary>
    ///     A viewer whose window also reports the user's input, as SDL reports a pygame window's and GLFW a MuJoCo viewer's:
    ///     keys, the close button, resizes and the mouse. <c>Gym.Rendering.Display</c> pumps the keys, the close button and
    ///     resizes into pygame's event queue (<c>Gym.Rendering.EventQueue</c>), which is how <c>Gym.Utils.Play</c> lets a
    ///     person play an env from the keyboard; a MuJoCo env's human window reads all of it for Gymnasium's viewer controls
    ///     (DR-70: orbit, pan and zoom with the mouse, the keys of its menu).
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
    ///     driven by events posted to the queue instead, as Gymnasium's own tests post them). An implementation of before the
    ///     mouse joined the contract simply never reports the mouse kinds, which every reader tolerates.
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
