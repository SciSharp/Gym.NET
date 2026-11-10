/*
* Copyright (c) 2006-2009 Erin Catto http://www.box2d.org
*
* This software is provided 'as-is', without any express or implied
* warranty.  In no event will the authors be held liable for any damages
* arising from the use of this software.
* Permission is granted to anyone to use this software for any purpose,
* including commercial applications, and to alter it and redistribute it
* freely, subject to the following restrictions:
* 1. The origin of this software must not be misrepresented; you must not
* claim that you wrote the original software. If you use this software
* in a product, an acknowledgment in the product documentation would be
* appreciated but is not required.
* 2. Altered source versions must be plainly marked as such, and must not be
* misrepresented as being the original software.
* 3. This notice may not be removed or altered from any source distribution.
*/
// Altered source version (Gym.NET, 2026-11-10): Box2D/Common/b2Settings.h and b2Settings.cpp of pybox2d 2.3.10's
// bundled Box2D 2.3.0, translated to C#. The constants keep their float32 values; b2Assert keeps pybox2d's behavior
// (USE_EXCEPTIONS: a failed assertion raises Python's AssertionError carrying the asserted expression's text); the
// allocator and logging hooks are dropped, as .NET allocates and nothing logs.
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using NumSharp;

namespace Gym.Box2D {
    /// <summary>
    ///     Box2D's global tuning constants, in meters, kilograms and seconds, with float32 values exactly as the C++ defines
    ///     them, and its assertion. Every engine file reads them through <c>using static</c>, so the translated code keeps
    ///     the C++ spelling (<c>b2_linearSlop</c>).
    /// </summary>
    /// <remarks>
    ///     Footgun: these values are part of the bit-for-bit contract with pybox2d. Changing one changes every trajectory.
    ///     <para>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Common/b2Settings.h">Box2D/Common/b2Settings.h</see>.</para>
    /// </remarks>
    internal static class b2Settings {
        /// <summary>FLT_MAX: the "no limit" value the engine starts searches and AABBs from.</summary>
        public const float b2_maxFloat = float.MaxValue;

        /// <summary>FLT_EPSILON (2^-23): the engine's tolerance for "practically zero" lengths and determinants.</summary>
        public const float b2_epsilon = 1.1920929E-07f;

        /// <summary>π as Box2D writes it, <c>3.14159265359f</c>, which rounds to the float32 nearest π.</summary>
        public const float b2_pi = 3.14159265359f;

        /// <summary>The maximum number of contact points between two convex shapes. Do not change.</summary>
        public const int b2_maxManifoldPoints = 2;

        /// <summary>The maximum number of vertices on a convex polygon; <see cref="b2PolygonShape"/> refuses more.</summary>
        public const int b2_maxPolygonVertices = 16;

        /// <summary>How far the dynamic tree fattens AABBs (meters), so small moves don't re-insert a proxy.</summary>
        public const float b2_aabbExtension = 0.1f;

        /// <summary>The multiplier on a proxy's displacement that predicts where its fattened AABB must reach.</summary>
        public const float b2_aabbMultiplier = 2.0f;

        /// <summary>A small length used as the collision and constraint tolerance (meters).</summary>
        public const float b2_linearSlop = 0.005f;

        /// <summary>A small angle used as the collision and constraint tolerance (radians): <c>2.0f / 180.0f * b2_pi</c> in float32.</summary>
        public const float b2_angularSlop = 2.0f / 180.0f * b2_pi;

        /// <summary>The skin radius of polygons and edges, <c>2.0f * b2_linearSlop</c>.</summary>
        public const float b2_polygonRadius = 2.0f * b2_linearSlop;

        /// <summary>The maximum number of sub-steps per contact in continuous physics.</summary>
        public const int b2_maxSubSteps = 8;

        /// <summary>The maximum number of contacts handled when solving a time-of-impact event.</summary>
        public const int b2_maxTOIContacts = 32;

        /// <summary>Relative normal speeds below this (m/s) are treated as inelastic, whatever the restitution.</summary>
        public const float b2_velocityThreshold = 1.0f;

        /// <summary>The maximum linear position correction per constraint solve (meters).</summary>
        public const float b2_maxLinearCorrection = 0.2f;

        /// <summary>The maximum angular position correction per constraint solve (radians).</summary>
        public const float b2_maxAngularCorrection = 8.0f / 180.0f * b2_pi;

        /// <summary>The maximum translation of a body per step (meters); faster bodies are clamped.</summary>
        public const float b2_maxTranslation = 2.0f;

        /// <summary><see cref="b2_maxTranslation"/> squared.</summary>
        public const float b2_maxTranslationSquared = b2_maxTranslation * b2_maxTranslation;

        /// <summary>The maximum rotation of a body per step (radians); faster spins are clamped.</summary>
        public const float b2_maxRotation = 0.5f * b2_pi;

        /// <summary><see cref="b2_maxRotation"/> squared.</summary>
        public const float b2_maxRotationSquared = b2_maxRotation * b2_maxRotation;

        /// <summary>How fast overlap is resolved per step: 1 would remove it at once but overshoots.</summary>
        public const float b2_baumgarte = 0.2f;

        /// <summary>The overlap-resolution factor of time-of-impact sub-steps.</summary>
        public const float b2_toiBaugarte = 0.75f;

        /// <summary>The time (seconds) a body must stay still before it may sleep.</summary>
        public const float b2_timeToSleep = 0.5f;

        /// <summary>A body faster than this (m/s) cannot sleep.</summary>
        public const float b2_linearSleepTolerance = 0.01f;

        /// <summary>A body spinning faster than this (rad/s) cannot sleep.</summary>
        public const float b2_angularSleepTolerance = 2.0f / 180.0f * b2_pi;

        /// <summary>
        ///     Box2D's <c>b2Assert</c> as pybox2d compiles it: a failed check raises Python's <see cref="AssertionError"/>
        ///     whose message is the asserted expression's source text, which pybox2d surfaces to the caller.
        /// </summary>
        /// <remarks>
        ///     The message is the C# expression text (<see cref="CallerArgumentExpressionAttribute"/>) when it reads exactly
        ///     as the C++ one. Where C# spells the condition differently (<c>bd.position</c> for <c>bd-&gt;position</c>, enum
        ///     prefixes, <c>GetShapeType()</c> for <c>GetType()</c>, an indexed node for a node pointer), the call passes
        ///     C++'s stringified text (<c>#A</c>) itself, so every assertion raises pybox2d's message. Footgun: a new or edited
        ///     assertion must keep that rule; the parity suite's assertion cell and a scan of the C++ sources catch drift.
        /// </remarks>
        /// <param name="condition">The condition that must hold.</param>
        /// <param name="expression">
        ///     The message: the condition's text, filled by the compiler; pass C++'s text explicitly only where the C#
        ///     spelling differs from it.
        /// </param>
        /// <exception cref="AssertionError">Thrown when <paramref name="condition"/> is false.</exception>
        public static void b2Assert([DoesNotReturnIf(false)] bool condition, [CallerArgumentExpression(nameof(condition))] string expression = "") {
            if (!condition)
                throw new AssertionError(expression);
        }
    }
}
