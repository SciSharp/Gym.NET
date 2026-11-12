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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/b2WorldCallbacks.h and b2WorldCallbacks.cpp of pybox2d
// 2.3.10's bundled Box2D 2.3.0, translated to C#. PreSolve and PostSolve stay internal (their arguments are engine
// structures no Gymnasium env reads); the listener's public hooks are BeginContact and EndContact.
using System.Runtime.CompilerServices;
using static Gym.Box2D.b2Settings;

namespace Gym.Box2D {
    /// <summary>Told when the world destroys joints and fixtures implicitly, because their body was destroyed.</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2WorldCallbacks.h">Box2D/Dynamics/b2WorldCallbacks.h</see>.</remarks>
    public abstract class b2DestructionListener {
        /// <summary>Called when a joint is about to be destroyed because one of its bodies is.</summary>
        /// <param name="joint">The joint.</param>
        public abstract void SayGoodbye(b2Joint joint);

        /// <summary>Called when a fixture is about to be destroyed because its body is.</summary>
        /// <param name="fixture">The fixture.</param>
        public abstract void SayGoodbye(b2Fixture fixture);
    }

    /// <summary>Decides whether two fixtures may collide; the default reads their <see cref="b2Filter"/>s.</summary>
    public class b2ContactFilter {
        /// <summary>
        ///     Box2D's default filter: fixtures of the same non-zero group always (positive) or never (negative) collide;
        ///     otherwise each one's mask must accept the other's category.
        /// </summary>
        /// <param name="fixtureA">The first fixture.</param>
        /// <param name="fixtureB">The second fixture.</param>
        /// <returns>True when contact calculations should be performed between these two fixtures.</returns>
        public virtual bool ShouldCollide(b2Fixture fixtureA, b2Fixture fixtureB) {
            b2Filter filterA = fixtureA.GetFilterData();
            b2Filter filterB = fixtureB.GetFilterData();

            if (filterA.groupIndex == filterB.groupIndex && filterA.groupIndex != 0)
                return filterA.groupIndex > 0;

            bool collide = (filterA.maskBits & filterB.categoryBits) != 0 && (filterA.categoryBits & filterB.maskBits) != 0;
            return collide;
        }
    }

    /// <summary>Two floats, inline (C++ <c>float32[b2_maxManifoldPoints]</c>).</summary>
    [InlineArray(b2_maxManifoldPoints)]
    internal struct b2Float2 {
        /// <summary>The first element; the runtime lays the second after it.</summary>
        private float _element0;
    }

    /// <summary>The impulses a contact received in the solver, for <c>PostSolve</c>.</summary>
    internal struct b2ContactImpulse {
        /// <summary>The normal impulse of each point.</summary>
        public b2Float2 normalImpulses;

        /// <summary>The tangent impulse of each point.</summary>
        public b2Float2 tangentImpulses;

        /// <summary>The number of points.</summary>
        public int count;
    }

    /// <summary>
    ///     Told about contacts as they begin and end touching (pybox2d's <c>contactListener</c>, which Gymnasium's envs
    ///     subclass as <c>ContactDetector</c>). The callbacks run inside <see cref="b2World.Step(double, int, int)"/>, while the world is
    ///     locked.
    /// </summary>
    /// <remarks>
    ///     Footgun: do not create or destroy bodies, fixtures or joints from a callback; the world is locked and refuses with
    ///     an <see cref="NumSharp.AssertionError"/>. Record what to change and do it after the step, as Gymnasium's envs do.
    /// </remarks>
    public class b2ContactListener {
        /// <summary>Called when two fixtures begin to touch (the contact's first touching update; sensors included).</summary>
        /// <param name="contact">The contact.</param>
        public virtual void BeginContact(b2Contact contact) { }

        /// <summary>Called when two fixtures cease to touch, including when their contact is destroyed while touching.</summary>
        /// <param name="contact">The contact.</param>
        public virtual void EndContact(b2Contact contact) { }

        /// <summary>Called after a touching, non-sensor contact is updated and before it is solved.</summary>
        /// <param name="contact">The contact.</param>
        /// <param name="oldManifold">The manifold before the update.</param>
        internal virtual void PreSolve(b2Contact contact, in b2Manifold oldManifold) { }

        /// <summary>Called after each solved contact with the impulses it received.</summary>
        /// <param name="contact">The contact.</param>
        /// <param name="impulse">The impulses.</param>
        internal virtual void PostSolve(b2Contact contact, in b2ContactImpulse impulse) { }
    }

    /// <summary>Told about each fixture whose fat box overlaps a <see cref="b2World.QueryAABB"/> box.</summary>
    public abstract class b2QueryCallback {
        /// <summary>Called for each fixture found.</summary>
        /// <param name="fixture">The fixture.</param>
        /// <returns>False to stop the query.</returns>
        public abstract bool ReportFixture(b2Fixture fixture);
    }

    /// <summary>
    ///     Told about each fixture a <see cref="b2World.RayCast(b2RayCastCallback, b2Vec2, b2Vec2)"/> ray hits (pybox2d's
    ///     <c>rayCastCallback</c>, which BipedalWalker's lidar subclasses). Hits arrive in tree order, not by distance.
    /// </summary>
    public abstract class b2RayCastCallback {
        /// <summary>Called for each fixture hit.</summary>
        /// <param name="fixture">The fixture hit.</param>
        /// <param name="point">The point of initial intersection.</param>
        /// <param name="normal">The normal at that point.</param>
        /// <param name="fraction">The fraction along the ray.</param>
        /// <returns>-1 to ignore this fixture, 0 to stop, the fraction to clip the ray to it, or 1 to continue.</returns>
        public abstract float ReportFixture(b2Fixture fixture, b2Vec2 point, b2Vec2 normal, float fraction);
    }
}
