/*
* Copyright (c) 2006-2011 Erin Catto http://www.box2d.org
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
// Altered source version (Gym.NET, 2026-11-10): Box2D/Dynamics/b2TimeStep.h of pybox2d 2.3.10's bundled Box2D 2.3.0,
// translated to C#; b2SolverData holds the island's position and velocity arrays by reference instead of by pointer.
namespace Gym.Box2D {
    /// <summary>Timing of the last step's phases in milliseconds; this port leaves them at zero (no timer, nothing reads them).</summary>
    /// <remarks>Reference: <see href="https://github.com/pybox2d/pybox2d/blob/ed9d7b04f7099a15984c727dbce46c13abd64409/Box2D/Dynamics/b2TimeStep.h">Box2D/Dynamics/b2TimeStep.h</see>.</remarks>
    internal struct b2Profile {
        /// <summary>The whole step.</summary>
        public float step;

        /// <summary>The narrow-phase.</summary>
        public float collide;

        /// <summary>The solver.</summary>
        public float solve;

        /// <summary>The solver's initialization.</summary>
        public float solveInit;

        /// <summary>The velocity iterations.</summary>
        public float solveVelocity;

        /// <summary>The position iterations.</summary>
        public float solvePosition;

        /// <summary>The broad-phase.</summary>
        public float broadphase;

        /// <summary>The time-of-impact solver.</summary>
        public float solveTOI;
    }

    /// <summary>A time step's parameters, as the world hands them to the solvers.</summary>
    internal struct b2TimeStep {
        /// <summary>The time step.</summary>
        public float dt;

        /// <summary>The inverse time step (0 if dt == 0).</summary>
        public float inv_dt;

        /// <summary><c>dt * inv_dt0</c>: the ratio to the previous step, which scales the warm-start impulses.</summary>
        public float dtRatio;

        /// <summary>The number of velocity iterations.</summary>
        public int velocityIterations;

        /// <summary>The number of position iterations.</summary>
        public int positionIterations;

        /// <summary>Whether to warm-start the solvers with the last step's impulses.</summary>
        public bool warmStarting;
    }

    /// <summary>A body's solver position: its center of mass and angle.</summary>
    internal struct b2Position {
        /// <summary>The center of mass.</summary>
        public b2Vec2 c;

        /// <summary>The angle.</summary>
        public float a;
    }

    /// <summary>A body's solver velocity: linear and angular.</summary>
    internal struct b2Velocity {
        /// <summary>The linear velocity.</summary>
        public b2Vec2 v;

        /// <summary>The angular velocity.</summary>
        public float w;
    }

    /// <summary>What the joint solvers read and write: the step and the island's positions and velocities.</summary>
    internal struct b2SolverData {
        /// <summary>The step.</summary>
        public b2TimeStep step;

        /// <summary>The island's positions, indexed by each body's island index.</summary>
        public b2Position[] positions;

        /// <summary>The island's velocities, indexed by each body's island index.</summary>
        public b2Velocity[] velocities;
    }
}
