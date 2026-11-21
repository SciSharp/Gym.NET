/*
* pybox2d -- http://pybox2d.googlecode.com
*
* Copyright (c) 2010 Ken Lauer / sirkne at gmail dot com
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
// Altered source version (Gym.NET, 2026-11-10): the b2FixtureDef and b2Fixture extensions of pybox2d 2.3.10's
// Box2D/Box2D_bodyfixture.i and the definition's keyword initializer (Box2D_kwargs.i), translated to C#: the filter
// shortcuts (categoryBits, maskBits, groupIndex) and the fixture's read-write properties.
using NumSharp;

namespace Gym.Box2D {
    public partial class b2FixtureDef {
        /// <summary>A fixture definition with Box2D's defaults (friction 0.2, no shape yet).</summary>
        public b2FixtureDef() { }

        /// <summary>
        ///     A fixture definition from pybox2d's keyword arguments (<c>fixtureDef(shape=..., density=..., ...)</c>);
        ///     scalars round to float32 as SWIG converts them, and a refused one raises pybox2d's
        ///     <c>Failed on kwargs for b2FixtureDef.density: …</c> (its keyword named).
        /// </summary>
        /// <param name="shape">The shape, cloned by each fixture created from this definition.</param>
        /// <param name="density">The density in kg/m².</param>
        /// <param name="friction">The friction.</param>
        /// <param name="restitution">The restitution.</param>
        /// <param name="isSensor">Whether the fixture is a sensor.</param>
        /// <param name="categoryBits">The collision category bits.</param>
        /// <param name="maskBits">The collision mask bits.</param>
        /// <param name="groupIndex">The collision group.</param>
        /// <param name="userData">The fixture's user data.</param>
        /// <exception cref="OverflowError">Thrown for a scalar beyond the float32 range.</exception>
        public b2FixtureDef(b2Shape? shape, double density = 0.0, double friction = 0.2, double restitution = 0.0, bool isSensor = false,
                            ushort categoryBits = 0x0001, ushort maskBits = 0xFFFF, short groupIndex = 0, object? userData = null) {
            this.shape = shape;
            // The float32 conversions can fail; pybox2d's _init_kwargs prefixes their errors with the class and keyword.
            Box2DConvert.Kwarg("b2FixtureDef", "density", () => this.density = Box2DConvert.ToFloat32(density, "b2FixtureDef_density_set", 2));
            Box2DConvert.Kwarg("b2FixtureDef", "friction", () => this.friction = Box2DConvert.ToFloat32(friction, "b2FixtureDef_friction_set", 2));
            Box2DConvert.Kwarg("b2FixtureDef", "restitution", () => this.restitution = Box2DConvert.ToFloat32(restitution, "b2FixtureDef_restitution_set", 2));
            this.isSensor = isSensor;
            filter.categoryBits = categoryBits;
            filter.maskBits = maskBits;
            filter.groupIndex = groupIndex;
            this.userData = userData;
        }

        /// <summary>The shape (pybox2d's <c>shape</c>); fixtures clone it, so changing it later affects only new fixtures.</summary>
        public b2Shape? Shape {
            get => shape;
            set => shape = value;
        }

        /// <summary>The density in kg/m² (stored as float32).</summary>
        public double Density {
            get => density;
            set => density = Box2DConvert.ToFloat32(value, "b2FixtureDef_density_set", 2);
        }

        /// <summary>The friction (stored as float32).</summary>
        public double Friction {
            get => friction;
            set => friction = Box2DConvert.ToFloat32(value, "b2FixtureDef_friction_set", 2);
        }

        /// <summary>The restitution (stored as float32).</summary>
        public double Restitution {
            get => restitution;
            set => restitution = Box2DConvert.ToFloat32(value, "b2FixtureDef_restitution_set", 2);
        }

        /// <summary>Whether the fixture is a sensor.</summary>
        public bool IsSensor {
            get => isSensor;
            set => isSensor = value;
        }

        /// <summary>The whole filter.</summary>
        public b2Filter Filter {
            get => filter;
            set => filter = value;
        }

        /// <summary>The collision category bits (pybox2d's <c>categoryBits</c>, a shortcut into <see cref="Filter"/>).</summary>
        public ushort CategoryBits {
            get => filter.categoryBits;
            set => filter.categoryBits = value;
        }

        /// <summary>The collision mask bits (pybox2d's <c>maskBits</c>).</summary>
        public ushort MaskBits {
            get => filter.maskBits;
            set => filter.maskBits = value;
        }

        /// <summary>The collision group (pybox2d's <c>groupIndex</c>).</summary>
        public short GroupIndex {
            get => filter.groupIndex;
            set => filter.groupIndex = value;
        }

        /// <summary>The fixture's user data.</summary>
        public object? UserData {
            get => userData;
            set => userData = value;
        }
    }

    public partial class b2Fixture {
        /// <summary>The fixture's own shape (pybox2d's <c>shape</c>): a clone of the definition's, so changing it changes the fixture.</summary>
        public b2Shape Shape => m_shape!;

        /// <summary>The body (pybox2d's <c>body</c>).</summary>
        public b2Body Body => m_body!;

        /// <summary>The shape kind (pybox2d's <c>type</c>).</summary>
        public b2ShapeType Type => m_shape!.m_type;

        /// <summary>The filter (pybox2d's <c>filterData</c>); setting it flags the fixture's contacts for re-filtering.</summary>
        public b2Filter FilterData {
            get => m_filter;
            set => SetFilterData(value);
        }

        /// <summary>Whether the fixture is a sensor (pybox2d's <c>sensor</c>); changing it wakes the body.</summary>
        public bool Sensor {
            get => m_isSensor;
            set => SetSensor(value);
        }

        /// <summary>The density (pybox2d's <c>density</c>); the body's mass changes only at its next <c>ResetMassData</c>.</summary>
        /// <exception cref="AssertionError">Thrown by the setter for a negative or non-finite density.</exception>
        public float Density {
            get => m_density;
            set => SetDensity(value);
        }

        /// <summary>The friction (pybox2d's <c>friction</c>); existing contacts keep their mixed friction.</summary>
        public float Friction {
            get => m_friction;
            set => m_friction = value;
        }

        /// <summary>The restitution (pybox2d's <c>restitution</c>); existing contacts keep their mixed restitution.</summary>
        public float Restitution {
            get => m_restitution;
            set => m_restitution = value;
        }

        /// <summary>The user data (pybox2d's <c>userData</c>).</summary>
        public object? UserData {
            get => m_userData;
            set => m_userData = value;
        }
    }
}
