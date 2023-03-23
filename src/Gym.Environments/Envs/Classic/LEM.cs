using System;
using System.Collections.Generic;

namespace Gym.Environments.Envs.Classic
{
    public class LEM
    {
        /// <summary>
        /// The miles above the surface
        /// </summary>
        public float Altitude { get; private set; }
        /// <summary>
        /// The residual feet in the altitude
        /// </summary>
        public float AltitudeFeet
        {
            get
            {
                return (float)Math.Truncate(5280f * (Altitude - Math.Truncate(Altitude)));
            }
        }
        public float Mass { get; private set; }
        public float FuelMass { get; private set; }
        public float Speed { get; private set; }
        public float SpeedMPH { get { return (float)Math.Truncate(Speed * 3600.0f); } }
        public float Gravity { get; private set; }

        private const float Z = 1.8f;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="a">Initial Altitude</param>
        /// <param name="m">Initial Total Mass of LEM (includes fuel)</param>
        /// <param name="n">Initial Mass of Fuel</param>
        /// <param name="v">Initial Speed</param>
        /// <param name="g">The gravity</param>
        public LEM(float a =120f, float m =33000f, float n =16500f, float v =1f, float g =0.001f)
        {
            Altitude = a;
            Mass = m;
            FuelMass = n;
            Speed = v;
            Gravity = g;
        }

        public float NetFuel
        {
            get
            {
                return (Mass - FuelMass);
            }
        }
        /// <summary>
        /// Returns true when the total mass of the LEM is less than the initial fuel mass
        /// </summary>
        public bool OutOfFuel
        {
            get
            {
                return (NetFuel < 0.001);
            }
        }

        /// <summary>
        /// Computes the drift time for the lander to reach zero altitude and returns the time
        /// </summary>
        /// <returns></returns>
        public float Drift()
        {
            float step = (-Speed + (float)Math.Sqrt(Speed * Speed + 2 * Altitude * Gravity)) / Gravity;
            Speed = Speed + Gravity * step;
            return (step);
        }
        /// <summary>
        /// Computes the next speed and altitude of the LEM given the step and burn rate
        /// </summary>
        /// <param name="step"></param>
        /// <param name="burn"></param>
        /// <returns>(speed,altitude)</returns>
        private Tuple<float, float> Update(float step, float burn)
        {
            float Q = step * burn / Mass;
            float Q2 = Q * Q;
            float Q3 = Q2 * Q;
            float Q4 = Q3 * Q;
            float Q5 = Q4 * Q;
            float J = Speed + Gravity * step + Z * (-Q - Q2 / 2 - Q3 / 3 - Q4 / 4 - Q5 / 5);
            float I = Altitude - Gravity * step * step / 2 - Speed * step + Z * step * (Q / 2 + Q2 / 6 + Q3 / 12 + Q4 / 20 + Q5 / 30);
            return new Tuple<float, float>(J, I);
        }

        private void Apply(float step, float burn, Tuple<float, float> result)
        {
            Mass -= step * burn;
            Altitude = result.Item2;
            Speed = result.Item1;
        }

        public float ApplyBurn(float burn, float time =10.0f)
        {
            float elapsed = 0;
            // Time decay loop
            while (time > 0.001)
            {
                float step = time;
                if (OutOfFuel)
                {
                    break;
                }
                if (Mass < FuelMass + step * burn)
                {
                    step = (Mass - FuelMass) / burn; // line 190
                }
                Tuple<float, float> r = Update(step, burn); // Line 200 -> 420
                if (r.Item2 <= 0.0) // Line 200
                {
                    while (step > 5e-3) // line 340
                    {
                        float D = Speed + (float)Math.Sqrt(Speed * Speed + 2 * Altitude * (Gravity - Z * burn / Mass));
                        step = 2 * Altitude / D;
                        r = Update(step, burn); // -> line 420
                        // Line 330 in basic
                        elapsed += step;
                        time -= step;
                        Apply(step, burn, r); // -> Line 330
                    }
                    // Done
                    return elapsed;
                }
                if (Speed <= 0.0) // Line 210
                {
                    elapsed += step;
                    time -= step;
                    Apply(step, burn, r);
                    continue; // Line 230
                }
                if (r.Item1 < 0.0)
                {
                    do
                    {
                        // Line 370
                        float U = (1 - Mass * Gravity / (Z * burn)) / 2;
                        step = Mass * Speed / (Z * burn * (U + (float)Math.Sqrt(U * U + Speed / Z))) + 0.5f;
                        r = Update(step, burn);
                        if (r.Item2 <= 0.0)
                        {
                            while (step > 5e-3)
                            {
                                float D = Speed + (float)Math.Sqrt(Speed * Speed + 2 * Altitude * (Gravity - Z * burn / Mass));
                                step = 2 * Altitude / D;
                                r = Update(step, burn);
                                // Line 330 in basic
                                elapsed += step;
                                time -= step;
                                Apply(step, burn, r);
                            }
                            return elapsed;
                        }
                        elapsed += step;
                        time -= step;
                        Apply(step, burn, r);
                        if (r.Item1 > 0.0)
                        {
                            // Line 390 -> 160
                            continue;
                        }
                    }
                    while (Speed > 0.0);
                }
                else
                {
                    elapsed += step;
                    time -= step;
                    Apply(step, burn, r);
                }
            }
            return elapsed;
        }
    }
}
