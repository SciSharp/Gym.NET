using System;
using System.Collections.Generic;

namespace Gym.Environments.Envs.Classic
{
    /// <summary>
    /// LEM physics for a simple landing vehicle with a straight drop to the landing pad.
    /// </summary>
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

        /// <summary>
        /// Updates the mass of the LEM for the given burn for the given step time. The altitude and
        /// speed are updated from the result parameter.
        /// </summary>
        /// <param name="step">Time of the burn</param>
        /// <param name="burn">Amount of fuel burn</param>
        /// <param name="result">Output from a call to Update(step,burn)</param>
        private void Apply(float step, float burn, Tuple<float, float> result)
        {
            Mass -= step * burn;
            Altitude = result.Item2;
            Speed = result.Item1;
        }

        /// <summary>
        /// Main method to apply the descent burn to the LEM. Computes the new flight parameters.
        /// </summary>
        /// <param name="burn">The number of pounds of fuel to burn per second</param>
        /// <param name="time">The number of seconds to perform the burn</param>
        /// <returns>The elapsed time for the burn procedure</returns>
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
/*
 * ORIGINAL BASIC CODE TRANSCRIBED FROM COMPUTE MAGAZINE
 * LUNAR LEM ROCKET
 * 
10 PRINT TAB(33);"LUNAR"
20 PRINT TAB(15);"CREATIVE COMPUTING MORRISTOWN, NEW JERSEY"
25 PRINT:PRINT:PRINT
30 PRINT "THIS IS A COMPUTER SIMULATION OF AN APOLLO LUNAR"
40 PRINT "LANDING CAPSULE.": PRINT: PRINT
50 PRINT "THE ON-BOARD COMPUTER HAS FAILED (IT WAS MADE BY"
60 PRINT "(XEROX) SO YOU HAVE TO LAND THE CASULE MANUALLY."
70 PRINT: PRINT "SET BURN RATE OF RETRO ROCKETS TO ANY VALUE BETWEEN"
80 PRINT "0 (FREE FALL) AND 200 (MAX BURN) POUNDS PER SECOND."
90 PRINT "SET NEW BURN RATE EVERY 10 SECONDS." :PRINT
100 PRINT "CAPSULE WEIGHT 32,500 LBS; FUEL WEIGHT 16,500 LBS."
110 PRINT: PRINT: PRINT: PRINT "GOOD LUCK"
120 L=0
130 PRINT: PRINT "SEC","MI + FT","MPH","LB FUEL","BURN RATE" :PRINT
140 A=120: V=1: M=33000: N=16500: G=1E-03: Z=1.8
150 PRINT L, INT(A); INT(5280*(A-INT(A))),3600*V,M-N,:INPUT K:T=10
160 IF M-N <1E-03 THEN 240
170 IF T<1E-03 THEN 150
180 S=T: IF M>= N+S*K THEN 200
190 S=(M-N)/K
200 GOSUB 420: IF I<= 0 THEN 340
210 IF V<=0 THEN 230
220 IF J<0 THEN 370
230 GOSUB 330: GOTO 160
240 PRINT "FUEL OUT AT";L;"SECONDS": S=(-V+SQR(V*V+2*A*G))/G
250 V=V+G*S: L=L+S
260 U=3600*V:PRINT "ON MOON AT";L;"SECONDS - IMPACT VELOCITY";U;"MPH"
270 IF U <= 1.2 THEN PRINT "PERFECT LANDING!": GOTO 440
280 IF U <= 10 THEN PRINT "GOOD LANDING (COULD BE BETTER)" : GOTO 440
282 IF U > 60 THEN 300
284 PRINT "CRAFT DAMAGE ... YOU'RE STRANDED HERE UNTIL A RESCUE"
286 PRINT "PARTY ARRIVES. HOPE YOU HAVE ENOUGH OXYGEN!"
288 GOTO 440
300 PRINT "SORRY THERE WERE NO SURVIVORS. YOU BLEW IT!"
310 PRINT "IN FACT, YOU BLASTED A NEW LUNAR CRATER";U*.277;"FEET DEEP!"
320 GOTO 440
330 L=L+S: T=T-S: M=M-S*K: A=I: V=J: RETURN
340 IF S < 5E-03 THEN 260
350 D=V+SQR(V*V+2*A*(G-Z*K/M)): S=2*A/D
360 GOSUB 420: GOSUB 330: GOTO 340
370 U=(1-M*G/(Z*K))/2: S=M*V/(Z*K*(U+SQR(U*U+V/Z)))+0.05: GOSUB 420
380 IF I<=0 THEN 340
390 GOSUB 330: IF J>0 THEN 160
400 IF V>0 THEN 370
410 GOTO 160
420 Q=S*K/M: J=V+G*S+Z*(-Q-Q*Q/2-Q^3/3-Q^4/4-Q^5/5)
430 I=A-G*S*S/2-V*S+Z*S*(Q/2+Q^2/6+Q^3/12+Q^4/20+Q^5/30):RETURN
440 PRINT:PRINT:PRINT:PRINT "TRY AGAIN??": GOTO 70
*/