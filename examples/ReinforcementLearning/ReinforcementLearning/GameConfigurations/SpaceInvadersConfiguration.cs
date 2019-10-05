using Gym.Core;
using Gym.Helper;
using Python.Runtime;

namespace AtariDeepQLearner.GameConfigurations
{
    internal sealed class SpaceInvadersConfiguration : IGameConfiguration
    {
        public string Code => "SpaceInvaders-v0";
        public int MemoryFrames => 4;
        public int AvailableActions => 6;
        public int FrameWidth => 160;
        public int FrameHeight => 210;
        public int ScaledImageWidth => 200;
        public int ScaledImageHeight => 200;
        
        public byte[][] ParseEnvResult(EnvResult env)
        {
            var list = new TupleSolver().TupleToList((PyObject)env.Observation);
            var result = new byte[list.Length][];
            for (var x = 0; x < list.Length; x++)
            {
                var pixels = list[x].GetData<byte>();
                result[x] = new byte[pixels.Length / 3];
                for (var y = 0; y < pixels.Length / 3; y++)
                {
                    var value = 0;
                    for (var k = 0; k < 3; k++)
                    {
                        value += pixels[y * 3 + k];
                    }

                    result[x][y] = (byte)(value / 3);
                }
            }

            return result;
        }
    }
}
