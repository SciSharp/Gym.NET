using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AtariDeepQLearner
{
    public class Observation
    {
        public Observation(int images)
        {
            Images = new Image<Rgba32>[images];
        }

        public Image<Rgba32>[] Images { get; set; }
        public int ActionTaken { get; set; }
        public float Reward { get; set; }
    }
}