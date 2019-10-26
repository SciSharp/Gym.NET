using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning
{
    public abstract class Observation<TData>
    {
        public int Id { get; set; }
        public TData[] Data { get; set; }
        public int ActionTaken { get; set; }
        public float Reward { get; set; }

        protected Observation(int dataCount)
        {
            Data = new TData[dataCount];
        }
    }

    public class ImageObservation : Observation<Image<Rgba32>>
    {
        public ImageObservation(int dataCount) : base(dataCount)
        { }
    }

    public class ParamterObservation : Observation<float[]>
    {
        public ParamterObservation(int dataCount) : base(dataCount)
        { }
    }
}