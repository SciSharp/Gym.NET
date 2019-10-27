using System.Linq;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.DataBuilders
{
    public class ImageDataBuilder : DataBuilder<Image<Rgba32>>
    {
        public ImageDataBuilder(IGameConfiguration configuration, int outputs) : base(configuration, outputs)
        { }

        public override float[] BuildInput(Image<Rgba32>[] dataGroup) => Imager.Load(dataGroup)
            .Crop(Configuration.FramePadding)
            .ComposeFrames(Configuration.ScaledImageWidth, Configuration.ScaledImageHeight, Configuration.ImageStackLayout)
            .InvertColors()
            .Greyscale()
            .Compile()
            .Rectify()
            .ToArray();
    }
}