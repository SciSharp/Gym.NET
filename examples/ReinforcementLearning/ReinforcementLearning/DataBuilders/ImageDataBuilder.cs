using System.Linq;
using ReinforcementLearning.GameConfigurations;
using SixLabors.ImageSharp;

namespace ReinforcementLearning.DataBuilders {
    public class ImageDataBuilder : DataBuilder<IImageGameConfiguration, Image> {
        public ImageDataBuilder(IImageGameConfiguration configuration, int outputs) : base(configuration, outputs) {
        }

        public override float[] BuildInput(Image[] dataGroup) => new Imager()
            .Load(dataGroup)
            .Crop(Configuration.FramePadding)
            .ComposeFrames(Configuration.ScaledImageWidth, Configuration.ScaledImageHeight, Configuration.ImageStackLayout)
            .InvertColors()
            .Greyscale()
            .Compile()
            .Rectify()
            .ToArray();
    }
}