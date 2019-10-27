using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.GameEngines
{
    public class ImageGameEngine : GameEngine<IImageGameConfiguration, Image<Rgba32>>
    {
        public ImageGameEngine(IImageGameConfiguration game) : base(game)
        { }

        internal override ReplayMemory<Image<Rgba32>> BuildMemory(IImageGameConfiguration game) =>
            new ImageReplayMemory(game.MemoryStates, game.FrameWidth, game.FrameHeight, game.MemoryCapacity);

        internal override Trainer<IImageGameConfiguration, Image<Rgba32>> BuildTrainer(IImageGameConfiguration game) =>
            new Trainer<IImageGameConfiguration, Image<Rgba32>>(game, DataBuilder, game.EnvInstance.ActionSpace.Shape.Size);

        protected override DataBuilder<IImageGameConfiguration, Image<Rgba32>> BuildDataBuilder(IImageGameConfiguration game) =>
            new ImageDataBuilder(game, game.EnvInstance.ActionSpace.Shape.Size);
    }
}