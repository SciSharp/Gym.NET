using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;
using SixLabors.ImageSharp;

namespace ReinforcementLearning.GameEngines {
    public class ImageGameEngine : GameEngine<IImageGameConfiguration, Image> {
        public ImageGameEngine(IImageGameConfiguration game) : base(game) {
        }

        internal override ReplayMemory<Image> BuildMemory(IImageGameConfiguration game) =>
            new ImageReplayMemory(game.MemoryStates, game.FrameWidth, game.FrameHeight, game.MemoryCapacity);

        internal override Trainer<IImageGameConfiguration, Image> BuildTrainer(IImageGameConfiguration game) =>
            new Trainer<IImageGameConfiguration, Image>(game, DataBuilder, game.EnvInstance.ActionSpace.Shape.Size);

        protected override DataBuilder<IImageGameConfiguration, Image> BuildDataBuilder(IImageGameConfiguration game) => 
            new ImageDataBuilder(game, game.EnvInstance.ActionSpace.Shape.Size);
    }
}