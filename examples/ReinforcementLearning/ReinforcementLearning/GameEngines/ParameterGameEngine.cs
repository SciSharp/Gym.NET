using ReinforcementLearning.DataBuilders;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;

namespace ReinforcementLearning.GameEngines {
    public class ParameterGameEngine : GameEngine<IParametersGameConfiguration, float[]> {
        public ParameterGameEngine(IParametersGameConfiguration game) : base(game) {
        }

        internal override ReplayMemory<float[]> BuildMemory(IParametersGameConfiguration game) =>
            new ParameterReplayMemory(game.MemoryStates, game.ParametersLength, game.MemoryCapacity);

        internal override Trainer<IParametersGameConfiguration, float[]> BuildTrainer(IParametersGameConfiguration game) =>
            new Trainer<IParametersGameConfiguration, float[]>(game, DataBuilder, game.EnvInstance.ActionSpace.Shape.Size);

        protected override DataBuilder<IParametersGameConfiguration, float[]> BuildDataBuilder(IParametersGameConfiguration game) =>
            new ParameterDataBuilder(game, game.EnvInstance.ActionSpace.Shape.Size);
    }
}