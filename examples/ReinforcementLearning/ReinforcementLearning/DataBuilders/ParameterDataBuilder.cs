using System.Linq;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.DataBuilders
{
    public class ParameterDataBuilder : DataBuilder<float[]>
    {
        public ParameterDataBuilder(IGameConfiguration configuration, int outputs) : base(configuration, outputs)
        { }

        public override float[] BuildInput(float[][] dataGroup) =>dataGroup
            .SelectMany(x => x)
            .ToArray();
    }
}