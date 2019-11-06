using System.Linq;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.DataBuilders {
    public class ParameterDataBuilder : DataBuilder<IParametersGameConfiguration, float[]> {
        public ParameterDataBuilder(IParametersGameConfiguration configuration, int outputs) : base(configuration, outputs) {
        }

        public override float[] BuildInput(float[][] dataGroup) => dataGroup
            .SelectMany(x => x)
            .ToArray();
    }
}