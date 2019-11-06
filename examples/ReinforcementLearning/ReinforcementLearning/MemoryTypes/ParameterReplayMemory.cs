using System;
using System.Linq;
using Gym.Observations;
using SixLabors.ImageSharp;

namespace ReinforcementLearning.MemoryTypes {
    public class ParameterReplayMemory : ReplayMemory<float[]> {
        private readonly int _parameterLength;

        public ParameterReplayMemory(int stageFrames, int parameterLength, int episodesCapacity) : base(stageFrames,
            episodesCapacity) {
            _parameterLength = parameterLength;
        }

        protected override float[] GetDataInput(Image currentFrame, Step currentStep) {
            if (currentStep.Observation == null) {
                return Enumerable.Repeat(0F, _parameterLength).ToArray();
            }

            var data = currentStep
                .Observation
                .GetData()
                .Cast<double>()
                .Select(x => (float) x)
                .ToArray();

            if (data.Length != _parameterLength) {
                throw new ArgumentException($"Parameters size [{data.Length}] differs from expected size [{_parameterLength}]");
            }

            return data;
        }
    }
}