using System;
using System.Linq;
using Gym.Observations;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.MemoryTypes
{
    public class ParameterReplayMemory : ReplayMemory<float[]>
    {
        private readonly int _parameterLength;

        public ParameterReplayMemory(int stageFrames, int parameterLength, int episodesCapacity) : base(stageFrames, episodesCapacity)
        {
            _parameterLength = parameterLength;
        }

        protected override float[] GetDataInput(Image<Rgba32> currentFrame, Step currentStep)
        {
            var data = currentStep
                .Observation
                .GetData()
                .Cast<float>()
                .ToArray();

            if (data.Length != _parameterLength)
            {
                throw new ArgumentException($"Parameters size [{data.Length}] differs from expected size [{_parameterLength}]");
            }

            return data;
        }
    }
}