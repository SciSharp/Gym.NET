using System;

namespace ReinforcementLearning.MemoryTypes
{
    internal class ParameterReplayMemory : ReplayMemory<float[]>
    {
        private readonly int _parameterLength;

        public ParameterReplayMemory(int stageFrames, int parameterLength, int episodesCapacity) : base(stageFrames, episodesCapacity)
        {
            _parameterLength = parameterLength;
        }

        protected override void ValidateInput(float[] currentData)
        {
            if (currentData.Length != _parameterLength)
            {
                throw new ArgumentException($"Parameters size [{currentData.Length}] differs from expected size [{_parameterLength}]");
            }
        }
    }
}