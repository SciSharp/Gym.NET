using System;
using Gym.Observations;
using SixLabors.ImageSharp;

namespace ReinforcementLearning.MemoryTypes {
    public class ImageReplayMemory : ReplayMemory<Image> {
        private readonly int _imageWidth;
        private readonly int _imageHeight;

        public ImageReplayMemory(int stageFrames, int imageWidth, int imageHeight, int episodesCapacity) : base(
            stageFrames, episodesCapacity) {
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
        }

        protected override Image GetDataInput(Image currentFrame, Step currentStep) {
            if (currentFrame.Width != _imageWidth || currentFrame.Height != _imageHeight) {
                throw new ArgumentException($"Frame size differs from expected size");
            }

            return currentFrame;
        }
    }
}