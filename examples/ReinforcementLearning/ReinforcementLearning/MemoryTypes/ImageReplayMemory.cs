using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning.MemoryTypes
{
    internal class ImageReplayMemory : ReplayMemory<Image<Rgba32>>
    {
        private readonly int _imageWidth;
        private readonly int _imageHeight;

        public ImageReplayMemory(int stageFrames, int imageWidth, int imageHeight, int episodesCapacity) : base(stageFrames, episodesCapacity)
        {
            _imageWidth = imageWidth;
            _imageHeight = imageHeight;
        }

        protected override void ValidateInput(Image<Rgba32> currentData)
        {
            if (currentData.Width != _imageWidth || currentData.Height != _imageHeight)
            {
                throw new ArgumentException($"Frame size differs from expected size");
            }
        }
    }
}