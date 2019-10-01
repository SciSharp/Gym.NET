using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Gym.Environments
{
    public interface IEnvViewer : IDisposable
    {
        void Render(Image<Rgba32> img);
        void Close();
    }
}
