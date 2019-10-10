using Gym.Envs;

namespace ReinforcementLearning.GameConfigurations
{
    public interface IGameConfiguration
    {
        IEnv EnvIstance { get; }
        ImageStackLayout ImageStackLayout { get; }
        int MemoryFrames { get; }
        int MemoryCapacity { get; }
        int SkippedFrames { get; }
        int FrameWidth { get; }
        int FrameHeight { get; }
        int ScaledImageWidth { get; }
        int ScaledImageHeight { get; }
        FramePadding FramePadding { get; }
        float StartingEpsilon { get; }
        int Episodes { get; }
        int BatchSize { get; }
        int Epochs { get; }
    }

    public enum ImageStackLayout
    {
        Horizontal,
        Vertical
    }

    public class FramePadding
    {
        public int Left { get; set; }
        public int Top { get; set; }
        public int Right { get; set; }
        public int Bottom { get; set; }
    }
}