using Gym.Envs;

namespace ReinforcementLearning.GameConfigurations
{
    public interface IGameConfiguration
    {
        IEnv EnvIstance { get; }
        ImageStackLayout ImageStackLayout { get; }
        int MemoryFrames { get; }
        int SkippedFrames { get; }
        int FrameWidth { get; }
        int FrameHeight { get; }
        int ScaledImageWidth { get; }
        int ScaledImageHeight { get; }
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
}