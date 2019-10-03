using Gym.Envs;

namespace AtariDeepQLearner.GameConfigurations
{
    public interface IGameConfiguration
    {
        IEnv EnvIstance { get; }
        int MemoryFrames { get; }
        int FrameWidth { get; }
        int FrameHeight { get; }
        int ScaledImageWidth { get; }
        int ScaledImageHeight { get; }
        int Episodes { get; }
        int BatchSize { get; }
        int Epochs { get; }
    }
}