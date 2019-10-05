using AtariDeepQLearner.GameConfigurations;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;

namespace AtariDeepQLearner.Runner
{
    public sealed class CartPoleConfiguration : IGameConfiguration
    {
        public IEnv EnvIstance => new CartPoleEnv(AvaloniaEnvViewer.Run);
        public int MemoryFrames => 1;
        public int SkippedFrames => 2;
        public int FrameWidth => 600;
        public int FrameHeight => 400;
        public int ScaledImageWidth => 50;
        public int ScaledImageHeight => 50;
        public int Episodes => 500;
        public int BatchSize => 100;
        public int Epochs => 10;
    }
}
