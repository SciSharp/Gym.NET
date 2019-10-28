using System;
using Gym.Environments.Envs.Classic;
using Gym.Envs;
using Gym.Rendering.Avalonia;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using ReinforcementLearning.GameConfigurations;

namespace ReinforcementLearning.Parameters.Runner {
    public sealed class CartPoleConfiguration : IParametersGameConfiguration {
        private readonly Lazy<IEnv> _env = new Lazy<IEnv>(() => new CartPoleEnv(AvaloniaEnvViewer.Run));

        public IEnv EnvInstance => _env.Value;
        public int MemoryStates => 4;
        public int MemoryCapacity => 100;
        public int SkippedFrames => 0;
        public int ParametersLength => 4; // x, x_dot, theta, theta_dot
        public float StartingEpsilon => 1F;
        public int Episodes => 2000;
        public int BatchSize => 100;
        public int Epochs => 10;

        public INeuralNetwork BuildNeuralNetwork() =>
            NetworkManager.NewSequential(TensorInfo.Linear(ParametersLength * MemoryStates),
                NetworkLayers.FullyConnected(50, ActivationType.ReLU),
                NetworkLayers.FullyConnected(20, ActivationType.ReLU),
                NetworkLayers.Softmax(EnvInstance.ActionSpace.Shape.Size));
    }
}