using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Interfaces.Data;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp.PixelFormats;

namespace AtariDeepQLearner
{
    public class Trainer
    {
        private readonly INeuralNetwork _network;
        private readonly Random _random = new Random();
        private readonly Imager _imager = new Imager();
        private readonly int _outputs;
        private readonly int _epochs;
        private readonly int _inputImgWidth;
        private readonly int _inputImgHeight;

        public Trainer(int inputImgWidth, int inputImgHeight, int outputs, int epochs)
        {
            _inputImgWidth = inputImgWidth;
            _inputImgHeight = inputImgHeight;
            _outputs = outputs;
            _epochs = epochs;

            if (CuDnnNetworkLayers.IsCudaSupportAvailable)
            {
                Console.ForegroundColor = ConsoleColor.DarkGreen;
                Console.WriteLine("Cuda Gpu support available");
                Console.ResetColor();

                _network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(inputImgHeight, inputImgWidth),
                    CuDnnNetworkLayers.Convolutional((5, 5), 50, ActivationType.Identity),
                    //NetworkLayers.Pooling(ActivationType.LeakyReLU), // used to reduce the spatial dimensions
                    CuDnnNetworkLayers.FullyConnected(10, ActivationType.Sigmoid),
                    CuDnnNetworkLayers.Softmax(outputs));
                return;
            }

            _network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(inputImgHeight, inputImgWidth),
                NetworkLayers.Convolutional((5, 5), 50, ActivationType.Identity),
                //NetworkLayers.Pooling(ActivationType.LeakyReLU), // used to reduce the spatial dimensions
                NetworkLayers.FullyConnected(10, ActivationType.Sigmoid),
                NetworkLayers.Softmax(outputs));
        }

        public ITrainingDataset BuildStuff(ReplayMemory memory)
        {
            var bestEpisodes = memory
                .Episodes
                .OrderByDescending(x => x.TotalReward)
                .Take((int)Math.Ceiling((double)memory.Episodes.Count / 5))
                .ToList();

            Console.WriteLine($"Trainint on best {bestEpisodes.Count} {nameof(bestEpisodes)} out of {memory.Episodes.Count}");

            var observations = bestEpisodes
                .SelectMany(x => x.Observations)
                .ToList();

            //var bestObservations = observations
            //    .OrderByDescending(x => x.Reward)
            //    .Take((int)Math.Ceiling((double)observations.Count / 10))
            //    .ToList();

            //Console.WriteLine($"Trainint on best {bestObservations.Count} {nameof(bestObservations)} out of {observations.Count}");

            return DatasetLoader.Training(BuildRawData(observations), 20);
        }

        public void TrainOnMemory(ITrainingDataset trainingData)
        {
            // Train the network
            var result = NetworkManager.TrainNetwork(_network,
                trainingData,
                TrainingAlgorithms.AdaDelta(),
                _epochs, 0.5f,
                TrackBatchProgress,
                TrainingProgress);
            Console.WriteLine("\nTraining session completed, moving to next one");

            var backupName = $"backup-network-{DateTime.Now:yyyyMMdd-HH-mm-ss}";
            _network.Save(File.Create(backupName));
            Console.WriteLine($"Backup model {backupName} saved");
        }

        public float[] Predict(float[] input, float epsilon) //epsilon = percentage of explorarion)
        {
            if (_random.NextDouble() > epsilon)
            {
                return _network.Forward(input);
            }

            var result = new float[_outputs];
            for (var i = 0; i < _outputs; i++)
            {
                result[i] = (float)_random.NextDouble();
            }

            return result;
        }

        public float[] Predict(float[] input) =>
            _network.Forward(input);

        private IEnumerable<(float[] x, float[] y)> BuildRawData(IEnumerable<Observation> bestObservations)
        {
            foreach (var observation in bestObservations)
            {
                var x = _imager.Load(observation.Images)
                    .ComposeFrames(_inputImgWidth, _inputImgHeight)
                    .InvertColors()
                    .Grayscale()
                    .Compile()
                    .Rectify()
                    .ToArray();

                var y = new float[_outputs];
                y[observation.ActionTaken] = 1;
                yield return (x, y);
            }
        }

        // Training monitor
        private static void TrackBatchProgress(BatchProgress progress)
        {
            Console.SetCursorPosition(0, Console.CursorTop);
            var n = (int)(progress.Percentage * 32 / 100); // 32 is the number of progress '=' characters to display
            var c = new char[32];
            for (var i = 0; i < 32; i++) c[i] = i <= n ? '=' : ' ';
            Console.Write($"[{new string(c)}] ");
        }

        // Training monitor 2
        private void TrainingProgress(TrainingProgressEventArgs progress)
        {
            Console.Write($"Epoch [{progress.Iteration}]/[{_epochs}]");
        }
    }
}