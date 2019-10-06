﻿using System;
using System.IO;
using System.Threading;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.SupervisedLearning.Progress;
using SixLabors.ImageSharp.PixelFormats;

namespace ReinforcementLearning
{
    public class Trainer
    {
        private INeuralNetwork _network;
        private readonly Random _random = new Random();
        private readonly DatasetBuilder _datasetBuilder;
        private readonly int _outputs;
        private readonly int _epochs;

        public Trainer(int inputImgWidth, int inputImgHeight, int outputs, int batchSize, int epochs)
        {
            _outputs = outputs;
            _epochs = epochs;
            _datasetBuilder = new DatasetBuilder(inputImgWidth, inputImgHeight, outputs, batchSize);

            if (CuDnnNetworkLayers.IsCudaSupportAvailable)
            {
                Console.ForegroundColor = ConsoleColor.DarkGreen;
                Console.WriteLine("Cuda Gpu support available");
                Console.ResetColor();

                _network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(inputImgHeight, inputImgWidth),
                    CuDnnNetworkLayers.Convolutional((3, 3), 40, ActivationType.Identity),
                    CuDnnNetworkLayers.Softmax(outputs));
                return;
            }

            _network = NetworkManager.NewSequential(TensorInfo.Image<Alpha8>(inputImgHeight, inputImgWidth),
                NetworkLayers.Convolutional((3, 3), 40, ActivationType.Identity),
                NetworkLayers.Softmax(outputs));
        }

        public void StartAsyncTraining(IConcurrentMemory memory, CancellationToken ct = default)
        {
            new Thread(() =>
            {
                while (!ct.IsCancellationRequested)
                {
                    TrainOnMemory(memory);
                }
            }).Start();
        }

        public void Load(Stream modelStream)
        {
            using (var backupStream = new MemoryStream())
            {
                modelStream.CopyTo(backupStream);
                modelStream.Seek(0, SeekOrigin.Begin);
                _network = NetworkLoader.TryLoad(modelStream, ExecutionModePreference.Cuda);
                if (_network == null)
                {
                    backupStream.Seek(0, SeekOrigin.Begin);
                    _network = NetworkLoader.TryLoad(backupStream, ExecutionModePreference.Cpu);
                }
            }
            if (_network == null)
            {
                throw new InvalidOperationException($"Cannot load model");
            }
        }

        public void TrainOnMemory(IConcurrentMemory memory)
        {
            var trainingData = _datasetBuilder.BuildDataset(memory);
            if (trainingData.Count == 0)
            {
                return;
            }

            var clonedInstance = _network.Clone();
            // Train the network
            var result = NetworkManager.TrainNetwork(clonedInstance,
                trainingData,
                TrainingAlgorithms.AdaDelta(),
                _epochs, 0.5f,
                TrackBatchProgress,
                TrainingProgress);

            Console.WriteLine("\nTraining session completed, moving to next one");

            var backupName = $"backup-network-{DateTime.Now:yyyyMMdd-HH-mm-ss}.modl";
            _network.Save(File.Create(backupName));
            Console.WriteLine($"Backup model {backupName} saved");
            _network = clonedInstance;
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

        // Training monitor
        private static void TrackBatchProgress(BatchProgress progress)
        {
            //Console.SetCursorPosition(0, Console.CursorTop);
            //var n = (int)(progress.Percentage * 32 / 100); // 32 is the number of progress '=' characters to display
            //var c = new char[32];
            //for (var i = 0; i < 32; i++) c[i] = i <= n ? '=' : ' ';
            //Console.Write($"[{new string(c)}] ");
        }

        // Training monitor 2
        private void TrainingProgress(TrainingProgressEventArgs progress)
        {
            //Console.Write($"Epoch [{progress.Iteration}]/[{_epochs}]");
        }
    }
}