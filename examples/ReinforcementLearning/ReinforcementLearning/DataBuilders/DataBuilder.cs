using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworkNET.APIs;
using NeuralNetworkNET.APIs.Interfaces.Data;
using ReinforcementLearning.GameConfigurations;
using ReinforcementLearning.MemoryTypes;

namespace ReinforcementLearning.DataBuilders {
    public abstract class DataBuilder<TConfiguration, TData>
        where TConfiguration : IGameConfiguration {
        protected readonly TConfiguration Configuration;
        protected readonly int Outputs;

        protected readonly Dictionary<int, (float[] x, float[] y)> ObservationDictionary =
            new Dictionary<int, (float[] x, float[] y)>();

        protected DataBuilder(TConfiguration configuration, int outputs) {
            Configuration = configuration;
            Outputs = outputs;
        }

        public abstract float[] BuildInput(TData[] dataGroup);

        public ITrainingDataset BuildDataset(IConcurrentMemory<TData> memory) {
            if (memory.Episodes.Count < Configuration.MemoryCapacity) {
                return default;
            }

            var episodes = memory.Episodes;

            var observations = episodes
                .SelectMany(x => x.Observations.Take(x.Observations.Length * 2 / 3))
                .ToList();

            Console.WriteLine($"Training on best {episodes.Count} episodes out of {memory.Episodes.Count}. {observations.Count} observations");

            return DatasetLoader.Training(BuildRawData(observations), Configuration.BatchSize);
        }

        private IEnumerable<(float[] x, float[] y)> BuildRawData(IEnumerable<Observation<TData>> bestObservations) {
            foreach (var observation in bestObservations) {
                if (ObservationDictionary.ContainsKey(observation.Id)) {
                    yield return ObservationDictionary[observation.Id];
                    continue;
                }

                var x = BuildInput(observation.Data);
                var y = new float[Outputs];
                y[observation.ActionTaken] = 1;
                ObservationDictionary.Add(observation.Id, (x, y));
                yield return (x, y);
            }
        }
    }
}