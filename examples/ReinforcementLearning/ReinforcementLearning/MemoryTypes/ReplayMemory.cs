using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;

namespace ReinforcementLearning.MemoryTypes
{
    internal abstract class ReplayMemory<TData> : IConcurrentMemory<TData>
    {
        public ConcurrentBag<Episode<TData>> Episodes { get; private set; } = new ConcurrentBag<Episode<TData>>();
        protected List<Observation<TData>> Observations = new List<Observation<TData>>();
        protected Queue<TData> DataQueue = new Queue<TData>();
        protected int CurrentId;
        protected readonly int StageFrames;
        protected readonly int EpisodesCapacity;

        protected ReplayMemory(int stageFrames, int episodesCapacity)
        {
            StageFrames = stageFrames;
            EpisodesCapacity = episodesCapacity;
        }

        protected abstract void ValidateInput(TData currentData);

        public void Memorize(int action, float currentReward, bool done)
        {
            if (DataQueue.Count < StageFrames)
            {
                return;
            }

            Observations.Add(new Observation<TData>(StageFrames)
            {
                Id = CurrentId++,
                ActionTaken = action,
                Reward = currentReward,
                Data = DataQueue.ToArray()
            });
        }

        public TData[] Enqueue(TData currentData)
        {
            ValidateInput(currentData);

            DataQueue.Enqueue(currentData);
            if (DataQueue.Count < StageFrames)
            {
                return null;
            }

            if (DataQueue.Count > StageFrames)
            {
                DataQueue.Dequeue();
            }

            return DataQueue.ToArray();
        }

        public void EndEpisode(float episodeReward)
        {
            DataQueue = new Queue<TData>();

            var lowestEpisode = Episodes.OrderBy(x => x.TotalReward).FirstOrDefault();
            if (lowestEpisode == null || episodeReward >= lowestEpisode.TotalReward || Episodes.Count < EpisodesCapacity)
            {
                Episodes.Add(new Episode<TData> { Observations = Observations.ToArray() });
            }

            if (Episodes.Count > EpisodesCapacity)
            {
                Episodes = new ConcurrentBag<Episode<TData>>(Episodes.Except(new[] { lowestEpisode }));
            }

            Observations = new List<Observation<TData>>();
        }

        public void Clear()
        {
            DataQueue = new Queue<TData>();
            Episodes = new ConcurrentBag<Episode<TData>>();
            Observations = new List<Observation<TData>>();
        }

        public void Save(string fileName, int? maxItems = null)
        {
            if (maxItems == null)
            {
                File.WriteAllText(fileName, JsonConvert.SerializeObject(Episodes, Formatting.None));
                return;
            }

            var episodes = Episodes
                .OrderByDescending(x => x.TotalReward)
                .Take(maxItems.Value);

            File.WriteAllText(fileName, JsonConvert.SerializeObject(episodes, Formatting.None));
        }
    }
}