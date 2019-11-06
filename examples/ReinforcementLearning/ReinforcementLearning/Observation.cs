namespace ReinforcementLearning {
    public class Observation<TData> {
        public int Id { get; set; }
        public TData[] Data { get; set; }
        public int ActionTaken { get; set; }
        public float Reward { get; set; }

        public Observation(int dataCount) {
            Data = new TData[dataCount];
        }
    }
}