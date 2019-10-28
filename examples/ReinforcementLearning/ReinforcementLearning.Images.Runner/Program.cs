using ReinforcementLearning.GameEngines;

namespace ReinforcementLearning.Images.Runner {
    class Program {
        static void Main(string[] args) {
            new ImageGameEngine(new CartPoleConfiguration()).Play();
        }
    }
}