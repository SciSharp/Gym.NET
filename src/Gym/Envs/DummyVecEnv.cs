namespace Gym.Envs {
    public class DummyVecEnv : VecEnvWrapper {
        public DummyVecEnv(IEnv env) : base(env.ObservationSpace, env.ActionSpace, env) { }
    }
}