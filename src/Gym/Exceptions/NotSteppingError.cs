using System;

namespace Gym.Exceptions {
    public class NotSteppingError : Exception {
        public NotSteppingError() : base("not running an async step") { }
    }
}