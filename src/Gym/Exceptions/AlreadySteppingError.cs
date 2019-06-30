using System;

namespace Gym.Exceptions {
    // 
    //     Raised when an asynchronous step is running while
    //     step_async() is called again.
    //     
    public class AlreadySteppingError : Exception {
        public AlreadySteppingError() : base("already running an async step") { }
    }
}