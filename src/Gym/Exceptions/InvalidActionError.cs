using System;
using System.Collections.Generic;
using System.Text;

namespace Gym.Exceptions
{
    public class InvalidActionError : Exception {
        public InvalidActionError() : base("Action is outside of the configured action space.") { }
        public InvalidActionError(string msg) : base(msg) { }
    }
}
