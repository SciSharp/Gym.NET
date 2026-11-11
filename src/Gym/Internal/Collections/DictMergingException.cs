// Gym.Internal is 0.2's code, under review for deletion (R5 of docs/plans/CSHARP_CODE_REVIEW.md): it keeps no nullable
// context until that decision (DR-54), rather than 89 annotations for code that may go.
#nullable disable
using System;

namespace Gym.Collections {
    [Serializable]
    public class DictMergingException : Exception {
        public DictMergingException() { }
        public DictMergingException(string message) : base(message) { }
        public DictMergingException(string message, Exception inner) : base(message, inner) { }
    }
}