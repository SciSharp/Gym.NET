// Gym.Internal is 0.2's code, under review for deletion (R5 of docs/plans/CSHARP_CODE_REVIEW.md): it keeps no nullable
// context until that decision (DR-54), rather than 89 annotations for code that may go.
#nullable disable
using System;

namespace Gym.Collections {
    [Serializable]
    public class DictInlineInitializationException : Exception {
        public DictInlineInitializationException() { }
        public DictInlineInitializationException(string message) : base(message) { }
        public DictInlineInitializationException(string message, Exception inner) : base(message, inner) { }
    }
}