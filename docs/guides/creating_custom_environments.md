# Guide: Creating Custom Gymnasium Environments in C#

[Back to Documentation Index](../README.md) · [View PRD](../PRD.md)

---

## 1. Overview

Custom environments in `Gym.NET` inherit from `Gym.Envs.Env` and implement the standardized Gymnasium lifecycle contract.

---

## 2. Step-by-Step Implementation

### Step 1: Define Observation & Action Spaces
```csharp
using Gym.Collections;
using Gym.Core;
using Gym.Envs;
using Gym.Environments.Rendering;
using Gym.Spaces;
using NumSharp;
using SixLabors.ImageSharp;

public class SimpleCustomEnv : Env {
    private readonly NumPyRandom _random;
    private NDArray _state;
    private int _stepCount;

    public SimpleCustomEnv(string renderMode = null, IEnvironmentViewerFactoryDelegate viewerFactory = null) {
        // 1. Define discrete action space (e.g. 3 discrete actions)
        ActionSpace = new Discrete(3);

        // 2. Define bounded continuous observation space
        var low = np.array(-1.0f, -1.0f);
        var high = np.array(1.0f, 1.0f);
        ObservationSpace = new Box(low, high);

        // 3. Configure metadata and PRNG
        Metadata = new Dict("render_modes", new[] { "human", "rgb_array" }, "render_fps", 30);
        _random = np.random.RandomState();
    }

    public override (NDArray Observation, Dict Information) Reset(int? seed = null, Dict options = null) {
        if (seed.HasValue) {
            _random.seed(seed.Value);
        }
        _stepCount = 0;
        _state = _random.uniform(-0.1f, 0.1f, new Shape(2));
        var info = new Dict("initial_state", _state);
        return (_state, info);
    }

    public override StepResult Step(object action) {
        int act = (int)action;
        _stepCount++;

        // Update state dynamics based on action
        _state = _state + (float)act * 0.05f;

        float reward = 1.0f;
        bool terminated = Math.Abs((float)_state[0]) > 1.0f;
        bool truncated = _stepCount >= 200;
        var info = new Dict("step", _stepCount);

        return new StepResult(_state, reward, terminated, truncated, info);
    }

    public override Image Render(string mode = "human") {
        // Generate and return ImageSharp canvas frame
        return null;
    }

    public override void Close() {
        // Cleanup resources
    }

    public override void Seed(int seed) {
        _random.seed(seed);
    }
}
```

---

## 3. Best Practices Checklist

1. **Object Calisthenics**: Keep methods $\le 15$ lines, avoid `else` via early returns.
2. **Headless Safety**: Ensure environments execute completely without GUI initialization by default.
3. **Deterministic Seeding**: Respect `seed` parameter in `Reset()`.
4. **Tuple Deconstruction**: Return `StepResult` supporting 5-tuple deconstruction.
