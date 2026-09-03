# Gymnasium Source of Truth (SOT) Reference

## 1. Overview & Architectural Alignment

**Gym.NET** is a pure C# (.NET) port of reinforcement learning environment toolkits originally modeled after `openai/gym`. 
Following OpenAI's transition and deprecation of the original Gym repository, the **Farama Foundation Gymnasium** project (`https://github.com/Farama-Foundation/Gymnasium` / `https://gymnasium.farama.org/`) is the canonical, actively maintained **Source of Truth (SOT)** for environment interfaces, space specifications, and standard reference environments.

---

## 2. Canonical Submodule Topography

The canonical Gymnasium Python source is registered as a direct Git submodule under `refs/Gymnasium`:

- **Submodule Path:** `refs/Gymnasium`
- **Upstream Repository:** `https://github.com/Farama-Foundation/Gymnasium.git`
- **Official Documentation:** `https://gymnasium.farama.org/index.html`

---

## 3. Key Gymnasium API Evolution & SOT Mapping

### 3.1 Step Function Signature Transition
- **Legacy OpenAI Gym (v0.21 and earlier):**
  $$\text{step}(\text{action}) \to (\text{observation}, \text{reward}, \text{done}, \text{info})$$
- **Modern Farama Gymnasium (v0.26+ / v1.0+):**
  $$\text{step}(\text{action}) \to (\text{observation}, \text{reward}, \text{terminated}, \text{truncated}, \text{info})$$
  - `terminated`: Indicates the MDP reached a terminal state (e.g. pole fell, task accomplished).
  - `truncated`: Indicates the episode was stopped due to an out-of-MDP constraint (e.g. time limit reached / max steps exceeded).

### 3.2 Reset Function Signature Transition
- **Legacy OpenAI Gym:**
  $$\text{reset}() \to \text{observation}$$
- **Modern Farama Gymnasium:**
  $$\text{reset}(\text{seed}=\text{None}, \text{options}=\text{None}) \to (\text{observation}, \text{info})$$

### 3.3 Classic Control Environments SOT
Reference implementations for classic control environments are located in:
- `refs/Gymnasium/gymnasium/envs/classic_control/cartpole.py`
- `refs/Gymnasium/gymnasium/envs/classic_control/pendulum.py`
- `refs/Gymnasium/gymnasium/envs/classic_control/mountain_car.py`
- `refs/Gymnasium/gymnasium/envs/classic_control/acrobot.py`
