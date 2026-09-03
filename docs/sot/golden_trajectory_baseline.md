# SOT: Golden Trajectory Baselines & Verification Vectors

[Back to Documentation Index](../README.md) · [View PRD](../PRD.md)

---

## 1. Overview & Methodology

To ensure 100% mathematical and behavioral fidelity with Farama Gymnasium, `Gym.NET` employs **Golden Trajectory Verification**. Deterministic seed sequences are run in Python Gymnasium to generate state-action-reward golden vectors, which are then validated by automated MSTest suites.

---

## 2. Classic Control Trajectory Baselines

### 2.1 `CartPole-v1`
- **Initial State Range**: Uniform random in $[-0.05, 0.05]^4$.
- **Physics Constants**:
  - Gravity: $g = 9.8\text{ m/s}^2$
  - Mass Cart: $m_c = 1.0\text{ kg}$
  - Mass Pole: $m_p = 0.1\text{ kg}$
  - Length: $l = 0.5\text{ m}$ (half-pole)
  - Force Magnitude: $F = 10.0\text{ N}$
  - Tau (timestep): $\tau = 0.02\text{ s}$
- **Termination Thresholds**:
  - $|x| > 2.4\text{ m}$
  - $|\theta| > 12^\circ \approx 0.2094395\text{ rad}$
- **Max Steps (`TimeLimit`)**: 500 steps.

### 2.2 `Pendulum-v1`
- **Dynamics**: $\ddot{\theta} = -\frac{3g}{2l} \sin(\theta + \pi) + \frac{3}{m l^2} u$, with continuous torque $u \in [-2.0, 2.0]$.
- **Reward Function**: $r = -(\theta^2 + 0.1\dot{\theta}^2 + 0.001u^2)$, where $\theta \in [-\pi, \pi]$ (normalized angle).
- **Max Steps**: 200 steps.

### 2.3 `LunarLander-v3`
- **Deterministic Baseline (Seed 1000 with Heuristic PID)**:
  - **Expected Steps**: 245 steps.
  - **Expected Total Reward**: $35.515747$.
  - **Tolerance**: $\epsilon < 1e-5$.
