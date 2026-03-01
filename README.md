# ⚛️ PI-QDQN: Physics-Informed Hybrid Quantum-Classical Deep Q-Network

> **Amrita QuantumLeap Bootcamp 2026 — Hackathon**  
> Theme: Quantum Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PennyLane](https://img.shields.io/badge/PennyLane-latest-purple)](https://pennylane.ai)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🧠 Overview

This project implements a **Physics-Informed Hybrid Quantum-Classical Deep Q-Network (PI-QDQN)** to solve the CartPole-v1 control task — a canonical robotics stabilisation problem.

We replace the hidden layers of a classical DQN with a **Variational Quantum Circuit (VQC)**, achieving a **96.1% reduction in trainable parameters** while outperforming the classical baseline. Additionally, a **Hamiltonian physics loss** — inspired by Physics-Informed Neural Networks (PINNs) — is applied to guide the agent toward energy-conserving, physically valid solutions.

---

## 🏆 Key Results

| Metric | Classical DQN | PI-QDQN (Ours) |
|---|---|---|
| Trainable Parameters | 770 | **30** |
| Parameter Reduction | — | **96.1%** |
| Max Reward Achieved | 500 (unstable) | **500 ✓ (3× sustained)** |
| Physics Constraint | ❌ None | ✅ Hamiltonian H = T + V |
| Gradient Stability | Standard | Data Re-Uploading (Barren Plateau fix) |

![Results]
<img width="1202" height="695" alt="Graph_Cartpole_agent" src="https://github.com/user-attachments/assets/46f935c0-3c4c-43f1-a45f-42fd0e7275e0" />

---

## 💡 Research Contributions

### 1. Hybrid Quantum-Classical Architecture
The classical neural network's hidden layers are replaced by a 4-qubit VQC. Physical state variables `[x, ẋ, θ, θ̇]` are encoded as qubit rotation angles via `AngleEmbedding`. Entanglement between qubits captures physical correlations (e.g., pole angle ↔ angular velocity) more parameter-efficiently than classical matrix multiplication.

```
Input [x, ẋ, θ, θ̇]
    ↓
[Input Normalisation ÷ Bounds]
┌─ Re-Upload Layer × 5 ────────┐
│  AngleEmbedding (Y-rotation)  │
│  BasicEntanglerLayers (RX)    │
└───────────────────────────────┘
[PauliZ ⟨Z⟩ measurement × 4 qubits]
    ↓
[Linear: 4 → 2]  (classical)
    ↓
Q(← left),  Q(right →)
```

### 2. Data Re-Uploading (Barren Plateau Mitigation)
In deep VQCs, quantum state gradients vanish exponentially with circuit depth:

$$\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_k}\right] \propto \frac{1}{2^n}$$

We solve this with **Data Re-Uploading**: the physical state is re-injected before every entangling layer, anchoring the quantum state to the physics at each circuit depth.

### 3. Input Normalisation (Bloch Sphere Aliasing Fix)
Raw CartPole velocities (up to ±4 m/s) cause encoding angles to wrap multiple times around the Bloch sphere, destroying information. All inputs are normalised to `[-1, 1]` before quantum encoding using physical bounds `[4.8, 4.0, 0.418, 4.0]`.

### 4. Physics-Informed Hamiltonian Loss (PINN-Style)
Inspired by PINNs, we penalise violations of **energy conservation** using the CartPole Hamiltonian:

$$H = \frac{1}{2}(M+m)\dot{x}^2 + \frac{1}{2}ml^2\dot{\theta}^2 + ml\dot{x}\dot{\theta}\cos\theta - mgl\cos\theta$$

The physics loss is applied **selectively** — only near-upright states (`|θ| < 0.2 rad`) where the system approximates a conservative pendulum:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{TD}} + \lambda \cdot \text{Var}[H(\text{near-upright states})]$$

> **Why Hamiltonian and not Newtonian?**  
> The Hamiltonian is a single scalar energy functional (H = T + V), requiring minimal information encoding compared to multiple coordinate-dependent Newtonian force equations. This places less pressure on the quantum state during training — especially critical on NISQ-era hardware where information overload increases decoherence risk.

---

## 🔬 Quantum Circuit

```
0: ─╭AngleEmbedding─╭BasicEntanglerLayers─╭AngleEmbedding─╭BasicEntanglerLayers ··· ┤ <Z>
1: ─├AngleEmbedding─├BasicEntanglerLayers─├AngleEmbedding─├BasicEntanglerLayers ··· ┤ <Z>
2: ─├AngleEmbedding─├BasicEntanglerLayers─├AngleEmbedding─├BasicEntanglerLayers ··· ┤ <Z>
3: ─╰AngleEmbedding─╰BasicEntanglerLayers─╰AngleEmbedding─╰BasicEntanglerLayers ··· ┤ <Z>

(5 re-uploading layers total)
```

---

## 📁 Repository Structure

```
├── main.py                  # Full training script (single file)
├── pi_qdqn_results.png      # 4-panel result chart (generated after training)
├── cartpole_agent.gif       # Trained agent animation (generated after training)
└── README.md
```

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install pennylane torch gymnasium numpy matplotlib pillow
```

### Run Training

```bash
python main.py
```

This will:
1. Print the quantum circuit architecture
2. Train the PI-QDQN agent (600 episodes)
3. Train the classical DQN baseline (600 episodes)
4. Save `pi_qdqn_results.png` — 4-panel comparison chart
5. Save `cartpole_agent.gif` — animation of the trained agent

---

## ⚙️ Configuration

All hyperparameters are at the top of `main.py`:

| Parameter | Value | Description |
|---|---|---|
| `N_QUBITS` | 4 | Matches CartPole state dimension |
| `N_LAYERS` | 5 | Re-uploading VQC depth |
| `LR` | 0.003 | Adam learning rate |
| `EPISODES` | 600 | Max training episodes |
| `TARGET_UPDATE` | 10 | Target network sync interval |
| `LAMBDA_PHYSICS` | 0.1 | Hamiltonian loss weight |
| `THETA_THRESHOLD` | 0.2 rad | Near-upright physics gate |

---

## 🧩 Technical Stack

- **[PennyLane](https://pennylane.ai)** — Quantum circuit definition, simulation, and `TorchLayer` integration
- **[PyTorch](https://pytorch.org)** — Autograd, optimisation, hybrid classical layers
- **[Gymnasium](https://gymnasium.farama.org)** — CartPole-v1 environment
- **[Matplotlib](https://matplotlib.org)** — Result visualisation and GIF animation

---

## 📐 Physics Background

CartPole is governed by the Lagrangian mechanics of a cart-pole system. The Hamiltonian (total mechanical energy) is:

$$H = T + V$$

Where:
- **T** (Kinetic): $\frac{1}{2}(M+m)\dot{x}^2 + \frac{1}{2}ml^2\dot{\theta}^2 + ml\dot{x}\dot{\theta}\cos\theta$
- **V** (Potential): $-mgl\cos\theta$

When the pole is near-upright, the system approximates a conservative pendulum. Our physics loss enforces this conservation law during learning, guiding the agent toward physically valid control strategies rather than purely reward-maximising ones.

---

## 🎓 Context

Built as part of the **Amrita QuantumLeap Bootcamp 2026 Hackathon** under the Quantum Machine Learning theme at Amrita Vishwa Vidyapeetham, Kollam.

---

## 📜 License

MIT License — free to use, modify, and distribute.

