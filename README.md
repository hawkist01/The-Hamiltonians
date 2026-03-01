# The-Hamiltonians
Amrita QuantumLeap Bootcamp 2026- Hackathon

⚛️ PI-QDQN: Physics-Informed Hybrid Quantum-Classical Deep Q-Network
Amrita QuantumLeap Bootcamp 2026 — Hackathon Submission
Theme: Quantum Machine Learning

🧠 Project Overview
This project implements a Physics-Informed Hybrid Quantum-Classical Deep Q-Network (PI-QDQN) to solve the CartPole-v1 robotic control problem.

We replace the hidden layers of a classical Deep Q-Network (DQN) with a Variational Quantum Circuit (VQC), reducing trainable parameters by 96.1% while achieving superior performance. Additionally, we incorporate a Hamiltonian-based physics loss (PINN-style) to ground the agent in the actual mechanics of the CartPole system.

🔬 Research Contributions
1. Hybrid Quantum-Classical Architecture
Classical DQN hidden layers are replaced by a 4-qubit VQC using PennyLane

AngleEmbedding encodes the 4 CartPole state variables as Y-rotations on qubits

BasicEntanglerLayers creates trainable entanglement between qubits (RX + circular CNOT)

Classical nn.Linear(4→2) maps quantum measurements to Q-values

2. Data Re-Uploading (Barren Plateau Mitigation)
The CartPole state is re-injected before every entangling layer

Prevents the quantum state from drifting into a Barren Plateau where:

\text{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta_k}\right] \propto \frac{1}{2^n}

Anchors the Bloch sphere trajectory to physics at each circuit depth

3. Input Normalisation (Bloch Sphere Aliasing Fix)
Raw CartPole velocities (±4 m/s) would wrap around the Bloch sphere >1 time

All state variables normalised to [-1, 1] using physical bounds before encoding

Ensures non-aliased, injective mapping from state space to Hilbert space

4. Selective Hamiltonian Physics Loss (PINN-Style)
Inspired by Physics-Informed Neural Networks (PINNs)

The CartPole Hamiltonian (total mechanical energy) is computed per batch:

H = \frac{1}{2}(M+m)\dot{x}^2 + \frac{1}{2}ml^2\dot{\theta}^2 + ml\dot{x}\dot{\theta}\cos\theta - mgl\cos\theta

Physics constraint applied only when |θ| < 0.2 rad (near-upright), where energy is conserved

Loss: $\mathcal{L}{\text{total}} = \mathcal{L}{\text{TD}} + \lambda \cdot \text{Var}[H(\text{states})]$

Hamiltonian formulation chosen over Newtonian — single scalar energy term minimises information pressure on qubits

5. Target Network Stabilisation
Dual-network DQN: policy net trains every step, target net syncs every 10 episodes

Stops Q-value bootstrapping from chasing a moving target (eliminates oscillation)

📊 Results
Metric	Classical DQN	PI-QDQN (Ours)
Trainable Parameters	770	30
Parameter Reduction	—	96.1%
Max Reward Achieved	~200	500 (×3 times)
Energy Conservation	✗	✅ Hamiltonian → 0
Barren Plateau Risk	N/A	✅ Mitigated
Key finding: At episode ~500, the Hamiltonian variance collapses to near-zero simultaneously with the reward breakthrough — proving the agent learned physically valid energy-conserving behaviour, not just a statistical shortcut.

🗂️ Repository Structure
text
.
├── main.py                  # Full training + evaluation code (single file)
├── pi_qdqn_results.png      # 4-panel results chart (reward, params, physics loss, arch)
├── cartpole_agent.gif       # Trained PI-QDQN agent animation
└── README.md                # This file
⚙️ Installation
bash
pip install pennylane torch gymnasium numpy matplotlib pillow
Python 3.9+ recommended.

🚀 Running the Project
bash
python main.py
This will:

Print the quantum circuit architecture to terminal

Train the PI-QDQN agent (600 episodes, ~10–20 min on CPU)

Train the Classical DQN baseline for comparison

Save pi_qdqn_results.png — the 4-panel results chart

Save cartpole_agent.gif — animation of the trained agent balancing

🔧 Key Hyperparameters
Parameter	Value	Reason
N_QUBITS	4	Matches CartPole state dimension
N_LAYERS	5	Deeper VQC for richer entanglement
LR	0.003	Low LR for stable quantum gradient updates
LAMBDA_PHYSICS	0.1	Physics loss weight
THETA_THRESHOLD	0.2 rad	Selective physics region (near-upright only)
TARGET_UPDATE	10 ep	Target network sync frequency
BATCH_SIZE	32	Larger batches for stable Bellman targets
🧱 Architecture Diagram
text
Input: [x, ẋ, θ, θ̇]  (4 physical state variables)
       ↓
[Input Normalisation ÷ STATE_BOUNDS]
┌─ Re-Upload Layer × 5 ──────────────────────────────────┐
│  AngleEmbedding(inputs × π, rotation="Y")              │
│  BasicEntanglerLayers(weights[layer], rotation=RX)     │
└────────────────────────────────────────────────────────┘
[PauliZ ⟨Z⟩ Measurement × 4 qubits]  →  output ∈ [-1, 1]⁴
       ↓
[Classical Linear: 4 → 2]
       ↓
Q(← left),  Q(right →)
📐 Quantum Circuit
The data re-uploading VQC alternates encoding and entanglement for all 5 layers:

text
0: ─╭AngleEmb─╭Entangle─╭AngleEmb─╭Entangle─ ··· ─╭Entangle─┤ ⟨Z⟩
1: ─├AngleEmb─├Entangle─├AngleEmb─├Entangle─ ··· ─├Entangle─┤ ⟨Z⟩
2: ─├AngleEmb─├Entangle─├AngleEmb─├Entangle─ ··· ─├Entangle─┤ ⟨Z⟩
3: ─╰AngleEmb─╰Entangle─╰AngleEmb─╰Entangle─ ··· ─╰Entangle─┤ ⟨Z⟩
👥 Team
Amrita Vishwa Vidyapeetham — M.Tech AI
Submitted for: Amrita QuantumLeap Bootcamp 2026 Hackathon
Theme: Quantum Machine Learning

📚 References
Pérez-Salinas et al. (2020) — Data re-uploading for a universal quantum classifier

McClean et al. (2018) — Barren plateaus in quantum neural network training landscapes

Skolik et al. (2021) — Layerwise learning for quantum neural networks

Chen et al. (2022) — Variational Quantum Circuits for Deep Reinforcement Learning

Raissi et al. (2019) — Physics-informed neural networks (PINNs)

PennyLane Documentation — https://pennylane.ai
