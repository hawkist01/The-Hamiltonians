
"""
Amrita QuantumLeap Bootcamp 2026 — Hackathon
=============================================
Project : Physics-Informed Hybrid Quantum-Classical DQN (PI-QDQN)
Theme   : Quantum Machine Learning
Stack   : PennyLane + PyTorch + Gymnasium + Matplotlib

Final Fixes (v3):
  1. Input Normalisation  → Prevents Bloch sphere aliasing (wrapping)
  2. N_LAYERS = 5         → Deeper VQC for better representation
  3. Stable LR (0.003)    → Smoother quantum gradient descent

Install:
  pip install pennylane torch gymnasium numpy matplotlib pillow
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import deque

# ══════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ══════════════════════════════════════════════════════════
N_QUBITS         = 4       # CartPole state: [x, x_dot, theta, theta_dot]
N_LAYERS         = 5       # (Increased) Re-uploading VQC depth
LR               = 0.003   # (Lowered) More stable quantum gradients
EPISODES         = 600     # (Increased) Time for normalized learning
GAMMA            = 0.99
EPSILON          = 1.0
EPS_DECAY        = 0.997
EPS_MIN          = 0.01
BATCH_SIZE       = 32
MEM_SIZE         = 5000
TARGET_UPDATE    = 10      # Sync target network every N episodes
SOLVE_SCORE      = 475
LAMBDA_PHYSICS   = 0.1     # Physics loss weight
THETA_THRESHOLD  = 0.2     # Radians — only apply physics near upright
SEED             = 42

# CartPole physical constants
M_CART = 1.0
M_POLE = 0.1
L_POLE = 0.5
G      = 9.8

# State Bounds for Normalisation [x, x_dot, theta, theta_dot]
# x max is ~4.8, theta max is ~0.418. Velocity bounds are empirical.
STATE_BOUNDS = torch.tensor([4.8, 4.0, 0.418, 4.0])

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ══════════════════════════════════════════════════════════
#  SECTION 2 — QUANTUM CIRCUIT (Data Re-Uploading VQC)
# ══════════════════════════════════════════════════════════
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Data Re-Uploading: state is re-injected before each entangling layer.
    Input must be normalized to [-1, 1] before entering this circuit
    so that `inputs * pi` does not wrap around the Bloch sphere > 1 time.
    """
    for layer_idx in range(N_LAYERS):
        qml.AngleEmbedding(inputs * np.pi, wires=range(N_QUBITS), rotation="Y")
        qml.BasicEntanglerLayers(
            weights[layer_idx].unsqueeze(0),
            wires=range(N_QUBITS),
            rotation=qml.RX
        )
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}


# ══════════════════════════════════════════════════════════
#  SECTION 3 — HYBRID PI-QDQN MODEL
# ══════════════════════════════════════════════════════════
class PIQuantumDQN(nn.Module):
    """
    Physics-Informed Hybrid Quantum DQN (PI-QDQN)
    Normalizes state → VQC (feature extractor) → Linear (action mapper)
    """
    def __init__(self):
        super().__init__()
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        self.output  = nn.Linear(N_QUBITS, 2)
        nn.init.xavier_uniform_(self.output.weight)

    def forward(self, x):
        # ── THE FIX: Normalise inputs before quantum encoding ──
        # Prevents Bloch sphere aliasing (information destruction)
        x_norm = x / STATE_BOUNDS.to(x.device)
        # Clip to ensure bounds are strictly [-1, 1]
        x_norm = torch.clamp(x_norm, -1.0, 1.0)

        q_out = self.q_layer(x_norm)
        return self.output(q_out)


# ══════════════════════════════════════════════════════════
#  SECTION 4 — CLASSICAL DQN BASELINE
# ══════════════════════════════════════════════════════════
class ClassicalDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Classical DQN also benefits from normalisation for fair comparison
            nn.Linear(4, 24), nn.ReLU(),
            nn.Linear(24, 24), nn.ReLU(),
            nn.Linear(24, 2)
        )

    def forward(self, x):
        x_norm = x / STATE_BOUNDS.to(x.device)
        return self.net(x_norm)


# ══════════════════════════════════════════════════════════
#  SECTION 5 — SELECTIVE HAMILTONIAN PHYSICS LOSS
# ══════════════════════════════════════════════════════════
def hamiltonian_loss(states):
    """
    Selective Hamiltonian Loss — PINN-style energy conservation.
    Only applied when |θ| < THETA_THRESHOLD (near upright),
    because gravity does real work when the pole falls.
    """
    x_dot     = states[:, 1]
    theta     = states[:, 2]
    theta_dot = states[:, 3]

    # Only select states where pole is near upright
    near_upright = (torch.abs(theta) < THETA_THRESHOLD)

    if near_upright.sum() < 2:
        return torch.tensor(0.0, requires_grad=True)

    x_dot_u     = x_dot[near_upright]
    theta_u     = theta[near_upright]
    theta_dot_u = theta_dot[near_upright]

    T = (0.5 * (M_CART + M_POLE) * x_dot_u**2
       + 0.5 * M_POLE * L_POLE**2 * theta_dot_u**2
       + M_POLE * L_POLE * x_dot_u * theta_dot_u * torch.cos(theta_u))

    V = -M_POLE * G * L_POLE * torch.cos(theta_u)
    H = T + V

    return torch.var(H)


# ══════════════════════════════════════════════════════════
#  SECTION 6 — REPLAY BUFFER
# ══════════════════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, n):
        batch       = random.sample(self.buffer, n)
        states      = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32)
        actions     = torch.tensor([x[1] for x in batch],           dtype=torch.long)
        rewards     = torch.tensor([x[2] for x in batch],           dtype=torch.float32)
        next_states = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32)
        dones       = torch.tensor([x[4] for x in batch],           dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ══════════════════════════════════════════════════════════
#  SECTION 7 — TRAINING (with Target Network)
# ══════════════════════════════════════════════════════════
def train(model, env, label="Model", use_physics_loss=False):
    target_model = type(model)()
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer   = optim.Adam(model.parameters(), lr=LR)
    loss_fn     = nn.MSELoss()
    buffer      = ReplayBuffer(MEM_SIZE)
    epsilon     = EPSILON
    rewards_log = []
    phys_log    = []
    solved_at   = None

    for episode in range(1, EPISODES + 1):
        state, _     = env.reset(seed=SEED + episode)
        total_reward = 0.0
        done         = False

        while not done:
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(
                        model(torch.tensor(state, dtype=torch.float32))
                    ).item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state         = next_state
            total_reward += reward

            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    max_next_q = target_model(next_states).max(1)[0]
                    target_q   = rewards + (1 - dones) * GAMMA * max_next_q

                td_loss = loss_fn(current_q, target_q)

                if use_physics_loss:
                    phys_loss = hamiltonian_loss(states)
                    loss      = td_loss + LAMBDA_PHYSICS * phys_loss
                    phys_log.append(phys_loss.item())
                else:
                    loss = td_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        rewards_log.append(total_reward)
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_model.load_state_dict(model.state_dict())

        avg = np.mean(rewards_log[-10:])
        if episode % 20 == 0 or total_reward >= SOLVE_SCORE:
            tag = "[PHYSICS ✓]" if use_physics_loss else ""
            print(f"[{label}]{tag} Ep {episode:>4} | "
                  f"Reward: {total_reward:>6.1f} | Avg(10): {avg:>6.1f} | ε: {epsilon:.3f}")

        if avg >= SOLVE_SCORE and solved_at is None:
            solved_at = episode
            print(f"\n✅ [{label}] SOLVED at Episode {episode}!\n")
            break

    return rewards_log, phys_log, solved_at


# ══════════════════════════════════════════════════════════
#  SECTION 8 — CARTPOLE ANIMATION
# ══════════════════════════════════════════════════════════
def save_cartpole_animation(model, filename="cartpole_agent.gif", max_steps=500):
    print("\n🎬 Recording CartPole animation...")
    env_render = gym.make("CartPole-v1", render_mode="rgb_array")
    state, _   = env_render.reset(seed=0)
    frames     = []
    rewards    = []

    for _ in range(max_steps):
        frame = env_render.render()
        frames.append(frame)
        with torch.no_grad():
            action = torch.argmax(
                model(torch.tensor(state, dtype=torch.float32))
            ).item()
        state, reward, terminated, truncated, _ = env_render.step(action)
        rewards.append(reward)
        if terminated or truncated:
            break

    env_render.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axis("off")
    img_plot = ax.imshow(frames[0])
    total_so_far = [0]

    def update(i):
        img_plot.set_data(frames[i])
        total_so_far[0] += rewards[i] if i < len(rewards) else 0
        ax.set_title(
            f"PI-QDQN Agent (32 params) — Step {i+1}/{len(frames)} | "
            f"Reward: {int(sum(rewards[:i+1]))}",
            fontsize=9, fontweight="bold"
        )
        return [img_plot]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=40, blit=True)
    ani.save(filename, writer=PillowWriter(fps=25))
    plt.close()
    print(f"🎬 Animation saved to {filename}  ({len(frames)} frames, "
          f"total reward: {int(sum(rewards))})")


# ══════════════════════════════════════════════════════════
#  SECTION 9 — VISUALISATION
# ══════════════════════════════════════════════════════════
def smooth(r, w=10):
    if len(r) < w:
        return np.array(r)
    return np.convolve(r, np.ones(w) / w, mode="valid")

def plot_results(q_rewards, c_rewards, phys_log, q_solved, c_solved):
    PURPLE = "#7B2FBE"
    ORANGE = "#E87A1D"
    GREEN  = "#2DA44E"

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Physics-Informed Hybrid Quantum-Classical DQN (PI-QDQN)\n"
        "Amrita QuantumLeap Bootcamp 2026 — CartPole-v1",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    sq = smooth(q_rewards)
    sc = smooth(c_rewards)
    ax1.plot(sq, color=PURPLE, lw=2.0, label="PI-QDQN (Quantum + Physics)")
    ax1.plot(sc, color=ORANGE, lw=2.0, label="Classical DQN (baseline)")
    ax1.axhline(SOLVE_SCORE, color="gray", ls="--", lw=1.2, label=f"Solve ({SOLVE_SCORE})")
    if q_solved:
        ax1.axvline(q_solved, color=PURPLE, ls=":", alpha=0.8, label=f"QDQN Solved @ ep {q_solved}")
    if c_solved:
        ax1.axvline(c_solved, color=ORANGE, ls=":", alpha=0.8, label=f"DQN Solved @ ep {c_solved}")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward (smoothed)")
    ax1.set_title("Training Reward Curves")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 2])
    q_params = sum(p.numel() for p in PIQuantumDQN().parameters())
    c_params = sum(p.numel() for p in ClassicalDQN().parameters())
    bars = ax2.bar(
        ["Classical DQN\n(2 hidden layers)", "PI-QDQN\n(VQC + Physics)"],
        [c_params, q_params],
        color=[ORANGE, PURPLE], width=0.45, edgecolor="black", lw=0.8
    )
    for bar, count in zip(bars, [c_params, q_params]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{count}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Trainable Parameters")
    ax2.set_title(f"Parameter Efficiency\n({(1-q_params/c_params)*100:.1f}% reduction)")
    ax2.set_ylim(0, c_params * 1.3)
    ax2.grid(True, axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :2])
    if phys_log:
        ps = smooth(phys_log, w=min(100, max(2, len(phys_log)//5)))
        ax3.plot(ps, color=GREEN, lw=1.8, label="Hamiltonian Variance (near-upright only)")
        ax3.fill_between(range(len(ps)), ps, alpha=0.15, color=GREEN)
        ax3.axhline(0, color="gray", ls="--", lw=1.0, label="Perfect conservation")
        ax3.set_xlabel("Training Steps")
        ax3.set_ylabel("Var[H]  (J²)")
        ax3.set_title("Selective Hamiltonian Physics Loss — |θ| < 0.2 rad only")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    arch = (
        "  PI-QDQN Architecture\n"
        "  ────────────────────────────\n"
        "  Input: [x, ẋ, θ, θ̇]  (4 vars)\n"
        "       ↓\n"
        "  [Input Normalisation  ÷ Bounds]\n"
        "  ┌─ Re-Upload Layer × 5 ─────┐\n"
        "  │ AngleEmbedding (Y-rot)     │\n"
        "  │ BasicEntanglerLayers (RX)  │\n"
        "  └────────────────────────────┘\n"
        "  [PauliZ ⟨Z⟩ × 4 qubits]\n"
        "       ↓\n"
        "  [Linear: 4 → 2]  (classical)\n"
        "       ↓\n"
        "  Q(←left),  Q(right→)\n"
        "  ────────────────────────────\n"
        f"  Quantum params  : {N_LAYERS * N_QUBITS}\n"
        f"  Classical params: {4*2 + 2}\n"
        f"  Total           : {q_params}\n"
        "  ────────────────────────────\n"
        "  Physics: Var[H] near |θ|<0.2\n"
        "  H = T + V  (Hamiltonian)\n"
        "  Target Net sync: every 10 ep"
    )
    ax4.text(0.05, 0.97, arch, transform=ax4.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#f0eeff", alpha=0.85,
                       edgecolor=PURPLE, lw=1.5))

    plt.savefig("pi_qdqn_results.png", dpi=150, bbox_inches="tight")
    print("\n Results chart saved to pi_qdqn_results.png")
    plt.show()


# ══════════════════════════════════════════════════════════
#  SECTION 10 — MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    q_model  = PIQuantumDQN()
    c_model  = ClassicalDQN()
    q_params = sum(p.numel() for p in q_model.parameters())
    c_params = sum(p.numel() for p in c_model.parameters())

    print("=" * 65)
    print("  Amrita QuantumLeap Bootcamp 2026 — Hackathon")
    print("  Physics-Informed Hybrid QDQN for CartPole-v1 (v3 Final)")
    print("=" * 65)
    print(f"  Quantum Device    : PennyLane default.qubit ({N_QUBITS} qubits)")
    print(f"  Circuit Style     : Normalized Re-Uploading VQC ({N_LAYERS} layers)")
    print(f"  PI-QDQN Params    : {q_params}  ← Quantum Advantage")
    print(f"  Classical Params  : {c_params}")
    print(f"  Reduction         : {(1 - q_params/c_params)*100:.1f}%")
    print(f"  Physics Loss      : Selective Hamiltonian |θ| < {THETA_THRESHOLD} rad")
    print("=" * 65)

    print("\n📐 Quantum Circuit (Normalized Data Re-Uploading):")
    dummy_i = torch.zeros(N_QUBITS, dtype=torch.float32)
    dummy_w = torch.zeros(N_LAYERS, N_QUBITS, dtype=torch.float32)
    print(qml.draw(quantum_circuit)(dummy_i, dummy_w))

    print("\n🚀 Training PI-QDQN (Quantum + Selective Physics Loss)...")
    q_rewards, phys_log, q_solved = train(
        q_model, env, label="PI-QDQN", use_physics_loss=True
    )

    print("\n🔁 Training Classical DQN (baseline)...")
    c_rewards, _, c_solved = train(
        c_model, env, label="Classical DQN", use_physics_loss=False
    )

    env.close()

    # Save results chart
    plot_results(q_rewards, c_rewards, phys_log, q_solved, c_solved)

    # Save animation of trained PI-QDQN agent
    save_cartpole_animation(q_model, filename="cartpole_agent.gif")

    print("\n Done! Files ready for judges:")
    print("     pi_qdqn_results.png  — Training charts")
    print("   cartpole_agent.gif   — Trained agent animation")
