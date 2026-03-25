import os
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

from env.quadruple_env import QuadrupleInvertedPendulumEnv


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results" / "v2_quadruple"
LOG_DIR = ROOT_DIR / "logs" / "quadruple_ppo"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def train_and_test(total_timesteps=5_000_000, model_name="quadruple_pendulum_ppo"):
    model_path = MODELS_DIR / model_name
    summary_path = RESULTS_DIR / "test_summary.txt"

    vec_env = make_vec_env(QuadrupleInvertedPendulumEnv, n_envs=1)

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        learning_rate=1e-4,
        gamma=0.99,
        tensorboard_log=str(LOG_DIR),
    )

    print(f"Training PPO agent for quadruple inverted pendulum: {total_timesteps} steps")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save(str(model_path))
    print(f"Model saved to: {model_path}.zip")

    print("Testing trained model...")

    test_env = QuadrupleInvertedPendulumEnv()
    total_reward_sum = 0.0
    num_episodes = 5
    max_test_steps = 10000

    results = []

    for ep in range(num_episodes):
        obs, _ = test_env.reset()
        episode_reward = 0.0

        for step in range(max_test_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        total_reward_sum += episode_reward
        results.append(f"Episode {ep + 1}: reward={episode_reward:.2f}, steps={step + 1}")
        print(results[-1])

    avg_reward = total_reward_sum / num_episodes
    results.append(f"Average reward: {avg_reward:.2f}")

    with open(summary_path, "w", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Test summary saved to: {summary_path}")


if __name__ == "__main__":
    train_and_test()
