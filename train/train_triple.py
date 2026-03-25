import os
import sys
from pathlib import Path
import io
from base64 import b64encode

import moviepy.editor as mpy
from IPython.display import HTML

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# 상위 폴더 import 가능하게 설정
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))

from env.triple_env import TripleInvertedPendulumEnv


MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results" / "v1_triple"
LOG_DIR = ROOT_DIR / "logs" / "triple_ppo"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def display_video(file_name, width=700):
    if not os.path.exists(file_name):
        print(f"Video file not found: {file_name}")
        return None

    video_file = io.open(file_name, "r+b").read()
    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    html_code = f"""
    <video controls autoplay loop width="{width}">
        <source src="{video_url}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    """
    return HTML(html_code)


def train_and_visualize(
    total_timesteps=500_000,
    model_name="triple_pendulum_ppo",
    video_name="triple_pendulum_control.mp4",
    max_test_steps=5000,
):
    model_path = MODELS_DIR / model_name
    video_path = RESULTS_DIR / video_name

    vec_env = make_vec_env(TripleInvertedPendulumEnv, n_envs=1)

    if not os.path.exists(f"{model_path}.zip"):
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            tensorboard_log=str(LOG_DIR),
        )
        print(f"Training PPO agent for triple inverted pendulum: {total_timesteps} steps")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(str(model_path))
        print(f"Model saved to: {model_path}.zip")
    else:
        print(f"Loading existing model: {model_path}.zip")
        model = PPO.load(str(model_path))

    print("Recording simulation video...")

    test_env = TripleInvertedPendulumEnv(render_mode="rgb_array")
    obs, _ = test_env.reset()
    frames = []

    episode_reward = 0.0

    for step in range(max_test_steps):
        frame = test_env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        episode_reward += reward

        if terminated or truncated:
            print(f"Episode finished at step {step + 1}, total reward: {episode_reward:.2f}")
            break

    test_env.close()

    if not frames:
        print("No frames were generated. Check render settings.")
        return None

    fps = test_env.metadata["render_fps"]
    clip = mpy.ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(
        str(video_path),
        fps=fps,
        verbose=False,
        logger=None,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    print(f"Video saved to: {video_path}")
    return display_video(str(video_path))


if __name__ == "__main__":
    train_and_visualize()
