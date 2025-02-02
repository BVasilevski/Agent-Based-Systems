import subprocess
import sys

if __name__ == "__main__":
    total_timesteps = int(input("Enter timesteps: "))
    learning_rate = float(input("Enter learning rate: "))
    num_steps = int(input("Enter number of steps: "))
    num_envs = int(input("Enter number of parallel environments: "))
    env_id = "CartPole-v1"
    python_executable = sys.executable

    command = [
        python_executable, "pqn.py",
        "--env-id", env_id,
        "--total-timesteps", str(total_timesteps),
        "--learning-rate", str(learning_rate),
        "--num-steps", str(num_steps),
        "--num-envs", str(num_envs)
    ]

    try:
        subprocess.run(command, text=True, capture_output=True, check=True)
        print("Model trained and saved successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running the original script: {e}")
