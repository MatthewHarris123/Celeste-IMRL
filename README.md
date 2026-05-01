# Celeste AI Gym

A Gymnasium-compatible environment for Celeste gameplay, providing an interface for reinforcement learning agents.

## Overview

This repository contains a custom Gym environment that interfaces with Celeste through TCP communication with the companion Celeste AI mod. This is a base environment implementation designed to be integrated with your own training code.

## Features

- **Custom Gym Environment**: Full gymnasium-compatible environment for Celeste gameplay
- **Real-time Interface**: Direct communication with Celeste game through TCP socket
- **Flexible Observation Space**: RGB frame observations from the game
- **Multi-Binary Actions**: Support for all Celeste inputs (movement, jump, dash, grab)

## Architecture

```
Python RL Environment (celeste-ai-gym)
    ↕ TCP Socket Communication (port 5000)
Celeste Game + AI Mod (celeste-ai-mod)
```

## Installation

### Prerequisites

- Python 3.8+
- Celeste game with the companion AI mod installed

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd celeste-ai-gym
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Testing the Environment

```bash
python debug_env.py
```

This will connect to a running Celeste instance and allow you to test the environment interface.

### Using in Your Own Code

```python
from CelesteEnv import CelesteEnv

# Create environment
env = CelesteEnv()

# Reset environment
obs, info = env.reset()

# Take actions
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

# Close when done
env.close()
```

## Environment Details

### Observation Space

- **Type**: RGB images
- **Shape**: (180, 320, 3) 
- **Data Type**: uint8
- **Range**: [0, 255]

### Action Space

- **Type**: MultiBinary(7)
- **Actions**: [up, down, left, right, jump, dash, grab]
- **Binary**: Each action is 0 (not pressed) or 1 (pressed)

### Reward Function

The environment provides reward signals based on game state information received from the Celeste mod. The reward calculation is designed to be customizable for different training objectives.

**Default Reward Components:**

- **Progress Reward**: Based on the `distance` metric from the game state, which tracks progress towards the target
- **Death Penalty**: Negative reward when `isDead` flag is true
- **Level Completion**: Bonus reward when `completedLevel` flag is true

## TCP Communication Protocol

The environment communicates with the Celeste mod using JSON messages over TCP:

### Reset Message
```json
{"type": "reset"}
```

### Action Message
```json
{
  "type": "action",
  "actions": [0, 0, 1, 0, 1, 0, 0]
}
```

### State Response
```json
{
  "type": "state",
  "image": "base64_encoded_screenshot",
  "gameState": {
    "position": [x, y],
    "velocity": [vx, vy],
    "room": "room_name",
    "distance": 123.45,
    "isDead": false,
    "completedLevel": false
  }
}
```

## Integrating with Training Code

This environment can be used with any RL library that supports Gymnasium. Example with Stable-Baselines3:

```python
from stable_baselines3 import PPO
from CelesteEnv import CelesteEnv

env = CelesteEnv()
model = PPO("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

## File Structure

```
celeste-ai-gym/
├── CelesteEnv.py          # Main Gym environment
├── debug_env.py           # Environment testing script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Troubleshooting

### Common Issues

**Connection Refused**: Ensure the Celeste AI mod is loaded and the game is running.

**TCP Connection Timeout**: Check that the mod is listening on the correct port (default: 5000).

**Environment Freezing**: Check TCP connection stability and mod compatibility.

## Dependencies

Core dependencies:

- `gymnasium`: RL environment interface
- `numpy`: Numerical computing
- `opencv-python`: Image processing
- `pillow`: Image handling
- `mss`: Screen capture

## License

[Specify your license here]

## Related Projects

- Celeste AI Mod: Companion C# mod for Celeste integration (required to use this environment)

## Citation

If you use this work in research, please cite:

```bibtex
@misc{celeste-ai-gym,
  title={Celeste AI Gym: Gymnasium Environment for Celeste},
  author={[Your Name]},
  year={2026},
  url={[Repository URL]}
}
```