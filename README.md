# Celeste IMRL

This project uses an environment seen at https://github.com/ethanrcampbell02/celeste-ai-gym, along side the companion mod https://github.com/ethanrcampbell02/celeste-ai-mod, to train an intrinsically motivated RL agent.
Also uses https://github.com/RLE-Foundation/RLeXplore/tree/main

## Installation

### Prerequisites

- Python 3.8+
- Celeste game with a SLIGHTLY ALTERED companion AI mod installed

### Altering the companion mod
before installing the companion mod, go to "MadelAIneModule.cs" in the companion mod's source folder, find the "ApplyInputs" function, and change it to
```C#
private void ApplyInputs(JsonElement response)
    {
        try
        {
            // Parse input values from response
            // Expected format: {"type": "ACK", "moveX": 0.0, "moveY": 0.0, "jump": false, "dash": false, "grab": false}
            Vector2 aim = new Vector2(0, 0);
            // Movement axes
            if (response.TryGetProperty("moveX", out var moveXProp))
            {
                float moveX = moveXProp.GetSingle();
                if (moveX == 0)
                    GameplayBinds.MoveX.SetNeutral();
                else
                    GameplayBinds.MoveX.SetValue(moveX);
                aim.X = moveX;
            }
            
            if (response.TryGetProperty("moveY", out var moveYProp))
            {
                float moveY = moveYProp.GetSingle();
                if (moveY == 0)
                    GameplayBinds.MoveY.SetNeutral();
                else
                    GameplayBinds.MoveY.SetValue(moveY);
                aim.Y=moveY;
            }
            
            // Button presses
            if (response.TryGetProperty("jump", out var jumpProp))
            {
                if (jumpProp.GetBoolean())
                    GameplayBinds.Jump.Press();
                else
                    GameplayBinds.Jump.Release();
            }
            
            if (response.TryGetProperty("dash", out var dashProp))
            {
                if (dashProp.GetBoolean()) {
                    GameplayBinds.Aim.SetDirection(aim);
                    GameplayBinds.Dash.Press();
                }
                else
                    GameplayBinds.Dash.Release();
            }
            
            if (response.TryGetProperty("grab", out var grabProp))
            {
                if (grabProp.GetBoolean())
                    GameplayBinds.Grab.Press();
                else
                    GameplayBinds.Grab.Release();
            }
            
            Logger.Debug(nameof(MadelAIneModule), "Applied inputs from Python");
        }
        catch (Exception ex)
        {
            Logger.Error(nameof(MadelAIneModule), $"Error applying inputs: {ex.Message}");
        }
    }
```
(Note the addition of changing "aim", this allows madeline to dash vertically and diagonally)
### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd celeste-IMRL
   ```

3. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Testing the Environment

```bash
python debug_env.py
```

This will connect to a running Celeste instance and allow you to test the environment interface.

### Running Agents

```bash
python .\PPO_forsaken.py
```
(replace PPO_forsaken with any of the other IMRL.py files)

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
