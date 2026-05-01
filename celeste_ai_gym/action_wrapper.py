import gymnasium as gym
from celeste_ai_gym import CelesteEnv

DASHLESS_SIMPLE = [
    ['GRAB'],
    ['GRAB', 'RIGHT'],
    ['GRAB', 'RIGHT', 'JUMP'],
    ['GRAB', 'JUMP']
]

DASHLESS_COMPLEX = [
    ['GRAB'],
    ['GRAB', 'RIGHT'],
    ['RIGHT'],
    ['GRAB', 'RIGHT', 'JUMP'],
    ['GRAB', 'JUMP'],
    ['GRAB', 'LEFT'],
    ['LEFT'],
    ['GRAB', 'LEFT', 'JUMP'],
    ['GRAB', 'DOWN'],
    ['GRAB', 'UP']
]

DASH_RESTRICTED = DASHLESS_COMPLEX + [
    ['RIGHT', 'DASH'],
    ['RIGHT', 'UP', 'DASH'],
    ['UP', 'DASH'],
    ['LEFT', 'UP', 'DASH'],
    ['LEFT', 'DASH']
]

class CelesteActionWrapper(gym.ActionWrapper):
    """
    Converts discrete actions to multibinary actions.
    """

    # Action mapping: each index maps to [up, down, left, right, jump, dash, grab]
    _key_map = {
        'NOOP':  [0, 0, 0, 0, 0, 0, 0],
        'UP':    [1, 0, 0, 0, 0, 0, 0],
        'DOWN':  [0, 1, 0, 0, 0, 0, 0],
        'LEFT':  [0, 0, 1, 0, 0, 0, 0],
        'RIGHT': [0, 0, 0, 1, 0, 0, 0],
        'JUMP':  [0, 0, 0, 0, 1, 0, 0],
        'DASH':  [0, 0, 0, 0, 0, 1, 0],
        'GRAB':  [0, 0, 0, 0, 0, 0, 1]
    }
    
    def __init__(self, env: CelesteEnv, actions: list):
        """
        Initialize a new multibinary to discrete action space wrapper.

        Args:
            env: the environment to wrap
            actions: an ordered list of actions (as lists of buttons).
                The index of each button list is its discrete coded value

        Returns:
            None

        """

        super().__init__(env)
        # create the new action space
        self.action_space = gym.spaces.Discrete(len(actions))
        # create the action map from the list of discrete actions
        self._action_map = {}
        self._action_meanings = {}
        # iterate over all the actions (as button lists)
        for action, button_list in enumerate(actions):
            # the value of this action's multibinary
            multi_action = [0] * 7  # 7 buttons
            # iterate over the buttons in this button list
            for button in button_list:
                multi_action = [max(a, b) for a, b in zip(multi_action, self._key_map[button])]
            # set this action maps value to the multibinary action value
            self._action_map[action] = multi_action
            self._action_meanings[action] = ' '.join(button_list)

    def step(self, action):
        """
        Take a step using the given action.

        Args:
            action: the discrete action to take

        Returns:
            observation, reward, done, info
        """
        multi_action = self._action_map[action]
        return self.env.step(multi_action)
        