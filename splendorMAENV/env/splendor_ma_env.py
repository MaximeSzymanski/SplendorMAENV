from pettingzoo import AECEnv
import functools
import random
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils import agent_selector, wrappers
from model.game_manager import GameManager
from copy import copy

import numpy as np
class SplendorMAEnv(AECEnv):
    """Splendor Multi-Agent Environment for a two player game.
    """
    """The metadata holds environment constants.

      The "name" metadata allows the environment to be pretty printed.
      """

    metadata = {
        'name': 'splendorMAEnv_v0'
    }

    def __init__(self):
        self.possible_agents = ["player_1", "player_2"]
        self.agent_name_mapping = dict(zip(range(len(self.possible_agents)), self.possible_agents))
        self.game : GameManager = GameManager(2)
        self.game.launch_game(2)




    def reset(
        self,
        seed = None,
        options = None,
    ) -> None:
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()




    def observe(self, agent):
            """
            Observe should return the observation of the specified agent. This function
            should return a sane observation (though not necessarily the most up to date possible)
            at any time after reset() is called.
            """
            # observation of one agent is the previous state of the other
            return np.array(self.observations[agent])

    def step(self, action):
        """
               step(action) takes in an action for the current agent (specified by
               agent_selection) and needs to update
               - rewards
               - _cumulative_rewards (accumulating the rewards)
               - terminations
               - truncations
               - infos
               - agent_selection (to the next agent)
               And any internal state used by observe() or render()
               """

        if(
            self.terminations[self.agent_selection] == True
            or self.truncations[self.agent_selection] == True
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        # stores action of current agent
        self.state[self.agent_selection] = action

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            self.rewards[]

        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return MultiDiscrete(
            [8, 8, 8, 8, 8, 6, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91,
             91, 91, 91, 91, 91, 11, 11, 11, 11, 11, 91, 91, 91, 8, 8, 8, 8, 8, 6, 91, 91, 91, 91, 91, 91, 91, 91, 91,
             91, 91, 91, 91, 91, 91,
             91, 91, 91, 91, 91, 11, 11, 11, 11, 11, 92, 92, 92, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 11, 11,
             11, 11, 11, 40, 30, 20])

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(66)






