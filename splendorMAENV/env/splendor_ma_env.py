from pettingzoo import AECEnv
import functools
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.utils import agent_selector
from model.game_manager import GameManager
from model.token_array import TokenArray
from model.checker import Checker
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
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.from_board_states_to_obs_train(1)
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.last_action = -1
        print(f'=====================================================')
        print(f'RESET')
        print(f'=====================================================')




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


        player_1_vp = self.game.get_player_victory_point(0)
        player_2_vp = self.game.get_player_victory_point(1)
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
        self.apply_action(action)
        both_players_passed = action == 65 and self.last_action == 65
        if both_players_passed:
            print(f'=====================================================')
            print(f'Both player passed')
            print(f'=====================================================')

        if self._agent_selector.is_last():

            self.rewards[self.agents[0]] , self.rewards[self.agents[1]] = self.game.get_player_victory_point(0), self.game.get_player_victory_point(1)
            self.terminations = {agent: self.game.is_last_turn() or both_players_passed  for agent in self.agents}
            self.truncations = {agent: self.game.is_last_turn() or both_players_passed for agent in self.agents}

            for i in self.agents:
                self.observations[i] = self.from_board_states_to_obs_train(i)
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self.num_moves += 1
        self._accumulate_rewards()

        self.last_action = action


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

    def from_board_states_to_obs_train(self, playerId: int):
        if playerId == ['Player_1']:
            playerId = 1
        else:
            playerId = 2
        state = self.game.gather_ia_board_state(playerId)
        player_string = 'player1' if playerId == 1 else 'player2'
        player2 = (playerId + 1) % 2
        opponent_string = 'player' + str(player2)

        obs = np.zeros(88)
        obs[0:6] = state[player_string]['tokens']
        # if the length of the list smaller than 20, we padd with 90
        list_of_cards = [
            card.cardId for card in state[player_string]['cards'] if card is not None]
        if len(list_of_cards) < 20:
            for i in range(20 - len(list_of_cards)):
                list_of_cards.append(90)

        obs[6:26] = list_of_cards
        # same for nobles, if the length of the list smaller than 5, we padd with 10
        list_of_nobles = [
            patron.patron_id for patron in state[player_string]['nobles']]
        if len(list_of_nobles) < 5:
            for i in range(5 - len(list_of_nobles)):
                list_of_nobles.append(10)

        obs[26:31] = list_of_nobles
        # we save the reserved cards in a list, if the card is None, we padd with 90

        self.reserved_cards = [
            card.cardId for card in state[player_string]['reserved'] if card is not None]
        if len(self.reserved_cards) < 3:
            for i in range(3 - len(self.reserved_cards)):
                self.reserved_cards.append(90)
        # same for reserved cards, if the length of the list smaller than 3, we padd with 90
        list_of_reserved = [
            card.cardId for card in state[player_string]['reserved'] if card is not None]
        if len(list_of_reserved) < 3:
            for i in range(3 - len(list_of_reserved)):
                list_of_reserved.append(90)
        obs[31:34] = list_of_reserved

        obs[34:40] = state[opponent_string]['tokens']
        # same for player 2
        list_of_cards = [
            card.cardId for card in state[opponent_string]['cards'] if card is not None]
        if len(list_of_cards) < 20:
            for i in range(20 - len(list_of_cards)):
                list_of_cards.append(90)

        obs[40:60] = list_of_cards
        list_of_nobles = [
            patron.patron_id for patron in state[opponent_string]['nobles']]
        if len(list_of_nobles) < 5:
            for i in range(5 - len(list_of_nobles)):
                list_of_nobles.append(10)
        obs[60:65] = list_of_nobles
        list_of_reserved = [
            card.cardId for card in state[opponent_string]['reserved'] if card is not None]

        if len(list_of_reserved) < 3:
            for i in range(3 - len(list_of_reserved)):
                list_of_reserved.append(90)

        obs[65:68] = list_of_reserved

        # same for shop
        self.shop1_cards = state['shop']['rank1_cards']
        list_of_cards_rank1 = [
            card.cardId for card in state['shop']['rank1_cards']]

        if len(list_of_cards_rank1) < 4:
            for i in range(4 - len(list_of_cards_rank1)):
                list_of_cards_rank1.append(90)
        obs[68:72] = list_of_cards_rank1

        self.shop2_cards = state['shop']['rank2_cards']
        list_of_cards_rank2 = [
            card.cardId for card in state['shop']['rank2_cards']]
        if len(list_of_cards_rank2) < 4:
            for i in range(4 - len(list_of_cards_rank2)):
                list_of_cards_rank2.append(90)
        obs[72:76] = list_of_cards_rank2

        self.shop3_cards = state['shop']['rank3_cards']
        list_of_cards_rank3 = [
            card.cardId for card in state['shop']['rank3_cards']]
        if len(list_of_cards_rank3) < 4:
            for i in range(4 - len(list_of_cards_rank3)):
                list_of_cards_rank3.append(90)
        obs[76:80] = list_of_cards_rank3

        list_of_nobles = [
            patron.patron_id for patron in state['shop']['nobles']]
        if len(list_of_nobles) < 5:
            for i in range(5 - len(list_of_nobles)):
                list_of_nobles.append(10)
        obs[80:85] = list_of_nobles

        obs[85] = state['shop']['rank1_size']
        obs[86] = state['shop']['rank2_size']
        obs[87] = state['shop']['rank3_size']
        # save it as a pickle

        obs = self.normalize_obs(obs)
        self.obs = obs

        self.encoded_state = state
        return obs

    def normalize_obs(self, obs):
        """
                Player 1 state:
                5 tokens: 0-4
                1 gold token: 5
                20 player cards: 6-25
                5 noble cards: 26-30
                3 reserved cards: 31-33

                Player 2 state:
                5 tokens: 34-38
                1 gold token: 39
                20 player cards: 40-59
                5 noble cards: 60-64
                3 reserved cards: 65-67

                Shop state:
                12 cards: 68-79
                5 noble cards: 80-84
                1 tier 1 number of cards: 85
                1 tier 2 number of cards: 86
                1 tier 3 number of cards: 87

                """
        obs[0:6] = obs[0:6] / 10
        obs[6:26] = obs[6:26] / 90
        obs[26:31] = obs[26:31] / 10
        obs[31:34] = obs[31:34] / 90

        obs[34:40] = obs[34:40] / 10
        obs[40:60] = obs[40:60] / 90
        obs[60:65] = obs[60:65] / 10
        obs[65:68] = obs[65:68] / 90

        obs[68:80] = obs[68:80] / 90
        obs[80:85] = obs[80:85] / 10
        obs[85:88] = obs[85:88] / 30

        return obs

    def apply_action(self, action):
        # print the action done
        # print('action done : ', action)
        if action == 0:
            # take [1,1,1,0,0,0] tokens
            self.game.take_token(TokenArray([1, 1, 1, 0, 0, 0]))
        elif action == 1:
            # take [1,1,0,1,0,0] tokens
            self.game.take_token(TokenArray([1, 1, 0, 1, 0, 0]))
        elif action == 2:
            # take [1,1,0,0,1,0] tokens
            self.game.take_token(TokenArray([1, 1, 0, 0, 1, 0]))
        elif action == 3:
            # take [1,0,1,1,0,0] tokens
            self.game.take_token(TokenArray([1, 0, 1, 1, 0, 0]))
        elif action == 4:
            # take [1,0,1,0,1,0] tokens
            self.game.take_token(TokenArray([1, 0, 1, 0, 1, 0]))
        elif action == 5:
            # take [1,0,0,1,1,0] tokens
            self.game.take_token(TokenArray([1, 0, 0, 1, 1, 0]))
        elif action == 6:
            # take [0,1,1,1,0,0] tokens
            self.game.take_token(TokenArray([0, 1, 1, 1, 0, 0]))
        elif action == 7:
            # take [0,1,1,0,1,0] tokens
            self.game.take_token(TokenArray([0, 1, 1, 0, 1, 0]))
        elif action == 8:
            # take [0,1,0,1,1,0] tokens
            self.game.take_token(TokenArray([0, 1, 0, 1, 1, 0]))
        elif action == 9:
            # take [0,0,1,1,1,0] tokens
            self.game.take_token(TokenArray([0, 0, 1, 1, 1, 0]))
        elif action == 10:
            self.game.take_token(TokenArray([2, 0, 0, 0, 0, 0]))
        elif action == 11:
            self.game.take_token(TokenArray([0, 2, 0, 0, 0, 0]))
        elif action == 12:
            self.game.take_token(TokenArray([0, 0, 2, 0, 0, 0]))
        elif action == 13:
            self.game.take_token(TokenArray([0, 0, 0, 2, 0, 0]))
        elif action == 14:
            self.game.take_token(TokenArray([0, 0, 0, 0, 2, 0]))
        elif action == 15:
            self.game.buy_card(self.shop1_cards[0].cardId)
        elif action == 16:
            self.game.buy_card(self.shop1_cards[1].cardId)
        elif action == 17:
            self.game.buy_card(self.shop1_cards[2].cardId)
        elif action == 18:
            self.game.buy_card(self.shop1_cards[3].cardId)
        elif action == 19:
            self.game.buy_card(self.shop2_cards[0].cardId)
        elif action == 20:
            self.game.buy_card(self.shop2_cards[1].cardId)
        elif action == 21:
            self.game.buy_card(self.shop2_cards[2].cardId)
        elif action == 22:
            self.game.buy_card(self.shop2_cards[3].cardId)
        elif action == 23:
            self.game.buy_card(self.shop3_cards[0].cardId)
        elif action == 24:
            self.game.buy_card(self.shop3_cards[1].cardId)
        elif action == 25:
            self.game.buy_card(self.shop3_cards[2].cardId)
        elif action == 26:
            self.game.buy_card(self.shop3_cards[3].cardId)
        elif action == 27:
            self.game.buy_card(self.reserved_cards[0])
        elif action == 28:
            self.game.buy_card(self.reserved_cards[1])
        elif action == 29:
            self.game.buy_card(self.reserved_cards[2])
        elif action == 30:
            self.game.reserve_card(self.shop1_cards[0].cardId)
        elif action == 31:
            self.game.reserve_card(self.shop1_cards[1].cardId)
        elif action == 32:
            self.game.reserve_card(self.shop1_cards[2].cardId)
        elif action == 33:
            self.game.reserve_card(self.shop1_cards[3].cardId)
        elif action == 34:
            self.game.reserve_card(self.shop2_cards[0].cardId)
        elif action == 35:
            self.game.reserve_card(self.shop2_cards[1].cardId)
        elif action == 36:
            self.game.reserve_card(self.shop2_cards[2].cardId)
        elif action == 37:
            self.game.reserve_card(self.shop2_cards[3].cardId)
        elif action == 38:
            self.game.reserve_card(self.shop3_cards[0].cardId)
        elif action == 39:
            self.game.reserve_card(self.shop3_cards[1].cardId)
        elif action == 40:
            self.game.reserve_card(self.shop3_cards[2].cardId)
        elif action == 41:
            self.game.reserve_card(self.shop3_cards[3].cardId)
        elif action == 42:
            self.game.reserve_pile_card(0)
        elif action == 43:
            self.game.reserve_pile_card(1)
        elif action == 44:
            self.game.reserve_pile_card(2)
        elif action == 45:
            self.game.take_token(TokenArray([1, 0, 0, 0, 0, 0]))
        elif action == 46:
            self.game.take_token(TokenArray([0, 1, 0, 0, 0, 0]))
        elif action == 47:
            self.game.take_token(TokenArray([0, 0, 1, 0, 0, 0]))
        elif action == 48:
            self.game.take_token(TokenArray([0, 0, 0, 1, 0, 0]))
        elif action == 49:
            self.game.take_token(TokenArray([0, 0, 0, 0, 1, 0]))
        elif action == 50:
            self.game.take_token(TokenArray([1, 1, 0, 0, 0, 0]))
        elif action == 51:
            self.game.take_token(TokenArray([1, 0, 1, 0, 0, 0]))
        elif action == 52:
            self.game.take_token(TokenArray([1, 0, 0, 1, 0, 0]))
        elif action == 53:
            self.game.take_token(TokenArray([1, 0, 0, 0, 1, 0]))
        elif action == 54:
            self.game.take_token(TokenArray([0, 1, 1, 0, 0, 0]))
        elif action == 55:
            self.game.take_token(TokenArray([0, 1, 0, 1, 0, 0]))
        elif action == 56:
            self.game.take_token(TokenArray([0, 1, 0, 0, 1, 0]))
        elif action == 57:
            self.game.take_token(TokenArray([0, 0, 1, 1, 0, 0]))
        elif action == 58:
            self.game.take_token(TokenArray([0, 0, 1, 0, 1, 0]))
        elif action == 59:
            self.game.take_token(TokenArray([0, 0, 0, 1, 1, 0]))
        elif action == 60:
            self.game.take_token(TokenArray([2, 0, 0, 0, 0, 0]))
        elif action == 61:
            self.game.take_token(TokenArray([0, 2, 0, 0, 0, 0]))
        elif action == 62:
            self.game.take_token(TokenArray([0, 0, 2, 0, 0, 0]))
        elif action == 63:
            self.game.take_token(TokenArray([0, 0, 0, 2, 0, 0]))
        elif action == 64:
            self.game.take_token(TokenArray([0, 0, 0, 0, 2, 0]))
        elif action == 65:
            self.game.pass_turn()

    def get_mask(self, playerId):
            state = self.encoded_state
            # print the key of state

            player_str = 'player1' if playerId=='player_1' else 'player2'
            # read the pickle

            mask = np.zeros(88)
            shop_cards_rank1 = state['shop']['rank1_cards']
            shop_cards_rank2 = state['shop']['rank2_cards']
            shop_cards_rank3 = state['shop']['rank3_cards']
            # if the shop of rank 1 is empty, padd the shop with none
            if len(shop_cards_rank1) != 4:
                # padd with none shop_cards_rank1 until it has 4 cards
                for i in range(4 - len(state['shop']['rank1_cards'])):
                    shop_cards_rank1.append(None)

            # if the shop of rank 2 is empty, padd the shop with none
            if len(shop_cards_rank2) != 4:
                for i in range(4 - len(state['shop']['rank2_cards'])):
                    shop_cards_rank2.append(None)
            # if the shop of rank 3 is empty, padd the shop with none
            if len(shop_cards_rank3) != 4:
                for i in range(4 - len(state['shop']['rank3_cards'])):
                    shop_cards_rank3.append(None)
            # remove the none from the reserved cards
            number_card_reserved = len(
                [x for x in state[player_str]['reserved'] if x is not None])
            shop_cards = [shop_cards_rank1, shop_cards_rank2, shop_cards_rank3]

            mask = Checker.get_mask(state[player_str]['cards'], shop_cards, state['shop']['rank1_size'],
                                    state['shop']['rank2_size'], state['shop']['rank3_size'], number_card_reserved,
                                    TokenArray(state[player_str]['tokens']), state['shop']['tokens'],
                                    state[player_str]['reserved'],
                                    state[player_str]['object'])

            return mask.astype(np.int8)