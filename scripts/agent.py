#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
November 2024
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from collections import defaultdict

class Agent:
    """Agent interacting with some environment.
    
    Parameters
    ----------
    model : object of class Environment
        Model.
    policy : function or string
        Policy of the agent (default = random).
    player : int
        Player for games (1 or -1, default = default player of the game).
    """
    
    def __init__(self, model, policy='random', player=None):
        if type(policy) == str:
            if policy == 'random':
                policy = self.random_policy
            elif policy == 'one_step':
                policy = self.one_step_policy
            else:
                raise ValueError("The policy must be either 'random', 'one_step', or a custom policy.")            
        if player is None:
            if model.is_game():
                player = model.player
            else:
                player = 1            
        self.model = model
        self.policy = policy
        self.player = player
            
    def get_actions(self, state):
        """Get available actions."""
        if self.model.is_game():
            return self.model.get_actions(state, self.player)
        else:
            return self.model.get_actions(state)
        
    def random_policy(self, state):
        """Random choice among possible actions."""
        probs = []
        actions = self.get_actions(state)
        if len(actions):
            probs = np.ones(len(actions)) / len(actions)
        return probs, actions
    
    def one_step_policy(self, state):
        """One-step policy for games, looking for win moves or moves avoiding defeat."""
        if not self.model.is_game():
            raise ValueError('The one-step policy is applicable to games only.')
        player, board = state
        if player == self.player:
            actions = self.get_actions(state)
            # win move
            for action in actions:
                next_state = self.model.get_next_state(state, action)
                if self.model.get_reward(next_state) == player:
                    return [1], [action]
            # move to avoid defeat
            for action in actions:
                adversary_state = -player, board
                next_state = self.model.get_next_state(adversary_state, action)
                if self.model.get_reward(next_state) == -player:
                    return [1], [action]
            # otherwise, random move
            if len(actions):
                probs = np.ones(len(actions)) / len(actions)
                return probs, actions
        return [1], [None]

    def get_action(self, state):
        """Get selected action."""
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action      
         
    def get_episode(self, state=None, horizon=100):
        """Get the states and rewards for an episode, starting from some state."""
        self.model.reset(state)
        state = self.model.state
        states = [state] 
        reward = self.model.get_reward(state)
        rewards = []
        stop = self.model.is_terminal(state)
        if not stop:
            for t in range(horizon):
                action = self.get_action(state)
                reward, stop = self.model.step(action)
                state = self.model.state
                states.append(state)
                rewards.append(reward)
                if stop:
                    break
        return stop, states, rewards
    
    def get_gains(self, state=None, horizon=100, n_runs=100, gamma=1):
        """Get the gains (cumulative rewards) over independent runs, starting from some state."""
        gains = []
        for t in range(n_runs):
            _, _, rewards = self.get_episode(state, horizon)
            gains.append(sum(rewards * np.power(gamma, np.arange(len(rewards)))))
        return gains
    
    
class OnlineEvaluation(Agent):
    """Online evaluation. The agent interacts with the environment and learns the value function of its policy.
    
    Parameters
    ----------
    model : object of class Environment
        Model.
    policy : function or string
        Policy of the agent (default = random).
    player : int
        Player for games (1 or -1, default = default player of the game).
    gamma : float
        Discount rate (in [0, 1], default = 1).
    init_value : float
        Initial value of the value function.
    """
    
    def __init__(self, model, policy='random', player=None, gamma=1, init_value=0):
        super(OnlineEvaluation, self).__init__(model, policy, player)   
        self.gamma = gamma 
        self.value = defaultdict(lambda: init_value) # value of a state
        self.count = defaultdict(lambda: 0) # count of a state (number of visits)        
            
    def add_state(self, state):
        """Add a state if unknown."""
        code = self.model.encode(state)
        if code not in self.value:
            self.value[code] = 0
            self.count[code] = 0
        
    def get_known_states(self):
        """Get known states."""
        states = [self.model.decode(code) for code in self.value]
        return states
    
    def is_known(self, state):
        """Check if some state is known."""
        code = self.model.encode(state)
        return code in self.value

    def get_values(self, states=None):
        """Get the values of some states (default = all states).""" 
        if states is None:
            states = self.model.get_all_states()
            if not len(states):
                raise ValueError("The state space is too large. Please specify some states.")
        codes = [self.model.encode(state) for state in states]
        values = [self.value[code] for code in codes]
        return values
    
    def get_best_actions(self, state):
        """Get the best actions in some state according to the value function.""" 
        actions = self.get_actions(state)
        if len(actions) > 1:
            values = []
            for action in actions:
                probs, next_states = self.model.get_transition(state, action)
                rewards = [self.model.get_reward(next_state) for next_state in next_states]
                next_values = self.get_values(next_states)
                # expected value
                value = np.sum(np.array(probs) * (np.array(rewards) + self.gamma * np.array(next_values)))
                values.append(value)
            if self.player == 1:
                best_value = max(values)
            else:
                best_value = min(values)
            actions = [action for action, value in zip(actions, values) if value==best_value]
        return actions        
    
    def get_policy(self):
        """Get the best policy according to the value function."""
        def policy(state):
            actions = self.get_best_actions(state)
            if len(actions):
                probs = np.ones(len(actions)) / len(actions)
            else:
                probs = []
            return probs, actions
        return policy    
    
    def randomize_policy(self, epsilon=0):
        """Get the epsilon-greedy policy according to the value function."""
        def policy(state):
            actions = self.get_actions(state)
            best_actions = self.get_best_actions(state)
            probs = []
            for action in actions:
                prob = epsilon / len(actions)
                if action in best_actions:
                    prob += (1 - epsilon) / len(best_actions)
                probs.append(prob)
            return probs, actions
        return policy
     
        
class OnlineControl(OnlineEvaluation):
    """Online control. The agent interacts with the model and learns the best policy.
    
    Parameters
    ----------
    model : object of class Environment
        Model.
    policy : function or string
        Initial policy of the agent (default = random).
    player : int
        Player for games (1 or -1, default = default player of the game).
    gamma : float
        Discount rate (in [0, 1], default = 1).
    init_value : float
        Initial value of the action-value function.
    """
    
    def __init__(self, model, policy='random', player=None, gamma=1, init_value=0):
        super(OnlineControl, self).__init__(model, policy, player, gamma)  
        self.action_value = defaultdict(lambda: defaultdict(lambda: init_value))
        self.action_count = defaultdict(lambda: defaultdict(lambda: 0))
            
    def get_known_states(self):
        """Get known states."""
        states = [self.model.decode(code) for code in self.action_value]
        return states
    
    def get_best_actions(self, state):
        """Get the best actions in some state according to the action-value function.""" 
        actions = self.get_actions(state)
        if len(actions) > 1:
            code = self.model.encode(state)
            values = [self.action_value[code][action] for action in actions]
            if self.player == 1:
                best_value = max(values)
            else:
                best_value = min(values)
            actions = [action for action, value in zip(actions, values) if value==best_value]
        return actions
    
    def randomize_best_action(self, state, epsilon=0):
        """Get the action of the epsilon-greedy policy."""
        policy = self.randomize_policy(epsilon=epsilon)
        probs, actions = policy(state)
        i = np.random.choice(len(actions), p=probs)
        return actions[i]
    

class Bandit(Agent):
    """Bandit algorithm with random policy. 

    Parameters
    ----------
    model : object of class MAB
        The model.
    init_value : float
        Initial value of the action-value function.
    init_count : int
        Initial count of the action-count function.
    """
    
    def __init__(self, model, init_value=0, init_count=0):
        # if not isinstance(model, MAB):
        #     raise ValueError('The model must be a multi-armed bandit.')
        self.model = model
        self.policy = self.random_policy
        actions = model.get_actions()
        self.values = len(actions) * [init_value]
        self.counts = len(actions) * [init_count]

    def get_actions(self, state=None):
        """Get all possible actions."""
        return self.model.get_actions()
    
    def get_episode(self, horizon=100):
        """Get the rewards for an episode and update the values."""
        state = None
        rewards = []
        for t in range(horizon):
            action = self.get_action(state)
            reward, _ = self.model.step(action)
            rewards.append(reward)
            self.counts[action] += 1
            diff = reward - self.values[action]
            # update by temporal difference
            self.values[action] += diff / self.counts[action]
        return rewards    
    

class Greedy(Bandit):
    """Bandit algorithm with epsilon-greedy policy. 

    Parameters
    ----------
    model : object of class MAB
        The model.
    epsilon : float in [0, 1]
        Exploration rate.
    init_value : float
        Initial value of the action-value function.
    init_count : int
        Initial count of the action-count function.
    """
    
    def __init__(self, model, epsilon=0.1, init_value=0, init_count=0):
        super(Greedy, self).__init__(model, init_value, init_count) 
        self.epsilon = epsilon

    def get_action(self, state=None):
        """Get action with eps-greedy policy."""
        actions = self.get_actions()
        if np.random.random() > self.epsilon:
            # select the best action(s) with probability 1 - epsilon
            values = np.array(self.values)
            actions = np.flatnonzero(values==np.max(values))
        return np.random.choice(actions)


class UCB(Bandit):
    """Bandit algorithm with UCB policy. 

    Parameters
    ----------
    model : object of class MAB
        The model.
    const : float in [0, 1]
        Multiplicative constant for the UCB bonus.
    init_value : float
        Initial value of the action-value function.
    init_count : int
        Initial count of the action-count function.
    """
    
    def __init__(self, model, const=1, init_value=0, init_count=0):
        super(UCB, self).__init__(model, init_value, init_count) 
        self.const = const

    def get_action(self, state=None):
        """Get action with UCB policy."""
        values = np.array(self.values)
        counts = np.array(self.counts)
        actions = self.get_actions()
        # to be modified
        # not visited actions
        not_visited = np.flatnonzero(counts==0)
        if len(not_visited):
            return np.random.choice(not_visited)
        # visited actions
        ucb = values + self.const * np.sqrt(np.log(np.sum(counts)) / counts)
        actions = np.flatnonzero(ucb==np.max(ucb))
        return np.random.choice(actions)
    

class TS(Bandit):
    """Bandit algorithm with Thompson sampling. 

    Parameters
    ----------
    model : object of class MAB
        The model.
    """
    
    def __init__(self, model):
        super(TS, self).__init__(model) 
        self.distribution = model.distribution
            
    def get_action(self, state=None):
        """Get action with TS policy."""
        values = np.array(self.values)
        counts = np.array(self.counts)
        if self.distribution == 'bernoulli':
            # to be modified
            alpha = values * counts + 1
            beta = counts - values * counts + 1
            samples = np.random.beta(alpha, beta)
        else:
            # to be modified
            mean = values * counts / (counts + 1)
            std = 1 / np.sqrt(counts + 1)
            samples = np.random.normal(mean, std)
        return np.argmax(samples)
