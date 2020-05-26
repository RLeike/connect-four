import numpy as np

"""This as a storage for the information necessary for training
i.e. it stores quadruple (old_state, action, reward, new_state) and offers easy minibatch accessibility"""

class Memory:
    """
    Stores the last couple of experiences of the AI. An experence is a quadruple (old_state, action, reward, new_state)
    If full it proceeds to overwrite the oldest experience.
    """
    def __init__(self, state_shape, action_shape, size=1000000):
        self.memcounter = 0
        self.old_state = np.zeros((size,)+state_shape)
        self.action = np.zeros((size,)+action_shape)
        self.new_state = np.zeros((size,)+state_shape)
        self.reward = np.zeros((size,1))
        #saves the player that wins
        self.win = np.zeros((size,1))
        self.player = np.zeros((size,1))
        self.size = size
        self.full = False
        self.empty = True
        self.data_augmentation = True

    def store(self, state, action, reward, player):
        """
        Stores an experience. DOES NOT COPY, do not modify input.
        :param state: the current state that the AI saw
        :param action: the action it took when evaluating that state
        :param reward: the reward it got as a consequence to its action
        :param player: the player that is took the action
        """
        if self.empty:
            self.last_state = state
            self.last_action = action
            self.last_reward = reward
            self.last_player = player
            self.game_began = 0
            self.empty = False
            return
        self.old_state[self.memcounter] = self.last_state
        self.action[self.memcounter] = self.last_action
        self.reward[self.memcounter] = self.last_reward
        self.player[self.memcounter] = self.last_player
        self.new_state[self.memcounter] = np.copy(state)
        if self.last_reward != 0:
            if self.memcounter >= self.game_began:
                self.win[self.game_began:self.memcounter+1] = self.last_reward*self.last_player
            else:
                self.win[:self.memcounter+1] = self.last_reward*self.last_player
                self.win[self.game_began:] = self.last_reward*self.last_player
        self.memcounter += 1
        if self.memcounter >= self.size:
            self.memcounter = 0
            self.full = True
        if self.last_reward != 0:
            self.game_began = self.memcounter
        self.last_player = player
        self.last_state = np.copy(state)
        self.last_action = np.copy(action)
        self.last_reward = reward

    def display(self, game, begin=0, end=10):
        #displays the experience. assumes game has attribute .state and method.show]
        for i in range(begin, end):
            game.state = self.old_state[i,:,:,0] #this is still gamespecific
            print("old state:")
            game.show()
            print("action:")
            print(self.action[i])
            print("reward:",self.reward[i])
            print("new state:")
            game.state = self.new_state[i,:,:,0] #this is still gamespecific
            game.show()
            
    def get_minibatch(self, size = 10, include_wins=False):
        """
        makes a minibatch of random experiences. output should be handled read only!
        :param size: how many things are in the minibatch
        :return: the minibatch as a quadruple (old_state, action, reward, new_state). e.g. reward is a shape (size,) np array
        """
        available = 0
        if self.full:
            available = self.size
        else:
            available = self.memcounter -1
        batch_index = np.random.choice(available, size)
        flip = np.random.randint(0,2)
        ols = self.old_state[batch_index]
        nes = self.new_state[batch_index]
        ac = self.action[batch_index]
        if flip and self.data_augmentation:
            ols = np.flip(ols, axis = 1)
            nes = np.flip(nes, axis = 1)
            ac = np.flip(ac, axis = 1)
        if include_wins == False:
            return (ols, ac, self.reward[batch_index],nes)
        return (ols, ac, self.reward[batch_index], nes, (self.win[batch_index]*self.player[batch_index])/2.+0.5)
    
    
class FakeMemory:
    def __init__(self,mem):
        """
        Mimics the interface of the memory class but delivers only fake data
        """
        self.state_shape = mem.old_state.shape[1:]
        self.action_shape = mem.action.shape[1:]
        
    
        
    def get_minibatch(self, size = 10, include_wins=False):
        old_state = np.random.randint(0,3, size=(size,)+self.state_shape)-1
        new_state = np.zeros((size,)+self.state_shape)
        reward = np.random.randint(0,2,size=(size,1))*2-1
        action = np.zeros((size,)+self.action_shape)
        action[[np.arange(size),np.random.choice(7,size=size)]] = 1.
        #put some information about the reward into the data
        old_state[:,3,:,0] = reward
        if include_wins == False:
            return (old_state, action, reward,new_state)
        return (old_state, action, reward,new_state, reward)

    
    
    
