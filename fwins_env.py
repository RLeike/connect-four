import numpy as np

const_lines = np.zeros((16,4,2),dtype=np.int32)
def _lines():
    #all diagonal, vertical and horizontal line of length 4 that contain one point
    if const_lines[0,1,0] != 0:
        return const_lines
    for i in range(4):
        const_lines[i,:,0] = np.arange(-i,-i+4)
    for i in range(4):
        const_lines[i+4,:,1] = np.arange(-i,-i+4)
    for i in range(4):
        const_lines[i+8,:,0] = np.arange(-i,-i+4)
        const_lines[i+8,:,1] = np.arange(-i,-i+4)
    for i in range(4):
        const_lines[i+12,:,0] = np.flip(np.arange(i-3,i+1),axis=0)
        const_lines[i+12,:,1] = np.arange(-i,-i+4)
    return const_lines


class FourWins:
    #The class that performs all game-specific stuff
    #Contains state which is a numpy array
    def __init__(self, preprocessing= 'addAxis'):
        self.dims = (7,6)
        self.top_stone = np.ones(self.dims[0],dtype=np.int32)*(self.dims[1]-1)
        self.state = np.zeros(self.dims)
        self.preprocessing = preprocessing

    def play(self, player, column):
        #Takes a player (-1 for +, 1 for o)
        #and a column to put the stone in. Returns 1 if the player wins
        #-1 if the column is full
        pos = np.array([column,self.top_stone[column]])
        if pos[1] == -1:
            return -1
        self.state[pos[0],pos[1]]=player
        self.top_stone[column] -= 1
        return self._checkwin(pos)*player
    
    def preprocessedState(self, player):
        """returns the state of the game but preprocessed for the AI"""
        if self.preprocessing == 'None':
            return self.state*player
        if self.preprocessing == 'addAxis':
            return self.state[np.newaxis,:,:,np.newaxis]*player
        if self.preprocessing == 'distinguishPlayers':
            x = np.zeros((1,)+self.dims+(2,))
            x = np.stack([np.clip(self.state*player,0,1),
                    np.clip(-self.state*player,0,1)], axis=2)
            return x.reshape((1,)+self.dims+(2,))
            
    
    def undoPlay(self, column):
        #Takes a column to remove the top stone from. Raises Value Error if column is empty
        pos = np.array([column,self.top_stone[column]+1])
        if pos[1] == self.dims[1]:
            raise ValueError("Column emtpy, cannot undo")
        self.state[pos[0],pos[1]]=0
        self.top_stone[column] += 1
        
    def soft_play(self, player, dist):
        #takes an array of floats and plays in the highest possible column
        col = np.exp(dist)*(self.top_stone != -1)
        return self.play(player, np.argmax(col))

    def reset(self):
        #starts new game
        self.top_stone = np.ones(self.dims[0],dtype=np.int32)*(self.dims[1]-1)
        self.state = np.zeros(self.dims)
  
    def _checkwin(self, pos):
        l = _lines()+pos.reshape((1,1,2))
        mask = 1-np.any(np.any(l<0,axis=1),axis=1)
        mask *= 1-np.any(l[:,:,0]>=self.dims[0],axis=1)
        mask *= 1-np.any(l[:,:,1]>=self.dims[1],axis=1)
        mask = mask.astype(np.bool)
        winlines = self.state[l[mask,:,0],l[mask,:,1]]
        if np.any(np.all(winlines==1,axis=1),axis=0):
            return 1
        if np.any(np.all(winlines==-1,axis=1),axis=0):
            return -1
        return 0

    def show(self,row_labels=False):
        #draws the game in asci art into the console
        if row_labels:
            s = " "
            for i in range(self.dims[0]):
                s += " {}".format(i)
            s += " "
            print(s)
        else:
            print('_'*(self.dims[0]*2+3))
        for j in range(self.dims[1]):
            str = "|"
            for i in range(self.dims[0]):
                if self.state[i,j] == 1:
                    str += ' o'
                elif self.state[i, j] == -1:
                    str += " +"
                else:
                    str += '  '
            print(str+' |')
        print('-'*(self.dims[0]*2+3))


