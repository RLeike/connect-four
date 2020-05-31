import numpy as onp
from jax import random
import jax.numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import fwins_env as env
import memory
import agent
import pickle

load_model = False
save_model = True


trainSteps = 10000000
memSize = 2000000
#whether some games are shown once completed during training
showGames = True
epsStart = 1.
epsEnd = 0.1
epsDecaySteps = 2.0e6
lr = .01
mom=.99
model_name = "testNet_{}".format(lr)
used_loss = agent.zero_sum_loss
preprocessing='distinguishPlayers'



g = env.FourWins(preprocessing=preprocessing)
state_shape = g.preprocessedState(1)[0].shape
ag = agent.FeedforwardAgent(state_shape, [(4,3,16),(3,2,64), (100,)], (g.state.shape[0],), nonlinearity = ['relu','relu','relu','id'], used_loss = used_loss, model_name = model_name, learning_rate=lr)

if load_model:
    try:
        #     Restore variables from disk.
        vs = np.load(model_name+'.npy', allow_pickle=True)
        ag.variables = [[np.array(a) for a in b] for b in vs]
        print("succesfully loaded model "+ ag.model_name+" from file!")
    except :
        raise ValueError("model could not be loaded. Are you sure a model with these parameters exists?")
        

if not load_model:
    rng = random.PRNGKey(0)
    ag.init_variables(rng)

def update_eps(step, agent):
    #let exploration epsilon decay
    t = (step-mem.size/2)*1./epsDecaySteps
    agent.epsilon = np.exp(t*np.log(epsEnd) + (1-t)*np.log(epsStart))
    agent.epsilon = max(agent.epsilon,epsEnd)
    agent.epsilon = min(agent.epsilon,epsStart)

def display_progress(step, agent, cumulative_loss):
    print("Trained for",step,"/",trainSteps,"steps")
    print("cumulative loss:", cumulative_loss)
    print("exploration epsilon:",agent.epsilon)
    print("current start estimates:")
    print(agent.prediction(np.zeros((1,)+agent.shapes[0])))
    s = onp.zeros((1,)+agent.shapes[0])
    s[0,3,5,1] = 1
    print("current opponent start estimates:")
    print(agent.prediction(s))
    return 0.

def display_game(game, reward, action, player):
        print(reward,action)
        p = ""
        if player == 1:
            p = "o"
        else:
            p = "+"
        print("Player "+p+" {}.".format(["lost","won"][(reward+1)//2]))
        game.show()
        
#*************TRAINING*****************
mem = memory.Memory(g.preprocessedState(1)[0].shape, (g.state.shape[0],),size=memSize)
player = 1
cumulativ_loss = 0.
for step in range(trainSteps):
    if step % 100000 == 0:
        update_eps(step, ag)
        cumulativ_loss = display_progress(step, ag, cumulativ_loss/100000.)
    #the following code plays games according to epsilon greedy and stores them in mem
    s = g.preprocessedState(player)
#    move = ag.think_ahead(g,player,1,noise=0.05)
    move = ag.training_move(s)
    a = np.argmax(move)
    move = onp.zeros(move.shape)
    move[a]=1
    r = g.play(player,a)
    mem.store(s[0],move,r,player)
    if r != 0:
        if showGames and onp.random.rand()<1./2000:
            display_game(g,r,a,player)
        g.reset()
    player *= -1
    if r!= 0:
        player=1
    #Here the actual training happens in minibatches
    if step>mem.size/2 and step % 15 == 0:
        cumulativ_loss += 15*ag.train(mem, minibatchSize=50)
#***********END TRAINING****************

#with open("memory.pcl",'wb') as f:
#    pickle.dump(mem,f)

# Save the variables to disk.
if trainSteps>=20000:
    if save_model:
        save_path = model_name + '.npy'
        np.save(save_path, ag.variables)
        print("Model saved in file: %s" % save_path)
    print("average loss over training data:",agent.compute_avg_loss(ag, mem))


#rest of the stuff is just for testing the AI in 1v1 games
def get_ai_move(game, player, noise = 0., think_ahead = 0):
    s = game.preprocessedState(player)
    aiMove = None
    if think_ahead:
        aiMove = ag.think_ahead(game, player, think_ahead)[np.newaxis,:]
    else:
        aiMove = ag.prediction(s)
    a = onp.argmax(aiMove + onp.random.normal(size = aiMove.shape)*noise)
    print(aiMove, a)
    #print("expected reward:",aiMove[0,a])
    return a
    
def get_user_move(game):
    while(True):
        game.show(row_labels=True)
        userMove = int(input("Enter row (0-6):"))
        if userMove < 0 or userMove > 6:
            continue
        return userMove


def testgame(game = None, ai_begins=False, ai_noise=0., thinking = 0):
    if game == None:
        game = env.FourWins(preprocessing=preprocessing)
    # performs a testgame user vs agent
    print("Playing against trained agent!")
    val = 0
    while (True):
        a = None
        if ai_begins:
            a = get_ai_move(game, 1, ai_noise, thinking)
        else:
            a = get_user_move(game)
        val = game.play(1, a)
        if val == -1:
            print("Player o played illegal move and loses")
            break
        if val == 1:
            game.show()
            print("Player o wins!")
            break
        if not ai_begins:
            a = get_ai_move(game, -1, ai_noise, thinking)
        else:
            a = get_user_move(game)
        val = game.play(-1, a)
        if val == -1:
            print("Player + played illegal move and loses")
            break
        if val == 1:
            game.show()
            print("Player + wins!")
            break
testgame(thinking=2)
