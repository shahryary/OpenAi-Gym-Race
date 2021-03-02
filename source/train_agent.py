"""

"""

import numpy as np
import gym
import time, tqdm
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

import cma
import multiprocessing as mp

from train_model import load_trian

emb_size = 32
number_predict = 2
num_act = 3
num_par = number_predict * emb_size + number_predict

def nrm_obs(observation):
    return observation.astype('float32') / 255.

def weight_bias(params):
    weights = params[:num_par - number_predict]
    bias = params[-number_predict:]
    weights = np.reshape(weights, [emb_size, number_predict])
    return weights, bias

def decide(sess, network, observation, params):
    observation = nrm_obs(observation)
    embedding = sess.run(network.z, feed_dict={network.image: observation[None, :,  :,  :]})
    weights, bias = weight_bias(params)

    action = np.zeros(num_act)
    predict = np.matmul(np.squeeze(embedding), weights) + bias
    predict = np.tanh(predict)

    action[0] = predict[0]
    if predict[1] < 0:
        action[1] = np.abs(predict[1])
        action[2] = 0
    else:
        action[2] = predict[1]
        action[1] = 0

    return action

env = CarRacing()

def play(params, render=True, verbose=False):
    sess, network = load_trian()
    num_try = 12
    agent_reward = 0
    for trial in range(num_try):
        observation = env.reset()
        #  random positions in the race-track
        np.random.seed(int(str(time.time()*1000000)[10:13]))
        position = np.random.randint(len(env.track))
        env.car = Car(env.world, *env.track[position][1:4])

        total_reward = 0.0
        steps = 0
        while True:
            if render:
                env.render()
            action = decide(sess, network, observation, params)
            observation, r, done, info = env.step(action)
            total_reward += r
            #     random init of position
            if verbose and (steps % 200 == 0 or steps == 999):
                print("\naction " + str(["{:+0.2f}".format(x) for x in action]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))

            steps += 1
            if steps == 999:#
                break

        # If reward is out of scale, clip it
        total_reward = np.maximum(-100, total_reward)
        agent_reward += total_reward

    return - (agent_reward / num_try)

def train():
    #CMA-ES stochastic optimizer class with ask-and-tell interface.
    sto = cma.CMAEvolutionStrategy(num_par * [0], 0.1, {'popsize': 10})#15 if you have more memory feel free to increase it

    rewards_through_gens = []
    generation = 1
    try:
        while not sto.stop():
            solutions = sto.ask()
            with mp.Pool(mp.cpu_count()) as parallel:
                rewards = list(tqdm.tqdm(parallel.imap(play, list(solutions)), total=len(solutions)))
            #rewards=list(tqdm.tqdm((play, list(solutions)), total=len(solutions)))
            sto.tell(solutions, rewards)

            rewards = np.array(rewards) *(-1.)
            print("\n**************")
            print("Generation: {}".format(generation))
            print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
            print("Avg reward: {:.3f}".format(np.mean(rewards)))
            print("**************\n")

            generation+=1
            rewards_through_gens.append(rewards)
            np.save('rewards', rewards_through_gens)

    except (KeyboardInterrupt, SystemExit):
        print("Manual Interrupt")
    except Exception as e:
        print("Exception: {}".format(e))
    return sto

if __name__ == '__main__':
    sto = train()
    np.save('best_params', sto.best.get()[0])
    RENDER = True
    score = play(sto.best.get()[0], render=RENDER, verbose=True)
    print("Final Score: {}".format(-score))