import numpy as np
import multiprocessing as mp
import gym
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

# define parameters 
batch_size = 12
batch_number = 30
steps = 200
render = True


# generate action 
def create_action(last_action):
    if np.random.randint(3) % 3:
        return last_action

    index = np.random.randn(3)
    index[1] = np.abs(index[1])
    index = np.argmax(index)
    mask = np.zeros(3)
    mask[index] = 1

    action = np.random.randn(3)
    action = np.tanh(action)
    action[1] = (action[1] + 1) / 2
    action[2] = (action[2] + 1) / 2

    return action*mask

# normalize numbers  
def norm_obse(observation):
    return observation.astype('float32') / 255.

# 
def simulate_batch(batch_num):
    car_env = CarRacing()

    obs_data = []
    action_data = []
    action = car_env.action_space.sample()
    for item in range(batch_size):
        en_observ = car_env.reset()
        # this make car to start in random positions 
        position = np.random.randint(len(car_env.track))
        car_env.car = Car(car_env.world, *car_env.track[position][1:4])
        en_observ = norm_obse(en_observ)

        obs_sequence = []

        # time steps
        for i in range(steps):
            if render:
                car_env.render()

            action = create_action(action)

            en_observ, reward, done, info = car_env.step(action)
            en_observ = norm_obse(en_observ)

            obs_data.append(en_observ)

    print("Saving dataset for batch {}".format(batch_num))
    np.save('data/TR_data_{}'.format(batch_num), obs_data)
    
    car_env.close()

def main():
    print("Generating data for env CarRacing-v0")

    # this make you run in parallel mode
    with mp.Pool(mp.cpu_count()) as parallel:
        parallel.map(simulate_batch, range(batch_number))

if __name__ == "__main__":
    main()