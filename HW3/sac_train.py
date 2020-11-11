import json
from datetime import datetime
from pdb import set_trace
from time import time

import gym
import munch
import numpy as np
import tensorflow as tf
from sac_model import SAC, ReplayBuffer
from scipy import stats

class SACAgent:
    def __init__(self, config, env):
        self.env = env
        self.config = config
        self.iter_num = self.config.iter_num
        self.sac = SAC(self.config)
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
        self.max_iter = config.max_iter
        self.log_freq = config.log_freq
        self.buffer = ReplayBuffer(self.config)
    
    def train(self, render=False):
        train_log_dir = "logs/sac/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            state = self.env.reset()
            step_num = 0

            while not is_terminal:
                # Get action
                state = np.expand_dims(state, axis=0)
                action, log_pi = self.sac.actor.get_action(state)
                action = action[0].numpy()
                # Step
                next_state, reward, is_terminal, _ = self.env.step(action)
                # Save sample to replay buffer
                sample = {
                    "state": state,
                    "next_state": next_state,
                    "reward": reward,
                    "action": action,
                    "terminal": is_terminal,
                }

                self.buffer.add_sample(sample)

                if render:
                    self.env.render()

                # Prepare for next time step
                state = next_state
                cummulative_reward += reward
                step_num += 1
                if step_num >= self.max_iter:
                    break
            
            set_trace()
    
    def train_step(self):
        pass
    
    @tf.function
    def opt_actor(self):
        pass
    
    @tf.function
    def opt_critic(self):
        pass
    
    def test(self):
        pass



def main():
    env = gym.make("BipedalWalkerHardcore-v3").unwrapped
    # env = gym.wrappers.Monitor(env, "ppo_recording", force=True)
    config_path = "sac_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    sac_agent = SACAgent(config, env)
    sac_agent.train(render=False)
    # ppo_agent.test("models/actor_200/variables/variables")


if __name__ == "__main__":
    main()
