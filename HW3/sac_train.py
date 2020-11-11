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
        self.polyak_constant = config.polyak_constant

    def train(self, render=False):
        train_log_dir = "logs/sac/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        update_counter = 0

        # Assign eval_net weights to target_net
        self.sac.critic1_target.net.set_weights(self.sac.critic1_eval.net.get_weights())
        self.sac.critic2_target.net.set_weights(self.sac.critic2_eval.net.get_weights())

        # Placeholder of loss
        actor_loss = None
        critic_loss = None  

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
                
                # Update model weights
                if update_counter >= self.config.update_threshold:
                    if update_counter % self.config.update_freq:
                        actor_loss, critic_loss = self.train_step()
                update_counter += 1

            print("Iteration: {}, Reward: {}".format(
            episode, cummulative_reward))

            # Log to TensorBoard
            if actor_loss is not None:
                with train_summary_writer.as_default():
                    tf.summary.scalar("actor_loss_value",
                                    actor_loss, step=episode)
                    tf.summary.scalar("critic_loss_value",
                                    critic_loss, step=episode)
                    tf.summary.scalar("reward", cummulative_reward, step=episode)

            # Save model
            if (episode + 1) % self.log_freq == 0:
                names = ["actor", "critic1_eval", "critic1_target", "critic2_eval", "critic2_target"]
                nets = [self.sac.actor, self.sac.critic1_eval, self.sac.critic1_target, self.sac.critic2_eval, self.sac.critic2_target]
                for name, net in zip(names, nets):
                    filename = "models/{}/{}_{}".format(
                        name, self.stamp, episode + 1)
                    net.net.save(filename)
                    print("Model of {} saved at {}".format(name, filename))

    def train_step(self):
        actor_loss = self.opt_actor()
        critic_loss = self.opt_critic()

        critics = [[self.sac.critic1_eval, self.sac.critic1_target], 
                   [self.sac.critic2_eval, self.sac.critic2_target]]

        # Polyak Averaging
        for nets in critics:
            eval_weights = nets[0].net.get_weights()
            target_weights = nets[1].net.get_weights()
            update_weights = []
            for eval_weight, target_weight in zip(eval_weights, target_weights):
                update_weights.append(
                    self.polyak_constant * eval_weight
                    + (1 - self.polyak_constant) * target_weight
                )
            nets[1].net.set_weights(update_weights)
        return actor_loss, critic_loss

    @tf.function
    def opt_actor(self):
        loss_value = 0
        return loss_value

    @tf.function
    def opt_critic(self):
        loss_value = 0
        return loss_value

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
    sac_agent.train(render=True)


if __name__ == "__main__":
    main()
