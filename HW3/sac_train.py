import json
from datetime import datetime
from pdb import set_trace
from time import time

import gym
import munch
import numpy as np
import tensorflow as tf
from sac_model import SAC, ReplayBuffer, BipedalWalkerHardcoreWrapper
from scipy import stats

# Implemented some tricks in https://mp.weixin.qq.com/s/8vgLGcpsWkF89ma7T2twRA
# run using: xvfb-run -a python3 sac_train.py


class SACAgent:
    def __init__(self, config, env, test_env, expert_sac_file):
        self.env = env
        self.test_env = test_env
        self.config = config
        self.iter_num = self.config.iter_num
        self.sac = SAC(self.config)
        self.expert_sac = SAC(self.config)
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
        self.max_iter = config.max_iter
        self.log_freq = config.log_freq
        self.buffer = ReplayBuffer(self.config)
        self.polyak_constant = config.polyak_constant
        self.batch_size = config.batch_size
        self.test_run = 0
        self.expert_sac_file = expert_sac_file

        # Setup tensorboard logdir
        test_log_dir = "logs/sac/" + self.stamp + "/test"
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        train_log_dir = "logs/sac/" + self.stamp + "/train"
        self.train_summary_writer = tf.summary.create_file_writer(
            train_log_dir)

    def burn_in(self, filename=None):
        if filename is not None:
            self.expert_sac.actor.load(filename)

        state = self.env.reset()
        for i in range(self.config.burn_in_size):
            state = np.expand_dims(state, axis=0)

            action, _ = self.expert_sac.actor.get_action(state)
            action = action[0].numpy()

            # action = self.env.action_space.sample()
            next_state, reward, is_terminal, _ = self.env.step(action)
            sample = {
                "state": state,
                "next_state": next_state,
                "reward": reward,
                "action": action,
                "terminal": is_terminal,
            }

            self.buffer.add_sample(sample)

            if is_terminal:
                state = self.env.reset()
            else:
                state = next_state

    def train(self, render=False):
        self.update_counter = 0
        max_test_reward = -1000

        # Assign eval_net weights to target_net
        self.sac.critic1_target.net.set_weights(
            self.sac.critic1_eval.net.get_weights())
        self.sac.critic2_target.net.set_weights(
            self.sac.critic2_eval.net.get_weights())

        # Placeholder of loss
        actor_loss = None
        critic_loss = None

        self.burn_in(filename=self.expert_sac_file)

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            state = self.env.reset()
            step_num = 0
            self.epoch = episode

            while not is_terminal:
                state = np.expand_dims(state, axis=0)
                # Get action
                action, _ = self.sac.actor.get_action(state)
                action = action[0].numpy()
                self.update_counter += 1

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
            for _ in range(10000):
                actor_loss, critic_loss = self.train_step()
                self.polyak_averaging()

            print(
                "Iteration: {}, Reward: {}, t: {}".format(
                    episode, cummulative_reward, self.update_counter
                )
            )

            # Log to TensorBoard
            with self.train_summary_writer.as_default():
                tf.summary.scalar("actor_loss_value", actor_loss, step=episode)
                tf.summary.scalar("critic_loss_value",
                                  critic_loss, step=episode)
                tf.summary.scalar("reward", cummulative_reward, step=episode)

            # Save model
            if (episode + 1) % self.log_freq == 0:
                test_reward = self.test()

                # If best model save it
                if test_reward > max_test_reward:
                    max_test_reward = test_reward
                    names = [
                        "actor",
                        "critic1_eval",
                        "critic1_target",
                        "critic2_eval",
                        "critic2_target",
                    ]
                    nets = [
                        self.sac.actor,
                        self.sac.critic1_eval,
                        self.sac.critic1_target,
                        self.sac.critic2_eval,
                        self.sac.critic2_target,
                    ]
                    for name, net in zip(names, nets):
                        filename = "models/{}/{}_{}".format(
                            name, self.stamp, episode + 1
                        )
                        net.net.save(filename)
                        print("Model of {} saved at {}".format(name, filename))

    def train_step(self):
        batch = self.buffer.get_samples(self.batch_size)
        # Shape [batch_size, observation_size]
        batch_state = np.array([sample["state"] for sample in batch])[:, 0, :]
        batch_state = tf.cast(batch_state, tf.float32)
        # Shape [batch_size, observation_size]
        batch_next_state = np.array([sample["next_state"] for sample in batch])
        batch_next_state = tf.cast(batch_next_state, tf.float32)
        # Shape [batch_size, ]
        batch_reward = np.array([sample["reward"] for sample in batch])
        batch_reward = np.expand_dims(batch_reward, axis=1)
        batch_reward = tf.cast(batch_reward, tf.float32)
        # Shape [batch_size, action_size]
        batch_action = np.array([sample["action"] for sample in batch])
        # Shape [batch_size, ]
        batch_terminal = np.array([sample["terminal"] for sample in batch])
        batch_terminal = np.expand_dims(batch_terminal, axis=1)
        batch_terminal = tf.cast(batch_terminal, tf.float32)

        # Get Q update target
        action, log_prob = self.sac.actor.get_action(batch_next_state)
        q1_target = self.sac.critic1_target.net([batch_next_state, action])
        q2_target = self.sac.critic2_target.net([batch_next_state, action])
        q_target = tf.math.minimum(q1_target, q2_target)
        y = batch_reward + self.config.gamma * (1 - batch_terminal) * (
            q_target - self.config.alpha * log_prob
        )

        # Update actor and critic networks
        critic_loss = self.opt_critic(batch_state, batch_action, y)
        actor_loss = self.opt_actor(batch_state)

        return actor_loss, critic_loss

    def polyak_averaging(self):
        # Update Q1
        eval_weights = self.sac.critic1_eval.net.get_weights()
        target_weights = self.sac.critic1_target.net.get_weights()
        update_weights = []
        for eval_weight, target_weight in zip(eval_weights, target_weights):
            update_weights.append(
                self.polyak_constant * target_weight
                + (1 - self.polyak_constant) * eval_weight
            )
        self.sac.critic1_target.net.set_weights(update_weights)

        # Update Q2
        eval_weights = self.sac.critic2_eval.net.get_weights()
        target_weights = self.sac.critic2_target.net.get_weights()
        update_weights = []
        for eval_weight, target_weight in zip(eval_weights, target_weights):
            update_weights.append(
                self.polyak_constant * target_weight
                + (1 - self.polyak_constant) * eval_weight
            )
        self.sac.critic2_target.net.set_weights(update_weights)

    @tf.function
    def opt_actor(self, batch_state):
        with tf.GradientTape() as tape:
            action, log_prob = self.sac.actor.get_action(batch_state)
            q1_eval = self.sac.critic1_eval.net([batch_state, action])
            q2_eval = self.sac.critic2_eval.net([batch_state, action])
            q_eval = tf.math.minimum(q1_eval, q2_eval)
            loss_value = tf.reduce_mean(self.config.alpha * log_prob - q_eval)

        grads = tape.gradient(loss_value, self.sac.actor.net.trainable_weights)
        self.sac.actor.optimizer.apply_gradients(
            zip(grads, self.sac.actor.net.trainable_weights)
        )
        return loss_value

    @tf.function
    def opt_critic(self, batch_state, batch_action, y):
        with tf.GradientTape() as tape:
            q1 = self.sac.critic1_eval.net([batch_state, batch_action])
            q2 = self.sac.critic2_eval.net([batch_state, batch_action])
            loss_q1 = tf.reduce_mean((q1 - y) ** 2)
            loss_q2 = tf.reduce_mean((q2 - y) ** 2)
            loss_value = loss_q1 + loss_q2

        weights = self.sac.critic1_eval.net.trainable_weights
        weights.extend(self.sac.critic2_eval.net.trainable_weights)
        grads = tape.gradient(loss_value, weights)
        self.sac.critic_opt.apply_gradients(zip(grads, weights))

        return loss_value

    def test(self, filename=None, render=False, standalone=False):
        # Load file
        if filename is not None:
            self.sac.actor.load(filename)

        test_iter_num = self.config.test_iters
        test_rewards = 0

        for _ in range(test_iter_num):
            cummulative_reward = 0
            is_terminal = False
            state = self.env.reset()
            step_num = 0

            while not is_terminal:
                state = np.expand_dims(state, axis=0)
                # Get action
                action, _ = self.sac.actor.get_action(state, test=True)
                action = action[0].numpy()
                # Step
                next_state, reward, is_terminal, _ = self.test_env.step(action)

                if render:
                    self.env.render()

                # Prepare for next time step
                state = next_state
                cummulative_reward += reward
                step_num += 1
                if step_num >= self.max_iter:
                    break

            self.test_run += 1
            print("test run {}, reward: {}".format(
                self.test_run, cummulative_reward))
            test_rewards += cummulative_reward

        # Log to TensorBoard
        test_rewards = test_rewards / test_iter_num
        if not standalone:
            with self.test_summary_writer.as_default():
                tf.summary.scalar("test reward", test_rewards,
                                  step=self.update_counter)

        return test_rewards


def main():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[2], "GPU")
    tf.config.experimental.set_memory_growth(gpus[2], True)

    with tf.device("/device:GPU:2"):
        env = gym.make("BipedalWalkerHardcore-v3")
        # train_env = BipedalWalkerHardcoreWrapper(env)
        # env = gym.wrappers.Monitor(env, "sac_recording", force=True)
        config_path = "sac_config.json"
        with open(config_path) as json_file:
            config = json.load(json_file)
        config = munch.munchify(config)

        # filename = "models/actor/20201112-140453_1700/variables/variables"
        filename = "models/actor/20201112-152302_2600/variables/variables"

        sac_agent = SACAgent(config, env, env, filename)
        sac_agent.train(render=False)
        # sac_agent.test(
        #     filename=filename,
        #     render=True,
        #     standalone=True,
        # )


if __name__ == "__main__":
    main()
