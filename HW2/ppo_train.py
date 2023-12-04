import json
from datetime import datetime
from pdb import set_trace
from time import time

import gym
import munch
import numpy as np
import tensorflow as tf
from scipy import stats

from ppo_model import PPO, discount_cumsum


class PPOAgent:
    def __init__(self, config, env):
        self.env = env
        self.config = config
        self.iter_num = self.config.iter_num
        self.ppo = PPO(self.config)
        self.stamp = datetime.fromtimestamp(time()).strftime("%Y%m%d-%H%M%S")
        self.max_iter = config.max_iter
        self.log_freq = config.log_freq

    def train(self, render=False):
        train_log_dir = "logs/ppo/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            step_num = 0

            while not is_terminal:
                # Get action
                state = np.expand_dims(current_state, axis=0)
                action_prob = self.ppo.actor.net(state)

                set_trace()
                action = np.random.choice(
                    self.env.action_space.n, 1, p=action_prob[0].numpy()
                )[0]

                # Step
                next_state, reward, is_terminal, _ = self.env.step(action)

                # Obtain needed values
                value = self.ppo.critic.net(state)
                log_policy = tf.math.log(action_prob[:, action])

                # Save to buffer
                sample = {
                    "state": current_state,
                    "reward": reward,
                    "action": action,
                    "value": value,
                    "log_policy": log_policy,
                    "is_terminal": is_terminal,
                }
                self.ppo.buffer.append(sample)

                if render:
                    self.env.render()

                # Prepare for next time step
                current_state = next_state
                cummulative_reward += reward
                step_num += 1
                if step_num >= self.max_iter:
                    break

            actor_loss_value, critic_loss_value = self.train_step()
            self.ppo.buffer = []

            print("Iteration: {}, Reward: {}".format(episode, cummulative_reward))

            # Log to TensorBoard
            with train_summary_writer.as_default():
                tf.summary.scalar("actor_loss_value", actor_loss_value, step=episode)
                tf.summary.scalar("critic_loss_value", critic_loss_value, step=episode)
                tf.summary.scalar("reward", cummulative_reward, step=episode)

            # Save model
            if (episode + 1) % self.log_freq == 0:
                filename_actor = "models/{}/actor_{}".format(self.stamp, episode + 1)
                filename_critic = "models/{}/critic_{}".format(self.stamp, episode + 1)
                self.ppo.actor.net.save(filename_actor)
                self.ppo.critic.net.save(filename_critic)
                print(
                    "Model saved at {} and {}".format(filename_actor, filename_critic)
                )

    def train_step(self):
        # Load episode data
        episode_state = np.array([sample["state"] for sample in self.ppo.buffer])
        episode_reward = np.array([sample["reward"] for sample in self.ppo.buffer])
        episode_action = np.array([sample["action"] for sample in self.ppo.buffer])
        episode_is_terminal = np.array(
            [sample["is_terminal"] for sample in self.ppo.buffer]
        )
        episode_value = np.array([sample["value"] for sample in self.ppo.buffer])[
            :, 0, 0
        ]
        episode_log_policy_old = np.array(
            [sample["log_policy"] for sample in self.ppo.buffer]
        )[:, 0]

        # Placeholder to make dimension align with episode value
        episode_reward = np.concatenate((episode_reward, np.array([0.0])))
        # Value function at terminal state is 0
        episode_value = np.concatenate((episode_value, np.array([0.0])))
        # Calculate TD-error of each step, Eq(12) in paper
        episode_delta = (
            episode_reward[:-1]
            + self.ppo.gamma * episode_value[1:] * (1 - episode_is_terminal)
            - episode_value[:-1]
        )
        # Get advantage at each step, Eq(11) in paper
        episode_advantage = discount_cumsum(
            episode_delta, self.ppo.gamma * self.ppo.lam
        )
        episode_reward_to_go = discount_cumsum(episode_reward, self.ppo.gamma)[:-1]
        # Make sure the data type aligns
        episode_reward_to_go = tf.cast(episode_reward_to_go, tf.float32)

        # Normalize advantage
        advantage_stats = stats.describe(episode_advantage)
        advantage_mean = advantage_stats.mean
        advantage_std = np.sqrt(advantage_stats.variance)
        normalized_episode_advantage = (
            episode_advantage - advantage_mean
        ) / advantage_std
        # Make sure the data type aligns
        normalized_episode_advantage = tf.cast(normalized_episode_advantage, tf.float32)

        one_hot_actions = tf.keras.utils.to_categorical(
            episode_action, self.config.actor_output_size, dtype="float32"
        )

        actor_loss_value = 0
        critic_loss_value = 0

        for i in range(self.config.actor_update_per_iter):
            actor_loss_value_tmp, kl = self.optimize_actor_step(
                episode_state,
                episode_action,
                normalized_episode_advantage,
                episode_log_policy_old,
                one_hot_actions,
            )
            actor_loss_value += actor_loss_value_tmp
            if kl > 1.5 * self.config.target_kl:
                break

        actor_loss_value /= i

        for _ in range(self.config.critic_update_per_iter):
            critic_loss_value_tmp = self.optimize_critic_step(
                episode_state, episode_reward_to_go
            )
            critic_loss_value += (
                critic_loss_value_tmp / self.config.critic_update_per_iter
            )
        return actor_loss_value, critic_loss_value

    @tf.function
    def optimize_actor_step(
        self,
        episode_state,
        episode_action,
        normalized_episode_advantage,
        log_policy_old,
        one_hot_actions,
    ):
        with tf.GradientTape() as tape:
            action_probs = self.ppo.actor.net(episode_state)
            log_probs = tf.math.log(
                tf.reduce_sum(action_probs * one_hot_actions, axis=1)
            )
            ratio = tf.exp(log_probs - log_policy_old)
            clip_ratio = tf.cast(self.config.clip_ratio, tf.float32)
            tf_one = tf.constant(1.0, dtype=tf.float32)
            clipped_advantage = (
                tf.clip_by_value(ratio, tf_one - clip_ratio, tf_one + clip_ratio)
                * normalized_episode_advantage
            )
            loss_value = -tf.minimum(
                ratio * normalized_episode_advantage, clipped_advantage
            )
            loss_value = tf.reduce_mean(loss_value)

        kl = tf.reduce_mean(log_probs - log_policy_old)

        grads = tape.gradient(loss_value, self.ppo.actor.net.trainable_weights)
        self.ppo.actor.optimizer.apply_gradients(
            zip(grads, self.ppo.actor.net.trainable_weights)
        )
        return loss_value, kl

    @tf.function
    def optimize_critic_step(self, episode_state, episode_reward_to_go):
        with tf.GradientTape() as tape:
            value_estimate = self.ppo.critic.net(episode_state)
            loss_value = tf.reduce_mean(
                tf.square(value_estimate[:, 0] - episode_reward_to_go)
            )
        grads = tape.gradient(loss_value, self.ppo.critic.net.trainable_weights)
        self.ppo.critic.optimizer.apply_gradients(
            zip(grads, self.ppo.critic.net.trainable_weights)
        )
        return 0

    def test(self, filename, render=False):
        self.ppo.actor.load(filename)
        test_log_dir = "logs/ppo/" + self.stamp + "/test"
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for episode in range(self.config.test_iters):
            cummulative_reward = 0
            is_terminal = False
            current_state = self.env.reset()
            step_num = 0

            while not is_terminal:
                state = np.expand_dims(current_state, axis=0)
                action_prob = self.ppo.actor.net(state)
                action = np.random.choice(
                    self.env.action_space.n, 1, p=action_prob[0].numpy()
                )[0]
                next_state, reward, is_terminal, _ = self.env.step(action)
                if render:
                    self.env.render()

                current_state = next_state
                cummulative_reward += reward
                step_num += 1

                if step_num >= self.max_iter:
                    break

            self.env.close()

            with test_summary_writer.as_default():
                tf.summary.scalar("reward", cummulative_reward, step=episode)


def main():
    # env = gym.make("CartPole-v1").unwrapped
    # env = gym.wrappers.Monitor(env, "ppo_recording", force=True)
    env = gym.make("Pendulum-v0").unwrapped
    config_path = "ppo_pendulum_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)
    ppo_agent = PPOAgent(config, env)
    ppo_agent.train()
    # ppo_agent.test("models/actor_200/variables/variables")


if __name__ == "__main__":
    main()
