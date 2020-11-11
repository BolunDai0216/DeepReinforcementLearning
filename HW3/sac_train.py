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
        self.batch_size = config.batch_size
        self.test_run = 0

        # Setup tensorboard logdir
        test_log_dir = "logs/sac/" + self.stamp + "/test"
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    def train(self, render=False):
        train_log_dir = "logs/sac/" + self.stamp + "/train"
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        update_counter = 0

        # Assign eval_net weights to target_net
        self.sac.critic1_target.net.set_weights(
            self.sac.critic1_eval.net.get_weights())
        self.sac.critic2_target.net.set_weights(
            self.sac.critic2_eval.net.get_weights())

        # Placeholder of loss
        actor_loss = None
        critic_loss = None

        for episode in range(self.iter_num):
            cummulative_reward = 0
            is_terminal = False
            state = self.env.reset()
            step_num = 0

            while not is_terminal:
                state = np.expand_dims(state, axis=0)

                # Get action
                if update_counter <= self.config.start_steps:
                    action = self.env.action_space.sample()
                else:
                    action, log_pi = self.sac.actor.get_action(state)
                    action = action[0].numpy()
                # Step
                next_state, reward, is_terminal, _ = self.env.step(action)

                # Clip falling reward
                if is_terminal and reward == -100:
                    reward = 0

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
                    if update_counter % self.config.update_freq == 0:
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
                    tf.summary.scalar(
                        "reward", cummulative_reward, step=episode)

            # Save model
            if (episode + 1) % self.log_freq == 0:
                names = ["actor", "critic1_eval", "critic1_target",
                         "critic2_eval", "critic2_target"]
                nets = [self.sac.actor, self.sac.critic1_eval, self.sac.critic1_target,
                        self.sac.critic2_eval, self.sac.critic2_target]
                for name, net in zip(names, nets):
                    filename = "models/{}/{}_{}".format(
                        name, self.stamp, episode + 1)
                    net.net.save(filename)
                    print("Model of {} saved at {}".format(name, filename))
                
                self.test()


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
        batch_reward = tf.cast(batch_reward, tf.float32)
        # Shape [batch_size, action_size]
        batch_action = np.array([sample["action"] for sample in batch])
        # Shape [batch_size, ]
        batch_terminal = np.array([sample["terminal"] for sample in batch])
        batch_terminal = tf.cast(batch_terminal, tf.float32)

        actor_loss = self.opt_actor(batch_state)
        critic_loss = self.opt_critic(
            batch_state, batch_next_state, batch_reward, batch_action, batch_terminal)

        critics = [[self.sac.critic1_eval, self.sac.critic1_target],
                   [self.sac.critic2_eval, self.sac.critic2_target]]

        # Polyak Averaging
        for nets in critics:
            eval_weights = nets[0].net.get_weights()
            target_weights = nets[1].net.get_weights()
            update_weights = []
            for eval_weight, target_weight in zip(eval_weights, target_weights):
                update_weights.append(
                    self.polyak_constant * target_weight
                    + (1 - self.polyak_constant) * eval_weight
                )
            nets[1].net.set_weights(update_weights)
        return actor_loss, critic_loss

    @tf.function
    def opt_actor(self, batch_state):
        with tf.GradientTape() as tape_a:
            action, log_prob = self.sac.actor.get_action(batch_state)
            q1_eval = self.sac.critic1_eval.net([batch_state, action])
            q2_eval = self.sac.critic2_eval.net([batch_state, action])
            q_eval = tf.math.minimum(q1_eval, q2_eval)
            loss_value = tf.reduce_mean(self.config.alpha*log_prob - q_eval)

        grads = tape_a.gradient(loss_value, self.sac.actor.net.trainable_weights)
        self.sac.actor.optimizer.apply_gradients(
            zip(grads, self.sac.actor.net.trainable_weights)
        )
        return loss_value

    @tf.function
    def opt_critic(self, batch_state, batch_next_state, batch_reward, batch_action, batch_terminal):
        action, log_prob = self.sac.actor.get_action(batch_next_state)
        q1_target = self.sac.critic1_target.net([batch_next_state, action])
        q2_target = self.sac.critic2_target.net([batch_next_state, action])
        q_target = tf.math.minimum(q1_target, q2_target)
        y = batch_reward + self.config.gamma * \
            (1-batch_terminal)*(q_target-self.config.alpha*log_prob)

        with tf.GradientTape(persistent=True) as tape:
            q1 = self.sac.critic1_eval.net([batch_state, batch_action])
            q2 = self.sac.critic2_eval.net([batch_state, batch_action])
            loss_q1 = tf.reduce_mean((q1 - y)**2)
            loss_q2 = tf.reduce_mean((q2 - y)**2)
            loss_value = loss_q1 + loss_q2

        grads1 = tape.gradient(
            loss_q1, self.sac.critic1_eval.net.trainable_weights)
        grads2 = tape.gradient(
            loss_q2, self.sac.critic2_eval.net.trainable_weights)
        self.sac.critic1_eval.optimizer.apply_gradients(
            zip(grads1, self.sac.critic1_eval.net.trainable_weights)
        )
        self.sac.critic2_eval.optimizer.apply_gradients(
            zip(grads2, self.sac.critic2_eval.net.trainable_weights)
        )
        return loss_value

    def test(self, filename=None, render=False):
        # Load file
        if filename is not None:
            self.sac.actor.load(filename)

        for i in range(self.config.test_iters):
            cummulative_reward = 0
            is_terminal = False
            state = self.env.reset()
            step_num = 0

            while not is_terminal:
                state = np.expand_dims(state, axis=0)
                # Get action
                action, log_pi = self.sac.actor.get_action(state, test=True)
                action = action[0].numpy()
                # Step
                next_state, reward, is_terminal, _ = self.env.step(action)

                if render:
                    self.env.render()

                # Prepare for next time step
                state = next_state
                cummulative_reward += reward
                step_num += 1
                if step_num >= self.max_iter:
                    break

            self.test_run += 1
            print("test run {}, reward: {}".format(self.test_run, cummulative_reward))

            # Log to TensorBoard
            with self.test_summary_writer.as_default():
                tf.summary.scalar(
                    "test reward", cummulative_reward, step=self.test_run)


def main():
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_visible_devices(gpus[2], "GPU")
    tf.config.experimental.set_memory_growth(gpus[2], True)

    with tf.device("/device:GPU:2"):
        env = gym.make("BipedalWalkerHardcore-v3")
        env = BipedalWalkerHardcoreWrapper(env)
        # env = gym.wrappers.Monitor(env, "ppo_recording", force=True)
        config_path = "sac_config.json"
        with open(config_path) as json_file:
            config = json.load(json_file)
        config = munch.munchify(config)
        sac_agent = SACAgent(config, env)
        sac_agent.train(render=False)
        # sac_agent.test(filename="models/actor/20201111-161316_9100/variables/variables")


if __name__ == "__main__":
    main()
