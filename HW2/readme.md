## DQN
A double DQN model is used to solve OpenAI Gym's `CarRacing-v0` environment. The double DQN algorithm is given in the [DQN Nature Paper](https://www.nature.com/articles/nature14236) and this implementation is inspired by [Andy Wu's repo](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN). 

Some tricks are applied to enhance performance:
* Each episode is terminated once the cumulative reward is below 0;
* After 300 time steps, if negative rewards are encountered more than 25 instances the episode is terminated;
* The episode is terminated after 1000 episodes;
* After each time step an optimization step is performed.

### DQN performance 
![DQN Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/carrace_train.png)
It can be seen that the test performance is superior to the training performance, this is probably due to using an epsilon-greedy policy in training. A video recording one of the test runs can be found [here](https://www.youtube.com/watch?v=KQclb-CsLTE).

## REINFORCE
A REINFORCE model is used to solve OpenAI Gym's `CartPole-v1` environment. During training, the episode length is set to be 1e4 time steps while in testing the episode length is 1e3. 

### REINFORCE performance
![REINFORCE Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/cartpole.png)
It can be seen that during testing the cartpole is able to balance at the top for all of the 1000 time steps, which lets it earn the maximum reward of 1000. A video recording one of the test runs can be found [here](https://www.youtube.com/watch?v=zldhflojbXc).

## PPO
In this section a PPO model is used to solve the OpenAI Gym `CartPole-v1` environment. This implementation is inspired by [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

### PPO performance
![PPO Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/cartpole_ppo.png)
It can be seen that its training efficiency is much higher than REINFORCE, it only takes 200 episodes to balance the cart-pole. In a separate experiment, when using similar settings to PPO, REINFORCE could not balance the cart-pole even after 1000 episodes.

### PPO Algorithm
![PPO Each Time Step](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/PPO_episode.png)
![PPO Algorithm](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/ppo_alg.png)
