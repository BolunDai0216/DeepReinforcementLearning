## DQN
A double DQN model is used to solve the OpenAI Gym's `CarRacing-v0` environment. The double DQN algorithm is given in the [DQN Nature Paper](https://www.nature.com/articles/nature14236) and the implementation is inspired by [Andy Wu's repo](https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN). Some tricks are used to make the algorithm behave better:
* Each episode is terminated once the cumulative reward is below 0;
* After 300 time steps, if negative rewards are encountered more than 25 instances the episode is terminated;
* The episode is terminated after 1000 episodes;
* After each time step an optimization step is performed.

### DQN performance 
![DQN Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/carrace_train.png)
It can be seen that the test performance is superior to the training performance, this is probably due to using an epsilon-greedy policy in training. A video recording one of the testing runs can be found [here](https://www.youtube.com/watch?v=KQclb-CsLTE).

## REINFORCE
A REINFORCE model is used to solve the OpenAI Gym `CartPole-v1` environment. During training, the episode length is set to be 1e4 time steps while in testing the episode length is 1e3. 

### REINFORCE performance
![REINFORCE Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/cartpole.png)
It can be seen that during testing the cartpole is able to balance at the top for all of the 1000 time steps, which lets it earn the maximum reward of 1000. A video recording one of the testing runs can be found [here](https://www.youtube.com/watch?v=zldhflojbXc).
