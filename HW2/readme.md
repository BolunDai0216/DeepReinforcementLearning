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
In this section a PPO model is used to solve the OpenAI Gym `CartPole-v1` environment. 

### PPO performance
![PPO Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/cartpole_ppo.png)
It can be seen that its training efficiency is much higher than REINFORCE, it only takes 200 episodes to balance the cart-pole. In a separate experiment, when using similar settings to PPO, REINFORCE could not balance the cart-pole even after 1000 episodes.

### PPO Algorithm
Below I provide a detailed walk-through of the PPO algorithm, which is inspired by [OpenAI Spinning Up](https://spinningup.openai.com). At each time step the policy <img src="https://render.githubusercontent.com/render/math?math=\pi(s)"> outputs a probability distribution over the actions <img src="https://render.githubusercontent.com/render/math?math=a_i">, and an action <img src="https://render.githubusercontent.com/render/math?math=a"> is sampled. Then the state <img src="https://render.githubusercontent.com/render/math?math=s">, action <img src="https://render.githubusercontent.com/render/math?math=a">, its corresponding probability density <img src="https://render.githubusercontent.com/render/math?math=p(a)">, its log probability <img src="https://render.githubusercontent.com/render/math?math=\log[p(a)]">, the value function estimation at the current state <img src="https://render.githubusercontent.com/render/math?math=v">, the terminal info and the reward <img src="https://render.githubusercontent.com/render/math?math=r"> is stored in a buffer.
![PPO Each Time Step](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW2/img/PPO_episode.png)
After each episode ends PPO will update the actor and critic network. To update the actor network we first need to calculate the TD-error at each time step <img src="https://render.githubusercontent.com/render/math?math=\delta_t = r_t + \gamma(1 - d)V(s_{t+1}) - V(s_t)">, where <img src="https://render.githubusercontent.com/render/math?math=d = 1"> only at terminal states. Also for the terminal state we have the value function as 0, i.e., at state <img src="https://render.githubusercontent.com/render/math?math=s_t"> after applying action <img src="https://render.githubusercontent.com/render/math?math=a_{i, t}"> the episode is terminated, then we set <img src="https://render.githubusercontent.com/render/math?math=V(s_{t+1}) = 0">. Then we can calculate the generalized advantage at each time step as <img src="https://render.githubusercontent.com/render/math?math=\hat{A}_t = \delta_t - (-1)(\gamma\lambda)\delta_{t+1} + \cdots - (-1)(\gamma\lambda)^{T-(t-1)}\delta_{T-1}">. Then we normalize the advantage array for this episode. Before performing any gradient updates the log probabilities of the actions performed based on the current policy <img src="https://render.githubusercontent.com/render/math?math=\log(\pi_{\mathrm{old}}(s_i, a_i))"> are stored. The loss is then calculated as <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\mathrm{actor}} = -\frac{1}{T}\sum_{t=0}^{T}\min\{\exp{(\mathrm{ratio})}\hat{A}_t, \mathrm{clip}(\mathrm{ratio}, 1 - \mathrm{ratio}, 1 - (-\mathrm{ratio}))\hat{A}_t\}"> where <img src="https://render.githubusercontent.com/render/math?math=\mathrm{ratio} = \log(\pi_{\mathrm{current}}(s_i, a_i)) - \log(\pi_{\mathrm{old}}(s_i, a_i))"> and <img src="https://render.githubusercontent.com/render/math?math=T"> is the episode length. We take a fixed number of gradient steps on <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\mathrm{actor}}"> or when the KL-divergence of <img src="https://render.githubusercontent.com/render/math?math=\pi_{\mathrm{current}}(s_i, a_i))"> and <img src="https://render.githubusercontent.com/render/math?math=\pi_{\mathrm{old}}(s_i, a_i)"> is larger then a threshold, whichever happens first. 

To update the critic, we simply calculate the target as the reward-to-go <img src="https://render.githubusercontent.com/render/math?math=G_t">. The critic loss is <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\mathrm{critic}} = \frac{1}{T}\sum_{t=0}^{T}(G_t - V(s_t))^2">. We take a fixed number of gradient steps on <img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}_{\mathrm{critic}}">.

The <img src="https://render.githubusercontent.com/render/math?math=-(-1)">'s are shown because for some unknown reason I cannot show the plus sign.
