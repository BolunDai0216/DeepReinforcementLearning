## SAC

The soft actor-critic (SAC) algorithm is adapted from [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html). Additional changes to the training are implemented inspired by [CreateAMind](https://mp.weixin.qq.com/s/8vgLGcpsWkF89ma7T2twRA).

Some tricks are applied to enhance performance:
* During training the actions are repeated for three steps.
* When the agent falls, the terminal reward is clipped to zero.
* The reward is scaled by 5.
* Noises are added to the actions.
* When training, the probability of encountering a stump is increased.

### SAC performance 
![SAC Performance](https://github.com/BolunDai0216/DeepReinforcementLearning/blob/main/HW3/img/bipedalwalkerharcore-v3.png)
The discrepancy between the reward during training and testing is due to reward scaling during training. A video showing the trained model can be found [Here](https://youtu.be/SyonPJc8hMw).


