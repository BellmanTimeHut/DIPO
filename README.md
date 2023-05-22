## Policy Representation via Diffusion Probability Model for Reinforcement Learning

**Policy Representation via Diffusion Probability Model for Reinforcement Learning**<br>


Abstract: *Popular reinforcement learning (RL) algorithms tend to produce a unimodal policy distribution, which weakens the expressiveness of complicated policy and decays the ability of exploration. The diffusion probability model is powerful to learn complicated multimodal distributions, which has shown promising and potential applications to RL. In this paper, we formally build a theoretical foundation of policy representation via the diffusion probability model and provide practical implementations of diffusion policy for online model-free RL. Concretely, we character diffusion policy as a stochastic process, which is a new approach to representing a policy. Then we present a convergence guarantee for diffusion policy, which provides a theory to understand the multimodality of diffusion policy. Furthermore, we propose the DIPO which is an implementation for model-free online RL with DIffusion POlicy. To the best of our knowledge, DIPO is the first algorithm to solve model-free online RL problems with the diffusion model. Finally, extensive empirical results show the effectiveness and superiority of DIPO on the standard continuous control MoJoCo benchmark.*

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `DIPO` can be created and activated with:
```.bash
conda create DIPO
conda activate DIPO
```
To get started, install the additionally required python packages into you environment.
```.bash
pip install -r requirements.txt
```

### Running
Running experiments based our code could be quite easy, so below we use `Hopper-v3` task as an example. 

```.bash
python main.py --env_name Hopper-v3 --num_steps 1000000 --n_timesteps 100 --cuda 0 --seed 0
```


### Hyperparameters
Hyperparameters for DIPO have been shown as follow for easily reproducing our reported results.

#### Hyper-parameters for algorithms
| Hyperparameter | DIPO | SAC | TD3 | PPO |
| -------------- | ---- | --- | --- | --- |
| No. of hidden layers | 2 | 2 | 2 | 2 |
| No. of hidden nodes | 256 | 256  | 256  | 256  |
| Activation | mish | relu | relu | tanh |
| Batch size | 256 | 256 | 256 | 256 |
| Discount for reward $\gamma$ | 0.99 | 0.99 | 0.99 | 0.99 |
| Target smoothing coefficient $\tau$ | 0.005 | 0.005 | 0.005 | 0.005 |
| Learning rate for actor | $3 × 10^{-4}$ | $3 × 10^{-4}$ | $3 × 10^{-4}$ | $7 × 10^{-4}$ |
| Learning rate for actor | $3 × 10^{-4}$ | $3 × 10^{-4}$ | $3 × 10^{-4}$ | $7 × 10^{-4}$ |
| Actor Critic grad norm | 2 | N/A | N/A | 0.5 |
| Memeroy size | $1 × 10^6$ | $1 × 10^6$ | $1 × 10^6$ | $1 × 10^6$ |
| Entropy coefficient | N/A | 0.2 | N/A | 0.01 |
| Value loss coefficient | N/A | N/A | N/A | 0.5 |
| Exploration noise | N/A | N/A | $\mathcal{N}$(0, 0.1) | N/A |
| Policy noise | N/A | N/A | $\mathcal{N}$(0, 0.2) | N/A |
| Noise clip | N/A | N/A | 0.5 | N/A |
| Use gae | N/A | N/A | N/A | True |

#### Hyper-parameters for MuJoCo.(DIPO)
| Hyperparameter | Hopper-v3 | Walker2d-v3 | Ant-v3 | HalfCheetah-v3 | Humanoid-v3 |
| --- | --- | --- | --- | --- | --- |
| Learning rate for action | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| Actor Critic grad norm | 1 | 2 | 0.8 | 2 | 2 |
| Action grad norm ratio | 0.3 | 0.08 | 0.1 | 0.08 | 0.1 |
| Action gradient steps | 20 | 20 | 20 | 40 | 20 |
| Diffusion inference timesteps | 100 | 100 | 100 | 100 | 100 |
| Diffusion beta schedule | cosine | cosine | cosine | cosine | cosine |
| Update actor target every | 1 | 1 | 1 | 2 | 1 |


### Difussion Policy
