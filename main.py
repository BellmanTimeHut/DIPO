import argparse
import numpy as np
import torch

from agent.DiPo import DiPo
from agent.replay_memory import ReplayMemory, DiffusionMemory

from tensorboardX import SummaryWriter
import gym
import os


def readParser():
    parser = argparse.ArgumentParser(description='Diffusion Policy')
    parser.add_argument('--env_name', default="Hopper-v3",
                        help='Mujoco Gym environment (default: Hopper-v3)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')

    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='env timesteps (default: 1000000)')

    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--update_actor_target_every', type=int, default=1, metavar='N',
                        help='update actor target per iteration (default: 1)')

    parser.add_argument("--policy_type", type=str, default="Diffusion", metavar='S',
                        help="Diffusion, VAE or MLP")
    parser.add_argument("--beta_schedule", type=str, default="cosine", metavar='S',
                        help="linear, cosine or vp")
    parser.add_argument('--n_timesteps', type=int, default=100, metavar='N',
                        help='diffusion timesteps (default: 100)')
    parser.add_argument('--diffusion_lr', type=float, default=0.0003, metavar='G',
                        help='diffusion learning rate (default: 0.0003)')
    parser.add_argument('--critic_lr', type=float, default=0.0003, metavar='G',
                        help='critic learning rate (default: 0.0003)')
    parser.add_argument('--action_lr', type=float, default=0.03, metavar='G',
                        help='diffusion learning rate (default: 0.03)')
    parser.add_argument('--noise_ratio', type=float, default=1.0, metavar='G',
                        help='noise ratio in sample process (default: 1.0)')

    parser.add_argument('--action_gradient_steps', type=int, default=20, metavar='N',
                        help='action gradient steps (default: 20)')
    parser.add_argument('--ratio', type=float, default=0.1, metavar='G',
                        help='the ratio of action grad norm to action_dim (default: 0.1)')
    parser.add_argument('--ac_grad_norm', type=float, default=2.0, metavar='G',
                        help='actor and critic grad norm (default: 1.0)')

    parser.add_argument('--cuda', default='cuda:0',
                        help='run on CUDA (default: cuda:0)')

    return parser.parse_args()


def evaluate(env, agent, writer, steps):
    episodes = 10
    returns = np.zeros((episodes,), dtype=np.float32)

    for i in range(episodes):
        state = env.reset()
        episode_reward = 0.
        done = False
        while not done:
            action = agent.sample_action(state, eval=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        returns[i] = episode_reward

    mean_return = np.mean(returns)

    writer.add_scalar(
            'reward/test', mean_return, steps)
    print('-' * 60)
    print(f'Num steps: {steps:<5}  '
              f'reward: {mean_return:<5.1f}')
    print('-' * 60)


def main(args=None):
    if args is None:
        args = readParser()

    device = torch.device(args.cuda)

    dir = "record"
    # dir = "test"
    log_dir = os.path.join(dir, f'{args.env_name}', f'policy_type={args.policy_type}', f'ratio={args.ratio}', f'seed={args.seed}')
    writer = SummaryWriter(log_dir)

    # Initial environment
    env = gym.make(args.env_name)
    state_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))
    print(action_size)

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    memory_size = 1e6
    num_steps = args.num_steps
    start_steps = 10000
    eval_interval = 10000
    updates_per_step = 1
    batch_size = args.batch_size
    log_interval = 10
    
    memory = ReplayMemory(state_size, action_size, memory_size, device)
    diffusion_memory = DiffusionMemory(state_size, action_size, memory_size, device)

    agent = DiPo(args, state_size, env.action_space, memory, diffusion_memory, device)

    steps = 0
    episodes = 0

    while steps < num_steps:
        episode_reward = 0.
        episode_steps = 0
        done = False
        state = env.reset()
        episodes += 1
        while not done:
            if start_steps > steps:
                action = env.action_space.sample()
            else:
                action = agent.sample_action(state, eval=False)
            next_state, reward, done, _ = env.step(action)

            mask = 0.0 if done else args.gamma

            steps += 1
            episode_steps += 1
            episode_reward += reward

            agent.append_memory(state, action, reward, next_state, mask)

            if steps >= start_steps:
                agent.train(updates_per_step, batch_size=batch_size, log_writer=writer)

            if steps % eval_interval == 0:
                evaluate(env, agent, writer, steps)
                # self.save_models()
                done =True

            state = next_state

        # if episodes % log_interval == 0:
        #     writer.add_scalar('reward/train', episode_reward, steps)

        print(f'episode: {episodes:<4}  '
            f'episode steps: {episode_steps:<4}  '
            f'reward: {episode_reward:<5.1f}')


if __name__ == "__main__":
    main()
