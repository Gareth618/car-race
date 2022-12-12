import gym
import sys
import itertools
import numpy as np
from agent import Agent
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt

def is_int(string):
    try: return int(string) > 0
    except: return False

args = sys.argv[1:]
if len(args) == 2 and args[0] in ['train', 'continue'] and is_int(args[1]):
    mode = args[0]
    episodes = int(args[1])
elif len(args) == 1 and args[0] == 'test':
    mode = args[0]
else:
    print('wrong format')
    exit(1)

env = gym.make('CarRacing-v2', render_mode='rgb_array', domain_randomize=True)
env.action_space.seed(42)
env.reset(seed=13, options={'randomize': True})

def snapshot():
    matrix = env.render()
    matrix = matrix[:-50, 125:475]
    image = Image.fromarray(matrix)
    image = image.convert('L')
    matrix = np.array(image)
    for x in range(270, 335):
        for y in range(150, 200):
            matrix[x][y] = 100
    matrix = np.array([[[col, col, col] for col in row] for row in matrix])
    image = Image.fromarray(matrix)
    image = image.resize((100, 100))
    image = ImageEnhance.Contrast(image).enhance(3)
    matrix = np.array(image)
    matrix = np.array([[col[0] for col in row] for row in matrix])
    return matrix

steering = [-1, 0, 1]
gas = [0, 1]
breaking = [0, .2]
agent = Agent(
    list(itertools.product(steering, gas, breaking)), 1000,
    alpha=.001, gamma=.95, epsilon=1, epsilon_lower=.1, epsilon_decay=.001
)
if mode in ['continue', 'test']:
    agent.model.load_weights('model')

if mode in 'test':
    def take_action(action):
        step_reward = 0
        step_game_over = False
        for _ in range(3):
            _, reward, game_over, _, _ = env.step(action)
            step_reward += reward
            step_game_over |= game_over
        return snapshot(), step_reward, step_game_over

    step = 0
    while True:
        agent.step(snapshot(), take_action)
        step += 1
        if step % 10 == 0:
            plt.imshow(env.render())
            plt.show()

for episode in range(episodes):
    env.reset()
    episode_reward = 0
    negative_rewards = 0
    should_break = False

    def take_action(action):
        global episode_reward, negative_rewards, should_break
        step_reward = 0
        step_game_over = False
        for _ in range(3):
            _, reward, game_over, _, _ = env.step(action)
            step_reward += reward
            step_game_over |= game_over
        episode_reward += step_reward
        negative_rewards = negative_rewards + 1 if step_reward < 0 else 0
        should_break |= step_game_over
        return snapshot(), step_reward, step_game_over

    step = 0
    while True:
        agent.step(snapshot(), take_action)
        step += 1
        if step < 300:
            negative_rewards = 0
        should_break |= negative_rewards == 10
        if should_break: break
    agent.model.save_weights('model')
    print('finished episode', episode + 1, 'of', episodes, 'with reward', episode_reward)
