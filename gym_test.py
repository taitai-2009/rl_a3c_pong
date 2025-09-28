import torch
import gym
from gym.wrappers import RecordVideo

# 環境の作成
env = gym.make('PongDeterministic-v4')

env = RecordVideo(
    env,
    ".",
    episode_trigger=lambda ep_no: True,   # 1 エピソード目だけ録画
    name_prefix="test_video"
)

# 環境の初期化
state = env.reset()
state = torch.from_numpy(state)
hx = torch.zeros(1, 256)
cx = torch.zeros(1, 256)

for _ in range(1000):
    action = env.action_space.sample()  # ランダムな行動を選択
    state, reward, done, _ = env.step(action)  # 行動を環境に適用

    if done:
        break

env.close()
