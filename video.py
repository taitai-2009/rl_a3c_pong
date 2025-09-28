from collections import deque
import torch, torch.nn.functional as F
import gym
from gym.wrappers import RecordVideo        # ← 追加
from envs import create_atari_env
from model import ActorCritic

def record_once(env_name, shared_model, episode_counter, save_dir="./videos"):
    env = create_atari_env(env_name)

    name_prefix = f"pong_a3c_{episode_counter:3d}".replace(" ", "_")
    env = RecordVideo(
        env,
        save_dir,
        episode_trigger=lambda ep_no: True,   # 1 エピソード目だけ録画
        name_prefix=name_prefix
    )
    seed = 40
    env.seed(seed)
    torch.manual_seed(seed)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.load_state_dict(shared_model.state_dict())
    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    hx = torch.zeros(1, 256)
    cx = torch.zeros(1, 256)

    done = False
    while not done:
        with torch.no_grad():
            _, logits, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logits, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        state = torch.from_numpy(state)

    env.close()
    print("MP4 has been saved in ", save_dir)

