import numpy as np
import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, counter, lock, optimizer):
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0
    max_episode_length = 10000
    
    # ��ԊO��repeat�ɑ���
    while True:
        # �{����Reset gradients�̈ʒu: (global����Actor, Critic�����Ƃ�)
        # optimizer.zero_grad(set_to_none=True)

        # Synchronize thread-specific parameters = and v = v
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []   # ← 追加：エントロピー蓄積

        for step in range(args.num_steps):
            episode_length += 1
            # model�o�͂ɂ��Ă�model.py����def forward()�̍ŏI�s���Q��
            value, logit, (hx, cx) = model((state.unsqueeze(0), # state���A1�o�b�`���̃f�[�^�Ƃ���PyTorch���f���ɓn��
                                            (hx, cx)))          # LSTM�� hidden state��cell state

            # logit�����Ɏ�action������
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1).detach() # �m�����zprob�ɉ����ă����_����1�̍s�����T���v�����O
            # 方策（確率）と対数確率
            logp_full = F.log_softmax(logit, dim=-1)
            prob      = torch.softmax(logit, dim=-1)

            # エントロピー（探索性）：-sum p*logp
            entropy_t = -(logp_full * prob).sum(dim=1)  # [batch=1], H(π(s_t))
            entropies.append(entropy_t.mean())

            # 行動サンプリング
            action = prob.multinomial(num_samples=1).detach()

            # ���ɑ΂�action���N����
            state, reward, done, _ = env.step(action.numpy().item())
            done = done or episode_length >= max_episode_length

            # ��Ō��z�v�Z�ɕK�v�� log�m�� �ރ�_v * log��(a|s;��')��ۑ�����
            # ���ۂɎ�����s����log�m�������𔲂�����Ă���
            #log_prob = F.log_softmax(logit, dim=-1)
            #log_prob = -log_prob.gather(1, action)
            # 取った行動の -log π(a|s)
            log_prob = -logp_full.gather(1, action)

            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)

            # ���̎��̉��l�A��V�Alog��(a|s;��')��ۑ����A�Ō�̃��[�v�Ŏg��
            values.append(value)
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        policy_loss = 0
        value_loss = 0
        for i in reversed(range(len(rewards))):
            R = R * args.gamma + rewards[i]  # WRITE ME
            advantage = R - values[i]  # WRITE ME
            policy_loss = policy_loss + log_probs[i] * advantage.detach()
            value_loss = value_loss + 0.5 * advantage.pow(2)


        # Reset gradients: (global����Actor, Critic�����Ƃ�)
        # �_���ʂ�Ȃ�Ζ{�� while True: �̒���Ȃ̂����A��������Ɗw�K�������������
        optimizer.zero_grad()

        #(policy_loss + value_loss).backward()
        # エントロピー正則化を追加
        entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
        loss = policy_loss + value_loss - args.entropy_coef * entropy
        loss.backward()
        
        # amount of gradient norm clipping�͘_��5.1�͂Ɍ��y����
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()
