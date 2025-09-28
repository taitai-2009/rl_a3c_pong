# 強化学習A3CによるPongオペレーションの改善 (A3C-based Reinforcement Learning for Pong Improvement)

`Code Name: rl_a3c_pong`

## サンプルコードからエントロピー正則化を加えた際の変更点

`main.py`

```
--- a/main.py
+++ b/main.py
@@ -33,6 +33,8 @@ parser.add_argument('--num-steps', type=int, default=20,
                     help='number of forward steps in A3C (default: 20)')
 parser.add_argument('--value-loss-coef', type=float, default=0.5,
                     help='value loss coefficient (default: 0.5)')
+parser.add_argument('--entropy-coef', type=float, default=0.01,
+                    help='entropy bonus coefficient (default: 0.01)')
 parser.add_argument('--max-grad-norm', type=float, default=50,
                     help='value loss coefficient (default: 50)')
```

`train.py`

```
@@
 def train(rank, args, shared_model, counter, lock, optimizer):
@@
-        values = []
-        log_probs = []
-        rewards = []
+        values = []
+        log_probs = []
+        rewards = []
+        entropies = []   # ← 追加：エントロピー蓄積
 
         for step in range(args.num_steps):
             episode_length += 1
@@
-            prob = F.softmax(logit, dim=-1)
-            action = prob.multinomial(num_samples=1).detach() # …
+            # 方策（確率）と対数確率
+            logp_full = F.log_softmax(logit, dim=-1)
+            prob      = torch.softmax(logit, dim=-1)
+
+            # エントロピー（探索性）：-sum p*logp
+            entropy_t = -(logp_full * prob).sum(dim=1)  # [batch=1]
+            entropies.append(entropy_t.mean())
+
+            # 行動サンプリング
+            action = prob.multinomial(num_samples=1).detach()
@@
-            log_prob = F.log_softmax(logit, dim=-1)
-            log_prob = -log_prob.gather(1, action)
+            # 取った行動の -log π(a|s)
+            log_prob = -logp_full.gather(1, action)
@@
         policy_loss = 0
         value_loss = 0
         for i in reversed(range(len(rewards))):
             R = R * args.gamma + rewards[i]  # WRITE ME
             advantage = R - values[i]  # WRITE ME
             policy_loss = policy_loss + log_probs[i] * advantage.detach()
             value_loss = value_loss + 0.5 * advantage.pow(2)
 
@@
-        (policy_loss + value_loss).backward()
+        # エントロピー正則化を追加
+        entropy = torch.stack(entropies).mean() if entropies else torch.tensor(0.0)
+        loss = policy_loss + value_loss - args.entropy_coef * entropy
+        loss.backward()
```

## サンプルコードの挙動

- 初めて +21 点で勝利するまで  
  → 約 **5分33秒 / 105万ステップ**  
  → 報酬推移：-21 → -2 → -21 → -2 → -19 → +21  
- 勝てるが立ち上がりが遅く、安定性に欠ける  


```
(venv) $ python3 main.py 
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
env.observation_space.shape: (1, 42, 42)
env action meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
num of processes: 16
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]

A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
Time 00h 00m 00s, total num steps 984, FPS 1655, episode reward -21.0, episode length 813
recording
MP4 has been saved in  ./videos
test.py (line 76): test(): episode 1
Time 00h 01m 03s, total num steps 200848, FPS 3183, episode reward -2.0, episode length 106
test.py (line 76): test(): episode 2
Time 00h 02m 09s, total num steps 411548, FPS 3184, episode reward -21.0, episode length 2575
test.py (line 76): test(): episode 3
Time 00h 03m 17s, total num steps 626747, FPS 3178, episode reward -2.0, episode length 4912
test.py (line 76): test(): episode 4
Time 00h 04m 26s, total num steps 844732, FPS 3174, episode reward -19.0, episode length 3562
test.py (line 76): test(): episode 5
Time 00h 05m 33s, total num steps 1056619, FPS 3169, episode reward 21.0, episode length 2836
recording
/home/XXXXX/venv/lib/python3.10/site-packages/gym/wrappers/record_video.py:41: UserWarning: WARN: Overwriting existing videos at /home/YYYYY/videos folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)
  logger.warn(
MP4 has been saved in  ./videos
test.py (line 76): test(): episode 6
Time 00h 06m 46s, total num steps 1282977, FPS 3159, episode reward 21.0, episode length 2836
test.py (line 76): test(): episode 7
Time 00h 07m 52s, total num steps 1489764, FPS 3154, episode reward 21.0, episode length 2836
test.py (line 76): test(): episode 8
Time 00h 08m 59s, total num steps 1699905, FPS 3150, episode reward 21.0, episode length 2836
test.py (line 76): test(): episode 9
Time 00h 10m 04s, total num steps 1901878, FPS 3147, episode reward 21.0, episode length 2836
test.py (line 76): test(): episode 10
Time 00h 11m 11s, total num steps 2110088, FPS 3142, episode reward 21.0, episode length 2836
recording
/home/XXXXX/venv/lib/python3.10/site-packages/gym/wrappers/record_video.py:41: UserWarning: WARN: Overwriting existing videos at /home/YYYYY/videos folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)
  logger.warn(
MP4 has been saved in  ./videos
```

## エントロピー正則化を加えた時の挙動

- 改善前：勝利まで **105万ステップ（5分33秒）**  
- 改善後：勝利まで **74万ステップ（4分17秒）（約30%短縮）**  
- 報酬推移：-21 → -19 → -18 → +21  
  → **スムーズに改善**  
- 勝利時エピソード長も短縮  
  → 2836 → **2535**  

```
(venv) $ python3 main.py --env-name PongDeterministic-v4 --num-processes 8 --num-steps 20 --entropy-coef 0.01
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
env.observation_space.shape: (1, 42, 42)
env action meanings: ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
num of processes: 8
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
A.L.E: Arcade Learning Environment (version 0.7.4+069f8bd)
[Powered by Stella]
Time 00h 00m 00s, total num steps 1700, FPS 3013, episode reward -21.0, episode length 765
recording
/home/XXXXX/venv/lib/python3.10/site-packages/gym/wrappers/record_video.py:41: UserWarning: WARN: Overwriting existing videos at /home/YYYYY/videos folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)
  logger.warn(
MP4 has been saved in  ./videos
test.py (line 76): test(): episode 1
Time 00h 01m 02s, total num steps 183238, FPS 2912, episode reward -21.0, episode length 765
test.py (line 76): test(): episode 2
Time 00h 02m 05s, total num steps 364617, FPS 2902, episode reward -19.0, episode length 2175
test.py (line 76): test(): episode 3
Time 00h 03m 12s, total num steps 555975, FPS 2889, episode reward -18.0, episode length 3530
test.py (line 76): test(): episode 4
Time 00h 04m 17s, total num steps 742133, FPS 2881, episode reward 21.0, episode length 2535
test.py (line 76): test(): episode 5
Time 00h 05m 22s, total num steps 928012, FPS 2875, episode reward 21.0, episode length 2535
recording
/home/XXXXX/venv/lib/python3.10/site-packages/gym/wrappers/record_video.py:41: UserWarning: WARN: Overwriting existing videos at /home/YYYYY/videos folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)
  logger.warn(
MP4 has been saved in  ./videos
test.py (line 76): test(): episode 6
Time 00h 06m 29s, total num steps 1115719, FPS 2862, episode reward 21.0, episode length 2535
test.py (line 76): test(): episode 7
Time 00h 07m 35s, total num steps 1299729, FPS 2856, episode reward 21.0, episode length 2535
test.py (line 76): test(): episode 8
Time 00h 08m 38s, total num steps 1479262, FPS 2850, episode reward 21.0, episode length 2535
test.py (line 76): test(): episode 9
Time 00h 09m 41s, total num steps 1654833, FPS 2846, episode reward 21.0, episode length 2535
test.py (line 76): test(): episode 10
Time 00h 10m 44s, total num steps 1832063, FPS 2841, episode reward 21.0, episode length 2535
recording
/home/XXXXX/venv/lib/python3.10/site-packages/gym/wrappers/record_video.py:41: UserWarning: WARN: Overwriting existing videos at /home/YYYYY/videos folder (try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)
  logger.warn(
MP4 has been saved in  ./videos
```

## 考察
- エントロピー正則化により  
  - 初期探索が促進され、学習立ち上がりが速くなった  
  - 方策が多様化し、局所解からの脱出が容易に  
- 学習後は係数を小さくすると収束性がさらに改善  


## まとめ
- A3CでPongの自動プレイを実現  
- **エントロピー正則化**により学習効率・安定性が向上  
- 強化学習は「探索と活用のバランス」が鍵  
- 今後：Breakoutなど他Atariゲームへの展開


## 参考文献：
- [1] Asynchronous Methods for Deep Reinforcement Learning, 2016, https://arxiv.org/abs/1602.01783
- [2] Function Optimization using Connectionist Reinforcement Learning Algorithms, 1991, https://researchwith.montclair.edu/en/publications/function-optimization-using-connectionist-reinforcement-learning-
- [3] Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning, 1992, https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
