import torch.multiprocessing as mp
import time

def thread(th_num, params, counter):
    # 演習(2) 以下のコメントを外して実行してみてください
    #for i in range(3):
    #    with counter.get_lock():  # 排他ロックしてカウンタ更新
    #        counter.value += 1
    #        print(f"Thread No.: {th_num}, Count: {counter.value}")
    #    time.sleep(th_num)  # th_num秒待機
    print(f"Thread No.: {th_num} finished.")

processes = []

counter = mp.Value('i', 0)
lock = mp.Lock()

for i in range(3):
    p = mp.Process(target=thread, args=(i, i, counter))
    p.start()
    processes.append(p)

# 演習(3) 以下のコメントを外して実行してみてください
#for p in processes:
#    p.join()

print('main finished')
