"""dump_lr_mmengine.py — 在 p2r-torch 环境 dump 官方 param_scheduler 的逐 iter LR 序列（C2 golden）。

官方 schedule_1x.py（v2 三个 config 均未覆盖）：
    LinearLR(start_factor=1/3, by_epoch=False, begin=0, end=500)
    MultiStepLR(milestones=[8,11], gamma=0.1, by_epoch=True)
按 mmengine ParamSchedulerHook 的真实调度方式驱动真实的 mmengine scheduler 类：
by_epoch=False 的每 iter step 一次，by_epoch=True 的每 epoch step 一次；
记录每个 iter 更新时 optimizer 实际生效的 lr。

用法：
    conda activate p2r-torch
    python tools/dump_lr_mmengine.py  # 产出 tests/parity/golden/lr_sequence.npz
"""
import os

import numpy as np
import torch
from mmengine.optim.scheduler import LinearLR, MultiStepLR

BASE_LR = 5e-5          # 官方 AdamW lr
ITERS_PER_EPOCH = 120   # 测试用 epoch 长度：warmup 500 iter 横跨 epoch 0–4，能压出交叠段
EPOCHS = 12

OUT = os.path.join(os.path.dirname(__file__), '..', 'tests', 'parity', 'golden',
                   'lr_sequence.npz')


def main():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=BASE_LR, betas=(0.9, 0.999), weight_decay=0.05)
    s_iter = LinearLR(opt, start_factor=1.0 / 3, by_epoch=False, begin=0, end=500)
    s_epoch = MultiStepLR(opt, milestones=[8, 11], gamma=0.1, by_epoch=True)

    lrs = []
    for _ in range(EPOCHS):
        for _ in range(ITERS_PER_EPOCH):
            lrs.append(opt.param_groups[0]['lr'])  # 本 iter 更新使用的 lr
            s_iter.step()      # ParamSchedulerHook.after_train_iter
        s_epoch.step()         # ParamSchedulerHook.after_train_epoch

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, lr=np.array(lrs, dtype=np.float64),
             iters_per_epoch=ITERS_PER_EPOCH, epochs=EPOCHS, base_lr=BASE_LR,
             start_factor=1.0 / 3, warmup_iters=500, milestones=[8, 11], gamma=0.1)
    print(f'saved {len(lrs)} lrs -> {OUT}')
    print('head:', lrs[:3], '... warmup end:', lrs[499:502],
          '... epoch8 head:', lrs[8 * ITERS_PER_EPOCH - 1:8 * ITERS_PER_EPOCH + 2])


if __name__ == '__main__':
    main()
