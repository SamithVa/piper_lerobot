[English](README.md) | **中文**

# deploy — 在 Piper 机械臂上运行任意策略

**服务端（server）** 在策略所需的任意环境中托管模型；**客户端（client）**
驱动机械臂和相机，通过 localhost HTTP 流式执行动作块（action chunk），
并采用异步重叠（async overlap），推理期间机械臂不会停顿。任何模型都可以接入 —
openpi、GR00T、lerobot 等等：只需写一个小小的 adapter，其余代码完全不用改。

## 三步部署你自己的策略

先零成本跑通整条流水线（不需要 GPU、checkpoint 或机械臂）：

```bash
python -m deploy.launch example        # dummy 策略服务端
curl -s http://127.0.0.1:8090/info
```

1. **Adapter** — 把 `deploy/adapters/dummy.py` 复制为
   `deploy/adapters/<name>.py`，实现它的三个方法：`info()`、
   `predict_chunk(...)`、`reset()`。完整契约（张量形状、dtype、参数）见
   `dummy.py` 的模块 docstring 和 `adapters/base.py`。然后在
   `deploy/adapters/__init__.py::make_adapter` 里注册。
2. **Preset** — 把 `deploy/presets/example.json` 复制为
   `presets/<name>.json`；把 `server.python` 指向你模型所需的环境
   （服务端只依赖 stdlib+numpy，任何环境都能跑），再加上 `client` 配置段
   （照抄 `pi05.json` 的 cameras / camera_map）。
3. **运行** — `python3 -m deploy.launch <name> --task="..."`。

## 运行已有策略

```bash
python3 -m deploy.launch pi05 --task="Stack the cup on top of the bowl." --duration_s=60
# 等价写法：
bash deploy/run.sh pi05 "Stack the cup on top of the bowl." 60
```

| preset    | 策略                                              | 说明                          |
|-----------|---------------------------------------------------|-------------------------------|
| `pi05`    | `outputs/pi05`（samithva/pi05_stack_cup_bowl）    | 环境版本锁定 — 见下方说明     |
| `smolvla` | smolvla_bimanual_stack_cup_bowl（最新 checkpoint）| lerobot 环境                  |
| `example` | dummy adapter — 无 GPU、无 checkpoint、无机械臂   | 仅服务端的教学示例            |

一个 preset 收纳了你本来必须了解的一切：服务环境、checkpoint、端口、相机配置、
相机→图像键映射。启动器对 preset 端口的处理：**复用（reuse）** 已在服务同一
checkpoint 的热服务端（跳过 pi05 约 15-20 秒的编译冷启动）；**拒绝（refuse）**
被其他策略占用的端口（绝不杀掉别人的进程）；否则 **拉起（spawn）** 一个后台
服务端（日志：`deploy/logs/server-<preset>.log`）。Ctrl-C 只停客户端 —
服务端保持热状态，供下次运行复用。可覆盖参数：`--checkpoint=`、`--port=`、
`--fps=`，以及任意客户端参数（`--key=value` 形式）。

## 工作原理

```
launch.py ──拉起/复用──▶ server.py（策略环境）◀── adapters/<name>.py（你的代码）
    │                        ▲   /info /predict /reset
    └──运行──▶ client.py ────┘   npz 观测 → npy 动作块
              （base python：机械臂 + 相机）
```

- **协议**（`protocol.py`）：`GET /info` → `{name, image_keys, state_dim,
  action_dim, chunk_size, fps, checkpoint}`；`POST /predict` — 请求体为 npz
  图像（`img_<key>`，HWC uint8 RGB）+ `state`（float32）+ `task`，返回 `.npy`
  字节，形状 `(chunk_size, action_dim)`；`POST /reset` 清空回合状态。
  出错 → HTTP 500，正文为 traceback。
- **异步重叠**（`chunking.py`）：当前动作块消耗过半时就发出下一次
  `/predict`；请求在途期间已执行的行会从新块中跳过；若队列耗尽，客户端
  保持当前位置并重新请求。

## 本机注意事项（pi05）

- 服务环境必须与训练环境 **完全一致**（此处：lerobot v0.5.1 + transformers
  5.3.0 — 更新版本的 transformers 能跑，但会静默忽略输入）。完整来龙去脉
  （含 HF 离线配置）见 `deploy/presets/pi05.json` 的 `_notes`。
- 服务端拉起后的第一次 `/predict` 需要约 15-20 秒（`torch.compile` +
  CUDA-graph 捕获）；客户端用 `first_predict_timeout_s`（90 秒）吸收它。
  **不要** 在服务端加预热 predict — 那会破坏 pi05 的 CUDA-graph 状态。
- 双环境划分：客户端始终跑在 base python（pyAgxArm + 相机）；每个策略在
  自己的环境里提供服务。

## 测试

```bash
PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python -m pytest deploy/tests -v
# 集成测试（会下载 smolvla_base，需要 GPU）：
DEPLOY_IT=1 PYTHONPATH=src:. /home/embodied/miniconda3/envs/lerobot/bin/python \
    -m pytest deploy/tests/test_lerobot_adapter.py -v
```
