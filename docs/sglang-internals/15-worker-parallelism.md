# Worker 与并行：TP / PP / DP / EP 与分布式

## 1. 一句话职责

本章讲的是「一个 SGLang 实例如何把一个大模型铺到多张 GPU、多台机器上跑前向」：
`TpModelWorker` 是 scheduler 进程里真正持有模型、执行 forward 的那一层；
四种并行（TP / PP / DP / EP）各自切分模型或请求的不同维度；
`distributed/` 目录负责建立 NCCL / gloo 的 process group，把切开的张量在合适的时机用 `all-reduce` / `all-gather` / `reduce-scatter` 重新拼回去；
`DataParallelController` 则在「多个完整副本」之间做请求级负载均衡。

一句话区分四者：

- **TP（tensor parallel）**：把每一层的权重矩阵切开，多卡协作算 *同一层*。
- **PP（pipeline parallel）**：把模型的 *层* 切成若干段，每段放一组卡，按流水线接力。
- **DP（data parallel）**：跑 *多个完整副本*，不同副本处理不同请求（SGLang 里还有特殊的 **DP attention**）。
- **EP（expert parallel）**：MoE 模型里把 *专家（experts）* 分到不同卡，每张卡只持有一部分专家。

## 2. 在端到端链路中的位置

```
客户端 → 分词(TokenizerManager) → [DataParallelController] → 调度(Scheduler) → KV缓存
                                          │                        │
                                          │ 多副本时才有             ▼
                                          │                  ┌─────────────────┐
                                          └─────────────────►│  TpModelWorker  │ ← 本章
                                                             │  + ModelRunner  │
                                                             └────────┬────────┘
                                                                      ▼
                                                              前向(forward) → 采样 → 解码 → 响应
```

注意两个位置：

1. `DataParallelController` 介于 tokenizer 和 scheduler 之间，**只有开启 `--dp-size > 1` 时才存在**（见 `python/sglang/srt/managers/data_parallel_controller.py:283`）。
2. `TpModelWorker` 在每个 scheduler 进程内部，是 scheduler 调用 `forward_batch_generation` 的承接者（[调度器](12-scheduler.md) 把 `ModelWorkerBatch` 交给它，它再委托给 `ModelRunner`，见 [模型执行器](14-model-executor.md)）。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
|------|-----------|------|
| `TpModelWorker` | `python/sglang/srt/managers/tp_worker.py:47` | scheduler 进程里持有 `ModelRunner`、tokenizer，执行 forward / sample / 权重更新 |
| `TpModelWorker.forward_batch_generation` | `python/sglang/srt/managers/tp_worker.py:183` | 把 `ModelWorkerBatch` 变成 `ForwardBatch`，处理 PP 收发，跑 forward + sample |
| `TpModelWorkerClient` | `python/sglang/srt/managers/tp_worker_overlap_thread.py:51` | overlap scheduler 下对 `TpModelWorker` 的包装，用后台线程异步跑 forward |
| `TpModelWorkerClient.forward_thread_func_` | `python/sglang/srt/managers/tp_worker_overlap_thread.py:125` | 后台 forward 线程主循环，配合 future token id 实现 CPU/GPU 重叠 |
| `DataParallelController` | `python/sglang/srt/managers/data_parallel_controller.py:57` | 在多个 DP 副本间分发请求 |
| `DataParallelController.round_robin_scheduler` | `python/sglang/srt/managers/data_parallel_controller.py:249` | 默认的轮询负载均衡 |
| `GroupCoordinator` | `python/sglang/srt/distributed/parallel_state.py:165` | 对 PyTorch ProcessGroup 的封装，提供 all-reduce/all-gather/PP 收发 |
| `init_distributed_environment` | `python/sglang/srt/distributed/parallel_state.py:1046` | 初始化全局 `_WORLD` group（`torch.distributed.init_process_group`） |
| `initialize_model_parallel` | `python/sglang/srt/distributed/parallel_state.py:1101` | 构建 `_TP` 和 `_PP` 两个 group |
| `tensor_parallel` | `python/sglang/srt/model_parallel.py:121` | 基于 `_tp_plan` 用 DTensor 对 module 做列/行切分 |
| `initialize_dp_attention` | `python/sglang/srt/layers/dp_attention.py:61` | 建立 attention 专用 TP 子 group `_ATTN_TP_GROUP` |
| `compute_dp_attention_world_info` | `python/sglang/srt/layers/dp_attention.py:33` | 由 `tp_rank` 推出 `attn_tp_rank / attn_tp_size / attn_dp_rank` |
| `_dp_gather` / `dp_scatter` | `python/sglang/srt/layers/dp_attention.py:224` / `:274` | DP attention 在 attn 与 MLP 之间 gather/scatter token |
| `LayerCommunicator` 相关 | `python/sglang/srt/layers/communicator.py:38` 起 | 描述每层在 attn / MLP 间的 scatter 模式与通信选择 |
| `ModelRunner.init_torch_distributed` | `python/sglang/srt/model_executor/model_runner.py:414` | 真正把上面这些初始化串起来的入口 |

## 4. 核心数据流 / 执行流程

### 4.1 进程模型：一个 scheduler 进程 = 一个 rank

SGLang 的关键设计：**TP/PP 的每个 rank 是一个独立的 OS 进程**，每个进程跑一份 `run_scheduler_process`，进程内有一个 `Scheduler`，`Scheduler` 内有一个 `TpModelWorker`，`TpModelWorker` 内有一个 `ModelRunner`，`ModelRunner` 内有切好的模型分片。

进程是怎么拉起来的？在 `DataParallelController.launch_tensor_parallel_group` 里（`python/sglang/srt/managers/data_parallel_controller.py:170`）用 `mp.Process(target=run_scheduler_process, ...)` 为每个 `(pp_rank, tp_rank)` 组合各起一个进程（`:224`），并把 `gpu_id / tp_rank / pp_rank / dp_rank` 作为参数传进去。即使不开 DP，这条路径也会被走（`launch_dp_schedulers` 默认 `dp_size=1`）。

rank 编号规则（`model_runner.py:451`）：

```
global_rank = tp_size * pp_rank + tp_rank
world_size  = tp_size * pp_size
```

### 4.2 process group 的建立（distributed/）

每个 rank 进程启动后都会走 `ModelRunner.init_torch_distributed`（`model_runner.py:414`），三步建组：

```
init_distributed_environment(world_size=tp*pp, rank=tp*pp_rank+tp_rank, ...)   # → _WORLD
        │   torch.distributed.init_process_group(backend=nccl, ...)
        ▼
initialize_model_parallel(tp_size, pp_size)                                     # → _TP, _PP
        │   _TP : 把相邻 tp_size 个 rank 分成一组   [g0,g1],[g2,g3],...
        │   _PP : 跨步取 rank 组成 pipeline        [g0,g2,...],[g1,g3,...]
        ▼
initialize_dp_attention(...)                                                    # → _ATTN_TP_GROUP
```

`GroupCoordinator.__init__`（`parallel_state.py:201`）对每个 group 同时建 **两个** ProcessGroup：一个用 `nccl`（`device_group`，GPU 上通信）、一个用 `gloo`（`cpu_group`，用于 CPU 上的协调与 `broadcast_pyobj`，见 `parallel_state.py:229`）。它还会按需创建 `PyNcclCommunicator` 和 `CustomAllreduce`（`:260`、`:267`）——后者是 SGLang/vLLM 自带的小张量快速 all-reduce 实现。

`_TP` group 的 rank 划分（`parallel_state.py:1145`）：

```
TP groups (tp=2, pp=1, world=2):   [g0, g1]
PP groups (tp=2, pp=2, world=4):   TP=[g0,g1],[g2,g3]   PP=[g0,g2],[g1,g3]
```

### 4.3 TP：切权重，靠 all-reduce 拼回

TP 的切分由两条路径完成：

- 自研模型走各层自己的并行 linear（如 `ColumnParallelLinear` / `RowParallelLinear`，不在本章重点）。
- HuggingFace 风格模型走 `model_parallel.py:121` 的 `tensor_parallel()`：它读 module 的 `_tp_plan`，对标 `"Colwise"` / `"Rowwise"` / `"Colwise_Sharded"` 三种切法，用 `DTensor` 把权重切片（`:138` 的 `tplize`）。

经典 Transformer 一层里 TP 的通信点：

```
        x (每卡都有完整 x)
          │  ColumnParallel  (按输出维切, 无通信)
          ▼
   QKV / gate&up  分片
          │  attention / SwiGLU  (各卡算自己那份 head / 中间维)
          ▼
   o_proj / down_proj  RowParallel (按输入维切)
          │
          ▼  all-reduce   ← 把各卡的部分和加起来
        x' (每卡又都有完整 x')
```

即：**Column 切完不通信，Row 切完做一次 all-reduce**。一层 attention + MLP 通常有 2 次 all-reduce。`tensor_model_parallel_all_reduce` 最终落到 `GroupCoordinator.all_reduce`（`parallel_state.py:397`），它会按张量大小在 `CustomAllreduce`（小张量）/ pynccl / `torch.distributed`（`:455`）之间择优。

### 4.4 PP：切层，靠点对点收发接力

PP 把模型层切成 `pp_size` 段。`TpModelWorker.forward_batch_generation`（`tp_worker.py:183`）里能清楚看到流水线逻辑：

```python
if not self.pp_group.is_first_rank:        # tp_worker.py:194
    pp_proxy_tensors = PPProxyTensors(self.pp_group.recv_tensor_dict(...))   # 收上一段的 hidden_states

if self.pp_group.is_last_rank:             # tp_worker.py:201
    logits_output, ... = self.model_runner.forward(forward_batch, pp_proxy_tensors=...)
    next_token_ids = self.model_runner.sample(...)   # 只有最后一段做采样
    return logits_output, next_token_ids, ...
else:
    pp_proxy_tensors, ... = self.model_runner.forward(...)
    return pp_proxy_tensors.tensors, None, ...        # 中间段把 hidden_states 传给下一段
```

- 非第一段：先 `recv_tensor_dict`（`parallel_state.py:830`）收上游的 `hidden_states`（即 `PPProxyTensors`）。
- 只有最后一段（`is_last_rank`，`parallel_state.py:331`）才有 `logits` 和 `sample`。
- PP 用的是点对点 send/recv，不是集合通信。注意 `recv_tensor_dict` 传了 `all_gather_group=self.get_attention_tp_group()`——在跨 TP 的 PP 收发里要配合 attention TP group 做 all-gather。

### 4.5 EP：切专家，复用 TP group

EP 在 SGLang 里**没有单独的 process group**——它复用 `_TP` group。看 `python/sglang/srt/layers/moe/ep_moe/layer.py:166`：

```python
self.tp_rank = get_tensor_model_parallel_rank()
...
self.start_expert_id = self.tp_rank * self.num_experts_per_partition   # :172
```

即每个 TP rank 用自己的 `tp_rank` 算出「我负责哪几个专家」。开关是 `--enable-ep-moe`（透传到 `global_server_args_dict["enable_ep_moe"]`，`model_runner.py:201`，使用处 `ep_moe/layer.py:1204`）。
EP 的通信是 MoE 特有的 **all-to-all / dispatch-combine**（每个 token 被路由到它选中的专家所在的卡，算完再收回来），区别于 TP 的 all-reduce。EP 和 TP 复用同一组卡、同一个 NCCL group，只是把「按矩阵切」换成「按专家切」。

### 4.6 DP attention：解决「KV cache 被 TP 复读」的浪费

普通 TP 下，**attention 也按 head 切**，每张卡只存 1/tp_size 的 KV head——这本来没问题。但对 GQA/MLA（KV head 很少，甚至像 DeepSeek MLA 只有 1 个 latent KV）的模型，head 数 < tp_size 时无法整除切分，只能在多卡上 **复制** KV cache，造成显存浪费。

**DP attention** 的思路：attention 部分不做 TP，而是按 **DP** 切——把 tp group 再划成 `dp_size` 个「attention DP 副本」，每个副本独立持有自己那批请求的 KV cache，attention 内部只在更小的 `attn_tp_size = tp_size // dp_size` 范围里做 TP；而 MLP / MoE 部分仍按全 TP（或 EP）跑。

切分公式在 `compute_dp_attention_world_info`（`dp_attention.py:33`）：

```python
attn_tp_size = tp_size // dp_size
attn_dp_rank = tp_rank // attn_tp_size     # 我属于哪个 attention DP 副本
attn_tp_rank = tp_rank % attn_tp_size      # 我在副本内的 TP rank
```

`initialize_dp_attention`（`dp_attention.py:61`）据此建一个专用子 group `_ATTN_TP_GROUP`（`:94`，每 `attn_tp_size` 个 rank 一组）。

一层内的数据流在 attention 与 MLP 边界处需要 gather/scatter（`dp_attention.py:224`、`:274`，由 `communicator.py` 的 `LayerCommunicator` 编排）：

```
       各 DP 副本各自的 local tokens
   attention (DP: 每副本独立, KV cache 不再复制)
              │
              ▼  dp_gather  (把所有副本的 token 拼成 global)  ← all-reduce / all-gather
   MLP / MoE (按全 TP / EP, 需要看到全部 token)
              │
              ▼  dp_scatter (再切回各副本的 local 段)
       各 DP 副本各自的 local tokens
```

`_dp_gather`（`:224`）对非浮点（如 input_ids）且 tp 在单机 8 卡内时用 `inplace_all_reduce`，否则走 `tensor_model_parallel_all_reduce`（`:250`、`:255`）。开关是 `--enable-dp-attention`，配合 `--dp-size`。

> 关键区分：普通 `--dp-size`（4.7 节）= 多个**完全独立的模型副本**，由 `DataParallelController` 分发；而 `--enable-dp-attention` 复用同一套 TP group，只在 attention 层做 DP，权重不复制。两者的 `dp_size` 含义不同，初始化路径也不同（见 `data_parallel_controller.py:88` 的 `if enable_dp_attention` 分支）。

### 4.7 DataParallelController：在副本间分发请求

不开 DP attention 时（普通 DP），每个 DP 副本是一套完整的 TP/PP group。`DataParallelController` 的工作流：

```
TokenizerManager
   │  zmq PUSH (scheduler_input_ipc_name)
   ▼
recv_from_tokenizer (zmq.PULL)            data_parallel_controller.py:72
   │  event_loop: recv_pyobj              :261
   ▼
dispatching(req)  = round_robin_scheduler :249
   │  self.workers[counter].send_pyobj(req); counter = (counter+1) % dp_size
   ▼
workers[dp_rank] (zmq.PUSH → 各副本 scheduler)   :98
```

要点：

- 两种分发策略：`ROUND_ROBIN`（默认，已实现）与 `SHORTEST_QUEUE`（`shortest_queue_scheduler` 仍是 `NotImplementedError`，`:258`）。
- 控制类消息（非 generate/embedding 请求）不走负载均衡，而是广播给每个 TP group 的首个 worker（`event_loop` 的 else 分支，`:279`，步长 `control_message_step`）。
- disaggregation 模式下用 `bootstrap_room % len(workers)` 而非轮询（`:256`），保证同一请求落到固定副本。
- 只有 `node_rank == 0` 的节点跑真正的分发循环（`:301`）；其他节点只负责拉起本地的 scheduler 进程。

### 4.8 overlap scheduler 下的 worker（TpModelWorkerClient）

开 overlap（默认开，`--disable-overlap-schedule` 关）时，scheduler 用的是 `TpModelWorkerClient`（`scheduler.py:272`）。它在后台开一个 `forward_thread`（`tp_worker_overlap_thread.py:82`），主线程把 batch 塞进 `input_queue`、立刻拿到一批 **future token ids**（负数占位，`:227`）就返回去做下一批的调度；后台线程真正跑 forward，把结果写回 `future_token_ids_map`（`:158`）。`resolve_future_token_ids`（`:43`）在下一批 forward 前用 `torch.where` 把负数占位换成真实 token。这就是 CPU 调度与 GPU 计算重叠的核心机制（详见 [调度器](12-scheduler.md) 与 [模型执行器](14-model-executor.md)）。

## 5. 设计难点与权衡

1. **「一 rank 一进程」而不是一进程多卡**。SGLang 选择多进程（每卡一个 scheduler 进程），好处是没有 Python GIL 争用、各卡调度互不阻塞；代价是需要靠 NCCL/gloo 在进程间同步（如 `broadcast_pyobj` 同步随机种子，`tp_worker.py:139`）。这也是为什么很多状态需要在 TP group 内显式同步。

2. **EP 复用 TP group 是有意为之**。不额外建 EP group 简化了拓扑，但意味着 `ep_size == tp_size`，专家数必须能被 TP rank 数整除（`ep_moe/layer.py:172` 的 `start_expert_id` 计算），灵活性受限。

3. **DP attention 是「省 KV cache 显存」与「多一次 gather/scatter 通信」的权衡**。对 MLA/GQA 类模型收益大（KV head 少，TP 切不动），对普通 MHA 模型反而可能因额外通信变慢。所以它是个独立开关而非默认。

4. **all-reduce 的多后端择优**。`GroupCoordinator.all_reduce`（`parallel_state.py:397`）里有一长串分支：CPU 走 ipex、小张量走 `CustomAllreduce`、否则 pynccl 或 torch。`reduce_scatter`（`:462`）目前只支持 torch 后端（注释 `TODO(ch-wan): support other backends`），是已知的扩展点。

5. **future token ids 的负数编码很微妙**。`forward_batch_generation` 返回的 `future_next_token_ids` 是 `-(ct+1)` 这样的负数（`tp_worker_overlap_thread.py:227`），后续 batch 把它们当 input_ids 用，靠 `resolve_future_token_ids` 在 GPU 上替换。负数与 `future_token_ids_map` 的环形复用（`% future_token_ids_limit`，`:234`）是 overlap 正确性的关键，改动这里极易出 off-by-one。

6. **保持 `model_worker_batch` 引用防止 CUDA 非法访问**。`forward_thread_func_` 里特意把 batch 存进 `batch_lists`（`:139`），注释说明否则 tensor 会被 PyTorch 提前释放导致 illegal memory access——这是异步 forward 的隐藏坑。

## 6. 改代码 / 修 bug 指南

「想改 X → 看这里」：

- **加一种新的 DP 负载均衡策略** → `data_parallel_controller.py:42` 的 `LoadBalanceMethod` 枚举 + `dispatch_lookup`（`:78`）+ 实现一个 `xxx_scheduler` 方法（参考 `round_robin_scheduler:249`）。`shortest_queue_scheduler` 是现成的空壳。
- **改 TP 切分规则 / 支持新模型的 TP** → 模型文件里给 module 加 `_tp_plan`，或看 `model_parallel.py:138` 的 `tplize` 三种 style。自研并行 linear 在 `layers/linear.py`（本章未展开）。
- **改 PP 收发的张量内容** → `tp_worker.py:194`（recv）与 `:217`（中间段 return），以及 `parallel_state.py:777`(`send_tensor_dict`)/`:830`(`recv_tensor_dict`)。
- **改 EP 专家划分** → `layers/moe/ep_moe/layer.py:166`-`172`（`start_expert_id`），注意它依赖 `get_tensor_model_parallel_rank`。
- **改 DP attention 的 gather/scatter** → `dp_attention.py:224`(`_dp_gather`)、`:274`(`dp_scatter`)、`:295`(`attn_tp_reduce_scatter`)；编排逻辑在 `communicator.py`。
- **改 process group 拓扑 / rank 编号** → `parallel_state.py:1101`(`initialize_model_parallel`) 与 `model_runner.py:448`-`459` 的 rank 计算。

常见 bug 高发区：

- **NCCL 初始化卡死 / 端口冲突**：`dist_init_method`（`model_runner.py:440`）和 `nccl_port`。多 DP 副本时端口由 `launch_dp_schedulers` 用 `bind_port` 预占（`data_parallel_controller.py:122`）；DP attention 下所有 dp rank 共用同一个 nccl_port（`:215`，有注释说明 DP 复用 TP group）。
- **某个 rank 静默退出但整体 hang**：任一 scheduler 进程崩了会向父进程发 `SIGQUIT`（`tp_worker_overlap_thread.py:122`、`data_parallel_controller.py:311`）。查日志里第一个抛异常的 rank。
- **CUDA illegal memory access**：八成在 overlap 路径，检查 batch 引用保持（`:139`）和 stream 同步事件（`sync_event`，`:219`）。
- **TP 间随机性不一致 / 采样发散**：随机种子靠 `broadcast_pyobj` 在 world group 内同步（`tp_worker.py:139`），TP rank 间结果不一致先查这里和 `SYNC_TOKEN_IDS_ACROSS_TP`（`dp_attention.py:101`）。
- **显存不均衡报错**：`init_torch_distributed` 里有 `min_per_gpu_memory < local * 0.9` 的检查（`model_runner.py:481`），被别的进程占了卡会触发，可用 `SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK` 临时跳过。

调试入手点：

- 打开 `--node-rank` / `--tp-size` / `--pp-size` / `--dp-size` / `--enable-dp-attention` / `--enable-ep-moe` 这几个 server arg，对照 `data_parallel_controller.py:170` 的进程拉起逻辑确认到底起了几个进程、各自的 `(gpu_id, tp_rank, pp_rank, dp_rank)`。
- 在 `GroupCoordinator.__init__`（`parallel_state.py:201`）加日志打印每个 group 的 `ranks`，能直观看到 TP/PP/ATTN_TP group 的实际划分。
- 用 `setproctitle` 设置的进程名（`sglang::data_parallel_controller` 等，`data_parallel_controller.py:288`）配合 `ps` / `py-spy` 定位是哪个角色的进程卡住。

## 7. 一台多卡机上的进程/通信拓扑图

下图：单机 8 卡，`tp_size=4, pp_size=2, dp_size=1`（即 world_size = tp*pp = 8）。

```
  Host (1 node, 8 GPUs)                 进程数 = tp_size * pp_size = 8

  PP stage 0 (前半层)                    PP stage 1 (后半层)
  ┌───────────────────────────┐         ┌───────────────────────────┐
  │ rank0  rank1  rank2  rank3 │         │ rank4  rank5  rank6  rank7 │
  │ GPU0   GPU1   GPU2   GPU3  │         │ GPU4   GPU5   GPU6   GPU7  │
  │  └──────┴──all-reduce┴───┘ │         │  └──────┴──all-reduce┴───┘ │
  │      TP group [0,1,2,3]    │         │      TP group [4,5,6,7]    │
  └─────────┬─────────────────┘         └─────────┬─────────────────┘
            │  PP send/recv (hidden_states, 点对点)
            │  PP groups: [0,4] [1,5] [2,6] [3,7]
            └───────────────────────►─────────────┘

  每个 rankN = 1 个 OS 进程:
     run_scheduler_process → Scheduler → TpModelWorker(Client) → ModelRunner → 模型分片

  通信原语:
     TP  : all-reduce (RowParallel 之后) / all-gather  —— _TP group, NCCL+CustomAllreduce
     PP  : send/recv tensor_dict                       —— _PP group, NCCL p2p
     EP  : all-to-all (dispatch/combine)               —— 复用 _TP group
     DP-attn: dp_gather/dp_scatter (all-reduce/all-gather) —— _ATTN_TP_GROUP

  若开 dp_size>1 (普通 DP, 非 dp-attention):
     上面整块 ×N 份完整副本, 由 DataParallelController 轮询分发请求:
        TokenizerManager → DPController(node_rank0) → [副本0 sched, 副本1 sched, ...]
```

跨节点时（`nnodes > 1`），`launch_tensor_parallel_group`（`data_parallel_controller.py:186`-`197`）会按 `nnodes_per_tp_group` / `tp_size_per_node` / `pp_size_per_node` 把上面的格子拆到不同 node，每个 node 只在本地起属于自己的那部分 rank 进程。

---

相关章节：[调度器](12-scheduler.md)、[模型执行器](14-model-executor.md)、[KV 缓存与 RadixAttention](13-memcache-radixattention.md)、[术语表](02-glossary.md)。
