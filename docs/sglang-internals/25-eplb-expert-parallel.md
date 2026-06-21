# EPLB：专家并行负载均衡

> 适用于：想读懂 SGLang 在大规模 Expert Parallel(EP) 部署下如何监控并重平衡 MoE 专家负载、并能改代码 / 修 bug 的工程师。

---

## 1. 一句话职责

EPLB(Expert-Parallel Load Balancer) 解决的是 **MoE 模型在大规模 EP 部署下，token 路由到各 expert 的负载严重不均** 的问题：它在运行时统计每个 logical expert 的命中次数(`expert_distribution`)，用 DeepSeek 的重平衡算法算出一份更优的 **物理专家放置方案**(`expert_location`，含冗余副本)，再在不重启服务的前提下，通过 GPU 间 P2P 通信原地搬运专家权重(`expert_location_updater`)，让各张卡承担的 token 数尽量均衡。

一句话拆成三块：**统计(record) → 重算放置(rebalance) → 在线更新权重(update)**。

---

## 2. 在端到端链路中的位置

EPLB 不是请求链路上的"必经站"，而是挂在 forward 之后、跨多个 forward pass 周期性触发的**旁路控制环路**。它读 MoE forward 产生的统计，写回供下一次 MoE forward 使用的 `expert_location_metadata`。

```
客户端 → 分词 → 调度 → KV缓存 → 前向(forward) ─┬─→ 采样 → 解码 → 响应
                                              │
                          ┌───────────────────┘
                          │ MoE topk 阶段
                          ▼
              [logical → physical 映射]          ← expert_location_dispatch
              topk_ids_logical_to_physical()
                          │
                          ▼
              on_select_experts / on_deepep_dispatch
              (统计每层每个 expert 命中次数)        ← expert_distribution(Recorder)
                          │
   每个 forward pass 结束  ▼
   ┌──────────────────────────────────────────────┐
   │ EPLBManager.on_forward_pass_end(pass_id)      │
   │   if pass_id % N == 0: rebalance()            │  ← 旁路控制环路
   │     1. dump_record() → logical_count          │
   │     2. ExpertLocationMetadata.init_by_eplb()  │  ← deepseek_eplb 算法
   │     3. model_runner.update_expert_location()  │  ← expert_location_updater(P2P 搬权重)
   └──────────────────────────────────────────────┘
                          │
                          ▼
            更新 _global_expert_location_metadata
            (下一次 MoE forward 的 dispatch 立刻生效)
```

forward 入口处的挂钩见 `python/sglang/srt/model_executor/model_runner.py:1167`(用 `with_forward_pass` 包住整个 forward 来界定一个 "single pass")与 `python/sglang/srt/model_executor/model_runner.py:1175`(forward 结束触发 `eplb_manager.on_forward_pass_end`)。

与 forward / MoE 层本身的关系见 [模型执行器](14-model-executor.md)；分布式与 EP 概念见 [Worker 与并行](15-worker-parallelism.md)。

---

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `EPLBManager` | `python/sglang/srt/managers/eplb_manager.py:18` | 重平衡的"调度器"：按 `eplb_rebalance_num_iterations` 周期触发 `rebalance()` |
| `EPLBManager.on_forward_pass_end` | `python/sglang/srt/managers/eplb_manager.py:36` | 每个 forward pass 结束判断是否到了重平衡周期 |
| `EPLBManager.rebalance` | `python/sglang/srt/managers/eplb_manager.py:40` | dump 统计 → 算新放置 → 调 `update_expert_location` |
| `ExpertDistributionRecorder` | `python/sglang/srt/managers/expert_distribution.py:40` | 统计模块抽象基类，含 Noop/Real 两种实现 |
| `_ExpertDistributionRecorderReal` | `python/sglang/srt/managers/expert_distribution.py:104` | 真正记录：管理 single-pass gatherer + accumulator |
| `_SinglePassGatherer` 及子类 | `python/sglang/srt/managers/expert_distribution.py:242` | 单次 forward 内收集每层 expert 命中数(select / deepep normal / low_latency) |
| `_StatAccumulator` | `python/sglang/srt/managers/expert_distribution.py:550` | 把多次 pass 的 physical count 存进 circular buffer，dump 时转 logical count |
| `ExpertLocationMetadata` | `python/sglang/srt/managers/expert_location.py:33` | 全局专家放置的核心数据结构(4 张映射表) |
| `ExpertLocationMetadata.init_by_eplb` | `python/sglang/srt/managers/expert_location.py:124` | 用 EPLB 算法从 `logical_count` 算出放置 |
| `ExpertLocationMetadata.init_trivial` | `python/sglang/srt/managers/expert_location.py:81` | 平凡放置(logical i = physical i) |
| `compute_logical_to_rank_dispatch_physical_map` | `python/sglang/srt/managers/expert_location.py:296` | 为每个 rank 预计算"static dispatch"用的副本选择表 |
| `ExpertLocationDispatchInfo` / `topk_ids_logical_to_physical` | `python/sglang/srt/managers/expert_location_dispatch.py:24` / `:58` | MoE forward 时把 logical expert id 翻译成 physical id |
| `rebalance_experts` | `python/sglang/srt/managers/deepseek_eplb.py:256` | DeepSeek 重平衡算法入口(分 prefill/decode 两种 phase) |
| `make_redundant_experts_chunkwise` | `python/sglang/srt/managers/deepseek_eplb.py:36` | 核心：贪心放冗余副本 + GPU 间二次均衡 |
| `update_expert_location` | `python/sglang/srt/model_executor/expert_location_updater.py:29` | 在线更新：搬权重 + 替换 metadata |
| `update_expert_weights_single_layer` | `python/sglang/srt/model_executor/expert_location_updater.py:84` | 单层权重搬运的 5 种情况(unchanged/same-gpu/free-rider/same-node/cross-node) |

---

## 4. 核心数据流 / 执行流程

### 4.1 概念：logical vs physical expert

理解 EPLB 的前提是分清两种 expert 编号：

- **logical expert**：模型权重里"真实"的专家，数量固定 = `num_logical_experts`(对 DeepSeek-V3 即 `n_routed_experts=256`)。router 输出的 topk_ids 是 logical 的。
- **physical expert**：实际放在某张 GPU 上的一份专家权重。数量 = `num_logical_experts + ep_num_redundant_experts`，见 `python/sglang/srt/managers/expert_location.py:165`。多出来的就是**冗余副本(redundant experts)**：把热门 logical expert 复制成多份，分散到不同 GPU 来分摊负载。

`ExpertLocationMetadata` 用 4 张表描述两者关系(`python/sglang/srt/managers/expert_location.py:34-37`)：

```
physical_to_logical_map          (layers, num_physical)    每个物理槽位放哪个 logical expert
logical_to_all_physical_map      (layers, num_logical, X)  每个 logical 有哪些 physical 副本(-1 padding)
logical_to_all_physical_map_num_valid (layers, num_logical) 上表每行有几个有效副本
logical_to_rank_dispatch_physical_map (layers, num_logical) 给"当前 rank"预选好的那一个 physical 副本
```

### 4.2 统计阶段：expert_distribution

入口是 `model_runner.forward`：每次 forward 用 `with_forward_pass(forward_pass_id, forward_batch)` 包住(`expert_distribution.py:132`)，进出时分别 reset/collect 所有 single-pass gatherer。

MoE forward 内部，topk 选完专家后会调 `on_select_experts`(见 `python/sglang/srt/layers/moe/topk.py:379`)；用 DeepEP 时则走 `on_deepep_dispatch_normal/low_latency`。这些 hook 把"本层每个 expert 命中多少 token"喂给 gatherer：

- `_SelectExpertsSinglePassGatherer`(`expert_distribution.py:325`)：朴素实现，把 topk_ids 搬到 CPU 逐个计数，慢但通用。
- `_DeepepNormalSinglePassGatherer` / `_DeepepLowLatencySinglePassGatherer`(`:347` / `:372`)：直接拿 DeepEP dispatch 已经算好的 `local_physical_count`，生产环境用这条路径，低开销。

注意一个 GPU 只看得到自己负责的 `num_local_physical_experts` 那段，`_convert_local_to_global_physical_count`(`:404`)把它放到全局向量对应区间(其它区间为 0)，后续靠 all-reduce 拼成全局视图。

一次 forward 结束，`_StatAccumulator.append`(`:564`) 把这一 pass 的 `global_physical_count` 压进一个 **circular buffer**(`_CircularBuffer`,`:631`)，长度 = `expert_distribution_recorder_buffer_size`。

dump 时(`_StatAccumulator.dump`,`:580`)做两件事：
1. `_convert_global_physical_count_to_logical_count`(`:678`)：用 `physical_to_logical_map` 做 `scatter_add_`，把"物理计数"折叠回"逻辑计数"——因为重平衡算的是 logical expert 的热度。
2. `all_reduce(SUM)` 把所有 rank 的 logical_count 求和成真正的全局分布。

返回结构 `dict(rank, logical_count)`，`logical_count` 形状是 `(buffer_size_or_steps, num_layers, num_logical_experts)`。

### 4.3 重平衡阶段：deepseek_eplb 算法

`EPLBManager.rebalance`(`eplb_manager.py:40`)取 `dump_record(output_mode="object")["logical_count"]`,丢给 `ExpertLocationMetadata.init_by_eplb`(`expert_location.py:124`)，后者再调 `deepseek_eplb.rebalance_experts`(`deepseek_eplb.py:256`)。

phase 选择(`expert_location.py:138`)：PD 分离的 `disaggregation_mode` 是 `"prefill"` 走 group-aware 的两层分配；否则当 `"decode"` 走简单 chunkwise。

```
tokens_per_expert (steps, layers, num_logical)
        │
        ├─ phase=="prefill" → prefill_rebalance_experts (deepseek_eplb.py:198)
        │     1. 把 logical experts 按 num_groups 分组，pack_groups 把组均衡到各 node
        │     2. 重排成 "mlog"(per-node 连续) 视图
        │     3. make_redundant_experts_chunkwise(每个 chunk = 一个 node)
        │     4. 把 mlog 映射逆变换回 logical
        │
        └─ phase=="decode" → decode_rebalance_experts (deepseek_eplb.py:185)
              整个 EP 当作一个 chunk 直接 make_redundant_experts_chunkwise
        │
        ▼
返回 (physical_to_logical_map, logical_to_all_physical_map, logical_count)
```

`make_redundant_experts_chunkwise`(`deepseek_eplb.py:36`)是算法核心，两步：

1. **放冗余副本(贪心)**：每个 chunk 先把 logical experts 各放一份(`:83`)，然后循环 `num_redundancy_experts_per_group` 次，每次给"分摊后单副本负载最高"的 logical expert 再加一份副本。打分用 `tokens_per_expert / logical_count`(`:98`，logical_count 即当前副本数)，加副本后的预期负载用 `tokens / (count+1)`(`:100`)；`argmin(values.sum(0))`(`:109`)选能让"最大单副本负载"下降最多的那个。`tokens_per_expert_all_diff`(`:95`)加上 `1e-4*arange` 是为了让分数互不相等，避免 tie-break 不确定。
2. **GPU 间二次均衡**(`num_local_physical_experts>1` 时，`:145`)：在一个 chunk 内把物理槽位按负载排序，用蛇形(`flip` + `1::2`,`:166`)分配到各 GPU，让每张卡内部的累计负载尽量接近，同时同步更新 `logical_to_physical_map`。

回到 `_init_raw`(`expert_location.py:181`)：把 `logical_to_all_physical_map` pad 到 `num_physical` 宽度、统计每行有效副本数，再调 `compute_logical_to_rank_dispatch_physical_map`。

### 4.4 dispatch：logical → physical 的运行时翻译

MoE forward 拿到 logical topk_ids 后，必须翻译成实际 physical id 才能找到权重。`ExpertLocationDispatchInfo.init_new`(`expert_location_dispatch.py:35`)按层切出当前 metadata 的几行，`topk_ids_logical_to_physical`(`:58`)按 `ep_dispatch_algorithm` 翻译：

- `"static"`(`_topk_ids_logical_to_physical_static`,`:71`)：直接查 `logical_to_rank_dispatch_physical_map[topk_ids]`。这张表在 `compute_logical_to_rank_dispatch_physical_map`(`expert_location.py:296`)里预计算：用 `base_seed + ep_rank` 的随机数为每个 logical expert 选一个副本，但若某副本恰好在本 rank 上则**优先选本地**(`:330` 的 `is_same_gpu` 覆盖)，从而尽量减少跨卡通信。
- `"dynamic"`(`:77`)：每个 token 实时随机选一个有效副本，更均衡但有随机开销。

### 4.5 在线更新阶段：expert_location_updater

`model_runner.update_expert_location`(`model_runner.py:593`)把模型的 `routed_experts_weights_of_layer`(DeepSeek 侧在 `models/deepseek_v2.py:1611` 构建)交给 `update_expert_location`(`expert_location_updater.py:29`)。流程：先 `_update_expert_weights` 搬权重，再 `old_metadata.update(new_metadata)`(`expert_location.py:213`)**原地** in-place 覆盖那 4 张表(`dst[...] = ...`)，所以持有同一个 metadata 引用的 dispatch 端立刻看到新放置。

`update_expert_weights_single_layer`(`expert_location_updater.py:84`)对本 rank 每个目标物理槽位，按代价从低到高分 5 种情况(`_handle_recv_of_dst_expert_location`,`:134`)：

```
case 1 unchanged   旧=新, 不动                               (:140)
case 2 same-gpu    本卡已有该 logical 的另一副本, 本地 copy   (:148)
case 3 free-rider  本卡更前面的目标槽已经收到了同一 logical   (:164)
case 4 same-node   同 node 内某卡有, 走节点内 P2P            (:182)
case 5 cross-node  只能跨 node P2P                          (:201)
```

跨卡通信用 `torch.distributed` 的 `P2POp(isend/irecv)` + `batch_isend_irecv`(`:340`)，先收进 `temp_buffers`、再 `_execute_buffer2weight_copies` 拷回正式权重(`:344`)，避免 send/recv 用同一块内存导致竞争。`_ChunkUtils`(`:370`)负责把多个 src rank 公平地映射到多个 dst rank(一对多/多对多分块)。

> 关键设计：updater 只在每个 rank 上处理 `local_expert_location_range` 内的目标(`:108`)，全程不重建权重张量、不重启，整轮重平衡耗时由 `EPLBManager.rebalance` 用 `torch.cuda.synchronize` + 计时打到日志(`eplb_manager.py:42-55`)。

---

## 5. 设计难点与权衡

1. **physical/logical 解耦 + 冗余副本是整个机制的地基。** router 永远输出 logical id，放置层完全可以独立变化。代价是每次 forward 都要做一次 `topk_ids_logical_to_physical` gather，以及维护 4 张映射表的一致性(任何一张算错都会让 token 路由到错误权重)。

2. **统计开销 vs 精度。** `_SelectExpertsSinglePassGatherer` 要把 topk_ids 搬 CPU + `torch.cuda.synchronize`(`expert_distribution.py:329`)，注释直言"pretty slow, but we will use the DeepEP Gatherer in production"。DeepEP 路径直接复用 dispatch 阶段已有的 count，几乎零额外开销——这是为什么大规模部署强烈建议配 DeepEP。

3. **circular buffer 与重平衡周期的耦合约束。** `EPLBManager.__init__` 有断言 `eplb_rebalance_num_iterations <= expert_distribution_recorder_buffer_size`(`eplb_manager.py:25`)，否则 buffer 里会混入上一轮的陈旧数据。改这两个参数时务必成对考虑。

4. **prefill 与 decode 用不同算法。** prefill 阶段 batch 大、要考虑 group/node 局部性(DeepSeek 的 group-limited routing)，所以先 `pack_groups` 做 node 级均衡再放副本；decode 阶段 token 稀疏，直接全局 chunkwise 即可。phase 由 `disaggregation_mode` 推断(`expert_location.py:138`)，非 PD 部署一律按 decode。

5. **static dispatch 的本地优先 + 随机性。** `compute_logical_to_rank_dispatch_physical_map` 用 `base_seed + ep_rank` 让不同 rank 选不同副本(分散负载)，又强制本地副本优先(省通信)。这是"均衡"和"通信量"之间的折中；注释 `# TODO use more sophisticated approaches`(`:295`)说明这只是当前启发式。

6. **更新期间的正确性靠 5 种 case 的优先级保证。** free-rider(case 3) 依赖"同一 logical 的多个目标槽位按升序处理、后者复用前者已收到的数据",这隐含了 `_handle_recv` 必须按 `dst_expert_location` 递增顺序遍历(`:129`)。打乱顺序就会破坏 free-rider 假设。

7. **算法刻意制造"互不相等的分数"。** `make_redundant_experts_chunkwise` 里 `tokens_per_expert_all_diff`(`:95`)和注释 `Values in score must be different`(`:99`)是为了让 `argmax/argmin` 在有并列时也确定性地选择，保证各 rank 算出完全一致的放置(否则不同 rank 放置不一致 = 灾难)。

---

## 6. 改代码 / 修 bug 指南

### 想改 X → 看这里

| 需求 | 入手点 |
| --- | --- |
| 调整重平衡频率 | `--eplb-rebalance-num-iterations`(`server_args.py:177`)，注意与 buffer size 的断言(`eplb_manager.py:25`) |
| 增/减冗余副本数量 | `--ep-num-redundant-experts`(`server_args.py:173`)，影响 `num_physical_experts`(`expert_location.py:165`) |
| 换/调重平衡算法 | `deepseek_eplb.make_redundant_experts_chunkwise`(`deepseek_eplb.py:36`)，打分逻辑在 `:96-114` |
| 换 dispatch 策略 | `--ep-dispatch-algorithm`(`server_args.py:174`)+ `expert_location_dispatch.py:58` |
| 改统计来源/粒度 | `expert_distribution.py` 的 gatherer 子类(`:242` 起)；`expert_distribution_recorder_mode`(`server_args.py:178`) |
| 改权重在线搬运 | `update_expert_weights_single_layer`(`expert_location_updater.py:84`) |
| 让新 MoE 模型支持 EPLB | 给模型类加 `get_model_config_for_expert_location`(参考 `models/deepseek_v2.py:1847`)和 `routed_experts_weights_of_layer`(`:1611`) |
| 手动 dump/分析分布 | HTTP 端点 `/start|stop|dump_expert_distribution_record`(`http_server.py:365-388`)，落盘到 `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`(`expert_distribution.py:605`) |
| 用离线分布初始化放置 | `--init-expert-location`(`server_args.py:175`)接 `.pt`/`.json`，解析见 `compute_initial_expert_location_metadata`(`expert_location.py:361`) |

### 常见 bug 高发区

- **映射表不一致 / token 路由错位**：症状是开 EPLB 后精度骤降。先确认 `_convert_global_physical_count_to_logical_count`(`expert_distribution.py:678`)用的 `physical_to_logical_map` 是 **dump 那一刻** 的旧 map(因为 buffer 里的 physical count 是按旧放置统计的)。重平衡后再 dump 必须配套新 map。
- **各 rank 放置不一致**：检查 `deepseek_eplb` 输入是否真的是 all-reduce 后的全局 `logical_count`(`expert_distribution.py:587`)，以及"互不相等分数"的 trick 是否被你改动破坏。
- **更新卡死 / hang**：`_execute_p2p_ops` 用 `batch_isend_irecv` 后 `req.wait()`(`expert_location_updater.py:340`)；若某 rank 的 send/recv 配对不上(case 计算 bug)会全体死锁。用 `update_expert_weights_single_layer(..., debug=True)`(`:92`)打开 `output_logs` 看每个槽位走了哪个 case。
- **buffer 陈旧数据**：违反 `eplb_rebalance_num_iterations <= buffer_size` 断言时；以及忘记 `start_record`(`scheduler.py:2184`)就 dump 会得到全 0。
- **draft worker 误触发**：`EPLBManager` 仅在 `enable_eplb and not is_draft_worker` 时创建(`model_runner.py:262`)，speculative decoding 场景注意别在 draft 上重平衡。

### 调试入手点

1. 看日志：`[EPLBManager] rebalance start/end time=...`(`eplb_manager.py:41,55`)确认是否触发、耗时多少。
2. 看均衡度：开 `--enable-expert-distribution-metrics`(`server_args.py:182`)，日志 `[Expert Balancedness] ... balancedness=`(`expert_distribution.py:524`)给出 `avg/max` 的利用率比(越接近 1 越均衡，定义见 `compute_utilization_rate`,`:714`)。
3. 离线复现算法：直接对 `deepseek_eplb.rebalance_experts` 喂一个构造的 `tokens_per_expert` 张量，单机就能验证放置是否合理，不需要起多卡服务。
4. 单层 updater：`update_expert_weights_single_layer` 是纯函数式、可单测，传入手写的 old/new map 即可断言 P2P 计划。

---

## 相关章节

- [模型执行器](14-model-executor.md)：forward 入口与 EPLB hook 的挂载点。
- [Worker 与并行](15-worker-parallelism.md)：EP/TP、`torch.distributed` 基础。
- [调度器](12-scheduler.md)：`expert_distribution_handle` 处理 start/stop/dump 请求。
- [PD 分离](21-pd-disaggregation.md)：`disaggregation_mode` 决定 prefill/decode 两种重平衡算法。
- [术语表](02-glossary.md)：logical/physical expert、redundant expert、EP 等名词。
