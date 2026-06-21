# PD 分离（Prefill/Decode Disaggregation）

## 1. 一句话职责

把一次推理拆成两个阶段、跑在两类不同的实例上：**prefill 实例**只负责处理 prompt（compute-bound 的 extend forward），算完后把生成的 KV cache 通过 RDMA 传给 **decode 实例**；**decode 实例**只负责逐 token 生成（memory-bound 的 decode forward）。两类实例各自独立调度、独立扩缩容，中间靠一套 **KV transfer 后端**（mooncake / nixl）+ **bootstrap 握手协议**把 KV cache 从 prefill 的显存搬到 decode 的显存。

对应代码全部位于 `python/sglang/srt/disaggregation/`。

## 2. 在端到端链路中的位置

普通（非分离）模式下，一个 scheduler 实例从头跑到尾。PD 分离把链路从中间剖开成两个进程/两台机器：

```
                          ┌──────────────── PREFILL 实例 ────────────────┐
客户端 → mini_lb(LB) ──┬──→ 分词 → 调度(Bootstrap/Waiting/Inflight 队列)   │
   (注入 bootstrap_*)  │      → KV缓存(RadixAttention) → extend forward    │
                       │      → 采样(只采 1 个 token) → KVSender.send() ───┼──┐
                       │                                                   │  │ RDMA write
                       │                                                   │  │ (KV cache + aux)
                       │   ┌──────────────── DECODE 实例 ────────────────┐ │  │
                       └──→│ 分词(同 prompt) → 调度(Prealloc/Transfer/    │ │  │
                           │ Waiting 队列) → 预分配 KV 槽位 → KVReceiver  │←┼──┘
                           │ 接收 → fake extend(跳过 prefill forward)     │ │
                           │ → decode forward → 采样 → 解码 → 响应 ───────┼─┼──→ 客户端
                           └──────────────────────────────────────────────┘ │
                          └──────────────────────────────────────────────────┘
```

关键点：

- **客户端不直接连两台实例**。它连 [mini_lb](#3-关键文件与类) 这个 LB，LB 用 `select_pair()` 选一对 (prefill, decode)，并把 `bootstrap_host/port/room` 三元组注入请求体，再把同一个请求**同时** POST 给 prefill 和 decode 两台实例（见 `mini_lb.py:62` 的 `generate()`）。
- prefill 侧只采 **1 个 token**（`max_new_tokens=1`），这个 token + logprobs 作为「aux 元数据」一起传给 decode；decode 侧用一个「假 extend」把这个 token 接上，然后正常往下 decode。
- 与 [调度器](12-scheduler.md) 的关系：两类实例都复用同一个 `Scheduler`，只是 event loop 换成 disagg 版本（见第 4 节）。与 [RadixAttention](13-memcache-radixattention.md) 的关系：prefill 侧 KV 仍进 radix tree，传完后再 unlock。

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `DisaggregationMode` | `disaggregation/utils.py:27` | 枚举 `NULL/PREFILL/DECODE`，决定 scheduler 走哪个 event loop |
| `TransferBackend` | `disaggregation/utils.py:72` | 枚举 `MOONCAKE/NIXL/FAKE` |
| `get_kv_class()` | `disaggregation/utils.py:85` | 工厂：按 backend + 角色返回对应的 Manager/Sender/Receiver/BootstrapServer 类 |
| `MetadataBuffers` | `disaggregation/utils.py:204` | 存「首 token + logprobs」的 CPU 侧 pinned buffer，跟随 KV 一起 RDMA 传输 |
| `ReqToMetadataIdxAllocator` | `disaggregation/utils.py:49` | 给每个请求分配一个 metadata buffer 槽位 |
| `KVPoll` | `disaggregation/base/conn.py:23` | 传输状态机：`Bootstrapping→WaitingForInput→Transferring→Success/Failed` |
| `KVArgs` | `disaggregation/base/conn.py:11` | 描述本地 KV pool 的裸指针/长度，注册给 RDMA engine |
| `BaseKVManager/Sender/Receiver/BootstrapServer` | `disaggregation/base/conn.py:31-114` | 抽象基类，三个 backend 各自实现 |
| `PrefillBootstrapQueue` | `disaggregation/prefill.py:57` | prefill 侧请求生命周期管理：握手 + 预算估计 |
| `SchedulerDisaggregationPrefillMixin` | `disaggregation/prefill.py:203` | prefill 的两个 event loop + `send_kv_chunk` |
| `DecodePreallocQueue` | `disaggregation/decode.py:71` | decode 侧握手 + 预分配 KV 槽位 |
| `DecodeTransferQueue` | `disaggregation/decode.py:329` | decode 侧 poll 传输状态、收 aux 元数据 |
| `SchedulerDisaggregationDecodeMixin` | `disaggregation/decode.py:439` | decode 的两个 event loop + 「假 extend」构造 |
| `ScheduleBatchDisaggregationDecodeMixin` | `disaggregation/decode_schedule_batch_mixin.py:19` | `prepare_for_prebuilt_extend`/`process_prebuilt_extend`：跳过 prefill forward |
| `MooncakeKVManager` | `disaggregation/mooncake/conn.py:120` | mooncake 后端核心：bootstrap 线程 + transfer 线程 |
| `MooncakeTransferEngine` | `disaggregation/mooncake/transfer_engine.py:9` | 对 mooncake C++ engine 的薄封装（`transfer_sync_write`）|
| `NixlKVManager` | `disaggregation/nixl/conn.py:119` | nixl 后端核心，用 `nixl_agent` 做 GPUDirect WRITE |
| `MiniLoadBalancer` | `disaggregation/mini_lb.py:47` | 测试用 LB，注入 bootstrap 信息、并发转发、合并 logprobs |
| `ZmqEventPublisher` | `disaggregation/kv_events.py:104` | KV cache events 的 ZMQ PUB 发布器（供外部 KV-aware 路由器消费）|
| `KVEventBatch`/`BlockStored`/`BlockRemoved` | `disaggregation/kv_events.py:74/58/66` | KV events 协议的消息体 |

scheduler 侧的接线：`disaggregation_mode` 在 `managers/scheduler.py:460` 读取；三套队列在 `scheduler.py:557-633` 初始化；event loop 在 `scheduler.py:2304-2321` 按模式分派。

## 4. 核心数据流 / 执行流程

### 4.1 整体握手 + 传输时序

bootstrap 三元组的含义：`bootstrap_host/bootstrap_port` 指向 **prefill 实例的 bootstrap server**（一个 HTTP 服务，`MooncakeKVBootstrapServer`，`mooncake/conn.py:734`）；`bootstrap_room` 是这次请求的全局唯一 ID（LB 随机生成，`mini_lb.py:302`），prefill 和 decode 用它来配对。

```
 decode 实例                          prefill 实例 (含 bootstrap server)
 ───────────                          ─────────────────────────────────
 Receiver.__init__
   GET /route?engine_rank=-1   ─────► 返回 prefill 的 tp_size/dp_size
   (算出 target_tp_rank / dp_group)
   GET /route?engine_rank=R    ─────► 返回该 rank 的 (rank_ip, rank_port)
   (首次) _register_kv_args:
     ZMQ PUSH 本地 KV/aux 裸指针 ───► 存入 decode_kv_args_table[session]
 [KVPoll.WaitingForInput]

 pop_preallocated:
   预分配 KV 槽位 → kv_indices
   Receiver.init(kv_indices, aux_idx)
     ZMQ PUSH (room, dst_kv_indices,  ─► bootstrap_thread 收到, 存 transfer_infos[room]
               dst_aux_index, ...)       当收齐 required_dst_info_num 个 →
                                         update_status(room, WaitingForInput)
                                         ◄──────────── prefill 侧 Sender 此时变 WaitingForInput
                                         (prefill 此前一直 Bootstrapping, 卡在 pop_bootstrapped)

                                       prefill: 跑完 extend forward, send_kv_chunk:
                                         Sender.send(page_indices)
                                         → add_transfer_request → transfer_queue
                                       transfer_thread:
                                         send_kvcache(): RDMA WRITE 每层 KV ──┐
                                         send_aux():     RDMA WRITE 首token ──┼──► 写进 decode 显存/CPU buffer
                                         update_status(room, Success)         │
   decode_thread 收到 status ◄──── sync_status_to_decode_endpoint (ZMQ PUSH) ┘
   update_status(room, Success)
 pop_transferred:
   poll==Success → 从 metadata_buffers 读出首 token → req.output_ids
   → waiting_queue → 假 extend → 正常 decode
```

注意传输方向是 **prefill 主动 WRITE 到 decode**（RDMA single-sided write），不是 decode 来读。decode 只是先把目标 KV 槽位的地址（`dst_kv_indices` + 之前注册的 `dst_kv_ptrs`）告诉 prefill。

### 4.2 prefill 侧请求三段式生命周期

代码注释（`prefill.py:1-18`）总结得很清楚，对应三个队列：

1. **Bootstrap Queue**（`PrefillBootstrapQueue`，`prefill.py:57`）：请求进来时 `add()`（`prefill.py:132`）给它建一个 `KVSender`，状态置 `Bootstrapping`。`pop_bootstrapped()`（`prefill.py:152`）每轮 poll 所有 sender，等到 decode 端把 `transfer_infos` 注册齐、状态翻成 `WaitingForInput`，再 `Sender.init(num_pages, metadata_buffer_index)` 通知传输长度，移入 waiting queue。这里有一个**资源预算**：`max_new_tokens=1`（`prefill.py:150`）使 `PrefillAdder` 的显存估计准确。
2. **Waiting Queue**：复用普通 scheduler 的 `get_new_batch_prefill` 跑 extend forward。
3. **Inflight Queue**（`disagg_prefill_inflight_queue`）：forward 完后在 `process_batch_result_disagg_prefill`（`prefill.py:288`）里把请求 push 进去，并调用 `send_kv_chunk(req, last_chunk=True)`（`prefill.py:452`）触发 KV 传输。`process_disagg_prefill_inflight_queue`（`prefill.py:383`）每轮非阻塞 poll：`Success` 就 `cache_finished_req` 解锁 radix tree、`stream_output` 回给客户端（其实 prefill 的「响应」只是 input logprobs 等元信息，真正的 token 流由 decode 出）。

`send_kv_chunk`（`prefill.py:452`）的细节：

- 它支持 **chunked prefill** —— `start_send_idx` 到 `end_idx` 是本次要传的区间；非最后一块时会把不满一页的尾巴留到下次（`end_idx - end_idx % page_size`，`prefill.py:471`），保证页对齐。
- 用 `kv_to_page_indices`（`utils.py:130`）把 token 级索引压成 page 级索引（page_size>1 时取 `indices[::page_size] // page_size`）。
- 只有 `last_chunk` 才 `set_buf(req)`（`utils.py:257`），把首 token / logprobs 写进 `MetadataBuffers`，随最后一块一起传。

### 4.3 decode 侧请求四段式生命周期

代码注释见 `decode.py:1-19`，对应队列在 `scheduler.py:576-606` 创建，`process_decode_queue`（`decode.py:640`）每轮串起来：

1. **PreallocQueue**（`DecodePreallocQueue`，`decode.py:71`）：`add()`（`decode.py:149`）建 `KVReceiver`（构造函数里就完成了 GET /route 握手）。`pop_preallocated()`（`decode.py:199`）在握手完成（`waiting_for_input`）且有足够 KV 显存时，调 `_pre_alloc`（`decode.py:287`）预分配 `req_to_token` 和 KV 槽位，再 `Receiver.init(page_indices, metadata_buffer_index)` 把目标地址推给 prefill。
   - **显存预算**`_allocatable_tokens()`（`decode.py:264`）：会为 running / transfer / waiting 队列里每个请求各预留 `num_reserved_decode_tokens`（默认 512，`SGLANG_NUM_RESERVED_DECODE_TOKENS`）个 token 的 decode 空间，避免传完 KV 后没地方继续 decode。
2. **TransferQueue**（`DecodeTransferQueue`，`decode.py:329`）：`pop_transferred()`（`decode.py:355`）poll receiver。`Success` 时从 `metadata_buffers.get_buf(idx)` 读出 prefill 算的首 token、append 到 `req.output_ids`（`decode.py:396`），移入 waiting queue。
3. **WaitingQueue → 假 extend**：`get_new_prebuilt_batch`（`decode.py:592`）把这些请求组成一个 batch，调 `prepare_for_prebuilt_extend` + `process_prebuilt_extend`（`decode_schedule_batch_mixin.py:21/96`）。这两个函数**只填充 metadata、不跑 prefill forward**——KV 已经从 prefill 传过来了，所以这里是个「假的」extend，目的只是把 batch 状态对齐到「已 prefill 完、有 1 个 output token」。
4. **RunningBatch**：`get_next_disagg_decode_batch_to_run`（`decode.py:560`）把假 extend batch merge 进 running batch，之后就是普通 decode forward。

### 4.4 KV transfer 后端：mooncake vs nixl

两个后端都实现 `BaseKVManager/Sender/Receiver`，核心都是「按层、按连续块分组、RDMA WRITE」：

- **分组**`group_concurrent_contiguous`（`mooncake/conn.py:37`、`nixl/conn.py` 同名）：把 src/dst 索引里**连续**的段合并成一次大传输（`np.diff != 1` 处切断），减少 RDMA 描述符数量。
- **逐层传输**`send_kvcache`：KV pool 每层一个连续 buffer（`get_contiguous_buf_infos`），地址 = `层基址 + 块首索引 * item_len`，长度 = `item_len * 块长`。
  - mooncake（`mooncake/conn.py:189`）用线程池**并发**跑每一层（`process_layer`），调 `engine.transfer_sync` → `transfer_sync_write`（`transfer_engine.py:66`）。
  - nixl（`nixl/conn.py:207`）把所有层的 descs 攒成一个 `xfer`，一次 `initialize_xfer("WRITE", ...)` + `transfer()`，走 GPUDirect（`"VRAM"` 内存类型，带 `dst_gpu_id`）。
- **aux 数据**`send_aux`：首 token / logprobs 在 CPU pinned buffer（mooncake 走 DRAM，nixl 也用 `"DRAM"`），单独一次小传输。
- **状态回传**：mooncake 传完后 `sync_status_to_decode_endpoint`（`mooncake/conn.py:265`）通过 ZMQ PUSH 把 `Success/Failed` 推回 decode 的 `decode_thread`（`mooncake/conn.py:383`），decode 侧据此更新本地 `request_status`。

`FakeKVSender/Receiver`（`fake/conn.py`）不做任何真实传输，poll 立即返回 `Success`，专门给 warmup 请求用（`bootstrap_host == FakeBootstrapHost == "2.2.2.2"`，`utils.py:21`，prefill.py:133 / decode.py:151 判定）。

### 4.5 KV events 协议

这是**独立于 KV transfer** 的另一套机制：把「某个 KV block 被存入 / 移除 radix tree」作为事件广播出去，供**外部 KV-aware 路由器**（如 prefix-cache-aware LB）订阅，从而把后续相同前缀的请求路由到已有缓存的实例。

- 触发点在 [RadixAttention](13-memcache-radixattention.md) 里：`_record_store_event`（`mem_cache/radix_cache.py:463`）造 `BlockStored`、`_record_remove_event`（`radix_cache.py:477`）造 `BlockRemoved`，进队列。
- scheduler 每轮调 `_publish_kv_events`（`scheduler.py:2226`）：`take_events()` 取出后包成 `KVEventBatch`（`kv_events.py:74`），交给 `ZmqEventPublisher.publish`。
- `ZmqEventPublisher`（`kv_events.py:104`）用单独线程发 ZMQ PUB，带**单调递增 seq**和**replay buffer**（`buffer_steps` 个历史 batch），订阅者可通过 ROUTER socket 发起始 seq 请求补漏（`_service_replay`，`kv_events.py:268`），实现 at-least-once。
- 通过 `--kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557"}'` 启用（`KVEventsConfig.from_cli`，`kv_events.py:326`）。

### 4.6 mini_lb 负载均衡

`mini_lb.py` 是**测试/参考实现**（顶部注释明说 "Minimal HTTP load balancer ... for testing"），生产环境通常用独立的 Rust LB。它做三件事：

1. **配对**`select_pair()`（`mini_lb.py:53`）：从已注册的 prefill / decode 列表里**纯随机**各选一个。没有任何负载感知。
2. **注入 bootstrap 信息**：`handle_generate_request`（`mini_lb.py:231`）把 prefill 的 hostname、bootstrap_port、随机 room 塞进请求体；batch 请求时按 batch_size 各生成一份（`mini_lb.py:240`）。
3. **并发转发 + 合并**`generate`/`generate_stream`（`mini_lb.py:62/100`）：把同一请求**同时** POST 给 prefill 和 decode，`asyncio.gather` 等两边。最终 token 流以 **decode 的响应**为准；若 `return_logprob`，把 prefill 的 `input_token_logprobs` 拼到 decode 结果前面（`mini_lb.py:139`）——因为 input logprobs 只有 prefill 算得出来。

实例可主动 `POST /register`（`mini_lb.py:331`）把自己加入 LB，对应 `register_disaggregation_server`（`utils.py:164`）。

### 4.7 与大规模 EP 部署的关系（96×H100）

PD 分离是大规模 expert-parallel（EP）部署（参考官方 96×H100 DeepSeek 部署 blog）的前提条件，原因在 `docs/backend/pd_disaggregation.md:7-14` 讲得很直接：

- **prefill 与 decode 的最优并行策略不同**。看 `pd_disaggregation.md:42-48` 的 DeepSeek 多节点示例：prefill 用 `--deepep-mode normal`（吞吐优先），decode 用 `--deepep-mode low_latency` 且限 `--max-running-requests 128`（延迟优先）。两阶段拆开后才能各自选最优 EP/DP 配置。
- **避免 prefill 打断 decode**：unified 模式下，新来的 prefill batch 会打断正在 decode 的 batch（`pd_disaggregation.md:11`），并在 DP attention 下造成 worker 间负载不均（`pd_disaggregation.md:12`）。分离后 decode 实例只跑 decode，TBT（time-between-tokens）平稳。
- 在 disagg 模式下两类实例都**强制 `dp_size==1` 除非开 `enable_dp_attention`**（`mooncake/conn.py:142`），因为 KV 传输的 rank 映射依赖明确的 TP/DP 拓扑（见第 5 节）。

## 5. 设计难点与权衡

**(a) 为什么 prefill 只采 1 个 token 并单独传 aux？**
prefill forward 的最后一步会产出第一个 output token；如果不把它传给 decode，decode 的「假 extend」就缺了起点。`MetadataBuffers`（`utils.py:204`）刻意把它放在 CPU pinned buffer 而非 KV pool，单独一次小 RDMA（`send_aux`），并 padding 到 ≥64 字节（`utils.py:209` 注释：RDMA 最小传输单位）。

**(b) prefill 与 decode 的 TP size 不一致怎么办？**
最复杂的逻辑在 `MooncakeKVReceiver.__init__`（`mooncake/conn.py:547-590`）。三种情况：
- TP 相等：1 对 1，`required_dst_info_num=1`。
- decode TP > prefill TP（仅 MLA 支持）：多个 decode rank 共享一个 prefill rank 的 KV，一个 prefill rank 要被多个 decode rank 注册，`required_dst_info_num>1`（`mooncake/conn.py:557`）。
- decode TP < prefill TP（仅 MLA）：一个 decode rank 要从多个 prefill rank 收 KV，构造 `target_tp_ranks` 列表，对其中真正要数据的发实数据、其余发 **dummy 请求**（`is_dummy`，`mooncake/conn.py:606`）。dummy 的存在是为了让 prefill 侧 `transfer_infos[room]` 能收齐 `required_dst_info_num` 个、从而把状态正确翻成 `WaitingForInput`（`mooncake/conn.py:304`），否则会**永久卡在 Bootstrapping**。非 MLA 模型不允许 TP 不一致（`mooncake/conn.py:559` 的 assert）。

**(c) 状态机用 `max` 合并的玄机。**`update_status`（`mooncake/conn.py:417`）用 `max(old, new)` 更新状态。因为 prefill 侧可能**先**收到 decode 的 bootstrap 注册（`WaitingForInput=2`）、**后**才轮到自己把状态设成 `Bootstrapping=1`，用 max 保证状态只能单调前进，不会被回退。这是个非显然但关键的并发细节。

**(d) overlap scheduler 下 KV 传输要延迟。**`process_prefill_chunk`（`prefill.py:433`）里，开了 overlap 时**不立即** `send_kv_chunk`，而是记 `tmp_end_idx`、推迟到 `process_batch_result_disagg_prefill`（`prefill.py:377`）result resolve 之后再传——否则会传到尚未算完的 KV。见 [overlap scheduler](12-scheduler.md)。

**(e) decode 的显存死锁防护。**`_allocatable_tokens`（`decode.py:264`）的预留逻辑：如果不为正在传输和等待的请求预留 decode 空间，可能出现「KV 传进来了但没地方 decode」的死锁。这就是 `num_reserved_decode_tokens` 的意义。

**(f) page 对齐。**KV 以 page 为单位传输，非整页的尾部必须延后（`prefill.py:471`），否则 decode 侧 page 边界对不上。chunked prefill + paging 是 bug 高发组合。

## 6. 改代码 / 修 bug 指南

**想加一个新的 KV transfer 后端 →** 实现 `BaseKVManager/Sender/Receiver/BootstrapServer`（`base/conn.py`），在 `TransferBackend` 枚举（`utils.py:72`）加一项，在 `get_kv_class`（`utils.py:85`）注册映射。参考最简单的 `fake/conn.py`，真实实现对照 `nixl/conn.py`（比 mooncake 更紧凑）。

**想改负载均衡策略 →** `MiniLoadBalancer.select_pair`（`mini_lb.py:53`）。现在是纯随机，要做 KV-aware 路由就订阅第 4.5 节的 KV events、结合各实例负载。注意 mini_lb 只是参考实现。

**想改预分配 / 显存预算 →** decode 侧看 `_allocatable_tokens`（`decode.py:264`）和 `num_reserved_decode_tokens`；prefill 侧看 `PrefillBootstrapQueue._process_req`（`prefill.py:146`，`max_new_tokens=1`）。

**想改传出去的元数据（如支持 >128 top_logprobs）→** `MetadataBuffers`（`utils.py:204`），注意 `TODO: abort top_logprobs_num > 128`（`utils.py:206`）和 `set_buf`/`get_buf`。

**常见 bug 高发区：**
- **请求卡在 Bootstrapping 不动** → 多半是 TP/DP 不一致下 `required_dst_info_num` 没收齐（dummy 请求逻辑，`mooncake/conn.py:304`），或 bootstrap server 的 `/route` 没返回正确 rank 信息（`_handle_route_get`，`mooncake/conn.py:805`）。
- **高并发下 hang** → 看 `prefill.py:242` 那条 HACK 注释：disagg prefill loop 不进 `update_running_batch`，必须手动 reset `batch_is_full`，否则 hang。
- **KV 对不上 / 乱码** → page 对齐（`prefill.py:471`）、`group_concurrent_contiguous` 分组（`mooncake/conn.py:37`）、overlap 延迟传输（`prefill.py:377`）三处任一出错。
- **decode 死锁（KV 满但不前进）** → `_allocatable_tokens` 预留不足。

**调试入手点：**
- 设环境变量 `DISAGGREGATION_TEST_FAILURE_PROB`（`utils.py:24`）注入随机失败，验证失败路径（`prepare_abort`，`utils.py:189`）。
- 用 `FAKE` 后端（`bootstrap_host="2.2.2.2"`）跑通调度流程、排除 RDMA 因素。
- prefill 看 `disagg_prefill_inflight_queue` 长度、decode 看 `disagg_decode_transfer_queue`/`prealloc_queue` 长度（scheduler 的 `print_stats` 已打印，`scheduler.py:1151`）。
- 状态卡住时打印 `MooncakeKVManager.request_status[room]` 和 `transfer_infos[room]`。

## 交叉引用

- [请求生命周期](01-request-lifecycle.md)
- [调度器与 overlap scheduler](12-scheduler.md)
- [RadixAttention 与 KV cache](13-memcache-radixattention.md)
- [model executor / forward](14-model-executor.md)
- [术语表](02-glossary.md)
