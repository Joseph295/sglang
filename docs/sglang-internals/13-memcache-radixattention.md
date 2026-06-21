# 内存与 KV 缓存：RadixAttention 与 paged KV pool（核心）

> 本章是 SGLang 的「招牌特性」章节。读完后你应该能：看懂 radix tree 如何做 prefix caching、KV 显存如何按 page/token 管理、引用计数如何保证「正在用的 KV 不被释放」，并且知道改相关代码 / 修 bug 时该看哪几行。

## 1. 一句话职责

这一层负责 **把已经算好的 KV cache 尽可能复用**，并 **在显存里高效地分配 / 回收 KV cache 的物理位置**。它由两件事组成：

- **prefix caching（前缀复用）**：用一棵 radix tree（基数树）把「token 序列 → KV cache 物理位置」的映射组织起来，新请求只要前缀命中，就直接复用旧请求算过的 KV，省掉这段 prefill 的计算。这就是 **RadixAttention**。
- **KV 显存管理**：用两级内存池（`ReqToTokenPool` + `TokenToKVPoolAllocator` / `KVCache`）把「请求 → token 槽位 → KV 物理 buffer」的映射管起来，支持按 token 或按 page 分配、释放、LRU 驱逐。

## 2. 在端到端链路中的位置

```
客户端 → 分词 → 调度(scheduler) → ┌──────────────────────────┐ → 前向(forward) → 采样 → 解码 → 响应
                                  │   本章：KV 缓存 / 内存池   │
                                  └──────────────────────────┘
                                            ▲
                  scheduler 在 add 请求时调用 match_prefix() 拿到可复用前缀；
                  在 batch 结束时调用 cache_finished_req / cache_unfinished_req 把新算的 KV 插回 tree；
                  显存不够时调用 evict() 驱逐最久未用的 KV。
```

更细一点，本章模块同时被三方调用（见 [调度器](12-scheduler.md)）：

```
              ┌─────────────────────────────────────────────────────────┐
 Req 进来 ──▶ │ scheduler.get_new_batch_prefill / PrefillAdder            │
              │   tree_cache.match_prefix(token_ids)  ── 命中前缀 ─┐       │
              └───────────────────────────────────────────────────│──────┘
                                                                   ▼
                              prefix_indices (复用) + 新 token 需要新分配的槽位
                                                                   │
              ┌────────────────────────────────────────────────────────┐
 forward ──▶  │ attention backend 读 KVCache.get_kv_buffer / set_kv_buffer│
              └────────────────────────────────────────────────────────┘
                                                                   │
              ┌────────────────────────────────────────────────────────┐
 batch 结束 ▶ │ tree_cache.cache_finished_req / cache_unfinished_req     │
              │   把新算的 KV insert 回 radix tree；free 多余槽位          │
              └────────────────────────────────────────────────────────┘
```

## 3. 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `BasePrefixCache` | `python/sglang/srt/mem_cache/base_prefix_cache.py:5` | 所有 prefix cache 的抽象基类，定义 `match_prefix / insert / cache_finished_req / evict / inc_lock_ref / dec_lock_ref` 接口 |
| `RadixCache` | `python/sglang/srt/mem_cache/radix_cache.py:98` | radix tree 实现的 prefix cache，SGLang 默认缓存 |
| `TreeNode` | `python/sglang/srt/mem_cache/radix_cache.py:44` | radix tree 节点，存 `key`(token ids) / `value`(kv indices) / `lock_ref` / `last_access_time` |
| `RadixCache.match_prefix` | `python/sglang/srt/mem_cache/radix_cache.py:138` | 查最长可复用前缀 |
| `RadixCache._insert_helper` | `python/sglang/srt/mem_cache/radix_cache.py:380` | 插入新序列，必要时分裂节点 |
| `RadixCache.evict` | `python/sglang/srt/mem_cache/radix_cache.py:263` | LRU 驱逐叶子节点直到腾出 `num_tokens` |
| `RadixCache.inc_lock_ref / dec_lock_ref` | `python/sglang/srt/mem_cache/radix_cache.py:288` / `:302` | 引用计数，保护正在用的 KV 不被驱逐 |
| `ChunkCache` | `python/sglang/srt/mem_cache/chunk_cache.py:22` | 禁用 radix cache 时的退化缓存，只支持 chunked prefill，不做前缀复用 |
| `HiRadixCache` | `python/sglang/srt/mem_cache/hiradix_cache.py:23` | 分层缓存（GPU + CPU host），继承 `RadixCache` |
| `ReqToTokenPool` | `python/sglang/srt/mem_cache/memory_pool.py:54` | 第一级池：`req_pool_idx → 每个 token 的 kv 槽位` |
| `TokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/memory_pool.py:169` | 第二级分配器：管理 kv 槽位（token 粒度）的 free list |
| `KVCache` | `python/sglang/srt/mem_cache/memory_pool.py:102` | 真正持有 KV 物理显存的抽象基类 |
| `MHATokenToKVPool` | `python/sglang/srt/mem_cache/memory_pool.py:238` | 标准多头注意力的 KV buffer（K/V 各一份） |
| `MLATokenToKVPool` | `python/sglang/srt/mem_cache/memory_pool.py:511` | MLA（DeepSeek 类）的 KV buffer（压缩成单 buffer） |
| `PagedTokenToKVPoolAllocator` | `python/sglang/srt/mem_cache/paged_allocator.py:157` | page 对齐的分配器，输出 page-aligned 的连续 indices |
| `HostKVCache` | `python/sglang/srt/mem_cache/memory_pool.py:760` | host(CPU) 侧 KV buffer 抽象，供 HiRadixCache 使用 |
| `MultiModalCache` | `python/sglang/srt/mem_cache/multimodal_cache.py:6` | 缓存 VLM encoder 输出 embedding（与 KV 无关，独立小缓存） |

三种 cache 的选择逻辑在 scheduler 里，见 `python/sglang/srt/managers/scheduler.py:498`：

```
disable_radix_cache 且开启 chunked_prefill → ChunkCache
否则 enable_hierarchical_cache             → HiRadixCache
否则                                        → RadixCache (disable 标志可整体关掉前缀复用)
```

## 4. 核心数据流 / 执行流程

### 4.1 两级内存池：从「请求」到「物理 KV」

理解这一切的前提是看懂三层映射：

```
  Req (req_pool_idx)
        │  ReqToTokenPool.req_to_token[req_pool_idx]  形状 [max_context_len], dtype int32
        ▼
  每个 position 对应一个 "kv 槽位号" (token index)
        │  TokenToKVPoolAllocator 管理这些槽位号的 free list
        ▼
  槽位号 = KVCache.k_buffer[layer][slot] / v_buffer[layer][slot] 的下标
        │
        ▼
  真正的 KV 张量（显存）：[size+page_size, head_num, head_dim] per layer
```

- `ReqToTokenPool`（`memory_pool.py:54`）：一张 `[size, max_context_len]` 的 int32 大表（`memory_pool.py:72`）。每个活跃请求占一行（`req_pool_idx`），第 i 列存「该请求第 i 个 token 的 KV 槽位号」。`alloc` 从 `free_slots` 取行号，`free` 还回去（`memory_pool.py:83`、`:92`）。
- `TokenToKVPoolAllocator`（`memory_pool.py:169`）：管理「槽位号」这一维度的 free list。`page_size == 1` 时就是逐 token 分配：`free_slots` 是一个 `arange(1, size+1)` 的 tensor（`memory_pool.py:231`），`alloc` 切前 N 个，`free` 用 `torch.cat` 拼回去。**注意 slot 0 是留给 padding 的 dummy 槽**（`memory_pool.py:230` 注释）。
- `KVCache` / `MHATokenToKVPool`（`memory_pool.py:238`）：真正的显存。每层一个 `k_buffer` 和 `v_buffer`，形状 `[size+page_size, head_num, head_dim]`（`memory_pool.py:281`）。forward 时 attention backend 调 `set_kv_buffer(layer, loc, cache_k, cache_v)`（`memory_pool.py:378`）把新算的 KV 写到 `loc` 指定的槽位，调 `get_kv_buffer` 读出来做 attention。

MLA 模型走 `MLATokenToKVPool`（`memory_pool.py:511`），把 KV 压成单个 `kv_buffer`（`[size+page_size, 1, kv_lora_rank + qk_rope_head_dim]`，`memory_pool.py:541`），显存省一大截。

### 4.2 paged 分配：为什么需要 page，以及 page 对齐分配

`page_size == 1` 时是 token 粒度，逻辑最简单但 attention kernel 的访存不友好。多数后端用 **paged KV**：每 `page_size` 个连续 token 组成一个 page，分配以 page 为单位。这由 `PagedTokenToKVPoolAllocator`（`paged_allocator.py:157`）负责，关键特点：

- free list 是 **page 号** 而不是 token 号：`free_pages = arange(1, num_pages+1)`（`paged_allocator.py:318`）。
- `alloc(need_size)` 要求 `need_size` 是 page 对齐的，取若干 page，再展开成 token indices（`paged_allocator.py:196`）。
- 真正的难点在 **续写**：一个请求 prefill/decode 时，前缀已经占了若干 token（可能是半个 page），新 token 要先 **填满旧的半个 page**，再分配新的整 page，最后可能再开半个新 page。这个三段式逻辑用一个 triton kernel `alloc_extend_kernel`（`paged_allocator.py:29`）一次性算出所有 out_indices：
  - Part 1：填旧的 partial page（`paged_allocator.py:71`）
  - Part 2：填新的 full pages（`paged_allocator.py:85`）
  - Part 3：开新的 partial page（`paged_allocator.py:104`）
- decode 阶段每步只加 1 个 token，用更轻的 `alloc_decode_kernel`（`paged_allocator.py:116`）：要么续上当前 page 的下一个 slot（`last_loc + 1`），要么开一个新 page。

`free` 时把 token indices 整除回 page 号去重再还回 free list（`paged_allocator.py:288`）。

### 4.3 radix tree：match → insert → evict

radix tree 把所有「已缓存序列」组织成一棵压缩前缀树。每个 `TreeNode`（`radix_cache.py:44`）：

- `key`：一段 token id（边上的标签，radix tree 的特征是边可以是多 token 的串）
- `value`：这段 token 对应的 KV 槽位 indices（一个 tensor）
- `lock_ref`：引用计数，>0 表示有请求正在用，不可驱逐
- `last_access_time`：LRU 时间戳（`__lt__` 按它比较，`radix_cache.py:73`，配合 `heapq` 做最小堆）

**match_prefix**（`radix_cache.py:138` → `_match_prefix_helper:336`）：从 root 出发，用 `get_child_key_fn(key)` 取第一个 child key（page_size=1 时是 `key[0]`，否则是 `tuple(key[:page_size])`，见 `radix_cache.py:119`）找子节点；逐节点比较，匹配满整条边就继续往下走，匹配到一半就 `_split_node` 把节点劈成两段，返回到分裂点。返回 `(value, last_node)`：value 是拼接好的可复用 KV indices，last_node 是命中的最深节点。

```
插入 "Hello" 后:                 再插入 "Hello_LA":
   root                              root
    │ "Hello"                         │ "Hello"            <- 公共前缀被提取为一条边
    ●                                 ●  (split 点)
                                      │ "_LA"
                                      ●
match_prefix("Hello_world")  →  命中 "Hello"，在 ● 处分裂或停下，
                                返回 "Hello" 对应的 5 个 kv indices 复用，
                                "_world" 部分需要新算。
```

**insert**（`radix_cache.py:170` → `_insert_helper:380`）：和 match 类似地往下走，能匹配的部分累加 `total_prefix_length`（这部分 KV 已经存在，调用方随后会把重复分配的槽位 free 掉）；走到无法匹配处，剩余 key/value 挂成一个新 `TreeNode`，并把这段长度加进 `evictable_size_`（`radix_cache.py:409`）。匹配到边中间时同样 `_split_node`（`radix_cache.py:361`）。

**节点分裂 `_split_node`**（`radix_cache.py:361`）是 radix tree 的核心操作：把一个节点在 `split_len` 处切成「父段 new_node → 子段 child」，`key`/`value` 都按 split_len 切开，`lock_ref` 继承给 new_node。这保证了「公共前缀只存一份」。

**插回 tree 的入口**：batch 跑完后 scheduler 调用：

- `cache_finished_req`（`radix_cache.py:178`）：请求结束。取出该请求所有 token 的 kv_indices，做 page 对齐后 `insert` 回 tree；把 `[prefix_indices : new_prefix_len]` 这段「插入时发现已经存在、于是重复了」的槽位 `free` 掉；释放 req 槽位并 `dec_lock_ref(req.last_node)`（`radix_cache.py:211`）。
- `cache_unfinished_req`（`radix_cache.py:213`）：chunked prefill 中途，把已经算好的部分 insert 回去，重新 `match_prefix` 拿到更新后的 prefix_indices，更新 `ReqToTokenPool`，并把 lock 从旧 last_node 转到新 last_node（`radix_cache.py:244`）。

**evict（LRU 驱逐）**（`radix_cache.py:263`）：

```
1. _collect_leaves() 收集所有叶子节点，heapify 成最小堆（按 last_access_time）
2. while 还没驱逐够 num_tokens:
     x = 弹出最久未访问的叶子
     if x == root: break
     if x.lock_ref > 0: continue        # 正在用，跳过！
     token_to_kv_pool_allocator.free(x.value)   # 释放显存
     num_evicted += len(x.value)
     _delete_leaf(x)                    # 从父节点 children 删掉
     if 父节点变成叶子: 把父节点 push 回堆     # 自底向上继续驱逐
```

注意只驱逐 **叶子**，且驱逐后父节点若变叶子会被重新入堆，从而实现「从树梢往树根」的 LRU 回收。`lock_ref > 0` 的节点直接跳过——这就是「不释放正在用的 KV」的保证。

### 4.4 引用计数：怎么保证正在用的 KV 不被释放

`inc_lock_ref(node)`（`radix_cache.py:288`）从 node 一路 `+1` 到 root；当某节点 `lock_ref` 从 0 变 1 时，把它的大小从 `evictable_size_` 移到 `protected_size_`。`dec_lock_ref`（`radix_cache.py:302`）反向操作，从 1 变 0 时移回 evictable。

调用时机：

- 请求被加入 batch、确定要复用某条前缀路径时，对 `last_node` 调 `inc_lock_ref` → 整条路径（到 root）全部锁住，evict 时会因 `lock_ref > 0` 跳过。
- 请求结束（`cache_finished_req`）或前缀切换（`cache_unfinished_req`）时 `dec_lock_ref`，解锁后这些节点重新可被 LRU 驱逐。

`evictable_size()` / `protected_size()`（`radix_cache.py:316`）被 scheduler 用来算「还能放多少 token」（`scheduler.py:1138` 一带）。

### 4.5 HiRadixCache：GPU/CPU 分层缓存

`HiRadixCache`（`hiradix_cache.py:23`）继承 `RadixCache`，在 GPU radix tree 之上叠加一层 **host(CPU) KV 备份**，让被 GPU 驱逐的 KV 不立即丢弃，而是写到 CPU 内存（`MHATokenToKVPoolHost` / `MLATokenToKVPoolHost`，`memory_pool.py:935`/`:1020`），命中时再异步 load 回 GPU。这样有效缓存容量从「GPU 显存」扩展到「GPU + 大块 CPU 内存」。

`TreeNode` 为此预留了字段（`radix_cache.py:57`）：`host_value`（CPU 侧 indices）、`loading`（正在从 host 载回）、`evicted`（property，`value is None` 即 GPU 上已无，`radix_cache.py:65`）、`backuped`（`host_value is not None`，`radix_cache.py:69`）。

核心机制：

- **write-back / write-through**：由 `HiCacheController`（`managers/cache_controller.py:146`）的后台线程异步把 GPU KV 拷到 host。`inc_hit_count`（`hiradix_cache.py:106`）在命中次数超过 `write_through_threshold` 时触发 `write_backup`（`hiradix_cache.py:84`）。
- **分层 evict**（`hiradix_cache.py:154`）：被驱逐节点若还没备份，按写策略要么先 write-back 到 host 再驱逐（`_evict_backuped`，`hiradix_cache.py:191`，只清 GPU `value`、保留 `host_value`），要么直接 `_evict_regular` 删除（`hiradix_cache.py:199`）。
- **load back**（`hiradix_cache.py:229`）：match 命中一个 `evicted` 节点（GPU 没有但 host 有）时，沿父链收集所有 evicted 节点，`inc_lock_ref` 保护祖先，整段一起从 host load 回 GPU（「load it all or not at all」，`hiradix_cache.py:248`）；太小（< `load_back_threshold`）或超内存配额就跳过。
- **异步握手**：`writing_check`（`hiradix_cache.py:114`）和 `loading_check`（`hiradix_cache.py:136`）每个 batch 被 scheduler 调用（`scheduler.py:1365`），从 `ack_write_queue` / `ack_load_queue` 取完成通知并 `dec_lock_ref`。TP > 1 时 `writing_check` 用 `all_reduce(MIN)` 同步各 worker 的进度，保证所有 TP rank 对 radix tree 做一致更新（`hiradix_cache.py:124`）。

host 侧 buffer 有一套状态机 `MemoryStateInt`（`memory_pool.py:736`：IDLE→RESERVED→PROTECTED→SYNCED→BACKUP），仅在 debug 模式下校验状态迁移（`@synchronized(debug_only=True)`），防止「正在写/读的 host slot 被复用」。

### 4.6 ChunkCache：禁用 radix 时的退化路径

`ChunkCache`（`chunk_cache.py:22`）实现同样的接口，但 **不做任何前缀复用**：`match_prefix` 永远返回 `([], None)`（`chunk_cache.py:36`），`insert` 直接 `NotImplementedError`。它只在 `disable_radix_cache` + 开启 chunked prefill 时使用，作用是支撑 chunked prefill 把分块的 KV 槽位记到 `req.prefix_indices`（`chunk_cache.py:48`），请求结束就整段 free。选它通常是为了规避 radix cache 的开销/复杂度，或调试时排除前缀复用的影响。

## 5. 设计难点与权衡

1. **为什么前缀复用能省大量计算？** prefill 的计算量正比于序列长度（注意力是 O(n²)、其余是 O(n)）。多轮对话、few-shot、共享 system prompt 等场景，新请求和旧请求往往共享很长的前缀。命中前缀就意味着这段 KV 不必重算，直接拿现成的物理槽位接着算后面的 token。在高并发同前缀场景，吞吐可以翻数倍——这正是 RadixAttention 的核心收益。

2. **radix tree 而不是 hash 表**：hash 只能做「整条序列」精确命中，而 radix tree 天然支持 **最长公共前缀** 匹配，并且通过节点分裂让公共前缀只存一份 KV、自动支持「一个前缀派生多个分支」。代价是匹配/插入要逐边走、要处理分裂，逻辑比 hash 复杂。

3. **page_size 的两难**：`page_size=1` 逻辑最简单、前缀匹配最细（任意 token 边界都能复用），但 attention kernel 访存差；`page_size>1` 对 kernel 友好，但匹配必须 page 对齐（`match_prefix` 里 `page_aligned_len = len(key)//page_size*page_size`，`radix_cache.py:160`），即末尾不足一页的部分无法复用。`get_child_key_fn` 因此在两种模式下不同（`radix_cache.py:119`）。

4. **「插入时发现重复」的处理**：`match_prefix` 之后调用方已经为整个序列分配了槽位，但 insert 回 tree 时发现前缀已存在（KV 已有物理副本）。代码的处理是：以 tree 里已有的为准，把自己重复分配的那段 `free` 掉（`radix_cache.py:205`、`:233`）。理解这点对看懂 `cache_finished_req` 至关重要——否则会以为 KV 被泄漏或双重释放。

5. **引用计数 + LRU 的配合**：单靠 LRU 无法防止「驱逐了正在 forward 用的 KV」，所以加了 `lock_ref`。难点在于 lock 是 **沿路径到 root** 的：复用一条前缀，整条祖先链都必须锁住，否则祖先被驱逐会让子节点的 value 悬空。`inc/dec_lock_ref` 的 while 循环（`radix_cache.py:293`、`:307`）就是干这个。

6. **HiRadixCache 的异步一致性**：write/load 是后台线程异步做的，主线程必须在每个 batch 用 `writing_check/loading_check` 收割完成事件并解锁；TP 多卡还要 `all_reduce` 对齐进度。这是分层缓存最容易出 bug 的地方（见下一节）。

7. **slot 0 是 padding 专用**：`free_slots`/`free_pages` 都从 1 开始（`memory_pool.py:231`、`paged_allocator.py:318`），buffer 多申请了 `page_size` 行（`memory_pool.py:283`）。padded token 的 dummy 输出写到 slot 0，不会污染真实数据。改分配器时千万别把 0 也放进 free list。

## 6. 改代码 / 修 bug 指南

**想改 X → 看这里：**

- **改前缀匹配 / page 对齐规则** → `RadixCache.match_prefix`（`radix_cache.py:138`）、`_match_prefix_helper`（`:336`）、`key_match_fn` / `get_child_key_fn`（`:119`）。
- **改驱逐策略（不只是 LRU）** → `RadixCache.evict`（`radix_cache.py:263`）和 `TreeNode.__lt__`（`:73`，改比较键即可换排序依据）。
- **改 KV 物理布局 / 新增 KV pool 类型** → 继承 `KVCache`（`memory_pool.py:102`），实现 `get_kv_buffer` / `set_kv_buffer` / `get_flat_data` / `transfer`。参考 `MHATokenToKVPool`、`MLATokenToKVPool`。
- **改 paged 分配（如新 page_size 策略）** → `PagedTokenToKVPoolAllocator`（`paged_allocator.py:157`）及其三个 triton kernel（`alloc_extend_kernel:29`、`alloc_decode_kernel:116`）。
- **改分层缓存 / 写回策略** → `HiRadixCache`（`hiradix_cache.py:23`）+ `HiCacheController`（`managers/cache_controller.py:146`），改 `write_through_threshold` / `load_back_threshold`（`hiradix_cache.py:63`、`:66`）。
- **整体开关前缀复用** → scheduler 选择逻辑 `scheduler.py:498`；`--disable-radix-cache`、`--enable-hierarchical-cache` 等 server args。

**常见 bug 高发区：**

- **KV 泄漏 / 显存不断下降**：检查每个 `alloc` 是否有对应 `free`；`cache_finished_req` 里那两处 `free`（`radix_cache.py:205`、`:233`）的 slice 区间最易写错。可对照 `available_size()` + `evictable_size()` + `protected_size()` 的总和是否守恒。
- **驱逐了正在用的 KV → 读到脏数据 / 崩溃**：八成是 `inc_lock_ref` / `dec_lock_ref` 不配对（漏 inc、重复 dec），或者忘了 lock 是沿路径到 root 的。
- **page 不对齐 assert 失败**：`SGLANG_DEBUG_MEMORY_POOL=1` 会打开 `paged_allocator.py` 里一系列对齐断言（`:198`、`:225`、`:262`），调 paged 逻辑时务必先开它。
- **HiRadixCache 卡死 / KV 不一致**：`writing_check`/`loading_check`（`hiradix_cache.py:114`、`:136`）漏调，或 TP 下 `all_reduce` 进度不一致导致各 rank radix tree 分叉。
- **node 悬空（value 是 None 还被当成有效）**：HiRadixCache 里到处要先判 `node.evicted`（`radix_cache.py:65`）再用 `node.value`，漏判会 crash。

**调试入手点：**

- `RadixCache.pretty_print()`（`radix_cache.py:256`）打印整棵树（每节点 key 长度 + 前 10 个 token + `r=lock_ref`），是排查树结构/锁状态最直接的工具。
- 直接跑 `python -m sglang.srt.mem_cache.radix_cache`（文件末尾有 `__main__`，`radix_cache.py:499`）能单独玩 insert/match，不必起整个 server。
- `total_size()`（`radix_cache.py:260`）/ `evictable_size()` / `protected_size()` 三个数对账，能快速定位泄漏在「锁住的」还是「可驱逐的」部分。
- 看请求级别的复用情况，从 scheduler 调 `match_prefix` 的位置（`scheduler.py` 中 `PrefillAdder` 一带）打日志最有效；scheduler 层细节见 [调度器](12-scheduler.md)。

---

相关章节：[调度器](12-scheduler.md)、[术语表](02-glossary.md)。
