# DetokenizerManager 与输出回流

## 一句话职责

把 Scheduler 算出来的 **output token ids**，在一个独立子进程里**增量地**反查（detokenize）成人类可读的文本 —— 同时妥善处理 UTF-8 多字节字符被切成两半、stop string 的回切（trim）、增量发送 offset —— 然后把结果经 ZMQ 推回 `TokenizerManager`，唤醒每个请求挂着的 async generator，最终拼成 HTTP / SSE 响应。reasoning（思维链）、function-call / tool 的结构化解析则发生在更上层（OpenAI adapter），消费的是本阶段产出的纯文本。

本章覆盖三个相邻但职责分明的环节：

1. **Scheduler 侧 output processor** —— 每个 forward step 之后组装本步要往外吐的东西，并判定请求是否结束。
2. **DetokenizerManager 子进程** —— 真正做增量 detokenize。
3. **TokenizerManager 回流 + 上层解析** —— 把文本喂回 async generator，再在 OpenAI 层做 reasoning / tool 解析。

---

## 在端到端链路中的位置

```
客户端 → 分词(TokenizerManager) → 调度(Scheduler) → KV缓存 → 前向(ModelRunner) → 采样
                                       │                                              │
                                       │            next_token_ids (CPU list)         │
                                       ▼ ◀────────────────────────────────────────────┘
              ┌──────────────────────────────────────────────────┐
              │  Scheduler.process_batch_result_{prefill,decode}   │  ← 本章
              │  · req.output_ids.append(next_token_id)            │
              │  · req.check_finished()  (EOS / stop / max tokens) │
              │  · stream_output() → BatchTokenIDOut               │
              └───────────────────────┬──────────────────────────┘
                                      │ ZMQ PUSH (token ids)
                                      ▼
              ┌──────────────────────────────────────────────────┐
              │  DetokenizerManager (独立进程)                      │  ← 本章核心
              │  · 增量 detokenize (surr/read offset)              │
              │  · UTF-8 半字符处理 ("�")                          │
              │  · trim stop str / stop token                     │
              │  → BatchStrOut (text)                             │
              └───────────────────────┬──────────────────────────┘
                                      │ ZMQ PUSH (text)
                                      ▼
              ┌──────────────────────────────────────────────────┐
              │  TokenizerManager._handle_batch_output            │  ← 本章
              │  · state.text += chunk; state.event.set()         │
              │  → 唤醒 _wait_one_response 这个 async generator    │
              └───────────────────────┬──────────────────────────┘
                                      ▼
              ┌──────────────────────────────────────────────────┐
              │  OpenAI adapter: ReasoningParser / FunctionCallParser │ ← 本章
              └───────────────────────┬──────────────────────────┘
                                      ▼
                                   响应 (JSON / SSE)
```

`TokenizerManager`/`Scheduler` 的整体职责见 [TokenizerManager](11-tokenizer-manager.md) 与 [调度器](12-scheduler.md)。本章关注「token id → text → 结构化字段」这条出口管线。

---

## 关键文件与类

| 符号 | file:line | 作用 |
| --- | --- | --- |
| `DetokenizerManager` | `python/sglang/srt/managers/detokenizer_manager.py:68` | detokenize 子进程主体 |
| `DecodeStatus` | `python/sglang/srt/managers/detokenizer_manager.py:56` | 每个 rid 的增量解码游标（surr/read/sent offset） |
| `DetokenizerManager.handle_batch_token_id_out` | `python/sglang/srt/managers/detokenizer_manager.py:140` | 核心：批量增量 detokenize |
| `DetokenizerManager.trim_matched_stop` | `python/sglang/srt/managers/detokenizer_manager.py:113` | 把命中的 stop str / stop token 从输出里切掉 |
| `LimitedCapacityDict` | `python/sglang/srt/managers/detokenizer_manager.py:251` | 有界 LRU，超出 `DETOKENIZER_MAX_STATES` 淘汰最老 rid |
| `find_printable_text` | `python/sglang/utils.py:256` | 截出「可安全打印」的前缀（避免半个词/半个字） |
| `SchedulerOutputProcessorMixin` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:26` | Scheduler 的输出组装与结束判定 |
| `...process_batch_result_decode` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:185` | decode step 后处理 |
| `...stream_output_generation` | `python/sglang/srt/managers/scheduler_output_processor_mixin.py:457` | 组装 `BatchTokenIDOut` 并 PUSH 给 detokenizer |
| `Req.init_incremental_detokenize` | `python/sglang/srt/managers/schedule_batch.py:671` | 计算本请求要送去 detokenize 的 ids 与 read_offset |
| `Req.check_finished` | `python/sglang/srt/managers/schedule_batch.py:683` | 判定 EOS / stop token / stop str / max tokens |
| `TokenizerManager._handle_batch_output` | `python/sglang/srt/managers/tokenizer_manager.py:1119` | 接收 `BatchStrOut`，更新 state，set event |
| `TokenizerManager._wait_one_response` | `python/sglang/srt/managers/tokenizer_manager.py:632` | per-request 的 async generator |
| `ReasoningParser` | `python/sglang/srt/reasoning_parser.py:143` | `<think>...</think>` 思维链拆分 |
| `BaseReasoningFormatDetector.parse_streaming_increment` | `python/sglang/srt/reasoning_parser.py:47` | 流式增量解析 reasoning |
| `FunctionCallParser` | `python/sglang/srt/function_call/function_call_parser.py:18` | tool/function-call 解析入口 |
| `BaseFormatDetector.parse_streaming_increment` | `python/sglang/srt/function_call/base_format_detector.py:75` | 流式 partial-JSON tool 解析 |
| `Qwen25Detector` | `python/sglang/srt/function_call/qwen25_detector.py:15` | `<tool_call>{...}</tool_call>` 格式 |
| `CompletionTemplate` / FIM | `python/sglang/srt/code_completion_parser.py:36` | 代码补全 FIM 模板（输入侧，非输出侧） |

---

## 核心数据流 / 执行流程

### 1) Scheduler 侧：每步产出什么

每个 forward step 之后，`process_batch_result_decode`（`scheduler_output_processor_mixin.py:185`）对 batch 里每个 req 做三件事：

```python
# scheduler_output_processor_mixin.py:232-239
if batch.spec_algorithm.is_none():
    req.output_ids.append(next_token_id)   # 追加这一步采样到的 token
req.check_finished()                        # 判定是否结束
if req.finished():
    self.tree_cache.cache_finished_req(req)
```

随后调用 `stream_output` → `stream_output_generation`（`:457`）。注意这里**不做 detokenize**，只组装 token ids 与元信息。关键的「增量」逻辑在于：每个 req 维护若干 offset，只把「上次没送过」的部分打包：

```python
# scheduler_output_processor_mixin.py:541-552
decode_ids, read_offset = req.init_incremental_detokenize()
decode_ids_list.append(decode_ids[req.send_decode_id_offset :])  # 只发增量
req.send_decode_id_offset = len(decode_ids)
...
req.send_token_offset = len(req.output_ids)
```

`init_incremental_detokenize`（`schedule_batch.py:671`）返回的不是「全部 output ids」，而是从 `surr_offset` 起的一段尾巴，外加一个 `read_offset`：

```python
# schedule_batch.py:674-681
if first_iter:
    self.read_offset = len(self.origin_input_ids_unpadded)
    self.surr_offset = max(self.read_offset - INIT_INCREMENTAL_DETOKENIZATION_OFFSET, 0)  # 默认回看 5 个 token
all_ids = self.origin_input_ids_unpadded + self.output_ids
return all_ids[self.surr_offset :], self.read_offset - self.surr_offset
```

**为什么要回看 5 个 token（`INIT_INCREMENTAL_DETOKENIZATION_OFFSET = 5`，`schedule_batch.py:69`）？** 因为某些 tokenizer（特别是 BPE）解码时存在「上下文相关」：单独解码一个 token 得到的字符串，可能和它前面跟着几个 token 时不一样（典型是前导空格、合并字符）。多带几个「surrounding」token 做参照，能让增量出来的文本更稳定。

**什么时候才 stream（`should_output`）？** 见 `:512-529`：

- 请求 `finished()` → 必发一次（且用 `req.finished_output` 标志防止 overlap schedule 下重复发尾包，`:512-517`）。
- `req.stream=True` → 每 `stream_interval` 个 token 发一次。
- 非流式 → 每 `DEFAULT_FORCE_STREAM_INTERVAL = 50`（`:23`）个 token 也发一次（让 detokenizer 持续维护 `DecodeStatus`，避免最后一次性 decode 巨大序列）。

最终打包成 `BatchTokenIDOut` 经 `self.send_to_detokenizer.send_pyobj(...)`（`:650`）PUSH 出去。

### 2) DetokenizerManager：增量 detokenize 的核心

子进程是个极简事件循环（`detokenizer_manager.py:106-111`）：

```python
recv_obj = self.recv_from_scheduler.recv_pyobj()
output = self._request_dispatcher(recv_obj)   # 按类型分派
self.send_to_tokenizer.send_pyobj(output)
```

`handle_batch_token_id_out`（`:140`）是主菜。对每个 rid，先维护一份 `DecodeStatus`（`:56`）。第一次见到 rid 时新建，之后只 `extend` 新来的 decode_ids：

```python
# detokenizer_manager.py:155-157
s = self.decode_status[rid]
s.decode_ids.extend(recv_obj.decode_ids[i])   # 累积；scheduler 只发增量
```

然后构造两组 id 序列并各做一次 `batch_decode`：

```python
# detokenizer_manager.py:159-178
read_ids  = trim_matched_stop(s.decode_ids[s.surr_offset:], ...)   # surr_offset 起到结尾
surr_ids  = s.decode_ids[s.surr_offset : s.read_offset]            # surr_offset 到 read_offset
surr_texts = tokenizer.batch_decode(surr_ids, ...)
read_texts = tokenizer.batch_decode(read_ids, ...)
```

`new_text = read_texts[i][len(surr_texts[i]):]` 就是「这一段新增 token 相对于参照前缀多出来的文本」。这就是增量的本质：**用一段公共前缀（surr）去抵消 tokenizer 的上下文相关性，相减得到真正的新文本。**

```
decode_ids:  [ ......... | surr | new tokens ........ ]
                          ^      ^                    ^
                       surr_off read_off          末尾
             surr_text = decode(surr_off .. read_off)
             read_text = decode(surr_off .. 末尾)
             new_text  = read_text[len(surr_text):]   ← 新增文本
```

#### UTF-8 半字符处理（最容易踩的坑）

一个汉字/emoji 往往跨多个 token；解码到一半时，`batch_decode` 会吐出替换字符 `"�"`（U+FFFD）。如果直接把它送给用户，流式里会闪现乱码。处理逻辑在 `:195-203`：

```python
if recv_obj.finished_reasons[i] is None:        # 还没结束 → 流式 chunk
    if len(new_text) > 0 and not new_text.endswith("�"):
        # 文本完整：提交，推进 offset 窗口
        s.decoded_text += new_text
        s.surr_offset = s.read_offset
        s.read_offset = len(s.decode_ids)
        new_text = ""
    else:
        # 末尾是半个字符 → 不提交、不推进 offset，只截出能安全打印的部分
        new_text = find_printable_text(new_text)
```

要点：

- **末尾是 `"�"` 就不推进 offset**。下一步更多 token 进来后，同样的 `surr_offset` 会把这半个字符连同后续 token 一起重新解码，凑成完整字符。这是「等下一步补全」的机制。
- `find_printable_text`（`utils.py:256`）进一步保守：即便没有 `"�"`，也会把结尾不完整的「词」截掉（在最后一个空格处切），CJK 字符则特判放行（`:264-268`）。它的作用是**先行预览**已经确定的部分，而真正的 offset 推进只在文本干净时发生。

#### 增量发送 offset（`sent_offset`）

DetokenizerManager 自己也要做增量发送，避免每步把 `decoded_text` 全量重发：

```python
# detokenizer_manager.py:205-213
output_str = trim_matched_stop(s.decoded_text + new_text, finished_reason, no_stop_trim)
incremental_output = output_str[s.sent_offset:]   # 只发没发过的尾巴
s.sent_offset = len(output_str)
output_strs.append(incremental_output)
```

#### trim stop（回切停止串）

`trim_matched_stop`（`:113`）把命中的 stop 内容从输出里删掉：stop **string** 用 `output.find(matched)` 截断；stop **token** 直接 `output[:-1]` 去尾。`no_stop_trim=True` 或没有 `matched` 时原样返回。结束判定本身在 scheduler 侧（见下），detokenizer 只负责按 `finished_reason["matched"]` 做文本/ token 层面的清理。

最终打包 `BatchStrOut`（`:215`），把 `output_strs`（增量文本）和一大票 logprob / hidden_states 透传字段一起 PUSH 给 TokenizerManager。

### 3) 结束判定：`Req.check_finished`

`schedule_batch.py:683` 按优先级判定结束原因：

1. `to_abort` → `FINISH_ABORT`。
2. `len(output_ids) >= max_new_tokens` → `FINISH_LENGTH`（`:693`）。
3. grammar 终止 → `FINISH_MATCHED_TOKEN`（`:699`）。
4. EOS / stop_token_ids / tokenizer.eos / additional_stop_token_ids → `FINISH_MATCHED_TOKEN`（`:704-722`，`ignore_eos` 可跳过）。
5. stop strings → 把尾巴 decode 出来 `tail_str`，在 `tail_str` 或已有的 `decoded_text` 里找子串 → `FINISH_MATCHED_STR`（`:724-733`）。

注意第 5 步 stop str 判定**必须自己 decode 一小段尾巴**（`output_ids[-(stop_str_max_len + 1):]`），因为 scheduler 进程此刻还没有 detokenizer 给的完整文本 —— 这是两个进程职责分离带来的一个小冗余。

### 4) 回流到 TokenizerManager + async generator

`_handle_batch_output`（`tokenizer_manager.py:1119`）按 rid 找到 `ReqState`，对 `BatchStrOut` 做累加并唤醒：

```python
# tokenizer_manager.py:1163-1209
if isinstance(recv_obj, BatchStrOut):
    state.text += recv_obj.output_strs[i]       # 累加增量文本
    out_dict = {"text": state.text, "meta_info": meta_info}
...
state.finished = recv_obj.finished_reasons[i] is not None
if state.finished:
    del self.rid_to_state[rid]
state.out_list.append(out_dict)
state.event.set()                                # 唤醒 generator
```

`_wait_one_response`（`:632`）是 per-request 的 async generator：`await state.event.wait()` → 取 `state.out_list[-1]` → `yield` → `state.event.clear()` 进入下一轮。结束时再 yield 一次并 `break`（`:674-680`）。`state.text` 始终是「到目前为止的全量文本」，流式增量靠上层做差。

### 5) reasoning / function-call / tool 解析（上层）

这些解析**不在 detokenizer 里**，而在 OpenAI adapter（`openai_api/adapter.py`）消费 `state.text` 时进行。

**ReasoningParser**（`reasoning_parser.py:143`）把 `<think>…</think>` 拆成 `reasoning_text` 与 `normal_text`：

- 非流式 `parse_non_stream`（`:169`）→ `detect_and_parse`：以 `</think>` split（`:41-45`）。
- 流式 `parse_stream_chunk`（`:174`）→ `parse_streaming_increment`（`:47`）：内部维护 `_buffer`，遇到 `<think>` 先 strip（`:61-63`），遇到 `</think>` 把前面归 reasoning、后面归 normal（`:66-77`）；`stream_reasoning=False` 时会先攒着不吐 reasoning。adapter 里非流式默认用 `stream_reasoning=False`（`adapter.py:1308-1311`），流式则每个 index 维护一个 parser（`adapter.py:1553-1559`）。

**FunctionCallParser**（`function_call_parser.py:18`）按 `tool_call_parser` 选 detector（`llama3 / qwen25 / mistral / deepseekv3 / pythonic`，`:27-33`）。流式解析最微妙：`BaseFormatDetector.parse_streaming_increment`（`base_format_detector.py:75`）用 **partial-JSON** 解析未闭合的 arguments，并通过 `streamed_args_for_tool` 记录「已经吐过的 JSON 前缀」，每次只 emit `argument_diff`（`:151-165`、`:198-231`），从而把一个还没写完的 `{"city": "北` 也能增量推给客户端。tool name 只发一次（`current_tool_name_sent`，`:177-191`）；遇到非法 tool name 会 reset 整个状态（`:118-125`）。非流式 `parse_non_stream`（`:59`）一把抓：`detect_and_parse` 用正则切出 `<tool_call>…</tool_call>`（如 `qwen25_detector.py:46-52`）。

`tool_choice="required"` / 指定函数时，还会通过 `get_structure_constraint` / `get_ebnf`（`function_call_parser.py:136-175`）下发 EBNF/structural-tag 约束到采样阶段，强制模型只能产出合法工具调用 —— 这是「解析」反向影响「生成」的一环。

**code_completion_parser.py** 是个容易误会的文件：它处理的是 **FIM（fill-in-the-middle）输入模板**（`generate_completion_prompt`，`:129`），即把 prompt/suffix 包进 `<fim_begin>…<fim_middle>…` 拼成模型输入，属于**入口侧**，与输出回流没有直接关系，放这里是因为它常和「代码补全」输出一起被提到。

---

## 设计难点与权衡

1. **为什么 detokenize 要单独开一个进程？** detokenize 是纯 CPU 工作（tokenizer decode），且对每个 token step 都要做。放在 Scheduler 主循环里会和 GPU 调度抢 CPU、增加 step 延迟；放在 TokenizerManager（asyncio 单线程）里又会阻塞事件循环。独立进程 + ZMQ PUSH/PULL 把它从关键路径里摘出来，三方解耦。代价是：结束判定（stop str）需要的文本，scheduler 拿不到，只能自己 decode 一小段（`check_finished` 第 5 步）。

2. **surr/read 双 offset 而非「逐 token decode」。** 单 token 解码在 BPE 下不可靠（上下文相关、前导空格）。双 offset 用一段稳定前缀做差，是 vLLM 沿用过来的经典做法（代码注释直接引用了 vLLM detokenizer，`schedule_batch.py:670`）。`INIT_INCREMENTAL_DETOKENIZATION_OFFSET=5` 是个经验值。

3. **「不推进 offset」实现 UTF-8 半字符等待。** 没有显式的「byte buffer」，而是靠「末尾 `"�"` 就不动 surr/read offset、下一步连旧 token 重解一遍」来天然补全。简洁但隐式 —— 改这块时要意识到 offset 推进和文本提交是绑定的。

4. **三层各自的增量 offset。** scheduler 有 `send_decode_id_offset`/`send_token_offset`；detokenizer 有 `surr/read/sent_offset`；tokenizer_manager 有 `last_output_offset`。每层都「只传我新算出来的」，层层做差。理解 bug 时要分清是哪一层的 offset 算错了。

5. **`LimitedCapacityDict` 的有界状态。** detokenizer 给每个在飞请求存 `DecodeStatus`，上限 `DETOKENIZER_MAX_STATES`（默认 `1<<16`）。超了会 LRU 淘汰最老的 rid，之后该 rid 再来就 `KeyError`，抛出那条很长的报错（`:185-193`，指向 issue #2812）。高并发长尾场景下这是真实可能触发的。

6. **overlap schedule 下的重复输出防护。** 因为 overlap 会多算一个 delayed token，一个 finished 请求可能进入 `stream_output` 两次；靠 `req.finished_output` 标志只发一次尾包（`scheduler_output_processor_mixin.py:512-517`）。spec decoding 下 `next_token_ids` 与 `reqs` 长度不一致，多处用 `batch.spec_algorithm.is_none()` 绕开（`:232`、`:241`、`:265`）。

---

## 改代码 / 修 bug 指南

「想改 X → 看这里」：

- **流式输出乱码 / 末尾闪现 `�`** → `detokenizer_manager.py:194-203` 的 `"�"` 分支 + `find_printable_text`（`utils.py:256`）。先确认 `new_text.endswith("�")` 是否如期为 True；CJK 全量放行逻辑在 `find_printable_text`。
- **增量文本重复 / 丢字 / 错位** → 三层 offset：scheduler `send_decode_id_offset`（`scheduler_output_processor_mixin.py:546-548`）、detokenizer `surr/read/sent_offset`（`detokenizer_manager.py:199-212`）、tokenizer_manager `state.text` 累加（`tokenizer_manager.py:1164`）。逐层打印 offset 与 `len`。
- **stop string 没被切掉 / 把正文也切了** → `trim_matched_stop`（`detokenizer_manager.py:113`）+ `check_finished` 第 5 步（`schedule_batch.py:724-733`）。注意 `no_stop_trim` 与「多个 stop str 同时命中」尚未处理（`:123` 的 TODO）。
- **请求该结束没结束 / 提前结束** → `check_finished`（`schedule_batch.py:683`），按优先级核对 max_new_tokens / EOS / stop_token_ids / grammar。
- **新增一种 reasoning 格式（非 `<think>`）** → 在 `reasoning_parser.py` 加一个 `BaseReasoningFormatDetector` 子类并注册进 `DetectorMap`（`:154-157`）。
- **新增一种 tool-call 格式** → 在 `function_call/` 加 detector（实现 `detect_and_parse` / `parse_streaming_increment` / `has_tool_call` / `structure_info` / `build_ebnf`），注册进 `FunctionCallParser.ToolCallParserEnum`（`function_call_parser.py:27-33`）。流式逻辑大概率能复用 `BaseFormatDetector`（`base_format_detector.py:75`），主要改 `bot_token`/`eot_token` 与正则。
- **高并发下偶发 “Decode status not found”** → 调大环境变量 `SGLANG_DETOKENIZER_MAX_STATES`（`detokenizer_manager.py:53`、报错在 `:185-193`）。
- **代码补全 FIM 拼接不对** → 是输入侧，看 `code_completion_parser.py:129` 的 `generate_completion_prompt` 与模板注册（`:145-174`），与本章其余输出逻辑无关。

**调试入手点：**

- detokenizer 进程标题是 `sglang::detokenizer`（`detokenizer_manager.py:269`），崩溃会向父进程发 `SIGQUIT`（`:279`），所以「整个 server 突然退出」时要看它的日志。
- 想看每步往外发了什么文本：在 `handle_batch_token_id_out` 末尾、`BatchStrOut` 构造处（`:215`）打印 `output_strs`。
- 想看 scheduler 决定 stream 哪些请求：`stream_output_generation` 的 `should_output` 分支（`scheduler_output_processor_mixin.py:512-529`）。
- 想看上层如何把 `state.text` 转成 SSE：`tokenizer_manager.py:_wait_one_response`（`:632`）+ `openai_api/adapter.py` 中的 `parse_stream_chunk` 调用（`:1559`、`:1593`）。

相关章节：[TokenizerManager](11-tokenizer-manager.md)、[调度器](12-scheduler.md)、[入口与 API](10-entrypoints-api.md)、[术语表](02-glossary.md)。
