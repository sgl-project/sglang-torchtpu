# GLM-5.2 native MTP baseline 执行记录

当前状态：`exp-6mtkc6t2qi` 已完成 8-chip native MTP E2E 基线。两 rank 均为 `SUCCEEDED`，16-DP routed smoke、两组 workload warmup 和六轮正式测量全部完成。同一 allocation / application generation 1 / client attempt 1 跑完环境、加载、编译和全部请求，未重启或重试。Decode 使用用户授权的 shared prefix。以下是实际 TPU 结果；后半部分保留此前排障历史。

## 用户确认的基线与 #2198 验证边界

用户已确认将本次结果作为后续对照基线。固定实验 `exp-6mtkc6t2qi` 和实际执行源码 `811865ca48ca23aa14dc74c9dbebaf5b380279dc`；正式基准数字采用 SSE 实测：prefill TTFT 6302.24 ms、decode TPOT 18.849 ms、shared-prefix decode 3511.86 token/s（438.98/物理 chip）。Step-time 估算3520.73 token/s作为交叉核对，不替换实测值。

以下加载问题定位于 #2198 原 head `c2861ccb90f7b970218057c981eb30bab5bbef01`。本次发现两项会明显拖慢启动的加载路径缺口：

1. **Target 未启用 EP-local expert reads**：GLM target 未声明该能力，原 helper 返回 `None`，回退读取整 shard，再由正常 loader 过滤 experts。确认的是冗余 I/O 和 host 内存压力，没有证据说明该 fallback 读错 expert/scale。独立修复 `1879b469580f6a78fb8356c53c526a0dc79dec63` 精确启用 GLM target，保留全部 runtime guards；原回归 `None != (48,64)`，修复后19个 CPU tests和两个真实 shard FP8 weight/F32 scale byte oracles通过。
2. **MTP draft 重读整个 checkpoint**：`DeepseekWeightLoaderMixin.do_load_weights` 取得 tensor 后才跳过非 MTP layer。观察到282个 shards约14分钟只扫57个；不是已证实的死锁或权重映射错误。基线通过原始5个 shard的 symlink和仅含1569个MTP tensor的 index缩小读取范围；没有重写或重新量化权重。

上述 reader、GLM model、DeepSeek common loader和NextN文件相对 #2198 的 PR base `55f642b178a602689b852a0ecb51667310d4de29` 均 byte-for-byte 不变，因此应记为当前组合的既有启动性能缺口，不能归因为本 PR 新增回归。**成功基线包含 reader 修复与 MTP-only view，不是原样 #2198 head 的完整 E2E 通过。** #2198 后续已纳入配套改动：`023a32a934` 为 reader 修复，`7688ab8c2a` 修复 Python 3.10 的两处 `enterContext()`，`ed78e85c42` 在通用 loader 中按原始 index 自动筛选 NextN shards；不再要求手工 view。新 head `ed78e85c42d4dcbfbd2f5caadc607909b682d36a` 已转为 ready，Python 3.10.20 / Torch 2.14.1 的104个相关 CPU tests及pre-commit通过。新的自动筛选通过真实 index 核对恰为原始270–274五个shards、1569个MTP tensors；新 head 尚未重跑完整8-chip GLM E2E。

本次没有定位到新增 channelwise FP8 adapter 的错误权重字节或 scale 错配；但选定 shard的byte oracle和16-DP算术smoke不构成全模型数值等价验证。venv临时目录ENOSPC、runner缺tar、HTTP keep-alive/client阻塞属于本轮环境或benchmark脚本问题；服务期编译和清理时Gloo异常也未证实由 #2198 新增逻辑引起。

## 最终实测结果（2026-11-16）

| 场景 | 轮次 | TTFT 中位数 ms | Client TPOT 中位数 ms | 稳态 output token/s（总计） | output token/s / 物理 chip |
| --- | ---: | ---: | ---: | ---: | ---: |
| C16：49,152 cached + 4,096 extend，输出16 | 1 | 6327.23 | 17.57 | — | — |
| 同上 | 2 | 6301.60 | 20.00 | — | — |
| 同上 | 3 | 6302.24 | 20.28 | — | — |
| C64 shared：90,000 input，输出1,024 | 1 | 5234.92 | 18.85 | 3511.86 | 438.98 |
| 同上 | 2 | 5469.18 | 18.23 | 3628.18 | 453.52 |
| 同上 | 3 | 5640.27 | 19.39 | 3368.34 | 421.04 |

- 取三轮各自中位数的中位数：prefill TTFT **6302.24 ms**；decode TTFT **5469.18 ms**、TPOT **18.849 ms**。Decode 稳态吞吐中位数 **3511.86 output token/s**，按 **8 个物理 chips** 归一化为 **438.98 token/s/chip**；三轮范围 421.04–453.52。不是按16个逻辑 devices 归一化，也不是把请求平均 TPOT 倒数当作总吞吐。
- TTFT 是 client 发出 HTTP 请求至首个有输出的 SSE event，包含 HTTP、调度和 prefill。TPOT 是首末输出 event 的间隔除以 `completion_tokens - 1`，不是 TPU 单步时间。前置 prefix priming、workload round0 warmup 和启动编译不计入六轮正式结果。
- C64 稳态窗口取所有请求达到64个输出后、任何请求达到960个输出前的公共区间；三轮分别 **14.090 / 13.374 / 14.441 秒**。每个窗口内所有 `/v1/loads` 采样均通过16 DP、每 DP 4条 running requests 检查。整个轮次的满并发采样数分别707 / 732 / 768；C16分别23 / 24 / 30。吞吐按公共窗口内实际 SSE 累积 token 增量计算，剔除了尾部批量下降阶段。

### 根据 MTP step time 估算吞吐（后续核对）

通过 Falcon CPU analyzer `an-dink13hkrq` 读取同一正式三轮的完整 `*-loads.json`，在每轮 C64 公共满载窗口内，对每个 DP 的累计 `decode_moments` 做首末差分。源码 `scheduler.py::_accumulate_decode_moment` 的六项为 `n, sum_b, sum_step_us, sum_b2, sum_b_step_us, sum_generated`。16 DP 的差分全部满足 `sum_b=4*n`、`sum_b2=16*n`，因此仅覆盖每 DP 4条请求的满载 step。

| 轮次 | MTP 整轮 step ms | 每请求每 step 实际输出 tokens | 估算总 output token/s | 估算 token/s/物理 chip | SSE 实测总 token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 95.115 | 5.2324 | 3520.73 | 440.09 | 3511.86 |
| 2 | 91.732 | 5.2088 | 3634.07 | 454.26 | 3628.18 |
| 3 | 97.301 | 5.1246 | 3370.72 | 421.34 | 3368.34 |

计算为 `64 × 实际输出tokens/请求/step ÷ step秒`，再除以8个物理 chips。三轮吞吐估算的中位数 **3520.73 token/s**，即 **440.09 token/s/chip**，与 SSE 实测中位数3511.86/438.98相符。不是假定每轮固定接受6 tokens，也没有再乘 TP16。

这里 step time 是相邻 decode launch 的调度 wall-clock cadence，覆盖 native MTP draft/verify 整轮执行节奏；不是独立 device profiler 的 target-forward 时间。满载窗口的平均输出长度来自同窗口 `sum_generated / sum_b`，与整请求的 `completion_tokens / verify_ct` 范围不同。保留逐 DP 差分和原始边界时间：[step audit](glm52-e2e-baseline-step-audit-20261116.json)。没有新增 TPU 实验。

### Cache、MTP 与验证范围

- 16 DP 的真实 KV 容量均为 **219,008 tokens/DP**，FP8 `torch.float8_e4m3fn`；target KV 11.282 GB/DP。4条独立90K请求含输出和 speculative/page 余量的 slots 下限为 **364,288/DP**，因此选择 shared prefix；没有降低 C64、90K 输入、1,024 输出或 MTP 配置。
- Decode 的89,984-token相同 prefix 先在每个 DP 各 prime 一次；每 DP 内4条请求共享，后缀16 tokens各自不同。Prefix KV 仍在不同 DP 上分别驻留。该结果不证明4条独立90K prefix/DP能装入 KV。
- C16 的48条正式请求全部 `prompt_tokens=53,248`、`completion_tokens=16`，实际 cached 为49,088–49,152；C64 的192条正式请求全部 `prompt_tokens=90,000`、`completion_tokens=1,024`，实际 cached 为89,920–89,984。最多重算末尾一个64-token page，已计入实测 TTFT。
- 全部368条请求的 DP receipt 与指定 rank 相符，`num_retractions=0`，没有 abort。16条 chat smoke 覆盖 DP0–15，使用 checkpoint chat template，输出均以 `4` 开头。Smoke 强制16输出并 `ignore_eos=True`，答案后的续写不用于指令遵循/模型质量结论；这不是广泛准确率验收。
- 三轮 decode native counters：verify **38,097**，correct drafts **158,732**，proposed drafts **190,485**；接受率 `correct/proposed` = **83.330%**，三轮分别84.235% / 83.450% / 82.324%。按 native 输出口径 `196608 / verify` = **5.161 output tokens/verify**。计数器可能包含最后一次 verify 在输出长度裁剪前的 accepted run，不把它等同于逐 token 无裁剪计数。所有正式请求均有非零 MTP verify。
- 输入为排序后的真实 UTF-8 `python/sglang/*.py` source 经 checkpoint tokenizer 编码并冻结，未循环填充；1,142,656个 token IDs，SHA256 `607d4dc372753216266489042597330f2fc66c94b5e0fd6a760181a10162b3c6`。这是自然代码输入的缓存性能口径。

### 运行身份和服务期间编译

- 实验：`exp-6mtkc6t2qi` / `job-agyoxxt6mj` / `art-tfqylgfhvw`；1-prod，8 physical v7x chips / 2 hosts / 16 visible devices。Running 12:02:21.833 UTC，Falcon terminal success 12:38:56.976 UTC。原始 artifact：[Falcon view](https://falcon.infiscale-infra.org/v1/experiments/exp-6mtkc6t2qi/artifacts/art-tfqylgfhvw/view)。相对根目录：`application-attempts/1/glm52-e2e/`，正式数据在 `measurements/attempt-1/`。
- 实际源码 `811865ca48ca23aa14dc74c9dbebaf5b380279dc`，两 host benchmark source SHA256 均为 `a77e8a33f1edcc6140619e7f729bbcb8f7fdf3630a4965cfeeda3906ec9369d3`；client SHA256 `443049db573c97aa1a86ad3401ef564a97046836247e078839782d9d389bd598`。模型/reader/依赖相对 reader fix `1879b469580f6a78fb8356c53c526a0dc79dec63` 未变。两 rank checkpoint config/index/tokenizer_config SHA一致，与下文模型身份一致。
- 每 host 9份真实 serving-process runtime 文件（launcher + 8 workers），共18份；另两份 validation probe 文件未混入 serving 证据。全部实际解释器 `/tmp/glm52-baseline-runtime/venv/bin/python`，均记录实际 imports 和已映射 `libtpu.so`：Torch `2.11.0+cpu`、torchvision `0.26.0+cpu`、torch_tpu `0.2.1.dev20261028201649`、JAX/JAXLIB `0.11.2`、libtpu `0.0.50`、tpu-sync-torch `0.0.1.dev20261112081648`。
- 两 rank resolved args 均为 TP16 / attention-DP16 / EP16、attention backend `pallas_tpu_mla_sparse`、MoE `pallas_tpu`、quantization `fp8`、FP8 KV、每 DP chunk2048、DP lm_head=true。Native `NEXTN` 规范化为 `EAGLE`，5 steps / topk1 / 6 draft tokens。未改动模型、并行或 MTP 参数。
- 16 DP 在12:33:43完成 native warmup 并设置 Torch `fail_on_recompile`。**这不等于 XLA 服务期零编译**：其后日志记录128个新 XLA programs，每个25–46 ms；跨 ranks累计4029 ms，最大单条日志的 program 组222 ms，不能将跨 rank 累计当作 wall-clock 停顿。另有 persistent-cache loads，其加载耗时未测。
- 按 server 秒级时间戳与 request wall/monotonic 时间对齐（窗口边界保守扩1秒）：prefill三轮未见新 compile；decode round1尾部记录27个 programs（跨 ranks累计804 ms），round2 TTFT阶段14个（432 ms），round3无。三个 C64公共满载窗口内均未见新 XLA compile 记录。端到端 TTFT/TPOT 保留这些真实开销，未事后扣除；不宣称整个 serving 过程零编译。
- 最后一个测量 request 于12:38:30.455完成。12:38:33后 supervisor 退出清理触发 SIGTERM；rank1 在12:38:37–38出现 Gloo `Connection closed by peer` 和 scheduler teardown traceback。异常晚于全部请求完成，两个应用结果均 `SUCCEEDED`；本轮不证明无异常的 graceful shutdown。

### 取证与交付

- 通过 Falcon CLI 离线 CPU analyzer `an-a841nyzdcg` 成功取回声明 outputs，核对原始 summary、368份 request metadata、两 rank results/provenance、18份 serving runtime、完整日志匹配与 load audit；未为取证另起 TPU。前一个显式 TPU image 的 analyzer `an-sj4ul6hzue` 未成功启动，改用 Falcon 默认 analyzer image 后完成；不计作模型失败或硬件实验。
- 收集器的 `no-client-attention=False` 仅因成功路径也会保留 `client-attention.json`。实际内容是 **RESOLVED / attempt=1**，源码成功路径明确写入该 receipt；没有 client retry。该文件存在不能解释为 client 失败。
- 可读汇总及完整数字、provenance、逐条服务期编译和结果文件 SHA256：[glm52-e2e-baseline-results-20261116.json](glm52-e2e-baseline-results-20261116.json)。原始 streams、load snapshots、输入 manifest 和 source hashes 保留在上述 Falcon artifact；CPU fixtures、此前 holding 和 native warmup 均未计入正式结果。
- 本次正式基线已完成；停止该任务的定时跟进，不再为消除上述编译或退出清理日志自动启动实验。

## 固定目标

- 8 physical v7x chips / 2 hosts / 16 visible devices，TP16 / attention-DP16 / EP16。
- 用户的离线 channelwise FP8 checkpoint；Torch W8A8；FP8 KV；native NEXTN MTP 5 steps / 6 draft tokens。
- Prefill：C16，49,152 cached prefix + 4,096 extend，输出16，每 DP chunk 2,048。
- Decode：C64，90,000 input + 1,024 output；优先 independent prefix；容量不足时按用户本轮授权使用 shared prefix，单独标明实际缓存和并发。

## 机器与当前实验

- 1-prod 已实际获配 8-chip `2x2x2` slice。
- 首轮 allocation（已自然到期）：`exp-jjzf5biomm` / `job-ctbj0kddm6` / `art-tuqtvn2txw`。
- 主 holding command 有界；holding 本身的成功不等于模型成功。
- 后续不查询或使用已经退出的 antgroup / antgroup-v2。

## 与周末成功任务的对照

参考 `exp-k0nnadpb6q`（20/20 chat checks、16 DP receipts）和 `exp-r188nvdf96`（12K native long requests）。它们均为 8 chips、相同模型、未开启 MTP，不能替代本轮 MTP 和目标长上下文验收。

1. 镜像 tag、torch_tpu wheel SHA、模型 config/index SHA 相同；历史 image tag 不构成 immutable image identity。
2. 当前 declared TPU stack 与旧执行记录相比更新了 tpu-sync-torch；本轮遵从当前 source 声明，不降级。
3. `/tmp/tpu_logs` 实测仅 1 GiB tmpfs。最初 bootstrap 错把 venv 放在该目录，安装报 ENOSPC；周末成功的 venv 位于普通 `/tmp/glm52-peer-rank.../venv`。本轮修复为 `/tmp/glm52-baseline-runtime/venv`、同根 TMPDIR、关闭 pip 下载缓存；两机安装和声明版本核对已通过。
4. 周末 source 对 GLM target 有 EP-local reader opt-in，#2198 head 没有；本轮整 shard 加载观测明显缓慢。已在 plugin 内恢复精确 target-class 判断，保留全部 runtime guards，NextN、subclass 和其他模型行为不扩展。
5. 沿用周末的物理 host rank 查询、每 host checkpoint read preparation 和双 host rendezvous。

6. 本轮日志证实 target 在 37.78–43.54 秒完成加载；随后 draft 第二次扫描 282 个 shard，约14分钟只到57个。源码 `DeepseekWeightLoaderMixin.do_load_weights` 在读取 tensor 后才跳过非第78层。已保留原始日志并停止该 child server，allocation 保留；改为 metadata index + 原 shard symlink 的 MTP-only 视图，包含第78层完整1,569个 tensor、原始5个 shard，embedding/head 仍由原生路径共享 target。未转换权重或调整 MTP 参数。

## 代码身份与验证

- 用户 PR #2198 head：`c2861ccb90f7b970218057c981eb30bab5bbef01`；未改写其分支。
- 独立分支：`codex/glm52-e2e-baseline-20261116`。
- reader fix：`1879b469580f6a78fb8356c53c526a0dc79dec63`，已推送。
- 新增 regression 在原代码复现 `None != (48, 64)`；修复后 reader 文件 19 个 CPU tests 全部通过。
- 两个真实 shard 的 byte oracle 通过：expert [0,16) 与 [240,256)，涵盖 FP8 weights 和 F32 channel scales；分别验证 gap=0 与现有 gap=16MiB，未改变权重 bytes。
- Native CLI 已解析验证：每 DP chunk=2,048，local compile BS=[1,2,3,4]，NEXTN 被原生 resolver 规范化为 EAGLE，steps=5、draft tokens=6、draft quantization=fp8、DP lm_head 开启。
- Client transport 测试已验证 16 routed HTTP requests、fragmented SSE、pause/resume ingress 和 completion/MTP metadata；这是 transport 验证，不是模型通过证据。

## 前轮模型启动证据（历史）

- MTP-only 视图运行：`session-readerfix-1879-mtp-view`，implementation `1879b469580f6a78fb8356c53c526a0dc79dec63`，bundle SHA256 `22a3cbc8be05e5e8cd23bf1ea976d5184c874e6edc87f469215da4a62f52bb9f`。
- target 本轮 host0 各 DP 加载29.63–42.39秒；draft 5个 shard 的 iterator 232秒完成，TARGET_VERIFY BS1/2/3/4全部完成，分别181.6/182.6/195.9/197.2秒；BS4覆盖24个verify rows。EXTEND(FULL)的64-token无/有spec info两档分别194.4/58.3秒完成，2K无spec info档494.3秒完成，其余档也已完成；target共4个verify及4个extend shapes通过，正在native EAGLE FUSED DRAFT chain预编译。
- 两个 host 日志16 DP target/draft KV 容量219,008 tokens，FP8；4条独立90K请求的 page-rounded slots需求364,288，容量不足。正式 client 将再查询完整16 DP 容量，按用户授权选择 shared prefix，C64和90K输入不降低。
- 可复现 harness commit：`fa6269a0b1`（只新增 benchmark 脚本，模型/reader 代码与运行中的1879一致）；正式性能结果尚未产生。

## HTTP client 回归与本轮后续

- 10:53:50 全部target/draft图、global warmup及原生多DP生成warmup完成；16 DP启用 `fail_on_recompile`。原生decode日志有native MTP accept len 2–6。它不代替chat正确性和目标性能验收。
- 首个显式routed chat请求在client报 `ServerDisconnectedError`；两机无kernel/compiler异常，随后supervisor按client失败清理server。
- 已复现并修复：`/server_info`后同步import/初始化tokenizer阻塞event loop约7秒，超过server默认5秒HTTP keep-alive，aiohttp复用尚未处理EOF的连接。将import、初始化、输入编码放入 `asyncio.to_thread`。
- 最小HTTP复现：blocking=ServerDisconnectedError，to_thread=200；实际client.main回归：旧代码ServerDisconnectedError，新代码16模拟DP smoke全部PASS。这是transport验证，不是模型质量验证。
- client修复与MTP整shard host预读commit：`6bf809456f`。模型实现仍为1879，未修改kernel/策略；新bundle `6ebfe7b98972fc31cb4d0f7c393ff7f32b37e90a50d6ef563461f780a37af259`，run id `readerfix-1879-clientfix-v2`。

## 续跑资源与固定源码

- 旧allocation于11:10 UTC自然达到6,800秒holding上限，exit124；非模型错误，也未把holding误记为成功。clientfix-v2尚未完成验收。
- 已重新核查1-prod资源快照（包含running/pending allocations与active leases），只使用1-prod候选；未查询退出集群。
- 固定源码 `d8f342b878`（模型/reader/依赖相对1879无变化；含已验证的client修复及MTP host预读），续跑 `exp-7eqnrujaw6` / `job-dgowhe71v1` / `art-yi2osnr97j`。
- 续跑直接执行已提交的falcon_runner，不以holding作为application；挂载原25GB compile cache到 `/glm52-prior-compile-cache`，缓存不替代实际warmup/serve验收。
- 11:11–11:16 UTC多次admission快照仍为PENDING，ResourceReady=False，无法容纳完整2x2x2 slice；control-plane pending workloads也已核查。
- 2-prod候选已只读核查：单host4-chip pools、已占用64-chip pool，以及空闲大拓扑；没有已验证的完整8-chip route，且runtime=k8s_job与已验证1-prod的jobset不同。未据空闲节点数断言可调度，未试投第二份fallback。

## 模型身份

- checkpoint：`inference-model-storage-poc-tpu-hns/GLM5.2-fp8-channel-wise`，只读挂载 `/models/GLM5.2-fp8-channel-wise`。
- config SHA256：`0735624bbd5666283a1f135c59ff4c15e8b51c56fc50e022181525253f29463b`。
- index SHA256：`0ee9d61ce4e0ca912c2a5b8220f57b1060fb95ec09304da77c85a43d56aeb2a0`。
- tokenizer_config SHA256：`98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc`。
- 78 target layers，index 包含第78层 MTP；282 indexed shards。
- 抽查7个 header：target/MTP FP8 weights、[N] F32 scales、BF16 bridge/head/norm；保留实际 quantization_config 并补入 `weight_scale_granularity=channel`。

## 此前失败和当时待办（历史）

- `exp-wgzfz1wtp6`、`exp-v7sl08xwpe` 在 pip 安装阶段失败，未进入模型；第二轮日志证实 ENOSPC。
- CPU helper 未获资源，已终止；两个 custom analysis preflight 未产出可读启动日志，未作为任何验证依据。
- `exp-jjzf5biomm` 内原 c286 的慢加载已主动停止并归档到 `glm52-e2e/rank-*/pre-fix-whole-shard-load/`；allocation 保留。
- 当时待办是等待`exp-7eqnrujaw6`获配并完成正式测量；此后该任务的 runner 打包问题及最终成功任务见下文和本文开头。此条不代表当前状态。

## 2026-11-16 原生runner完整预检与资源保留

- `exp-7eqnrujaw6`于11:19获配，11:22在模型加载前失败：原生Git入口误依赖手工会话的`/tmp/benchmark-overlay.tar.gz`。这是runner打包错误，已去除，改为实际源码树SHA256与双host一致性校验。
- 用户要求减少启动次数：新增6900秒有界allocation supervisor；环境/模型失败后保留机器，只有明确修复后的显式指令才重跑应用。client失败保留已编译服务，可原地重跑完整client；退出码0本身不足以证明成功。
- 本地完整HTTP/API对照发现并修复`/flush_cache`纯文本解析；stream记录/输入哈希/汇总写入移到线程，避免阻塞event loop；连接池1秒过期低于server的5秒；同步ingress任一请求失败立即报错；active batch采样间隔20ms。
- 干净Git checkout `811865ca48`的两组CPU预检均通过：双rank完整控制流，实际metadata/host读，注入client失败后保留服务、显式重跑，16 routed smoke、两组warmup和六个measurement round共368次模拟HTTP请求；以及false-success防护、setup失败保留、恢复、有界到期。CPU fixture不计入任何TPU/模型/性能结论。详见`glm52-runner-preflight-20261116.json`。
- 固定代码`811865ca48`，当前任务`exp-6mtkc6t2qi` / `job-agyoxxt6mj` / `art-tfqylgfhvw`，12:02:21 UTC Running，保留前轮25GB compile cache。模型/reader/依赖实现相对1879无变化。
- 已启用本线程5分钟heartbeat `glm52-8-chip`，仅实质进展/异常/完成时通知。最新ID与监控状态以`glm52-e2e-baseline-status.json`为准。
