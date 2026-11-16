# GLM-5.2 native MTP baseline 执行记录

当前状态：已按用户“一次启动完成全流程、失败保留现场”的原则重新预检并获配，完整任务`exp-6mtkc6t2qi`正在运行。完整target/draft加载、全部启动图及原生MTP生成warmup已通过；显式routed chat和两组正式性能场景尚未验收，不能宣称完整E2E基线完成。

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

## 当前模型启动证据

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

## 失败和未完成项

- `exp-wgzfz1wtp6`、`exp-v7sl08xwpe` 在 pip 安装阶段失败，未进入模型；第二轮日志证实 ENOSPC。
- CPU helper 未获资源，已终止；两个 custom analysis preflight 未产出可读启动日志，未作为任何验证依据。
- `exp-jjzf5biomm` 内原 c286 的慢加载已主动停止并归档到 `glm52-e2e/rank-*/pre-fix-whole-shard-load/`；allocation 保留。
- 下一步：等待`exp-7eqnrujaw6`获配；runner将固定源码自动完成16-DP routed chat与至少3轮目标测量，记录实际KV容量、cache hits、active batch、接受统计和重编译。当前无可报告的正式TTFT/TPOT/吞吐结果。

## 2026-11-16 原生runner完整预检与资源保留

- `exp-7eqnrujaw6`于11:19获配，11:22在模型加载前失败：原生Git入口误依赖手工会话的`/tmp/benchmark-overlay.tar.gz`。这是runner打包错误，已去除，改为实际源码树SHA256与双host一致性校验。
- 用户要求减少启动次数：新增6900秒有界allocation supervisor；环境/模型失败后保留机器，只有明确修复后的显式指令才重跑应用。client失败保留已编译服务，可原地重跑完整client；退出码0本身不足以证明成功。
- 本地完整HTTP/API对照发现并修复`/flush_cache`纯文本解析；stream记录/输入哈希/汇总写入移到线程，避免阻塞event loop；连接池1秒过期低于server的5秒；同步ingress任一请求失败立即报错；active batch采样间隔20ms。
- 干净Git checkout `811865ca48`的两组CPU预检均通过：双rank完整控制流，实际metadata/host读，注入client失败后保留服务、显式重跑，16 routed smoke、两组warmup和六个measurement round共368次模拟HTTP请求；以及false-success防护、setup失败保留、恢复、有界到期。CPU fixture不计入任何TPU/模型/性能结论。详见`glm52-runner-preflight-20261116.json`。
- 固定代码`811865ca48`，当前任务`exp-6mtkc6t2qi` / `job-agyoxxt6mj` / `art-tfqylgfhvw`，12:02:21 UTC Running，保留前轮25GB compile cache。模型/reader/依赖实现相对1879无变化。
- 已启用本线程5分钟heartbeat `glm52-8-chip`，仅实质进展/异常/完成时通知。最新ID与监控状态以`glm52-e2e-baseline-status.json`为准。
