# Qwen3.5-0.8B 文本模型

本实现从本地 Hugging Face checkpoint 构建 DeepProve 原生计算图，覆盖
Qwen3.5-0.8B 的全部 24 个文本 decoder block、embedding、最终 RMSNorm 和
共享权重的语言模型输出层。18 层采用 Gated DeltaNet，另外 6 层采用完整
因果注意力。视觉编码器与辅助 MTP 权重不属于本次文本推理计算图。

浮点参考使用 `transformers==5.3.0` 的 `Qwen3_5ForCausalLM`，CPU float32、
eager attention。原生推理和证明均在 Rust 中执行，不调用 Python 模型代算。
数值参考数据的最长前向输入为 256 tokens；完整模型证明的长度验收门槛
为 32 tokens。模型与校准保持冻结配置的 256-token 容量。
`--max-context` 约束前向计算的输入行数，即
`prompt_len + max_new_tokens - 1 <= max_context`。最后一个预测 token
只作为输出认证，不再输入模型。

## 与 DeepProve 论文的对应

本仓库根目录的 `deep-prove.pdf` 是技术路线依据：

| 论文内容 | 实现 |
| --- | --- |
| §3.1 完整因果序列的单次认证 | 先进行带缓存的逐 token 生成，再清空 KV、卷积和递归状态重放完整序列，逐项核对生成 token，并证明完整因果图和 argmax 输出链。 |
| §3.2 静态量化 | 使用独立训练文本校准并冻结整数图；保留较高精度的中间乘积，在显式 Requant 节点执行有证明的舍入和截断。 |
| §4.1 模型解析与预处理 | 读取本地 safetensors 及索引文件，构造混合 decoder 图，提交模型权重和 lookup tables。 |
| §4.2–4.3 证明 | 使用 polynomial commitments、sumcheck 和 LogUp，绑定图边、线性运算、非线性查表与整数范围。 |

Qwen 的递归状态、卷积和归一化需要在论文路线之上扩展算子。特别是
Qwen 的归一化保留 `epsilon=1e-6`，没有采用论文 §4.3.3 中省略 epsilon
的简化公式。

## 关键模型语义

- 普通 RMSNorm 使用 `1 + weight`，Gated DeltaNet 输出 RMSNorm 使用
  `weight`。Q/K 的 L2 normalization 使用 `sqrt(sum(x²) + epsilon)`。
  DeltaNet 的 Q 在 L2 normalization 后还乘以 `1 / sqrt(128)`。
- 线性注意力使用 4-tap depthwise causal convolution；缓存保存前三个
  时间步，完整证明使用零初始历史。Q/K 各有 16 个 128 维 head，V 同样
  使用 16 个 128 维 head。
- Gated DeltaNet 的递推为
  `U = alpha * S_prev`、`memory = kᵀU`、
  `delta = beta * (v - memory)`、`S = U + k * deltaᵀ`、`o = qᵀS`。
  `alpha = exp(-exp(A_log) * softplus(a + dt_bias))`，`beta = sigmoid(b)`。
- 完整注意力保留 8 个 Q head、2 个 KV head、每个 head 256 维，按 head
  拆分 Q 与输出 gate。只对 head 的前 64 维应用 partial RoPE，频率基数
  为 `1e7`；其余维度保持不变。
- 直接使用 checkpoint 的 `tokenizer.json`，保留原始 Unicode 规则及
  解码空格。`--chat` 使用单轮、关闭 thinking 的文本模板。
  `248044` 和 `248046` 都是生成停止 token；prompt 内的结束标记不会
  提前终止生成。

## 新算子的整数证明

QwenNorm 对每一行证明整数平方和、整数平方根的上下界及倒数乘子的
舍入关系。分母包含 epsilon；根、余数及差值分解为有范围证明的 limbs。
最终输出与输入、归一化乘子、可选 gamma 的乘积通过 sumcheck 绑定。

GatedDeltaScan 使用冻结的定点乘子，对衰减、memory、value 转换、delta、
状态更新和输出分别保留舍入余数。状态与中间结果限定在有证明的 signed
32-bit 域；余数按实际 shift 位数约束。逐层提交完整时间轴上的状态，
sumcheck 证明递推关系和两个收缩运算，独立的时间移位关系绑定
`S_prev[t] = S[t-1]`，并约束零初始状态与 padding。递归状态缓存不作为
证明输入。

CausalConv1d 证明 causal shift 与卷积乘积。Partial RoPE 的证明保留
未旋转维度。Sigmoid、Softplus、ExpNegative 使用独立的查表定义，并将
物理 padding 的零输出纳入约束。

新构建的 EinSum context 使用 `EinSumV2`：stacked 与 contracted 维度
共同进入 sumcheck，输出位置的 equality polynomial 参与乘积约束，
最终左右输入在同一随机位置分别绑定。这样也能证明没有收缩维度的
逐元素门控乘法。多输出 QKV 的共享输入声明通过独立的随机聚合证明
合并。旧 `EinSum` context 只用于显式读取旧产物，不作为新 Qwen 证明
的回退路径。

验证器先检查分块节点身份、原图覆盖和边界承诺：合成 Split/Recombination
节点不能占用原始算子的 ID；不支持横向拆分的 scan 必须保留完整时间轴。
层见证、边界张量和查表见证的承诺 ID 不能相互覆盖。新协议在产生全局
Fiat–Shamir 挑战前吸收完整公开输入输出及其形状，协议版本由可信 verifier
context 决定；旧产物走显式兼容路径。

证明声明对应被冻结的整数计算图。浮点一致性、量化质量和生成内容通过
独立数值与语义评测验收；它们与密码学证明分别记录。

整序列认证保留逐 token 阶段的 Softmax shift witness，重新计算其余图，
并在证明中约束该 witness 的查表与归一化关系；它并非清空所有缓存后的
独立 Softmax 重算。Argmax 证明所选 token 达到该行最大 logit；当多个
token 并列最大时，证明不额外限定平局选择规则。

## 构建与生成

从仓库根目录执行，模型目录可以替换为其他同构本地 checkpoint。

```sh
cargo build --release --locked -p zkml \
  --bin extract-logits --bin bench-llm --bin verify-llm

RAYON_NUM_THREADS=8 target/release/extract-logits \
  --model /home/yliangh/models/Qwen/Qwen3.5-0.8B \
  --chat --text '中国的首都是哪里？只回答城市名。' \
  --max-context 256 --num-tokens 32 --mode float \
  --output-dir target/qwen35/float-generation
```

整数推理需要提供固定校准数据。校准集使用训练 split，测试集与语义
验收问题不参与校准。数据准备、参考模型、逐层对照和评价命令见
[reference and acceptance tools](../assets/scripts/llms/qwen35_validation.md)。

Qwen FFN 使用 16-bit SiLU 输入表；表的输出位宽按未截断 up projection
与 GLU 乘积后的 Requant 预算选择。本模型的冻结配置使用
`ZKML_BIT_LEN=14`，SiLU 输入表为 16-bit、输出表为 12-bit，残差为
14-bit。后续 down projection 的预算
由 GLU 后的独立 Requant 处理。SiLU 输出范围覆盖完整输入表域，包含
校准样本未触达的正负端点。仓库的通用默认值仍为 12-bit；下面显式
指定本模型通过开发集筛选的配置。

```sh
ZKML_BIT_LEN=14 RAYON_NUM_THREADS=8 target/release/extract-logits \
  --model /home/yliangh/models/Qwen/Qwen3.5-0.8B \
  --chat --text '中国的首都是哪里？只回答城市名。' \
  --max-context 256 --num-tokens 32 --mode int \
  --calibration-tokens target/qwen35/inputs/calibration-bilingual/calibration.json \
  --calibration-samples 8 \
  --output-dir target/qwen35/int-generation
```

`manifest.json` 记录输入/输出 token、模型文件指纹和量化摘要指纹。
`quantization.json` 冻结完整图结构、整数参数、内部递归尺度、归一化精度、
舍入/截断参数及选定算术实现文件的源码指纹。它不代替完整构建身份；
交付报告还应记录执行二进制、`Cargo.lock` 和代码版本或补丁的身份。
测试与证明必须使用相同冻结配置。

## 生成、证明与独立验证

```sh
/usr/bin/time -v env DEEP_PROVE_DISABLE_MEMORY_STATS=1 ZKML_BIT_LEN=14 \
  target/release/bench-llm \
  --model qwen35 --hf-dir /home/yliangh/models/Qwen/Qwen3.5-0.8B \
  --prompt-cases target/qwen35/inputs/proof32-semantic-case.json \
  --max-context 256 --num-threads 8 \
  --calibration-tokens target/qwen35/inputs/calibration-bilingual/calibration.json \
  --calibration-samples 8 \
  --artifacts-dir target/qwen35/proof32-final \
  --bench target/qwen35/proof32-final/bench.csv
```

该验收用例包含 31-token 英文问题和 2-token 输出预算，验收时检查实际
计算的完整 history 为 32 行。`--max-context 256` 保持校准与冻结图身份，
实际证明长度由用例的 prompt 和生成序列决定。

`DEEP_PROVE_DISABLE_MEMORY_STATS=1` 只关闭逐算子的操作系统内存采样，
避免大内存进程反复读取 `/proc/self/smaps` 的开销；计时、模型计算和
证明协议不变。不设置时保持原行为。上面的 `/usr/bin/time -v` 仍会
记录整个进程的峰值 RSS。

也可以使用 `--prompt-cases cases.json`，一次校准和 setup 后证明多组输入。
每项包含 `name`、`prompt_tokens`、`max_new_tokens`。

成功后输出 `proof.bin`、`io.bin`、`verifier.bin`、`result.json`、精确
prompt/generated token 文件及解码文本。证明失败时只可能留下明确标记为
未验证的 generation diagnostics，不能当作通过证明的结果。

验证者应从独立可信的模型 setup 记录取得 verifier 文件及其 SHA256，并
使用调用方预期的 prompt token 文件：

```sh
target/release/verify-llm \
  --artifacts-dir target/qwen35/proof32-final/en_factual_32 \
  --verifier /path/to/trusted/verifier.bin \
  --verifier-sha256 TRUSTED_VERIFIER_SHA256 \
  --prompt-tokens /path/to/expected-prompt-tokens.json \
  --tokenizer /home/yliangh/models/Qwen/Qwen3.5-0.8B/tokenizer.json \
  --tokenizer-sha256 TRUSTED_TOKENIZER_SHA256
```

`verify-llm` 验证可信 verifier 对应的计算、公开 token 链，以及指定可信
tokenizer 时的解码文本。`result.json` 中的模型文件、校准与量化指纹是
来源记录，验证器不会重新计算这些指纹或检查 `quantization.json`。
原始 checkpoint、冻结量化配置与 verifier 之间的对应关系须由独立可信
setup 记录建立；产物目录自行提供的元数据不能建立这种信任关系。

新文件使用带类型和版本的 `DPZKLLM` V2 envelope，拒绝类型错配、未知
版本与尾随数据。协议由独立可信 verifier context 选择，proof/IO 文件头
不能选择或降级协议。旧版 Llama 保存文件保留显式 legacy 读取路径；
没有绑定公开 IO 的旧 Qwen V1 verifier 被拒绝，需要重新生成证明。
新的混合模型证明
按小块顺序执行，并通过已提交的边界张量连接各块，以控制递归状态证明
的峰值内存。

## 验收记录

完整验收需要同时满足浮点全图一致性、冻结整数图的 held-out 数值质量、
中英文有意义生成、缓存与重放一致、32-token 上下文的完整模型证明
以及独立 verifier 和篡改拒绝检查。最终运行报告应记录数据指纹、量化
指纹、实际输出、测量值及证明文件路径；单算子测试不能替代完整模型验收。

本地 25 条、每条 256 tokens 的 held-out 浮点测试已通过，覆盖 6400 个
logit 行和 6375 个 next-token NLL 目标：top-1 一致率 100%，余弦相似度
`0.9999999999993702`，相对 L2 误差 `1.0957567e-6`，perplexity 比值
`0.99999997994`。24 个 decoder block 的逐层对照也已通过，最大相对
L2 误差 `2.521512e-6`。这些结果使用提供的本地真实权重。

同一份完整 held-out 测试上的冻结整数图也已通过：6400 行的 top-1
一致率为 `95.34375%`，余弦相似度 `0.9991477178068436`，相对 L2
误差 `0.038421807`。6375 个 NLL 目标上的 perplexity 为
`21.11528010`，HF 浮点参考为 `21.03122952`，比值
`1.0039964653`。测试使用预先冻结的 14-bit 配置和全部 25 条样本。

冻结整数图的 9 项中英文生成检查已通过，包括事实问答、算术、翻译、
创作和长 prompt 问答。实际输出包括 `Paris`、`北京。`、`5`、`早上好`
与 `A fluffy cat curled up in the sun, dreaming of a warm, golden nap.`。
检查覆盖所有保存 logits 的有限性、生成 token 的 argmax 一致性、精确
解码和停止标记，并另行逐条阅读。中文解释包含一处不严谨的附加科学
表述，已记录在人工复核报告中；有意义内容验收不保证每个生成事实正确。
记录位于 `target/qwen35/validation/integer/semantic-check.json`、
`semantic-manual-review.json` 和 `semantic-generated-texts.txt`。

独立缓存重放检查清除了包括 Softmax shift 在内的所有缓存。四个真实
模型用例共比较 27,315,200 个整数 logits，全部零差异，生成 token 也
完全一致。其中英文事实问答与原始文本压力用例均实际计算 32 行。
报告位于 `target/qwen35/independent-cache-replay32/summary.json`。

真实权重的 32-token 整模证明已成功生成并通过内置验证。用例为上面的
31-token 英文问题，生成 token 为 `[57590, 248046]`，精确解码为
`Paris`，EOS 前实际计算 32 行。覆盖全部 24 个 decoder block，包含
18 个 GatedDeltaScan 和 6 个全注意力 Softmax，共 422 个证明分块。
本次 CPU 运行使用 8 个线程：证明耗时 `2909.610 s`，内置验证
`41.861 s`；包括校准与 setup 的进程总耗时 `3434.82 s`，峰值 RSS
`61,312,188 KiB`（约 `58.47 GiB`）。证明文件为 `88,583,095` 字节
（约 `84.48 MiB`）。这些是本机本次测量值。

产物位于 `target/qwen35/proof32-final/en_factual_32/`；原始测量见
`target/qwen35/proof32-final/bench.csv` 和 `time.txt`。本次运行的
可信 setup、独立 verifier 副本、预期输入及 tokenizer 指纹已单独保存
在 `target/qwen35/trusted/`。独立验收命令：

```sh
DEEP_PROVE_DISABLE_MEMORY_STATS=1 python3 \
  zkml/assets/scripts/llms/verify_llama3_artifacts.py \
  --pins target/qwen35/trusted/proof32-pins.json \
  --verifier-bin target/release/verify-llm \
  --report target/qwen35/validation/proof32-artifact-verification.json \
  --require-cases 1 --threads 4 --timeout 600
```

脚本名称沿用 Llama，但通过显式词表、EOS、verifier 和 tokenizer pins
支持本 Qwen 用例。本次验收产物的 `result.json.serialization` 描述字符串
仍保留旧的 `v1` 文案；实际三个二进制文件头及协议均为 V2。原件保持
不变，最终源码已改为引用实际 artifact 版本生成该字段。验收运行与最终
交付程序的构建身份分别记录，不能以该描述字符串选择验证协议。

独立验收已通过：正常证明的 `verified` 和 `text_verified` 均为 `true`，
精确输出仍为 `Paris`；18 项篡改测试全部以明确验证错误拒绝，原始产物
指纹保持不变。最终汇总见 `target/qwen35/acceptance-summary.json`，
逐项结果见 `target/qwen35/validation/proof32-artifact-verification.json`。
