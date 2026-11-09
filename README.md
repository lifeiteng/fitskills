# fitskills

简体中文 | [English](README.en.md)

[![License: MIT](https://img.shields.io/badge/license-MIT-16A34A.svg?style=flat-square)](LICENSE)

**人工智能的 Skills 工具箱。** 作者 Feiteng Li（[@FeitengLi](https://x.com/FeitengLi)）。

装进 Claude Code、Codex 等支持 Skills 的 Agent 之后，可以在对话里直接用它们优化提高 LLM：数值精度对不上、找不到原因、改完不知道有没有改坏。

## 快速开始

**第 1 步：安装**

```bash
npx -y skills add lifeiteng/fitskills -g --all
```

在 Claude Code 里也可以通过插件市场安装：

```
/plugin marketplace add lifeiteng/fitskills
/plugin install fitskills@fitskills
```

**第 2 步：把问题交给 `/fit`**

在 Agent 的对话框里输入 `/fit`，后面写上你的情况。信息不完整也没关系，它会追问：

```text
/fit 我们把 MoE 模型从 vLLM 迁到自研引擎，greedy 解码第 300 个 token 起输出不一样了，logits 差在 1e-3 量级。先查哪里？
```

**第 3 步：在同一个对话里继续**

拿到结果后，直接补充数据或导出的张量，不用重新输入命令。

## 全部 Skill

### 入口

| 命令 | 用途 |
|---|---|
| [`/fit`](skills/fit/SKILL.md) | 根据你的问题选择技能，并安排先后顺序 |

### 数值与精度

| 命令 | 用途 |
|---|---|
| [`/fit-bitexact`](skills/fit-bitexact/SKILL.md) | 让同一个大模型在两套实现之间逐位算出相同结果。两套实现可以是不同 GPU、不同推理引擎，也可以是推理和训练。按依赖顺序推进：核对输入 → 第 0 层逐算子 → decode → 逐层 → 长记录 → 回收速度。附按位比较、导出与回放钩子、GEMM 探针三个模板 |

`fit-bitexact` 实战：DeepSeek V4.1 对齐 NVIDIA GPU 约 Mac Studio（Apple M5 Ultra）上的推理过程，与参考实现对比，4 条完整记录、共 23 亿个数，0 位不同；修复后 prefill 反而快了约 2%。完整过程见[公众号文章](链接待发布)。

## 路线图

下面这些技能整理中，还没有发布：

| 方向 | 计划内容 |
|---|---|
| MoE 量化与剪枝 | `TODO` |


## 关于作者

X：[@FeitengLi](https://x.com/FeitengLi)

公众号：Generative AI

<img src="assets/wechat-qr.png" alt="公众号二维码" width="480">

## 目录结构

```
skills/
├── fit/                    # 入口与调度
└── fit-bitexact/
    ├── SKILL.md            # 推进路线、比较判据、交接约定
    ├── references/         # 目标侧修法、参考侧建模、根因速查
    └── templates/          # compare.py、dump_hooks.py、probe_gemm.py
```

## 许可证

[MIT](LICENSE)。可以自由使用、修改和再发布，保留版权声明即可。
