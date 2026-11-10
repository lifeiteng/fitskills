# fitskills

[简体中文](README.md) | English

[![License: MIT](https://img.shields.io/badge/license-MIT-16A34A.svg?style=flat-square)](LICENSE)

**A Skills toolbox for AI.** By Feiteng Li ([@FeitengLi](https://x.com/FeitengLi)).

Install them into Claude Code, Codex or any agent that supports Skills, then use them right in chat to improve your LLM: numerics that don't match, a cause you can't find, changes you can't tell whether you broke something.

## Quick start

**1. Install**

```bash
npx -y skills add lifeiteng/fitskills -g --all
```

Or through the Claude Code plugin marketplace:

```
/plugin marketplace add lifeiteng/fitskills
/plugin install fitskills@fitskills
```

**2. Hand your problem to `/fit`**

Type `/fit` followed by your situation. It's fine if details are missing; it will ask.

```text
/fit We moved a MoE model from vLLM to our own engine. Greedy decoding diverges from token 300, logits differ around 1e-3. Where do we start?
```

**3. Keep going in the same conversation**

Add data or dumped tensors as you get them; no need to retype the command.

## Skills

### Entry

| Command | What it does |
|---|---|
| [`/fit`](skills/fit/SKILL.md) | Picks the right skill for your problem and plans the order |

### Numerics and precision

| Command | What it does |
|---|---|
| [`/fit-bitexact`](skills/fit-bitexact/SKILL.md) | Make the same LLM compute bit-identical results on two implementations: different GPUs, different inference engines, or inference vs training. Proceeds in dependency order: inputs → layer 0 op by op → decode → layer by layer → long records → recover speed. Ships three templates: bitwise compare, dump/replay hooks, GEMM probes |

`fit-bitexact` in practice: DeepSeek V4.1 inference on NVIDIA GPU aligned to Mac Studio (Apple M5 Ultra), compared against a reference implementation: 4 full records, 2.3 billion values, 0 bits differ; after the fixes, prefill got about 2% faster. Full story in the [WeChat article](https://mp.weixin.qq.com/s/9Fvh3UuxGfa8xDVZ1rQEag).

## Roadmap

These skills are being prepared and not released yet:

| Area | Plan |
|---|---|
| MoE quantization and pruning | `TODO` |

## About the author

X: [@FeitengLi](https://x.com/FeitengLi)

WeChat official account: Generative AI

<img src="assets/wechat-qr.png" alt="WeChat official account QR code" width="480">

## Layout

```
skills/
├── fit/                    # entry point and dispatcher
└── fit-bitexact/
    ├── SKILL.md            # roadmap, comparison criteria, handoff contract
    ├── references/         # target-side fixes, reference-side modeling, root-cause cheatsheet
    └── templates/          # compare.py, dump_hooks.py, probe_gemm.py
```

## License

[MIT](LICENSE). Free to use, modify and redistribute; keep the copyright notice.
