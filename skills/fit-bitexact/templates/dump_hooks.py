#!/usr/bin/env python3
"""PyTorch dump hooks, plus replay hooks that feed the target side's real inputs.

Dump (usable on either the target or the reference side):
    dumper = Dumper(model, out_dir="pack-A/records/r0", modules=["layers.0.attn", "layers.0.mlp"])
    model(input_ids); dumper.close()
  Each module writes <name>.in0.<dtype> and <name>.out.<dtype> (contiguous memory, raw bytes),
  and shape/dtype are recorded in manifest.json.

Replay (reference side, used when isolating single ops in S2):
    replay = Replay(ref_model, pack_dir="pack-A/records/r0", modules=["layers.0.attn"])
    ref_model(input_ids)
  On entry to these modules, the current input is replaced with the target side's dumped input,
  so only the module's own computation is compared and upstream differences do not leak in.

Non-PyTorch engines (vLLM / SGLang / custom C++ / Metal): add dumps in the same format at the
corresponding module's entry and exit; with the same file format, compare.py can compare them directly.
"""
import json
import os
import torch

_EXT = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32"}


def _write(t, path):
    t = t.detach().contiguous().cpu()
    t.view(torch.int16 if t.element_size() == 2 else torch.int32).numpy().tofile(path)


class Dumper:
    def __init__(self, model, out_dir, modules):
        os.makedirs(out_dir, exist_ok=True)
        self.out, self.manifest, self.handles = out_dir, {}, []
        named = dict(model.named_modules())
        for name in modules:
            self.handles.append(named[name].register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(mod, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            for tag, t in [("in0", inputs[0]), ("out", out)]:
                path = f"{self.out}/{name}.{tag}.{_EXT[t.dtype]}"
                _write(t, path)
                self.manifest[os.path.basename(path)] = {"shape": list(t.shape), "dtype": str(t.dtype)}
        return fn

    def close(self):
        for h in self.handles:
            h.remove()
        with open(f"{self.out}/manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=1)


class Replay:
    def __init__(self, model, pack_dir, modules):
        meta = json.load(open(f"{pack_dir}/manifest.json"))
        named = dict(model.named_modules())
        for name in modules:
            file = next(k for k in meta if k.startswith(f"{name}.in0."))
            named[name].register_forward_pre_hook(self._hook(f"{pack_dir}/{file}", meta[file]))

    @staticmethod
    def _hook(path, m):
        dtype = getattr(torch, m["dtype"].split(".")[-1])
        raw = torch.from_file(path, size=int(torch.tensor(m["shape"]).prod()),
                              dtype=torch.int16 if dtype in (torch.float16, torch.bfloat16) else torch.int32)
        target_in = raw.view(dtype).reshape(m["shape"])

        def fn(mod, inputs):
            return (target_in.to(inputs[0].device),) + tuple(inputs[1:])
        return fn
