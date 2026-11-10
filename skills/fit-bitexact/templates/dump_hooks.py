#!/usr/bin/env python3
"""PyTorch dump hooks, plus replay hooks that feed the target side's real inputs.

Dump (usable on either the target or the reference side):
    with Dumper(model, out_dir="pack-A/records/r0", modules=["layers.0.attn", "layers.0.mlp"]):
        model(input_ids)          # prefill and every decode step are recorded as separate calls
  Each call of each module writes <name>.c<call>.in<i>.<dtype> for tensor positional arg i,
  <name>.c<call>.kw-<key>.<dtype> for tensor kwarg <key>, and <name>.c<call>.out.<dtype>
  (contiguous memory, raw bytes); shape/dtype are recorded in manifest.json. <call> counts from 0
  per module, so repeated calls (prefill + each decode step) do not overwrite each other.
  Non-tensor arguments, and tensors of dtypes not in _EXT, are not dumped. Leaving the `with` block
  (or calling close()) removes the hooks and writes the manifest.

Replay (reference side, used when isolating single ops in S2):
    with Replay(ref_model, pack_dir="pack-A/records/r0", modules=["layers.0.attn"]):
        ref_model(input_ids)
  On entry to these modules, every dumped tensor positional arg and tensor kwarg is replaced with the
  target side's dumped tensor from the same call index (the k-th call replays dump call k), so only
  the module's own computation is compared and upstream differences do not leak in. Non-tensor
  arguments are passed through unchanged; make sure they match the target side yourself.
  Raises if the module is called more times than were dumped.

Non-PyTorch engines (vLLM / SGLang / custom C++ / Metal): add dumps in the same format at the
corresponding module's entry and exit; with the same file format, compare.py can compare them directly.
"""
import json
import math
import os
import torch

_EXT = {torch.float16: "f16", torch.bfloat16: "bf16", torch.float32: "f32",  # compare.py reads these
        torch.int32: "i32", torch.int64: "i64", torch.uint8: "u8", torch.bool: "b8"}  # e.g. positions, masks
_RAW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}  # raw-bytes view by element size


def _write(t, path):
    t = t.detach().contiguous().cpu()
    t.view(_RAW[t.element_size()]).numpy().tofile(path)


def _read(path, m):
    dtype = getattr(torch, m["dtype"].split(".")[-1])
    raw = torch.from_file(path, shared=False, size=math.prod(m["shape"]),
                          dtype=_RAW[torch.empty((), dtype=dtype).element_size()])
    return raw.view(dtype).reshape(m["shape"])


class _Hooks:
    def __init__(self):
        self.handles = []

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class Dumper(_Hooks):
    def __init__(self, model, out_dir, modules):
        super().__init__()
        os.makedirs(out_dir, exist_ok=True)
        self.out, self.manifest, self.calls, self.pending = out_dir, {}, {}, {}
        named = dict(model.named_modules())
        for name in modules:
            self.handles.append(named[name].register_forward_pre_hook(self._pre(name), with_kwargs=True))
            self.handles.append(named[name].register_forward_hook(self._post(name)))

    def _save(self, prefix, tag, t):
        path = f"{self.out}/{prefix}.{tag}.{_EXT[t.dtype]}"
        _write(t, path)
        self.manifest[os.path.basename(path)] = {"shape": list(t.shape), "dtype": str(t.dtype)}

    def _pre(self, name):
        def fn(mod, args, kwargs):
            call = self.calls.get(name, 0)
            self.calls[name] = call + 1
            self.pending[name] = call
            for i, t in enumerate(args):
                if isinstance(t, torch.Tensor) and t.dtype in _EXT:
                    self._save(f"{name}.c{call}", f"in{i}", t)
            for k, t in kwargs.items():
                if isinstance(t, torch.Tensor) and t.dtype in _EXT:
                    self._save(f"{name}.c{call}", f"kw-{k}", t)
        return fn

    def _post(self, name):
        def fn(mod, inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            call = self.pending.pop(name)
            if isinstance(out, torch.Tensor) and out.dtype in _EXT:
                self._save(f"{name}.c{call}", "out", out)
        return fn

    def close(self):
        super().close()
        with open(f"{self.out}/manifest.json", "w") as f:
            json.dump(self.manifest, f, indent=1)


class Replay(_Hooks):
    def __init__(self, model, pack_dir, modules):
        super().__init__()
        with open(f"{pack_dir}/manifest.json") as f:
            self.meta = json.load(f)
        self.dir, self.calls = pack_dir, {}
        named = dict(model.named_modules())
        for name in modules:
            self.handles.append(named[name].register_forward_pre_hook(self._hook(name), with_kwargs=True))

    def _load(self, prefix, tag):
        file = next((k for k in self.meta if k.startswith(f"{prefix}.{tag}.")), None)
        return None if file is None else _read(f"{self.dir}/{file}", self.meta[file])

    def _hook(self, name):
        def fn(mod, args, kwargs):
            call = self.calls.get(name, 0)
            self.calls[name] = call + 1
            prefix = f"{name}.c{call}"
            if not any(k.startswith(prefix + ".") for k in self.meta):
                raise RuntimeError(f"{name}: call {call} was not dumped (pack has fewer calls of this module)")
            new_args = []
            for i, t in enumerate(args):
                d = self._load(prefix, f"in{i}") if isinstance(t, torch.Tensor) else None
                new_args.append(t if d is None else d.to(t.device))
            new_kwargs = {}
            for k, t in kwargs.items():
                d = self._load(prefix, f"kw-{k}") if isinstance(t, torch.Tensor) else None
                new_kwargs[k] = t if d is None else d.to(t.device)
            return tuple(new_args), new_kwargs
        return fn
