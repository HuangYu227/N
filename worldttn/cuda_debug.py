"""Opt-in synchronized operator/layout tracing; never replace tensors or kernels."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import os
from pathlib import Path
import socket
import traceback

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_ACTIVE = ContextVar("ttn_cuda_trace", default=None)


def tensor_metadata(value):
    records = []
    def visit(item, path):
        if isinstance(item, torch.Tensor):
            record = {"path": path, "shape": list(item.shape), "dtype": str(item.dtype),
                      "device": str(item.device), "layout": str(item.layout)}
            if item.layout == torch.strided:
                pointer = item.data_ptr()
                record.update(stride=list(item.stride()), contiguous=item.is_contiguous(),
                              storage_offset=item.storage_offset(), element_size=item.element_size(),
                              pointer_mod={str(n): pointer % n for n in (16, 128, 256)})
            records.append(record)
        elif isinstance(item, dict):
            for name, child in item.items(): visit(child, f"{path}.{name}")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item): visit(child, f"{path}[{index}]")
    visit(value, "tensor")
    return records


class CUDATrace(TorchDispatchMode):
    def __init__(self, handle, device):
        super().__init__()
        self.handle, self.device = handle, torch.device(device)
        self.sequence = 0
        self.last_op = self.last_cuda_op = self.last_node = None
        self.failure_logged = False

    def emit(self, event, **fields):
        self.sequence += 1
        record = {"sequence": self.sequence, "event": event, **fields}
        self.handle.write(json.dumps(record) + "\n")
        self.handle.flush()  # Retain the final begin record even on a CUDA/Slurm abort.

    def synchronize(self):
        if self.device.type == "cuda": torch.cuda.synchronize(self.device)

    def failure(self, error, where):
        if self.failure_logged: return
        self.failure_logged = True
        record = {"where": where, "error": str(error), "last_op": self.last_op,
                  "last_cuda_op": self.last_cuda_op, "last_node": self.last_node,
                  "python_stack": traceback.format_exc()}
        self.emit("error", **record)
        print("[TTN CUDA error] " + json.dumps(record), flush=True)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        inputs = tensor_metadata((args, kwargs))
        self.last_op = {"name": str(func), "tensors": inputs}
        if any(row["device"].startswith("cuda") for row in inputs): self.last_cuda_op = self.last_op
        fields = dict(self.last_op)
        if "view_as_complex" in str(func) or "view_as_real" in str(func) or any(
                "complex" in row["dtype"] for row in inputs):
            fields["python_stack"] = traceback.format_stack(limit=12)
        self.emit("op.begin", **fields)
        try:
            result = func(*args, **kwargs)
            outputs = tensor_metadata(result)
            if any(row["device"].startswith("cuda") for row in outputs): self.last_cuda_op = self.last_op
            self.synchronize()
            self.emit("op.end", name=str(func), tensors=outputs)
            return result
        except Exception as error:
            self.failure(error, "operator")
            raise

    @contextmanager
    def backward(self, loss):
        handles, visited, pending = [], set(), [loss.grad_fn]
        def before(name, gradients):
            self.last_node = {"name": name, "grad_outputs": tensor_metadata(gradients)}
            self.emit("node.begin", **self.last_node)
            self.synchronize()
            # Return None: observe gradients without replacing their buffers/layout.
        def after(name, leaf, grad_inputs, grad_outputs):
            self.last_node = {"name": name, "grad_inputs": tensor_metadata(grad_inputs),
                              "grad_outputs": tensor_metadata(grad_outputs),
                              "leaf": tensor_metadata(leaf), "accumulated_grad": tensor_metadata(
                                  leaf.grad if leaf is not None else None)}
            self.synchronize()  # Also catches direct Triton launches absent from ATen dispatch.
            self.emit("node.end", **self.last_node)
        try:
            while pending:
                node = pending.pop()
                if node is None or node in visited: continue
                visited.add(node)
                name = node.name()
                leaf = getattr(node, "variable", None)
                handles.append(node.register_prehook(lambda gradients, name=name: before(name, gradients)))
                handles.append(node.register_hook(lambda gi, go, name=name, leaf=leaf: after(name, leaf, gi, go)))
                pending.extend(child for child, _ in node.next_functions)
            self.emit("backward.begin", nodes=len(visited), loss=tensor_metadata(loss))
            self.synchronize()
            yield
            self.synchronize()
            self.emit("backward.end")
        except Exception as error:
            self.failure(error, "backward")
            raise
        finally:
            for handle in handles: handle.remove()


@contextmanager
def cuda_diagnostics(directory, device):
    if not directory:
        yield
        return
    path = Path(directory) / f"rank-{os.environ.get('RANK', '0')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        mode = CUDATrace(handle, device)
        token = _ACTIVE.set(mode)
        mode.emit("trace.begin", hostname=socket.gethostname(), pid=os.getpid(), device=str(device),
                  launch_blocking=os.environ.get("CUDA_LAUNCH_BLOCKING"),
                  gdn_disable_compile=os.environ.get("GDN_DISABLE_COMPILE"),
                  gdn_disable_complex_compile=os.environ.get("GDN_DISABLE_COMPLEX_COMPILE"))
        print(f"[TTN CUDA trace] path={path} device={device} synchronized=true", flush=True)
        try:
            # Capture the forward origin of a failing backward node without NaN scans.
            with mode, torch.autograd.detect_anomaly(check_nan=False):
                yield
            mode.emit("trace.end")
        except Exception as error:
            mode.failure(error, "update")
            raise
        finally:
            _ACTIVE.reset(token)


@contextmanager
def trace_backward(loss):
    mode = _ACTIVE.get()
    if mode is None:
        yield
    else:
        with mode.backward(loss): yield
