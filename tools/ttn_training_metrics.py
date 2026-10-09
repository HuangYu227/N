"""Stream existing training logs; no weights, GPU calls or job submission."""
import argparse
import json
import math
from pathlib import Path


def numbers(value):
    if isinstance(value, dict):
        for item in value.values(): yield from numbers(item)
    elif isinstance(value, list):
        for item in value: yield from numbers(item)
    elif type(value) in (int, float): yield value


def mean(value):
    total = count = 0
    for item in numbers(value): total += item; count += 1
    return total / count if count else None


def maximum(value):
    return max(numbers(value), default=None)


def gib(value):
    return value / 2**30 if value is not None else None


def line(values):
    print(" | ".join("null" if v is None else f"{v:.6g}" for v in values), flush=True)


def change(before, after):
    b, a = mean(before), mean(after)
    return 100 * (a / b - 1) if b is not None and a is not None and b > 0 else None


def records(root):
    path = root / "train.jsonl"
    if not path.is_file():
        print("缺少训练日志：", path, flush=True)
        return
    with path.open(encoding="utf-8") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip(): continue
            try: yield json.loads(raw)
            except json.JSONDecodeError:
                print("跳过未完整写入行：", path, number, flush=True)


def anchor_rows(row):
    rank = next(r for r in row["ranks"] if r["rank"] == 0)
    for chunk in rank["chunks"]:
        if chunk["chunk"] + 1 not in (1, 5, 20, 40): continue
        for anchor in chunk["anchors"]:
            p = anchor.get("proximal", {}).get("per_head", {})
            spectrum = anchor.get("state_spectrum", {}).get("committed", {})
            calls = anchor.get("proximal_trajectory", [])
            noisy = calls[-1] if calls else {}
            query = noisy.get("heldout_query", {})
            energy = mean(spectrum.get("top1_energy_fraction"))
            effect = noisy.get("output_effect", {}).get("relative_delta")
            yield [chunk["chunk"] + 1, anchor["block"], anchor.get("state", {}).get("rms"),
                mean(spectrum.get("stable_rank")), 100 * energy if energy is not None else None,
                change(p.get("inner_objective_before"), p.get("inner_objective_after")),
                change(p.get("current_residual_mse_before"), p.get("current_residual_mse_after")),
                change(p.get("history_residual_mse_before"), p.get("history_residual_mse_after")),
                mean(p.get("history_current_gradient_ratio")), mean(p.get("history_current_gradient_cosine")),
                change(query.get("before"), query.get("after_support_solve")),
                100 * effect if effect is not None else None]


def report(run):
    run = Path(run)
    print("开始读取训练指标：", run, flush=True)
    plan = json.loads((run / "chain.json").read_text(encoding="utf-8"))
    roots = list(dict.fromkeys([Path(plan["source"]), run]))
    print("step | loss | grad | 秒 | GPU峰值GiB | 采样RSS最大GiB | 缓存最大GiB", flush=True)
    latest = None
    for root in roots:
        print("读取：", root / "train.jsonl", flush=True)
        for row in records(root):
            ranks = row["ranks"]
            phases = [p for r in ranks for p in r.get("memory_phases", [])]
            chunks = [c for r in ranks for c in r.get("chunks", [])]
            line([row["step"], row.get("loss"), row.get("outer_grad_norm"), row.get("seconds"),
                gib(maximum([r.get("peak_allocated_bytes") for r in ranks])),
                gib(maximum([p.get("host_rss_bytes") for p in phases])),
                gib(maximum([c.get("memory_storage_bytes") for c in chunks]))])
            print("  chunks/commits：", [(r["rank"], len(r.get("chunks", [])), r.get("commits")) for r in ranks],
                  "记录全部有限：", all(math.isfinite(x) for x in numbers(row)), flush=True)
            if any("offload_saved_tensors" in p for p in phases):
                print("  GPU保存预算GiB | 窗口累计GPU复制最大GiB | 窗口累计CPU复制最大GiB（累计量，不是RSS）：", flush=True)
                line([maximum([r.get("activation_gpu_budget_gib") for r in ranks]),
                      gib(maximum([p.get("offload_saved_tensors", {}).get("gpu_packed_tensor_bytes") for p in phases])),
                      gib(maximum([p.get("offload_saved_tensors", {}).get("cpu_packed_tensor_bytes") for p in phases]))])
            if latest is None or row["step"] >= latest["step"]:
                cfg = row.get("config", {})
                # Keep only small summaries; a complete step can be a large JSON object.
                latest = {"step": row["step"], "tbptt": row.get("tbptt"), "config": {k: cfg.get(k) for k in (
                    "memory_update", "memory_transport", "memory_selection", "memory_capacity_frames",
                    "memory_history_weight", "memory_kappa", "local_update", "persistent_meta", "persistent_update")},
                    "anchors": list(anchor_rows(row)), "health": [(r["rank"],
                        r.get("parameter_update", {}).get("missing_core_gradients"),
                        r.get("parameter_update", {}).get("by_origin")) for r in ranks]}
            del row, ranks, phases, chunks
    if latest is None: raise ValueError("没有完整训练记录")
    print("\n最后step：", latest["step"], "TBPTT：", latest["tbptt"], "配置：", latest["config"], flush=True)
    print("rank0，均值跨head；Δ%=(after/before-1)*100；null表示未记录/未定义。", flush=True)
    print("chunk | anchor | S_RMS | stable-rank | top1能量% | 内目标Δ% | 当前残差Δ% | 历史残差Δ% | 历史/当前梯度 | 梯度cos | noisy queryΔ% | noisy DirectS输出Δ%", flush=True)
    for values in latest["anchors"]: line(values)
    for rank, missing, origins in latest["health"]:
        print("rank", rank, "缺失核心梯度=", missing, "参数来源更新=", origins, flush=True)
    print("读取完成；未加载权重、未重算SHA、未提交作业。内部残差不是视频rollout MSE。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    report(parser.parse_args().run)


if __name__ == "__main__": main()
