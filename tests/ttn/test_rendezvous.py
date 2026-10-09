"""Real rendezvous checks without loading SANA or requesting CUDA."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import time

import pytest
import torch


def _launch(rank=0, *, job="10001", preferred=38688):
    return {"launcher": "slurm", "rank": rank, "world_size": 2,
            "master_addr": "127.0.0.1", "master_port": preferred,
            "job_id": job}


def _automatic_store(launch, path, timeout=60):
    from worldttn import distributed
    assert hasattr(distributed, "_automatic_tcp_store"), "automatic rendezvous is missing"
    return distributed._automatic_tcp_store(launch, str(path), timeout)


def _worker(index, output, preferred, ready):
    import torch.distributed as dist
    os.environ["USE_LIBUV"] = "0"
    torch.set_num_threads(1)
    group, rank = divmod(index, 2)
    output = Path(output)
    launch = _launch(rank, job=str(10001 + group), preferred=preferred)
    store = _automatic_store(launch, output / f"group-{group}.json")
    # All server handles must coexist to exercise concurrent groups on one IP.
    ready.wait(timeout=60)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        value = torch.tensor(rank + 1 + group * 10, dtype=torch.int64)
        dist.all_reduce(value)
        (output / f"result-{index}.json").write_text(json.dumps({
            "group": group, "rank": rank, "port": store.port, "sum": value.item()
        }))
    finally:
        dist.destroy_process_group()


def _run_groups(output, preferred, groups):
    from worldttn import distributed
    assert hasattr(distributed, "_automatic_tcp_store"), "automatic rendezvous is missing"
    ready = torch.multiprocessing.get_context("spawn").Barrier(groups * 2)
    torch.multiprocessing.spawn(_worker, args=(str(output), preferred, ready),
                                nprocs=groups * 2, join=True)
    return [json.loads((output / f"result-{index}.json").read_text())
            for index in range(groups * 2)]


def test_occupied_preferred_port_keeps_owner_and_connects_two_ranks(tmp_path):
    with socket.socket() as owner:
        owner.bind(("127.0.0.1", 0))
        owner.listen()
        preferred = owner.getsockname()[1]
        rows = _run_groups(tmp_path, preferred, 1)
        assert {row["sum"] for row in rows} == {3}
        assert len({row["port"] for row in rows}) == 1
        assert rows[0]["port"] != preferred
        # The unrelated process can still accept connections after rendezvous.
        with socket.create_connection(("127.0.0.1", preferred), timeout=2):
            connection, _ = owner.accept()
            connection.close()


def test_concurrent_groups_on_same_address_have_distinct_ports_and_state(tmp_path):
    rows = _run_groups(tmp_path, 38688, 2)
    ports = []
    for group in range(2):
        members = [row for row in rows if row["group"] == group]
        assert len(members) == 2
        assert {row["sum"] for row in members} == {3 + group * 20}
        assert len({row["port"] for row in members}) == 1
        ports.append(members[0]["port"])
    assert ports[0] != ports[1]


def test_missing_publication_has_bounded_timeout(tmp_path):
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        _automatic_store(_launch(1), tmp_path / "missing.json", timeout=.15)
    assert time.monotonic() - started < 3


def test_relative_publication_path_is_rejected():
    with pytest.raises(ValueError, match="absolute"):
        _automatic_store(_launch(), Path("relative.json"), timeout=.15)


def test_server_rejects_reused_publication_without_overwriting(tmp_path):
    path = tmp_path / "reused.json"
    original = '{"old": "another launch"}'
    path.write_text(original)
    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        _automatic_store(_launch(), path, timeout=.15)
    assert path.read_text() == original


@pytest.mark.parametrize("payload", [
    "not json",
    "[]",
    {},
    {"world_size": 3},
    {"job_id": "another-job"},
    {"master_addr": "127.0.0.2"},
    {"port": 0},
    {"port": 65536},
    {"port": "12345"},
])
def test_client_rejects_invalid_publication_before_connecting(tmp_path, payload):
    path = tmp_path / "invalid.json"
    if isinstance(payload, dict) and payload:
        payload = {"master_addr": "127.0.0.1", "world_size": 2,
                   "job_id": "10001", "port": 12345, **payload}
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
    with pytest.raises((ValueError, RuntimeError)):
        _automatic_store(_launch(1), path, timeout=.15)
