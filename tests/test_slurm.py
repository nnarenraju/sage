"""SLURM helpers: parsing the read-only queries, and chaining against a stand-in sbatch."""

import os

from sage.utils.slurm import query
from sage.utils.slurm.chain import chain

# sinfo -o "%P|%a|%l|%L|%D|%c|%m|%G|%C" on a cluster with two GPU node types in one partition.
PARTITIONS = """cpu*|up|30-00:00:00|1-00:00:00|42|384|773300|(null)|5376/9984/768/16128
gpu|up|10-00:00:00|1-00:00:00|1|192|773300|gpu:h100_80gb:2,gpu:h200_143gb:2|4/188/0/192
gpu|up|10-00:00:00|1-00:00:00|1|192|773300|gpu:h100_94gb:2,gpu:h200_143gb:2|4/188/0/192
"""
NODE_GPUS = """gpu    gpu:h100_80gb:2,gpu:h200_143gb:2    gpu:h100_80gb:1(IDX:0),gpu:h200_143gb:0(IDX:N/A)
cpu*   (null)                              (null)
"""


def test_queries():
    partitions = query.parse_partitions(PARTITIONS)
    assert partitions["cpu"]["default"] and not partitions["gpu"]["default"]
    assert partitions["gpu"]["cpus"] == [8, 376, 0, 384]
    assert [t["gpus"] for t in partitions["gpu"]["node_types"]] == [
        {"h100_80gb": 2, "h200_143gb": 2}, {"h100_94gb": 2, "h200_143gb": 2},
    ]
    assert query.parse_node_gpus(NODE_GPUS) == {"gpu": (1, 4), "cpu": (0, 0)}
    assert query.gpus("gpu:4") == {"gpu": 4}


def test_chain(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    sbatch = bin_dir / "sbatch"
    sbatch.write_text('#!/bin/bash\necho "$@" >> "$(dirname "$0")/calls"\n'
                      'echo "$((100 + $(wc -l < "$(dirname "$0")/calls")));cluster"\n')
    sbatch.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    assert chain("job.sh", 3, after="7") == ["101", "102", "103"]
    assert (bin_dir / "calls").read_text().splitlines() == [
        "--parsable --dependency=afterany:7 job.sh",
        "--parsable --dependency=afterany:101 job.sh",
        "--parsable --dependency=afterany:102 job.sh",
    ]
    assert chain("job.sh", 1, environment={"SAGE_CHECK_ONLY": "1"}) == ["104"]
    assert (bin_dir / "calls").read_text().splitlines()[-1] == "--parsable --export=ALL,SAGE_CHECK_ONLY=1 job.sh"
