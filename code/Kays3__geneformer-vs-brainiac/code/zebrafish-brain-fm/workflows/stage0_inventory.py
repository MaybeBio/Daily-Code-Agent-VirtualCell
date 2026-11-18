"""Stage 0 as a Neuro-Workflow graph: ManifestLoader -> ManifestCheck -> ManifestVerify.

Run from the zebrafish-brain-fm folder:

    uv run python workflows/stage0_inventory.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))

from neuroworkflow import WorkflowBuilder  # noqa: E402

from manifest_check import ManifestCheckNode  # noqa: E402
from manifest_loader import ManifestLoaderNode  # noqa: E402
from manifest_verify import ManifestVerifyNode  # noqa: E402

TITLE = "Stage 0: data inventory"


def build(argv=()):
    """Build the graph without running it."""
    loader, check, verify = ManifestLoaderNode("Manifest"), ManifestCheckNode("Check"), ManifestVerifyNode("Verify")
    loader.configure(manifest_path=str(ROOT / "data" / "MANIFEST.tsv"))
    verify.configure(root=str(ROOT))
    builder = WorkflowBuilder(
        "zfbrain_stage0_inventory",
        context={"species": "zebrafish", "resource_requirements": {"cpus": 1, "memory_gb": 1}},
    )
    for node in (loader, check, verify):
        builder.add_node(node)
    builder.connect("Manifest", "rows", "Check", "rows")
    builder.connect("Manifest", "rows", "Verify", "rows")
    builder.connect("Check", "ready", "Verify", "ready")
    return builder, {"check": check, "verify": verify}


def main():
    builder, nodes = build(sys.argv)
    success = builder.build().execute()
    check, verify = nodes["check"], nodes["verify"]
    ready, blocked = check.get_output("ready"), check.get_output("blocked")
    verified, failed, absent = (verify.get_output(k) for k in ("verified", "failed", "absent"))
    print("\n--- stage 0 inventory ---")
    print("workflow success:", success)
    print(f"ready: {len(ready)}  blocked: {len(blocked)}")
    print(f"of the ready: verified on this machine {len(verified)}, failed {len(failed)}, not on this machine {len(absent)}")
    for name in verified:
        print(f"  verified: {name}")
    for name, reason in failed.items():
        print(f"  FAILED: {name}: {reason}")
    for name in absent:
        print(f"  not here: {name}")
    for name, missing in blocked.items():
        print(f"  blocked: {name}: missing {', '.join(missing)}")


if __name__ == "__main__":
    main()
