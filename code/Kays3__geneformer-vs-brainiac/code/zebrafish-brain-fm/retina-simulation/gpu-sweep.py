"""Batch the browser model on CUDA and compare its summary to JavaScript."""
import gzip
import json
import sys

import torch


def run(payload):
    cases = payload["cases"]
    device = torch.device("cuda")
    dt = payload["dt"]

    matrix = torch.tensor([case["matrix"] for case in cases], device=device)
    alpha = torch.tensor([case["alpha"] for case in cases], device=device)
    cone_input = torch.tensor([case["input"] for case in cases], device=device)
    state = torch.zeros_like(alpha)
    drive_input = torch.nn.functional.pad(cone_input, (0, alpha.shape[1] - cone_input.shape[2]))
    on = torch.tensor(payload["on"], device=device)
    off = torch.tensor(payload["off"], device=device)
    on_trace, off_trace = [], []
    for step in range(drive_input.shape[1]):
        drive = torch.bmm(matrix, state.unsqueeze(2)).squeeze(2) + drive_input[:, step, :]
        state = (state + alpha * (drive.clamp(-2, 2) - state)).clamp(-2, 2)
        rate = (6 + 33 * state).clamp(0, 80)
        on_trace.append(rate.index_select(1, on).mean(1))
        off_trace.append(rate.index_select(1, off).mean(1))

    on_peaks = torch.stack(on_trace)[450 // dt:1150 // dt + 1].amax(0).tolist()
    off_peaks = torch.stack(off_trace)[450 // dt:1800 // dt + 1].amax(0).tolist()
    error = 0.0
    grouped = {}
    for case, on_peak, off_peak in zip(cases, on_peaks, off_peaks, strict=True):
        reference = case["reference"]
        error = max(error, abs(on_peak - reference["on"]), abs(off_peak - reference["off"]))
        key = (case["seed"], case["stimulus"])
        label = case["mode"] if case["stress"] or case["mode"] == "wild" else "zero"
        grouped.setdefault(key, {})[label] = (on_peak, off_peak, reference["total"])

    if error > 0.0001:
        raise AssertionError(f"CUDA and browser peaks differ by {error:.6f} Hz")

    changes = {}
    for (_, stimulus), results in grouped.items():
        wild, zero = results["wild"], results["zero"]
        synaptic, contacts = results["synaptic"], results["contacts"]
        if max(abs(a - b) for a, b in zip(wild, zero, strict=True)) > 0.0001:
            raise AssertionError("Zero-strength perturbation changed the baseline")
        if not synaptic[2] == wild[2] > contacts[2]:
            raise AssertionError("Unexpected anatomical contact count")
        changes.setdefault(stimulus, []).append((wild[0] - synaptic[0], wild[0] - contacts[0]))

    return {
        "device": torch.cuda.get_device_name(),
        "scenarios": len(cases),
        "wiring_seeds": len({case["seed"] for case in cases}),
        "max_browser_cuda_peak_difference_hz": error,
        "on_peak_drop_hz_ranges": {
            stimulus: {
                "transmission_only": [round(min(a for a, _ in values), 3), round(max(a for a, _ in values), 3)],
                "transmission_and_contact_loss": [
                    round(min(b for _, b in values), 3), round(max(b for _, b in values), 3)
                ],
            }
            for stimulus, values in changes.items()
        },
    }


if __name__ == "__main__":
    with gzip.open(sys.argv[1], "rt") as source:
        print(json.dumps(run(json.load(source)), indent=2))
