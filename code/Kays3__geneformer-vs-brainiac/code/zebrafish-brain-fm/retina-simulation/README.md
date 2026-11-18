# Zebrafish retina simulation

Open [index.html](https://github.com/Kays3/geneformer-vs-brainiac/blob/HEAD/zebrafish-brain-fm/retina-simulation/index.html) in a browser. The model and viewer run offline with no installation. Drag either retinal patch to rotate both. Select a cell to plot its paired activity traces, or select a matrix entry to highlight the corresponding connections. Use the shared timeline to replay the stimulus.

## What the model does

The schematic patch represents a cone-driven larval retina at about 6 days post-fertilization. It places seven neuronal groups in three dimensions and assigns signed connections using seeded nearest-neighbor rules. A 10 ms update integrates the visual input and propagates activity. Model coordinates and upstream response changes are in arbitrary units. Ganglion cell output is an illustrative rate in Hz. The peak readouts take the maximum of each ganglion group's mean rate in the time window shown.

- **Transmission only** weakens and slows modeled cone release. It preserves anatomical contact counts.
- **Transmission + contact loss** additionally removes a seeded subset of invaginating contacts. The contact-loss probability applies to that subset.
- **Zero transmission effect and zero contact loss** reproduce the wild-type simulation.

The perturbation is inspired by published [ribbon synapse morphology](https://pmc.ncbi.nlm.nih.gov/articles/PMC6729946/) and [ganglion cell physiology](https://pmc.ncbi.nlm.nih.gov/articles/PMC2141919/). All positions, edges, parameter values, and response traces are synthetic. To use this as a quantitative model, fit wild-type parameters against measured data and compare against independently held-out mutant recordings and anatomy. The viewer uses no expression or EM data.

## Repeat the GPU check

From this directory, generate the batch input with Node.js, then run the sweep using a CUDA-enabled PyTorch environment. The input can be copied to the GPU machine along with the Python script.

```sh
node sweep-input.js | gzip -c > /tmp/retina-sweep-input.json.gz
python gpu-sweep.py /tmp/retina-sweep-input.json.gz
```

The check covers 16 synthetic wiring seeds, three stimuli, and four perturbation settings, including the zero-effect control. It uses the same inputs and model coefficients as the browser, compares the two simulations' ganglion cell peaks, and checks the expected anatomical contact counts.

On November 18, 2026, the NVIDIA GB10 on `thinkstation1` completed all 192 cases over Tailscale SSH via `hokudaibook`. The largest discrepancy between JavaScript and CUDA across the ON and OFF peak summaries was 0.0000047 Hz. This check measures numerical agreement between the implementations.
