# significance_AI_infrastrucure_App
# AI Infrastructure Suite

A browser-based set of calculators for planning AI infrastructure. It covers training time, memory, power and cooling, network fabric, storage, and security posture. It is built from the seven-layer framework in the research paper *Significance of AI Infrastructure*.

The results are planning estimates, not vendor quotes.

## Running the app

No install or build step is needed. Open `index.html` in any modern browser.

## Files

| File | Purpose |
| --- | --- |
| `index.html` | Page structure, tabs, and inputs |
| `styles.css` | Dark theme styling |
| `calc.js` | Pure calculation engine. No DOM access, so it can be tested with Node. |
| `app.js` | UI wiring. Recalculates all outputs when any input changes. |

## Tabs

- **Dashboard:** Cluster summary and recommendations
- **Components:** Overview of the seven infrastructure layers
- **Compute:** Training time, GPU-hours, cost, and tokens per second
- **Memory:** Inference and training GPU memory needs
- **Power & Cooling:** IT load, facility load, energy cost, CO2, rack count, and cooling advice
- **Network:** Leaf-spine fabric sizing
- **Storage:** Dataset, checkpoint, and write-bandwidth needs
- **Security:** Weighted scoring of ten controls

## Shared inputs

The cluster bar at the top sets the GPU model, GPU count, PUE, and electricity price. Every tab uses these values.

## How the numbers are calculated

- **Training FLOPs:** $6 \times N \times D$, where $N$ is parameters and $D$ is training tokens.
- **Training time:** FLOPs divided by (GPU count × peak BF16 TFLOPS × MFU).
- **Inference memory:** weights × 1.2 + KV cache.
- **Training memory:** 16 bytes per parameter (mixed precision with Adam), plus an activation overhead you set.
- **Power:** GPU TDP × (1 + server overhead) × PUE. Annual energy is facility load × 8,760 hours × utilization.
- **Cooling advice:** air up to 15 kW per rack, enhanced air up to 30 kW, direct-to-chip liquid up to 50 kW, and immersion or full liquid above that.
- **Fabric:** leaf-spine sizing from switch radix and oversubscription. Designs needing more than two tiers report switch count as unavailable.
- **Storage:** checkpoint size is 14 bytes per parameter. Write bandwidth is checkpoint size divided by the write window you set.
- **Security score:** ten controls with weights totaling 100. Full counts as 100 percent of a control's weight and partial as 50 percent. Ratings are Strong (85+), Moderate (60–84), and Needs work (below 60).

## Editing values

- **GPU specifications:** Edit the `GPUS` table at the top of [calc.js](calc.js). The values are approximate public datasheet figures.
- **Constants:** Bytes per parameter, MFU defaults, and overhead factors are also set at the top of `calc.js`.

## Testing

Check syntax and run a quick calculation with Node:

```sh
node --check app.js && node --check calc.js
node -e "const C=require('./calc.js'); console.log(C.trainingEstimate({paramsB:7,tokensB:1000,gpuCount:512,gpu:'NVIDIA H100 SXM (80 GB)',mfu:0.4,pricePerGpuHour:2.5}))"
```

For example, 7B parameters, 1,000B tokens, and 512 H100s at 40 percent MFU give about 2.40 days.

## Limitations

- Estimates use simplified models. Real training runs depend on parallelism strategy, kernel efficiency, and network congestion.
- GPU specifications and prices change over time. Verify them before using results for a purchase decision.
- The fabric model assumes a symmetric leaf-spine design.
