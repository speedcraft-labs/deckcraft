# GPU shape-drag preview

DeckCraft keeps the CPU slide renderer as the canonical renderer, release path, and fallback. While dragging a single plain preset shape or picture, the editor can reuse the CPU-rendered slide background and composite the moved transparent shape with the existing eframe wgpu backend. It falls back to CPU rendering for unsupported shapes, z-order conflicts, and any state the preview cannot safely reuse. Releasing the drag settles through the CPU renderer.

The corpus check is on `HowWeRefactor.pptx`, slide 24, shape index 38, at 960×540 over 30 drag positions. The GPU image must stay within normalized RGBA mean absolute error (MAE) 0.02 both inside the selected shape's bounds and across the full editor canvas region. Each GPU benchmark invocation enforces that bound against the CPU renderer; the final settled CPU frame is also required to have the exact same pixel SHA-256 as the old path.

On an NVIDIA GeForce RTX 2080 Ti (Vulkan), the paired release run measured 12.673 ms/frame on the CPU path and 1.544 ms/frame on the GPU preview path: ratio 0.122, 95% CI 0.087–0.188, 10 runs per side, 2 warmups, 2,000 resamples. The paired report is publishable and output checked. The settled pixel SHA-256 was `8e6c6a0c79afefe0d0b9754f75cea4810b55b779a9473a9b12062ab41fabcbda`. A separate 30-frame oracle check measured selected-shape MAE 0.013351 and full-canvas MAE 0.000034, both below 0.02. The paired process CPU-time ratio was 1.186 (95% CI 1.061–1.315), so CPU overhead rises while frame wall latency falls.

Evidence:

- [`deckcraft-drag-gpu-vs-cpu.json`](../benchmarks/issue-237/deckcraft-drag-gpu-vs-cpu.json): paired timing, load guard, revisions, adapter, and output check.
- [`deckcraft-gpu-error-check.json`](../benchmarks/issue-237/deckcraft-gpu-error-check.json): adapter, per-frame timings, and CPU oracle MAE from the diagnostic pass.
