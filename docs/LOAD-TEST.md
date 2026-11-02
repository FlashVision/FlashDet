# Load & Chaos Simulation

Tool: `scripts/load_test.py`  
Environment: CPU-only container, PyTorch 2.13, Pico `repvit` checkpoint,
synthetic 128×128 images, one `Predictor` shared by a thread pool.

## Results

| Users | Concurrency | Wall (s) | RPS | p50 (ms) | p95 (ms) | p99 (ms) | Normal failures | Chaos rejected | Unexpected errors | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 8 | 0.26 | 194 | 44.9 | 49.7 | 50.3 | 0 | 5/5 | 0 | 894 MB |
| 1,000 | 32 | 6.37 | 157 | 203.1 | 223.5 | 346.1 | 0 | 20/20 | 0 | 986 MB |
| 5,000 | 64 | 34.99 | 143 | 448.3 | 491.1 | 586.5 | 0 | 50/50 | 0 | 1,117 MB |

## What did **not** break

- No process crashes, deadlocks or missing predictions.
- 1,000 and 5,000 simulated users completed with **zero normal-request failures**.
- Every malformed payload (wrong channels, wrong rank, zero-size, non-array) was
  rejected as a clean `ValueError`/`TypeError`; zero unexpected exceptions.
- Model loading from a local checkpoint took ~0.03 s; disk I/O is not the
  bottleneck in this test.
- Sharing one model across threads is safe for inference in this workload.

## What **is** breaking / where it hurts

1. **Throughput plateaus around 150 req/s on CPU.** Concurrency increases latency
   almost linearly (p50 45 → 203 → 448 ms) while RPS stays ~145–157. The predictor
   is CPU-bound; adding threads past the physical core count only adds queueing.
2. **p99 latency grows faster than p50.** At 64 workers p99 is 587 ms vs 448 ms
   p50. A web front end without timeouts will accumulate slow requests.
3. **Memory per process is ~1 GB.** With multiple worker processes (common for
   an HTTP server), a 4-worker deployment can need ~4 GB before request buffers.
   Under a cgroup limit this becomes OOM-kill, not a graceful 503.
4. **No batching.** Each request runs a separate forward pass. For high QPS,
   dynamic batching (8–32 images) is the biggest win on CPU/GPU.
5. **No request timeout/cancellation.** A single very large or stalled request
   occupies a worker thread.
6. **No rate limiting or queue depth limit.** 5,000 users are accepted at once;
   in production this is a denial-of-service lever.

## Recommended serving architecture

- One bounded process pool (workers ≈ physical cores), not unbounded threads.
- Dynamic batch queue with a max wait (e.g. 5–10 ms) and max batch size.
- Request/body size limits, image pixel limit (already enforced in
  `Predictor.preprocess`), and per-IP rate limits.
- Timeouts and a circuit breaker at the HTTP layer.
- Preload/warm the model once per worker; never load a checkpoint per request.
- Export to ONNX Runtime / TensorRT / OpenVINO for production throughput.
- Emit p50/p95/p99, queue depth, worker saturation and RSS metrics.

## Reproduce

```bash
python scripts/load_test.py --users 1000 --concurrency 32 --size 128
python scripts/load_test.py --users 5000 --concurrency 64 --size 128 \
  --output /tmp/load5000.json
```

These numbers are synthetic and hardware-dependent; use them as a relative
baseline, not an SLA.
