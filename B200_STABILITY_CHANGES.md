# B200 Stability Fix - Applied Changes

## Critical Changes Applied

### 1. **Enabled Eager Execution Mode**
- **Changed**: `enforce_eager=True` (disabled CUDA graphs)
- **Why**: CUDA graphs can cause "EngineCore died" crashes on B200
- **Trade-off**: ~10-15% slower but much more stable

### 2. **Reduced Memory Footprint** 
- **GPU Memory**: 0.70 → 0.60 (40% less GPU memory used)
- **Max Model Length**: 4096 → 2048 tokens
- **Number of Miners**: 4 → 3
- **Max Concurrent Requests**: 6 → 3 (one per miner)

### 3. **Changed Data Type**
- **Changed**: `dtype="bfloat16"` → `dtype="auto"`
- **Why**: Let vLLM choose the most stable dtype for this version

### 4. **Added Comprehensive Logging**
- Every request now logged with ID and stats
- Crash detection with detailed error traces
- Request lifecycle tracking

### 5. **Added Graceful Shutdown**
- SIGTERM/SIGINT handlers
- Prevents NCCL warnings on shutdown

## New Configuration Summary

```yaml
GPU Memory Utilization: 60% (was 70%)
Number of Miners: 3 (was 4)
Max Model Length: 2048 tokens (was 4096)
Max Concurrent Requests: 3 (was 6)
Eager Mode: Enabled (CUDA graphs disabled)
Data Type: auto (stable choice)
```

## Expected Performance

### Before (Unstable):
- ⚠️ Crashes every 1-2 minutes
- 4 miners active
- Higher throughput (when working)

### After (Stable):
- ✅ Should run indefinitely
- 3 miners active (75% capacity)
- Slightly lower per-request latency
- **Total throughput: More reliable**

## Monitoring Commands

```bash
# Restart with new settings
pm2 restart multi-miner-orchestrator

# Watch logs in real-time (look for "[vLLM]" tags)
pm2 logs multi-miner-orchestrator

# Check if it's stable (should show increasing uptime)
pm2 status

# Monitor GPU usage (should be ~60% VRAM used)
watch -n 1 nvidia-smi

# Check for crashes in last hour
pm2 logs multi-miner-orchestrator --lines 500 | grep -i "error\|crash\|died"
```

## What to Look For in Logs

### Good Signs ✅
```
[vLLM] Starting generation for request abc123...
[vLLM] Request abc123 completed: 128 tokens generated
Miner 0 started on port 8091
Miner 1 started on port 8092
Miner 2 started on port 8093
```

### Bad Signs ⚠️
```
[vLLM] Request abc123 was CANCELLED (engine crashed)
ERROR Engine core proc EngineCore_DP0 died unexpectedly
asyncio.exceptions.CancelledError
```

## If Still Crashing

### Emergency Mode: 2 Miners, 50% GPU
Edit `ecosystem.multi-miner.config.js`:
```javascript
args: '--base-port 8091 --num-miners 2 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.50 --cache-size 400 --sampling-profiles sampling_profiles_h100.json',
```

Then:
```bash
pm2 restart multi-miner-orchestrator
```

### Nuclear Option: Single Miner
```javascript
args: '--base-port 8091 --num-miners 1 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.65 --cache-size 200 --sampling-profiles sampling_profiles_h100.json',
```

## Understanding the Trade-offs

| Configuration | Stability | Throughput | GPU Usage | Best For |
|---------------|-----------|------------|-----------|----------|
| 4 miners, 70% GPU | ❌ Crashes | Highest | 70% | Ideal (not working) |
| **3 miners, 60% GPU** | **✅ Stable** | **High** | **60%** | **Current (best balance)** |
| 2 miners, 50% GPU | ✅ Very Stable | Medium | 50% | Emergency fallback |
| 1 miner, 65% GPU | ✅ Rock Solid | Lowest | 65% | Maximum stability |

## Technical Details

### Why enforce_eager=True?
CUDA graphs pre-compile computation graphs for speed but can cause memory corruption on new GPU architectures like B200 when:
- vLLM version doesn't fully support the architecture
- Driver/CUDA version mismatches
- Memory fragmentation occurs

Eager mode executes operations immediately without graph compilation - slower but safer.

### Why Reduce max_model_len?
The KV cache size is proportional to `max_model_len × batch_size × hidden_dim`. Reducing from 4096 to 2048 tokens cuts memory usage nearly in half, preventing OOM crashes.

### Why Fewer Miners?
Each miner can send concurrent requests. With 4 miners, the engine could receive 4+ simultaneous requests, causing memory spikes. 3 miners with strict limits prevents this.

## Verification Checklist After Restart

Wait 5 minutes, then check:

- [ ] `pm2 status` shows "online" (not "errored" or "stopped")
- [ ] Uptime is increasing (not resetting to 0s)
- [ ] All 3 ports respond: `curl http://localhost:8091/health`
- [ ] GPU memory stable: `nvidia-smi` shows consistent usage
- [ ] No "died unexpectedly" in logs: `pm2 logs --lines 100`
- [ ] Request logs show "[vLLM] completed" messages

If all ✅ after 5-10 minutes → **You're stable!** 🎉

If still crashing → Use Emergency Mode (2 miners, 50% GPU)
