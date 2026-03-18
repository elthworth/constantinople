# B200 GPU Crash Fix - vLLM Engine Stability

## Problem
The multi-miner orchestrator was experiencing frequent crashes with:
```
ERROR Engine core proc EngineCore_DP0 died unexpectedly, shutting down client
```

## Root Cause
The B200 GPU (newer architecture) requires specific vLLM configuration to prevent crashes:
1. **Data type**: B200 is optimized for `bfloat16` instead of `float16`
2. **Memory management**: Needs conservative memory limits and fragmentation prevention
3. **Worker stability**: Requires specific worker configuration for new GPU architectures
4. **Batch limits**: Needs stricter concurrent request and batch token limits

## Changes Made

### 1. vLLM Engine Configuration (`shared_vllm_multi_miner.py`)
✅ Changed dtype from `float16` to `bfloat16` (B200 optimized)
✅ Added `max_num_seqs` and `max_num_batched_tokens` limits
✅ Enabled `prefix_caching` to reduce memory fragmentation
✅ Disabled `custom_all_reduce` (fixes multi-GPU instability)
✅ Set `worker_use_ray=False` (more stable than Ray on new architectures)
✅ Added GPU detection and detailed logging
✅ Improved error handling with specific CancelledError detection

### 2. PM2 Configuration (`ecosystem.multi-miner.config.js`)
✅ Increased `gpu-memory-utilization` from 0.62 to 0.70 (balanced for B200)
✅ Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (reduces fragmentation)
✅ Added `VLLM_WORKER_MULTIPROC_METHOD=spawn` (more stable)
✅ Increased `max_memory_restart` to 10G
✅ Added restart limits: `max_restarts: 5`, `min_uptime: 30s`
✅ Extended `listen_timeout` to 180s (B200 initialization takes longer)

### 3. Diagnostic Tools
✅ Created `check_gpu_health.sh` - monitors GPU status and diagnoses issues

## How to Apply the Fix

### Step 1: Update vLLM (Important!)
```bash
pip install -U vllm
```
The latest vLLM version has better B200 support.

### Step 2: Make the health check script executable
```bash
chmod +x check_gpu_health.sh
```

### Step 3: Check current GPU status
```bash
./check_gpu_health.sh
```

### Step 4: Restart the multi-miner with new settings
```bash
pm2 restart multi-miner-orchestrator
```

### Step 5: Monitor for stability
```bash
# Watch logs in real-time
pm2 logs multi-miner-orchestrator

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## If Crashes Still Occur

### Option A: Reduce GPU Memory Utilization
Edit `ecosystem.multi-miner.config.js` and change:
```javascript
args: '--base-port 8091 --num-miners 4 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.60 --cache-size 800 --sampling-profiles sampling_profiles_h100.json',
```

Then restart:
```bash
pm2 restart multi-miner-orchestrator
```

### Option B: Reduce Number of Miners
If memory is still an issue, reduce from 4 to 3 miners:
```javascript
args: '--base-port 8091 --num-miners 3 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.70 --cache-size 600 --sampling-profiles sampling_profiles_h100.json',
```

### Option C: Check vLLM Version
The B200 requires vLLM >= 0.4.0 for proper support:
```bash
python3 -c "import vllm; print(vllm.__version__)"
```

If below 0.4.0, update:
```bash
pip install --upgrade vllm
```

## Monitoring Commands

```bash
# Check if processes are stable (should show 'online' status)
pm2 status

# View detailed logs
pm2 logs multi-miner-orchestrator --lines 100

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Run health check
./check_gpu_health.sh
```

## Expected Behavior After Fix

✅ No more "EngineCore died unexpectedly" errors
✅ Stable operation for hours/days without restart
✅ GPU memory usage stays below 80% of total
✅ All 4 miners respond to requests consistently
✅ PM2 shows 0 restarts after initial start

## Key Configuration Values

| Setting | Old Value | New Value | Reason |
|---------|-----------|-----------|--------|
| dtype | float16 | bfloat16 | B200 optimization |
| gpu_memory_utilization | 0.62 | 0.70 | Balanced for B200 |
| disable_custom_all_reduce | (not set) | True | Multi-GPU stability |
| worker_use_ray | (not set) | False | More stable workers |
| max_num_batched_tokens | (not set) | 8192 | Prevent memory spikes |
| listen_timeout | 120000 | 180000 | B200 init takes longer |

## Technical Details

### Why bfloat16?
- B200 has optimized hardware support for bfloat16
- More numerically stable than float16
- Better handles large gradients and activations
- Recommended by NVIDIA for Blackwell architecture (B200)

### Why disable_custom_all_reduce?
- Custom all-reduce kernels may not be optimized for B200 yet
- Standard NCCL implementation is more stable
- Slight performance trade-off for reliability

### Why worker_use_ray=False?
- Ray workers add complexity and potential failure points
- Multiprocessing workers are simpler and more stable
- Better for single-GPU deployments
- Reduces dependency on Ray infrastructure

## Support

If issues persist after these fixes:
1. Run `./check_gpu_health.sh` and share the output
2. Check `pm2 logs multi-miner-orchestrator` for detailed errors
3. Verify CUDA version compatibility: `nvidia-smi`
4. Check vLLM GitHub issues for B200-specific problems

## Verification Checklist

- [ ] vLLM updated to latest version
- [ ] PM2 config updated with new settings
- [ ] Health check script is executable
- [ ] Application restarted with new configuration
- [ ] No crashes for at least 30 minutes
- [ ] GPU memory usage stable under 80%
- [ ] All 4 miners responding to health checks
- [ ] PM2 shows stable uptime with 0 restarts
