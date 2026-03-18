# B200 Runtime Crash Fix - Complete Diagnostic Suite

## Problem Analysis

Your B200 GPU experiences **runtime crashes** (not startup failures). The pattern:
1. ✅ App starts successfully 
2. ✅ Runs for 15-20 minutes
3. ❌ `EngineCore_DP0 died unexpectedly` crash
4. 🔄 PM2 auto-restarts

**Root Causes Identified:**
1. **NCCL initialization issues** - Single-GPU setup shouldn't use NCCL
2. **Memory accumulation** - Possible memory leak over time
3. **GPU memory fragmentation** - Requests causing memory spikes
4. **Async cancellation** - Requests not completing cleanly

## 🔧 Fixes Applied

### 1. **NCCL Configuration** (Major Fix)
Added environment variables to disable unnecessary NCCL features:
- `NCCL_P2P_DISABLE=1` - No peer-to-peer on single GPU
- `NCCL_SHM_DISABLE=1` - Disable shared memory (stability)
- `TORCH_NCCL_BLOCKING_WAIT=1` - Prevent async hangs

### 2. **GPU Memory Monitoring**
- Automatic memory checks every 60 seconds
- Memory leak detection (tracks increasing trend)
- Auto garbage collection at 85% memory usage
- Detailed logging of memory patterns

### 3. **Request Tracking**
- Every request logged with ID and timing
- Success/failure rate tracking
- Crash detection with context
- Aggregate statistics

### 4. **Health Monitoring System**
- Separate monitor process checks endpoints every 30s
- Detects crashes immediately
- Tracks GPU and system memory
- Logs to `/tmp/vllm_health_stats.json`

## 🚀 Quick Start

### Option 1: Start with Automated Monitoring (Recommended)
```bash
./start_with_monitoring.sh
```

This will:
- Stop existing processes
- Start multi-miner orchestrator
- Wait for initialization
- Start health monitor
- Verify all endpoints

### Option 2: Manual Start
```bash
# Start main app
pm2 restart multi-miner-orchestrator

# Start monitor (in separate terminal or background)
python3 monitor_vllm_health.py --interval 30 &
```

## 📊 Monitoring Commands

### Real-time Logs
```bash
# Main application logs
pm2 logs multi-miner-orchestrator

# Health monitor logs
pm2 logs vllm-health-monitor

# Both together
pm2 logs
```

### Check Health Status
```bash
# Quick check all endpoints
curl -s http://localhost:8091/health | python3 -m json.tool
curl -s http://localhost:8092/health | python3 -m json.tool
curl -s http://localhost:8093/health | python3 -m json.tool

# View health stats file
cat /tmp/vllm_health_stats.json | python3 -m json.tool
```

### GPU Monitoring
```bash
# Watch GPU usage live
watch -n 1 nvidia-smi

# Check GPU memory trend
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5
```

## 🔍 Diagnostic Tools

### Run Full Diagnostic Report
```bash
./diagnose_crash.sh
```

This comprehensive report includes:
1. PM2 process status and restart count
2. GPU memory and utilization
3. Recent error logs
4. EngineCore crash events
5. NCCL warnings
6. Health endpoint status
7. Memory leak detection
8. Crash timestamps
9. Environment variables
10. Recommendations

### What the Logs Tell You

#### Good Signs ✅
```
[vLLM] Starting generation for request abc12345...
[vLLM] Request abc12345 completed: 128 tokens generated
[GPU Memory] 85.23GB / 191.5GB (44.5%)
[Stats] 150 requests | 100.0% success | 8.5 req/min
```

#### Warning Signs ⚠️
```
⚠️  HIGH GPU MEMORY: 86.2% - close to OOM!
⚠️  MEMORY LEAK DETECTED: Steady increase over last 5 checks
   Memory trend: [78.1, 79.3, 81.2, 83.5, 86.2]
```

#### Crash Detection 🚨
```
[vLLM] Request abc12345 was CANCELLED (engine crashed or shutdown)
🚨 ENGINE CRASH DETECTED 🚨
Stats before crash: {'total_requests': 234, 'success_rate': 99.1%, ...}
ERROR Engine core proc EngineCore_DP0 died unexpectedly
```

## 🛠️ Troubleshooting by Symptom

### Symptom: Crashes after 15-20 minutes
**Diagnosis**: Memory leak or fragmentation  
**Solution:**
```bash
# Reduce GPU memory utilization
# Edit ecosystem.multi-miner.config.js:
args: '... --gpu-memory-utilization 0.50 ...'

pm2 restart multi-miner-orchestrator
```

### Symptom: "EngineCore died" immediately on first request
**Diagnosis**: GPU memory too high  
**Solution:**
```bash
# Reduce to 2 miners
# Edit ecosystem.multi-miner.config.js:
args: '--base-port 8091 --num-miners 2 --gpu-memory-utilization 0.50 ...'

pm2 restart multi-miner-orchestrator
```

### Symptom: NCCL warnings in logs
**Diagnosis**: NCCL env vars not applied  
**Solution:**
```bash
# Verify environment variables are set
pm2 show multi-miner-orchestrator | grep -A 30 "env:"

# Should see NCCL_P2P_DISABLE=1, etc.
# If not, edit ecosystem.multi-miner.config.js and restart
```

### Symptom: GPU memory keeps increasing
**Diagnosis**: Memory leak  
**Solution:**
```bash
# Check memory trend in health stats
cat /tmp/vllm_health_stats.json | python3 -m json.tool

# If confirmed leak, this is a vLLM bug - workaround:
# Set aggressive restart threshold
# Edit ecosystem.multi-miner.config.js:
max_memory_restart: '8G',  # Restart if RAM exceeds 8GB

pm2 restart multi-miner-orchestrator
```

### Symptom: Random crashes, no clear pattern
**Diagnosis**: Race condition or async issue  
**Solution:**
```bash
# Reduce concurrent requests to absolute minimum
# Edit ecosystem.multi-miner.config.js:
args: '--base-port 8091 --num-miners 2 --gpu-memory-utilization 0.50 ...'

# This gives max_concurrent_requests = 2 (one per miner)
pm2 restart multi-miner-orchestrator
```

## 📈 Understanding the Stats

### Engine Stats (from health endpoint)
```json
{
  "engine_stats": {
    "uptime_seconds": 1234,
    "total_requests": 456,
    "successful_requests": 452,
    "failed_requests": 4,
    "success_rate": 99.1,
    "requests_per_minute": 22.3
  }
}
```

- **uptime_seconds**: Time since engine initialization (resets on restart)
- **success_rate**: Should be >95%, otherwise investigate errors
- **requests_per_minute**: Normal range: 5-30 rpm depending on load

### GPU Memory Stats
```json
{
  "gpu_memory": {
    "allocated_gb": 42.5,
    "reserved_gb": 85.2,
    "total_gb": 191.5,
    "utilization_pct": 44.5
  }
}
```

- **allocated_gb**: Active tensors in use
- **reserved_gb**: Memory reserved by CUDA (includes cache)
- **utilization_pct**: Should stay <80% for stability

## 🔄 Recovery Procedures

### If Crashes Persist: Emergency Mode

**Step 1: Minimal Configuration**
```javascript
// ecosystem.multi-miner.config.js
args: '--base-port 8091 --num-miners 2 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.45 --cache-size 300 --sampling-profiles sampling_profiles_h100.json',
```

**Step 2: Restart**
```bash
pm2 restart multi-miner-orchestrator
```

**Step 3: Monitor for 30 minutes**
```bash
pm2 logs multi-miner-orchestrator | grep -E "vLLM|Stats|GPU Memory"
```

**Step 4: If stable, gradually increase**
- After 1 hour stable: increase to 0.50 GPU memory
- After 2 hours stable: increase to 3 miners
- After 24 hours stable: increase to 0.55 GPU memory

### Nuclear Option: Single Miner Mode
```javascript
args: '--base-port 8091 --num-miners 1 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.60 --cache-size 200 --sampling-profiles sampling_profiles_h100.json',
```

This should **never crash**. If it does, the issue is:
- vLLM version incompatibility with B200
- CUDA driver issue
- Hardware problem

## 📝 What to Report if Issues Continue

If crashes persist after all fixes:

1. **Run diagnostic report:**
   ```bash
   ./diagnose_crash.sh > crash_report.txt
   ```

2. **Collect PM2 logs:**
   ```bash
   pm2 logs multi-miner-orchestrator --lines 500 > pm2_logs.txt
   ```

3. **Check vLLM version:**
   ```bash
   python3 -c "import vllm; print(vllm.__version__)"
   ```

4. **Check if it's a known issue:**
   - Search vLLM GitHub issues for "B200"
   - Check for "EngineCore died" issues
   - Look for NCCL + single GPU problems

## 🎯 Expected Outcome

After applying all fixes:

| Metric | Target | Good | Needs Work |
|--------|--------|------|------------|
| Uptime between crashes | >24 hours | >8 hours | <2 hours |
| Success rate | >99% | >95% | <90% |
| GPU memory (stable) | <75% | <85% | >85% |
| Memory leak rate | 0% increase | <2% per hour | >5% per hour |

**If achieving "Good" or better: You're stable enough for production**

**If in "Needs Work": Apply Emergency Mode settings**

## 📞 Quick Reference

```bash
# Start with monitoring
./start_with_monitoring.sh

# Check status
pm2 status

# View logs
pm2 logs multi-miner-orchestrator

# Diagnose crashes
./diagnose_crash.sh

# Check health
curl http://localhost:8091/health | python3 -m json.tool

# View health monitor stats
cat /tmp/vllm_health_stats.json | python3 -m json.tool

# Restart
pm2 restart multi-miner-orchestrator

# Stop everything
pm2 stop all
```

---

**Remember**: The B200 is a very new GPU. Some instability is expected until vLLM is fully optimized for it. The monitoring tools will help identify exactly what's failing.
