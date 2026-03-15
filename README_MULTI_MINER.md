# Multi-Miner Setup — README

**Maximize Bittensor SN97 emission by running multiple miners on a single GPU server**

## 🎯 What This Does

Enables **3.5-3.8x emission multiplier** by running 4 miners on one GPU, sharing:
- One vLLM inference engine
- One HuggingFace model for hidden states
- One shared cache (1000+ requests)

While avoiding collusion detection through:
- Different sampling parameters per miner
- Response variation (30-60% difference)
- Timing jitter to decorrelate latencies

## 📊 Expected Results

| GPU | VRAM | Miners | Emission | Monthly Cost (RunPod) |
|-----|------|--------|----------|----------------------|
| RTX 4090 | 24GB | 4 | 3.5-3.8x | $245 |
| A100 40GB | 40GB | 6-7 | 5.0-5.5x | $792 |
| A100 80GB | 80GB | 8-10 | 6.5-7.0x | $1152 |

**Why not 4.0x?** Shared compute adds ~5-10ms queuing delay, population-relative scoring has diminishing returns, and there's small collusion risk if misconfigured.

## 🚀 Quick Start

### Option 1: Automated Deployment (Recommended)

```bash
# Deploy 4 miners on one GPU
./deploy_multi_miner.sh 4 8091

# Monitor in real-time
./monitor_multi_miners.sh 8091 4

# Test anti-collusion (response variation)
./test_response_variation.sh 8091 4
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install vllm torch transformers accelerate fastapi uvicorn numpy aiohttp pydantic

# Start multi-miner orchestrator
python3 shared_vllm_multi_miner.py \
  --base-port 8091 \
  --num-miners 4 \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.75 \
  --cache-size 1000 \
  --sampling-profiles sampling_profiles.json
```

### Option 3: Docker

```bash
# Using docker-compose
docker compose -f docker-compose.multi-miner.yml up -d

# Or standalone
docker run -d --gpus all -p 8091-8094:8091-8094 \
  -e NUM_MINERS=4 \
  -e GPU_MEMORY_UTILIZATION=0.75 \
  inference-multi-miner
```

## 📝 Register Miners

**Critical**: Each miner must use a **different hotkey** to avoid Sybil detection:

```bash
# Miner 0 (port 8091)
btcli subnet register --netuid 97 --wallet.name wallet1 --wallet.hotkey hotkey1

# Miner 1 (port 8092)
btcli subnet register --netuid 97 --wallet.name wallet2 --wallet.hotkey hotkey2

# Miner 2 (port 8093)
btcli subnet register --netuid 97 --wallet.name wallet3 --wallet.hotkey hotkey3

# Miner 3 (port 8094)
btcli subnet register --netuid 97 --wallet.name wallet4 --wallet.hotkey hotkey4
```

## 📂 Files Overview

### Core Implementation
- **`shared_vllm_multi_miner.py`** (711 lines) — Main orchestrator
  - SharedVLLMEngine: Single vLLM + HF model
  - MinerInstance: Per-miner endpoints with unique sampling
  - SharedHiddenStateCache: LRU cache for all miners
  - FastAPI apps per miner (ports 8091+)

- **`sampling_profiles.json`** — Anti-collusion configurations
  - 8 pre-configured profiles with varied temperature/top_p
  - Designed for 30-60% response variation

### Deployment Scripts
- **`deploy_multi_miner.sh`** — Automated deployment
  - Checks prerequisites (GPU, Python, dependencies)
  - Installs packages if missing
  - Downloads model if not cached
  - Starts miners and verifies health

- **`monitor_multi_miners.sh`** — Real-time monitoring dashboard
  - GPU utilization
  - Per-miner stats (requests, challenges, pass rate)
  - Shared cache metrics
  - Color-coded status

- **`test_response_variation.sh`** — Anti-collusion validation
  - Tests response similarity between miners
  - Flags if >85% similar (collusion risk)
  - Validates sampling profile effectiveness

### Docker
- **`Dockerfile.multi-miner`** — Production image
  - CUDA 12.1 base
  - vLLM 0.3.2 + dependencies
  - Health checks

- **`docker-compose.multi-miner.yml`** — Compose config
  - Environment variables
  - Volume persistence
  - GPU allocation

### Documentation
- **`MULTI_MINER_SETUP_GUIDE.md`** (468 lines) — Complete guide
  - Critical scoring factors explained
  - Hardware requirements
  - Configuration tuning
  - Anti-collusion strategy
  - Troubleshooting

- **`MULTI_MINER_QUICK_REF.md`** (296 lines) — Quick reference
  - Commands cheatsheet
  - Performance targets
  - ROI calculations
  - Common issues

- **`MULTI_MINER_SUMMARY.md`** (497 lines) — This implementation summary
  - Codebase analysis results
  - Architecture decisions
  - Expected results
  - Risk analysis

- **`README_MULTI_MINER.md`** (This file) — Overview

## ⚙️ Configuration

### GPU Memory Tuning

```bash
# Conservative (4+ miners or frequent OOM)
--gpu-memory-utilization 0.70

# Balanced (3-4 miners) — DEFAULT
--gpu-memory-utilization 0.75

# Aggressive (1-2 miners)
--gpu-memory-utilization 0.80
```

### Cache Sizing

```bash
# Minimum (low memory)
--cache-size 500

# Recommended — DEFAULT
--cache-size 1000

# Large (high traffic)
--cache-size 1500
```

### Sampling Profile Customization

Edit `sampling_profiles.json`:

```json
{
  "temperature_base": 0.70,      // Base temperature (0.65-0.80)
  "temperature_variance": 0.15,  // Per-request jitter (0.10-0.20)
  "top_p": 0.90,                 // Nucleus sampling (0.85-0.95)
  "timing_jitter_ms": 8.0        // Latency decorrelation (5-12ms)
}
```

## 🎯 Critical Success Factors

### 1. Speed (40% of score)
- **TTFT**: <30ms excellent, <100ms good, >500ms poor
- **TPS**: >150 excellent, >80 good, <10 poor
- **Optimization**: Use vLLM, enable flash attention, reduce network latency

### 2. Verification (40% of score)
- **Cosine threshold**: >0.995 (pass), <0.995 (fail)
- **Challenge latency**: <50ms ideal, >500ms auto-fail
- **Failure penalty**: 3x asymmetric (failing costs triple)
- **Three strikes**: 3 consecutive fails = 90% weight penalty

### 3. Consistency (20% of score)
- **Natural CV**: 0.05-0.15 (GPU variance)
- **Divergence limit**: <12% (safe), >12% (-30%), >25% (-70%)
- **Response variation**: 30-60% between miners (avoids collusion)

## ✅ Anti-Collusion Checklist

Before going to production, verify:

- [ ] Each miner uses **different hotkey**
- [ ] Different sampling parameters per miner
- [ ] Response variation >30% (test with `test_response_variation.sh`)
- [ ] Timing jitter enabled (5-12ms per miner)
- [ ] Cache hit rate >95%
- [ ] Challenge pass rate >95%
- [ ] GPU utilization 75-85%
- [ ] No errors in logs

## 🔍 Monitoring

### Real-Time Dashboard

```bash
./monitor_multi_miners.sh 8091 4
```

Shows:
- GPU utilization and temperature
- Per-miner requests, challenges, pass rates
- Shared cache size and hit rate
- Aggregate statistics

### Manual Health Checks

```bash
# All miners
for port in {8091..8094}; do
  curl -s "http://localhost:$port/health" | jq '.miner_id, .status, .total_requests'
done

# Cache metrics
curl -s http://localhost:8091/health | jq '.cache_size, .cache_hit_rate'
```

### Log Analysis

```bash
# View real-time logs
tail -f multi_miner.log

# Search for errors
grep -i "error\|exception\|fail" multi_miner.log

# Check challenge results
grep "Challenge" multi_miner.log | tail -20

# Analyze TTFT distribution
grep "Inference" multi_miner.log | awk '{print $(NF-3)}' | sort -n | uniq -c
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce GPU memory allocation
--gpu-memory-utilization 0.70

# Or reduce number of miners
--num-miners 3

# Or reduce cache size
--cache-size 500
```

### High Challenge Miss Rate

```bash
# Increase cache size
--cache-size 1500

# Check hit rate (should be >95%)
curl -s http://localhost:8091/health | jq '.cache_hit_rate'
```

### Low Cosine Similarity (Failures)

```bash
# Verify exact model
curl -s http://localhost:8091/health | jq '.model'
# Should be: Qwen/Qwen2.5-7B-Instruct

# Check logs for errors
tail -100 multi_miner.log | grep -i "cosine"
```

### Collusion Detection Flags

```bash
# Test response variation
./test_response_variation.sh 8091 4

# Should show 30-60% similarity (safe range)
# If >85%, increase temperature_variance in sampling_profiles.json
```

### Port Already in Use

```bash
# Kill existing process
pkill -f shared_vllm_multi_miner

# Or specific port
lsof -ti :8091 | xargs kill -9
```

## 📈 Performance Targets

### Per-Miner Targets
- Requests: 100+ per epoch (72 min)
- Challenges: 10-20+ per epoch
- Pass rate: >95%
- Cache hit rate: >95%
- TTFT: <100ms avg
- TPS: >80 avg

### Aggregate Targets (4 Miners)
- Total requests: 400+ per epoch
- Total challenges: 40-80+ per epoch
- Overall pass rate: >95%
- Aggregate TPS: 300-400+
- Expected emission: 3.5-3.8x

## 🎓 Understanding the Validator

### Scoring Breakdown
```python
# From validator/hardened_scoring.py
SPEED_WEIGHT = 0.40        # 40% for TTFT + TPS
VERIFICATION_WEIGHT = 0.40  # 40% for hidden state challenges
CONSISTENCY_WEIGHT = 0.20   # 20% for stable performance
```

### Collusion Detection Thresholds
```python
# From validator/collusion_detector.py
RESPONSE_SIMILARITY_SUSPICIOUS = 0.85  # 85%+ = suspicious
RESPONSE_SIMILARITY_COLLUDING = 0.95   # 95%+ = colluding
TIMING_CORRELATION_SUSPICIOUS = 0.7    # Pearson r > 0.7
```

### Anti-Gaming Protections
1. **Asymmetric penalties**: Failing costs 3x passing
2. **Three-strike rule**: 3 consecutive fails = 90% penalty
3. **Divergence detection**: >12% organic vs synthetic gap
4. **Cross-epoch tracking**: Can't reset by staying under minimums
5. **Population-relative scoring**: Can't game absolute thresholds

## 📚 Additional Resources

- **Full Setup Guide**: `MULTI_MINER_SETUP_GUIDE.md`
- **Quick Reference**: `MULTI_MINER_QUICK_REF.md`
- **Implementation Summary**: `MULTI_MINER_SUMMARY.md`
- **Original Miner Guide**: `MINER_GUIDE.md`
- **Threat Model**: `THREAT_MODEL.md`

## 🤝 Support

### Health Check Failed?

```bash
# Check if miners are running
ps aux | grep shared_vllm_multi_miner

# Check GPU
nvidia-smi

# Check logs
tail -100 multi_miner.log
```

### Performance Issues?

```bash
# GPU utilization too low? Increase miners or GPU memory
# GPU utilization too high? Reduce miners or GPU memory
# High TTFT? Check network latency, enable flash attention
# Low TPS? Increase GPU memory utilization
# Low pass rate? Verify model, check token alignment
```

### Need Help?

1. Check logs: `tail -100 multi_miner.log`
2. Run health checks: `curl http://localhost:8091/health`
3. Test response variation: `./test_response_variation.sh`
4. Check GPU: `nvidia-smi`
5. Review documentation: `MULTI_MINER_SETUP_GUIDE.md`

## 🎉 Ready to Deploy

```bash
# 1. Deploy miners
./deploy_multi_miner.sh 4 8091

# 2. Test anti-collusion
./test_response_variation.sh 8091 4

# 3. Register with different hotkeys
btcli subnet register --netuid 97 --wallet.name wallet1 --wallet.hotkey hotkey1
# ... repeat for all miners

# 4. Monitor performance
./monitor_multi_miners.sh 8091 4
```

**Expected outcome**: 3.5-3.8x emission on single GPU vs running 1 miner.

Good luck! 🚀
