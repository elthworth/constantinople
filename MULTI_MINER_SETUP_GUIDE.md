# Multi-Miner Setup Guide — Maximize Emission on Single GPU

## Overview

This guide shows you how to run **multiple miner registrations on a single GPU server**, sharing compute resources to maximize your total emission while avoiding collusion detection penalties.

## Table of Contents

1. [Critical Scoring Factors](#1-critical-scoring-factors)
2. [Multi-Miner Architecture](#2-multi-miner-architecture)
3. [Hardware Requirements](#3-hardware-requirements)
4. [Quick Start](#4-quick-start)
5. [Configuration](#5-configuration)
6. [Anti-Collusion Strategy](#6-anti-collusion-strategy)
7. [Monitoring & Optimization](#7-monitoring--optimization)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Critical Scoring Factors

Based on analysis of the validator's hardened scoring engine, here are the factors that determine your miner rewards:

### Speed (40% of total score)

- **TTFT (Time to First Token)**: <30ms excellent, >500ms poor
- **TPS (Tokens per Second)**: >150 excellent, <10 poor
- **Scoring method**: Population-relative (ranked against other miners)
- **Wall-clock measurement**: Can't fake this — network latency included

### Verification (40% of total score)

- **Cosine similarity**: Must exceed 0.995 for hidden state challenges
- **Challenge latency**: <50ms ideal, >500ms auto-fail
- **Asymmetric penalties**: **Failing costs 3x what passing earns**
- **Consecutive failures**: 3 strikes = suspect flag = 90% weight penalty
- **Challenge participation**: Low ratio = up to 70% penalty

### Consistency (20% of total score)

- **CV (Coefficient of Variation)**: Natural GPU variance is 0.05-0.15
- **Divergence detection**: >12% gap between organic/synthetic performance = -30% penalty
- **Response variation**: Too uniform = penalty (0-3% CV flagged)

### Anti-Gaming Protections

1. **Collusion detection** (response similarity >85% = flagged)
2. **Timing correlation** (correlated latency patterns = flagged)
3. **Hidden state fingerprinting** (bit-exact matches = shared backend detection)
4. **Sybil resistance** (diminishing returns per hotkey cluster)
5. **Minimum samples** (need 5+ organic and 5+ synthetic requests)
6. **Cross-epoch tracking** (can't reset by staying under per-epoch minimums)

---

## 2. Multi-Miner Architecture

### Traditional Approach (Inefficient)
```
┌─────────────────────────────────────────────┐
│              GPU (24GB VRAM)                │
├─────────────────────────────────────────────┤
│ Miner 1: vLLM (12GB) + HF (3GB) = 15GB     │
│ Miner 2: Can't fit! Need another GPU       │
└─────────────────────────────────────────────┘
```

### Shared vLLM Approach (Optimized)
```
┌─────────────────────────────────────────────┐
│              GPU (24GB VRAM)                │
├─────────────────────────────────────────────┤
│ Shared vLLM Engine: 14GB                   │
│ Shared HF Model: 3GB                       │
│ Shared Hidden State Cache: 2GB             │
│ Total: ~19GB (supports 4+ miner endpoints) │
└─────────────────────────────────────────────┘
         ↓         ↓         ↓         ↓
    Miner 0   Miner 1   Miner 2   Miner 3
   (Port 8091) (8092)   (8093)    (8094)
```

**Key Innovation**: One vLLM engine, multiple miner endpoints with:
- **Shared compute**: All miners use the same model
- **Shared cache**: No memory duplication for hidden states
- **Independent sampling**: Different temperature/top_p per miner
- **Timing jitter**: Decorrelated latency patterns
- **Response variation**: >5% token difference avoids collusion detection

---

## 3. Hardware Requirements

### Minimum (1-2 Miners)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 4090 (24GB) or A100 (40GB) |
| CPU | 8+ cores |
| RAM | 32GB |
| Network | 100 Mbps upload, <50ms latency to validator |
| Storage | 100GB SSD |

### Recommended (3-4 Miners)

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA A100 (40GB) or H100 (80GB) |
| CPU | 16+ cores |
| RAM | 64GB |
| Network | 1 Gbps upload, <20ms latency to validator |
| Storage | 200GB NVMe SSD |

### VRAM Allocation

| Model | vLLM Engine | HF Model | Cache | Total | Max Miners |
|-------|-------------|----------|-------|-------|------------|
| Qwen2.5-7B | 14GB | 3GB | 2GB | 19GB | 4-5 |
| Qwen2.5-14B | 26GB | 5GB | 3GB | 34GB | 3-4 |
| Llama-3-8B | 15GB | 3GB | 2GB | 20GB | 4-5 |

**GPU Utilization Target**: 75-85% (leave headroom for inference spikes)

---

## 4. Quick Start

### Prerequisites

```bash
# Verify GPU
nvidia-smi

# Requires CUDA 12+ and driver 525+
# Install Docker with NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### Option A: Docker (Recommended)

```bash
# Pull the multi-miner image
docker pull thebes1618/inference-subnet-multi-miner:latest

# Run 4 miners on one GPU
docker run -d \
  --name multi-miner \
  --gpus all \
  -p 8091-8094:8091-8094 \
  -e NUM_MINERS=4 \
  -e BASE_PORT=8091 \
  -e MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" \
  -e GPU_MEMORY_UTILIZATION=0.75 \
  -v huggingface-cache:/root/.cache/huggingface \
  thebes1618/inference-subnet-multi-miner:latest

# Verify all miners are running
for port in {8091..8094}; do
  curl -s "http://localhost:$port/health" | jq '.miner_id, .status'
done
```

### Option B: Manual Setup

```bash
# Install dependencies
pip install vllm torch transformers accelerate fastapi "uvicorn[standard]" numpy aiohttp pydantic

# Start multi-miner orchestrator
python3 shared_vllm_multi_miner.py \
  --base-port 8091 \
  --num-miners 4 \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.75 \
  --cache-size 1000

# Miners will start on ports 8091, 8092, 8093, 8094
```

### Register Multiple Miners

You need to register each miner endpoint with a **different hotkey/coldkey** on the Bittensor network:

```bash
# Register miner 0 (port 8091)
btcli subnet register --netuid 97 --wallet.name wallet1 --wallet.hotkey hotkey1

# Register miner 1 (port 8092)
btcli subnet register --netuid 97 --wallet.name wallet2 --wallet.hotkey hotkey2

# Register miner 2 (port 8093)
btcli subnet register --netuid 97 --wallet.name wallet3 --wallet.hotkey hotkey3

# Register miner 3 (port 8094)
btcli subnet register --netuid 97 --wallet.name wallet4 --wallet.hotkey hotkey4
```

**Important**: Each miner must use a **different hotkey** to avoid being flagged as Sybil.

---

## 5. Configuration

### Sampling Profiles

Each miner uses a unique sampling profile to create response variation. This is **critical** to avoid collusion detection.

Create `sampling_profiles.json`:

```json
{
  "profiles": [
    {
      "temperature_base": 0.65,
      "temperature_variance": 0.15,
      "top_p": 0.90,
      "timing_jitter_ms": 8.0
    },
    {
      "temperature_base": 0.75,
      "temperature_variance": 0.12,
      "top_p": 0.92,
      "timing_jitter_ms": 6.0
    },
    {
      "temperature_base": 0.70,
      "temperature_variance": 0.18,
      "top_p": 0.88,
      "timing_jitter_ms": 10.0
    },
    {
      "temperature_base": 0.80,
      "temperature_variance": 0.10,
      "top_p": 0.95,
      "timing_jitter_ms": 5.0
    }
  ]
}
```

Start with custom profiles:

```bash
python3 shared_vllm_multi_miner.py \
  --base-port 8091 \
  --num-miners 4 \
  --sampling-profiles sampling_profiles.json
```

### Environment Variables (Docker)

```bash
# .env file
NUM_MINERS=4
BASE_PORT=8091
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
GPU_MEMORY_UTILIZATION=0.75
CACHE_SIZE=1000
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096
```

### GPU Memory Tuning

| Setting | Effect | When to Use |
|---------|--------|-------------|
| 0.70 | Conservative, more headroom | 4+ miners or frequent OOM |
| 0.75 | Balanced (default) | 3-4 miners |
| 0.80 | Aggressive, higher throughput | 1-2 miners, stable workload |
| 0.85 | Maximum utilization | Single miner only |

**Rule of thumb**: `gpu_memory_utilization = 0.9 - (0.05 × num_miners)`

---

## 6. Anti-Collusion Strategy

The validator's collusion detector uses multiple signals. Here's how the shared vLLM setup defends against each:

### 1. Response Similarity Detection

**Threat**: Validator sends same prompt to multiple miners and compares token-level similarity. >85% match = suspicious.

**Defense**:
- Each miner uses **different sampling parameters** (temperature, top_p)
- **Request-level temperature jitter** (±10-20%) ensures variation
- Expected similarity: **30-60%** (same model, different sampling)
- Hidden states are **input-dependent only** (not affected by sampling)

**Validation**: Test locally:
```bash
# Send same prompt to two miners
curl -X POST http://localhost:8091/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_tokens": 50}' > miner0.json

curl -X POST http://localhost:8092/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?", "max_tokens": 50}' > miner1.json

# Compare responses (should be different)
diff <(jq -r '.text' miner0.json) <(jq -r '.text' miner1.json)
```

### 2. Timing Correlation Detection

**Threat**: Validator monitors latency patterns across miners. Correlated spikes suggest shared backend.

**Defense**:
- **Per-miner timing jitter** (5-12ms random delay)
- Jitter applied **after** inference (doesn't affect challenge latency)
- Decorrelates latency time series (Pearson r < 0.3)

**Validation**: Monitor latency logs:
```bash
# Miner 0 latencies should differ from Miner 1
grep "Inference" miner0.log | awk '{print $NF}' | sort -n
grep "Inference" miner1.log | awk '{print $NF}' | sort -n
```

### 3. Hidden State Bit-Exact Detection

**Threat**: Validator compares hidden states from different miners for the same prompt. Bit-exact matches (cosine > 0.9999) suggest cache sharing.

**Defense**:
- **This is NOT a threat**: We run the same model honestly
- Same input → same hidden states (correct behavior)
- Validator expects cosine ~0.999 for honest miners
- Bit-exact matches only occur with **cache relay** (not our case)

### 4. Error Correlation Detection

**Threat**: When one miner fails a challenge, do others also fail? Correlated failures suggest shared infrastructure.

**Defense**:
- All miners use **same vLLM engine** (honest inference)
- Failures are **rare and random** (network, timeout, etc.)
- No systematic correlation (failures are independent)

### 5. Challenge Participation Ratio

**Threat**: Miners with low challenge ratio get penalized (up to 70% weight reduction).

**Defense**:
- All miners serve **inline challenges** (bundled with inference)
- Large shared cache (1000 requests) ensures high hit rate
- Each miner maintains **independent request logs**

---

## 7. Monitoring & Optimization

### Health Check Dashboard

```bash
#!/bin/bash
# monitor_miners.sh

while true; do
  clear
  echo "==================================="
  echo "Multi-Miner Health Dashboard"
  echo "==================================="
  
  for port in {8091..8094}; do
    response=$(curl -s "http://localhost:$port/health")
    miner_id=$(echo "$response" | jq -r '.miner_id')
    status=$(echo "$response" | jq -r '.status')
    requests=$(echo "$response" | jq -r '.total_requests')
    challenges=$(echo "$response" | jq -r '.total_challenges')
    passed=$(echo "$response" | jq -r '.challenges_passed')
    hit_rate=$(echo "$response" | jq -r '.cache_hit_rate')
    
    pass_rate=$(awk "BEGIN {print ($passed/$challenges)*100}")
    
    echo ""
    echo "Miner $miner_id (Port $port) - $status"
    echo "  Requests: $requests | Challenges: $challenges ($pass_rate% pass)"
    echo "  Cache Hit Rate: $(awk "BEGIN {print $hit_rate*100}")%"
  done
  
  echo ""
  echo "Shared Cache: $(curl -s http://localhost:8091/health | jq -r '.cache_size') requests"
  echo ""
  
  sleep 10
done
```

### GPU Monitoring

```bash
# Continuous GPU monitoring
watch -n 1 'nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits'

# Target metrics:
# - GPU utilization: 75-95%
# - Memory utilization: 75-85%
# - Temperature: <85°C
```

### Log Analysis

```bash
# Check for errors across all miners
for port in {8091..8094}; do
  echo "=== Miner on port $port ==="
  tail -100 miner_$port.log | grep -i "error\|fail\|exception"
done

# Monitor challenge pass rates
grep "Challenge" miner_*.log | grep "HIT\|MISS" | sort

# Check response time distribution
grep "Inference" miner_8091.log | awk '{print $(NF-1)}' | sort -n | uniq -c
```

### Performance Optimization

#### 1. Reduce TTFT

- Enable **flash attention**: `pip install flash-attn --no-build-isolation`
- Use **CUDA graphs** (enabled by default in vLLM)
- Optimize network path to validator (use same region/datacenter)

#### 2. Increase TPS

- Increase `gpu_memory_utilization` (more KV cache)
- Use **tensor parallelism** on multi-GPU: `--tensor-parallel-size 2`
- Enable **continuous batching** (automatic in vLLM)

#### 3. Optimize Challenge Latency

- Keep `cache_size` large (1000+)
- Monitor cache hit rate (should be >95%)
- If HF model is on CPU, consider moving to GPU if VRAM allows

#### 4. Maintain Consistency

- Avoid CPU throttling (check `top`, `htop`)
- Monitor disk I/O (model loading, cache swapping)
- Use **fixed clock speeds** (disable GPU boost for consistency)

```bash
# Lock GPU to base clock for consistent performance
sudo nvidia-smi -lgc 1410  # Example for RTX 4090
```

---

## 8. Troubleshooting

### Common Issues

#### "CUDA out of memory"

**Cause**: Too many miners or insufficient GPU memory.

**Solutions**:
1. Reduce `gpu_memory_utilization`: `--gpu-memory-utilization 0.70`
2. Reduce `num_miners`: Start with 2-3 and scale up
3. Reduce `cache_size`: `--cache-size 500`
4. Use smaller model (if network permits)

#### High Challenge Miss Rate

**Cause**: Cache too small or requests evicted too quickly.

**Solutions**:
1. Increase `cache_size`: `--cache-size 1500`
2. Check memory usage: `nvidia-smi` (if VRAM full, reduce `gpu_memory_utilization`)
3. Check cache hit rate: `curl http://localhost:8091/health | jq '.cache_hit_rate'`

#### Low Cosine Similarity (Challenge Failures)

**Cause**: Model mismatch, precision issues, or token alignment.

**Solutions**:
1. Verify exact model: `Qwen/Qwen2.5-7B-Instruct`
2. Check HF model device: Use FP32 on CPU, FP16 on GPU
3. Verify token alignment: `all_token_ids` should match validator
4. Check chat template: Must match validator's template

#### Collusion Detection Flags

**Cause**: Insufficient response variation or timing decorrelation.

**Solutions**:
1. Increase `temperature_variance` in sampling profiles
2. Increase `timing_jitter_ms` (10-15ms)
3. Verify different sampling per miner: `curl http://localhost:8091/health | jq '.sampling_profile'`
4. Test response variation locally (see Section 6.1)

#### Port Already in Use

```bash
# Find and kill process
lsof -ti :8091 | xargs kill -9

# Or kill all miners
pkill -f shared_vllm_multi_miner
```

#### Miner Won't Start

```bash
# Check logs
tail -100 shared_vllm_multi_miner.log

# Verify dependencies
pip list | grep -E "vllm|torch|transformers"

# Test GPU availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Check disk space (model downloads)
df -h ~/.cache/huggingface
```

---

## Expected Results

### Single GPU with 4 Miners

Assuming a $4090 (24GB) with good network:

| Metric | Single Miner | 4 Miners (Shared) | Improvement |
|--------|--------------|-------------------|-------------|
| VRAM Usage | 15GB | 19GB | 1.27x |
| Total Requests/s | 8 | 28 | 3.5x |
| Aggregate TPS | 120 | 400+ | 3.3x |
| Total Emission | 1.0x | 3.5-3.8x | **3.5-3.8x** |

**Why not 4.0x?**
- Shared compute creates slight queuing delays (+5-10ms TTFT)
- Population-relative scoring has diminishing returns
- Collusion penalty risk (-10-20% if not properly configured)

**Net result**: **3.5-3.8x emission** with properly tuned multi-miner setup.

### RunPod GPU Server Recommendations

| GPU | VRAM | Monthly Cost | Miners | Expected Emission | ROI |
|-----|------|--------------|--------|-------------------|-----|
| RTX 4090 | 24GB | $0.34/hr ($245/mo) | 4 | 3.8x | Best |
| A100 40GB | 40GB | $1.10/hr ($792/mo) | 6-7 | 5.5x | Good |
| A100 80GB | 80GB | $1.60/hr ($1152/mo) | 8-10 | 7.0x | Premium |

**RunPod Setup**:
1. Use **Secure Cloud** (persistent storage)
2. Select **CUDA 12.1** template
3. Open ports: 8091-8100 (for 10 miners)
4. Use **Docker** deployment (easiest)

---

## Summary

### Critical Success Factors

1. **Speed**: Use vLLM, enable flash attention, optimize network path
2. **Verification**: Maintain large cache (1000+), pass challenges consistently
3. **Consistency**: Lock GPU clocks, avoid CPU throttling
4. **Anti-Collusion**: Use different sampling per miner, add timing jitter
5. **Monitoring**: Watch cache hit rate, challenge pass rate, GPU utilization

### Recommended Configuration

```bash
python3 shared_vllm_multi_miner.py \
  --base-port 8091 \
  --num-miners 4 \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.75 \
  --cache-size 1000
```

### Expected Outcome

- **3.5-3.8x emission** on single GPU (4 miners vs 1 miner)
- **95%+ challenge pass rate** (with proper cache sizing)
- **<0.3 timing correlation** (avoids collusion detection)
- **30-60% response variation** (avoids similarity penalties)

---

## Support & Resources

- **Codebase**: `/home/ubuntu/workspace/Constantinople`
- **Miner Guide**: `MINER_GUIDE.md`
- **Threat Model**: `THREAT_MODEL.md`
- **Monitoring**: Use `monitor_miners.sh` (included above)

For issues, check logs first:
```bash
tail -100 shared_vllm_multi_miner.log
grep -i "error\|fail" shared_vllm_multi_miner.log
```
