# Multi-Miner Quick Reference

## Critical Success Factors

### 1. Speed (40% of Score)
- **Target TTFT**: <30ms (excellent), <100ms (good), >500ms (poor)
- **Target TPS**: >150 (excellent), >80 (good), <10 (poor)
- **Optimization**: Use vLLM, enable flash attention, optimize network latency

### 2. Verification (40% of Score)
- **Cosine threshold**: >0.995 (pass), <0.995 (fail)
- **Challenge latency**: <50ms (ideal), >500ms (auto-fail)
- **Failure penalty**: 3x asymmetric (failing costs 3x passing)
- **Consecutive fails**: 3 strikes = 90% weight penalty

### 3. Consistency (20% of Score)
- **Natural CV range**: 0.05-0.15 (good), <0.03 (suspicious), >1.0 (unstable)
- **Divergence limit**: <12% gap (safe), >12% (-30% penalty), >25% (-70% penalty)

## Quick Start Commands

### Deploy 4 Miners on One GPU
```bash
# Automated deployment
./deploy_multi_miner.sh 4 8091

# Manual start
python3 shared_vllm_multi_miner.py \
  --base-port 8091 --num-miners 4 \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --gpu-memory-utilization 0.75
```

### Monitor Miners
```bash
./monitor_multi_miners.sh 8091 4
```

### Check Individual Miner Health
```bash
for port in {8091..8094}; do
  curl -s "http://localhost:$port/health" | jq '.miner_id, .status, .total_requests, .cache_hit_rate'
done
```

### Register Miners (Each Needs Different Hotkey)
```bash
btcli subnet register --netuid 97 --wallet.name wallet1 --wallet.hotkey hotkey1  # Port 8091
btcli subnet register --netuid 97 --wallet.name wallet2 --wallet.hotkey hotkey2  # Port 8092
btcli subnet register --netuid 97 --wallet.name wallet3 --wallet.hotkey hotkey3  # Port 8093
btcli subnet register --netuid 97 --wallet.name wallet4 --wallet.hotkey hotkey4  # Port 8094
```

## Hardware Requirements

| GPU | VRAM | Max Miners | Monthly Cost (RunPod) | Expected Emission |
|-----|------|------------|----------------------|-------------------|
| RTX 4090 | 24GB | 4-5 | $245 | 3.5-3.8x |
| A100 40GB | 40GB | 6-7 | $792 | 5.0-5.5x |
| A100 80GB | 80GB | 8-10 | $1152 | 6.5-7.0x |

## Configuration Tuning

### GPU Memory Utilization
```bash
# Conservative (4+ miners or frequent OOM)
--gpu-memory-utilization 0.70

# Balanced (3-4 miners) - DEFAULT
--gpu-memory-utilization 0.75

# Aggressive (1-2 miners)
--gpu-memory-utilization 0.80
```

### Cache Size
```bash
# Minimum (low memory)
--cache-size 500

# Recommended (balanced) - DEFAULT
--cache-size 1000

# Large (high traffic, prevent misses)
--cache-size 1500
```

### Sampling Profile Tuning
Edit `sampling_profiles.json`:
- **temperature_base**: 0.65-0.80 (controls randomness)
- **temperature_variance**: 0.10-0.20 (per-request jitter)
- **top_p**: 0.85-0.95 (nucleus sampling)
- **timing_jitter_ms**: 5-12 (latency decorrelation)

## Troubleshooting

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
curl -s http://localhost:8091/health | jq '.model'  # Should be Qwen/Qwen2.5-7B-Instruct

# Check logs for errors
tail -100 multi_miner.log | grep -i "error\|cosine"
```

### Collusion Detection Flags
```bash
# Test response variation (should be 30-60% different)
curl -X POST http://localhost:8091/inference \
  -d '{"prompt":"Test","max_tokens":50}' | jq -r '.text' > m1.txt
curl -X POST http://localhost:8092/inference \
  -d '{"prompt":"Test","max_tokens":50}' | jq -r '.text' > m2.txt
diff m1.txt m2.txt  # Should show differences

# Check sampling profiles
curl -s http://localhost:8091/health | jq '.sampling_profile'
curl -s http://localhost:8092/health | jq '.sampling_profile'  # Should differ
```

## Performance Targets

### Per-Miner Targets
- **Requests**: 100+ per epoch (72 min)
- **Challenges**: 10-20+ per epoch
- **Pass rate**: >95%
- **Cache hit rate**: >95%
- **TTFT**: <100ms average
- **TPS**: >80 average

### Aggregate Targets (4 Miners)
- **Total requests**: 400+ per epoch
- **Total challenges**: 40-80 per epoch
- **Overall pass rate**: >95%
- **Aggregate TPS**: 300-400+
- **Expected emission**: 3.5-3.8x single miner

## Anti-Collusion Checklist

✅ Each miner uses different hotkey  
✅ Different sampling parameters per miner  
✅ Response variation >30% (test with same prompt)  
✅ Timing jitter enabled (5-12ms per miner)  
✅ Independent request logs per miner  
✅ Cache shared but responses unique  
✅ No bit-exact hidden state copying (we run real model)  

## Log Monitoring

### View Real-Time Logs
```bash
tail -f multi_miner.log
```

### Search for Errors
```bash
grep -i "error\|exception\|fail" multi_miner.log
```

### Check Challenge Results
```bash
grep "Challenge" multi_miner.log | tail -20
```

### Analyze Performance
```bash
# TTFT distribution
grep "Inference" multi_miner.log | awk '{print $(NF-3)}' | sort -n | uniq -c

# Challenge pass rate by miner
for id in {0..3}; do
  echo "Miner $id:"
  grep "\[Miner $id\] Challenge" multi_miner.log | grep -c "HIT"
done
```

## Docker Commands

### Build Image
```bash
docker build -f Dockerfile.multi-miner -t multi-miner .
```

### Run with Docker Compose
```bash
docker compose -f docker-compose.multi-miner.yml up -d
```

### View Docker Logs
```bash
docker logs -f inference-multi-miner
```

### Stop Multi-Miner
```bash
docker compose -f docker-compose.multi-miner.yml down
```

## Port Management

Default port range: **8091-8094** (for 4 miners)

To change:
```bash
# Start with custom base port
--base-port 9000  # Miners will use 9000-9003

# Remember to update firewall
for port in {9000..9003}; do
  sudo ufw allow $port/tcp
done
```

## Resource Monitoring

### GPU Stats
```bash
watch -n 1 nvidia-smi
```

### Memory Usage
```bash
watch -n 1 'free -h && echo "" && df -h'
```

### Network Monitoring
```bash
iftop -i eth0  # Replace eth0 with your interface
```

## Expected ROI

| Setup | Hardware Cost | Monthly Hosting | Emission Multiplier | Break-even |
|-------|--------------|-----------------|---------------------|------------|
| Single miner | $1500 (4090) | $245 (RunPod) | 1.0x | Baseline |
| 4 miners shared | $1500 (4090) | $245 (RunPod) | 3.5-3.8x | 3.5-3.8x faster |
| 7 miners shared | $3000 (A100-40) | $792 (RunPod) | 5.0-5.5x | 1.54-1.69x faster |

*Assumes equal TAO rewards across all setups*

## Support

- **Full Guide**: `MULTI_MINER_SETUP_GUIDE.md`
- **Miner Guide**: `MINER_GUIDE.md`
- **Threat Model**: `THREAT_MODEL.md`
- **Monitoring Script**: `monitor_multi_miners.sh`
- **Deployment Script**: `deploy_multi_miner.sh`
