#!/bin/bash
# GPU Health Check Script for B200 - Diagnose vLLM crashes

echo "=== GPU Health Check for B200 ==="
echo ""

# Check GPU status
echo "1. GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv
echo ""

# Check CUDA version
echo "2. CUDA Version:"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"
python3 -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')" 2>/dev/null || echo "PyTorch not found"
echo ""

# Check vLLM version
echo "3. vLLM Version:"
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null || echo "vLLM not installed"
echo ""

# Check for OOM in logs
echo "4. Recent OOM Errors:"
if [ -f logs/multi-miner-error.log ]; then
    grep -i "out of memory\|OOM\|CUDA error" logs/multi-miner-error.log | tail -5
else
    echo "No error log found"
fi
echo ""

# Check current GPU processes
echo "5. Current GPU Processes:"
nvidia-smi pmon -c 1 2>/dev/null || nvidia-smi
echo ""

# Check if PM2 is running
echo "6. PM2 Status:"
pm2 status multi-miner-orchestrator 2>/dev/null || echo "PM2 not running or process not found"
echo ""

# Memory recommendations
echo "7. Memory Recommendations for B200:"
TOTAL_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ ! -z "$TOTAL_MEM" ]; then
    RECOMMENDED_UTIL=$(python3 -c "print(f'{(0.70):.2f}')")
    RECOMMENDED_MEM=$(python3 -c "print(f'{int($TOTAL_MEM * 0.70 / 1024):.1f}')")
    echo "Total GPU Memory: ${TOTAL_MEM} MB"
    echo "Recommended --gpu-memory-utilization: 0.70"
    echo "This reserves ~${RECOMMENDED_MEM} GB for vLLM"
    echo ""
    echo "If crashes persist, try:"
    echo "  - Reduce to --gpu-memory-utilization 0.60"
    echo "  - Reduce --num-miners to 3 or 2"
    echo "  - Check: pm2 logs multi-miner-orchestrator"
fi

echo ""
echo "=== Troubleshooting Steps ==="
echo "1. If 'EngineCore died unexpectedly' error occurs:"
echo "   - Reduce GPU memory: --gpu-memory-utilization 0.60"
echo "   - Reduce miners: --num-miners 3 or 2"
echo "   - Update vLLM: pip install -U vllm"
echo ""
echo "2. Check logs:"
echo "   pm2 logs multi-miner-orchestrator --lines 100"
echo ""
echo "3. Restart with new settings:"
echo "   pm2 restart multi-miner-orchestrator"
echo ""
echo "4. Monitor GPU usage:"
echo "   watch -n 1 nvidia-smi"
