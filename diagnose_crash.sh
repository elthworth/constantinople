#!/bin/bash
# Comprehensive Diagnostic Script for B200 vLLM Crashes
# Run this when experiencing crashes to collect all relevant information

echo "==================================================================="
echo "vLLM Multi-Miner Crash Diagnostic Report"
echo "Generated: $(date)"
echo "==================================================================="
echo ""

# 1. Check PM2 Status
echo "1. PM2 Process Status:"
echo "-------------------------------------------------------------------"
pm2 status multi-miner-orchestrator 2>/dev/null || echo "PM2 process not found"
echo ""

# 2. Check uptime and restart count
echo "2. Uptime and Restart Count:"
echo "-------------------------------------------------------------------"
pm2 show multi-miner-orchestrator 2>/dev/null | grep -E "uptime|restart" || echo "No PM2 info available"
echo ""

# 3. GPU Information
echo "3. GPU Status:"
echo "-------------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw --format=csv
echo ""

# 4. GPU Memory Details
echo "4. Detailed GPU Memory Breakdown:"
echo "-------------------------------------------------------------------"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
echo ""

# 5. Check for OOM in dmesg
echo "5. Out of Memory Events (dmesg):"
echo "-------------------------------------------------------------------"
sudo dmesg | grep -i "out of memory\|oom\|killed process" | tail -10 || echo "No OOM events found"
echo ""

# 6. Recent Error Logs
echo "6. Recent Error Logs (Last 50 lines):"
echo "-------------------------------------------------------------------"
pm2 logs multi-miner-orchestrator --nostream --lines 50 2>/dev/null | grep -i "error\|crash\|died\|exception" | tail -20
echo ""

# 7. Check EngineCore crashes
echo "7. EngineCore Crash Events:"
echo "-------------------------------------------------------------------"
pm2 logs multi-miner-orchestrator --nostream --lines 500 2>/dev/null | grep "EngineCore.*died" | tail -10
echo ""

# 8. NCCL Warnings
echo "8. NCCL Warnings:"
echo "-------------------------------------------------------------------"
pm2 logs multi-miner-orchestrator --nostream --lines 200 2>/dev/null | grep -i "nccl\|ProcessGroup" | tail -10
echo ""

# 9. Check Health Endpoints
echo "9. Miner Health Check:"
echo "-------------------------------------------------------------------"
for port in 8091 8092 8093; do
    echo -n "Port $port: "
    response=$(curl -s --max-time 5 http://localhost:$port/health 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "✅ OK"
        echo "$response" | python3 -m json.tool 2>/dev/null | head -15
    else
        echo "❌ UNREACHABLE"
    fi
    echo ""
done

# 10. Check System Memory
echo "10. System Memory:"
echo "-------------------------------------------------------------------"
free -h
echo ""

# 11. Check Python Processes
echo "11. Python Processes (vLLM related):"
echo "-------------------------------------------------------------------"
ps aux | grep "[p]ython.*vllm\|[p]ython.*multi_miner" | head -10
echo ""

# 12. Check for Memory Leak Pattern
echo "12. Memory Leak Detection:"
echo "-------------------------------------------------------------------"
if [ -f /tmp/vllm_health_stats.json ]; then
    echo "Health stats file found:"
    cat /tmp/vllm_health_stats.json | python3 -m json.tool 2>/dev/null
else
    echo "No health stats file found (monitor may not be running)"
fi
echo ""

# 13. Recent Timestamps of Crashes
echo "13. Crash Timestamps (Last 10):"
echo "-------------------------------------------------------------------"
pm2 logs multi-miner-orchestrator --nostream --lines 1000 2>/dev/null | \
    grep -E "died unexpectedly|CancelledError|KeyboardInterrupt" | \
    awk '{print $1, $2}' | tail -10
echo ""

# 14. CUDA/PyTorch Versions
echo "14. CUDA and PyTorch Versions:"
echo "-------------------------------------------------------------------"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')" 2>/dev/null
python3 -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null
echo ""

# 15. Environment Variables
echo "15. Relevant Environment Variables:"
echo "-------------------------------------------------------------------"
pm2 show multi-miner-orchestrator 2>/dev/null | grep -A 20 "env:" | grep -E "CUDA|NCCL|TORCH|VLLM"
echo ""

# 16. Recommendations
echo "==================================================================="
echo "DIAGNOSTICS COMPLETE"
echo "==================================================================="
echo ""
echo "📋 Next Steps Based on Findings:"
echo ""
echo "If you see:"
echo "  • 'EngineCore died unexpectedly' → Check GPU memory (should be <85%)"
echo "  • Multiple NCCL warnings → NCCL env vars not set correctly"
echo "  • Steadily increasing GPU memory → Memory leak (restart required)"
echo "  • OOM in dmesg → Reduce --gpu-memory-utilization to 0.50"
echo "  • High restart count → Apply emergency mode (2 miners, 50% GPU)"
echo ""
echo "To apply fixes:"
echo "  1. Edit ecosystem.multi-miner.config.js"
echo "  2. Run: pm2 restart multi-miner-orchestrator"
echo "  3. Monitor: pm2 logs multi-miner-orchestrator"
echo ""
echo "For continuous monitoring, run:"
echo "  python3 monitor_vllm_health.py --interval 30 &"
echo ""
