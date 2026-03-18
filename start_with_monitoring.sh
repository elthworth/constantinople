#!/bin/bash
# Start vLLM Multi-Miner with Health Monitoring

set -e

echo "🚀 Starting vLLM Multi-Miner with Health Monitoring"
echo ""

# Stop any existing instances
echo "1. Stopping existing processes..."
pm2 delete multi-miner-orchestrator 2>/dev/null || true
pm2 delete vllm-health-monitor 2>/dev/null || true
sleep 2

# Start main application
echo "2. Starting multi-miner orchestrator..."
pm2 start ecosystem.multi-miner.config.js
sleep 5

# Wait for miners to initialize
echo "3. Waiting for miners to initialize (30s)..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

# Check if miners are up
echo "4. Testing miner endpoints..."
all_ok=true
for port in 8091 8092 8093; do
    echo -n "   Port $port: "
    if curl -s --max-time 5 http://localhost:$port/health >/dev/null 2>&1; then
        echo "✅"
    else
        echo "❌"
        all_ok=false
    fi
done
echo ""

if [ "$all_ok" = false ]; then
    echo "⚠️  WARNING: Some miners failed to start. Check logs:"
    echo "   pm2 logs multi-miner-orchestrator"
    exit 1
fi

# Start health monitor
echo "5. Starting health monitor..."
pm2 start python3 --name vllm-health-monitor -- monitor_vllm_health.py --interval 30
sleep 2

# Show status
echo "6. Current Status:"
pm2 status

echo ""
echo "✅ Startup Complete!"
echo ""
echo "📊 Monitoring Commands:"
echo "   • View logs:        pm2 logs multi-miner-orchestrator"
echo "   • View monitor:     pm2 logs vllm-health-monitor"
echo "   • Check status:     pm2 status"
echo "   • Diagnose crashes: ./diagnose_crash.sh"
echo "   • View health:      cat /tmp/vllm_health_stats.json | python3 -m json.tool"
echo ""
echo "🔍 The health monitor will check every 30s and alert on:"
echo "   • Engine crashes"
echo "   • High GPU memory (>85%)"
echo "   • Memory leaks"
echo "   • Endpoint failures"
echo ""
echo "💡 If crashes occur, run: ./diagnose_crash.sh"
echo ""
