#!/bin/bash
# Quick test to verify miners are working and track them for crashes

echo "🧪 Testing vLLM Multi-Miner Stability"
echo "This will send test requests and monitor for crashes..."
echo ""

# Check if miners are running
echo "1. Checking miner endpoints..."
all_ok=true
for port in 8091 8092 8093; do
    if curl -s --max-time 5 http://localhost:$port/health >/dev/null 2>&1; then
        echo "   Port $port: ✅ Running"
    else
        echo "   Port $port: ❌ Not responding"
        all_ok=false
    fi
done

if [ "$all_ok" = false ]; then
    echo ""
    echo "❌ Some miners are not running. Start them first:"
    echo "   ./start_with_monitoring.sh"
    exit 1
fi

echo ""
echo "2. Sending test inference requests..."
echo ""

# Test each miner
for i in {1..3}; do
    port=$((8090 + i))
    echo "Testing Miner $((i-1)) on port $port..."
    
    response=$(curl -s --max-time 30 -X POST "http://localhost:$port/inference" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Write a haiku about artificial intelligence",
            "max_tokens": 50,
            "temperature": 0.7
        }' 2>&1)
    
    if echo "$response" | grep -q "request_id"; then
        echo "   ✅ Success - Generated response"
        tokens=$(echo "$response" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('output_tokens', 0))" 2>/dev/null || echo "?")
        echo "   📊 Output tokens: $tokens"
    else
        echo "   ❌ Failed"
        echo "   Response: $response"
    fi
    echo ""
    sleep 2
done

echo "3. Checking health after test requests..."
for port in 8091 8092 8093; do
    health=$(curl -s --max-time 5 "http://localhost:$port/health" 2>/dev/null)
    if [ $? -eq 0 ]; then
        total_reqs=$(echo "$health" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('engine_stats', {}).get('total_requests', 0))" 2>/dev/null || echo "?")
        gpu_util=$(echo "$health" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('gpu_memory', {}).get('utilization_pct', '?'))" 2>/dev/null || echo "?")
        echo "   Port $port: ✅ (Total requests: $total_reqs, GPU: $gpu_util%)"
    else
        echo "   Port $port: ❌ Not responding!"
    fi
done

echo ""
echo "4. Current GPU Memory:"
echo "-------------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "✅ Test Complete!"
echo ""
echo "💡 Monitor for crashes over next 30 minutes with:"
echo "   pm2 logs multi-miner-orchestrator"
echo ""
echo "   Watch for these patterns:"
echo "   • '[vLLM] Request XXX completed' = Good ✅"
echo "   • 'ENGINE CRASH DETECTED' = Bad ❌"
echo "   • 'HIGH GPU MEMORY' = Warning ⚠️"
echo ""
