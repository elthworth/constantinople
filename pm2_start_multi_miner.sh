#!/bin/bash
# pm2_start_multi_miner.sh - Start multi-miner with PM2 on H100
# Usage: ./pm2_start_multi_miner.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Multi-Miner with PM2 (H100 Optimized)    ${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
echo ""

# Check PM2
if ! command -v pm2 &> /dev/null; then
    echo -e "${YELLOW}PM2 not found. Installing...${NC}"
    npm install -g pm2
fi

# Create logs directory
mkdir -p logs

# Stop existing instance if running
if pm2 list | grep -q "multi-miner-orchestrator"; then
    echo -e "${YELLOW}Stopping existing multi-miner instance...${NC}"
    pm2 delete multi-miner-orchestrator 2>/dev/null || true
    sleep 2
fi

# Start with PM2
echo -e "${GREEN}Starting multi-miner orchestrator with PM2...${NC}"
pm2 start ecosystem.multi-miner.config.js

# Wait for startup
echo ""
echo -e "${YELLOW}Waiting for miners to initialize (30 seconds)...${NC}"
sleep 30

# Check health
echo ""
echo -e "${GREEN}Checking miner health...${NC}"
all_healthy=1
for port in {8091..8094}; do
    if curl -s --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1; then
        miner_id=$(curl -s "http://localhost:$port/health" | jq -r '.miner_id // "?"')
        echo -e "${GREEN}✓ Miner $miner_id (port $port): healthy${NC}"
    else
        echo -e "${YELLOW}⚠ Miner on port $port: not responding yet${NC}"
        all_healthy=0
    fi
done

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
if [ $all_healthy -eq 1 ]; then
    echo -e "${GREEN}✓ All miners started successfully!${NC}"
else
    echo -e "${YELLOW}⚠ Some miners still initializing. Wait 30s more and check:${NC}"
    echo -e "  pm2 logs multi-miner-orchestrator"
fi
echo ""
echo -e "PM2 Commands:"
echo -e "  ${YELLOW}pm2 logs multi-miner-orchestrator${NC}  - View logs"
echo -e "  ${YELLOW}pm2 monit${NC}                          - Monitor resources"
echo -e "  ${YELLOW}pm2 restart multi-miner-orchestrator${NC} - Restart"
echo -e "  ${YELLOW}pm2 stop multi-miner-orchestrator${NC}   - Stop"
echo -e "  ${YELLOW}pm2 delete multi-miner-orchestrator${NC} - Remove"
echo ""
echo -e "Health Endpoints:"
for port in {8091..8094}; do
    echo -e "  Miner $((port-8091)): ${BLUE}http://localhost:$port/health${NC}"
done
echo ""
echo -e "Next Steps:"
echo -e "  1. Register each miner with different hotkey"
echo -e "  2. Monitor: ${YELLOW}./monitor_multi_miners.sh 8091 4${NC}"
echo -e "  3. Test variation: ${YELLOW}./test_response_variation.sh 8091 4${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════${NC}"
