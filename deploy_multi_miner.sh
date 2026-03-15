#!/bin/bash
# deploy_multi_miner.sh - Deploy multi-miner setup on RunPod or any GPU server
# Usage: ./deploy_multi_miner.sh [num_miners] [base_port]

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
NUM_MINERS=${1:-4}
BASE_PORT=${2:-8091}
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-7B-Instruct"}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.75}
CACHE_SIZE=${CACHE_SIZE:-1000}

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Multi-Miner Deployment Script - Bittensor SN97       ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Configuration:"
echo -e "  Miners:           ${GREEN}${NUM_MINERS}${NC}"
echo -e "  Base Port:        ${GREEN}${BASE_PORT}${NC}"
echo -e "  Model:            ${GREEN}${MODEL_NAME}${NC}"
echo -e "  GPU Memory:       ${GREEN}${GPU_MEMORY_UTIL}${NC} (utilization)"
echo -e "  Cache Size:       ${GREEN}${CACHE_SIZE}${NC} requests"
echo ""

# Step 1: Check prerequisites
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ $GPU_COUNT -eq 0 ]; then
    echo -e "${RED}Error: No GPUs detected.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found ${GPU_COUNT} GPU(s)${NC}"

# Check VRAM
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$(echo "scale=1; $VRAM_MB / 1024" | bc)
echo -e "${GREEN}✓ GPU VRAM: ${VRAM_GB}GB${NC}"

if [ $(echo "$VRAM_GB < 20" | bc -l) -eq 1 ]; then
    echo -e "${YELLOW}Warning: VRAM < 20GB. Recommend reducing num_miners or using smaller model.${NC}"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION}${NC}"

# Step 2: Install dependencies
echo ""
echo -e "${YELLOW}[2/6] Installing dependencies...${NC}"

if python3 -c "import vllm" &> /dev/null; then
    echo -e "${GREEN}✓ vLLM already installed${NC}"
else
    echo "Installing vLLM and dependencies..."
    pip install -q vllm torch transformers accelerate fastapi "uvicorn[standard]" numpy aiohttp pydantic
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Step 3: Download model (if not cached)
echo ""
echo -e "${YELLOW}[3/6] Checking model cache...${NC}"

MODEL_CACHE="$HOME/.cache/huggingface/hub"
if [ -d "$MODEL_CACHE" ] && [ "$(ls -A $MODEL_CACHE)" ]; then
    echo -e "${GREEN}✓ Model cache exists${NC}"
else
    echo "Downloading model (this may take several minutes)..."
    python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('${MODEL_NAME}')
print('Downloading model weights...')
AutoModelForCausalLM.from_pretrained('${MODEL_NAME}', torch_dtype='auto', device_map='auto')
print('Model cached successfully')
"
    echo -e "${GREEN}✓ Model downloaded${NC}"
fi

# Step 4: Create configuration files
echo ""
echo -e "${YELLOW}[4/6] Creating configuration files...${NC}"

# Create .env file
cat > .env.multi-miner <<EOF
NUM_MINERS=${NUM_MINERS}
BASE_PORT=${BASE_PORT}
MODEL_NAME=${MODEL_NAME}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTIL}
CACHE_SIZE=${CACHE_SIZE}
MAX_PORT=$((BASE_PORT + NUM_MINERS - 1))
EOF

echo -e "${GREEN}✓ Created .env.multi-miner${NC}"

# Check if sampling_profiles.json exists
if [ ! -f "sampling_profiles.json" ]; then
    echo -e "${YELLOW}Warning: sampling_profiles.json not found. Using default profiles.${NC}"
fi

# Step 5: Configure firewall (if needed)
echo ""
echo -e "${YELLOW}[5/6] Configuring firewall...${NC}"

if command -v ufw &> /dev/null; then
    for ((i=0; i<NUM_MINERS; i++)); do
        port=$((BASE_PORT + i))
        sudo ufw allow $port/tcp &> /dev/null || true
    done
    echo -e "${GREEN}✓ Firewall rules added${NC}"
else
    echo -e "${YELLOW}⚠ ufw not found, skipping firewall configuration${NC}"
fi

# Step 6: Start multi-miner
echo ""
echo -e "${YELLOW}[6/6] Starting multi-miner orchestrator...${NC}"

# Check if already running
if pgrep -f "shared_vllm_multi_miner.py" > /dev/null; then
    echo -e "${YELLOW}Stopping existing multi-miner process...${NC}"
    pkill -f "shared_vllm_multi_miner.py"
    sleep 3
fi

# Start in background
SAMPLING_ARG=""
if [ -f "sampling_profiles.json" ]; then
    SAMPLING_ARG="--sampling-profiles sampling_profiles.json"
fi

nohup python3 shared_vllm_multi_miner.py \
    --base-port ${BASE_PORT} \
    --num-miners ${NUM_MINERS} \
    --model "${MODEL_NAME}" \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --cache-size ${CACHE_SIZE} \
    ${SAMPLING_ARG} \
    > multi_miner.log 2>&1 &

MINER_PID=$!
echo -e "${GREEN}✓ Multi-miner started (PID: ${MINER_PID})${NC}"

# Wait for startup
echo ""
echo -e "${YELLOW}Waiting for miners to initialize (this may take 30-60 seconds)...${NC}"
sleep 10

# Check health of each miner
echo ""
echo -e "${YELLOW}Checking miner health...${NC}"

all_healthy=1
for ((i=0; i<NUM_MINERS; i++)); do
    port=$((BASE_PORT + i))
    
    # Try up to 6 times (60 seconds total)
    healthy=0
    for attempt in {1..6}; do
        if curl -s --max-time 5 "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Miner $i (port $port): healthy${NC}"
            healthy=1
            break
        else
            if [ $attempt -lt 6 ]; then
                sleep 10
            fi
        fi
    done
    
    if [ $healthy -eq 0 ]; then
        echo -e "${RED}✗ Miner $i (port $port): not responding${NC}"
        all_healthy=0
    fi
done

# Final status
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
if [ $all_healthy -eq 1 ]; then
    echo -e "${GREEN}✓ All miners deployed successfully!${NC}"
    echo ""
    echo -e "Miner endpoints:"
    for ((i=0; i<NUM_MINERS; i++)); do
        port=$((BASE_PORT + i))
        echo -e "  Miner $i: ${BLUE}http://localhost:$port${NC}"
    done
    echo ""
    echo -e "Next steps:"
    echo -e "  1. Register each miner with a ${GREEN}different hotkey${NC}:"
    echo -e "     ${YELLOW}btcli subnet register --netuid 97 --wallet.name wallet1 --wallet.hotkey hotkey1${NC}"
    echo -e "     ${YELLOW}btcli subnet register --netuid 97 --wallet.name wallet2 --wallet.hotkey hotkey2${NC}"
    echo -e "     ... (repeat for all miners)"
    echo ""
    echo -e "  2. Monitor miners:"
    echo -e "     ${YELLOW}./monitor_multi_miners.sh ${BASE_PORT} ${NUM_MINERS}${NC}"
    echo ""
    echo -e "  3. View logs:"
    echo -e "     ${YELLOW}tail -f multi_miner.log${NC}"
    echo ""
else
    echo -e "${RED}✗ Some miners failed to start. Check logs:${NC}"
    echo -e "  ${YELLOW}tail -100 multi_miner.log${NC}"
    echo ""
    echo -e "Common issues:"
    echo -e "  • CUDA OOM: Reduce --num-miners or --gpu-memory-utilization"
    echo -e "  • Port conflict: Change --base-port"
    echo -e "  • Model download failed: Check network and HuggingFace access"
    exit 1
fi
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
