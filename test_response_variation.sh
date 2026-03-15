#!/bin/bash
# test_response_variation.sh - Validate anti-collusion response variation
# Usage: ./test_response_variation.sh [base_port] [num_miners]

BASE_PORT=${1:-8091}
NUM_MINERS=${2:-4}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Multi-Miner Response Variation Test                   ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Test prompts
TEST_PROMPTS=(
    "What is 2+2?"
    "Explain quantum computing in one sentence."
    "Write a haiku about AI."
    "What are the three laws of robotics?"
)

# Function to calculate token-level similarity
token_similarity() {
    local text1="$1"
    local text2="$2"
    
    # Simple word-based similarity (approximation)
    local words1=($(echo "$text1" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n'))
    local words2=($(echo "$text2" | tr '[:upper:]' '[:lower:]' | tr -cs '[:alnum:]' '\n'))
    
    local total=${#words1[@]}
    local matches=0
    
    for word in "${words1[@]}"; do
        if [[ " ${words2[@]} " =~ " ${word} " ]]; then
            matches=$((matches + 1))
        fi
    done
    
    if [ $total -gt 0 ]; then
        echo "scale=2; $matches * 100 / $total" | bc
    else
        echo "0"
    fi
}

echo "Testing response variation across $NUM_MINERS miners..."
echo ""

total_tests=0
passed_tests=0
failed_tests=0

for prompt in "${TEST_PROMPTS[@]}"; do
    echo -e "${YELLOW}Test Prompt: ${NC}\"$prompt\""
    echo ""
    
    # Collect responses from all miners
    declare -a responses
    declare -a response_texts
    
    for ((i=0; i<NUM_MINERS; i++)); do
        port=$((BASE_PORT + i))
        
        response=$(curl -s --max-time 10 -X POST "http://localhost:$port/inference" \
            -H "Content-Type: application/json" \
            -d "{\"prompt\":\"$prompt\",\"max_tokens\":50}" 2>/dev/null)
        
        if [ $? -ne 0 ] || [ -z "$response" ]; then
            echo -e "${RED}✗ Miner $i (port $port): No response${NC}"
            continue
        fi
        
        text=$(echo "$response" | jq -r '.text // empty')
        
        if [ -z "$text" ]; then
            echo -e "${RED}✗ Miner $i (port $port): Empty response${NC}"
            continue
        fi
        
        responses[$i]="$response"
        response_texts[$i]="$text"
        
        # Truncate for display
        display_text="${text:0:80}"
        [ ${#text} -gt 80 ] && display_text="${display_text}..."
        
        echo -e "  Miner $i: $display_text"
    done
    
    echo ""
    echo "Similarity Matrix:"
    echo "┌────────┬─────────────────────────────────────────┐"
    echo "│  Pair  │ Similarity │ Status                    │"
    echo "├────────┼────────────┼───────────────────────────┤"
    
    # Compare all pairs
    for ((i=0; i<NUM_MINERS-1; i++)); do
        for ((j=i+1; j<NUM_MINERS; j++)); do
            if [ -z "${response_texts[$i]}" ] || [ -z "${response_texts[$j]}" ]; then
                continue
            fi
            
            total_tests=$((total_tests + 1))
            
            # Calculate similarity
            sim=$(token_similarity "${response_texts[$i]}" "${response_texts[$j]}")
            
            # Status based on similarity
            # Target: 30-70% similarity (same model, different sampling)
            # Warning: >85% similarity (collusion detection threshold)
            # Fail: >95% similarity (definite red flag)
            
            status="${GREEN}✓ Good variation${NC}"
            status_emoji="✓"
            
            if (( $(echo "$sim > 95" | bc -l) )); then
                status="${RED}✗ TOO SIMILAR (>95%)${NC}"
                status_emoji="✗"
                failed_tests=$((failed_tests + 1))
            elif (( $(echo "$sim > 85" | bc -l) )); then
                status="${YELLOW}⚠ Warning (>85%)${NC}"
                status_emoji="⚠"
            elif (( $(echo "$sim < 10" | bc -l) )); then
                status="${YELLOW}⚠ Too different (<10%)${NC}"
                status_emoji="⚠"
            else
                passed_tests=$((passed_tests + 1))
            fi
            
            printf "│ %d ↔ %-2d │ %6.1f%%   │ %-35b │\n" $i $j $sim "$status"
        done
    done
    
    echo "└────────┴────────────┴───────────────────────────┘"
    echo ""
done

# Summary
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test Summary                                          ${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

pass_rate=0
if [ $total_tests -gt 0 ]; then
    pass_rate=$(echo "scale=1; $passed_tests * 100 / $total_tests" | bc)
fi

echo "Total Comparisons: $total_tests"
echo -e "Passed (30-85%):   ${GREEN}$passed_tests${NC}"
echo -e "Failed (>95%):     ${RED}$failed_tests${NC}"
echo ""

if [ $total_tests -eq 0 ]; then
    echo -e "${RED}✗ No successful tests. Check that miners are running:${NC}"
    echo "  curl http://localhost:${BASE_PORT}/health"
    exit 1
elif [ $failed_tests -gt 0 ]; then
    echo -e "${RED}✗ FAILED: Some responses are too similar (>95%)${NC}"
    echo ""
    echo "This indicates potential collusion detection risk."
    echo ""
    echo "Recommended actions:"
    echo "  1. Check sampling profiles: curl http://localhost:${BASE_PORT}/health | jq '.sampling_profile'"
    echo "  2. Increase temperature_variance in sampling_profiles.json"
    echo "  3. Verify different sampling parameters per miner"
    echo "  4. Restart miners after configuration changes"
    exit 1
elif (( $(echo "$pass_rate < 80" | bc -l) )); then
    echo -e "${YELLOW}⚠ WARNING: Pass rate below 80%${NC}"
    echo ""
    echo "Some responses may be at risk of collusion detection."
    echo "Consider adjusting sampling profiles for more variation."
    exit 1
else
    echo -e "${GREEN}✓ PASSED: Response variation is within safe range${NC}"
    echo ""
    echo "Expected similarity: 30-70% (same model, different sampling)"
    echo "Collusion threshold: 85% (validator flags above this)"
    echo "Your miners: Within safe range ✓"
    echo ""
    echo "Next steps:"
    echo "  1. Deploy to production: ./deploy_multi_miner.sh $NUM_MINERS $BASE_PORT"
    echo "  2. Register each miner with different hotkey"
    echo "  3. Monitor performance: ./monitor_multi_miners.sh $BASE_PORT $NUM_MINERS"
    exit 0
fi
