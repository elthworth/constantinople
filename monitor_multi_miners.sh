#!/bin/bash
# Monitor multiple miners on shared vLLM backend
# Usage: ./monitor_multi_miners.sh [base_port] [num_miners]

BASE_PORT=${1:-8091}
NUM_MINERS=${2:-4}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to format numbers
format_number() {
    printf "%'d" $1 2>/dev/null || echo $1
}

# Function to format percentage
format_pct() {
    printf "%.1f%%" $(echo "$1 * 100" | bc -l 2>/dev/null || echo "0")
}

# Function to get miner health
get_miner_health() {
    local port=$1
    curl -s --max-time 2 "http://localhost:$port/health" 2>/dev/null
}

# Function to check if port is open
is_port_open() {
    nc -z localhost $1 2>/dev/null
    return $?
}

# Main monitoring loop
while true; do
    clear
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║           Multi-Miner Monitoring Dashboard                        ║${NC}"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # GPU Stats
    if command -v nvidia-smi &> /dev/null; then
        gpu_stats=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits 2>/dev/null)
        if [ $? -eq 0 ]; then
            IFS=',' read -r gpu_util mem_util mem_used mem_total temp power power_limit <<< "$gpu_stats"
            
            # GPU utilization color
            gpu_color=$GREEN
            [ $(echo "$gpu_util > 95" | bc -l) -eq 1 ] && gpu_color=$YELLOW
            [ $(echo "$gpu_util < 60" | bc -l) -eq 1 ] && gpu_color=$RED
            
            # Memory color
            mem_color=$GREEN
            [ $(echo "$mem_util > 90" | bc -l) -eq 1 ] && mem_color=$RED
            [ $(echo "$mem_util > 85" | bc -l) -eq 1 ] && mem_color=$YELLOW
            
            # Temperature color
            temp_color=$GREEN
            [ $(echo "$temp > 85" | bc -l) -eq 1 ] && temp_color=$RED
            [ $(echo "$temp > 75" | bc -l) -eq 1 ] && temp_color=$YELLOW
            
            echo -e "${BLUE}GPU Status:${NC}"
            echo -e "  GPU Util:  ${gpu_color}${gpu_util}%${NC}"
            echo -e "  Mem Util:  ${mem_color}${mem_util}%${NC} (${mem_used}MB / ${mem_total}MB)"
            echo -e "  Temp:      ${temp_color}${temp}°C${NC}"
            echo -e "  Power:     ${power}W / ${power_limit}W"
            echo ""
        fi
    fi
    
    # Shared cache stats (from any miner)
    shared_cache_size=0
    shared_cache_hit_rate=0
    
    # Individual miner stats
    echo -e "${BLUE}Miner Stats:${NC}"
    echo -e "┌──────────┬────────┬───────────┬────────────┬──────────┬───────────┬────────────┐"
    echo -e "│ Miner ID │ Status │ Requests  │ Challenges │ Pass Rate│ Avg TTFT  │ Cache Hits │"
    echo -e "├──────────┼────────┼───────────┼────────────┼──────────┼───────────┼────────────┤"
    
    total_requests=0
    total_challenges=0
    total_passed=0
    active_miners=0
    
    for ((i=0; i<NUM_MINERS; i++)); do
        port=$((BASE_PORT + i))
        
        if is_port_open $port; then
            health=$(get_miner_health $port)
            
            if [ -n "$health" ]; then
                miner_id=$(echo "$health" | jq -r '.miner_id // "?"')
                status=$(echo "$health" | jq -r '.status // "unknown"')
                requests=$(echo "$health" | jq -r '.total_requests // 0')
                challenges=$(echo "$health" | jq -r '.total_challenges // 0')
                passed=$(echo "$health" | jq -r '.challenges_passed // 0')
                cache_hit_rate=$(echo "$health" | jq -r '.cache_hit_rate // 0')
                
                # Get sampling profile
                temp=$(echo "$health" | jq -r '.sampling_profile.temperature_base // 0')
                
                # Calculate pass rate
                if [ "$challenges" -gt 0 ]; then
                    pass_rate=$(echo "scale=1; $passed * 100 / $challenges" | bc)
                    pass_rate="${pass_rate}%"
                else
                    pass_rate="N/A"
                fi
                
                # Status color
                if [ "$status" == "ok" ]; then
                    status_str="${GREEN}● OK${NC}"
                    active_miners=$((active_miners + 1))
                else
                    status_str="${RED}● ERR${NC}"
                fi
                
                # Pass rate color
                pass_color=$GREEN
                if [ "$challenges" -gt 0 ]; then
                    pass_pct=$(echo "scale=0; $passed * 100 / $challenges" | bc)
                    [ $pass_pct -lt 95 ] && pass_color=$YELLOW
                    [ $pass_pct -lt 80 ] && pass_color=$RED
                fi
                
                # TTFT estimate (not available in real-time, show placeholder)
                ttft="~35ms"
                
                # Format cache hit rate
                cache_hits=$(format_pct $cache_hit_rate)
                
                # Update totals
                total_requests=$((total_requests + requests))
                total_challenges=$((total_challenges + challenges))
                total_passed=$((total_passed + passed))
                
                # Update shared cache (same for all)
                if [ "$i" -eq 0 ]; then
                    shared_cache_size=$(echo "$health" | jq -r '.cache_size // 0')
                    shared_cache_hit_rate=$cache_hit_rate
                fi
                
                printf "│ %-8s │ %-12b │ %-9s │ %-10s │ ${pass_color}%-8s${NC} │ %-9s │ %-10s │\n" \
                    "$miner_id" "$status_str" \
                    "$(format_number $requests)" \
                    "$(format_number $challenges)" \
                    "$pass_rate" "$ttft" "$cache_hits"
            else
                printf "│ %-8s │ ${RED}%-6s${NC} │ %-9s │ %-10s │ %-8s │ %-9s │ %-10s │\n" \
                    "$i" "NO DATA" "-" "-" "-" "-" "-"
            fi
        else
            printf "│ %-8s │ ${RED}%-6s${NC} │ %-9s │ %-10s │ %-8s │ %-9s │ %-10s │\n" \
                "$i" "DOWN" "-" "-" "-" "-" "-"
        fi
    done
    
    echo -e "└──────────┴────────┴───────────┴────────────┴──────────┴───────────┴────────────┘"
    
    # Aggregated stats
    echo ""
    echo -e "${BLUE}Aggregate Stats:${NC}"
    echo -e "  Active Miners:     ${GREEN}${active_miners}${NC} / ${NUM_MINERS}"
    echo -e "  Total Requests:    $(format_number $total_requests)"
    echo -e "  Total Challenges:  $(format_number $total_challenges)"
    
    if [ $total_challenges -gt 0 ]; then
        overall_pass_rate=$(echo "scale=1; $total_passed * 100 / $total_challenges" | bc)
        pass_color=$GREEN
        [ $(echo "$overall_pass_rate < 95" | bc -l) -eq 1 ] && pass_color=$YELLOW
        [ $(echo "$overall_pass_rate < 80" | bc -l) -eq 1 ] && pass_color=$RED
        echo -e "  Overall Pass Rate: ${pass_color}${overall_pass_rate}%${NC} ($total_passed / $total_challenges)"
    else
        echo -e "  Overall Pass Rate: ${YELLOW}N/A${NC} (no challenges yet)"
    fi
    
    echo ""
    echo -e "${BLUE}Shared Resources:${NC}"
    echo -e "  Cache Size:        $(format_number $shared_cache_size) requests"
    echo -e "  Cache Hit Rate:    $(format_pct $shared_cache_hit_rate)"
    
    # Request rate (approximate)
    if [ $total_requests -gt 0 ]; then
        # Calculate requests per second (rough estimate based on uptime)
        # This is a simplified calculation; in production, track time deltas
        echo -e "  Est. Throughput:   ${GREEN}~$(echo "scale=1; $total_requests / 60" | bc) req/min${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to exit. Refreshing every 10 seconds...${NC}"
    
    sleep 10
done
