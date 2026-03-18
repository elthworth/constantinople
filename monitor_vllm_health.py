#!/usr/bin/env python3
"""
vLLM Health Monitor - Diagnose runtime crashes
Monitors GPU memory, request patterns, and logs crashes
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import aiohttp
import psutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("vllm_health_monitor")


class VLLMHealthMonitor:
    def __init__(self, ports: list[int], check_interval: int = 30):
        self.ports = ports
        self.check_interval = check_interval
        self.stats = {
            "total_checks": 0,
            "successful_checks": 0,
            "failed_checks": 0,
            "crash_count": 0,
            "last_crash_time": None,
            "uptime_seconds": 0,
            "start_time": time.time(),
        }
        
    async def check_endpoint(self, port: int) -> dict:
        """Check if miner endpoint is healthy"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://localhost:{port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"status": "ok", "port": port, "data": data}
                    else:
                        return {"status": "error", "port": port, "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"status": "error", "port": port, "error": str(e)}
    
    def get_gpu_memory(self) -> dict:
        """Get GPU memory usage"""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                return {
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                    "total_gb": round(total, 2),
                    "utilization_pct": round((reserved / total) * 100, 1),
                }
            else:
                return {"error": "No CUDA GPU"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_system_memory(self) -> dict:
        """Get system RAM usage"""
        mem = psutil.virtual_memory()
        return {
            "used_gb": round(mem.used / 1e9, 2),
            "total_gb": round(mem.total / 1e9, 2),
            "utilization_pct": round(mem.percent, 1),
        }
    
    async def monitor_loop(self):
        """Main monitoring loop"""
        log.info(f"Starting health monitor for ports: {self.ports}")
        log.info(f"Check interval: {self.check_interval}s")
        
        previous_status = {port: "unknown" for port in self.ports}
        
        while True:
            try:
                self.stats["total_checks"] += 1
                self.stats["uptime_seconds"] = int(time.time() - self.stats["start_time"])
                
                # Check all endpoints
                checks = await asyncio.gather(
                    *[self.check_endpoint(port) for port in self.ports]
                )
                
                # Analyze results
                all_ok = all(c["status"] == "ok" for c in checks)
                
                if all_ok:
                    self.stats["successful_checks"] += 1
                else:
                    self.stats["failed_checks"] += 1
                
                # Detect crashes (state change from ok to error)
                for check in checks:
                    port = check["port"]
                    current = check["status"]
                    previous = previous_status.get(port, "unknown")
                    
                    if previous == "ok" and current == "error":
                        # CRASH DETECTED
                        self.stats["crash_count"] += 1
                        self.stats["last_crash_time"] = datetime.now().isoformat()
                        log.error(f"🚨 CRASH DETECTED on port {port}!")
                        log.error(f"   Error: {check.get('error', 'unknown')}")
                        log.error(f"   Total crashes: {self.stats['crash_count']}")
                        log.error(f"   Uptime before crash: {self.stats['uptime_seconds']}s")
                    
                    previous_status[port] = current
                
                # Get resource usage
                gpu_mem = self.get_gpu_memory()
                sys_mem = self.get_system_memory()
                
                # Log status
                status_str = " | ".join([
                    f"Port {c['port']}: {'✅' if c['status'] == 'ok' else '❌'}"
                    for c in checks
                ])
                
                log.info(f"Health Check #{self.stats['total_checks']}: {status_str}")
                log.info(f"  GPU: {gpu_mem.get('utilization_pct', '?')}% ({gpu_mem.get('reserved_gb', '?')} / {gpu_mem.get('total_gb', '?')} GB)")
                log.info(f"  RAM: {sys_mem['utilization_pct']}% ({sys_mem['used_gb']} / {sys_mem['total_gb']} GB)")
                log.info(f"  Uptime: {self.stats['uptime_seconds']}s | Crashes: {self.stats['crash_count']}")
                
                # Check for memory leak warning
                if gpu_mem.get('utilization_pct', 0) > 85:
                    log.warning(f"⚠️  HIGH GPU MEMORY: {gpu_mem['utilization_pct']}% - possible memory leak!")
                
                if sys_mem['utilization_pct'] > 90:
                    log.warning(f"⚠️  HIGH RAM USAGE: {sys_mem['utilization_pct']}% - possible memory leak!")
                
                # Save stats to file
                with open("/tmp/vllm_health_stats.json", "w") as f:
                    json.dump({
                        **self.stats,
                        "gpu_memory": gpu_mem,
                        "system_memory": sys_mem,
                        "endpoints": checks,
                    }, f, indent=2)
                
            except Exception as e:
                log.error(f"Monitor error: {e}")
            
            await asyncio.sleep(self.check_interval)


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Monitor vLLM multi-miner health")
    parser.add_argument("--ports", type=int, nargs="+", default=[8091, 8092, 8093],
                        help="Miner ports to monitor")
    parser.add_argument("--interval", type=int, default=30,
                        help="Check interval in seconds")
    
    args = parser.parse_args()
    
    monitor = VLLMHealthMonitor(ports=args.ports, check_interval=args.interval)
    await monitor.monitor_loop()


if __name__ == "__main__":
    asyncio.run(main())
