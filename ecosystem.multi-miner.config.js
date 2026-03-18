module.exports = {
  apps: [
    {
      name: 'multi-miner-orchestrator',
      script: 'shared_vllm_multi_miner.py',
      interpreter: 'python3',
      // B200 stability fix: aggressive memory reduction + fewer miners
      args: '--base-port 8091 --num-miners 3 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.60 --cache-size 600 --sampling-profiles sampling_profiles_h100.json',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '10G',  // Increased for B200 stability
      max_restarts: 5,  // Limit restart loops
      min_uptime: '30s',  // Must run 30s to count as successful start
      env: {
        CUDA_VISIBLE_DEVICES: '0',
        PYTHONUNBUFFERED: '1',
        CUDA_LAUNCH_BLOCKING: '0',  // Async CUDA launches (faster)
        PYTORCH_CUDA_ALLOC_CONF: 'expandable_segments:True',  // Reduce memory fragmentation
        VLLM_WORKER_MULTIPROC_METHOD: 'spawn',  // More stable than fork on new GPUs
        VLLM_USE_RAY_COMPILED_DAG: '0',  // Disable Ray for stability
        VLLM_ALLOW_RUNTIME_LORA_UPDATING: '0',  // Reduce overhead
        // NCCL settings - prevent crashes on single-GPU B200
        NCCL_DEBUG: 'ERROR',  // Reduce NCCL logging noise
        NCCL_P2P_DISABLE: '1',  // Disable peer-to-peer (not needed on single GPU)
        NCCL_SHM_DISABLE: '1',  // Disable shared memory (improves stability)
        TORCH_NCCL_BLOCKING_WAIT: '1',  // Prevent async NCCL hangs
        TORCH_NCCL_ASYNC_ERROR_HANDLING: '1',  // Better error handling
      },
      error_file: 'logs/multi-miner-error.log',
      out_file: 'logs/multi-miner-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      kill_timeout: 30000,
      wait_ready: true,
      listen_timeout: 180000,  // 3 minutes for B200 initialization
    }
  ]
};
