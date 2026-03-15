module.exports = {
  apps: [
    {
      name: 'multi-miner-orchestrator',
      script: 'shared_vllm_multi_miner.py',
      interpreter: 'python3',
      args: '--base-port 8091 --num-miners 4 --model "Qwen/Qwen2.5-7B-Instruct" --gpu-memory-utilization 0.80 --cache-size 1500 --sampling-profiles sampling_profiles_h100.json',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '8G',
      env: {
        CUDA_VISIBLE_DEVICES: '0',
        PYTHONUNBUFFERED: '1',
      },
      error_file: 'logs/multi-miner-error.log',
      out_file: 'logs/multi-miner-out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      kill_timeout: 30000,
      wait_ready: true,
      listen_timeout: 120000,
    }
  ]
};
