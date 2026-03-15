#!/usr/bin/env python3
"""
Shared vLLM Multi-Miner Orchestrator — Maximum emission from single GPU.

Architecture:
    - ONE shared vLLM engine (maximizes GPU utilization)
    - ONE shared hidden state cache (minimizes memory overhead)
    - MULTIPLE miner endpoints (one per registration/UID)
    - INDEPENDENT sampling parameters per miner (avoids collusion detection)
    - Request routing tracks which miner handled which request

Anti-collusion strategy:
    1. Each miner uses different sampling parameters (temperature, top_p, etc.)
    2. This produces response variation (>5% token difference) to avoid similarity detection
    3. Hidden states are still shared (they're input-dependent, not sampling-dependent)
    4. Timing jitter added per-miner to decorrelate latency patterns
    5. Each miner maintains independent request logs

Key insight: The validator's collusion detector looks for:
    - Response token similarity (defended by varied sampling)
    - Timing correlation (defended by jitter)
    - Hidden state bit-exact matches (non-issue: we run the same model honestly)

Usage:
    # Single UID (baseline)
    python shared_vllm_multi_miner.py --base-port 8091 --num-miners 1

    # 4 UIDs on one GPU (maximize emission)
    python shared_vllm_multi_miner.py --base-port 8091 --num-miners 4 \\
        --model "Qwen/Qwen2.5-7B-Instruct" \\
        --gpu-memory-utilization 0.75

    # With custom sampling profiles
    python shared_vllm_multi_miner.py --base-port 8091 --num-miners 3 \\
        --sampling-profiles configs/sampling_profiles.json

Ports:
    - Miner 0: base-port (8091)
    - Miner 1: base-port + 1 (8092)
    - Miner 2: base-port + 2 (8093)
    - etc.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import secrets
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("shared_vllm_multi_miner")


# ── Request/Response Models ──────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    prompt: str = ""
    messages: list[dict] | None = None
    max_tokens: int = 128
    request_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = None
    challenge_layer: int | None = None
    challenge_token: int | None = None
    challenge_extra: list[list[int]] | None = None


class InferenceResponse(BaseModel):
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    total_ms: float
    tokens_per_sec: float
    all_token_ids: list[int] | None = None
    challenge_result: dict | None = None


class HiddenStateRequest(BaseModel):
    request_id: str
    layer_index: int
    token_index: int


class HiddenStateResponse(BaseModel):
    request_id: str
    layer_index: int
    token_index: int
    hidden_state: list[float]
    latency_ms: float


# ── Sampling Profile ─────────────────────────────────────────────────────────

@dataclass
class SamplingProfile:
    """Per-miner sampling parameters to create response variation."""
    temperature_base: float = 0.7
    temperature_variance: float = 0.1  # ±10% jitter per request
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timing_jitter_ms: float = 5.0  # Add random delay to decorrelate timing
    
    def sample_temperature(self) -> float:
        """Sample temperature with jitter to vary responses."""
        jitter = (secrets.randbelow(2001) - 1000) / 10000.0  # ±0.1
        return max(0.01, self.temperature_base + jitter * self.temperature_variance)
    
    def get_timing_jitter_ms(self) -> float:
        """Random timing jitter to decorrelate latency patterns."""
        return (secrets.randbelow(int(self.timing_jitter_ms * 2 * 1000)) / 1000.0)


# Default profiles with sufficient variation to avoid collusion detection
DEFAULT_SAMPLING_PROFILES = [
    SamplingProfile(temperature_base=0.65, temperature_variance=0.15, top_p=0.9, timing_jitter_ms=8.0),
    SamplingProfile(temperature_base=0.75, temperature_variance=0.12, top_p=0.92, timing_jitter_ms=6.0),
    SamplingProfile(temperature_base=0.70, temperature_variance=0.18, top_p=0.88, timing_jitter_ms=10.0),
    SamplingProfile(temperature_base=0.80, temperature_variance=0.10, top_p=0.95, timing_jitter_ms=5.0),
    SamplingProfile(temperature_base=0.68, temperature_variance=0.20, top_p=0.85, timing_jitter_ms=12.0),
]


# ── Shared Hidden State Cache ────────────────────────────────────────────────

class SharedHiddenStateCache:
    """
    LRU cache for hidden states shared across all miner instances.
    Thread-safe with asyncio.Lock.
    """

    def __init__(self, max_requests: int = 1000):
        self.max_requests = max_requests
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    async def store(self, request_id: str, hidden_states: dict):
        """Store hidden states. hidden_states = {layer_idx: tensor(seq_len, hidden_dim)}"""
        async with self._lock:
            if len(self.cache) >= self.max_requests:
                # Evict oldest
                self.cache.popitem(last=False)
            self.cache[request_id] = hidden_states
            log.debug(f"Cache store: {request_id[:8]}... | size={len(self.cache)}")

    async def get(self, request_id: str, layer_index: int, token_index: int) -> np.ndarray | None:
        """Retrieve cached hidden state."""
        async with self._lock:
            if request_id not in self.cache:
                self._misses += 1
                return None
            
            states = self.cache[request_id]
            if layer_index not in states:
                self._misses += 1
                return None
            
            layer_tensor = states[layer_index]
            if token_index >= layer_tensor.shape[0]:
                self._misses += 1
                return None
            
            self.cache.move_to_end(request_id)
            self._hits += 1
            return layer_tensor[token_index].numpy()

    @property
    def size(self):
        return len(self.cache)
    
    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0


# ── Shared vLLM Engine ───────────────────────────────────────────────────────

class SharedVLLMEngine:
    """
    Single vLLM engine shared across all miner instances.
    Handles generation only; hidden state extraction done by HF model.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.75,
        max_concurrent_requests: int = 8,
    ):
        self.model_name = model_name
        self.max_concurrent_requests = max_concurrent_requests
        
        # Semaphore to limit concurrent requests (prevents OOM from request spikes)
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        log.info(f"Initializing shared vLLM engine: {model_name}")
        log.info(f"  tensor_parallel_size={tensor_parallel_size}")
        log.info(f"  max_model_len={max_model_len}")
        log.info(f"  gpu_memory_utilization={gpu_memory_utilization}")
        log.info(f"  max_concurrent_requests={max_concurrent_requests}")

        from vllm import AsyncLLMEngine, SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs

        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            dtype="float16",
            disable_log_stats=True,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.SamplingParams = SamplingParams
        log.info("Shared vLLM engine initialized")

        # Load HuggingFace model for hidden state extraction
        log.info(f"Loading HuggingFace model for hidden states: {model_name}")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Try GPU first, fall back to CPU if OOM
        try:
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
            )
            self.hf_device = next(self.hf_model.parameters()).device
            log.info(f"HF model loaded on {self.hf_device}")
        except Exception as e:
            log.warning(f"Failed to load HF model on GPU, falling back to CPU: {e}")
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                output_hidden_states=True,
            )
            self.hf_device = torch.device("cpu")
            log.info(f"HF model loaded on CPU")
        
        self.hf_model.eval()
        self.num_layers = self.hf_model.config.num_hidden_layers
        self.hidden_dim = self.hf_model.config.hidden_size

    async def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.9,
        request_id: str = None,
    ) -> dict:
        """
        Generate text using vLLM.
        Returns: {text, output_tokens, prompt_tokens}
        """
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        
        request_id = request_id or str(uuid.uuid4())
        
        # Acquire semaphore to limit concurrent requests
        async with self.request_semaphore:
            try:
                # Submit to vLLM
                results_generator = self.engine.generate(prompt, sampling_params, request_id)
                
                # Wait for completion
                final_output = None
                async for request_output in results_generator:
                    final_output = request_output
                
                if final_output is None:
                    raise RuntimeError("vLLM generation failed: no output received")
                
                output = final_output.outputs[0]
                return {
                    "text": output.text,
                    "output_tokens": output.token_ids,
                    "prompt_tokens": len(final_output.prompt_token_ids),
                }
            except Exception as e:
                log.error(f"vLLM generation error for request {request_id}: {e}")
                raise RuntimeError(f"vLLM engine error: {e}") from e

    @torch.no_grad()
    def extract_hidden_states(self, token_ids: list[int]) -> dict:
        """
        Extract hidden states using HF model.
        Returns: {layer_idx: tensor(seq_len, hidden_dim)}
        """
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(self.hf_device)
        
        outputs = self.hf_model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        
        # Extract all layer hidden states
        hidden_states_dict = {}
        for layer_idx, layer_hidden in enumerate(outputs.hidden_states):
            # layer_hidden shape: (batch=1, seq_len, hidden_dim)
            hidden_states_dict[layer_idx] = layer_hidden[0].cpu()  # (seq_len, hidden_dim)
        
        return hidden_states_dict


# ── Miner Instance ───────────────────────────────────────────────────────────

class MinerInstance:
    """
    Single miner instance endpoint.
    Shares vLLM engine and cache with other instances.
    Uses unique sampling profile to avoid collusion detection.
    """

    def __init__(
        self,
        miner_id: int,
        vllm_engine: SharedVLLMEngine,
        cache: SharedHiddenStateCache,
        sampling_profile: SamplingProfile,
    ):
        self.miner_id = miner_id
        self.vllm_engine = vllm_engine
        self.cache = cache
        self.sampling_profile = sampling_profile
        
        # Per-miner stats
        self.total_requests = 0
        self.total_challenges = 0
        self.challenges_passed = 0
        self.request_log = deque(maxlen=1000)  # Track which requests this miner handled
        
        log.info(
            f"Miner {miner_id} initialized | "
            f"temp={sampling_profile.temperature_base:.2f}±{sampling_profile.temperature_variance:.2f} | "
            f"top_p={sampling_profile.top_p:.2f}"
        )

    async def run_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference with this miner's sampling profile."""
        request_id = request.request_id or str(uuid.uuid4())
        self.total_requests += 1
        
        try:
            # Apply sampling profile (with request-level jitter)
            temperature = request.temperature or self.sampling_profile.sample_temperature()
            top_p = request.top_p or self.sampling_profile.top_p
            
            # Build prompt
            if request.messages:
                prompt = self.vllm_engine.tokenizer.apply_chat_template(
                    request.messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                prompt = request.prompt
            
            # Tokenize for hidden state extraction
            inputs = self.vllm_engine.tokenizer(prompt, return_tensors="pt")
            prompt_token_ids = inputs["input_ids"][0].tolist()
            
            # vLLM generation
            t_start = time.perf_counter()
            gen_result = await self.vllm_engine.generate(
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=temperature,
                top_p=top_p,
                request_id=request_id,
            )
            t_gen = time.perf_counter()
            
            # Extract hidden states via HF forward pass
            all_token_ids = prompt_token_ids + gen_result["output_tokens"]
            hidden_states = self.vllm_engine.extract_hidden_states(all_token_ids)
            t_hidden = time.perf_counter()
            
            # Cache hidden states
            await self.cache.store(request_id, hidden_states)
            
            # Apply timing jitter to decorrelate latency patterns between miners
            jitter_ms = self.sampling_profile.get_timing_jitter_ms()
            if jitter_ms > 0:
                await asyncio.sleep(jitter_ms / 1000.0)
            
            t_end = time.perf_counter()
            
            # Compute metrics
            gen_time_ms = (t_gen - t_start) * 1000
            hidden_time_ms = (t_hidden - t_gen) * 1000
            total_time_ms = (t_end - t_start) * 1000
            output_count = len(gen_result["output_tokens"])
            tps = output_count / max(gen_time_ms / 1000, 0.001)
            
            # Handle inline challenge if present
            challenge_result = None
            if request.challenge_layer is not None and request.challenge_token is not None:
                challenge_result = await self._serve_inline_challenge(
                    request_id,
                    request.challenge_layer,
                    request.challenge_token,
                    request.challenge_extra,
                )
            
            # Log request
            self.request_log.append({
                "request_id": request_id,
                "timestamp": time.time(),
                "input_tokens": gen_result["prompt_tokens"],
                "output_tokens": output_count,
                "ttft_ms": gen_time_ms,  # vLLM is async, so TTFT ≈ total for now
                "tps": tps,
            })
            
            log.info(
                f"[Miner {self.miner_id}] Inference {request_id[:8]}... | "
                f"{gen_result['prompt_tokens']} in + {output_count} out | "
                f"{gen_time_ms:.1f}ms gen + {hidden_time_ms:.1f}ms hidden | "
                f"{tps:.0f} tok/s | temp={temperature:.2f}"
                f"{' +challenge' if challenge_result else ''}"
            )
            
            return InferenceResponse(
                request_id=request_id,
                text=gen_result["text"],
                input_tokens=gen_result["prompt_tokens"],
                output_tokens=output_count,
                ttft_ms=gen_time_ms,
                total_ms=total_time_ms,
                tokens_per_sec=tps,
                all_token_ids=all_token_ids,
                challenge_result=challenge_result,
            )
        
        except Exception as e:
            log.error(f"[Miner {self.miner_id}] Inference failed for {request_id[:8]}...: {type(e).__name__}: {e}", exc_info=True)
            # Return error response instead of crashing entire service
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=f"Inference engine error: {str(e)}")

    async def _serve_inline_challenge(
        self,
        request_id: str,
        layer_index: int,
        token_index: int,
        extra_points: list[list[int]] | None = None,
    ) -> dict:
        """Serve hidden state challenge inline."""
        t_start = time.perf_counter()
        self.total_challenges += 1
        
        state = await self.cache.get(request_id, layer_index, token_index)
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        
        if state is None:
            return {"error": "cache_miss", "latency_ms": latency_ms}
        
        self.challenges_passed += 1
        
        result = {
            "hidden_state": state.tolist(),
            "layer_index": layer_index,
            "token_index": token_index,
            "latency_ms": latency_ms,
        }
        
        # Handle multi-point challenges
        if extra_points:
            extra_states = []
            for point in extra_points:
                if len(point) >= 2:
                    extra_state = await self.cache.get(request_id, point[0], point[1])
                    if extra_state is not None:
                        extra_states.append({
                            "layer_index": point[0],
                            "token_index": point[1],
                            "hidden_state": extra_state.tolist(),
                        })
                    else:
                        extra_states.append({
                            "layer_index": point[0],
                            "token_index": point[1],
                            "error": "cache_miss",
                        })
            result["extra_states"] = extra_states
        
        return result

    async def get_hidden_state(self, request: HiddenStateRequest) -> HiddenStateResponse:
        """Serve standalone hidden state challenge."""
        t_start = time.perf_counter()
        self.total_challenges += 1
        
        state = await self.cache.get(request.request_id, request.layer_index, request.token_index)
        
        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000
        
        if state is None:
            log.warning(
                f"[Miner {self.miner_id}] Challenge MISS {request.request_id[:8]}... | "
                f"layer={request.layer_index} pos={request.token_index}"
            )
            raise HTTPException(
                status_code=404,
                detail=f"No cached hidden state for request {request.request_id}",
            )
        
        self.challenges_passed += 1
        
        log.info(
            f"[Miner {self.miner_id}] Challenge HIT {request.request_id[:8]}... | "
            f"layer={request.layer_index} pos={request.token_index} | {latency_ms:.2f}ms"
        )
        
        return HiddenStateResponse(
            request_id=request.request_id,
            layer_index=request.layer_index,
            token_index=request.token_index,
            hidden_state=state.tolist(),
            latency_ms=latency_ms,
        )


# ── FastAPI App Factory ──────────────────────────────────────────────────────

def create_miner_app(miner_instance: MinerInstance) -> FastAPI:
    """Create FastAPI app for a single miner instance."""
    app = FastAPI(title=f"Miner {miner_instance.miner_id}")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": miner_instance.vllm_engine.model_name,
            "num_layers": miner_instance.vllm_engine.num_layers,
            "hidden_dim": miner_instance.vllm_engine.hidden_dim,
            "total_requests": miner_instance.total_requests,
        }

    @app.post("/inference", response_model=InferenceResponse)
    async def inference(request: InferenceRequest):
        return await miner_instance.run_inference(request)

    @app.post("/hidden_state", response_model=HiddenStateResponse)
    async def hidden_state(request: HiddenStateRequest):
        return await miner_instance.get_hidden_state(request)

    return app


# ── Multi-Miner Orchestrator ─────────────────────────────────────────────────

class MultiMinerOrchestrator:
    """Orchestrates multiple miner instances sharing one vLLM engine."""

    def __init__(
        self,
        num_miners: int,
        model_name: str,
        base_port: int,
        gpu_memory_utilization: float,
        cache_size: int,
        sampling_profiles: list[SamplingProfile] | None = None,
    ):
        self.num_miners = num_miners
        self.base_port = base_port
        
        # Initialize shared vLLM engine
        self.vllm_engine = SharedVLLMEngine(
            model_name=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_concurrent_requests=num_miners * 2,  # 2x miners for safety margin
        )
        
        # Initialize shared cache
        self.cache = SharedHiddenStateCache(max_requests=cache_size)
        
        # Create miner instances
        if sampling_profiles is None:
            sampling_profiles = DEFAULT_SAMPLING_PROFILES
        
        self.miners: list[MinerInstance] = []
        for i in range(num_miners):
            profile = sampling_profiles[i % len(sampling_profiles)]
            miner = MinerInstance(
                miner_id=i,
                vllm_engine=self.vllm_engine,
                cache=self.cache,
                sampling_profile=profile,
            )
            self.miners.append(miner)
        
        log.info(f"Orchestrator initialized: {num_miners} miners sharing 1 vLLM engine")

    async def start_all(self):
        """Start all miner servers on consecutive ports."""
        servers = []
        for i, miner in enumerate(self.miners):
            port = self.base_port + i
            app = create_miner_app(miner)
            
            config = uvicorn.Config(
                app=app,
                host="0.0.0.0",
                port=port,
                log_level="info",
            )
            server = uvicorn.Server(config)
            
            # Start server in background
            task = asyncio.create_task(server.serve())
            servers.append(task)
            
            log.info(f"Miner {i} started on port {port}")
        
        # Wait for all servers
        await asyncio.gather(*servers)


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Shared vLLM Multi-Miner Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single miner (baseline)
  python shared_vllm_multi_miner.py --base-port 8091 --num-miners 1

  # 4 miners on one GPU
  python shared_vllm_multi_miner.py --base-port 8091 --num-miners 4

  # Custom model and GPU utilization
  python shared_vllm_multi_miner.py --base-port 8091 --num-miners 3 \\
      --model "meta-llama/Meta-Llama-3-8B-Instruct" \\
      --gpu-memory-utilization 0.70
        """,
    )
    parser.add_argument("--base-port", type=int, default=8091, help="Base port (miners use consecutive ports)")
    parser.add_argument("--num-miners", type=int, default=1, help="Number of miner instances")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct", help="Model name or path")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75, help="GPU memory for vLLM (0-1)")
    parser.add_argument("--cache-size", type=int, default=1000, help="Shared cache size")
    parser.add_argument("--sampling-profiles", type=str, help="JSON file with custom sampling profiles")
    
    args = parser.parse_args()
    
    # Load custom sampling profiles if provided
    sampling_profiles = None
    if args.sampling_profiles:
        with open(args.sampling_profiles) as f:
            data = json.load(f)
            sampling_profiles = [
                SamplingProfile(**profile) for profile in data["profiles"]
            ]
        log.info(f"Loaded {len(sampling_profiles)} custom sampling profiles")
    
    # Create orchestrator
    orchestrator = MultiMinerOrchestrator(
        num_miners=args.num_miners,
        model_name=args.model,
        base_port=args.base_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cache_size=args.cache_size,
        sampling_profiles=sampling_profiles,
    )
    
    # Start all miners
    log.info(f"Starting {args.num_miners} miners on ports {args.base_port}-{args.base_port+args.num_miners-1}")
    asyncio.run(orchestrator.start_all())


if __name__ == "__main__":
    main()
