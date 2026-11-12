import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int | list[int] = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_speculative_tokens: int = 0      # MTP draft tokens per step (hybrid Qwen3.5 only); 0 = off
    model_type: str = ""

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        cfg = AutoConfig.from_pretrained(self.model)
        self.model_type = cfg.model_type
        if hasattr(cfg, "text_config"):          # multimodal checkpoints: keep the language model only
            cfg = cfg.text_config
        self.hf_config = cfg
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if self.is_hybrid:
            # Restrictions for the hybrid model: single GPU, whole-prompt prefill,
            # no prefix caching (recurrent state cannot be rebuilt from cached KV blocks)
            assert self.tensor_parallel_size == 1, "Qwen3.5 baseline supports TP=1 only"
            self.max_num_seqs = min(self.max_num_seqs, 8)
            self.max_num_batched_tokens = max(self.max_num_batched_tokens, self.max_model_len)
        else:
            assert self.num_speculative_tokens == 0, "speculative decoding is implemented for Qwen3.5 MTP only"

    @property
    def is_hybrid(self) -> bool:
        return self.model_type.startswith("qwen3_5")
