gemini-2.5-flash gemini-2.5-flash-preview-09-2025 gemini-2.5-flash-lite gemini-2.5-flash-lite-preview-09-2025

Free Tier Models for AI Agent
Overview
This guide highlights the free Gemini 2.5 models that work well for an AI agent sandbox. Use it as a quick reference when you need to pick a model that balances reasoning depth, latency, and cost.

Model Catalog
gemini-2.5-pro
gemini-2.5-flash
gemini-2.5-flash-preview-09-2025
gemini-2.5-flash-lite
gemini-2.5-flash-lite-preview-09-2025
Comparison
Rank Model Strength Best Use When to Avoid
🥇 1 gemini-2.5-pro Deepest reasoning, best accuracy Research, coding, STEM, complex multimodal Overkill for light tasks
🥈 2 gemini-2.5-flash Balanced power + speed Chatbots, summarizers, general assistants None — good all-rounder
🥉 3 gemini-2.5-flash-preview-09-2025 Newest Flash updates Testing latest Gemini changes Production apps
🏅 4 gemini-2.5-flash-lite Fastest, cheapest High-traffic bots, lightweight tasks Deep reasoning
🎖️ 5 gemini-2.5-flash-lite-preview-09-2025 Experimental efficiency Benchmarking, low-latency testing Stability-critical apps
Selection Tips
Reach for gemini-2.5-pro when accuracy and reasoning beat latency concerns.
Default to gemini-2.5-flash if you need reliable speed without sacrificing too much quality.
Try the preview or lite variants only when you can tolerate experimental behavior or shallower reasoning.

export const MODELS = {
  // Strong general / reasoning
  STRONG_GENERAL_REASONING: "meta-llama/llama-3.3-70b-instruct:free",
  ULTRA_STRONG_REASONING: "nousresearch/hermes-3-llama-3.1-405b:free",
  HEAVY_GENERAL_REASONING: "nvidia/nemotron-3-super-120b-a12b:free",
  OPENAI_STRONG_GENERAL: "openai/gpt-oss-120b:free",
  OPENAI_FAST_GENERAL: "openai/gpt-oss-20b:free",
  QWEN_STRONG_REASONING: "qwen/qwen3-next-80b-a3b-instruct:free",

  // Writing / balanced chat
  GOOGLE_BALANCED_WRITING_31B: "google/gemma-4-31b-it:free",
  GOOGLE_BALANCED_WRITING_26B: "google/gemma-4-26b-a4b-it:free",
  GEMMA_GENERAL_CHAT_27B: "google/gemma-3-27b-it:free",
  GEMMA_GENERAL_CHAT_12B: "google/gemma-3-12b-it:free",

  // Coding
  CODING_EXPERT_QWEN: "qwen/qwen3-coder:free",
  CODING_REASONING_DOLPHIN: "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",

  // Fast lightweight models
  FAST_LIGHT_NEMOTRON_9B: "nvidia/nemotron-nano-9b-v2:free",
  FAST_LIGHT_NEMOTRON_12B_MULTIMODAL: "nvidia/nemotron-nano-12b-v2-vl:free",
  FAST_LIGHT_NEMOTRON_30B: "nvidia/nemotron-3-nano-30b-a3b:free",
  FAST_LIGHT_GEMMA_4B: "google/gemma-3-4b-it:free",
  FAST_LIGHT_LLAMA_3B: "meta-llama/llama-3.2-3b-instruct:free",

  // Flash / quick chat
  FLASH_CHAT_LING: "inclusionai/ling-2.6-flash:free",
  FLASH_CHAT_GLM: "z-ai/glm-4.5-air:free",
  FLASH_CHAT_MINIMAX: "minimax/minimax-m2.5:free",
  FLASH_CHAT_TENCENT: "tencent/hy3-preview:free",

  // Small reasoning models
  MICRO_REASONING_LFM: "liquid/lfm-2.5-1.2b-thinking:free",
  MICRO_INSTRUCT_LFM: "liquid/lfm-2.5-1.2b-instruct:free",
  SMALL_REASONING_GEMMA_3N: "google/gemma-3n-e4b-it:free",

  // OCR / tools
  OCR_FAST_BAIDU: "baidu/qianfan-ocr-fast:free",

  // Router fallback (auto model selection)
  AUTO_FREE_MODEL: "openrouter/free"
};

<!-- pkill -f langgraphjs -->
