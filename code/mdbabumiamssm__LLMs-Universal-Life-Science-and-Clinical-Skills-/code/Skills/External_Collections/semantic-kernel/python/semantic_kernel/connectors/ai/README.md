<!--
# COPYRIGHT NOTICE
# This file is part of the "Universal Biomedical Skills" project.
# Copyright (c) 2026 MD BABU MIA, PhD <md.babu.mia@mssm.edu>
# All Rights Reserved.
#
# This code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
#
# Provenance: Authenticated by MD BABU MIA

-->

# AI Connectors

This directory contains the implementation of the AI connectors (aka AI services) that are used to interact with AI models.

Depending on the modality, the AI connector can inherit from one of the following classes:

- [`ChatCompletionClientBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/chat_completion_client_base.py) for chat completion tasks.
- [`TextCompletionClientBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/text_completion_client_base.py) for text completion tasks.
- [`AudioToTextClientBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/audio_to_text_client_base.py) for audio to text tasks.
- [`TextToAudioClientBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/text_to_audio_client_base.py) for text to audio tasks.
- [`TextToImageClientBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/text_to_image_client_base.py) for text to image tasks.
- [`EmbeddingGeneratorBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/embeddings/embedding_generator_base.py) for text embedding tasks.

All base clients inherit from the [`AIServiceClientBase`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/services/ai_service_client_base.py) class.

## Existing AI connectors

| Services          | Connectors                          |
|-------------------------|--------------------------------------|
| OpenAI     | [`OpenAIChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_chat_completion.py) |
|            | [`OpenAITextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_text_completion.py) |
|            | [`OpenAITextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_text_embedding.py) |
|            | [`OpenAITextToImage`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_text_to_image.py) |
|            | [`OpenAITextToAudio`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_text_to_audio.py) |
|            | [`OpenAIAudioToText`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/open_ai_audio_to_text.py) |
| Azure OpenAI | [`AzureChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/azure_chat_completion.py) |
|            | [`AzureTextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/azure_text_embedding.py) |
|            | [`AzureTextToImage`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/azure_text_to_image.py) |
|            | [`AzureTextToAudio`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/azure_text_to_audio.py) |
|            | [`AzureAudioToText`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/open_ai/services/azure_audio_to_text.py) |
| Azure AI Inference | [`AzureAIInferenceChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/azure_ai_inference/services/azure_ai_inference_chat_completion.py) |
|            | [`AzureAIInferenceTextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/azure_ai_inference/services/azure_ai_inference_text_embedding.py) |
| Anthropic | [`AnthropicChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/anthropic/services/anthropic_chat_completion.py) |
| [Bedrock](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/bedrock/README.md) | [`BedrockChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/bedrock/services/bedrock_chat_completion.py) |
|         | [`BedrockTextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/bedrock/services/bedrock_text_completion.py) |
|         | [`BedrockTextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/bedrock/services/bedrock_text_embedding.py) |
| [Google AI](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/README.md) | [`GoogleAIChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/google_ai/services/google_ai_chat_completion.py) |
|           | [`GoogleAITextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/google_ai/services/google_ai_text_completion.py) |
|           | [`GoogleAITextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/google_ai/services/google_ai_text_embedding.py) |
| [Vertex AI](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/README.md) | [`GoogleAIChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/google_ai/services/google_ai_chat_completion.py) |
|           | [`GoogleAITextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/google_ai/services/google_ai_text_completion.py) |
|           | [`GoogleAITextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/google/google_ai/services/google_ai_text_embedding.py) |
| HuggingFace | [`HuggingFaceTextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/hugging_face/services/hf_text_completion.py) |
|             | [`HuggingFaceTextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/hugging_face/services/hf_text_embedding.py) |
| Mistral AI | [`MistralAIChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/mistral_ai/services/mistral_ai_chat_completion.py) |
|            | [`MistralAITextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/mistral_ai/services/mistral_ai_text_embedding.py) |
| [Nvidia](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/nvidia/README.md) | [`NvidiaTextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/nvidia/services/nvidia_text_embedding.py) |
| Ollama | [`OllamaChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/ollama/services/ollama_chat_completion.py) |
|        | [`OllamaTextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/ollama/services/ollama_text_completion.py) |
|        | [`OllamaTextEmbedding`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/ollama/services/ollama_text_embedding.py) |
| Onnx | [`OnnxGenAIChatCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/onnx/services/onnx_gen_ai_chat_completion.py) |
|      | [`OnnxGenAITextCompletion`](https://github.com/mdbabumiamssm/LLMs-Universal-Life-Science-and-Clinical-Skills-/blob/HEAD/Skills/External_Collections/semantic-kernel/python/semantic_kernel/connectors/ai/onnx/services/onnx_gen_ai_text_completion.py) |

<!-- AUTHOR_SIGNATURE: 9a7f3c2e-MD-BABU-MIA-2026-MSSM-SECURE -->