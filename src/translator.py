import openai
import os
import time
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dotenv import load_dotenv


# Language code to name mapping
LANGUAGE_NAMES = {
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "ko": "韩文",
    "es": "西班牙语",
    "fr": "法语",
    "de": "德语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "hi": "印地语",
    "pt": "葡萄牙语",
    "it": "意大利语",
    "nl": "荷兰语",
    "pl": "波兰语",
    "tr": "土耳其语",
    "vi": "越南语",
    "th": "泰语",
    "sv": "瑞典语",
    "auto": "自动检测"
}


def get_project_root() -> Path:
    """Get the project root directory (parent of src directory)"""
    # Start from the current file's location and go up to find project root
    current_file = Path(__file__).resolve()
    src_dir = current_file.parent
    return src_dir.parent


def get_prompt_path(prompt_path: str = "prompt.md") -> str:
    """
    Get the absolute path to prompt.md

    Args:
        prompt_path: Path to the prompt file (can be relative or absolute)

    Returns:
        Absolute path to the prompt file
    """
    # If already absolute, return as is
    if os.path.isabs(prompt_path):
        return prompt_path

    # If relative, resolve from project root
    project_root = get_project_root()
    absolute_path = project_root / prompt_path

    # If not found in project root, try current directory (for backward compatibility)
    if not absolute_path.exists():
        # Try as-is from current directory
        if os.path.exists(prompt_path):
            return prompt_path
        raise FileNotFoundError(f"Prompt file not found: {prompt_path} (tried: {absolute_path}, {os.path.abspath(prompt_path)})")

    return str(absolute_path)


class UniversalTranslator:
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat", base_url: Optional[str] = None):
        """
        Initialize the universal translator for different models

        Args:
            api_key: API key for the model. If not provided, will try to load from environment
            model: Model name to use (e.g., "deepseek-chat", "gemini", "qwen")
            base_url: Custom base URL for the API (if different from default)
        """
        load_dotenv()

        # Read from environment variables if not provided
        self.model = model or os.getenv('TRANSLATION_MODEL', 'deepseek-chat')
        self.base_url = base_url or os.getenv('TRANSLATION_URL')

        # Service provider configurations
        service_configs = {
            "deepseek": {
                "env_key": "DEEPSEEK_API_KEY",
                "default_url": "https://api.deepseek.com",
                "model_env": "DEEPSEEK_MODEL",
                "default_model": "deepseek-chat"
            },
            "gemini": {
                "env_key": "GEMINI_API_KEY",
                "default_url": "https://generativelanguage.googleapis.com/v1beta",
                "model_env": "GEMINI_MODEL",
                "default_model": "gemini-2.5-flash"
            },
            "qwen": {
                "env_key": "QWEN_API_KEY",
                "default_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_env": "QWEN_MODEL",
                "default_model": "qwen-plus"
            },
            "claude": {
                "env_key": "ANTHROPIC_API_KEY",
                "default_url": "https://api.anthropic.com",
                "model_env": "CLAUDE_MODEL",
                "default_model": "claude-3-sonnet-20240229"
            },
            "gpt": {
                "env_key": "OPENAI_API_KEY",
                "default_url": "https://api.openai.com/v1",
                "model_env": "OPENAI_MODEL",
                "default_model": "gpt-3.5-turbo"
            }
        }

        # Handle backward compatibility for old model names
        service_mapping = {
            "deepseek-chat": "deepseek",
            "gemini": "gemini",
            "gemini-pro": "gemini",
            "gemini-2.5-pro": "gemini",
            "qwen": "qwen",
            "qwen-max": "qwen",
            "claude": "claude",
            "claude-opus": "claude",
            "gpt": "gpt",
            "gpt-4": "gpt",
            "gpt-4-turbo": "gpt"
        }

        # Determine service provider
        if model == "custom":
            self.service = "custom"
        elif model in service_mapping:
            self.service = service_mapping[model]
        elif model in service_configs:
            self.service = model
        else:
            # For unknown models, treat as custom
            self.service = "custom"

        # Handle custom model
        if self.service == "custom":
            if not api_key and not os.getenv('TRANSLATION_API_KEY'):
                raise ValueError("API key is required for custom model. Set TRANSLATION_API_KEY environment variable or pass api_key parameter.")
            if not base_url and not os.getenv('TRANSLATION_URL'):
                raise ValueError("Base URL is required for custom model. Set TRANSLATION_URL environment variable or pass base_url parameter.")

            self.api_key = api_key or os.getenv('TRANSLATION_API_KEY')
            configured_url = base_url or os.getenv('TRANSLATION_URL')
            self.actual_model_name = model  # Use the provided model name directly
        else:
            service_config = service_configs[self.service]
            self.api_key = api_key or os.getenv(service_config["env_key"]) or os.getenv('TRANSLATION_API_KEY')
            if not self.api_key:
                raise ValueError(f"API key is required for {self.service}. Set {service_config['env_key']} or TRANSLATION_API_KEY environment variable or pass api_key parameter.")

            configured_url = base_url or os.getenv('TRANSLATION_URL') or service_config["default_url"]

            # Get model name from environment variable or use default
            self.actual_model_name = os.getenv(service_config["model_env"]) or service_config["default_model"]

        # Configure OpenAI client
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=configured_url
        )

    def load_prompt(self, prompt_path: str = "prompt.md") -> str:
        """
        Load translation prompt from file

        Args:
            prompt_path: Path to the prompt file

        Returns:
            Prompt content as string
        """
        # Resolve to absolute path
        absolute_prompt_path = get_prompt_path(prompt_path)

        if not os.path.exists(absolute_prompt_path):
            raise FileNotFoundError(f"Prompt file not found: {absolute_prompt_path}")

        with open(absolute_prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def translate_srt(self, srt_content: str, prompt_path: str = "prompt.md",
                     max_retries: int = 3, chunk_size: int = 20, target_language: str = "zh") -> str:
        """
        Translate SRT content to target language using the configured model with chunking

        Args:
            srt_content: Original SRT content in any language
            prompt_path: Path to the translation prompt file
            max_retries: Maximum number of retries for failed requests
            chunk_size: Number of subtitle entries per chunk (default: 20)
            target_language: Target language code (default: zh for Chinese)

        Returns:
            Translated SRT content in target language
        """
        print(f"Starting translation to {target_language} with {self.model} model...")

        # Parse SRT content into chunks
        chunks = self._split_srt_into_chunks(srt_content, chunk_size)
        print(f"Split into {len(chunks)} chunks for translation")

        # Load the translation prompt
        system_prompt = self.load_prompt(prompt_path)

        translated_chunks = []

        for i, chunk in enumerate(chunks, 1):
            print(f"Translating chunk {i}/{len(chunks)}...")

            # Get the starting index for this chunk to maintain numbering
            start_index = sum(len(self._parse_srt_content(c)) for c in chunks[:i])

            translated_chunk = self._translate_chunk(
                chunk, system_prompt, max_retries, start_index, target_language
            )

            translated_chunks.append(translated_chunk)

            # Add a small delay between chunks to avoid rate limiting (reduced from 1s to 0.1s)
            if i < len(chunks):
                time.sleep(0.1)

        # Combine all translated chunks
        final_translation = self._combine_translated_chunks(translated_chunks)
        print("Translation completed successfully!")

        return final_translation

    def save_translated_srt(self, content: str, output_path: str) -> None:
        """
        Save translated SRT content to file

        Args:
            content: Translated SRT content
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved translated subtitles to: {output_path}")

    def _split_srt_into_chunks(self, srt_content: str, chunk_size: int) -> List[str]:
        """
        Split SRT content into chunks of specified size

        Args:
            srt_content: Complete SRT content
            chunk_size: Number of subtitle entries per chunk

        Returns:
            List of SRT content chunks
        """
        # Parse SRT into individual subtitle blocks
        subtitle_blocks = self._parse_srt_content(srt_content)

        # Split into chunks
        chunks = []
        for i in range(0, len(subtitle_blocks), chunk_size):
            chunk = subtitle_blocks[i:i + chunk_size]
            chunk_content = '\n\n'.join(chunk)
            chunks.append(chunk_content)

        return chunks

    def _parse_srt_content(self, srt_content: str) -> List[str]:
        """
        Parse SRT content into individual subtitle blocks

        Args:
            srt_content: SRT content string

        Returns:
            List of subtitle blocks
        """
        # Split by empty lines to get individual subtitle blocks
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        return [block.strip() for block in blocks if block.strip()]

    def _translate_chunk(self, chunk: str, system_prompt: str, max_retries: int,
                         start_index: int, target_language: str = "zh") -> str:
        """
        Translate a single chunk of SRT content

        Args:
            chunk: SRT chunk to translate
            system_prompt: System prompt for translation
            max_retries: Maximum number of retries
            start_index: Starting subtitle index for this chunk
            target_language: Target language code (zh, en, ja, etc.)

        Returns:
            Translated chunk content
        """
        # Get target language name for the prompt
        target_lang_name = LANGUAGE_NAMES.get(target_language, target_language)

        # Enhanced user message with context about chunking
        user_message = f"""请翻译以下SRT字幕文件内容到{target_lang_name}。这是一个长字幕文件的第{start_index + 1}部分，请严格遵守以下所有规则：

1. 【格式保真】: 严格保持原文的SRT格式，包括但不限于序号、时间码（`HH:MM:SS,mmm --> HH:MM:SS,mmm`）以及空行，不得有任何改动
2. 【内容忠实】: 必须逐句、精准地翻译原文对话，不得添加、删减或篡改原文信息。翻译必须忠实于说话者的原始意图、情绪和语气
3. 【语气词处理】: 对于原文中频繁出现的无实义语气词，需根据上下文灵活处理。如果该语气词传递了犹豫、肯定、惊讶等情绪，则应翻译为对应的{target_lang_name}语气词；如果仅为口头禅或无意义的停顿，则可酌情省略，以保证{target_lang_name}文本的流畅性
4. 【错误推测】: 如果原文中存在疑似语音识别错误或语义不通顺的句子，请结合上下文进行最合理的推测性翻译，不需要括号注明
5. 【字幕序号处理】: 字幕序号从{start_index + 1}开始递增，确保序号连续性
6. 【输出格式】: **重要**：直接输出SRT格式内容，不要使用markdown代码块（```），不要添加任何额外格式

字幕内容：

{chunk}

请严格按照翻译规则进行翻译，并保持完整的SRT格式。"""

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.actual_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=6000  # Increased max_tokens for larger chunks
                )

                translated_content = response.choices[0].message.content

                if not translated_content:
                    raise ValueError("Empty response from DeepSeek API")

                # Clean the response to remove any markdown formatting
                cleaned_content = self._clean_srt_content(translated_content)

                return cleaned_content.strip()

            except Exception as e:
                print(f"Chunk translation attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Retrying chunk in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Chunk translation failed after {max_retries} attempts: {str(e)}")

    def _combine_translated_chunks(self, translated_chunks: List[str]) -> str:
        """
        Combine translated chunks into final SRT content
        Args:
            translated_chunks: List of translated chunk contents
        Returns:
            Combined SRT content
        """
        # Remove empty lines at the beginning and end of each chunk
        cleaned_chunks = []
        for chunk in translated_chunks:
            cleaned_chunk = chunk.strip()
            if cleaned_chunk:
                cleaned_chunks.append(cleaned_chunk)

        # Join chunks with double newlines
        return '\n\n'.join(cleaned_chunks)

    def _clean_srt_content(self, content: str) -> str:
        """
        Clean SRT content by removing markdown formatting and other artifacts

        Args:
            content: Raw content from translation API

        Returns:
            Cleaned SRT content
        """
        if not content:
            return content

        # Remove JSON objects that might appear in the content
        content = re.sub(r'\{[^}]*"translated_srt"[^}]*\}', '', content, flags=re.DOTALL)

        # Remove lines that contain "json" (with or without surrounding whitespace)
        content = re.sub(r'^\s*json\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\{\s*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*\}\s*$', '', content, flags=re.MULTILINE)

        # Remove markdown code blocks (```srt, ```, ```text, etc.)
        # Pattern to match ``` followed by optional language identifier and content until closing ```
        content = re.sub(r'```(?:srt|text)?\s*\n?(.*?)\n?```', r'\1', content, flags=re.DOTALL)

        # Remove any remaining standalone ``` markers
        content = re.sub(r'```\s*', '', content)

        # Remove any markdown-style headers or formatting that might have been added
        content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)

        # Remove any explanatory text that might appear before or after SRT content
        lines = content.split('\n')
        cleaned_lines = []
        in_srt_content = False

        for line in lines:
            line = line.strip()

            # Skip empty lines at the beginning
            if not line and not in_srt_content:
                continue

            # Check if this looks like the start of SRT content
            if line.isdigit() or '-->' in line:
                in_srt_content = True
                cleaned_lines.append(line)
            elif in_srt_content:
                # Once we're in SRT content, keep all lines (including empty ones for spacing)
                cleaned_lines.append(line)
            elif any(keyword in line.lower() for keyword in ['翻译', 'translation', '字幕', 'subtitle']):
                # Skip explanatory lines but start looking for SRT content after this
                continue

        # If we didn't find any SRT-like content, return the original content cleaned of markdown
        if not cleaned_lines:
            # Final fallback: just remove markdown and return
            cleaned_content = re.sub(r'```[^`]*```', '', content, flags=re.DOTALL)
            return cleaned_content.strip()

        # Rejoin the lines, preserving the structure
        result = '\n'.join(cleaned_lines)

        # Final cleanup: remove any multiple consecutive empty lines
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)

        return result.strip()

    @classmethod
    def create_translator(cls, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> 'UniversalTranslator':
        """
        Create a translator instance for a specific model

        Args:
            model: Model name (deepseek-chat, gemini, qwen, claude, gpt)
            api_key: API key for the model
            base_url: Custom base URL for the API

        Returns:
            Configured translator instance
        """
        return cls(api_key=api_key, model=model, base_url=base_url)

    @staticmethod
    def list_supported_models() -> List[str]:
        """
        List all supported service providers and models

        Returns:
            List of supported model names
        """
        return [
            "deepseek", "deepseek-chat",
            "gemini", "gemini-pro", "gemini-2.5-pro",
            "qwen", "qwen-max",
            "claude", "claude-opus",
            "gpt", "gpt-4", "gpt-4-turbo",
            "custom"
        ]

    @staticmethod
    def list_supported_services() -> List[str]:
        """
        List all supported service providers

        Returns:
            List of supported service provider names
        """
        return ["deepseek", "gemini", "qwen", "claude", "gpt", "custom"]

    @staticmethod
    def get_model_info(model: str) -> Dict[str, str]:
        """
        Get information about a specific model

        Args:
            model: Model name

        Returns:
            Dictionary containing model information
        """
        service_info = {
            "deepseek": {
                "name": "DeepSeek",
                "env_key": "DEEPSEEK_API_KEY",
                "model_env": "DEEPSEEK_MODEL",
                "default_url": "https://api.deepseek.com",
                "default_model": "deepseek-chat",
                "description": "DeepSeek AI service"
            },
            "gemini": {
                "name": "Google Gemini",
                "env_key": "GEMINI_API_KEY",
                "model_env": "GEMINI_MODEL",
                "default_url": "https://generativelanguage.googleapis.com/v1beta",
                "default_model": "gemini-1.5-flash",
                "description": "Google's Gemini AI service"
            },
            "qwen": {
                "name": "Alibaba Qwen",
                "env_key": "QWEN_API_KEY",
                "model_env": "QWEN_MODEL",
                "default_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "default_model": "qwen-plus",
                "description": "Alibaba's Qwen AI service"
            },
            "claude": {
                "name": "Anthropic Claude",
                "env_key": "ANTHROPIC_API_KEY",
                "model_env": "CLAUDE_MODEL",
                "default_url": "https://api.anthropic.com",
                "default_model": "claude-3-sonnet-20240229",
                "description": "Anthropic's Claude AI service"
            },
            "gpt": {
                "name": "OpenAI GPT",
                "env_key": "OPENAI_API_KEY",
                "model_env": "OPENAI_MODEL",
                "default_url": "https://api.openai.com/v1",
                "default_model": "gpt-3.5-turbo",
                "description": "OpenAI's GPT service"
            },
            "custom": {
                "name": "Custom Service",
                "env_key": "TRANSLATION_API_KEY",
                "model_env": "TRANSLATION_MODEL",
                "default_url": "TRANSLATION_URL",
                "default_model": "custom",
                "description": "Custom OpenAI-compatible service"
            }
        }

        # Handle service mapping for backward compatibility
        service_mapping = {
            "deepseek-chat": "deepseek",
            "gemini": "gemini",
            "gemini-pro": "gemini",
            "gemini-2.5-pro": "gemini",
            "qwen": "qwen",
            "qwen-max": "qwen",
            "claude": "claude",
            "claude-opus": "claude",
            "gpt": "gpt",
            "gpt-4": "gpt",
            "gpt-4-turbo": "gpt"
        }

        # Determine service
        if model in service_mapping:
            service = service_mapping[model]
        elif model in service_info:
            service = model
        else:
            service = "custom"

        if service not in service_info:
            raise ValueError(f"Unsupported service: {service}. Supported services: {list(service_info.keys())}")

        return service_info[service]


# Backward compatibility alias
class DeepSeekTranslator(UniversalTranslator):
    """
    Backward compatibility wrapper for DeepSeek translator
    """
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key, model="deepseek-chat")


# Example usage function
def example_usage():
    """
    Example usage of the UniversalTranslator class
    """
    # Example 1: Using DeepSeek (default)
    deepseek_translator = UniversalTranslator(api_key="your_deepseek_api_key", model="deepseek-chat")

    # Example 2: Using Gemini Flash
    gemini_translator = UniversalTranslator.create_translator("gemini", api_key="your_gemini_api_key")

    # Example 3: Using Qwen Max with custom URL
    qwen_translator = UniversalTranslator.create_translator(
        "qwen-max",
        api_key="your_qwen_api_key",
        base_url="https://custom-qwen-endpoint.com/v1"
    )

    # Example 4: List supported models
    print("Supported models:", UniversalTranslator.list_supported_models())

    # Example 5: Get model information
    model_info = UniversalTranslator.get_model_info("deepseek-chat")
    print(f"Model info: {model_info}")

    # Example 6: Using custom model
    custom_translator = UniversalTranslator.create_translator(
        "custom",
        api_key="your_custom_api_key",
        base_url="https://your-custom-endpoint.com/v1"
    )

    # Translation example
    srt_content = """
1
00:00:01,000 --> 00:00:03,000
こんにちは

2
00:00:04,000 --> 00:00:06,000
お元気ですか？
"""

    # Translate using DeepSeek
    translated_content = deepseek_translator.translate_srt(srt_content)
    print("Translated content:", translated_content)


if __name__ == "__main__":
    example_usage()