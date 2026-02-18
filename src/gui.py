#!/usr/bin/env python3
"""
OmniTranscribe GUI - Gradio Web Interface

A user-friendly web interface for OmniTranscribe that allows users to:
- Select source and target languages
- Upload audio/video files
- Upload background images for video generation
- Edit author/track information
- Choose output format (MP3 with metadata or MP4 with subtitles)
"""

import gradio as gr
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import shutil

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to load .env from the project root (parent of src directory)
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# Import local modules
from transcribe import AudioTranscriber
from translator import UniversalTranslator, get_prompt_path
from converter import SubtitleConverter
from video_converter import VideoConverter
from final_video_generator import FinalVideoGenerator
from simple_mp3_embedder import SimpleMP3Embedder
from config import get_config


# Language options
LANGUAGE_OPTIONS = {
    "自动检测": "auto",
    "中文": "zh",
    "英语": "en",
    "日语": "ja",
    "韩语": "ko",
    "西班牙语": "es",
    "法语": "fr",
    "德语": "de",
    "俄语": "ru",
    "阿拉伯语": "ar",
    "印地语": "hi",
    "葡萄牙语": "pt",
    "意大利语": "it",
    "荷兰语": "nl",
    "波兰语": "pl",
    "土耳其语": "tr",
    "越南语": "vi",
    "泰语": "th",
    "瑞典语": "sv",
}

# Whisper model options
MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large"]

# Device options
DEVICE_OPTIONS = ["auto", "cpu", "cuda", "mps"]

# Translation model options
TRANSLATION_MODEL_OPTIONS = ["google", "deepseek", "gemini", "qwen", "claude", "gpt"]

# Output format options
OUTPUT_FORMAT_OPTIONS = ["仅字幕 (SRT)", "MP4 视频 (带字幕)", "MP3 音频 (带歌词封面)"]


class OmniTranscribeGUI:
    """Gradio GUI for OmniTranscribe"""

    def __init__(self):
        self.config = get_config()
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

        # Set default paths
        project_root = Path(__file__).parent.parent
        self.default_background = project_root / "assets" / "default_background.png"
        self.default_font = project_root / "ChillDuanSansVF.ttf"

    def process_media(
        self,
        media_file: str,
        source_language: str,
        target_language: str,
        whisper_model: str,
        device: str,
        translation_model: str,
        output_format: str,
        background_image: Optional[str] = None,
        cover_image: Optional[str] = None,
        track_title: Optional[str] = "",
        artist_name: Optional[str] = "",
        album_name: Optional[str] = "",
        progress=gr.Progress()
    ) -> Tuple[str, str, str, Optional[str]]:
        """
        Process media file with transcription and translation

        Args:
            media_file: Path to uploaded media file
            source_language: Source language name (Chinese)
            target_language: Target language name (Chinese)
            whisper_model: Whisper model size
            device: Device for inference
            translation_model: Translation model to use
            output_format: Output format choice
            background_image: Background image for video generation
            cover_image: Cover image for MP3 metadata
            track_title: Track title for metadata
            artist_name: Artist name for metadata
            album_name: Album name for metadata
            progress: Gradio progress tracker

        Returns:
            Tuple of (original_srt_path, translated_srt_path, result_message, preview_file_path)
        """
        try:
            # Convert language names to codes
            source_lang_code = LANGUAGE_OPTIONS.get(source_language, source_language)
            target_lang_code = "none" if target_language == "不翻译" else LANGUAGE_OPTIONS.get(target_language, target_language)

            # Step 1: Validate inputs
            if not media_file:
                return None, None, "❌ 请上传音频/视频文件", None

            media_path = Path(media_file)
            if not media_path.exists():
                return None, None, f"❌ 文件不存在: {media_file}", None

            base_name = media_path.stem
            output_base = self.output_dir / base_name

            progress(0.1, desc="初始化中...")

            # Step 2: Convert video to audio if needed
            video_converter = VideoConverter()
            current_audio_path = str(media_path)

            if video_converter.is_video_file(str(media_path)):
                progress(0.2, desc="转换视频为音频...")
                current_audio_path, was_converted = video_converter.convert_to_audio(
                    str(media_path),
                    output_format='mp3',
                    output_dir=str(self.output_dir),
                    keep_original=True
                )

            # Step 3: Transcribe audio
            progress(0.3, desc=f"正在转录 ({source_language})...")
            transcriber = AudioTranscriber(model_size=whisper_model, device=device, use_cache=True)

            srt_content = transcriber.transcribe_audio(
                current_audio_path,
                language=source_lang_code,
                progress_callback=lambda current, total: progress(
                    0.3 + 0.2 * (current / total),
                    desc=f"转录中... {current}/{total}"
                )
            )

            original_srt_path = self.output_dir / f"{base_name}_original.srt"
            transcriber.save_srt(srt_content, str(original_srt_path))

            # Step 4: Translate subtitles
            if target_lang_code != "none":
                progress(0.5, desc=f"正在翻译为 {target_language}...")
                translator = UniversalTranslator(model=translation_model)

                translated_content = translator.translate_srt(
                    srt_content,
                    prompt_path=get_prompt_path(),
                    chunk_size=20,
                    target_language=target_lang_code
                )

                translated_srt_path = self.output_dir / f"{base_name}_translated.srt"
                translator.save_translated_srt(translated_content, str(translated_srt_path))
            else:
                translated_srt_path = None

            # Step 5: Generate output
            final_output_path = None

            if output_format == "MP4 视频 (带字幕)":
                # Use default background if none provided
                if not background_image:
                    if self.default_background.exists():
                        background_image = str(self.default_background)
                    else:
                        return str(original_srt_path), str(translated_srt_path) if translated_srt_path else "", "❌ 生成MP4需要背景图片（或默认背景不存在）", None

                progress(0.8, desc="生成MP4视频...")

                # Convert SRT to LRC for video generation
                converter = SubtitleConverter()
                if translated_srt_path:
                    # convert_file returns the actual output path
                    lyrics_file = converter.convert_file(str(translated_srt_path), "lrc")
                else:
                    # Use original subtitles
                    lyrics_file = converter.convert_file(str(original_srt_path), "lrc")

                video_output_path = self.output_dir / f"{base_name}_video.mp4"

                video_generator = FinalVideoGenerator()
                final_output_path = video_generator.create_video_from_existing_files(
                    audio_path=current_audio_path,
                    lyrics_path=lyrics_file,
                    image_path=background_image,
                    output_path=str(video_output_path)
                )

            elif output_format == "MP3 音频 (带歌词封面)":
                progress(0.8, desc="嵌入MP3元数据...")

                if translated_srt_path:
                    lyrics_file = str(translated_srt_path)
                else:
                    lyrics_file = str(original_srt_path)

                mp3_output_path = self.output_dir / f"{base_name}_with_metadata.mp3"

                embedder = SimpleMP3Embedder()
                final_output_path = embedder.embed_metadata(
                    audio_path=current_audio_path,
                    lyrics_path=lyrics_file,
                    cover_path=cover_image,
                    title=track_title if track_title else None,
                    artist=artist_name if artist_name else None,
                    album=album_name if album_name else None,
                    output_path=str(mp3_output_path)
                )

            progress(1.0, desc="完成!")

            # Return results
            result_message = "✅ 处理完成!\n\n"
            result_message += f"原始字幕: {original_srt_path}\n"
            if translated_srt_path:
                result_message += f"翻译字幕: {translated_srt_path}\n"
            if final_output_path:
                result_message += f"最终输出: {final_output_path}"

            return (
                str(original_srt_path),
                str(translated_srt_path) if translated_srt_path else "",
                result_message,
                str(final_output_path) if final_output_path else None
            )

        except Exception as e:
            return None, None, f"❌ 处理失败: {str(e)}", None

    def launch(self, share: bool = False):
        """Launch the Gradio interface"""

        custom_css = """
            .gradio-container {
                max-width: 100% !important;
                padding: 20px !important;
            }
            .header {
                text-align: center;
                padding: 20px;
            }
            .header h1 {
                margin-bottom: 5px;
            }
            /* 响应式布局 */
            @media (max-width: 768px) {
                .gradio-container {
                    padding: 10px !important;
                }
            }
            /* 确保列在小屏幕上堆叠 */
            .gradio-row {
                flex-wrap: wrap !important;
            }
            .gradio-column {
                min-width: 300px !important;
                flex: 1 1 300px !important;
            }
        """

        with gr.Blocks(
            title="OmniTranscribe - 多语言音频转录与翻译"
        ) as interface:

            # Header
            gr.HTML("""
                <div class="header">
                    <h1>🎙️ OmniTranscribe</h1>
                    <p>强大的多语言音频转录与翻译工具</p>
                </div>
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    # File upload
                    gr.Markdown("## 📁 文件上传")
                    media_file = gr.File(
                        label="上传音频/视频文件",
                        file_types=[".mp3", ".mp4", ".wav", ".m4a", ".avi", ".mov", ".mkv", ".flac", ".ogg"],
                        type="filepath"
                    )

                with gr.Column(scale=1):
                    # Language settings
                    gr.Markdown("## 🌍 语言设置")
                    source_language = gr.Dropdown(
                        choices=list(LANGUAGE_OPTIONS.keys()),
                        value="自动检测",
                        label="源语言",
                        info="音频/视频中的语言"
                    )
                    target_language = gr.Dropdown(
                        choices=["不翻译"] + list(LANGUAGE_OPTIONS.keys()),
                        value="中文",
                        label="目标语言",
                        info="翻译成的语言"
                    )

            with gr.Row():
                with gr.Column():
                    # Model settings
                    gr.Markdown("## ⚙️ 模型设置")
                    with gr.Row():
                        whisper_model = gr.Dropdown(
                            choices=MODEL_OPTIONS,
                            value="base",
                            label="Whisper 模型",
                            info="模型越大越准确但越慢"
                        )
                        device = gr.Dropdown(
                            choices=DEVICE_OPTIONS,
                            value="auto",
                            label="设备",
                            info="auto 会自动检测 GPU"
                        )
                    translation_model = gr.Dropdown(
                        choices=TRANSLATION_MODEL_OPTIONS,
                        value="deepseek",
                        label="翻译模型",
                        info="选择翻译服务"
                    )

                with gr.Column():
                    # Output format
                    gr.Markdown("## 📤 输出格式")
                    output_format = gr.Radio(
                        choices=OUTPUT_FORMAT_OPTIONS,
                        value="仅字幕 (SRT)",
                        label="选择输出格式"
                    )

                    # Show additional options based on output format
                    with gr.Group(visible=False) as mp4_options:
                        # Set default background if exists
                        default_bg_value = str(self.default_background) if self.default_background.exists() else None
                        background_image = gr.Image(
                            value=default_bg_value,
                            label="背景图片 (MP4) - 留空使用默认背景",
                            type="filepath",
                            sources=["upload", "clipboard"]
                        )
                        with gr.Row():
                            subtitle_position = gr.Radio(
                                choices=["top", "bottom", "center"],
                                value="bottom",
                                label="字幕位置",
                                scale=1
                            )
                            subtitle_fontsize = gr.Slider(
                                minimum=24,
                                maximum=72,
                                value=48,
                                step=4,
                                label="字幕大小",
                                scale=1
                            )

                    with gr.Group(visible=False) as mp3_options:
                        cover_image = gr.Image(
                            label="封面图片 (MP3)",
                            type="filepath"
                        )
                        track_title = gr.Textbox(
                            label="歌曲标题",
                            placeholder="输入歌曲标题"
                        )
                        with gr.Row():
                            artist_name = gr.Textbox(
                                label="艺术家",
                                placeholder="输入艺术家名称"
                            )
                            album_name = gr.Textbox(
                                label="专辑",
                                placeholder="输入专辑名称"
                            )

            # Process button
            process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

            # Output section
            gr.Markdown("## 📊 处理结果")
            with gr.Row():
                original_srt_output = gr.File(label="原始字幕 (SRT)")
                translated_srt_output = gr.File(label="翻译字幕 (SRT)")

            final_output = gr.Textbox(
                label="处理状态",
                lines=5,
                interactive=False
            )

            # Preview section
            gr.Markdown("## 🎬 预览")
            preview_path = gr.State(None)  # Hidden state for preview path
            with gr.Row():
                video_preview = gr.Video(
                    label="MP4 视频预览",
                    visible=False,
                    autoplay=True
                )
                audio_preview = gr.Audio(
                    label="MP3 音频预览",
                    visible=False
                )

            # Event handlers
            def update_output_options(output_format):
                """Show/hide options based on output format"""
                if output_format == "MP4 视频 (带字幕)":
                    return gr.update(visible=True), gr.update(visible=False)
                elif output_format == "MP3 音频 (带歌词封面)":
                    return gr.update(visible=False), gr.update(visible=True)
                else:
                    return gr.update(visible=False), gr.update(visible=False)

            output_format.change(
                update_output_options,
                inputs=[output_format],
                outputs=[mp4_options, mp3_options]
            )

            # Process button click
            def show_preview(output_format, preview_path):
                """Show appropriate preview based on output format and file path"""
                if output_format == "MP4 视频 (带字幕)" and preview_path:
                    return gr.update(visible=True, value=preview_path), gr.update(visible=False)
                elif output_format == "MP3 音频 (带歌词封面)" and preview_path:
                    return gr.update(visible=False), gr.update(visible=True, value=preview_path)
                else:
                    return gr.update(visible=False), gr.update(visible=False)

            process_btn.click(
                fn=self.process_media,
                inputs=[
                    media_file,
                    source_language,
                    target_language,
                    whisper_model,
                    device,
                    translation_model,
                    output_format,
                    background_image,
                    cover_image,
                    track_title,
                    artist_name,
                    album_name,
                ],
                outputs=[original_srt_output, translated_srt_output, final_output, preview_path]
            ).then(
                fn=show_preview,
                inputs=[output_format, preview_path],
                outputs=[video_preview, audio_preview]
            )

            # Footer
            gr.HTML("""
                <div style="text-align: center; margin-top: 30px; padding: 20px; border-top: 1px solid #e0e0e0;">
                    <p>用 ❤️ 为多语言音频内容爱好者打造</p>
                    <p style="font-size: 0.9em; color: #666;">
                        支持转录、翻译、字幕转换、视频生成、MP3 元数据嵌入
                    </p>
                </div>
            """)

        # Launch
        interface.launch(
            share=share,
            show_error=True,
            theme=gr.themes.Soft(),
            css=custom_css
        )


def main():
    """Main entry point for GUI"""
    import argparse

    parser = argparse.ArgumentParser(description="OmniTranscribe GUI")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    args = parser.parse_args()

    gui = OmniTranscribeGUI()
    gui.launch(share=args.share)


if __name__ == "__main__":
    main()
