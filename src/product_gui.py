from __future__ import annotations

import html
import inspect
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr

from app_settings import AppSettings
from project_store import ProjectStore
from resumable_pipeline import ResumablePipeline, bootstrap_runtime


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
CODE_TO_LANGUAGE = {value: key for key, value in LANGUAGE_OPTIONS.items()}
TRANSLATION_MODELS = [
    ("Google Translate · 免费免配置", "google"),
    ("DeepSeek", "deepseek"),
    ("Gemini", "gemini"),
    ("Qwen", "qwen"),
    ("Claude", "claude"),
    ("OpenAI", "gpt"),
]
STAGE_LABELS = {
    "imported": "已导入",
    "transcribing": "正在转录",
    "transcription_paused": "转录已暂停",
    "transcribed": "待校对",
    "editing": "校对中",
    "translating": "正在翻译",
    "translation_paused": "翻译已暂停",
    "translated": "译文待校对",
    "ready_to_export": "可导出",
    "exported": "已导出",
}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm", ".m4v"}


class ProductGUI:
    def __init__(self):
        bootstrap_runtime()
        self.settings = AppSettings()
        self.settings.apply_to_environment()
        self.store = ProjectStore()
        self.pipeline = ResumablePipeline(self.store)
        self.stop_flags: Dict[Tuple[str, str], threading.Event] = {}

    def _flag(self, project_id: str, task: str) -> threading.Event:
        key = (project_id, task)
        if key not in self.stop_flags:
            self.stop_flags[key] = threading.Event()
        return self.stop_flags[key]

    def _project_choices(self):
        return [
            (
                f"{project.name} · {STAGE_LABELS.get(project.stage, project.stage)} · {project.id[:8]}",
                project.id,
            )
            for project in self.store.list_projects()
        ]

    def _summary(self, project_id: Optional[str]) -> str:
        if not project_id:
            return "<div class='empty-state'>创建或打开一个项目后，这里会显示处理进度。</div>"
        project = self.store.get_project(project_id)
        transcribe = self.store.get_checkpoint(project_id, "transcription")
        translate = self.store.get_checkpoint(project_id, "translation")

        def progress_text(checkpoint):
            total = int(checkpoint.get("total_units", 0) or 0)
            done = int(checkpoint.get("completed_units", 0) or 0)
            return "尚未开始" if not total else f"{done}/{total}"

        stage = html.escape(STAGE_LABELS.get(project.stage, project.stage))
        name = html.escape(project.name)
        source = html.escape(Path(project.source_path).name)
        return f"""
        <div class="project-card">
          <div class="project-card-top">
            <div><div class="eyebrow">当前项目</div><h2>{name}</h2></div>
            <span class="stage-pill">{stage}</span>
          </div>
          <div class="meta-grid">
            <div><span>源文件</span><strong>{source}</strong></div>
            <div><span>转录</span><strong>{progress_text(transcribe)}</strong></div>
            <div><span>翻译</span><strong>{progress_text(translate)}</strong></div>
            <div><span>字幕</span><strong>{len(self.store.list_segments(project_id))} 条</strong></div>
          </div>
        </div>
        """

    def _editor_rows(self, project_id: Optional[str]):
        return [] if not project_id else self.pipeline.editor_rows(project_id)

    def _preview_updates(self, project_id: Optional[str]):
        if not project_id:
            return gr.update(visible=False, value=None), gr.update(visible=False, value=None)
        project = self.store.get_project(project_id)
        if Path(project.source_path).suffix.lower() in VIDEO_EXTENSIONS:
            return gr.update(visible=True, value=project.source_path), gr.update(visible=False, value=None)
        return gr.update(visible=False, value=None), gr.update(visible=True, value=project.source_path)

    def create_project(
        self,
        media_file: Optional[str],
        name: str,
        source_language: str,
        target_language: str,
        whisper_model: str,
        translation_model: str,
    ):
        if not media_file:
            raise gr.Error("请先选择音频或视频文件")
        project = self.pipeline.create_project(
            media_file,
            name=name or None,
            source_language=LANGUAGE_OPTIONS.get(source_language, source_language),
            target_language=LANGUAGE_OPTIONS.get(target_language, target_language),
            whisper_model=whisper_model,
            translation_model=translation_model,
        )
        return (
            project.id,
            gr.update(choices=self._project_choices(), value=project.id),
            self._summary(project.id),
            self._editor_rows(project.id),
            f"已创建项目「{project.name}」。下一步可以开始转录。",
            *self._preview_updates(project.id),
        )

    def load_project(self, project_id: Optional[str]):
        if not project_id:
            return None, self._summary(None), [], "请选择一个项目", gr.update(), gr.update(), "自动检测", "中文", "base", "google"
        project = self.store.get_project(project_id)
        return (
            project_id,
            self._summary(project_id),
            self._editor_rows(project_id),
            f"已打开「{project.name}」",
            *self._preview_updates(project_id),
            CODE_TO_LANGUAGE.get(project.source_language, "自动检测"),
            CODE_TO_LANGUAGE.get(project.target_language, "中文"),
            project.whisper_model,
            project.translation_model,
        )

    def run_transcription(
        self,
        project_id: Optional[str],
        source_language: str,
        whisper_model: str,
        chunk_minutes: int,
        progress=gr.Progress(),
    ):
        if not project_id:
            raise gr.Error("请先创建或打开项目")
        self.store.update_project(
            project_id,
            source_language=LANGUAGE_OPTIONS.get(source_language, source_language),
            whisper_model=whisper_model,
        )
        flag = self._flag(project_id, "transcription")
        flag.clear()

        def update(current: int, total: int, message: str):
            progress(0 if not total else min(1, current / total), desc=message)

        self.pipeline.transcribe(
            project_id,
            chunk_seconds=max(1, int(chunk_minutes)) * 60,
            progress=update,
            should_stop=flag.is_set,
        )
        return self._summary(project_id), self._editor_rows(project_id), "转录任务已结束。若刚才暂停，可再次点击继续。"

    def pause_task(self, project_id: Optional[str], task: str):
        if not project_id:
            return "当前没有项目"
        self._flag(project_id, task).set()
        label = "转录" if task == "transcription" else "翻译"
        return f"已请求暂停{label}。当前处理块完成后会安全停下。"

    def save_subtitles(self, project_id: Optional[str], table_rows):
        if not project_id:
            raise gr.Error("请先创建或打开项目")
        rows = table_rows.values.tolist() if hasattr(table_rows, "values") else table_rows
        self.pipeline.save_editor_rows(project_id, rows or [])
        return self._summary(project_id), self._editor_rows(project_id), "字幕修改已保存"

    def run_translation(
        self,
        project_id: Optional[str],
        target_language: str,
        translation_model: str,
        progress=gr.Progress(),
    ):
        if not project_id:
            raise gr.Error("请先创建或打开项目")
        if not self.store.list_segments(project_id):
            raise gr.Error("还没有字幕，请先完成转录")
        self.store.update_project(
            project_id,
            target_language=LANGUAGE_OPTIONS.get(target_language, target_language),
            translation_model=translation_model,
        )
        flag = self._flag(project_id, "translation")
        flag.clear()

        def update(current: int, total: int, message: str):
            progress(0 if not total else min(1, current / total), desc=message)

        self.pipeline.translate(project_id, batch_size=20, progress=update, should_stop=flag.is_set)
        return self._summary(project_id), self._editor_rows(project_id), "翻译任务已结束"

    def export(
        self,
        project_id: Optional[str],
        subtitle_mode: str,
        output_kind: str,
        background_image: Optional[str],
        title: str,
        artist: str,
        album: str,
        font_size: int,
    ):
        if not project_id:
            raise gr.Error("请先创建或打开项目")
        result = self.pipeline.export_media(
            project_id,
            subtitle_mode={"双语": "bilingual", "原文": "source", "译文": "translation"}[subtitle_mode],
            output_kind={"SRT 字幕": "srt", "MP4 视频": "mp4", "MP3 音频": "mp3"}[output_kind],
            background_image=background_image,
            title=title or None,
            artist=artist or None,
            album=album or None,
            font_size=int(font_size),
        )
        return str(result), self._summary(project_id), f"导出完成：{result.name}"

    def save_api_keys(self, deepseek_key: str, gemini_key: str, qwen_key: str, claude_key: str, openai_key: str):
        values = {
            "deepseek": deepseek_key,
            "gemini": gemini_key,
            "qwen": qwen_key,
            "claude": claude_key,
            "gpt": openai_key,
        }
        for provider, value in values.items():
            if (value or "").strip():
                self.settings.set_secret(provider, value)
        configured = [name for name, ok in self.settings.configured_providers().items() if ok]
        return "API 设置已保存。已配置：" + (", ".join(configured) if configured else "无（仍可使用 Google Translate）")

    def _css(self) -> str:
        return """
        .gradio-container { max-width: 1380px !important; margin: 0 auto !important; }
        .hero { padding: 12px 4px 18px; }
        .hero h1 { font-size: 30px; margin: 0 0 6px; letter-spacing: -0.03em; }
        .hero p { margin: 0; opacity: .68; }
        .project-card { border: 1px solid rgba(120,120,120,.18); border-radius: 20px; padding: 20px; margin: 6px 0 14px; background: rgba(127,127,127,.045); }
        .project-card-top { display:flex; justify-content:space-between; gap:16px; align-items:flex-start; }
        .project-card h2 { margin: 3px 0 0; font-size: 22px; }
        .eyebrow { font-size: 12px; opacity: .55; text-transform: uppercase; letter-spacing: .08em; }
        .stage-pill { border-radius: 999px; padding: 7px 11px; background: rgba(80,120,255,.12); font-size: 13px; }
        .meta-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin-top:18px; }
        .meta-grid div { padding: 12px; border-radius: 14px; background: rgba(127,127,127,.06); min-width:0; }
        .meta-grid span { display:block; font-size:12px; opacity:.55; margin-bottom:4px; }
        .meta-grid strong { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .empty-state { padding:20px; border:1px dashed rgba(120,120,120,.3); border-radius:18px; opacity:.7; }
        .primary-action button { min-height: 48px !important; font-weight: 650 !important; }
        footer { display:none !important; }
        @media (max-width: 760px) { .meta-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
        """

    def _gradio_major(self) -> int:
        try:
            return int(str(gr.__version__).split(".", 1)[0])
        except Exception:
            return 6

    def build(self):
        block_kwargs = {"title": "OmniTranscribe"}
        if self._gradio_major() < 6:
            block_kwargs.update(theme=gr.themes.Soft(), css=self._css())
        with gr.Blocks(**block_kwargs) as app:
            project_state = gr.State(None)
            gr.HTML("<div class='hero'><h1>OmniTranscribe</h1><p>转录、校对、翻译，再导出。每一步都会自动保存。</p></div>")

            with gr.Row(equal_height=True):
                with gr.Column(scale=3):
                    recent_projects = gr.Dropdown(choices=self._project_choices(), label="最近项目", info="中断过的项目可以直接从这里继续")
                with gr.Column(scale=1, min_width=150):
                    open_btn = gr.Button("打开项目", size="lg")

            project_summary = gr.HTML(self._summary(None))
            status = gr.Textbox(label="状态", interactive=False, lines=2)

            with gr.Tabs():
                with gr.Tab("① 导入"):
                    with gr.Row():
                        media_file = gr.File(
                            label="拖入音频或视频",
                            file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".mov", ".mkv", ".avi", ".wmv", ".flv", ".webm"],
                            type="filepath",
                            scale=2,
                        )
                        with gr.Column(scale=1):
                            project_name = gr.Textbox(label="项目名称（可选）")
                            source_language = gr.Dropdown(choices=list(LANGUAGE_OPTIONS.keys()), value="自动检测", label="源语言")
                            target_language = gr.Dropdown(
                                choices=[name for name in LANGUAGE_OPTIONS if name != "自动检测"], value="中文", label="目标语言"
                            )
                    with gr.Accordion("高级设置", open=False):
                        with gr.Row():
                            whisper_model = gr.Dropdown(choices=["tiny", "base", "small", "medium", "large"], value="base", label="Whisper 模型")
                            translation_model = gr.Dropdown(choices=TRANSLATION_MODELS, value="google", label="翻译服务")
                    create_btn = gr.Button("创建项目", variant="primary", elem_classes=["primary-action"])

                with gr.Tab("② 转录"):
                    gr.Markdown("转录按时间块自动保存。关闭软件或暂停后，再次点击即可从未完成的块继续。")
                    chunk_minutes = gr.Slider(1, 15, value=5, step=1, label="Checkpoint 间隔（分钟）")
                    with gr.Row():
                        transcribe_btn = gr.Button("开始 / 继续转录", variant="primary")
                        pause_transcribe_btn = gr.Button("安全暂停")

                with gr.Tab("③ 字幕校对"):
                    gr.Markdown("可直接修改时间、原文和译文。修改原文后，已有 AI 译文会自动标记为需要重译；人工修改的译文不会被自动覆盖。")
                    with gr.Row():
                        video_preview = gr.Video(label="视频预览", visible=False)
                        audio_preview = gr.Audio(label="音频预览", visible=False)
                    dataframe_kwargs = {
                        "headers": ["#", "开始", "结束", "原文", "译文", "状态"],
                        "datatype": ["number", "str", "str", "str", "str", "str"],
                        "interactive": True,
                        "wrap": True,
                        "row_count": 8,
                        "label": "字幕工作台",
                    }
                    parameters = inspect.signature(gr.Dataframe).parameters
                    if "column_count" in parameters:
                        dataframe_kwargs["column_count"] = 6
                    else:
                        dataframe_kwargs["col_count"] = (6, "fixed")
                    editor = gr.Dataframe(**dataframe_kwargs)
                    save_editor_btn = gr.Button("保存字幕修改", variant="primary")

                with gr.Tab("④ 翻译"):
                    gr.Markdown("只翻译尚未翻译或因原文校对而失效的字幕；已经人工校对的译文默认跳过。")
                    with gr.Row():
                        translate_btn = gr.Button("开始 / 继续翻译", variant="primary")
                        pause_translate_btn = gr.Button("安全暂停")

                with gr.Tab("⑤ 导出"):
                    with gr.Row():
                        subtitle_mode = gr.Radio(["双语", "原文", "译文"], value="双语", label="字幕内容")
                        output_kind = gr.Radio(["SRT 字幕", "MP4 视频", "MP3 音频"], value="SRT 字幕", label="输出")
                    with gr.Accordion("视频 / 音频外观与元数据", open=False):
                        background_image = gr.Image(label="背景图 / MP3 封面（音频转 MP4 时需要）", type="filepath")
                        font_size = gr.Slider(16, 64, value=24, step=1, label="视频字幕字号")
                        with gr.Row():
                            title = gr.Textbox(label="标题")
                            artist = gr.Textbox(label="艺术家")
                            album = gr.Textbox(label="专辑")
                    export_btn = gr.Button("导出", variant="primary")
                    exported_file = gr.File(label="导出文件")

                with gr.Tab("⚙ 设置"):
                    gr.Markdown("Google Translate 无需配置。只有选择其他翻译服务时才需要 API Key；密钥优先保存在系统凭据存储中。留空表示保留现有设置。")
                    with gr.Row():
                        deepseek_key = gr.Textbox(label="DeepSeek API Key", type="password")
                        gemini_key = gr.Textbox(label="Gemini API Key", type="password")
                    with gr.Row():
                        qwen_key = gr.Textbox(label="Qwen API Key", type="password")
                        claude_key = gr.Textbox(label="Claude API Key", type="password")
                    openai_key = gr.Textbox(label="OpenAI API Key", type="password")
                    save_api_btn = gr.Button("保存 API 设置", variant="primary")

            create_btn.click(
                self.create_project,
                [media_file, project_name, source_language, target_language, whisper_model, translation_model],
                [project_state, recent_projects, project_summary, editor, status, video_preview, audio_preview],
            )
            open_btn.click(
                self.load_project,
                [recent_projects],
                [project_state, project_summary, editor, status, video_preview, audio_preview, source_language, target_language, whisper_model, translation_model],
            )
            transcribe_btn.click(
                self.run_transcription,
                [project_state, source_language, whisper_model, chunk_minutes],
                [project_summary, editor, status],
            )
            pause_transcribe_btn.click(lambda project_id: self.pause_task(project_id, "transcription"), [project_state], [status])
            save_editor_btn.click(self.save_subtitles, [project_state, editor], [project_summary, editor, status])
            translate_btn.click(
                self.run_translation,
                [project_state, target_language, translation_model],
                [project_summary, editor, status],
            )
            pause_translate_btn.click(lambda project_id: self.pause_task(project_id, "translation"), [project_state], [status])
            export_btn.click(
                self.export,
                [project_state, subtitle_mode, output_kind, background_image, title, artist, album, font_size],
                [exported_file, project_summary, status],
            )
            save_api_btn.click(
                self.save_api_keys,
                [deepseek_key, gemini_key, qwen_key, claude_key, openai_key],
                [status],
            )
        return app

    def _launch_options(self):
        options = {"show_error": True}
        if self._gradio_major() >= 6:
            options.update(theme=gr.themes.Soft(), css=self._css())
        return options

    def launch(self, share: bool = False):
        app = self.build()
        app.queue(default_concurrency_limit=2).launch(share=share, inbrowser=True, **self._launch_options())

    def launch_desktop(self):
        try:
            import webview
        except ImportError:
            return self.launch(share=False)

        app = self.build()
        _, local_url, _ = app.queue(default_concurrency_limit=2).launch(
            share=False,
            inbrowser=False,
            prevent_thread_lock=True,
            server_name="127.0.0.1",
            **self._launch_options(),
        )
        webview.create_window("OmniTranscribe", local_url, width=1320, height=860, min_size=(900, 620))
        try:
            webview.start()
        finally:
            app.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OmniTranscribe product GUI")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    ProductGUI().launch(share=args.share)


if __name__ == "__main__":
    main()
