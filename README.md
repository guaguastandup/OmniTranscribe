# OmniTranscribe

<div align="center">

<img src="./omnitranscribe.png" alt="OmniTranscribe" width="400"/>

**强大的多语言音频转录与翻译工具**

[English](./README_EN.md) | 简体中文（默认）

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Whisper](https://img.shields.io/badge/Whisper-OpenAI-purple)](https://github.com/openai/whisper)

[功能特性](#功能特性) • [安装](#安装) • [使用方法](#使用方法) • [文档](#文档) • [贡献](#贡献)

</div>

---

## 项目简介

OmniTranscribe 是一款功能强大的多语言音频视频处理工具，支持 99+ 种语言的自动转录和翻译。结合 OpenAI Whisper 的语音识别技术与多种 AI 翻译模型，为您提供完整的多语言本地化解决方案。

### 工作流程

```
输入音频/视频  →  转录  →   翻译   →       输出
 [任意语言]     [SRT]   [任意语言]  [SRT/VTT/LRC/MP4]
```

## 功能特性

### 核心功能

- **多语言转录**：使用 OpenAI Whisper 支持 99+ 种语言的语音识别
- **灵活翻译**：支持多种 AI 翻译服务，可翻译至任意语言
- **免费翻译**：内置 Google Translate 支持，无需 API 密钥
- **格式转换**：支持 SRT、VTT、LRC 字幕格式互转
- **视频生成**：生成带同步字幕的 MP4 视频（默认背景图）
- **智能缓存**：基于文件哈希的缓存机制，避免重复转录
- **批量处理**：支持批量处理多个文件
- **交互模式**：友好的命令行交互界面
- **响应式 GUI**：自适应屏幕尺寸的 Web 界面

### 支持的语言

Whisper 支持 99+ 种语言，包括：

```
中文、英语、日语、韩语、西班牙语、法语、德语、俄语、阿拉伯语、印地语、
葡萄牙语、意大利语、荷兰语、波兰语、土耳其语、越南语、泰语、瑞典语、
以及更多...
```

### 翻译服务

- **Google Translate** ⭐ 免费无需 API 密钥（使用 deep-translator）
- **DeepSeek**（推荐）- 性价比高
- **Google Gemini** - 高质量翻译
- **阿里通义千问** - 中文优化
- **Anthropic Claude** - 先进推理
- **OpenAI GPT** - 行业标准
- **自定义 API** - 支持 OpenAI 兼容接口

### 支持的媒体格式

- **音频**：MP3、WAV、M4A、FLAC、OGG
- **视频**：MP4、AVI、MOV、MKV、WMV、FLV、WEBM

### 自定义字体
- 用于视频字幕生成

## 安装

### 前置要求

1. **Python 3.8 或更高版本**
2. **FFmpeg** - 媒体处理必需

#### 安装 FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载或使用：
```bash
choco install ffmpeg
```

### 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/guaguastandup/OmniTranscribe.git
cd OmniTranscribe
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置 API 密钥：
```bash
cp .env.example .env
# 编辑 .env 文件，添加您的 API 密钥
```

4. 验证安装：
```bash
python run.py --help
```

## 使用方法

### 图形界面模式（GUI）⭐ 推荐

```bash
# 启动图形界面
python run.py --gui

# 创建公共链接（可远程访问）
python run.py --gui --share
```

GUI 界面提供：
- 📁 拖拽上传音频/视频文件
- 🌍 可视化语言选择（支持 99+ 种语言）
- ⚙️ 模型和设备设置
- 📤 输出格式选择（仅字幕 / MP4 视频 / MP3 音频）
- 🖼️ 背景图片和封面上传
- 🎵 作者信息编辑（标题、艺术家、专辑）
- 📊 实时处理进度显示

### 交互模式（命令行）

```bash
python run.py
```

按照提示配置设置并处理文件。

### 基本用法（命令行）

```bash
# 使用智能默认设置处理音频文件
python run.py audio_file.mp3

# 处理视频文件
python run.py video_file.mp4

# 指定源语言
python run.py audio_file.mp3 --language ja

# 指定源语言和目标语言
python run.py audio_file.mp3 --language en --target-language zh
```

### 语言代码

常用语言代码：
- `auto` - 自动检测
- `zh` - 中文
- `en` - 英语
- `ja` - 日语
- `ko` - 韩语
- `es` - 西班牙语
- `fr` - 法语
- `de` - 德语
- `ru` - 俄语

### 使用示例

#### 多语言转录

```bash
# 转录日语音频
python run.py japanese_audio.mp3 --language ja

# 转录西班牙语视频
python run.py spanish_video.mp4 --language es

# 自动检测语言
python run.py audio.mp3 --language auto

# 使用特定 Whisper 模型
python run.py audio.mp3 --model medium --language en
```

#### 翻译选项

```bash
# 翻译为中文（默认）
python run.py audio.mp3 --language en

# 翻译为英语
python run.py audio.mp3 --language ja --target-language en

# 使用不同的翻译服务
python run.py audio.mp3 --translation-model gemini
python run.py audio.mp3 --translation-model qwen
python run.py audio.mp3 --translation-model claude
python run.py audio.mp3 --translation-model gpt

# 自定义 API 端点
python run.py audio.mp3 --translation-model custom \
  --translation-url "https://your-api.com/v1" \
  --translation-api-key "your-key"
```

#### GPU 加速

```bash
# 使用 GPU 加速（如果有）
python run.py audio.mp3 --device cuda --language en

# Apple Silicon (M1/M2/M3) 加速
python run.py audio.mp3 --device mps --language ja
```

#### 字幕格式转换

```bash
# 转换 SRT 为 VTT 或 LRC
python run.py audio.mp3 --convert-to vtt
python run.py audio.mp3 --convert-to lrc

# 仅转换现有字幕文件
python run.py --convert-only input.srt --convert-to vtt
```

#### 视频生成

```bash
# 生成带字幕的 MP4 视频
python run.py audio.mp3 --generate-video --background-image image.jpg

# 指定字幕位置
python run.py audio.mp3 --generate-video --background-image image.jpg --subtitle-position top
```

#### 批量处理

```bash
# 批量处理目录中的所有文件
python run.py --batch /path/to/media/files --language auto

# 递归处理子目录
python run.py --batch /path/to/files --recursive --language auto

# 删除原始视频文件以节省空间
python run.py --batch /path/to/videos --delete-video-files
```

#### 预设模式

```bash
# 快速处理（tiny 模型）
python run.py --fast audio_file.mp3

# 高质量（large 模型）
python run.py --quality audio_file.mp3

# GPU 加速
python run.py --gpu audio_file.mp3
```

#### 缓存管理

```bash
# 查看缓存统计
python run.py --cache-stats

# 清除所有缓存
python run.py --clear-cache

# 禁用缓存（强制重新转录）
python run.py audio.mp3 --no-cache
```

#### 仅转录不翻译

```bash
# 只转录，不翻译
python run.py audio.mp3 --target-language none

# GUI 模式：选择"不翻译"选项
python run.py --gui
```

## 配置

### 环境变量 (.env)

```bash
# DeepSeek API（推荐）
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_MODEL=deepseek-chat

# Google Gemini
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.5-flash

# 阿里通义千问
QWEN_API_KEY=sk-your-qwen-key
QWEN_MODEL=qwen-plus

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-your-claude-key
CLAUDE_MODEL=claude-3-5-sonnet-20241022

# OpenAI GPT
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini

# 默认服务
TRANSLATION_MODEL=deepseek

# Whisper 默认设置
WHISPER_MODEL=medium
WHISPER_LANGUAGE=auto
WHISPER_TARGET_LANGUAGE=zh
```

## 项目结构

```
OmniTranscribe/
├── run.py                     # 启动脚本
├── requirements.txt           # 依赖列表
├── .env.example               # 环境变量模板
├── README.md                  # 中文文档
├── README_EN.md               # 英文文档
├── LICENSE                    # MIT 许可证
├── CHANGELOG.md               # 变更日志
├── CONTRIBUTING.md            # 贡献指南
├── prompt.md                  # 翻译提示词
├── assets/                    # 默认资源
│   ├── default_background.png  # 默认视频背景
│   └── generate_default_bg.py  # 背景生成脚本
└── omnitranscribe/           # 源代码包（pip install . 可安装）
    ├── __init__.py
    ├── main.py                # 主入口（omnitranscribe 命令）
    ├── transcribe.py          # 音频转录
    ├── translator.py          # AI 翻译（含 Google Translate）
    ├── converter.py           # 字幕转换
    ├── video_converter.py     # 视频转音频
    ├── final_video_generator.py   # 视频生成
    ├── simple_mp3_embedder.py     # MP3 元数据
    ├── batch_processor.py     # 批量处理
    ├── cache.py               # 缓存管理
    ├── interactive.py         # 交互界面
    ├── config.py              # 配置管理
    └── gui.py                 # Web 图形界面
```

## 常见问题

### FFmpeg 未找到
```
Error: ffmpeg is not installed or not in PATH
```
**解决方案**：安装 FFmpeg（见[安装](#安装)部分）

### API 密钥错误
```
Error: DEEPSEEK_API_KEY not found in environment variables
```
**解决方案**：复制 `.env.example` 为 `.env` 并添加您的 API 密钥

### CUDA 内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：使用更小的 Whisper 模型或切换到 CPU：`--device cpu`

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 多语言语音识别
- [FFmpeg](https://ffmpeg.org/) - 媒体处理
- 所有翻译服务提供商（DeepSeek、Google、阿里、Anthropic、OpenAI）
- [寒蝉字体系列](https://github.com/Warren2060)（ChillDuanSans / ChillHuoFangSong，[OFL 1.1](./FONTS.md)）- 内置字幕字体

---

<div align="center">

用 ❤️ 为多语言音频内容爱好者打造

[![GitHub stars](https://img.shields.io/github/stars/guaguastandup/OmniTranscribe?style=social)](https://github.com/guaguastandup/OmniTranscribe/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/guaguastandup/OmniTranscribe?style=social)](https://github.com/guaguastandup/OmniTranscribe/network/members)

</div>
