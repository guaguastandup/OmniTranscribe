# 介绍
这是一个mp3/mp4转录+生成翻译后字幕+生成带字幕的mp3/mp4功能

# 已完成的功能
1. ✅ 优化GUI显示(当调整浏览器大小时，界面的模块要自适应)
2. ✅ 提供默认的视频背景
3. ✅ 提供默认字体
4. ✅ 提供默认的谷歌翻译(不需要配置LLM API)
5. ✅ 提供不翻译，仅转录的选项
6. ✅ 增加使用缓存的功能(如果某个视频的hash值+选择的whisper模型大小符合历史缓存，那么直接使用历史的字幕(如果有那个文件的话))
7. ✅ 优化README.md和README_en.md

# 新功能总结

## 1. 响应式GUI
- 使用Gradio的响应式布局特性
- 界面在调整浏览器大小时自动适应
- 移动端友好的界面设计

## 2. 默认视频背景
- 自动生成默认背景图片 (assets/default_background.png)
- GUI中如果没有上传背景图，自动使用默认背景
- 美观的渐变背景设计

## 3. 默认字体配置
- 自动检测并使用项目内置字体
- ChillDuanSansVF.ttf 作为首选默认字体
- 跨平台字体回退机制

## 4. Google Translate 支持
- 新增 Google Translate 翻译选项
- 无需 API 密钥，开箱即用
- 使用 `--translation-model google` 或在 GUI 中选择

## 5. 仅转录不翻译选项
- CLI: `--target-language none`
- GUI: 目标语言下拉菜单中选择"不翻译"
- 跳过翻译步骤，直接输出原语言字幕

## 6. 智能缓存系统
- 基于文件 SHA256 哈希 + 模型大小 + 语言的缓存键
- 自动缓存转录结果，避免重复处理
- 缓存管理命令:
  - `--cache-stats`: 查看缓存统计
  - `--clear-cache`: 清除所有缓存
  - `--no-cache`: 禁用缓存强制重新转录

# 使用示例

## 使用 Google 翻译（免费）
```bash
python run.py audio.mp3 --translation-model google
```

## 查看缓存状态
```bash
python run.py --cache-stats
```

## 仅转录不翻译
```bash
python run.py audio.mp3 --target-language none
```