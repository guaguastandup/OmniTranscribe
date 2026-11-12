# TODO / 进度

## 已完成

- ✅ 产品化断点续传管线（SQLite 项目库 + 分块转录/翻译检查点，`src/resumable_pipeline.py`）
- ✅ 新版项目式 GUI（导入 → 转录 → 校对 → 翻译 → 导出，`run.py --gui` / `--desktop`）
- ✅ 桌面打包（PyInstaller + Inno Setup / DMG，`.github/workflows/desktop-release.yml`）
- ✅ 2026-11 修复轮：`--language auto` 崩溃、批量模式强制日语、Claude/Gemini 兼容端点、
  `--fast/--quality/--gpu` 预设失效、`--subtitle-fontsize/-position` 无效、
  `.env` 的 WHISPER_*/TRANSLATION_MODEL 变量不生效等
- ✅ 健壮性/性能：Whisper 模型进程内缓存、翻译请求超时、eval()/mktemp()/裸 except 清理、
  缓存哈希单次计算
- ✅ GUI 视觉重做（渐变 Hero、语义化阶段胶囊、进度条卡片）
- ✅ ruff lint 接入并清零（`ruff.toml`）
- ✅ 单元测试扩充至 32 个（converter / cache / config / translator 纯函数 / project_store / subtitle_utils）
- ✅ CI：lint + 全量单测 + 全模块编译（`.github/workflows/product-tests.yml`）
- ✅ 字体合规：OFL 1.1 许可文本与来源说明（`FONTS.md`、`OFL-1.1.txt`）
- ✅ README 横幅压缩（5.5MB → 67KB）

## 待办

- [ ] **双管线收敛**：legacy CLI/GUI（`main.py`、`batch_processor.py`、`gui.py`）与
      resumable pipeline 存在约 300 行重复流程，应将 legacy 改为其薄封装
- [ ] **打包正规化**：`pyproject.toml` + 包内相对导入 + console_scripts 入口，支持 pip install
- [ ] **仓库瘦身**：两个字体共 35MB，考虑 Git LFS 或按需下载（历史改写需慎重）
- [ ] **真实端到端验证**：装齐 openai-whisper 后用真实音视频跑通 CLI/GUI 两条路径
- [ ] GUI 语言列表仅 18 种，与"99+ 语言"宣传不符，可开放自由输入
- [ ] 视频生成两遍编码合并为单遍（`-vf` 链式滤镜），导出耗时约可减半
