# Bundled Fonts

This repository redistributes two fonts from the **ChillType / 寒蝉** series by
Warren2060. Both are dual-licensed to us under the [SIL Open Font License 1.1](./OFL-1.1.txt),
which permits free use, embedding, and redistribution (including commercially)
as long as the license text travels along and the fonts stay under OFL.

| File | Font | Upstream project | Copyright / Reserved Font Name |
|---|---|---|---|
| `ChillDuanSansVF.ttf` | 寒蝉端黑体 ChillDuanSans | https://github.com/Warren2060/ChillDuanSans | © 2023 ChillType, RFN "Duan" (derived from Source Han Sans / 思源黑体 and Figtree) |
| `ChillHuoFangSong_Regular.otf` | 寒蝉活仿宋 ChillHuoFangSong | https://github.com/Warren2060/ChillMovableType | © 2023 ChillType, RFN "ChillMovableType" (derived from 朱雀仿宋 Zhuque Fangsong and Simonetta) |

Notes:

- The fonts are used as default subtitle faces for generated MP4 videos and are
  bundled into the desktop build (`packaging/OmniTranscribe.spec`).
- If you modify either font, the derivative must also stay under OFL 1.1 and
  must not use the Reserved Font Names above (see `OFL-1.1.txt`, conditions 3 & 5).
- Subtitle rendering falls back to system CJK fonts when these files are absent
  (`src/final_video_generator.py`).
