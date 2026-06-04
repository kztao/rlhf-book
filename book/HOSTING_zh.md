# RLHF Book - 托管/部署

本节记录 rlhfbook.com 的托管方式。对于本地构建，请参阅 `book/README_zh.md`。

## 快速参考

| 项目 | 详情 |
|------|-------|
| 站点 | [rlhfbook.com](https://rlhfbook.com) |
| 仓库 | [github.com/natolambert/rlhf-book](https://github.com/natolambert/rlhf-book) |
| 构建 | Pandoc + Make（HTML、PDF、EPUB、DOCX） |
| 部署 | GitHub Pages（`gh-pages` 分支，GitHub Actions） |
| CI | `.github/workflows/static.yml`（推送时自动构建 + 部署） |

## 构建与部署流程

1. **本地**: `make html` 将 `book/` 源文件 + 模板渲染到 `build/html/`
2. **CI**：在工作流运行器中运行 `make html`（macOS，pandoc）
3. **部署**：将 `build/html/` 推送到 `gh-pages` 分支，提供 rlhfbook.com

所有格式都通过 `Makefile` 构建：
```bash
make          # 构建所有格式
make html     # 站点 → build/html/
```

## 站点架构

- `/` → `build/html/index.html`
- `/c/01-introduction.html` → `build/html/c/01-introduction.html`
- `/library` → `build/html/library.html`
- `/assets/...` → 静态资源

构建运行 Pandoc 章节模板（`chapter.html`），该模板包含来自 `book/templates/` 的 `header.html` 和 `footer.html`，并生成每个章节的独立页面。

索引、404、课程、库和预购页面是通过 `make files` 目标复制或内联模板化的。
