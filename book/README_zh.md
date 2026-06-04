# RLHF Book - 构建系统

基于 [**Pandoc 书籍模板**](https://github.com/wikiti/pandoc-book-template)构建。

此目录包含用于以多种格式（HTML、PDF、EPUB、DOCX）构建 RLHF Book 的源文件。

## 用法

从仓库根目录：

```bash
make          # 构建所有格式
make html     # 构建 HTML 站点
make pdf      # 构建 PDF（需要 LaTeX）
make epub     # 构建 EPUB
make files    # 复制资源到构建输出
```

### 已知的转换问题

由于网站使用的嵌套结构，章节之间的 PDF 内部链接已损坏。
我们选择这样做是为了更好的网页体验，但最佳实践是不要在 Markdown 文件中放置任何指向 `rlhfbook.com` 的链接。非 HTML 版本将不适合使用它们。

### 使用编程代理编辑时的常见失败

编程代理（Claude、Cursor 等）经常引入 Unicode 字符，导致 Pandoc PDF 构建失败，错误如 `Cannot decode byte '\xe2'`。注意：

- **弯引号**（`'` U+2019）代替直引号（`'`）
- **全角破折号**（`—` U+2014）代替双连字符（`--`）
- **非断空格**（`\xa0` U+00A0）代替普通空格
- **弯双引号**（`"` `"` U+201C/U+201D）代替直双引号（`"`）

## 安装

### Linux

```sh
sudo apt-get install pandoc make
sudo apt-get install texlive-fonts-recommended texlive-xetex
```

### Mac
```
brew install pandoc make pandoc-crossref
```

小型 TeX 安装：
```sh
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install fvextra tcolorbox pdfcol
```

## 文件夹结构

```
book/
├── chapters/     # Markdown 源文件（每章一个）
├── images/       # 章节中引用的图片资源
├── assets/       # 品牌资源（封面、标志）
├── templates/    # 每种输出格式的 Pandoc 模板
├── scripts/      # 构建工具
├── data/         # 库数据（JSON）
└── preorder/     # 订单重定向页面
```

## 创建章节

创建新章节只需在 *chapters/* 文件夹中创建一个新的 markdown 文件：

```
chapters/01-introduction.md
chapters/02-installation.md
chapters/03-usage.md
chapters/04-references.md
```

Pandoc 和 Make 将按名称自动排序连接它们；这就是使用数字前缀的原因。

## 插入对象

### 插入图片

```md
![一只酷海鸥。](images/seagull.png)
```

调整大小：`![一只酷海鸥。](images/seagull.png){ width=50% height=50% }`

### 插入表格

```md
| 索引 | 名称 |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |

Table: 这是一个示例表格。
```

### 插入公式

行内公式：`$\mu = \sum_{i=0}^{N} \frac{x_i}{N}$`

居中公式：`$$\mu = \sum_{i=0}^{N} \frac{x_i}{N}$$`

### 交叉引用

使用 [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref)：

```md
- 查看 @fig:seagull。
- 查看 @tbl:table。
- 查看 @eq:equation。

![一只酷海鸥](images/seagull.png){#fig:seagull}

$$ y = mx + b $$ {#eq:equation}

| 索引 | 名称 |
| ----- | ---- |
| 0     | AAA  |

Table: 示例表格。 {#tbl:table}
```

## 输出

```sh
make pdf    # 导出为 PDF（输出到 build/pdf）
make epub   # 导出为 EPUB（输出到 build/epub）
make html   # 导出为 HTML（输出到 build/html）
make docx   # 导出为 DOCX（输出到 build/docx）
```

## 参考文献

- [Pandoc](http://pandoc.org/)
- [Pandoc 手册](http://pandoc.org/MANUAL.html)
- [维基百科：Markdown](http://wikipedia.org/wiki/Markdown)
