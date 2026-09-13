# Paper

`latex/` 包含 FCS 论文源码、期刊样式、参考文献和正文插图。LaTeX
编译产物由根目录的 `.gitignore` 统一忽略，不纳入版本控制。

## 编译

```bash
cd paper/latex
latexmk -pdf main.tex
```

生成的论文位于 `paper/latex/main.pdf`。

## 清理编译产物

```bash
cd paper/latex
latexmk -C main.tex
```


实验分为五个章节

- 实验介绍 shape 硬件
- 端到端性能
- 精度对比，（baseline，其他方法，体现我们的优势）
- abaltion （）
- overhead
