# LDA

## 环境配置

环境在终端中运行

```
pip install -r requirements.txt(修改为对应路径)
```



## html阅读

![img](https://github.com/wztsir/LDA/blob/main/data/imgs/html.png?raw=true)

在 pyLDAvis 主题可视化中，"Overall term frequency"（整体词频）和"Estimated term frequency within the selected topic"（在选定主题中的估计词频）是两个用于衡量主题关键词的重要性的指标。

### Overall Term Frequency（整体词频）：

含义： 衡量了语料库中所有文档中某个词语的总出现频率。
作用： 该值越高，表示该词在整个语料库中的使用越频繁。在主题模型中，这有助于确定哪些词在整个语料库中具有重要性。

### Estimated Term Frequency within the Selected Topic（在选定主题中的估计词频）：

含义： 衡量了某个词在选定的主题中的相对重要性。它是根据 LDA 模型中的概率分布计算得出的。
作用： 该值越高，表示该词在选定的主题中更为重要。在理解主题关键词时，这个指标可以帮助确定哪些词在给定主题中是核心的、有代表性的。

### Slide to adjust relevance metric(滑块)

在 pyLDAvis 中，通过滑块（Slide to adjust relevance metric）来调整关键词的相关性度量（relevance metric）。这个滑块控制了公式中的 λ 参数，即权衡两部分的相对重要性。

在具体的影响上，调整这个滑块可以改变可视化结果中关键词的显示方式。具体来说，关键词的大小和颜色是由相关性度量计算得到的，因此改变 λ 参数会调整关键词在可视化中的重要性。

当 λ 接近 0 时，关键词的显示受到主题中的概率分布较大的影响。
当 λ 接近 1 时，关键词的显示受到在整个语料库中的概率分布的影响更大。

