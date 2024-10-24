---
comments: true
title: 自定义布局
---
![layout](imgs/layout.jpg)

## 前言

在 `Gradio` 开发中，提前固定好布局可以帮助开发者更好地组织界面，提高开发效率，并确保最终应用的视觉效果。

以下是几种常用的布局方式。

1. 使用 `gradio.Interface` 构建基础布局
2. {==使用 `gradio.Blocks` 构建更复杂的布局==}
3. 使用样式表自定义布局

!!! question "布局选择"
	对于开发者来说，方法 1 往往不能满足日常使用场景，因为这种方法过于简单，不如方法 2 灵活，同时方法 3 又过于复杂，最适合大众开发者的就是方法 2，该方法灵活易用，可构建复杂布局，如果需要构建更复杂的布局，可以将方法 2 和方法 3 结合。

## `Row`（行布局）

默认情况下，`Blocks` 中的组件是垂直排列的。让我们开始用 `Rows` 重新排列组件。

当开发者希望多个组件并排排列，只需在 `with gradio.Row()` 下文中创建组件（各种组件将在后续一一说明）即可  (1)  。
{ .annotate }

1. 代码解读
	1. `with gr.Blocks() as demo:`：创建了一个 Gradio 应用程序，并将其赋值给 `demo`。
	2. `with gr.Row():`：创建了一个水平排列的容器，所有在这个容器中创建的元素都会被水平排列。
	3. `img = gr.Image()`：创建了一个图片组件，并将其赋值给 `img`。
	4. `audio = gr.Audio()`：创建了另一个图片组件，并将其赋值给 `audio`。
	5. `demo.launch()`：启动 Gradio 应用程序。

```python hl_lines="4"
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        img = gr.Image()
        audio = gr.Audio()
demo.launch()
```

![row](imgs/row.png)

下面是 `gradio.Row` 参数对照表：

| 参数名             | 类型                                         | 默认值         | 描述                                                                   |
| --------------- | ------------------------------------------ | ----------- | -------------------------------------------------------------------- |
| `variant`       | `Literal[('default', 'panel', 'compact')]` | `"default"` | 行类型，`'default'` (无背景)，`'panel'` (灰色背景和圆角)，或者 `'compact'` (圆角和无内部间距)。 |
| `visible`       | `bool`                                     | `True`      | 如果为 `False`，则该行将被隐藏。                                                 |
| `equal_height`  | `bool`                                     | `True`      | 如果为 `True`，则所有子元素的高度一致。                                              |
| `show_progress` | `bool`                                     | `False`     | 如果为 `True`，则在更新时显示进度动画。                                              |
| `elem_id`       | `str`                                      | `None`      | 可选的字符串，作为该组件在 HTML DOM 中的 id。可用于定位 CSS 样式。                           |
| `elem_classes`  | `list[str]`                                | `None`      | 可选的字符串或字符串列表，作为该组件在 HTML DOM 中的类。可用于定位 CSS 样式。                       |

下面是一些参数变化对照图：

=== "visible"

	```python hl_lines="4" title="visible"
	import gradio as gr

	with gr.Blocks() as demo:
	    with gr.Row(visible=False):
	        img = gr.Image()
	        audio = gr.Audio()
	demo.launch()
	```

	![visible=False](imgs/visible.png)

=== "equal_height"

	```python hl_lines="4" title="equal_height"
	import gradio as gr

	with gr.Blocks() as demo:
	    with gr.Row(equal_height=False):
	        img = gr.Image()
	        audio = gr.Audio()
	demo.launch()
	```

	![equal_height](imgs/equal_height.png)

=== "show_progress"

	```python hl_lines="4" title="show_progress"
	import gradio as gr

	with gr.Blocks () as demo:
	    with gr.Row (show_progress=True):
	        img = gr.Image ()
	        audio = gr.Audio ()
	demo.launch ()
	```

	![show_progress](imgs/show_progress.png)

## `Column`（列布局）

与 `Row` 类似，`Column` 用于垂直排列组件。

当开发者希望多个组件上下排列，只需在 `with gradio.Column ()` 下文中创建组件即可。

```python hl_lines="4"
import gradio as gr

with gr.Blocks() as demo:
    with gr.Column():
        img = gr.Image()
        audio = gr.Audio()
demo.launch()
```

![coulmn](imgs/coulmn.png)

下面是 `gradio.Column` 参数对照表：

| 参数名             | 类型                                         | 默认值         | 描述                                                                                   |
| --------------- | ------------------------------------------ | ----------- | ------------------------------------------------------------------------------------ |
| `scale`         | `int`                                      | `1`         | 相对于相邻列的相对宽度。例如，如果列 A 的 `scale` 为 2，而列 B 的 `scale` 为 1，则 A 的宽度将是 B 的两倍。               |
| `min_width`     | `int`                                      | `320`       | 列的最小像素宽度，如果屏幕空间不足以满足此值，则会换行。如果某个 `scale` 值导致列宽度小于 `min_width`，则会优先考虑 `min_width` 参数。 |
| `variant`       | `Literal[('default', 'panel', 'compact')]` | `"default"` | 列类型，`'default'` (无背景)，`'panel'` (灰色背景和圆角)，或者 `'compact'` (圆角和无内部间距)。                 |
| `visible`       | `bool`                                     | `True`      | 如果为 `False`，则该列将被隐藏。                                                                 |
| `show_progress` | `bool`                                     | `False`     | 如果为 `True`，则在更新时显示进度动画。                                                              |
| `render`        | `bool`                                     | `True`      | 如果为 `False`，则该组件不会在 `Blocks` 上下文中渲染。如果需要现在分配事件监听器，但稍后渲染组件，则应使用此参数。                   |
| `elem_id`       | `str`                                      | `None`      | 可选的字符串，作为该组件在 HTML DOM 中的 id。可用于定位 CSS 样式。                                           |
| `elem_classes`  | `list[str]`                                | `None`      | 可选的字符串或字符串列表，作为该组件在 HTML DOM 中的类。可用于定位 CSS 样式。                                       |

## `Nesting`（镶嵌布局）

在 `Gradio` 中，`Column` 布局默认情况下会将内部组件垂直排列，而这恰好是 `Blocks` 应用程序的默认布局方式。因此，为了使 `Column` 组件发挥作用，通常需要将其嵌套在 `Row` 组件中，以实现水平排列。

假设要创建下面示例的布局，或是更复杂的布局都可以使用 `Row` 与 `Column` 的嵌套来实现。

```python
import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        img1 = gr.Image()
        img2 = gr.Image()
    with gr.Row():
        img3 = gr.Image()
        with gr.Column():
            with gr.Column():
                btn1 = gr.Button()
                btn2 = gr.Button()
                btn3 = gr.Button()
                btn4 = gr.Button()
demo.launch()
```

![nesting](imgs/nesting.png)

## 参考资料

<div class="grid cards" markdown>

- `Gradio` 文档（布局部分）

    ---

    [用 `Blocks` 建立布局](https://www.gradio.app/guides/controlling-layout)

- flexbox 的基本概念

	---
	[布局介绍](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_flexible_box_layout/Basic_concepts_of_flexbox)
</div>