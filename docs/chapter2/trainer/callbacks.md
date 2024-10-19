---
comments: true
title: 回调函数
---

## 前言

回调函数是用于在训练过程中**查看（read only）**训练器行为的工具。它允许开发者在训练的不同阶段执行自定义操作，在这里，开发者往往搭配`Tensorboard`进行记录指标、可视化数据、调整超参数或进行早停等行为。

## 回调方法

HuggingFace社区有很多种回调方法，这里主要介绍[`TrainerCallback`](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/callback#transformers.TrainerCallback)。

在自定义回调函数的时候：

1. 继承`TrainerCallback`类；
2. 根据实际需求，重载下方第一个表内对应的方法；
3. 重载的方法的形参应该为下方第二个表内的参数名称；
4. 编写具体的逻辑。

<div class="grid" markdown>

| 方法名称                | 触发时间                       |
| ----------------------- | ------------------------------ |
| `on_train_begin`        | 训练开始前                     |
| `on_train_end`          | 训练结束后                     |
| `on_epoch_begin`        | 每个 epoch 开始前              |
| `on_epoch_end`          | 每个 epoch 结束后              |
| `on_step_begin`         | 每个训练步骤开始前             |
| `on_step_end`           | 每个训练步骤结束后             |
| `on_substep_end`        | 梯度累积过程中每个子步骤结束后 |
| `on_optimizer_step`     | 优化器步骤完成后，梯度清零前   |
| `on_pre_optimizer_step` | 优化器步骤前，梯度裁剪后       |
| `on_save`               | 保存检查点后                   |
| `on_log`                | 最后一次日志记录后             |
| `on_evaluate`           | 评估阶段结束后                 |
| `on_predict`            | 成功预测后                     |
| `on_prediction_step`    | 预测步骤完成后                 |
| `on_init_end`           | Trainer 初始化结束后           |

| 参数名称           | 描述                                    |
| ------------------ | --------------------------------------- |
| `args`             | `TrainingArguments` 对象                |
| `state`            | `TrainerState` 对象，包含训练状态信息   |
| `control`          | `TrainerControl` 对象，用于控制训练流程 |
| `model`            | 训练阶段的模型                          |
| `tokenizer`        | 编码数据的分词器                        |
| `optimizer`        | 训练阶段使用的优化器                    |
| `lr_scheduler`     | 训练阶段使用的学习率调度器              |
| `train_dataloader` | 训练阶段的训练集生成器                  |
| `eval_dataloader`  | 验证阶段的验证集生成器                  |
| `metrics`          | 上个阶段产生的评估指标                  |
| `logs`             | 日志信息                                |

</div>

下面是一则在`Trainer`使用过程中注册自定义回调函数的示例。

在使用`Trainer`进行训练时，训练开始会输出`Starting training`，训练结束会输出`Finishing training`:

```python
class MyCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print("Starting training")

    def on_train_end(self, args, state, control, **kwargs):
        print('Finishing training')

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],
)
```

## TrainerState

`TrainerState`：包含训练状态信息的对象。

| 参数名称                | 类型                            | 默认值  | 描述                                                                                                           |
| ----------------------- | ------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------- |
| `epoch`                 | `float`                         | `None`  | 仅在训练期间设置，表示当前训练所在的`epoch`（小数部分表示当前完成的百分比）。                                  |
| `global_step`           | `int`                           | `0`     | 训练期间，表示已完成的步骤数。                                                                                 |
| `max_steps`             | `int`                           | `0`     | 当前训练期间要执行的更新步骤数量。                                                                             |
| `logging_steps`         | `int`                           | `500`   | 每`X`个更新步骤记录一次日志。                                                                                  |
| `eval_steps`            | `int`                           | `500`   | 每`X`个步骤执行一次评估。                                                                                      |
| `save_steps`            | `int`                           | `500`   | 每`X`个更新步骤保存一次检查点。                                                                                |
| `train_batch_size`      | `int`                           | `None`  | 训练数据加载器的批次大小。                                                                                     |
| `num_input_tokens_seen` | `int`                           | `0`     | 训练期间看到的令牌数量（输入令牌数量，而不是预测令牌数量）。                                                   |
| `total_flos`            | `float`                         | `0`     | 模型从训练开始到现在的总浮点运算次数（存储为浮点数以避免溢出）。                                               |
| `log_history`           | `List[Dict[str, float]]`        | `None`  | 从训练开始到现在的日志列表。                                                                                   |
| `best_metric`           | `float`                         | `None`  | 跟踪最佳模型时的最佳指标值。                                                                                   |
| `best_model_checkpoint` | `str`                           | `None`  | 跟踪最佳模型时的最佳模型。                                                                                     |
| `is_local_process_zero` | `bool`                          | `True`  | 此进程是否为本地（例如，如果在多台机器上以分布式方式进行训练，则在一台机器上）主进程。                         |
| `is_world_process_zero` | `bool`                          | `True`  | 此进程是否为全局主进程（在多台机器上以分布式方式进行训练时，这仅对一个进程为`True`）。                         |
| `is_hyper_param_search` | `bool`                          | `False` | 我否正在使用 `Trainer.hyperparameter_search` 进行超参数搜索。这将影响数据在`TensorBoard`中的记录方式。         |
| `stateful_callbacks`    | `List[StatefulTrainerCallback]` | `None`  | 附加到`Trainer`的回调函数，这些回调函数应该保存或恢复其状态。相关的回调函数应该实现`state`和`from_state`函数。 |

## `TrainerControl`

`TrainerControl`：用于控制训练流程的对象。

| 参数名称               | 描述                                                                            | 默认值  |
| ---------------------- | ------------------------------------------------------------------------------- | ------- |
| `should_training_stop` | 是否中断训练。如果为`True`，此变量将不会被重置为`False`，训练将直接停止。       | `False` |
| `should_epoch_stop`    | 是否中断当前`epoch`。如果为`True`，此变量将在下一个`epoch`开始时重置为`False`。 | `False` |
| `should_save`          | 是否在当前步骤保存模型。如果为`True`，此变量将在下一个步骤开始时重置为`False`。 | `False` |
| `should_evaluate`      | 是否在当前步骤评估模型。如果为`True`，此变量将在下一个步骤开始时重置为`False`。 | `False` |
| `should_log`           | 是否在当前步骤报告日志。如果为`True`，此变量将在下一个步骤开始时重置为`False`。 | `False` |

## 参考资料

<div class="grid cards" markdown>

- [HuggingFace API callbacks回调函数](https://huggingface.co/docs/transformers/main_classes/callback)
- [什么是训练器和回调函数？](https://blog.csdn.net/qq_44324007/article/details/140811427)

</div>
