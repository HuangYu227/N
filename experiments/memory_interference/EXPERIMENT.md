# 旧内容检索与 TTT 效用诊断

## 当前阶段

这是待执行的诊断协议，不是已获得的实验结论。使用用户正在传输的 H100 State-TTT 权重；加载和 GPU 启动前再次确认传输完成。不用旧 L20 step5 或旧双 ψ step250 替代。只读主模型，不训练、不修改架构。

## 固定比较

- GPU6：full、no-local；GPU7：no-persistent、no-ttt，各卡顺序执行两组。
- 同一权重 SHA256、训练例 clip、首帧、文本、真实 camera、seed3407、121 latent、40 个预测 chunk。
- 逐 chunk 写入 GT clean 历史，原生 GDN/相机/FFN 缓存按现有生产逻辑保留，相机窗口2 chunk。
- full 开启临时和 clean State-TTT；no-local 只保留 clean；no-persistent 只保留临时；no-ttt 两者关闭。
- **四组全部保留 Correct 写入和 Cayley 输运**。no-ttt 不等于移除线性记忆，也不是原生 SANA。
- chunk2/6/11/21/31/40 上，用生产 scheduler 中最接近 σ=0.2/0.5/0.8 的索引构造同噪声 GT 去噪探针。仅条件预测，不做 CFG 外推。

## 测量与解释

1. 首帧 prefill 后，冻结真实模型的64个 Q/K/V token 及其初始 QS 响应。跟踪之后同一 S 对这些固定探针的读出。
2. 用实际 Cayley 因子的累计右乘矩阵补偿值通道旋转。只消除已知通道变换，不声称已经处理相机几何、遮挡或物理场景变化。
3. `old_key_binding_nmse` 衡量旧 K→V 关联的重建误差；它是内层记忆目标代理。
4. `old_query_response_drift` 衡量旧 Q 的读出相对最初响应的变化；变化本身不等于退化。最初响应也可能不准确。
5. 同一 clean 调用、相同输入特征和 pre-TTT S，计算仅 Correct 的反事实结果，与实际 TTT→Correct 比较，避免把上游网络变化误算成本层更新效果。
6. 生产 velocity MSE 提供任务层面的配对效用测量。训练例且使用 GT 历史，不能据此宣称长程视频质量提升或独立测试集泛化。
7. 所有点来自同一条序列；chunk/head 不是独立统计样本。不画伪置信区间，不预设干扰一定存在。

## 必须通过的验收

- 本地14项检查已通过：数学/快照/缓存隔离、真实 production TTNAnchor/StateTTT/TTNSession 的四组路径、结果配对拒绝检查，以及20 heads×112维、rank64的数值契约。它们不替代待执行的 L20 真实模型 GPU 验收。
- 正确实际 TTT 调用数；关闭临时更新不改变 clean 历史：full=no-local，no-persistent=no-ttt。
- 各组 prefill 同源；跨 GPU 权重、输入、代码和协议指纹相同。
- 真实 noisy forward 可重复；重复探针不改变持久 S/ψ、commit/revision；生产 loss 与手算 velocity MSE 一致。
- 同特征反事实在无 clean TTT 时与生产结果逐位一致。
- 未通过运行验收不输出汇总动机图；部分 JSON 仅用于排错。

## 输出

每组 JSON、逐 chunk partial JSON、checkpoint/输入/代码指纹、实际调用计数。运行结束验收后，生成旧关联曲线、旧 Q 响应变化曲线、token 误差热图、当前拟合/旧关联散点，以及任务 velocity MSE 曲线。保存 PNG 和 PDF。

这些图用来判断是否有值得解决的问题。只有真实测量支持后，才讨论结构创新和论文动机；当前没有假定能证明 TTT 有益或有害。

## 启动与资源

`prepare_launch.py` 必须在用户确认 H100 权重传输完成后执行。脚本先核对既有登记进程、只读生产源码指纹、指定 GPU 是否空闲，再创建本轮专属 tmux 窗口和记录。旧运行登记原样归档，不推断旧作业的退出原因。

两张卡各一个进程，每进程只看见其指定 GPU；推理分配器限制36GiB，每任务独立 systemd cgroup 限制64GiB RAM、不用 swap。若整机 MemAvailable 低于64GiB，保护器只会停止本轮已验证归属的进程。两组推理不需要重新分配数百 GiB 训练 RAM。四小时仅为诊断运行上限，不是预计耗时。

结束后执行 `plot_diagnostic.py <gpu6/results> <gpu7/results> --output <新图目录>`；只有四组完整且验收通过才生成图。
