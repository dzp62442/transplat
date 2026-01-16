# OmniScene 数据集实验说明（TransSplat 适配方案）

## 配置概览
- 入口配置计划新增 `config/dataset/omniscene.yaml`，结构参考 `re10k.yaml`，但与 DepthSplat 版本保持一致：`name: omniscene`、`roots: [datasets/omniscene]`、`defaults.view_sampler: all`（固定输出全部视角），并设置 `image_shape` 为 `224x400` 的默认值；`near=0.5`、`far=100.0`、`baseline_scale_bounds=false`、`make_baseline_1=false` 以对齐 DepthSplat 在 OmniScene 上的标定与尺度设置。
- 实验配置计划新增 `config/experiment/omniscene_112x200.yaml` 与 `config/experiment/omniscene_224x400.yaml`，以 `config/experiment/re10k.yaml` 为模板：
  - `defaults` 覆盖为 `dataset: omniscene`、`model/encoder: trans`、`loss: [mse, lpips]`。
  - `trainer.max_steps=100_001`；`trainer.val_check_interval=0.01`（每 0.01 个 epoch 验证）。
  - `data_loader.train/val/test.batch_size=1`（训练/验证/测试均为 1）。
  - `dataset.image_shape` 分别设为 `112x200`、`224x400`；同时沿用 `near=0.5`、`far=100.0`、`baseline_scale_bounds=false`、`make_baseline_1=false`。
  - 模型与损失参数保持本项目配置：`optimizer` 仍使用 `config/main.yaml` 的学习率/调度；`encoder` 的 `num_depth_candidates`、`costvolume_unet_*`、`gaussians_per_pixel` 等参数按 `re10k` 实验配置设置；`lpips.weight=0.05`。
- 训练/验证/测试节奏：
  - `trainer.max_steps=100_001`、`trainer.val_check_interval=0.01` 按需求固定在 experiment 中。
  - `train.eval_model_every_n_val=10`：本项目当前 `config/main.yaml`/`TrainCfg` 未提供该字段，暂不支持自动“每 N 次 val 跑一次 test”。若后续需要，可按 DepthSplat 的实现增补；不影响本次 OmniScene 基础适配。

## 数据加载流程
1. **注册入口**：在 `src/dataset/__init__.py` 中新增 `omniscene` → `DatasetOmniScene` 映射，并扩展 `DatasetCfg` 联合类型。
2. **数据类实现**：新增 `src/dataset/dataset_omniscene.py`，整体逻辑复用 DepthSplat 的 `DatasetOmniScene`：
   - 读取 `bins_*_3.2m.json` 列表：train 用 `bins_train_3.2m.json`，val 用 `bins_val_3.2m.json` 的稀疏子集（前 30000 里每 3000 取 1，再取前 10），test 默认使用 mini-test 抽样（`[0::14][:2048]`）。
   - `__getitem__` 读取 `bin_infos_3.2m/{token}.pkl`，固定 6 个环视摄像头的 key-frame 作为输入视图，再额外选同一 bin 的 `[1,2]` 帧作为渲染监督，并拼回输入帧形成 18 个 target 视图。
3. **图像/掩码/相对深度加载**：新增 `src/dataset/utils_omniscene.py`，基本复用 DepthSplat 的 `load_info/load_conditions`：
   - `load_info` 返回 `img_path` 与 `c2w/w2c`，坐标系处理保持 DepthSplat 的现状（不做 `flip_yz`）。
   - `load_conditions` 读取 `samples_small/sweeps_small`，缩放到配置分辨率并同步缩放内参；输出视图读取 `*_mask_small` 动态掩码，输入视图生成全 1 掩码。
   - 当 `load_rel_depth=True` 时，额外读取 DepthAnything-v2 预测的 disparity（`samples_dpt_small/sweeps_dpt_small`），转相对深度并归一化到 `[0,1]`。
4. **返回结构**：返回包含 `context/target` 的字典，字段与本项目兼容：`extrinsics`、`intrinsics`、`image`、`near`、`far`、`index`；`target` 额外包含 `masks` 与（仅 test 阶段）`rel_depth`。
5. **与本项目现有加载差异**：
   - `re10k` 使用 `IterableDataset + view_sampler` 在 `__iter__` 内采样；OmniScene 将采用 `Dataset` + 固定视角规则，不依赖 `view_sampler`。
   - `DatasetRE10k` 中的 `apply_augmentation_shim`、`apply_crop_shim` 对 OmniScene 不一定适用，默认不启用。

## 主程序调用方式
- `src/main.py` 仍按现有逻辑通过 Hydra 创建 `DataModule`，新增 `+experiment=omniscene_112x200`/`+experiment=omniscene_224x400` 即可切换 OmniScene。
- 本项目 `ModelWrapper` 不包含 DepthSplat 的 `eval_model_every_n_val` 与 `save_video_omniscene` 逻辑：
  - 若仅对齐训练/测试流程，可直接复用现有 `train/test` 入口；
  - 若需要 OmniScene 专用视频或“每 N 次 val 跑 test”，再按 DepthSplat 代码增补。

## PCC 指标补充方案
1. **相对深度加载（仅 test）**：
   - 在 `utils_omniscene.load_conditions` 中加入 `load_rel_depth` 开关，读取 `*_dpt_small` disparity，按 DepthSplat 的逻辑转换为相对深度并归一化。
   - `DatasetOmniScene` 在 `stage="test"` 时传入 `load_rel_depth=True`，并将 `rel_depth` 放入 `target`。
2. **测试时渲染深度**：
   - `DecoderSplattingCUDA` 已支持 `depth_mode` 输出深度。本项目 `ModelWrapper.test_step` 需在检测到 `rel_depth` 且 `compute_scores=true` 时调用 `decoder.forward(..., depth_mode="depth")`。
3. **PCC 计算位置**：
   - 在 `src/evaluation/metrics.py` 新增 `compute_pcc`（使用 `torchmetrics.PearsonCorrCoef`），接口与 DepthSplat 一致。
   - 在 `ModelWrapper.test_step` 中与 PSNR/SSIM/LPIPS 同步计算 `pcc`，输入为 `rel_depth` 与 `output.depth`。
4. **PCC 统计与汇总**：
   - 沿用 `test_step_outputs` 和 `on_test_end` 的 JSON 汇总方式，新增 `scores_pcc_all.json` 并写入 `scores_all_avg.json`。

## 与 DepthSplat 的差异与可复用性
1. **视角采样与 `num_context_views`**：
   - 本项目 `EncoderTrans` 依赖 `get_cfg().dataset.view_sampler.num_context_views`；`view_sampler: all` 在当前实现里不提供该字段（默认返回 0）。
   - 需要在 OmniScene 适配时补充该值（例如在 `view_sampler_all` 或 OmniScene 配置中明确设置为 6），或改为从 batch 动态推断输入视角数。
2. **动态掩码**：
   - 已在本项目补齐 `TrainCfg.use_dynamic_mask`，训练时可使用 `target.masks` 过滤损失；OmniScene 实验配置已默认开启该开关。
   - 目前对 `mse/lpips` 生效；若后续引入其它损失，可按相同方式传入掩码。
3. **Patch shim**：
   - 本项目 `apply_patch_shim` 仅处理 `image/intrinsics`；要支持 OmniScene 的 `masks/rel_depth`，需按 DepthSplat 的逻辑扩展裁剪。
4. **实验节奏**：
   - DepthSplat 提供 `eval_model_every_n_val` 与 `run_full_test_sets_eval`，本项目暂无该评测流程；若需要等价对齐可后续增补，但不影响基础训练/测试。

## 小结
本方案以 DepthSplat 的 OmniScene 适配为对照，保持本项目模型与训练超参不变，仅引入 OmniScene 数据集的配置/加载/调用与 PCC 指标支持。完成文档确认后，再按此方案逐步落地代码实现。
