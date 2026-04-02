# 中文版玉米精量播种机变量施肥决策系统

本项目面向国产玉米精量播种机的电动变量施肥场景，提供一套中文本地桌面应用，用于完成以下链路：

- 导入 `CSV` 网格处方图
- 按单个排肥器独立生成变量施肥决策
- 基于论文最优前向 KAN 模型执行单点排肥量预测
- 根据处方图和机具参数进行作业仿真
- 导出时间线控制指令、模型路由追踪和仿真摘要
- 导出地图总览图、当前时刻细节图和独立图例图

系统默认复用同级目录 [`ComPare`](D:/Personal/eclipse_workspace/12.10/ComPare) 中已经导出的论文模型工件：

- `inverse_KAN`：域内优先模型
- `inverse_MLP`：外推优先模型
- `forward_KAN`：正向预测最佳模型（开度、转速 -> 排肥量）

## 1. 功能概览

### 1.1 决策逻辑

系统固定采用如下链路：

`目标施肥量(kg/ha) -> 单排目标排肥量(g/min) -> 三档开度(20/35/50 mm) -> 目标转速(r/min)`

其中单排目标排肥量计算公式为：

```text
target_mass_g_min = target_rate_kg_ha * row_spacing_m * travel_speed_kmh * 1.6666667
```

### 1.2 模型路由

- 当 `target_mass_g_min` 和 `opening_mm` 落在 KAN 训练域内时，使用 `inverse_KAN`
- 当请求超出训练域，或未来启用非标准开度档位时，切换到 `inverse_MLP`
- 所有决策都会记录 `selected_model` 与 `domain_status`

### 1.3 中文桌面界面

桌面端基于标准库 `tkinter` 实现，默认三栏布局：

- 左侧：处方图、模型包、正向预测、机器参数
- 中间：带坐标轴、热力图图例、机具轨迹、当前跨排行走状态的地图视图
- 右侧：当前时刻排位决策表、模型路由摘要、导出按钮

### 1.4 正向预测

左侧面板新增“正向预测”卡片，默认加载论文主结果中的 `forward_KAN` 工件，支持输入：

- 开度 `opening_mm`
- 转速 `speed_r_min`

点击“执行正向预测”后，界面会显示：

- 预测排肥量 `g/min`
- 按当前“行距 + 作业速度”反算的等效施肥量 `kg/ha`
- 训练域状态（域内 / 开度外推 / 转速外推 / 双外推）
- 预测状态（正常 / 低值钳制）

若当前行距或作业速度无效，则仍显示 `g/min` 预测结果，但不显示等效 `kg/ha`。

### 1.5 导出内容

每次导出默认生成 6 份文件：

- `row_command_timeline.csv`
- `model_routing_trace.csv`
- `simulation_summary.json`
- `map_overview.png`
- `map_current_frame.png`
- `map_legend.png`

## 2. 目录结构

```text
.
├── main.py
├── README.md
├── requirements.txt
├── samples/
│   └── prescription_grid.csv
├── tests/
│   ├── test_engine_and_export.py
│   ├── test_forward_prediction.py
│   ├── test_model_runtime.py
│   ├── test_prescription.py
│   └── test_ui_smoke.py
└── vrf_system/
    ├── controller.py
    ├── defaults.py
    ├── domain.py
    ├── engine.py
    ├── exporters.py
    ├── model_runtime.py
    ├── prescription.py
    ├── simulator.py
    └── ui.py
```

## 3. 运行方式

推荐直接使用已能加载论文模型的 Python 环境运行：

```powershell
D:\ProgramData\miniconda3\envs\fertilizer_gpu\python.exe main.py
```

如果你不想每次都进 PyCharm 或终端，项目根目录已经提供了双击启动文件：

- [Launch_VRF_System.bat](D:/Personal/eclipse_workspace/12.10/Variable_Rate_Fertilization_System/Launch_VRF_System.bat)：普通双击启动
- [Launch_VRF_System_Hidden.vbs](D:/Personal/eclipse_workspace/12.10/Variable_Rate_Fertilization_System/Launch_VRF_System_Hidden.vbs)：隐藏命令行窗口启动

日常使用建议直接双击 `Launch_VRF_System_Hidden.vbs`。

如果你使用当前系统 Python，也可以先安装依赖后运行：

```powershell
python -m pip install -r requirements.txt
python main.py
```

## 4. 测试方式

项目测试采用标准库 `unittest`，可直接运行：

```powershell
python -m unittest discover -s tests -v
```

## 5. 处方图 CSV 格式

首版只支持局部米制坐标的矩形网格处方图，字段如下：

| 字段名 | 含义 |
| --- | --- |
| `cell_id` | 网格唯一编号 |
| `center_x_m` | 网格中心 X 坐标（米） |
| `center_y_m` | 网格中心 Y 坐标（米） |
| `width_m` | 网格宽度（米） |
| `height_m` | 网格高度（米） |
| `target_rate_kg_ha` | 目标施肥量（kg/ha） |
| `zone_id` | 分区标识 |

样例文件位于 [samples/prescription_grid.csv](D:/Personal/eclipse_workspace/12.10/Variable_Rate_Fertilization_System/samples/prescription_grid.csv)。

## 6. 默认模型来源

默认模型工件读取以下目录：

- [inverse_KAN](D:/Personal/eclipse_workspace/12.10/ComPare/runs/20260326T200342_compare_all/artifacts/inverse/inverse_KAN)
- [inverse_MLP](D:/Personal/eclipse_workspace/12.10/ComPare/runs/20260326T200342_compare_all/artifacts/inverse/inverse_MLP)
- [forward_KAN](D:/Personal/eclipse_workspace/12.10/ComPare/runs/20260326T200342_compare_all/artifacts/forward/KAN)

前向 KAN 的参考预测文件位于：

- [forward_model_predictions.csv](D:/Personal/eclipse_workspace/12.10/ComPare/runs/20260326T200342_compare_all/forward_model_predictions.csv)

如果后续你重新训练并导出了新模型，也可以在界面中选择新的模型目录重新加载。
