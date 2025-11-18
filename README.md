# RK3588 YOLOv5 + ByteTrack 视觉追踪系统

## 项目简介

这是一个基于瑞芯微RK3588平台的实时人体检测与追踪系统，结合了YOLOv5目标检测算法和ByteTrack多目标追踪算法。系统能够在RK3588的NPU（Neural Processing Unit）上高效运行，实现实时的人体检测和持续追踪功能。

### 主要特性

- **高性能目标检测**：基于YOLOv5算法，使用RKNN模型加速，充分利用RK3588的NPU算力
- **智能目标追踪**：集成ByteTrack算法，实现稳定的多目标追踪
- **双模式运行**：支持检测模式（DETECTING）和追踪模式（TRACKING）的智能切换
- **实时视频处理**：支持USB摄像头和视频文件输入
- **高效状态管理**：采用卡尔曼滤波进行目标状态预测，提高追踪稳定性
- **自适应图像处理**：使用letterbox方法保持图像纵横比，提高检测精度

## 硬件要求

- **开发板**: 瑞芯微RK3588开发板
- **NPU**: 支持RKNN Runtime的NPU核心
- **摄像头**: USB摄像头（默认/dev/video1）或视频文件
- **内存**: 建议4GB以上
- **存储**: 至少2GB可用空间

## 软件依赖

### 系统库
- **RKNN Runtime**: RK3588的神经网络运行时库（librknnrt.so）
- **OpenCV 4.x**: 用于图像处理和视频I/O
- **Eigen3**: 用于矩阵运算和卡尔曼滤波
- **CMake 3.4.1+**: 构建系统
- **GCC/G++**: 支持C++14标准的编译器

### 库文件位置
```
/usr/lib/librknnrt.so          # RKNN运行时库
/usr/include/rknn/             # RKNN头文件
/usr/local/include/eigen3/     # Eigen3数学库
/usr/local/share/opencv4/      # OpenCV配置文件
```

## 项目结构

```
person/
├── CMakeLists.txt                      # CMake构建配置文件
├── build-linux_RK3588.sh              # Linux/RK3588编译脚本
├── README.md                          # 项目说明文档（本文件）
│
├── include/                           # 头文件目录
│   ├── BYTETracker.h                 # ByteTrack追踪器类定义
│   ├── STrack.h                      # 单个追踪目标类定义
│   ├── dataType.h                    # 数据类型定义（Eigen矩阵类型）
│   ├── kalmanFilter.h                # 卡尔曼滤波器定义
│   └── lapjv.h                       # 线性分配问题求解器
│
├── src/                               # 源代码目录
│   ├── bytetrack.cpp                 # 主程序入口，包含检测和追踪逻辑
│   ├── BYTETracker.cpp               # ByteTrack追踪器实现
│   ├── STrack.cpp                    # 单个追踪目标实现
│   ├── kalmanFilter.cpp              # 卡尔曼滤波器实现
│   ├── lapjv.cpp                     # LAPJV算法实现（匈牙利算法）
│   └── utils.cpp                     # 工具函数实现
│
├── model/                             # 模型文件目录
│   ├── yolov5s-640-640.rknn          # YOLOv5s RKNN格式模型（640x640输入）
│   ├── best_nofocus_new_21x80x80.rknn # 自定义训练的RKNN模型
│   ├── labels.txt                    # 类别标签文件（person, car）
│   └── PRC_9resize.mp4               # 测试视频文件
│
├── build/                             # 构建输出目录
│   └── build_linux_aarch64/          # Linux ARM64架构构建文件
│
└── install/                           # 安装输出目录
    └── rknn_yolov5_3588_new_Linux/   # Linux平台安装文件
        ├── rknn_yolov5_3588_bytetrack # 可执行文件
        ├── yolov5s-640-640.rknn      # 模型文件（拷贝）
        ├── model/                     # 模型目录（拷贝）
        └── output.avi                 # 输出视频文件
```

### 核心文件说明

#### 主程序文件
- **bytetrack.cpp**: 主程序，集成了RKNN模型加载、推理、目标检测、追踪和可视化功能

#### 追踪算法文件
- **BYTETracker.cpp/h**: ByteTrack多目标追踪算法核心实现
  - 实现了基于IoU的数据关联
  - 支持高低置信度目标的二次匹配
  - 处理目标的新建、更新、丢失和删除状态

- **STrack.cpp/h**: 单个追踪目标（Single Track）的管理
  - 维护目标的位置、速度、状态信息
  - 集成卡尔曼滤波进行状态预测
  - 支持目标的激活、重新激活、更新等操作

#### 算法支持文件
- **kalmanFilter.cpp/h**: 卡尔曼滤波器实现
  - 用于目标状态预测（位置、速度、加速度）
  - 提供预测和更新接口

- **lapjv.cpp/h**: LAPJV（Jonker-Volgenant）算法实现
  - 解决线性分配问题
  - 用于检测框与追踪目标的最优匹配

- **utils.cpp**: 工具函数库
  - 包含辅助函数实现

#### 配置文件
- **CMakeLists.txt**: CMake构建配置
  - 配置编译选项（C++14标准）
  - 设置库依赖（RKNN、OpenCV、Eigen3）
  - 定义安装规则

- **build-linux_RK3588.sh**: 自动化编译脚本
  - 设置交叉编译环境（如需要）
  - 执行CMake配置和编译
  - 自动安装到指定目录

## 编译与安装

### 编译步骤

1. **克隆或下载项目**
```bash
git clone <repository-url>
cd person
```

2. **使用自动编译脚本**
```bash
chmod +x build-linux_RK3588.sh
./build-linux_RK3588.sh
```

3. **手动编译（可选）**
```bash
# 创建构建目录
mkdir -p build/build_linux_aarch64
cd build/build_linux_aarch64

# 配置CMake
cmake ../.. -DCMAKE_SYSTEM_NAME=Linux -DTARGET_SOC=rk3588

# 编译（使用4个线程）
make -j4

# 安装到指定目录
make install
```

### 编译输出

编译成功后，在 `install/rknn_yolov5_3588_new_Linux/` 目录下会生成：
- `rknn_yolov5_3588_bytetrack`: 可执行文件
- `model/`: 模型文件目录（包含labels.txt和视频文件）
- `yolov5s-640-640.rknn`: YOLOv5模型文件

## 使用说明

### 基本使用

1. **进入安装目录**
```bash
cd install/rknn_yolov5_3588_new_Linux/
```

2. **运行程序**
```bash
# 使用默认YOLOv5模型
./rknn_yolov5_3588_bytetrack yolov5s-640-640.rknn

# 使用自定义模型
./rknn_yolov5_3588_bytetrack model/best_nofocus_new_21x80x80.rknn
```

### 程序参数

```bash
./rknn_yolov5_3588_bytetrack <model_path>
```

- `model_path`: RKNN格式的模型文件路径（必需参数）

### 摄像头配置

程序默认使用 `/dev/video1` 作为视频输入源，配置如下：

```cpp
// 在bytetrack.cpp中的配置（第601-611行）
cv::VideoCapture cap("/dev/video1", cv::CAP_V4L2);
cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
cap.set(cv::CAP_PROP_FPS, 30);
```

**修改视频源：**
- 如需使用其他摄像头设备，修改 `/dev/video1` 为对应设备号
- 如需使用视频文件，可以修改代码中的摄像头初始化部分

### 运行模式

系统支持两种运行模式，会自动切换：

#### 1. 检测模式（DETECTING）
- **功能**: 持续检测场景中的人体目标
- **策略**: 每帧选择置信度最高的人体目标作为候选
- **切换条件**: 检测到稳定目标90帧后，自动切换到追踪模式
- **视觉标识**: 
  - 绿色边框标注检测到的最佳目标
  - 显示置信度分数
  - 画面上显示"DETECT"模式标识

#### 2. 追踪模式（TRACKING）
- **功能**: 持续追踪已锁定的目标
- **策略**: 使用ByteTrack算法保持目标的连续追踪
- **切换条件**: 目标丢失超过30帧后，自动切换回检测模式
- **视觉标识**: 
  - 红色边框标注追踪目标
  - 显示追踪ID
  - 画面上显示"TRACK"模式标识

### 键盘控制

- **ESC键**: 退出程序
- 程序会实时显示处理结果窗口

### 输出说明

程序运行时会在终端输出：
```
sdk api version: <版本号>
driver version: <驱动版本>
model input num: 1, output num: 3
index=0, name=input, n_dims=4, dims=[1, 3, 640, 640], ...
num of boxes before nms: <NMS前检测框数量>
num of boxes: <NMS后检测框数量>
```

## 配置参数详解

### 模型输入参数（bytetrack.cpp）

```cpp
static const int INPUT_W = 640;              // 模型输入宽度
static const int INPUT_H = 640;              // 模型输入高度
```

### 检测参数

```cpp
#define NMS_THRESH 0.5                       // NMS（非极大值抑制）阈值
#define NUM_ANCHORS 3                        // 每个网格的anchor数量
#define NUM_CLASSES 2                        // 检测类别数（person, vehicle）
#define BBOX_CONF_THRESH 0.1                 // 边界框置信度阈值
#define OBJ_NUMB_MAX_SIZE 64                 // 最大检测目标数量
```

### Anchor配置

```cpp
// YOLOv5的anchor尺寸（3个检测层，每层3个anchor）
const int anchor[3][6] = {
    10, 13, 16, 30, 33, 23,      // 小目标检测层（stride=8）
    30, 61, 62, 45, 59, 119,     // 中目标检测层（stride=16）
    116, 90, 156, 198, 373, 326  // 大目标检测层（stride=32）
};
```

### 追踪参数

```cpp
BYTETracker tracker(fps, 50);                // fps: 视频帧率, 50: 追踪缓冲帧数
const int max_lost = 30;                     // 最大丢失帧数（超过则切换到检测模式）
const int DETECT_FRAME_THRESHOLD = 90;       // 检测模式持续帧数阈值
```

### 类别标签

系统支持两个检测类别（model/labels.txt）：
```
person    # 类别0：人体
car       # 类别1：车辆
```

程序主要追踪person类别（label=0），置信度阈值为0.3。

## 算法原理

### YOLOv5目标检测

**YOLOv5**（You Only Look Once v5）是一种单阶段目标检测算法，具有以下特点：

1. **网络结构**
   - Backbone: CSPDarknet53（特征提取）
   - Neck: PANet（特征融合）
   - Head: YOLOv5检测头（3个检测层）

2. **多尺度检测**
   - 三个检测层：stride=8, 16, 32
   - 分别检测小、中、大目标
   - 输出特征图尺寸：80x80, 40x40, 20x20

3. **Anchor机制**
   - 每个检测层使用3个预定义anchor
   - Anchor与目标框进行匹配
   - 预测相对于anchor的偏移量

4. **后处理流程**
   ```
   模型输出 → 解码边界框 → 置信度筛选 → NMS抑制 → 最终检测结果
   ```

### ByteTrack多目标追踪

**ByteTrack**是一种基于检测的多目标追踪（MOT）算法，核心思想：

1. **检测框分类**
   - 高置信度检测框（>0.5）
   - 低置信度检测框（0.1-0.5）
   - 分别处理以提高鲁棒性

2. **数据关联流程**
   ```
   第一轮匹配：高置信度检测框 ←→ 追踪目标
   第二轮匹配：低置信度检测框 ←→ 未匹配追踪目标
   ```

3. **匹配策略**
   - 使用IoU（Intersection over Union）度量相似度
   - LAPJV算法求解最优分配
   - 卡尔曼滤波预测目标位置

4. **状态管理**
   - New: 新检测到的目标
   - Tracked: 正在追踪的目标
   - Lost: 临时丢失的目标
   - Removed: 永久删除的目标

### 卡尔曼滤波

用于预测目标的下一帧位置，平滑追踪轨迹：

1. **状态向量**（8维）
   ```
   [x, y, w, h, vx, vy, vw, vh]
   x, y: 边界框中心坐标
   w, h: 边界框宽度和高度
   vx, vy, vw, vh: 对应的速度
   ```

2. **预测-更新循环**
   - 预测：基于运动模型预测下一帧状态
   - 更新：基于新的检测结果修正预测

### Letterbox图像预处理

保持原始图像纵横比的缩放方法：

```
原始图像 → 等比缩放 → 边缘填充（灰色） → 640x640输入
```

优点：
- 避免图像变形，保持目标真实形状
- 提高检测精度
- 需要记录缩放比例和填充尺寸，用于坐标还原

## RKNN模型信息

### 模型文件

1. **yolov5s-640-640.rknn**
   - 标准YOLOv5s模型
   - 输入尺寸：640x640x3
   - 输出层：3个（对应不同检测尺度）
   - 量化格式：INT8（RKNN量化）

2. **best_nofocus_new_21x80x80.rknn**
   - 自定义训练模型
   - 可能针对特定场景优化

### 模型转换

RKNN模型由浮点模型（ONNX/PyTorch）转换而来：

```bash
# 使用RKNN-Toolkit转换（示例）
python convert.py \
    --model yolov5s.onnx \
    --output yolov5s-640-640.rknn \
    --quantize int8 \
    --dataset dataset.txt
```

### 量化说明

- **量化类型**: Affine quantization（仿射量化）
- **量化参数**: 
  - zero_point (zp): 零点偏移
  - scale: 缩放因子
- **量化公式**:
  ```
  float_value = (int8_value - zero_point) * scale
  int8_value = clip((float_value / scale) + zero_point, -128, 127)
  ```

## 性能优化

### NPU加速

- RK3588的NPU可提供高达6TOPS算力
- RKNN Runtime自动将模型调度到NPU执行
- 使用 `RKNN_NPU_CORE_AUTO` 自动选择最优NPU核心

### 多线程优化

代码中使用OpenMP并行化：
```cpp
#pragma omp parallel sections
{
    #pragma omp section
    { /* 左子树排序 */ }
    #pragma omp section
    { /* 右子树排序 */ }
}
```

### 性能建议

1. **降低分辨率**: 如需更高帧率，可降低输入分辨率（如320x320）
2. **调整阈值**: 提高BBOX_CONF_THRESH可减少检测框数量，加快处理速度
3. **减少追踪目标**: 限制最大追踪目标数量
4. **优化模型**: 使用更轻量的YOLOv5n模型

## 常见问题与解决方案

### 1. 编译错误

**问题**: 找不到RKNN库
```
Could not find RKNN_RT_LIB
```
**解决**: 确保已安装RKNN Runtime库到 `/usr/lib/librknnrt.so`

**问题**: 找不到OpenCV
```
Could not find OpenCV
```
**解决**: 检查OpenCV是否正确安装，配置文件位于 `/usr/local/share/opencv4`

**问题**: Eigen3头文件缺失
```
fatal error: eigen3/Eigen/Core: No such file or directory
```
**解决**: 安装Eigen3
```bash
sudo apt-get install libeigen3-dev
```

### 2. 运行时错误

**问题**: 打开摄像头失败
```
Failed to open camera!
```
**解决**: 
- 检查摄像头设备是否存在：`ls /dev/video*`
- 修改代码中的设备路径
- 确认摄像头驱动已加载

**问题**: 模型加载失败
```
Open rknn model file xxx.rknn failed.
```
**解决**: 
- 确认模型文件路径正确
- 检查模型文件是否完整（未损坏）

**问题**: NPU初始化失败
```
rknn_init error ret=-1
```
**解决**: 
- 确认RKNN驱动已正确加载：`lsmod | grep rknpu`
- 检查设备节点：`ls /dev/rknpu*`
- 重启开发板

### 3. 性能问题

**问题**: 帧率过低
**解决**:
- 降低输入分辨率
- 提高检测阈值（减少检测框）
- 使用更轻量的模型
- 检查NPU是否正常工作

**问题**: 追踪不稳定
**解决**:
- 调整追踪参数（track_buffer, max_lost）
- 降低BBOX_CONF_THRESH以检测更多候选框
- 改善光照条件

### 4. 显示问题

**问题**: 无法显示窗口
```
cv::imshow not working
```
**解决**:
- 确保系统支持GUI显示
- 如果是SSH远程连接，启用X11转发或使用VNC
- 可选择保存视频文件而不是实时显示

## 扩展开发

### 添加新的检测类别

1. 修改 `NUM_CLASSES` 宏定义
2. 更新 `model/labels.txt` 文件
3. 重新训练和转换模型

### 切换为视频文件输入

修改 `bytetrack.cpp` 中的视频捕获部分：

```cpp
// 替换摄像头输入
// cv::VideoCapture cap("/dev/video1", cv::CAP_V4L2);

// 使用视频文件输入
cv::VideoCapture cap("model/PRC_9resize.mp4");
```

### 保存输出视频

取消注释代码中的VideoWriter部分：

```cpp
// 在主循环前初始化VideoWriter
cv::VideoWriter writer("output.mp4", 
                       cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
                       fps, 
                       cv::Size(img_w, img_h));

// 在主循环中保存帧
writer.write(raw_img);

// 在程序结束时释放
writer.release();
```

### 添加其他追踪算法

项目架构支持替换追踪算法：
1. 实现新的Tracker类（继承或参考BYTETracker接口）
2. 在主程序中实例化新的追踪器
3. 调用update()方法更新追踪状态

## 技术参考

### 相关论文

1. **YOLOv5**
   - Repository: https://github.com/ultralytics/yolov5
   - Paper: 无正式论文（工程项目）

2. **ByteTrack**
   - Paper: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box" (ECCV 2022)
   - Repository: https://github.com/ifzhang/ByteTrack

3. **LAPJV Algorithm**
   - Paper: "A Shortest Augmenting Path Algorithm for Dense and Sparse Linear Assignment Problems" (Computing, 1987)

### RK3588相关资源

- RKNN-Toolkit2: https://github.com/rockchip-linux/rknn-toolkit2
- RK3588技术文档: https://www.rock-chips.com/a/cn/product/RK35xilie/2022/0926/1660.html

### 开发工具

- **CMake**: https://cmake.org/
- **OpenCV**: https://opencv.org/
- **Eigen**: https://eigen.tuxfamily.org/

## 许可证

本项目的许可证信息请参考源代码仓库。

## 贡献者

- 项目作者：yfyfy200121

## 更新日志

### Version 1.0
- 初始版本发布
- 支持YOLOv5目标检测
- 集成ByteTrack追踪算法
- 支持RK3588 NPU加速
- 实现检测和追踪双模式切换

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: 在项目仓库中提交issue
- Email: （如有请补充）

---

**最后更新**: 2024年11月

感谢使用RK3588 YOLOv5 + ByteTrack视觉追踪系统！
