## 项目启动文档：层叠瞬态自由能网络 (LTFN) 实验框架 [C++/Eigen 实现版]

**版本**：v1.0-cpp  
**目标**：使用 C++ 和 Eigen 库，从零实现无需反向传播、统一推理与学习的 LTFN 网络，并在 MNIST 无监督重建任务上验证其有效性。  
**读者**：AI Coding Agent（你将被要求根据本文档编写完整 C++ 代码）

---

## 1. 项目概述

本项目旨在实现一种全新的神经网络学习范式——**层叠瞬态自由能网络 (LTFN)**，该框架受大脑预测编码理论启发，彻底打破传统深度学习的“前向推理/反向传播”阶段分离。在 LTFN 中，网络的状态（神经元活动）和参数（突触权重）在同一连续时间动力学中依据局部信息同步更新，从而让“思考”和“学习”融为一体。

我们将通过一个**无监督图像重建任务**（MNIST 手写数字）来验证该框架的学习能力。网络接收原始图像作为输入，同时将其作为顶层向下生成的预测目标，通过最小化全局预测误差能量，自行学会压缩、表征并重建输入。整个过程无需标签，无需反向传播算法。

本实现使用 C++ 语言与 **Eigen** 线性代数库，旨在构建一个独立于现有深度学习框架、完全自主可控的基础原型，为后续定制化硬件和新一代 AI 框架奠定底层代码基础。

---

## 2. 理论背景与核心思想

### 2.1 预测编码与信用分配

传统反向传播（Backpropagation）解决信用分配问题的方式是：先将输入前向传播得到输出，再计算误差，并将误差梯度逐层反向传递以更新权重。这要求：
- 严格的前向-反向串行阶段；
- 全局同步的误差信号；
- 正向与反向使用对称的权重。

神经科学证据表明生物大脑并不符合这些条件。大脑采用的是一种**预测编码**机制：高级皮层不断向低级皮层发送对感官输入的**预测**，低级皮层仅将**预测误差**回传。整个系统持续并行运行，所有神经元仅凭局部可得的信息（来自上一层的预测、来自下一层的误差）来更新自身的活动状态和突触权重。

### 2.2 LTFN 的核心原理

LTFN 将上述机制形式化为一个能量最小化过程。网络由多个层级组成，每一层都在尝试预测下一层的活动。全局能量定义为所有层级的预测误差平方和：

\[
E = \frac{1}{2} \sum_{l=0}^{L-1} \| e_l \|^2
\]

其中第 \(l\) 层的预测误差 \(e_l\) 是该层实际表示 \(r_l\) 与上层生成的预测之间的差异：

\[
e_l = r_l - f(W_{l+1\to l}^g \cdot r_{l+1})
\]

\(f\) 为元素级非线性（如 sigmoid），\(W^g\) 是从高层向低层的生成权重。最底层（\(l=0\)）作为感官输入层，其“误差”就是输入图像与顶层向下生成的重建之间的差异。

整个网络的状态演化和权重更新都遵循同一个能量函数的梯度下降：

- **状态弛豫（推理）**：每层的表示单元 \(r_l\) 根据局部误差快速更新，驱动网络走向能量最低状态，即对当前输入的最佳解释。
- **权重更新（学习）**：突触权重 \(W^g\) 也沿着能量梯度进行缓慢调整，逐渐塑造能够更好预测输入的世界模型。

关键创新点：
1. **无阶段分离**：状态更新和权重更新在每一时刻同时发生，推理即学习。
2. **完全局部规则**：任何连接权重的修改只依赖该连接的前后神经单元活动（局部误差和激活值），无需全局梯度回传。
3. **生物学合理性**：信息双向流动（预测向下、误差向上），异步并行，无需中央控制器。

---


## 3. 架构详细描述（C++ 实现视角）

### 3.1 网络结构

采用对称漏斗形全连接网络：

| 层索引 | 名称 | 维度 | 说明 |
|--------|------|------|------|
| 0 | 输入层 | 784 | 固定为输入图像，不可更新 |
| 1 | 隐层1 | 256 | 动态状态向量 |
| 2 | 隐层2 | 64  | 动态状态向量 |
| 3 | 顶层 (L=3) | 32 | 最高层表示，无来自上层的预测 |

**权重矩阵**（使用 Eigen 动态矩阵类型）：
- `W[0]`：`(784, 256)`，从层1预测层0。
- `W[1]`：`(256, 64)` ，从层2预测层1。
- `W[2]`：`(64, 32)`  ，从层3预测层2。

所有非线性激活函数均采用 logistic sigmoid：

\[
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
\]

### 3.2 能量函数

对于输入 \(s\)（784 维向量），固定 `r[0] = s`。能量定义：

```
e[0] = r[0] - σ(W[0] * r[1])
e[1] = r[1] - σ(W[1] * r[2])
e[2] = r[2] - σ(W[2] * r[3])
E   = 0.5 * (e[0].squaredNorm() + e[1].squaredNorm() + e[2].squaredNorm())
```

### 3.3 离散时间动力学（C++ 更新公式）

以下公式将直接映射到 Eigen 操作。固定时间常数 `tau_r`，学习率 `lr_w`。

**状态更新**（快速，步长 `dt_r`）：

- **顶层 (l=3)**：
  ```cpp
  VectorXd z3 = W[2] * r[3];                // (64,)
  VectorXd fprime3 = sigmoid_deriv(z3);      // (64,)
  VectorXd delta3 = e[2].cwiseProduct(fprime3); // (64,)
  VectorXd dr3 = (W[2].transpose() * delta3) * (dt_r / tau_r);
  r[3] += dr3;
  ```

- **隐层2 (l=2)**：
  ```cpp
  VectorXd z2 = W[1] * r[2];                // (256,)
  VectorXd fprime2 = sigmoid_deriv(z2);
  VectorXd back = W[1].transpose() * (e[1].cwiseProduct(sigmoid_deriv(W[1] * r[2])));
  // 注意：上述 back 计算中，W[1]*r[2] 即为 z2，可复用
  VectorXd grad2 = e[2] - fprime2.cwiseProduct(back);
  r[2] -= (dt_r / tau_r) * grad2;
  ```

- **隐层1 (l=1)**：
  ```cpp
  VectorXd z1 = W[0] * r[1];
  VectorXd fprime1 = sigmoid_deriv(z1);
  VectorXd back1 = W[0].transpose() * (e[0].cwiseProduct(sigmoid_deriv(z1)));
  VectorXd grad1 = e[1] - fprime1.cwiseProduct(back1);
  r[1] -= (dt_r / tau_r) * grad1;
  ```

**权重更新**（慢速，步长 `dt_w`）：

对于每个 `l = 0, 1, 2`：
```cpp
VectorXd z = W[l] * r[l+1];                 // 前向预测输入
VectorXd fprime = sigmoid_deriv(z);
VectorXd delta_w = e[l].cwiseProduct(fprime); // 局部误差调制
MatrixXd dW = delta_w * r[l+1].transpose();   // 外积，尺寸与 W[l] 相同
W[l] += lr_w * dW * dt_w;
```

**注意**：权重更新步骤可以使用更新后的 `r` 向量（状态弛豫后）来计算，即先做完一次状态更新，再用新的状态计算误差和权重梯度，这符合伪代码顺序。

### 3.4 统一松弛过程（单步）

在一个时间步内，顺序执行以下操作（对应 `step()` 方法）：
1. 根据当前 `r` 计算所有层的误差 `e[0], e[1], e[2]`。
2. 根据误差计算各层状态梯度并更新 `r[1], r[2], r[3]`。
3. 使用更新后的 `r` 重新计算误差（或直接复用上一步的误差，两种方案均可，推荐复用以保证局部性），然后计算权重梯度并更新所有权重。

### 3.5 训练与推理流程

- **训练（在线学习）**：对每个输入图像，调用 `relax()` 方法执行固定步数（如 `K=200`），在这些步数内同时更新状态和权重。
- **测试（纯推理）**：冻结权重（不调用权重更新代码），对输入执行相同步数的状态弛豫，返回最终的重建图像（`σ(W[0] * r[1])`）。

---

## 4. 伪代码（C++ 风格，基于 Eigen）

```cpp
#include <Eigen/Dense>
#include <vector>
#include <random>

using namespace Eigen;
using std::vector;

// --- 辅助函数 ---
inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(500.0, x))));
}

inline double sigmoid_deriv(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// 对向量逐元素应用 sigmoid
VectorXd sigmoid_vec(const VectorXd& x) {
    return x.unaryExpr([](double v) { return sigmoid(v); });
}

// 对向量逐元素应用 sigmoid 导数
VectorXd sigmoid_deriv_vec(const VectorXd& x) {
    return x.unaryExpr([](double v) { return sigmoid_deriv(v); });
}

class LTFN {
public:
    int L;                      // 可学习层数 (总层数 = L+1)
    vector<int> dims;
    vector<MatrixXd> W;         // 权重 W[l]: (dims[l], dims[l+1])
    double tau_r;
    double lr_w;

    LTFN(const vector<int>& dimensions, double tau_r = 0.1, double lr_w = 1e-5)
        : L(dimensions.size() - 1), dims(dimensions), tau_r(tau_r), lr_w(lr_w) {
        // Xavier 初始化
        std::random_device rd;
        std::mt19937 gen(rd());
        for (int l = 0; l < L; ++l) {
            int n_in = dims[l+1];   // 注意 W[l] 输入维度是 dims[l+1]
            int n_out = dims[l];
            double limit = std::sqrt(6.0 / (n_in + n_out));
            std::uniform_real_distribution<> dis(-limit, limit);
            W.push_back(MatrixXd::Zero(n_out, n_in));
            for (int i = 0; i < n_out; ++i)
                for (int j = 0; j < n_in; ++j)
                    W[l](i, j) = dis(gen);
        }
    }

    // 单步松弛（推理 + 学习交织）
    void step(vector<VectorXd>& r, double dt_r, double dt_w) {
        // ---- 1. 计算所有层的预测误差 ----
        vector<VectorXd> e(L);
        vector<VectorXd> pred(L); // 预测值
        for (int l = 0; l < L; ++l) {
            VectorXd z = W[l] * r[l+1];
            pred[l] = sigmoid_vec(z);
            e[l] = r[l] - pred[l];
        }

        // ---- 2. 状态更新 (快速) ----
        vector<VectorXd> r_new = r; // 复制一份，以便同时使用旧 r 计算权重更新
        // 顶层 (L)
        {
            VectorXd z3 = W[L-1] * r[L];
            VectorXd fprime3 = sigmoid_deriv_vec(z3);
            VectorXd delta3 = e[L-1].cwiseProduct(fprime3);
            VectorXd dr_L = (W[L-1].transpose() * delta3) * (dt_r / tau_r);
            r_new[L] += dr_L;
        }
        // 中间层 l = L-1 down to 1
        for (int l = L-1; l >= 1; --l) {
            VectorXd z = W[l-1] * r[l];
            VectorXd fprime = sigmoid_deriv_vec(z);
            VectorXd back = W[l-1].transpose() * (e[l-1].cwiseProduct(sigmoid_deriv_vec(z)));
            VectorXd grad = e[l] - fprime.cwiseProduct(back);
            r_new[l] -= (dt_r / tau_r) * grad;
        }
        // 输入层 r[0] 保持不变
        // 更新 r 向量
        for (int l = 1; l <= L; ++l)
            r[l] = r_new[l];

        // ---- 3. 权重更新 (慢速，使用更新后的 r) ----
        for (int l = 0; l < L; ++l) {
            VectorXd z = W[l] * r[l+1];
            VectorXd fprime = sigmoid_deriv_vec(z);
            VectorXd delta_w = e[l].cwiseProduct(fprime); // 此处仍用旧误差 e[l]? 
            // 为严格符合局部规则，可重新计算误差，但此处允许稍许异步，为简化使用原误差。
            // 若需精确，应重新计算 e[l] = r[l] - sigmoid(W[l] * r[l+1]);
            // 这里选择重新计算以保证梯度精确落在新状态：
            VectorXd new_pred = sigmoid_vec(z);
            VectorXd new_e = r[l] - new_pred;
            delta_w = new_e.cwiseProduct(fprime);
            MatrixXd dW = delta_w * r[l+1].transpose();
            W[l] += lr_w * dW * dt_w;
        }
    }

    // 执行多步松弛，返回最终重建
    VectorXd relax(const VectorXd& input, int steps, double dt_r = 0.1, double dt_w = 1.0) {
        vector<VectorXd> r(L+1);
        r[0] = input;
        for (int l = 1; l <= L; ++l)
            r[l] = VectorXd::Zero(dims[l]);

        for (int t = 0; t < steps; ++t) {
            step(r, dt_r, dt_w);
        }
        // 返回重建
        return sigmoid_vec(W[0] * r[1]);
    }

    // 纯推理重建（不更新权重）
    VectorXd reconstruct(const VectorXd& input, int steps = 200, double dt_r = 0.1) {
        vector<VectorXd> r(L+1);
        r[0] = input;
        for (int l = 1; l <= L; ++l)
            r[l] = VectorXd::Zero(dims[l]);

        for (int t = 0; t < steps; ++t) {
            // 与step类似，但省略权重更新
            vector<VectorXd> e(L);
            for (int l = 0; l < L; ++l) {
                VectorXd z = W[l] * r[l+1];
                VectorXd pred = sigmoid_vec(z);
                e[l] = r[l] - pred;
            }
            // 状态更新
            VectorXd z_top = W[L-1] * r[L];
            VectorXd delta_top = e[L-1].cwiseProduct(sigmoid_deriv_vec(z_top));
            r[L] += (W[L-1].transpose() * delta_top) * (dt_r / tau_r);
            for (int l = L-1; l >= 1; --l) {
                VectorXd z = W[l-1] * r[l];
                VectorXd fprime = sigmoid_deriv_vec(z);
                VectorXd back = W[l-1].transpose() * (e[l-1].cwiseProduct(sigmoid_deriv_vec(z)));
                VectorXd grad = e[l] - fprime.cwiseProduct(back);
                r[l] -= (dt_r / tau_r) * grad;
            }
            r[0] = input;
        }
        return sigmoid_vec(W[0] * r[1]);
    }
};
```

---

## 5. 实验设计

### 5.1 数据集
- **MNIST 手写数字**：60,000 训练图像，10,000 测试图像，28×28 灰度。
- **预处理**：像素值除以 255 归一化到 [0,1]，拉平为 784 维向量。无需任何其他变换或数据增强。
- **数据加载**：使用 C++ 读取 MNIST 原始二进制文件（train-images-idx3-ubyte 等）。可自行实现或引入轻量头文件 `mnist_reader.hpp`（开源，仅需包含）。

### 5.2 模型配置
- 维度：`[784, 256, 64, 32]` (L=3)
- 时间常数 `tau_r = 0.1`
- 权重学习率 `lr_w = 1e-5`
- 状态更新步长 `dt_r = 0.1`，权重更新步长 `dt_w = 1.0`（乘积为 `lr_w * dt_w = 1e-5`）
- 每张图像的松弛步数 `steps = 200`（根据收敛情况可调整，初始阶段可稍多）

### 5.3 训练流程（在线学习）
1. 加载训练图像（`vector<VectorXd>`）。
2. 可打乱顺序，然后遍历每一张图像：
   - 调用 `net.relax(img, steps, dt_r, dt_w)`。
3. 监控进度：每处理 500 张图像，在测试集上用 `reconstruct()` 计算平均重建 MSE，并保存几幅重建图像。

**性能考量**：200 步 × 60k 张 ≈ 12M 次网络更新，在单线程 CPU 上可能需要数小时。开发调试时建议先用 5000 张图像快速验证能量下降和重建趋势。

### 5.4 评估指标
- **重建均方误差 (MSE)**：`(input - reconstruction).squaredNorm() / 784`，在测试集报告平均。
- **可视化**：将原图和重建图保存为 PGM 文件（二进制灰度图），或使用 `stb_image_write` 输出 PNG。
- **能量日志**：记录每 500 张后某固定测试图像的能量值。

### 5.5 成功标准
- 学习约 20k-50k 张图像后，测试集平均 MSE < 0.02（像素平均误差 < 0.14）。
- 肉眼可清晰辨认数字轮廓，背景噪声低。

---

## 6. 边界条件与实现细节

### 6.1 数值稳定性
- `sigmoid` 输入做裁剪：`x = std::clamp(x, -500.0, 500.0)`。
- Xavier 初始化时，确保矩阵值服从均匀分布，范围 $\pm \sqrt{\frac{6}{fan\_in + fan\_out}}$。
- 状态更新步长 `dt_r` 不宜超过 0.2，否则动态可能发散。可通过监控能量是否单调下降来调节。

### 6.2 内存与复用
- 在 `step()` 中频繁创建临时 `VectorXd` 和 `MatrixXd` 会导致大量动态内存分配。建议将常用临时变量（如 `z`, `fprime`, `delta`, `back`）作为循环外复用的成员变量或静态分配，提升效率。
- 权重更新时，外积 `delta * r[l+1].transpose()` 产生临时矩阵，可接受；但若追求极致效率，可手动循环赋值。

### 6.3 数据加载
- MNIST 文件格式：文件头 16 字节（magic, 图像数, 行, 列），之后为连续字节（0-255）。实现解析函数时注意字节序为大端 (big-endian)，可使用 `ntohl` 或手动转换。
- 可选使用已验证的开源 `mnist_reader.hpp`（MIT 许可，单头文件）。

### 6.4 编译优化
- 开启编译优化 `-O3 -DNDEBUG`。
- 若 Eigen 启用 OpenMP，可利用多核加速矩阵乘法（需链接 `-fopenmp`）。
- 确保所有向量和矩阵使用列优先存储（Eigen 默认），与我们的维度定义一致。

### 6.5 调试建议
- 初期可在极小的合成数据集（如固定 10 张二进制图像）上运行，检查能量是否每步下降。
- 打印权重梯度的范数，确保不为零且量级合理。
- 重建效果差时，尝试降低学习率或增加松弛步数。

---

## 7. 实现步骤建议

1. **搭建开发环境**：下载 Eigen（仅需头文件），配置 CMakeList 或 Makefile。
2. **实现辅助函数**：`sigmoid`、`sigmoid_deriv` 的标量与向量版本。
3. **实现 LTFN 类**：
   - 构造函数：权重 Xavier 初始化。
   - `step()` 方法：先不加权重更新，测试状态弛豫能否从随机初始状态重建出某固定图像的近似。
4. **测试状态弛豫**：固定权重，对单张图像调用 `reconstruct()`，打印 MSE。
5. **添加权重更新**：完成完整的 `step()`。
6. **集成 MNIST 数据加载**：解析原始文件或引入 `mnist_reader.hpp`。
7. **编写训练循环**：在线学习，每 500 张评估一次测试 MSE。
8. **添加可视化**：将原图与重建保存为 PGM 文件，可用 Python 脚本离线查看。
9. **参数调优**：根据收敛曲线调整 `lr_w`、`steps`。
10. **最终报告**：记录训练曲线、重建示例，分析 LTFN 的有效性。

---

## 8. 预期交付物

- 一个 C++ 项目，包含：
  - `main.cpp`：训练与评估循环
  - `ltfn.h` / `ltfn.cpp`：LTFN 类实现
  - `utils.h`：数据加载、图像保存等工具函数
  - `Makefile` 或 `CMakeLists.txt`
- 运行后可开始在线学习，控制台打印定期 MSE。
- 学习结束后自动保存重建示例图（PGM 文件）。
- 一份简短的分析文档（可选）：LTFN 是否在没有反向传播的前提下学到了有效的重建？统一动力学是否稳定？

---

现在，请根据以上文档开始编写完整的 C++ 实验代码。如遇任何库使用或实现细节的疑问，随时请求澄清。
