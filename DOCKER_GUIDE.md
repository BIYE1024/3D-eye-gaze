# 🐳 Docker 完全入门指南 — 以你的 3D Eye Gaze 项目为例

> 本文档是专为你这个 PyTorch 项目定制的 Docker 教程。
> **理念：少讲理论，多动手。每个概念都配实际命令。**

---

## 目录

1. [Docker 是什么？（30秒理解）](#1-docker-是什么30秒理解)
2. [核心概念：镜像 vs 容器](#2-核心概念镜像-vs-容器)
3. [Dockerfile 逐行解析](#3-dockerfile-逐行解析)
4. [快速上手：构建并运行](#4-快速上手构建并运行)
5. [GPU 配置详解](#5-gpu-配置详解)
6. [docker-compose：一键操作](#6-docker-compose一键操作)
7. [日常开发工作流](#7-日常开发工作流)
8. [常用命令速查表](#8-常用命令速查表)
9. [故障排查](#9-故障排查)
10. [最佳实践总结](#10-最佳实践总结)

---

## 1. Docker 是什么？（30秒理解）

```
┌─────────────────────────────────────────────────────┐
│              没有 Docker                              │
│  你: "我跑不起来"  同事: "我电脑上能跑啊"              │
│  问题: Python 版本、CUDA 版本、OS 差异...              │
│                                                       │
│              有 Docker                                │
│  Docker = 带环境的"集装箱"                             │
│  一次构建，到处运行 ✓                                 │
│  "我环境在 Dockerfile 里，自己看"                     │
└─────────────────────────────────────────────────────┘
```

**Docker 本质**: 轻量级虚拟化（不是虚拟机！它直接共享宿主机内核）

| 对比 | 虚拟机 (VM) | Docker 容器 |
|------|-------------|-------------|
| 启动速度 | 分钟级 | 秒级 |
| 占用空间 | GB 级 | MB 级 |
| 性能损耗 | 大 | 几乎无 |
| 隔离程度 | 完全隔离 | 进程级隔离 |

---

## 2. 核心概念：镜像 vs 容器

```
┌──────────────────────────────────────────────────────┐
│                                                       │
│   Dockerfile  ──build──▶  Image  ──run──▶  Container │
│   (食谱/配方)            (做好的菜)       (正在吃的)   │
│   构建一次               只读、可分享     可写、在运行  │
│                                                       │
│   类比面向对象编程:                                    │
│   Dockerfile = 类定义 (class)                         │
│   Image      = 编译好的 .class 文件                    │
│   Container  = 实例化的对象 (new 出来的)                │
└──────────────────────────────────────────────────────┘
```

- **Dockerfile** → 你写的文本文件，描述"怎么构建这个环境"
- **Image（镜像）** → 由 Dockerfile 构建出来的只读模板
- **Container（容器）** → 镜像的运行实例，可以读写
- **Registry（仓库）** → 存放镜像的地方（Docker Hub、私有仓库等）

---

## 3. Dockerfile 逐行解析

你的 `Dockerfile` 里我已经加了详细注释，这里做个总结：

```dockerfile
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
# ↑ 基础镜像：NVIDIA 官方 CUDA 11.0 + cuDNN 8 + Ubuntu 20.04
#   Docker 是分层结构，每层在上一层基础上修改
#   就像 git commit 链

ENV DEBIAN_FRONTEND=noninteractive
# ↑ 环境变量：告诉 apt 不要弹交互窗口（容器里没人点"确定"）

RUN apt-get update && apt-get install -y python3 ...
# ↑ RUN: 在镜像中执行命令，结果保存为新的一层
#   合并多条命令 (用 &&) 减少层数 = 更小的镜像

WORKDIR /workspace
# ↑ 设置工作目录，后续命令都在这里执行
#   如果目录不存在会自动创建

COPY requirements.txt .
COPY . .
# ↑ 将宿主机文件复制进镜像
#   注意顺序！先复制 requirements.txt 再安装，再复制全部代码
#   这样改代码时不会重新安装依赖（利用 Docker 层缓存）

CMD ["python", "run.py", "--help"]
# ↑ 容器启动时的默认命令
#   可以被 docker run 后面的参数覆盖
```

### Docker 层缓存机制（重点！）

```
每次 docker build，Docker 检查每一步:
"这个命令和上次一样吗？COPY 的文件变了吗？"
如果没变 → 直接用缓存（秒级完成）
如果变了 → 重新执行这一步及之后所有步骤

这就是为什么 COPY requirements.txt 在 COPY . . 之前:
改了代码 → 只重跑最后两步
改了依赖 → 才重跑 pip install（很慢）
```

---

## 4. 快速上手：构建并运行

### 4.1 构建镜像

```bash
# 进入项目目录
cd Model-aware_3D_Eye_Gaze-main/Model-aware_3D_Eye_Gaze-main

# 构建镜像
#   -t eye-gaze:latest  给镜像打标签（名字:版本）
#   .                   指定构建上下文（Dockerfile 所在目录）
docker build -t eye-gaze:latest .
```

**首次构建耗时**: 5~10 分钟（需要下载基础镜像和 pip 包）
**后续构建**: 几秒钟（如果只改了代码）

### 4.2 运行容器

```bash
# --- 交互式模式（推荐新手）---
# 进入容器内的 bash，手动跑命令
#   --gpus all    使用所有 GPU
#   -it           交互模式 + 伪终端
#   --rm          退出时自动删除容器
docker run --gpus all -it --rm eye-gaze:latest /bin/bash

# 进去了！现在可以用容器里的 Python 环境：
python -c "import torch; print(torch.cuda.is_available())"  # 应该输出 True
python run.py --help
```

```bash
# --- 直接运行训练 ---
docker run --gpus all --rm \
    -v /path/to/your/Datasets:/workspace/Datasets:ro \
    -v /path/to/your/Results:/workspace/Results \
    eye-gaze:latest \
    python run.py --exp_name="my_run" --path_data="/workspace/Datasets/All" ...
```

### 4.3 数据卷挂载（Volume）

```
┌──────────────────────────────────────────────────────┐
│  宿主机 (Windows)          容器内 (Linux)             │
│                                                       │
│  D:\Datasets\     ←───→    /workspace/Datasets       │
│  (你的数据)      挂载       (容器里看到的)             │
│                                                       │
│  C:\Results\      ←───→    /workspace/Results         │
│  (结果保存到这)   挂载      (容器写入的)               │
└──────────────────────────────────────────────────────┘
```

**为什么用 Volume？**
1. 容器删除后数据还在（否则一起没了）
2. 数据集很大（几十 GB），绝不能打包进镜像
3. 可以直接在 Windows 里查看训练结果

```bash
# 示例：挂载数据目录
docker run --gpus all -it --rm \
    -v D:/Datasets/TEyeD:/workspace/Datasets \
    -v D:/Results:/workspace/Results \
    eye-gaze:latest /bin/bash
```

---

## 5. GPU 配置详解

### 5.1 前提条件

你的环境已经满足：
- ✅ Windows 11 + WSL 2
- ✅ Docker Desktop (已安装)
- ✅ NVIDIA 驱动 (537.70)
- ✅ NVIDIA Container Toolkit (Docker Desktop 内置)

### 5.2 在容器中使用 GPU

```bash
# 方式 1: 使用所有 GPU
docker run --gpus all eye-gaze:latest

# 方式 2: 指定 GPU 编号
docker run --gpus '"device=0"' eye-gaze:latest

# 方式 3: 指定 GPU 能力
docker run --gpus all --capabilities=gpu eye-gaze:latest
```

### 5.3 验证 GPU 可用

```bash
# 进入容器后运行
docker run --gpus all -it --rm eye-gaze:latest /bin/bash

# 在容器内:
nvidia-smi                           # 应该能看到你的 GTX 1660 Ti
python -c "import torch; print(torch.cuda.is_available())"  # True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # GTX 1660 Ti
```

---

## 6. docker-compose：一键操作

手动敲 `docker run` 太累？`docker-compose` 来帮忙。

### 快速上手

```bash
# 构建镜像
docker-compose build

# 交互式开发模式（进入容器终端）
docker-compose run --rm dev

# 训练模式
docker-compose run --rm train

# Jupyter Notebook 模式
docker-compose run --rm jupyter
# 然后用浏览器打开 http://localhost:8888
```

### docker-compose 常用命令

```bash
docker-compose build      # 构建/重新构建镜像
docker-compose up -d      # 后台启动所有服务
docker-compose down       # 停止并删除所有容器
docker-compose ps         # 查看服务状态
docker-compose logs -f    # 实时查看日志
docker-compose run --rm train   # 运行一次性命令
```

---

## 7. 日常开发工作流

### 场景 A: 我想在容器里调试代码

```bash
# 1. 启动开发容器（代码通过 volume 挂载，修改即时生效）
docker-compose run --rm dev

# 2. 在容器里运行调试
python run.py --exp_name="DEBUG" --model="res_50_3"

# 3. 改代码 → 在 VS Code 里直接改项目文件
#    因为代码是通过 volume 挂载的，容器里立即看到改动！
```

### 场景 B: 我想完整训练一次

```bash
# 1. 先确保数据目录正确（修改 docker-compose.yml 中的 volumes）
# 2. 启动训练
docker-compose run --rm train

# 3. 监控训练
docker-compose logs -f
```

### 场景 C: 我想跑 Jupyter Notebook

```bash
docker-compose run --rm jupyter
# 浏览器打开 http://localhost:8888
```

### 场景 D: 我要分享环境给同事

```bash
# 导出镜像
docker save -o eye-gaze.tar eye-gaze:latest

# 同事导入镜像
docker load -i eye-gaze.tar
```

---

## 8. 常用命令速查表

```bash
# ---------- 镜像管理 ----------
docker images                    # 列出所有镜像
docker rmi <镜像名>              # 删除镜像
docker build -t name:tag .       # 构建镜像（从当前目录的 Dockerfile）
docker pull python:3.8           # 从 Docker Hub 拉取镜像
docker push name:tag             # 推送镜像到仓库
docker save -o file.tar img:tag  # 导出镜像为文件
docker load -i file.tar          # 从文件导入镜像

# ---------- 容器管理 ----------
docker ps                        # 查看运行中的容器
docker ps -a                     # 查看所有容器（包括已停止的）
docker run <镜像> <命令>         # 创建并启动容器
docker start <容器ID>            # 启动已停止的容器
docker stop <容器ID>             # 停止容器
docker rm <容器ID>               # 删除容器
docker exec -it <容器ID> bash    # 进入运行中的容器

# ---------- 清理 ----------
docker system prune              # 清理所有未使用的资源
docker builder prune             # 清理构建缓存
docker image prune               # 清理未使用的镜像
docker volume prune              # 清理未使用的数据卷

# ---------- 调试 ----------
docker logs <容器ID>             # 查看容器日志
docker inspect <容器ID>          # 查看容器详细信息
docker stats                     # 实时查看容器资源占用
docker top <容器ID>              # 查看容器内进程
```

---

## 9. 故障排查

### 问题 1: `docker: Error response from daemon: could not select device driver "nvidia"`

**原因**: NVIDIA Container Toolkit 未安装或未正确配置。

**解决**:
```powershell
# 检查 Docker Desktop 是否使用 WSL 2 后端
# Docker Desktop → Settings → General → Use WSL 2 based engine ✓

# 如果还有问题，在 Docker Desktop → Settings → Resources → WSL Integration
# 确保启用 WSL 集成
```

### 问题 2: `torch.cuda.is_available()` 返回 False

**原因**: 基础镜像 CUDA 版本与运行时不匹配。

**解决**:
```bash
# 在容器内检查
nvidia-smi                              # GPU 是否可见？
echo $CUDA_VISIBLE_DEVICES             # 是否为 -1 (禁用 GPU)？
pip list | grep torch                   # PyTorch 是否正确安装？
```

### 问题 3: 构建很慢

```bash
# 利用 Docker 层缓存：确保 Dockerfile 中
# 先 COPY requirements.txt → pip install → COPY . .
# 这样改代码时不会重新安装依赖
```

### 问题 4: 镜像太大

```bash
# 查看镜像大小
docker images eye-gaze

# 减小镜像体积的技巧：
# 1. 使用 slim 基础镜像（如果不需要编译工具）
# 2. pip install 时加 --no-cache-dir
# 3. apt-get install 后加 rm -rf /var/lib/apt/lists/*
# 4. 用 .dockerignore 排除大文件
```

---

## 10. 最佳实践总结

| 原则 | 说明 |
|------|------|
| **一个容器一个进程** | 不要把数据库、Web服务、训练全塞一个容器 |
| **镜像要小** | 用 `.dockerignore`、清理缓存、选合适的基础镜像 |
| **数据在外部** | 数据集、模型权重、结果都通过 volume 挂载，不打包进镜像 |
| **层缓存优化** | 变化频率低的放前面（系统依赖 → Python 包 → 代码） |
| **版本标签** | 使用有意义的标签（`eye-gaze:v1.0`）而不是只用 `latest` |
| **不要用 root** | 生产环境应创建非 root 用户运行（本项目为简便暂时用了 root） |
| **多阶段构建** | 构建和运行用不同镜像（build 阶段装编译器，run 阶段只要运行时） |

---

## 🎯 下一步学习路线

完成本指南后，建议按以下顺序深入学习：

1. **Docker Compose 网络** — 多容器通信
2. **Docker Volume 类型** — bind mount vs named volume vs tmpfs
3. **多阶段构建 (Multi-stage build)** — 进一步减小镜像体积
4. **Docker Registry** — 搭建私有镜像仓库
5. **Kubernetes (K8s)** — 大规模容器编排（你项目里已经有 `k8s-pod.yaml`）

---

> 💡 **最重要的建议**: 不要只看文档，动手敲命令！Docker 是一个"动手学"的工具。
> 从 `docker run hello-world` 开始，然后构建你自己的镜像，过程中遇到报错就查 — 
> 每个报错你都会真正理解和记住一个概念。
