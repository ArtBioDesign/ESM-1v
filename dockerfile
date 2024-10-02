# 使用基础镜像
ARG CUDA=12.2.0
FROM nvidia/cuda:${CUDA}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的工具
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制打包的环境到容器
COPY esm1v.tar.gz /opt/esm1v.tar.gz

# 解压环境
RUN mkdir /opt/esm1v && tar -xzf /opt/esm1v.tar.gz -C /opt/esm1v && \
    rm /opt/esm1v.tar.gz

# 设置 PATH 环境变量并运行 conda-unpack
RUN export PATH=/opt/esm1v/bin:$PATH && /opt/esmfold/bin/conda-unpack


# 设置 PATH 环境变量
ENV PATH=/opt/esm1v/bin:/opt/conda/bin:$PATH

# 设置工作目录
WORKDIR /workspace/esm1v

# 复制项目文件到容器
COPY ./esm1v/ /workspace/esm1v

# 设置默认入口点，激活环境并运行脚本

ENTRYPOINT ["/bin/bash", "-c", "source /opt/esm1v/bin/activate esm1v && /workspace/esm1v/predict.sh \"$@\"", "--"]

#["/bin/bash", "-c", "source activate fastMSA && exec /workspace/DHR/fastMSA.sh \"$@\"", "--"]

# 定义容器启动时的命令
CMD ["AAAAAi", "1"]

