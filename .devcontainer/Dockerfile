FROM mcr.microsoft.com/devcontainers/base:resolute

# Switch apt mirror
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources \
  && sed -i 's|http://security.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources

# Install basic development tools and iptables/ipset
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  git \
  gnupg \
  sudo \
  zsh \
  jq \
  vim \
  python3 \
  python3-pip \
  python3-venv \
  bubblewrap \
  && chmod 4755 /usr/bin/bwrap \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 during image build so downstream feature setup can reuse
# Docker's layer cache instead of reinstalling Node on each container rebuild.
RUN mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
  && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
    > /etc/apt/sources.list.d/nodesource.list \
  && apt-get update \
  && apt-get install -y --no-install-recommends nodejs \
  && corepack enable \
  && node --version \
  && npm --version \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# npm/pnpm/yarn/corepack
ENV NPM_CONFIG_REGISTRY=https://registry.npmmirror.com/

# pip source mirror switch
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

# uv source mirror switch
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# install uv
# RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pip3 install --no-cache-dir uv --break-system-packages
