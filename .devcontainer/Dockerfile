FROM mcr.microsoft.com/devcontainers/base:resolute

# Install basic development tools and iptables/ipset
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  curl \
  git \
  sudo \
  zsh \
  jq \
  vim \
  python3 \
  python3-pip \
  python3-venv \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
