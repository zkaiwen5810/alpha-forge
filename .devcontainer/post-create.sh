#!/usr/bin/env bash
set -euo pipefail

# Fix permissions for mounted volumes
sudo chown -R "$(id -u)":"$(id -g)" \
  "$HOME/.codex" \
  "$HOME/.claude" \
  "$HOME/.config" \
  "$HOME/.cache" \
  "$HOME/.persist"

# Ensure persist directory exists
PERSIST_DIR="$HOME/.persist"
mkdir -p "$PERSIST_DIR"

# Install a dedicated Codex launcher for cases where the container itself is the
# intended isolation boundary and nested bwrap sandboxing is not available.
LOCAL_BIN_DIR="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN_DIR"
cat > "$LOCAL_BIN_DIR/codex-dfa" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

exec /usr/local/bin/codex --sandbox danger-full-access "$@"
EOF
chmod 0755 "$LOCAL_BIN_DIR/codex-dfa"

# Claude root file persistence via symlink
PERSIST_FILE="$PERSIST_DIR/.claude.json"
TARGET_FILE="$HOME/.claude.json"

# Create file if not exists
if [ ! -f "$PERSIST_FILE" ]; then
  printf '{}\n' > "$PERSIST_FILE"
fi

# Create symlink: ~/.claude.json -> ~/.persist/.claude.json
if [ ! -L "$TARGET_FILE" ]; then
  rm -f "$TARGET_FILE"
  ln -s "$PERSIST_FILE" "$TARGET_FILE"
fi
