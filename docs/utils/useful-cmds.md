# Kill anything using ComfyUI's default port

## Windows PowerShell

```
Get-NetTCPConnection -LocalPort 8188 -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique |
  ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }
```

## Linux/macOS

```
lsof -ti tcp:8188 | xargs -r kill -9
```

If `xargs -r` is not available on macOS, use:

```
pids="$(lsof -ti tcp:8188)"; [ -n "$pids" ] && kill -9 $pids
```


## Nuke Codex VS Code Extension
```
# 1. Fully close VS Code
Stop-Process -Name Code -Force -ErrorAction SilentlyContinue

# 2. See exact OpenAI/Codex extension IDs installed
code --list-extensions | findstr /i "openai codex chatgpt"

# 3. Uninstall likely OpenAI/Codex extension IDs
code --uninstall-extension openai.chatgpt
code --uninstall-extension openai.codex

# 4. Delete leftover extension folders
Remove-Item "$env:USERPROFILE\.vscode\extensions\openai.*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.vscode\extensions\*codex*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\.vscode\extensions\*chatgpt*" -Recurse -Force -ErrorAction SilentlyContinue

# 5. Delete Codex/OpenAI VS Code persisted state
Remove-Item "$env:APPDATA\Code\User\globalStorage\openai.*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\User\globalStorage\*codex*" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\User\globalStorage\*chatgpt*" -Recurse -Force -ErrorAction SilentlyContinue

# 6. Delete VS Code caches that often preserve broken extension webview/login state
Remove-Item "$env:APPDATA\Code\Cache" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\CachedData" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\Code Cache" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\GPUCache" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\Service Worker" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item "$env:APPDATA\Code\CachedExtensionVSIXs" -Recurse -Force -ErrorAction SilentlyContinue

# 7. Optional: nuke per-workspace extension state
# This removes VS Code workspace UI/session state, not your code.
Remove-Item "$env:APPDATA\Code\User\workspaceStorage" -Recurse -Force -ErrorAction SilentlyContinue
```

## Reisntall Codex VS Code Extension
```
code --install-extension openai.chatgpt
```

# Start and kill AI-toolkit UI
## Kill running ai-toolkit process

```
tmux kill-session -t aitk_ui
```

## Spin up ai-tookit UI
```
tmux new -d -s aitk_ui 'export PATH=/workspace/bin:$PATH; cd /workspace/ai-toolkit/ui && npm run start'
```