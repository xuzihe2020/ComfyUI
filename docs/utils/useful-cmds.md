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
