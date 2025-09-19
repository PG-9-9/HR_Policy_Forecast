# Network Binding Explanation

## Why `0.0.0.0` vs `localhost`?

### **Server Binding (uvicorn --host parameter):**
- `--host 0.0.0.0` = Server listens on ALL network interfaces
- `--host 127.0.0.1` = Server listens only on localhost interface

### **Browser Access:**
- ✅ **Use:** `http://localhost:8000` or `http://127.0.0.1:8000`
- ❌ **Don't use:** `http://0.0.0.0:8000` (this won't work in browsers)

## Why Use `0.0.0.0` for Server Binding?

1. **Docker compatibility** - Containers need to bind to all interfaces
2. **Network access** - Allows access from other machines on network (if needed)
3. **AWS deployment** - Required for container deployments

## Local Development vs Production

### **Local Development:**
```bash
# Server binding
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Browser access
http://localhost:8000
```

### **Docker/AWS Production:**
```bash
# Container binding (same as local)
uvicorn app.main:app --host 0.0.0.0 --port 8000

# External access through load balancer/domain
https://your-domain.com
```

## Quick Reference

| Context | Server Binding | Browser URL |
|---------|---------------|-------------|
| Local Dev | `0.0.0.0:8000` | `localhost:8000` |
| Docker | `0.0.0.0:8000` | `localhost:8000` |
| AWS/Production | `0.0.0.0:8000` | `your-domain.com` |

**Bottom line:** Always use `localhost:8000` for local browser access, regardless of server binding!