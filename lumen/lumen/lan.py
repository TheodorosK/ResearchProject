"""Opening Lumen to phones and tablets on your home Wi-Fi, behind a PIN.

Off by default. When on, a second server listens on all network interfaces (port + 1), and every
request that doesn't come from this computer must carry a session cookie obtained with the PIN.
Remote devices can browse, search and organise, but can't browse the computer's folders, delete
files or change settings.
"""
import hashlib
import hmac
import secrets
import socket
import threading
import time

from . import db

SESSION_DAYS = 30
MAX_ATTEMPTS = 5            # wrong PINs per address...
LOCKOUT_SECONDS = 300       # ...before a pause
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}

# Paths remote devices may open without logging in (the login page and what it needs).
PUBLIC = ("/login.html", "/api/login", "/app.css", "/favicon.ico")
# Things only the person at the computer may do.
LOCAL_ONLY = (
    ("GET", "/api/fs"), ("POST", "/api/folders"), ("DELETE", "/api/folders"), ("POST", "/api/photos/trash"),
    ("POST", "/api/scan"), ("PUT", "/api/settings"), ("POST", "/api/lan"), ("GET", "/api/lan"),
    ("POST", "/api/index"), ("POST", "/api/autostart"),
)
LOCAL_ONLY_SUFFIXES = ("/reveal", "/open", "/copy")

_attempts = {}
_server = {"thread": None, "server": None, "port": None}
_lock = threading.Lock()


def is_remote(host: str | None) -> bool:
    return (host or "") not in LOCAL_HOSTS


def local_only(method: str, path: str) -> bool:
    return any(method == m and path.startswith(p) for m, p in LOCAL_ONLY) or \
        any(path.endswith(s) for s in LOCAL_ONLY_SUFFIXES)


_ip_cache = {"ip": None, "at": 0.0}


def allowed_host(host_header: str | None) -> bool:
    """Only answer requests addressed to this computer by IP or as localhost. This blocks DNS
    rebinding, where a web page's own domain is pointed at 127.0.0.1 to reach local apps."""
    host = (host_header or "").strip().lower()
    if host.startswith("["):
        host = host[1 : host.find("]")] if "]" in host else host
    else:
        host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    if host in ("127.0.0.1", "localhost", "::1", "testserver"):
        return True
    if time.time() - _ip_cache["at"] > 60:
        _ip_cache.update(ip=lan_ip(), at=time.time())
    return host == _ip_cache["ip"]


def lan_ip() -> str:
    """This computer's address on the local network (no traffic is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _secret() -> str:
    sec = db.get_setting("session_secret")
    if not sec:
        sec = secrets.token_hex(32)
        db.set_setting("session_secret", sec)
    return sec


def _hash(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 200_000).hex()


def set_pin(pin: str | None = None) -> str:
    """Set (or generate) the PIN. Returns it so it can be shown once."""
    pin = pin or f"{secrets.randbelow(10**6):06d}"
    if not (pin.isdigit() and 4 <= len(pin) <= 12):
        raise ValueError("The PIN must be 4 to 12 digits")
    salt = secrets.token_hex(16)
    db.set_setting("pin_hash", f"{salt}:{_hash(pin, salt)}")
    return pin


def pin_is_set() -> bool:
    return bool(db.get_setting("pin_hash"))


def check_pin(pin: str, client: str) -> bool:
    now = time.time()
    recent = [t for t in _attempts.get(client, []) if now - t < LOCKOUT_SECONDS]
    _attempts[client] = recent
    if len(recent) >= MAX_ATTEMPTS:
        raise PermissionError(f"Too many tries. Wait {int(LOCKOUT_SECONDS - (now - recent[0]))} seconds.")
    stored = db.get_setting("pin_hash") or ""
    if ":" in stored:
        salt, h = stored.split(":", 1)
        if hmac.compare_digest(_hash(pin or "", salt), h):
            _attempts.pop(client, None)
            return True
    recent.append(now)
    return False


def make_session() -> str:
    expiry = int(time.time()) + SESSION_DAYS * 86400
    # Tied to the current PIN: changing the PIN signs every device out.
    sig = hmac.new(_secret().encode(), f"{expiry}:{db.get_setting('pin_hash')}".encode(), "sha256").hexdigest()
    return f"{expiry}.{sig}"


def valid_session(token: str | None) -> bool:
    if not token or "." not in token:
        return False
    expiry, sig = token.split(".", 1)
    if not expiry.isdigit() or int(expiry) < time.time():
        return False
    good = hmac.new(_secret().encode(), f"{expiry}:{db.get_setting('pin_hash')}".encode(), "sha256").hexdigest()
    return hmac.compare_digest(sig, good)


def status(port: int | None):
    running = _server["thread"] is not None and _server["thread"].is_alive()
    p = _server["port"] or ((port or 8765) + 1)
    return {"enabled": db.get_setting("lan") == "1", "running": running, "port": p,
            "url": f"http://{lan_ip()}:{p}", "pin_set": pin_is_set()}


def start(app, port: int):
    """Serve the same app on all interfaces, in a background thread."""
    import uvicorn

    with _lock:
        if _server["thread"] is not None and _server["thread"].is_alive():
            return
        server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning", lifespan="off"))
        t = threading.Thread(target=server.run, name="lumen-lan", daemon=True)
        _server.update(thread=t, server=server, port=port)
        t.start()


def stop():
    with _lock:
        if _server["server"] is not None:
            _server["server"].should_exit = True
        _server.update(thread=None, server=None)
