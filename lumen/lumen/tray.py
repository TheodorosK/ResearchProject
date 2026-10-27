"""A menu-bar / system-tray icon for the desktop app: Open Lumen, Quit."""
import webbrowser

from PIL import Image, ImageDraw


def icon_image(size=64):
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    for y in range(size):  # the brand gradient, top-left orange to bottom-right blue
        t = y / size
        d.line([(0, y), (size, y)], fill=(int(255 - 131 * t), int(180 - 24 * t), int(84 + 171 * t), 255))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=size // 4, fill=255)
    im.putalpha(mask)
    r = size * 0.19
    c = size / 2
    d.ellipse([c - r, c - r, c + r, c + r], outline=(255, 255, 255, 255), width=max(2, size // 12))
    return im


def run(url: str, on_quit):
    """Blocks in the tray's event loop (must be the main thread on macOS)."""
    import pystray

    def open_lumen(icon, item):
        webbrowser.open(url)

    def quit_lumen(icon, item):
        icon.stop()
        on_quit()

    icon = pystray.Icon("Lumen", icon_image(), "Lumen", menu=pystray.Menu(
        pystray.MenuItem("Open Lumen", open_lumen, default=True),
        pystray.MenuItem("Quit", quit_lumen),
    ))
    icon.run()
