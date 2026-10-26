"""Draw the app icon from simple shapes; no external assets."""
from pathlib import Path
from PIL import Image, ImageDraw

base = Path(__file__).resolve().parent.parent / "assets"
base.mkdir(exist_ok=True)
image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)
draw.rounded_rectangle((8, 8, 248, 248), radius=54, fill="#16243c")
draw.rounded_rectangle((48, 42, 204, 213), radius=14, fill="#f6f8ff")
for box, color in [((59, 53, 120, 123), "#599ce5"), ((131, 53, 193, 123), "#edc16b"),
                   ((59, 134, 120, 201), "#74b7a8"), ((131, 134, 193, 201), "#d58da0")]:
    draw.rounded_rectangle(box, radius=6, fill=color)
draw.line((86, 186, 181, 75), fill="#16243c", width=14)
draw.ellipse((172, 65, 193, 86), fill="#16243c")
image.save(base / "app.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
image.save(base / "app.png")
