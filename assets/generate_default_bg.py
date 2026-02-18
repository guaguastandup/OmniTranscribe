#!/usr/bin/env python3
"""Generate default background image for video generation."""

from PIL import Image, ImageDraw, ImageFont
import os

# Create a beautiful gradient background
width, height = 1920, 1080
img = Image.new('RGB', (width, height))

# Create a gradient from top to bottom
draw = ImageDraw.Draw(img)

# Create a nice gradient from dark blue to purple
for y in range(height):
    # Calculate color based on y position
    r = int(30 + (60 - 30) * y / height)
    g = int(30 + (40 - 30) * y / height)
    b = int(80 + (120 - 80) * y / height)
    draw.rectangle([(0, y), (width, y + 1)], fill=(r, g, b))

# Add some decorative circles
for i in range(5):
    x = int(width * (0.2 + i * 0.15))
    y = int(height * 0.3)
    radius = int(100 + i * 30)
    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)],
                 outline=(255, 255, 255, 30), width=2)

# Add text
try:
    # Try to use a nice font
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
except:
    font = ImageFont.load_default()

text = "OmniTranscribe"
# Get text size
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]
text_height = bbox[3] - bbox[1]

# Draw text centered
x = (width - text_width) // 2
y = (height - text_height) // 2
draw.text((x, y), text, fill=(255, 255, 255, 200), font=font)

# Save the image
output_path = os.path.join(os.path.dirname(__file__), "default_background.png")
img.save(output_path, "PNG")
print(f"Default background created: {output_path}")
