# -*- coding: utf-8 -*-
"""Fix demo.py: vibration modality + emoji encoding"""
import sys

filepath = r"C:\Users\A\.qclaw\workspace\DataAgent\demo.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: "vibration" is not a valid modality, change to "tactile"
content = content.replace('"vibration"', '"tactile"')

# Fix 2: replace emojis that break GBK console
replacements = [
    ('\u2705', '[OK]'),
    ('\U0001f4c1', '[DIR]'),
    ('\U0001f4a1', '[TIP]'),
    ('\u26a0\ufe0f', '[WARN]'),
]
for old, new in replacements:
    content = content.replace(old, new)

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed demo.py: vibration modality + emoji encoding")
