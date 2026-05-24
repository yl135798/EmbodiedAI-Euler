# -*- coding: utf-8 -*-
"""修复 domain_adapter.py 中的 bug: ic_result → logic_result"""
import sys

file_path = r"C:\Users\A\.qclaw\workspace\DataAgent\cognition\domain_adapter.py"

# 1. 读取文件
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 2. 统计 bug 数量
bug_count = content.count("blend(ic_result,")
print(f"[修复] 发现 {bug_count} 处 bug (ic_result)")

# 3. 修复 bug
content = content.replace("blend(ic_result, emotion_result)", "blend(logic_result, emotion_result)")

# 4. 验证修复
fixed_count = content.count("blend(logic_result,")
remaining_bugs = content.count("ic_result")
print(f"[验证] 修复后 logic_result 出现 {fixed_count} 次")
print(f"[验证] 剩余 ic_result 数量: {remaining_bugs}")

# 5. 写回文件
with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"[OK] Bug 修复完成: {bug_count} 处 ic_result → logic_result")
