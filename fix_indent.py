#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fix all indentation issues in cla.py"""

with open('cla.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = False

# Fix line 1167 (index 1166) - return statement after else
if len(lines) > 1166:
    current = lines[1166]
    if current.startswith('        return jsonify({'):  # 8 spaces - wrong, should be 12
        # Fix the return and all lines in the jsonify block
        for i in range(1166, min(1181, len(lines))):
            line = lines[i]
            if line.startswith('        ') and not line.startswith('            '):
                lines[i] = '            ' + line[8:]
                fixed = True
        print(f"Fixed return statement indentation around line 1167")

# Write back
if fixed:
    with open('cla.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Done fixing indentation!")
else:
    print("No fixes needed or file structure different than expected")
