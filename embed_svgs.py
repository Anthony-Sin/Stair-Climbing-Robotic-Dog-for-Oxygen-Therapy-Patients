import os
import re

with open('poster/index_new.html', 'r', encoding='utf-8') as f:
    html = f.read()

with open('poster/svgs_dump.html', 'r', encoding='utf-8') as f:
    svgs = f.read()

# Replace the svg-container contents
html = re.sub(r'<div class="svg-container">.*?</div>', f'<div class="svg-container">\n{svgs}\n</div>', html, flags=re.DOTALL)
html = re.sub(r'<script>.*fetch.*?</script>', '', html, flags=re.DOTALL)

with open('poster/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Done!')
