import os
import re

with open('poster/recovered.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

start_idx = 0
for i, l in enumerate(lines):
    if '1: <!DOCTYPE html>' in l:
        start_idx = i
        break

restored_lines = []
for l in lines[start_idx:]:
    if re.match(r'^\d+: ', l):
        restored_lines.append(re.sub(r'^\d+: ', '', l))
    elif 'The above content shows the entire' in l or 'NOTE: The output was truncated' in l:
        break
    else:
        restored_lines.append(l)

restored = '\n'.join(restored_lines)
truncation_str = '<div className="subtitle">An autonomous Unitree Go2 quadruped that follows a patient on supp'
idx = restored.find(truncation_str)
if idx != -1:
    restored = restored[:idx] + '''<div className="subtitle">An autonomous Unitree Go2 quadruped that follows a patient on supplemental oxygen, recognizes stairs, and climbs them — carrying the tank so the patient never has to face a staircase alone.</div>
      </div>
      <div className="header-right">
        <div className="org">USF College of Engineering</div>
        <div className="author">Anthony Sinchi, Tasnia Simin, Khanh Hung Truong</div>
        <div className="tags">
          <span className="tag-pill">UI & Controls</span>
          <span className="tag-pill">Perception</span>
          <span className="tag-pill">Isaac Sim</span>
          <span className="tag-pill">Jetson Deploy</span>
        </div>
        <div className="badge">Team Robodog (Project #6)</div>
        <div className="advisor">Faculty Advisors: Dr. Yu Sun & Dr. William Kearns</div>
      </div>
    </div>
    <div className="poster" id="poster">
      {layout.columns.map(c=>renderColumn(c))}
    </div>
  </>);
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<PosterApp />);
</script>
</body>
</html>'''
    with open('poster/index.html', 'w', encoding='utf-8') as out:
        out.write(restored)
    print(f'Restored original poster and fixed tail from recovered.txt!')
else:
    print('Truncation string not found!')
