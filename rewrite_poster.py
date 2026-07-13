import os
import re

with open('poster/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

def extract_card(card_id):
    pattern = rf"{card_id}: \{{\s*title:'(.*?)',\s*color:'.*?',\s*eyebrow:'(.*?)'.*?body:\((.*?)\)\}},"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        title, eyebrow, body = match.groups()
        body = body.strip()
        if body.startswith('<>'): body = body[2:]
        if body.endswith('</>'): body = body[:-3]
        body = body.replace('className=', 'class=')
        body = re.sub(r'style=\{\{.*?\}\}', '', body)
        # Fix missing chart titles if any
        body = body.replace('class="pchart2" dangerouslySetInnerHTML={{__html: LEARNING_SVG + CURRICULUM_SVG}} /', '')
        return {'title': title, 'eyebrow': eyebrow, 'body': body}
    return {'title': '', 'eyebrow': '', 'body': ''}

cards = {}
for cid in ['motivation', 'hardware', 'system', 'approach', 'results', 'limits', 'conclusion']:
    cards[cid] = extract_card(cid)

html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
<title>Patient-assisting Quadruped Robot</title>
<style>
  :root {{
    --bg: #f9f9f9;
    --text: #222;
    --border: #ccc;
    --accent: #2c5282;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 40px;
    font-family: 'Open Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
  }}
  .poster-container {{
    max-width: 3456px;
    margin: 0 auto;
    background: #fff;
    padding: 60px;
    border: 1px solid #ddd;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
  }}
  .header {{
    display: flex;
    flex-direction: column;
    align-items: center;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 30px;
    margin-bottom: 40px;
    text-align: center;
  }}
  .header h1 {{
    font-family: 'Playfair Display', serif;
    font-size: 80px;
    margin: 0 0 20px 0;
    color: var(--text);
  }}
  .authors {{
    font-size: 32px;
    font-weight: 600;
    color: #444;
  }}
  .org {{
    font-size: 28px;
    color: #666;
    margin-top: 10px;
  }}
  
  .columns {{
    display: flex;
    gap: 40px;
  }}
  .col {{
    display: flex;
    flex-direction: column;
    gap: 30px;
  }}
  .col-1, .col-3 {{
    flex: 1;
  }}
  .col-2 {{
    flex: 1.5;
  }}
  
  .card {{
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 30px;
  }}
  .card h2 {{
    font-family: 'Open Sans', sans-serif;
    font-size: 40px;
    margin-top: 0;
    margin-bottom: 20px;
    text-align: center;
    color: var(--text);
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
  }}
  
  .card p, .card li {{
    font-size: 20px;
    line-height: 1.6;
  }}
  
  .fig-wrap img {{
    width: 100%;
    border-radius: 8px;
  }}
  .image-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 30px;
  }}
  .image-grid img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #ddd;
  }}
  
  .svg-container {{
    display: flex;
    justify-content: space-around;
    align-items: flex-end;
    margin-top: 30px;
  }}
  .svg-container svg {{
    width: 45%;
    height: auto;
  }}
  
</style>
</head>
<body>
<div class="poster-container">
  <div class="header">
    <h1>Stair-Climbing Robotic Dog for Oxygen Therapy Patients</h1>
    <div class="authors">Anthony Sinchi, Tasnia Simin, Khanh Hung Truong</div>
    <div class="org">USF College of Engineering • Team Robodog (Project #6)</div>
  </div>
  
  <div class="columns">
    <div class="col col-1">
      <div class="card">
        <h2>{cards['motivation']['title']}</h2>
        {cards['motivation']['body']}
      </div>
      <div class="card">
        <h2>{cards['hardware']['title']}</h2>
        {cards['hardware']['body']}
      </div>
      <div class="card">
        <h2>{cards['system']['title']}</h2>
        {cards['system']['body']}
      </div>
    </div>
    
    <div class="col col-2">
      <div class="card" style="border: 2px solid var(--accent);">
        <h2>Results and Discussions</h2>
        
        <div class="image-grid">
          <img src="assets/climb_pen_cinematic.png" alt="Climbing View" />
          <img src="assets/stairs_climb.png" alt="Stairs Climb" />
          <img src="assets/patient_carry.png" alt="Patient Carry" />
          <img src="assets/real_go2.png" alt="Real Go2 Robot" />
        </div>
        
        <p style="text-align: center; font-weight: 600; font-size: 24px; margin: 40px 0 20px 0;">Training Progress</p>
        
        <div class="svg-container">
          <!-- We will inject the SVGs here -->
          <div id="learning-svg"></div>
          <div id="curriculum-svg"></div>
        </div>
        <p style="text-align: center; margin-top: 20px; font-style: italic; color: #555;">
          Blind-RL retrain — 6,000 iterations, Go2 + 2.22 kg tank. Mean reward ~ 95.7, terrain curriculum plateaued at 138 mm riser.
        </p>
      </div>
    </div>
    
    <div class="col col-3">
      <div class="card">
        <h2>{cards['approach']['title']}</h2>
        {cards['approach']['body']}
      </div>
      <div class="card">
        <h2>{cards['results']['title']}</h2>
        {cards['results']['body']}
      </div>
      <div class="card">
        <h2>Conclusions & Limitations</h2>
        {cards['limits']['body']}
        {cards['conclusion']['body']}
      </div>
    </div>
  </div>
</div>

<script>
  // Inject the SVGs
  fetch('learning.svg').then(r => r.text()).then(html => document.getElementById('learning-svg').innerHTML = html);
  fetch('curriculum.svg').then(r => r.text()).then(html => document.getElementById('curriculum-svg').innerHTML = html);
</script>

</body>
</html>
'''

with open('poster/index_new.html', 'w', encoding='utf-8') as f:
    f.write(html)
print('Generated static poster!')
