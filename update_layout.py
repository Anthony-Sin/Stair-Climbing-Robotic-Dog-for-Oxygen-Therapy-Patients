import os
with open('poster/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

gallery_card = '''
  gallery: { title:'Visual Results & Training', color:'teal', eyebrow:'GALLERY', grow:true, body:(
    <>
      <div style={{display:'flex', flexWrap:'wrap', gap:'10px'}}>
        <div className="fig-wrap" style={{flex:'1 1 45%'}}><img src="assets/climb_pen_cinematic.png" style={{width:'100%', borderRadius:'6px'}} /></div>
        <div className="fig-wrap" style={{flex:'1 1 45%'}}><img src="assets/stairs_climb.png" style={{width:'100%', borderRadius:'6px'}} /></div>
        <div className="fig-wrap" style={{flex:'1 1 45%'}}><img src="assets/patient_carry.png" style={{width:'100%', borderRadius:'6px'}} /></div>
        <div className="fig-wrap" style={{flex:'1 1 45%'}}><img src="assets/real_go2.png" style={{width:'100%', borderRadius:'6px'}} /></div>
      </div>
      <div className="fig" style={{marginTop:'20px'}}>
        <div className="pchart2" dangerouslySetInnerHTML={{__html: LEARNING_SVG + CURRICULUM_SVG}} />
        <div className="cap">Blind-RL retrain — 6,000 iterations, Go2 + 2.22 kg tank.</div>
      </div>
    </>
  )},
'''

content = content.replace('  conclusion: { title', gallery_card + '  conclusion: { title')

old_layout = '''const DEFAULT_LAYOUT = { columns:[
  { id:'col1', widthMm:250,  cards:['hero','motivation','hardware'] },
  { id:'col2', widthMm:null, cards:['system','approach','methods'] },
  { id:'col3', widthMm:300,  cards:['results','training','limits','conclusion'] },
]};'''

new_layout = '''const DEFAULT_LAYOUT = { columns:[
  { id:'col1', widthMm:250,  cards:['hero', 'motivation','hardware'] },
  { id:'col2', widthMm:null, cards:['gallery'] },
  { id:'col3', widthMm:300,  cards:['system', 'approach','results','limits','conclusion'] },
]};'''

content = content.replace(old_layout, new_layout)

with open('poster/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated index.html successfully')
