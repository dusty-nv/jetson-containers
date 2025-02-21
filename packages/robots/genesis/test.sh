#!/bin/bash
python3 - <<EOF
print('testing genesis...')

import genesis as gs
gs.init(backend=gs.cuda)

scene = gs.Scene(show_viewer=False)
plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

scene.build()

for i in range(1000):
    scene.step()

print('genesis OK\\n')
EOF
