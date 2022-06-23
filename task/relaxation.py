from database import get_task_id, insert_and_execute_task, prepare_task
import sys
import numpy as np

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

task_id = get_task_id("gp_minimum", iteration)
gp_minimum = np.loadtxt("work/%d/gp_minimum.dat" % task_id)
max = config["max"]

for index, row in enumerate(gp_minimum):
    if index > max:
        break
    v_real = [x * 180 + 180 for x in row]
    task_id_relax = insert_and_execute_task(
        "dft_relax", iteration, key=index + 1, config={"vector": v_real})
