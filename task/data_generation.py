from database import prepare_task, run_task, get_task_id
import numpy as np
import json
import sys

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

if iteration == 1:
    task_id = get_task_id("main", 0)
    data = np.loadtxt("work/%d/initial.dat" % task_id)
else:
    task_id = get_task_id("data_generation", iteration - 1)
    data_pool = np.loadtxt("work/%d/data_pool.dat" % task_id)
    new_data = np.loadtxt("work/%d/new_data.dat" % task_id)
    data = np.vstack((data_pool, new_data))

np.savetxt("%s/data_pool.dat" % folder, data)

task_id_vae = run_task("train_vae", iteration)
task_id_sample = run_task("sample", iteration)

recon = np.loadtxt("work/%d/recon.dat" % task_id_sample)
new_data = []
for index, row in enumerate(recon):
    v_real = [x * 180 + 180 for x in row]
    task_id_energy = run_task(
        "dft_energy", iteration, key=index + 1, config={"vector": v_real})
    result = json.load(open("work/%d/result.json" % task_id_energy))
    if result["energy"] is not None:
        v = [x for x in row]
        new_data.append(v + [result["energy"]])
new_data = np.array(new_data)
np.savetxt("%s/new_data.dat" % folder, new_data)
