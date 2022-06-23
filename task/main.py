from database import prepare_task, insert_and_execute_task
import numpy as np
import sys
import model.log as log

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

initial = np.loadtxt("lib/material/%s/initial.dat" % config["material"])
args = np.argsort(initial[:, -1])[0:config["initial_number"]]
initial = initial[args]
initial[:, 0:-1] = initial[:, 0:-1] / 180.0 - 1

np.savetxt("work/%d/initial.dat" % id, initial)

# run
for iter in range(1, config["max_iteration"] + 1):
    log.title("< Date Generation : %d >" % iter)
    insert_and_execute_task("data_generation", iter)

    if iter % config["energy_model_interval"] == 0:
        log.title("< Energy Model : %d >" % iter)
        insert_and_execute_task("energy_model", iter)

        log.title("< Relaxation : %d >" % iter)
        insert_and_execute_task("relaxation", iter)
