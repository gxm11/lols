from database import get_task_id, prepare_task
import sys
import model.boss as boss
import numpy as np
import json
import os

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

task_id = get_task_id("data_generation", iteration)
data_pool = np.loadtxt("work/%d/data_pool.dat" % task_id)
np.savetxt("%s/gp_data.dat" % folder, data_pool)

boss.setup(config["config"], data_pool, folder)

# after modify the boss source code, the s option can be used.
# or you should run the o mode instead.
if os.system("boss -v | grep -q 'boss s'") == 0:
    boss.run(folder, config["boss_execute"])
else:
    boss.run(folder, config["boss_execute"], "o")

hypers = boss.hyper_parameters(folder)

with open("%s/gp_hypers.json" % folder, "w") as f:
    json.dump(hypers["gp_parameters"], f)
