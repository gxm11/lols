from database import run_task, prepare_task
import sys

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

task_id_gp_model = run_task("gp_model", iteration)
task_id_gp_minimum = run_task("gp_minimum", iteration)
