from database import insert_and_execute_task, prepare_task
import sys

# preparing folder and config.json
id = int(sys.argv[1])
folder = "work/%d" % id
iteration, key, config = prepare_task(id)

task_id_gp_model = insert_and_execute_task("gp_model", iteration)
task_id_gp_minimum = insert_and_execute_task("gp_minimum", iteration)
