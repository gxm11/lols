import json
import sqlite3
import shutil
import os
import model.log as log


def create_database():
    with sqlite3.connect("lols.db") as db:
        c = db.cursor()
        sql = """
        CREATE TABLE task (
            id INTEGER PRIMARY KEY,
            state,
            iteration,
            key,
            name,
            config
        );
        """
        c.executescript(sql)


def insert_task(name, iteration, key=0, config={}):
    if os.path.exists("config.json"):
        task_config = json.load(open("config.json"))[name]
    else:
        task_config = json.load(open("config-test.json"))[name]
    task_config.update(config)
    with sqlite3.connect("lols.db") as db:
        c = db.cursor()
        sql = """
        SELECT id FROM task WHERE
        name = ? and iteration = ? and key = ?
        """
        c.execute(sql, (name, iteration, key))
        ret = c.fetchone()
        if ret is not None:
            # if state = 0, refresh task config
            sql = """
            UPDATE task SET config = ? 
            WHERE id = ? and state = 0
            """
            c.execute(sql, (json.dumps(task_config), ret[0]))
            return ret[0]

        sql = """
        INSERT INTO task (state, iteration, key, name, config)
        VALUES (0, ?, ?, ?, ?);
        """

        c.execute(sql, (iteration, key, name, json.dumps(task_config)))
        return c.lastrowid


def update_task(id, **d):
    with sqlite3.connect("lols.db") as db:
        c = db.cursor()
        for key, value in d.items():
            sql = "UPDATE task SET %s = ? WHERE id = ?" % key
            c.execute(sql, (value, id))


def select_task(id):
    with sqlite3.connect("lols.db") as db:
        c = db.cursor()
        sql = """
        SELECT id, state, iteration, key, name, config
        FROM task WHERE id = ?
        """
        c.execute(sql, (id,))
        return c.fetchone()


def execute_task(id):
    id, state, iteration, key, name, config = select_task(id)
    log.title("Task [%d] <%s> start" % (id, name))
    log.info(" - iteration: %d, index: %d" % (iteration, key))
    if state >= 2:
        log.info(" - skipped")
        return
    update_task(id, state=1)
    ret = os.system("python task/%s.py %d" % (name, id))
    if ret != 0:
        log.info("Task [%d] <%s> error" % (id, name))
        log.error("Error occurs when running task [%d]" % id)
        raise
    update_task(id, state=2)
    log.info("Task [%d] <%s> finish" % (id, name))


def insert_and_execute_task(name, iteration, key=0, config={}):
    task_id = insert_task(name, iteration, key, config)
    execute_task(task_id)
    return task_id


def prepare_task(id):
    id, state, iteration, key, name, config = select_task(id)
    config = json.loads(config)
    workdir = "work/%d" % id
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.mkdir(workdir)
    with open("%s/config.json" % workdir, "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)
    return iteration, key, config


def get_task_id(name, iteration, key=0):
    with sqlite3.connect("lols.db") as db:
        c = db.cursor()
        sql = """
        SELECT id FROM task WHERE
        name = ? and iteration = ? and key = ?
        """
        c.execute(sql, (name, iteration, key))
        return c.fetchone()[0]
