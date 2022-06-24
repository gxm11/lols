from database import run_task, create_database
import os

if __name__ == "__main__":
    if not os.path.exists("work"):
        os.mkdir("work")

    if not os.path.exists("lols.db"):
        create_database()

    run_task("main", 0)
