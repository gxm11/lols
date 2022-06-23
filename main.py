from database import insert_and_execute_task, create_database
import os

if __name__ == "__main__":    
    if not os.path.exists("work"):
        os.mkdir("work")

    if not os.path.exists("lols.db"):
        create_database()

    insert_and_execute_task("main", 0)
