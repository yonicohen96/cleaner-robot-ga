import multiprocessing
import time
from typing import Callable



def timeout_function(timeout: int, function: Callable, *args, **kwargs):
    # Start bar as a process
    p = multiprocessing.Process(target=function, args=args, kwargs=kwargs)
    p.start()

    # Wait for 10 seconds or until process finishes
    p.join(timeout)

    # If thread is still active
    if p.is_alive():
        print("running... let's kill it...")

        # Terminate - may not work if process is stuck for good
        p.terminate()
        # OR Kill - will work for sure, no chance for process to finish nicely however
        # p.kill()

        p.join()

# bar
def bar(sleep, to_print_1, to_print_2):
    for i in range(100):
        print(to_print_1)
        print(to_print_2)
        time.sleep(sleep)



if __name__ == '__main__':
    timeout_function(10, bar, 0.5, to_print_2='', to_print_1='1')
