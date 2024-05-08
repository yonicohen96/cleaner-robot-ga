import multiprocessing
import time
from typing import Callable
import timeout_decorator
import tqdm

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
@timeout_decorator.timeout(5)
def bar(sleep, to_print_1):
    for i in range(100):
        print(to_print_1)
        time.sleep(sleep)


if __name__ == '__main__':
    for i in tqdm.tqdm(range(10), desc="d"):

        time.sleep(1)
