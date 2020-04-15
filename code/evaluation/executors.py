from concurrent.futures import Future, Executor, ProcessPoolExecutor, ThreadPoolExecutor
from threading import Lock
import os


class DummyExecutor(Executor):
    def __init__(self):
        self._shutdown = False
        self._shutdownLock = Lock()

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")

            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)

            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


def get_executor():
    # return ThreadPoolExecutor(os.cpu_count())
    # return DummyExecutor()
    # return ProcessPoolExecutor(int(os.cpu_count()*0.75))
    return ProcessPoolExecutor(4)


def get_nested_executor():
    return ThreadPoolExecutor()
