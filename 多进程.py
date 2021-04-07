# python3
# Create date: 2021-04-08
# Authore: Scc_hy
# Func: 多进程运行性能比较



from datetime import datetime 
import numpy as np
from functools import reduce
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from contextlib import contextmanager
import time

@contextmanager
def timer(title):
    st = time.perf_counter()
    yield
    print(f'{title} - done in {time.perf_counter()-st:.2f}s')


class SomeTry:
    def C(self, m, n):
        return self.reduce_multi(n) / self.reduce_multi(m) 

    def C_tuple(self, m_n_tuple):
        n, m = m_n_tuple
        return self.reduce_multi(n) / self.reduce_multi(m) 


    def A(self, m, n):
        return self.reduce_multi(n) / self.reduce_multi(n-m) 

    def reduce_multi(self, x):
        return reduce(lambda x1, x2: x1*x2, range(1, x+1))
    
    @classmethod
    def C_for_parallel(cls, tuple_flag = False):
        """
        新加载类固化method -> function
        """
        t = cls()
        if tuple_flag:
            return t.C_tuple
        return t.C

    @staticmethod
    def parallel_func(func_, param_iterator):
        worker = os.cpu_count() * 3 // 5
        executor = ProcessPoolExecutor(worker)
        with timer(f'{func_.__name__} Parallel run.'):
            task_list = [executor.submit(func_, *parami) for parami in param_iterator]
            res = [task.result() for task in as_completed(task_list)]
        return res

    @staticmethod
    def parallel_func_map(func_, param_iterator):
        worker = os.cpu_count() * 3 // 5
        with timer(f'{func_.__name__} Parallel run.'):
            with ProcessPoolExecutor(worker) as executor:
                res = executor.map(func_, param_iterator)
        return res

    def parallel_C(self, m_list, n_list):
        func_c = self.C_for_parallel(tuple_flag=False)
        func_tuple = self.C_for_parallel(tuple_flag=True)
        param_iterator = zip(m_list, n_list)
    
        res_map = self.parallel_func_map(self.C_tuple, param_iterator)
        res_task = self.parallel_func(self.C, param_iterator)

        res_map_func = self.parallel_func_map(func_tuple, param_iterator)
        res_task_func = self.parallel_func(func_c, param_iterator)
        return list(res_map), list(res_task), list(res_map_func), list(res_task_func)


if __name__ == "__main__":
    s = SomeTry()
    a, b, c, d = s.parallel_C(list(range(1, 5000)), list(range(500, 5500)))
    # print(a)
