

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
    print(f'{title} - done in {time.perf_counter()-st:.5f}s')


class SomeTry:
    def C(self, m, n):
        return self.reduce_multi(n) / self.reduce_multi(m) 

    def C_tuple(self, m_n_tuple):
        n, m = m_n_tuple
        return self.reduce_multi(n) / self.reduce_multi(m) 

    def C_next(self, m, n):
        out = self.C(m, n)
        if out > 10**100:
            return out
        return return self.A(8, 100)

    def C_tuple_next(self, m_n_tuple):
        out = self.C_tuple(m_n_tuple)
        if out > 10**100:
            return out
        return return self.A(8, 100)

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
            return t.C_tuple_next
        return t.C_next

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
        res_map, res_task, res_map_func, res_task_func = '', '', '', ''
        res_map = self.parallel_func_map(self.C_tuple_next, zip(m_list, n_list))
        res_task = self.parallel_func(self.C_next, zip(m_list, n_list))
        print('--'*25)
        print('use function')
        res_map_func = self.parallel_func_map(func_tuple, zip(m_list, n_list))
        res_task_func = self.parallel_func(func_c, zip(m_list, n_list))
        return list(res_map), list(res_task), list(res_map_func), list(res_task_func)

    def parallel_C2(self, m_list, n_list):
        func_c = self.C_for_parallel(tuple_flag=False)
        func_tuple = self.C_for_parallel(tuple_flag=True)
        res_map, res_task, res_map_func, res_task_func = '', '', '', ''
        worker = os.cpu_count() * 3 // 5
        with timer('C_tuple Parallel run.'):
            with ProcessPoolExecutor(worker) as executor:
                res_map = executor.map(self.C_tuple_next, zip(m_list, n_list))

        executor = ProcessPoolExecutor(worker)
        with timer(f'C Parallel run.'):
            task_list = [executor.submit(self.C_next, *parami) for parami in  zip(m_list, n_list)]
            res_task = [task.result() for task in as_completed(task_list)]

        print('--'*25)
        print('use function')
        res_map_func = self.parallel_func_map(func_tuple, zip(m_list, n_list))
        res_task_func = self.parallel_func(func_c, zip(m_list, n_list))
        return list(res_map), list(res_task), list(res_map_func), list(res_task_func)



if __name__ == "__main__":
    s = SomeTry()
    a, b, c, d = s.parallel_C(list(range(1, 100))*120, list(range(50, 150))*120)
    print('\n','**'*25)
    a, b, c, d = s.parallel_C2(list(range(1, 100))*120, list(range(50, 150))*120)
    # print(a, c)

"""
C_tuple_next Parallel run. - done in 14.85585s
C_next Parallel run. - done in 15.34117s
--------------------------------------------------
use function
C_tuple_next Parallel run. - done in 16.36587s
C_next Parallel run. - done in 16.27177s

 **************************************************
C_tuple Parallel run. - done in 16.23173s
C Parallel run. - done in 15.66818s
--------------------------------------------------
use function
C_tuple_next Parallel run. - done in 16.43504s
C_next Parallel run. - done in 15.63722s
"""
