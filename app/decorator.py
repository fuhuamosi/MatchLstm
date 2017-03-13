# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import time


# 计算一个函数的运行时间
def exe_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        back = func(*args, **kwargs)
        end = time.time()
        print("Function {0} cost {1}s".format(func.__name__, end - start))
        return back

    return wrapper
