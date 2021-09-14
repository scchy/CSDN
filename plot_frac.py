# python3
# Create date: 2021-09-14
# Author: Scc_hy
# Func: plor frac
# =================================================================================================


import turtle
from decimal import Decimal, getcontext
getcontext().prec = 10000
import time
import numpy as np


class PlotFrac():
    """
    turtle 简单绘制可以参考：https://www.cnblogs.com/nowgood/p/turtle.html
    """
    def __init__(self, forward_len=15, angle_scale=10, plot_len=100):
        self.forward_len = forward_len
        self.angle_scale = angle_scale
        self.plot_len = plot_len

    def plot_angle(self, ang):
        """
        基于小数点第n位大小，换算成角度
        顺时针转角度，然后绘制直线
        """
        turtle.right(ang * self.angle_scale)
        turtle.forward(self.forward_len)
    
    def plot_frac(self, frac):
        """
        绘制除不尽小数 或 无理数
        """
        frac_plot = str(frac).split('.')[1]
        if len(frac_plot) <= 900:
            return '可除尽小数，不绘制'

        for i in range(self.plot_len):
            turtle.speed(10)
            ang_i = int(frac_plot[i])
            self.plot_angle(ang_i)

        turtle.penup()
        time.sleep(5)
        # turtle.mainloop()
        turtle.bye()



if __name__ == '__main__':
    p = PlotFrac(plot_len=1000)
    for i in range(17, 20):
        loop = True
        while loop:
            try:
                print(f'plot 1 / {i}')
                frac = Decimal(1) / Decimal(i)
                p.plot_frac(frac)
                loop = False
            except Exception as e:
                continue       
    
    # 无理数集合
    l_ = [Decimal(np.pi), Decimal(np.e), ( Decimal(np.sqrt(5)) - Decimal(1))/ 2]
    p = PlotFrac(plot_len=9000)

    for i in l_:
        loop = True
        while loop:
            try:
                print(f'plot 1 / {i}')
                frac = Decimal(1) / Decimal(i)
                p.plot_frac(frac)
                loop = False
            except Exception as e:
                continue       
    