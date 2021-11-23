'''
ryPat_Optimization_01.py

3個條件 constraints (c1,c2,c3)
外加 4個 正交 界線 xyBounds
夾出一個 最佳化的搜尋空間
用來 尋找 f() 的極小值
並做 3-d 視覺化
'''
# 
# NonLinear Optimization
# Wonderful!!
#

import numpy as np
import matplotlib.pyplot as pl
import sympy as sm

import scipy.optimize as sopt
from ryPat import ryPlot3d


# 這是 objective function
# 目前本程式處理 2-d 函數 f(s)
# 其中
# s= [x,y]= sm.symbols('x,y')
#
def f(s):
    x, y= s
    
    z= -2**(
        ((x-5)**2 + (y-5)**2)
        /25)
    
    return z

# constraint函數, 必須同時有 x,y 項，
# 允許用 1e-10 當作 近似 0 的係數，
# 但絕大多數時候不需要，因為可以用 xyBounds 達成相同的目的
def c1(s):
    x,y= s
    z= x + y - 1
    #z= x + y*1e-3 +10
    return z

def c2(s):
    x,y= s
    z= -x + 2*y -2
    #z= x*1e-3 +y +10
    
    return z

def c3(s):
    x,y= s
    z= -(x + 3*y -9)
    #z= x + y + 20
    return z

xyBounds= [(-5, 5),  # xmin, xmax
           (-5, 5)   # ymin, ymax
           ]

        
def ryOptimization(
        f= f, 
        c1= c1, 
        c2= c2, 
        c3= c3, 
        xyBounds= xyBounds):

    #
    # Objective function s.t. Constraints
    #
    x,y= sm.symbols('x,y')
    
    print(f'Objective function: f()={f([x,y])}')
    q= f([x,y])
    display(q)
    
    print('Constraints: c() >= 0')
    print(f'c()= {c1([x,y])},{c2([x,y])},{c3([x,y])}')
    
    print('xyBounds:  [xmin, xmax] [ymin, ymax]')
    print(f'{xyBounds}')
    
    q= c1([x,y])
    display(q)
    q= c2([x,y])
    display(q)
    q= c3([x,y])
    display(q)
    
    x0= np.random.random(2)*10-5
    
    opt= sopt.minimize(
        f,
        x0= x0,
        constraints= [
            {'fun':c1, 'type': 'ineq'},
            {'fun':c2, 'type': 'ineq'},
            {'fun':c3, 'type': 'ineq'}        
            ],
        #bounds= [(None, None),
        #         (-3, None)
        #         #(None, -3)]
        bounds= xyBounds                
    )
    print(f'optimum= \n{opt}')
    
    
    #
    # plot objective function
    #
    x,y= sm.symbols('x,y')
    #f([x,y])
    '''
    fg, ax= ryPlotGradientMap(
        f([x,y]), 
        xrange= (-10,10), 
        yrange= (-10,10)
    )
    '''
    fg, ax, _, _, _=  ryPlot3d(
        f([x,y]), 
        xrange= (-10,10), 
        yrange= (-10,10))
    
    
    #
    # plot constraints 
    #
    
    xx= np.linspace(-10,10,1001)
    
    '''
    yy1= 3*xx+6
    yy2= (-xx+4)/2
    yy3= xx*0 -3
    '''
    
    # 上面3行 constraints 
    # 如何自動從 c1, c2, c3 撈出來？？
    
    y1= sm.lambdify(x,
            sm.solve(
                c1([x,y]),
                y)[0]
            )
    y2= sm.lambdify(x,
            sm.solve(
                c2([x,y]),
                y)[0]
            )
    y3= sm.lambdify(x,
            sm.solve(
                c3([x,y]),
                y)[0]
            )
    yy1= y1(xx)
    yy2= y2(xx)
    yy3= y3(xx)
    #-----------------------
    
    ax.plot(xx,yy1,'r',
            xx,yy2,'g',
            xx,yy3,'b',
            linestyle='--'
           )
    #
    # 畫 最佳點 opt.x 
    #
    
    ax.scatter3D(
        xs= [opt.x[0]]*2, 
        ys= [opt.x[1]]*2, 
        zs= [0, opt.fun],
        color= 'magenta',
        marker= 's'    
    )
    
    
    ax.plot3D(
        xs= [opt.x[0]]*2, 
        ys= [opt.x[1]]*2, 
        zs= [0, opt.fun],
        color= 'magenta',
        #marker= '.',
        linestyle= ':'
    )
    
    
    
    #
    # 畫坐標軸 (x軸,y軸) 
    #
    ax.plot(xx,  xx*0,'k--',
            xx*0,xx,  'k--',
           )
    
    #
    #  畫 xyBounds (平行x軸,y軸) 
    #
    
    ax.plot(xx,  xyBounds[1][0]*np.ones_like(xx),'c--',
            xx,  xyBounds[1][1]*np.ones_like(xx),'c--',
            xyBounds[0][0]*np.ones_like(xx), xx, 'c--',
            xyBounds[0][1]*np.ones_like(xx), xx, 'c--',
           )
    
    
    
    #
    # 設定 2軸 顯示範圍
    #
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    
    
    #
    # 再加碼 ，把 constraints 升至 立體的曲面 
    #
    
    xx= np.linspace(-10,10,1001)
    
    #yy1= 3*xx+6
    #yy2= (-xx+4)/2
    #yy3= xx*0 -3
    
    zz1= f([xx,yy1])
    zz2= f([xx,yy2])
    zz3= f([xx,yy3])
    
    ax.plot3D(xx,yy1,zz1, color='r', linestyle='-')
    ax.plot3D(xx,yy2,zz2, color='g', linestyle='-')
    ax.plot3D(xx,yy3,zz3, color='b', linestyle='-')
    
    pl.show()

if __name__=='__main__':
    
    #ryOptimization()
    
    def g(s):
        x, y= s
        z=  (x+9)*(x+5)*(x-5)*(x-9)
        z+=  -(y+12)*(y+8)*(y-8)*(y-10)
      
        return z
    
    ryOptimization(f=g)

