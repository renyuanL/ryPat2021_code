'''
ryPat_Optimization_02.py

3個條件 constraints (c1,c2,c3)
外加 4個 正交 界線 xyBounds
夾出一個 最佳化的搜尋空間
用來 尋找 f() 的極小值
並做 2-d 視覺化

特別強調 constraints 的變化
constraints 夾出一個 凸型區域，我把它塗成灰色，
但這個區域的決定尚未自動化，仍須想想看。
'''

from ryPat import *

from ryPat_Optimization_01 import *

#%matplotlib qt # 若你使用 jupyter，這一行應在jupyter 的碼細胞格(code cell)中執行

#f([x,y])

# 這是 objective function
# 目前本程式處理 2-d 函數 f(s)
# 其中
# s= [x,y]= sm.symbols('x,y')
#
# 在本章中 (x,y) 濃縮在 s 裡面，
# 這是 為了 scipy.optimize 的要求

def f(s):
    x, y= s
    
    x0,y0= (8, 1)    
    z=  (x-x0)**2 + ((y-y0)/3)**2 
    
    return z

# constraint函數, 必須同時有 x,y 項，
# 允許用 1e-10 當作 近似 0 的係數，
# 但絕大多數時候不需要，因為可以用 xyBounds 達成相同的目的
def c1(s):
    x,y= s
    #z= x + y*1e-3 +10  # 模擬 x + 10 > =0
    z= x + y + 6        # x + y - 1 >= 0
    return z

def c2(s):
    x,y= s
    #z= x*1e-3 +y +10   # 模擬 y + 10 > =0
    z= -x + y +6      # -x + 2*y - 2 >= 0
    return z

def c3(s):
    x,y= s
    #z= -(x + 3*y -9)
    #z= x + y + 20
    #z= -x - 3*y +30
    ## 試試看2次函數當作constraints
    z= -(x/2)**2 - y +10  # -x**2 - y +10 >=0
    return z

Constraints= [
    {'fun':c1, 'type':'ineq'},
    {'fun':c2, 'type':'ineq'},
    {'fun':c3, 'type':'ineq'},
]

xyBounds= [(-5, 5),  # xmin, xmax
           (-5, 5)   # ymin, ymax
           ]

ryOptimization(f,c1,c2,c3, xyBounds)

pl.figure()

'''
def ryOptimization(
        f= f, 
        c1= c1, 
        c2= c2, 
        c3= c3, 
        xyBounds= xyBounds):
'''
# 接下來我們回到 平面上來細部拆分

#
# Objective function s.t. Constraints
#
x,y= sm.symbols('x,y')

# sympy function --> numpy function
z= sm.lambdify([x,y],f([x,y])) 

xx= np.linspace(-10,10,1001)
yy= np.linspace(-10,10,1001)
zz= z(xx,yy)

xm, ym= np.meshgrid(xx,yy)
zm= z(xm,ym)
zz.shape, zm.shape



#
# plot constraints 
#

#  Constraints 
# 如何自動從 c1, c2, c3 撈出來？？
# 需要 sympy.solve()
#
y1= sm.solve(
        c1([x,y]),
        y)[0]
y2= sm.solve(
        c2([x,y]),
        y)[0]
y3= sm.solve(
        c3([x,y]),
        y)[0]

# sympy 解方程式之後，仍然需要numpy

y1= sm.lambdify(x, y1)
y2= sm.lambdify(x, y2)
y3= sm.lambdify(x, y3)

yy1= y1(xx)
yy2= y2(xx)
yy3= y3(xx)
#-----------------------

#
#  平行x軸,y軸 界線 (xyBounds) 可以如下製造出來 
#
[(xmin,xmax),(ymin,ymax)]= xyBounds
           
y_ymin= ymin *np.ones_like(xx)
y_ymax= ymax *np.ones_like(xx)
x_xmin= xmin *np.ones_like(yy)
x_xmax= xmax *np.ones_like(yy)

#
# 坐標軸 (x軸,y軸) 可以如下製造出來 
#
y_0= np.zeros_like(xx)
x_0= np.zeros_like(yy)

ax= pl.axes(xlim=(-10,10),ylim=(-10,10))#, projection='3d')

ax.contour(xm,ym,zm, 100, cmap='rainbow')

ax.plot(xx,yy1,'r',
        xx,yy2,'g',
        xx,yy3,'b',
        linestyle='--'
       )    





#
# 畫坐標軸 (x軸,y軸) 
#
ax.plot(xx, y_0,'w-',
        x_0,yy, 'w-',
        linewidth= 3,
        alpha= 0.3 
       )

#
#  畫 xyBounds (平行x軸,y軸) 
#
ax.plot(xx,     y_ymin,
        xx,     y_ymax,
        x_xmin, yy, 
        x_xmax, yy, 
        linewidth= 1,
        linestyle= '--',
        color= 'gray'
       )

# 把 符合限制條件的區域標示出來 (塗滿灰色)
# 這裡比較難以全自動，再想想、、、

ax.fill_between(xx, yy3, yy1,                
                where= (yy3 >= yy1)&(yy1>=yy2),
                alpha=0.50, 
                color='gray', 
                interpolate=True
               )
ax.fill_between(xx, yy3, yy2,                
                where= (yy3 >= yy2)&(yy2>=yy1),
                alpha=0.50, 
                color='gray', 
                interpolate=True
               )

for yyy in [yy1,yy2,yy3]:
    
    ax.fill_between(xx, y_ymax, yyy,                 
                    where= (yyy>=y_ymax),
                    alpha= 0.5, 
                    color='white', 
                    interpolate=True
                   )
    ax.fill_between(xx, y_ymin, yyy,                 
                    where= (yyy<=y_ymin),
                    alpha= 0.5, 
                    color='white', 
                    interpolate=True
                   )



# 在這個區域內找最小 f([x,y])
import scipy.optimize as sopt

x0= [0, 0]

opt= sopt.minimize(
    f,
    x0,
    #method= 'SLSQP',
    bounds= xyBounds,
    constraints=     Constraints
)

opt

#
# 畫 最佳點 opt.x 
#

xopt= opt.x[0]
yopt= opt.x[1]
fopt= opt.fun

ax.scatter(
    x= xopt,
    y= yopt, 
    color= 'magenta',
    marker= 's'    
)

ax.text(
    x= xopt, 
    y= yopt,
    s= f'{xopt:.3f},{yopt:.3f}'
)


fLtx=  sm.latex(f([x,y]))
c1Ltx= sm.latex(c1([x,y]))
c2Ltx= sm.latex(c2([x,y]))
c3Ltx= sm.latex(c3([x,y]))

titleStr= f'''
Optimize:      f(x,y)= ${fLtx}$
Subject to:    Constraints and Bounds
opt= [x= {xopt:.3f}, y= {yopt:.3f}; f= {fopt:.3f}]
'''

ax.set_title(titleStr)
ax.set_xlabel('x')
ax.set_ylabel('y')

infoStr= f'''
$Objective: f(x,y)= {fLtx}$
$c_1(x,y)= {c1Ltx} ≥ 0$ 
$c_2(x,y)= {c2Ltx} ≥ 0$ 
$c_3(x,y)= {c3Ltx} ≥ 0$
$x \in [{xmin},{xmax}]$ 
$y \in [{ymin},{ymax}]$
opt= [x= {xopt:.3f}, y= {yopt:.3f}; f= {fopt:.3f}]
'''
ax.text(
    x= -10, 
    y= -10,
    s= infoStr
)

pl.show()
