'''
ryPat_Optimization.py
'''
# 
# NonLinear Optimization
# Wonderful!!
#

from ryPat import *
import scipy.optimize as sopt

def f(s):
    x, y= s
    
    #z= (x-1)**2 + (y+1)**2
    z= (x-5)**2 + (y-5)**2
    z= - 2**(-z/25)
    
    return z

def c1(s):
    x,y= s
    z= -(-3*x + 1*y -6)
    return z

def c2(s):
    x,y= s
    z= -1*x -2*y +4
    return z

def c3(s):
    x,y= s
    z= 1*y + 3
    return z

#
# Objective function s.t. Constraints
#
x,y= sm.symbols('x,y')

print(f'Objective function: f()={f([x,y])}')
q= f([x,y])
display(q)

print('Constraints: c() >= 0')
print(f'c()= {c1([x,y])},{c2([x,y])},{c3([x,y])}')

q= c1([x,y])
display(q)
q= c2([x,y])
display(q)
q= c3([x,y])
display(q)

opt= sopt.minimize(
    f,
    x0= [0,0],
    constraints= [
        {'fun':c1, 'type': 'ineq'},
        {'fun':c2, 'type': 'ineq'},
        #{'fun':c3, 'type': 'ineq'}        
        ],
    bounds= [(None, None),
             (-3, None)
             #(None, -3)
            ]
)
print(f'optimum= \n{opt}')


#
# plot objective function
#
x,y= sm.symbols('x,y')
#f([x,y])
fg, ax= ryPlotGradientMap(
    f([x,y]), 
    xrange= (-10,10), 
    yrange= (-10,10)
)

#
# plot constraints 
#

xx= np.linspace(-10,10,1001)

yy1= 3*xx+6
yy2= (-xx+4)/2
yy3= xx*0 -3

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
# 設定 2軸 顯示範圍
#
ax.set_xlim([-10,10])
ax.set_ylim([-10,10])


#
# 再加碼 ，把 constraints 升至 立體的曲面 
#

xx= np.linspace(-10,10,1001)

yy1= 3*xx+6
yy2= (-xx+4)/2
yy3= xx*0 -3

zz1= f([xx,yy1])
zz2= f([xx,yy2])
zz3= f([xx,yy3])

ax.plot3D(xx,yy1,zz1, color='r', linestyle='-')
ax.plot3D(xx,yy2,zz2, color='g', linestyle='-')
ax.plot3D(xx,yy3,zz3, color='b', linestyle='-')

pl.show()