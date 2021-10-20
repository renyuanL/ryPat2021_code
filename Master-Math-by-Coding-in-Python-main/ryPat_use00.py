import numpy
import numpy as np

import sympy
import sympy as sm
from   sympy.abc import *

import matplotlib.pyplot as pl
import matplotlib.pyplot as plt

import seaborn as sb
sb.set_style('whitegrid')

from IPython.display import display, Math

#%matplotlib qt
#%matplotlib inline
sm.init_printing()
π= sm.pi
#%%

def f(x,y):
    z= x + y**2
    return z

z= f(x,y)
sm.plotting.plot3d(
    z, 
    (x, 0,1),
    (y, 0,1))
#%%

x= sm.symbols('x')

f= sm.sin(x**2)

F= sm.Integral(f,(x,0,x))
ϕ= sm.Derivative(f,x)

display(F)
F1= F.doit()
display(F1)

display(ϕ)
ϕ1= ϕ.doit()
display(ϕ1)


#%%
sm.plotting.plot(f, (x,0,10), title=f'f= {f}')
sm.plotting.plot(F1, (x,0,10), title=f'F= {F}')
sm.plotting.plot(ϕ1, (x,0,10), title=f'ϕ= {ϕ}')
#%%
def f(x,y):
    z= x+y**2
    return z

q= sm.Integral(
    sm.Derivative(
        f(x,y),
        x),
    x)
q1= q.doit()

#%%
θ= sm.symbols('θ')

x= sm.cos(θ) 
y= sm.sin(θ*1e-10)

sm.plotting.plot3d_parametric_line(
    x,
    y,
    θ,
    (θ, -π*10, π*10),
    xlabel= f'x= {x}',
    ylabel= f'y= {y}'
)

#%%
def f(x):
    y= x**2
    return y

π= sm.pi

x= sm.symbols('x')

p= sm.plotting.plot(
    f(x), 
    f(x).diff(x),
    f(x).integrate(x),
    (x,-5, 5), 
    show=False
)

p[0].line_color= 'r'
p[1].line_color= 'g'
p[2].line_color= 'b'
p.legend= True
p.show()
#%%
def f(x):
    y= sm.sin(x)
    return y

def df(f,x=None,dx=None):
    '''
    尚不成熟，要再想想...
    '''
    if x==None:
        x= sm.symbols('x')
    if dx==None:
        dx= sm.symbols('dx')
    df= f(x+dx)-f(x)
    dff= lambda x:df
    return dff
#%%

# Antiderivative

def f(x):
    y= sm.sqrt(1-x**2)
    return y

F= sm.Integral(f(x),x)

display(F)
display(F.doit())


#%%
def f(x):
    y= 4*sm.sin(x) +(2*x**5-sm.sqrt(x))/x
    return y

F= sm.Integral(f(x),x)
F1= F.doit()

display(F)
display(F1)

#%%

# sin^(−1)⁡〖(x)〗,  cos^(−1)⁡〖(x)〗,  tan^(−1)⁡〖(x)〗
def f(x):
    #y= sm.asin(x)
    #y= sm.acos(x)
    y= sm.atan(x)
    return y

df_dx= lambda x: f(x).diff(x)

def ryPlot(f,x,x0=-1, x1=1):
    q= sm.plotting.plot(
        f(x),
        (x,x0,x1),
        show=False
        )
    q.legend=True
    q.show()

ryPlot(f,x,-10,10)
ryPlot(df_dx,x,-10,10)

#%%
def f(x):
    '''
    Hyperbolic functions, 
    '''        
    #y= sm.sinh(x)
    #y= sm.cosh(x)
    y= sm.tanh(x)
       
    return y

df_dx= lambda x: f(x).diff(x)


def ryPlot(f,x,x0=-1, x1=1):
    q= sm.plotting.plot(
        f(x),
        (x,x0,x1),
        title= f'{f.__name__}',
        show=False
        )
    q.legend=True
    q.show()

ryPlot(f,x,-10,10)
ryPlot(df_dx,x,-10,10)
#%%

#%%

#%%
mgrid= np.mgrid[0:10,0:10]
ogrid= np.ogrid[0:10,0:10]
#%%
# Solve the differential equation y" − y = e^t.

f= sm.Function('f')
解= sm.dsolve(
    sm.Eq(
        f(t).diff(t, t) - f(t), 
        sm.exp(t)), 
    f(t))

#%%
q= 解.rhs.as_expr(
    ).subs({"C1":10, "C2":20})

#%%
# with initial conditions
# y(0) = 0
# y'(0)=10

y= 解.rhs.as_expr()

eq1= sm.Eq(y.subs(t,0),0)
eq2= sm.Eq(y.diff(t).subs(t,0),10)


#C1, C2= sm.symbols('C1, C2')

q= sm.solve(
  [eq1, eq2],
  #sm.symbols('C1, C2')
  'C1','C2'
  )

#%%
x= sm.Symbol('x')

#%%

























































