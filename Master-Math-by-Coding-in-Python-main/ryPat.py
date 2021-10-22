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
sm.init_printing(use_unicode=True)
#%%
π= sm.pi

#%%

def ryGradient_demo(z=None):
    '''
    input: z= f(x,y)
    '''
    x,y= sm.symbols('x,y')

    if z == None:
        z= x**2 +y**2
        #z= x*(x-10)*(x+10) +y*(y-10)*(y+10)
        #z= sm.sin(x)/x+sm.sin(y)/y

    dz_dx= z.diff(x)
    dz_dy= z.diff(y)

    print(f'z= {z}')
    print(f'dz_dx= {dz_dx}')
    print(f'dz_dy= {dz_dy}')
    
    display(z)
    display(dz_dx)
    display(dz_dy)


    ϵ= 0.2

    xrange= np.arange(-10, 10+ϵ, ϵ)
    yrange= np.arange(-10, 10+ϵ, ϵ)

    xm, ym= np.meshgrid(xrange,yrange)
    zm= sm.lambdify((x,y),z)(xm,ym)

    fg= pl.figure()
    ax= pl.axes(projection='3d', 
                xlabel= 'x', 
                ylabel= 'y', 
                #title=  f'z= {z}'
                title= f'z= ${sm.latex(z)}$'
               )

    ax.contour3D(xm, ym, zm, 100, cmap='rainbow')

    #--------------------------------
    '''
    ax.plot_surface(xm,ym,zm, 
                    #rstride= 1, 
                    #cstride= 1,
                    cmap=     'rainbow', #'viridis', #'rainbow',  
                    edgecolor='none'
                   )
    '''

    #-------------------------------------------
    downSampleFactor= 5
    x0, y0, z0= np.meshgrid(
        xrange[::downSampleFactor],
        yrange[::downSampleFactor],
        np.linspace(np.min(zm), np.min(zm)+1, 1)
        )

    #dz_dx= 2*x0
    #dz_dy= 2*y0

    dz_dx_m= sm.lambdify((x,y),dz_dx)(x0,y0)
    dz_dy_m= sm.lambdify((x,y),dz_dy)(x0,y0)

    u= dz_dx_m
    v= dz_dy_m
    w= np.zeros_like(x0)
    ax.quiver(x0, y0, z0, 
              u, v, w, 
              length= 1/np.max([u,v]), 
              color = 'gray')

if __name__=='__main__':
    ryGradient_demo()
    