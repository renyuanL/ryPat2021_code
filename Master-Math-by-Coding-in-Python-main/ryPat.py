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


kwargs={'xlabel': 'x', 'ylabel': 'y'}
    
def ryPlot3d_ver0_(z, xrange=(-3,+3), yrange=(-3,+3), kwargs= None):
    
    if kwargs == None:
        kwargs={'xlabel': 'x', 'ylabel': 'y'}
    
    xmin, xmax= xrange
    ymin, ymax= yrange
    
    xm,ym= np.meshgrid(
        np.linspace(xmin,xmax,101),
        np.linspace(ymin,ymax,101)
    )
    zm= sm.lambdify((x,y),z)(xm,ym)

    fg= pl.figure()
    ax= pl.axes(projection='3d',
                title= f'z= ${sm.latex(z)}$',
                **kwargs)
    ax.contour3D(xm, ym, zm, 100,
                 cmap= 'rainbow')
    return fg, ax    
    
# 進入真正的深水區
# 運用 ax.quiver() 結合 Gradient Vector ，
# 把 2D 函數 的梯度圖畫出。

def ryPlot3d(z, xrange=(-3,+3), yrange=(-3,+3), kwargs= None):
    
    if kwargs == None:
        kwargs={'xlabel': 'x', 'ylabel': 'y'}
    
    xmin, xmax= xrange
    ymin, ymax= yrange
    
    xrange= np.linspace(xmin,xmax,101)
    yrange= np.linspace(ymin,ymax,101)
    xm,ym=  np.meshgrid(xrange,yrange)
    
    zm= sm.lambdify((x,y),z)(xm,ym)

    fg= pl.figure()
    ax= pl.axes(projection='3d',
                title= f'z= ${sm.latex(z)}$',
                **kwargs)
    ax.contour3D(xm, ym, zm, 100,
                 cmap= 'rainbow')
    return fg, ax, zm, xrange, yrange

def ryPlotGradientMap(z,
                      xrange=(-3,+3), 
                      yrange=(-3,+3),
                      downSampleFactor= 4, 
                      lengthFactor= .1):
    
    fg, ax, zm, xrange, yrange=  ryPlot3d(z, 
                                          xrange= xrange, 
                                          yrange=yrange)
    
    z10= z.diff(x,1,y,0)
    z01= z.diff(x,0,y,1)

    #downSampleFactor= 2
    xm, ym, zm0= np.meshgrid(
        xrange[::downSampleFactor],
        yrange[::downSampleFactor],
        np.linspace(np.min(zm), np.min(zm)+1, 1)
        )

    #dz_dx= 2*x0
    #dz_dy= 2*y0

    dz_dx_m= sm.lambdify((x,y),z10)(xm,ym)
    dz_dy_m= sm.lambdify((x,y),z01)(xm,ym)

    u= dz_dx_m
    v= dz_dy_m
    w= np.zeros_like(xm)
    ax.quiver(xm, ym, zm0, 
              u, v, w, 
              length= lengthFactor, #*np.max([u,v]), 
              color = 'gray')    

if __name__=='__main__':
    
    ryGradient_demo()
    
    z= sm.exp(-(x**2+y**2))
    ryPlotGradientMap(z)
    