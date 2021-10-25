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


# 封裝測試一下...，允許輸入不同函數
# 加碼練習 subplot()
# 加碼觀察 cumsum()

def ryHistogram_demo(func= None):
    
    # 在時間軸上 不同的訊號，
    # 使用 np.histogram() 把它們視為相同的機率分布！
    t= np.linspace(-10,10,1001)
    #t= np.arange(-10,10)
    
    if func==None:
        func= np.sin
        
    X= func(t) #t**2
    
    X1= np.sort(X)
    X2= X.copy()
    np.random.shuffle(X2)  # 這個 .shuffle() 每次結果都不同！！
    
    
    # 針對離散型，也許 nbins 要精細一點計算
    nbins= len(set(X))
    nbins= min(nbins, 20)
    
    f, x= np.histogram(X, bins= nbins)
    f= np.insert(f,0,0)  
    # 上面這一行是 ry 特殊補丁，
    # .histogram() 做出的 f, x 長度不同，影響後面程式碼的簡潔！
    f1, x1= np.histogram(X1, bins= nbins)
    f1= np.insert(f1,0,0)  
    f2, x2= np.histogram(X2, bins= nbins)
    f2= np.insert(f2,0,0)
    t, X, X1, X2
    x, f, x1, f1, x2, f2
    
    # 加碼觀察 cumsum
    F= np.cumsum(f)
    
    #fg1= pl.figure('fg1')
    #ax= pl.axes()
    
    fg, ax = plt.subplots(2,2,figsize=(10,5))
    
    ax[0,0].plot(t,X,'ro',t,X1,'gx',t,X2,'b.')
    ax[0,0].set_xlabel('t')
    ax[0,0].set_ylabel('X, X1, X2')
    ax[0,0].legend(['X','X1','X2'])
    
    #'''
    #fg2= pl.figure('fg2')
    #ax2= pl.axes()
    ax[0,1].plot(f,x,'ro--',f1,x1,'gx-.',f2,x2,'b.-',F,x,'cs--')
    ax[0,1].set_xlabel('f, f1, f2, F')
    ax[0,1].set_ylabel('x')
    ax[0,1].legend(['f','f1','f2','F'])
    #'''
    #fg3= pl.figure('fg3')
    #ax3= pl.axes()
    ax[1,0].plot(x,f,'ro--',x1,f1,'gx-',x2,f2,'b.-', x,F,'cs--')
    ax[1,0].set_ylabel('f, f1, f2, F')
    ax[1,0].set_xlabel('x')
    ax[1,0].legend(['f','f1','f2','F'])
    
    # 加碼觀察 cumsum
    # 本想簡單達成
    # 但 加 legend() 出問題，只好把以下幾行混進上面程式碼中，殘念！！！
    
    '''
    F= np.cumsum(f)

    ax[1,0].plot(x,F,'cs-')
    #ax[1,0].legend(['F'])

    ax[0,1].plot(F,x,'cs-')
    #ax[0,1].legend(['F'])
    '''
 
    
def ryDistribution_demo(distributionType= 'normal', sampleSize= 1000):
    
    #sampleSize= 1000

    t= np.arange(sampleSize)
    
    if distributionType in ['normal','gauss','gaussian']:
        X= np.random.normal(size= sampleSize)
    elif distributionType == 'uniform':
        X= np.random.uniform(0,10, size= sampleSize)
    elif distributionType == 'binomial':
        X= np.random.binomial(n=10, p=.5, size= sampleSize)
    elif distributionType == 'poisson':
        X= np.random.poisson(lam=5, size= sampleSize)
    elif distributionType == 'uniform_d':
        X= np.random.randint(low= 0, high= 10, size= sampleSize)
    else: # 以上皆非時，以下 sin() 取代
        X= np.sin(t/sampleSize*2*np.pi)
        X += np.sin(t/sampleSize*2*np.pi *2)
        X += np.sin(t/sampleSize*2*np.pi *3)
        
    #X= X*X

    fig,ax = plt.subplots(3, 1,figsize=(6,3))
    ax[0].plot(t,X,'r.',alpha=.5)

    X1= X.copy()
    X1.sort()

    ax[0].plot(t,X1,'g.',alpha=.3)

    ax[0].set_xlabel('t')
    ax[0].set_ylabel('X')
    ax[0].legend(['X','X1= X.sort()'])
    ax[0].set_title(f'{distributionType}, X= X(t)')

    ax[1].plot(X,t,'r.',alpha=.5)
    ax[1].plot(X1,t,'g.',alpha=.3)

    ax[1].set_xlabel('X')
    ax[1].set_ylabel('t')
    ax[1].legend(['X','X1= X.sort()'])
    ax[1].set_title('$t= t(X)=X^{-1}(X(t))$, t in vertical, X in horizontal')

    # 針對離散型，也許 nbins 要精細一點計算
    nbins= len(set(X))
    nbins= min(nbins, 20)
    
    f, x= np.histogram(X, bins= nbins)
    x= x[:-1] # 原本的 x 比 f 多 1 點，故在此把它弄掉
    
    xwidth=  (X.max()-X.min())/nbins/2
    
    ax[2].bar(x, f, width= xwidth, color='r', alpha=.5)

    F= f.cumsum()
    ax[2].bar(x, F, width= xwidth, color= 'g', alpha=.3)

    ax[2].set_xlabel('x')
    ax[2].set_ylabel('f, F')
    ax[2].legend(['f','F= f.cumsum()'])
    ax[2].set_title('f= histogram(X)= f(x)')


    plt.show()
    
if __name__=='__main__':
    
    #ryGradient_demo()
    
    #z= sm.exp(-(x**2+y**2))
    #ryPlotGradientMap(z)
    
    def f(t): 
        #y= np.sin(t)
        #y= np.random.random(size= len(t))
        y= np.random.normal(size= len(t))
        
        return y

    ryHistogram_demo(f)
    
    ryDistribution_demo('normal')
    