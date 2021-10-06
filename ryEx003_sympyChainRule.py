# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 09:42:24 2021

@author: renyu
"""

import sympy as sy

# 用 sympy 來驗證 Chain-Rule

import sympy as sy

def f(x):
    y= 1 + x + x**2
    #y= sy.exp(x)
    
    return y

def g(x):
    y= x**3
    #y= sy.sqrt(x)
    #y= sy.cos(x)
    
    return y

x   = sy.symbols('x')
ϕ, γ= sy.symbols('ϕ, γ') # ϕ, γ are Greek letters for f, g


print(f'''
f(x)=
{f(x)}

g(x)=
{g(x)}

f(γ)=
{f(γ)}

g(ϕ)=
{g(ϕ)}

f(g(x))=
{f(g(x))}

g(f(x))=
{g(f(x))}

''')

#%%
# d(f(g(x)))/dx == df/dg * dg/dx
#

F= f(γ)
print(f'''
F= f(γ)=
{F}
''')

F= f(g(x))
print(f'''
F= f(g(x))=
{F}
''')


Left= f(g(x)).diff(x)
Left= Left.expand()


Right=  f(γ).diff(γ) * g(x).diff(x)
Right= Right.subs(γ, g(x))
Right= Right.expand()


assert Left==Right

print(f'''
Left= dF/dx=
{Left}

Right= df/dγ * dγ/dx=
{Right}
'''
)

#%%
#
# d(g(f(x))) / dx == dg/df * df/dx
#


G= g(ϕ)
print(f'''
G= g(ϕ)=
{G}
''')

G= g(f(x))
print(f'''
G= g(f(x))=
{G}
''')

Left= g(f(x)).diff(x)
Left= Left.expand()

Right=  g(ϕ).diff(ϕ) * f(x).diff(x)
Right= Right.subs(ϕ, f(x))
Right= Right.expand()



assert Left==Right

print(f'''
Left= dG/dx=
{Left}

Right= dg/dϕ * dϕ/dx= 
{Right}
'''
)

#%%
# 用 sympy 來驗證 product-Rule

fg= f(x)*g(x)

Left= (f(x)*g(x)).diff(x)
Left= Left.expand()

Right= f(x).diff(x) * g(x) + g(x).diff(x) * f(x)
Right= Right.expand()

assert Left == Right

print(f'''
Left= d(fg)/dx=
{Left}

Right= df/dx *g + dg/dx  *f = 
{Right}
'''
)

#%%
# general product Rule

def f1(x):
    y= 1+x
    return y

def f2(x):
    y= x**2
    return y

def f3(x):
    y= x**3
    return y

f123= f1(x)*f2(x)*f3(x)

print(f123)


Left= f123.diff(x)
Left= Left.expand()

Right= \
 f1(x).diff(x)*f123/f1(x) \
+f2(x).diff(x)*f123/f2(x) \
+f3(x).diff(x)*f123/f3(x)
Right= Right.expand()

assert Left==Right

print(f'''
Left= d(f1*f2*f3)/dx=
{Left}

Right= Σ (dfi/dx * F/fi) = 
{Right}
'''
)
#%%

q= sy.asin(x).diff(x)
print(q)

q= sy.acos(x).diff(x)
print(q)

q= sy.atan(x).diff(x)
print(q)
#%%
q= sy.log(x).diff(x)
print(q)

#%%
q= (1/x).integrate(x)#.diff(x)
print(q)

q= (1/x).diff(x)#.integrate(x)
print(q)
#%%

q= -1/sy.sqrt(1-x**2)
q= q.integrate(x)
print(q)
#%%
q= sy.acos(x).diff(x).integrate(x)
print(q)
#%%
def f(x):
    y= 4 * sy.sin(x) + (2*x**5 -sy.sqrt(x))/x
    return(y)

print(f(x))

fi= f(x).integrate(x) 

print(fi)

fd= f(x).diff(x)

print(fd)

fid= fi.diff(x)
fdi= fd.integrate(x)

assert fid.expand() == fdi.expand()

#%%


q= f(x).integrate((x,0,1))
print(q)
q= float(q)
print(q)

#%%
i, n, m, c= sy.symbols('i, n, m, c')

#a= i**2+i+1

def a(i):
    y= i**2 + i + 1
    return y

q= sy.Sum(a(i), (i,0,n))
print(q)

q= q.doit()
print(q)

q= q.subs(n,100)
print(q)

#%%


def f(x):
    y= 2*x*sy.sqrt(x**2+1)
    return y


print(f(x))

F= f(x).integrate(x)
print(F)

F= F.subs(x,1)-F.subs(x,0)
print(F)

F= f(x).integrate((x,0,1))
F= float(F)
print(F)
#%%
# 來試試看一個有名的積分，Gauss 分布

def f(x):
    y= sy.exp(-x**2)
    return y


print(f(x))

F= f(x).integrate(x)
print(F)

F= F.subs(x,1)-F.subs(x,0)
print(F)

F= f(x).integrate((x,0,1))
F= float(F)
print(F)

F= f(x).integrate((x,-sy.oo,sy.oo))
F= float(F)
print(F)

#%%
π= sy.pi

assert f(x).integrate((x,-sy.oo,sy.oo)) == sy.sqrt(π)


#%%

def f(x):
    y= 1+x+x**2
    return y
def g(x):
    y= x**3
    return y

q1= f(x) * g(x).diff(x) + g(x) * f(x).diff(x)
print(q1)

q2= (f(x)*g(x)).diff(x)
print(q2)

assert q1.expand() == q2.expand()
#%%
q= q1.integrate(x)
print(q)

q= f(x)*g(x)
print(q)
#%%
# partianl integration
#
q1= (f(x) * g(x).diff(x)).integrate(x) 

q2= (g(x) * f(x).diff(x)).integrate(x)

q3= f(x)*g(x)

assert (q1+q2).expand() == q3.expand()


#%%

def f(x,n):
    y= x**n * sy.exp(x)
    return y
q= f(x,0).integrate(x)
print(q)

#%%

def f(x,y):
    z= 5*y**3 +7*x**2*y -3*y +11
    return z

x,y = sy.symbols('x,y')

q= f(x,y)
print(q)

q_x= q.diff(x)
print(q_x)

q_y= q.diff(y)
print(q_y)

q_x_y= q_x.diff(y)
print(q_x_y)

q_y_x= q_y.diff(x)
print(q_x_y)

s= q.subs({x:1, y:1})
print(s)
#%%








