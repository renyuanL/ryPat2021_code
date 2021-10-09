#!/usr/bin/env python
# coding: utf-8

# # COURSE: Master math by coding in Python
# ## SECTION: Introduction to Sympy and LaTeX
# 
# #### https://www.udemy.com/course/math-with-python/?couponCode=MXC-DISC4ALL
# #### INSTRUCTOR: sincxpress.com
# 
# Note about this code: Each video in this section of the course corresponds to a section of code below. Please note that this code roughly matches the code shown in the live recording, but is not exactly the same -- the variable names, order of lines, and parameters may be slightly different. 

# In[ ]:





# # VIDEO: Intro to sympy, part 1

# In[1]:


# import the sympy package
import sympy as sym

# optional setup for "fancy" printing (used later)
# sym.init_printing()

# import additional functions for nice printing
from IPython.display import display, Math


# In[2]:


# create symbolic variables
x,y,z = sym.symbols('x,y,z')

x


# In[3]:


# x doesn't have a value; it's just a symbol
x + 4


# In[4]:


# for "fancy" printing (note: you need to run this line only once)
sym.init_printing()
x + 4


# In[5]:


# more fun...
display( x**y )
display( y/z )


# In[6]:


# let's compare with numpy
import numpy as np

display(sym.sqrt(2)) # square root in symbolic math
display(np.sqrt(2)) # square root in numeric/precision math


# ### Exercises

# In[7]:


# 1)
display(y*x**2)

# 2)
display( sym.sqrt(4)*x )

# 3)
display( sym.sqrt(x)*sym.sqrt(x) )


# # VIDEO: Intro to Latex

# In[8]:


# basic latex coding is easy:
display(Math('4+5=7'))


# In[9]:


# special characters are indicated using \\
display(Math('\\sigma = \\mu \\times \\sqrt{5}'))

# outside Python, use one \
display(Math('\\sigma = \\mu \times \\sqrt{5}'))

# subscripts and superscripts
display(Math('x_n + y^m - z^{m+n}'))

# fractions
display(Math('\\frac{1+x}{2e^{\pi}}'))

# right-click to change properties


# In[10]:


# regular text requires a special tag
f = 4
display(Math('Set x equal to %g'%f))

display(Math('\\text{Set x equal to %g}'%f))


# ### Latex code in a markdown cell
# 
# Note: this is not for evaluating variables or other numerical code!
# 
# 
# $$ \frac{1+\sqrt{2x}}{e^{i\pi}} $$

# ### Exercises!

# In[11]:


# 1) 
display(Math('4x+5y-8z=17'))

# 2) 
display(Math('\\sin(2\\pi f t+\\theta)'))

# 3)
display(Math('e=mc^2'))

# 4)
display(Math('\\frac{4+5x^2}{(1+x)(1-x)}'))


# # VIDEO: Intro to sympy, part 2

# In[12]:


# using Greek characters as variable names

mu,alpha,sigma = sym.symbols('mu,alpha,sigma')

expr = sym.exp( (mu-alpha)**2/ (2*sigma**2) )

display(expr)


# In[13]:


# can also use longer and more informative variable names
hello = sym.symbols('hello')

hello/3


# In[14]:


# substituting numbers for variables

# don't forget to define variables before using them!
display(x+4)

(x+4).subs(x,3)


# In[15]:


# substituting multiple variables

x,y = sym.symbols('x,y') # can create multiple variables in one line
expr = x+y+5

# substituting one variable
print( expr.subs(x,5) )

# substituting multiple variables
print( expr.subs({x:5,y:4}) )


# In[16]:


# using sympy to print latex code
expr = 3/x

print( sym.latex(expr) )

# notice:
print( sym.latex(3/4) )
print( sym.latex('3/4') )
# but
print( sym.latex(sym.sympify('3/4')) )


# ### Exercise!

# In[17]:


for i in range(-2,3):
    ans = (x+4).subs(x,i**2)
    display(Math('\\text{With }x=%g:\; x^2+4 \\quad \\Rightarrow \\quad %g^2+4 =%g' %(i,i,ans)))


# # VIDEO: Example: Use sympy to understand the law of exponents

# In[18]:


x,y,z = sym.symbols('x,y,z')

ex = x**y * x**z

display(ex)
display( sym.simplify(ex) )


# In[19]:


expr1 = x**y * x**z
expr2 = x**y / x**z
expr3 = x**y * y**z

display(Math('%s = %s' %( sym.latex(expr1),sym.latex(sym.simplify(expr1)) ) ))
display(Math('%s = %s' %( sym.latex(expr2),sym.latex(sym.simplify(expr2)) ) ))
display(Math('%s = %s' %( sym.latex(expr3),sym.latex(sym.simplify(expr3)) ) ))


# In[20]:


# using sym.Eq

sym.Eq(4,2+2)


# In[21]:


display( sym.Eq(x-3,4) )

# using variables
lhs = x-3
rhs = x
display( sym.Eq(lhs,rhs) )


# In[22]:


lhs = x-3
rhs = x+4#-7
display( sym.Eq(lhs,rhs) )

# or
sym.Eq( lhs-rhs )


# #### NOTE ABOUT `SYM.EQ`
# Sympy changed the behavior of `sym.Eq` since I made this course. To test against zero, you need to write `sym.Eq(expr,0)` instead of just `sym.Eq(expr)`.
# 
# If you are using a sympy version earlier than 1.5, you don't need to change anything. If you get a warning or error message, then simply add `,0` as the second input to `sym.Eq`.
# 
# To test which version of sympy you have, type
# `sym.__version__`

# In[23]:


display( sym.Eq(expr1,expr1) )

# but...
display( sym.Eq(expr1 - sym.simplify(expr1)) )

display( sym.Eq(sym.expand(  expr1-sym.simplify(expr1)  )) )


# In[24]:


# btw, there's also a powsimp function:
display( sym.powsimp(expr1) )

# and its converse
res = sym.powsimp(expr1)
display( sym.expand_power_exp(res) )


# # VIDEO: printing with f-strings

# In[25]:


# basic intro to f-strings

svar = 'Mike'
nvar = 7

print(f'Hi my name is {svar} and I eat {nvar} chocolates every day.')


# In[26]:


# now with latex integration

x,y = sym.symbols('x,y')
expr = 3/x


# trying to print using symbolic variables
display(Math(f'\\frac{x}{y}'))
display(Math(f'\\frac{{sym.latex(expr)}}{{y}}'))
display(Math(f'\\frac{{{x}}}{{{y}}}'))
display(Math(f'\\frac{{{sym.latex(expr)}}}{{{y}}}'))


# my preference for mixing replacements with latex
display(Math('\\frac{%s}{%s}'%(sym.latex(expr),y)))


# In[27]:


# print using numeric variables
u,w = 504,438

display(Math(f'\\frac{u}{w}'))
display(Math(f'\\frac{{u}}{{w}}'))
display(Math(f'\\frac{{{u}}}{{{w}}}'))

# my preference for mixing replacements with latex
display(Math('\\frac{%g}{%g}'%(u,w)))


# In[ ]:





# # VIDEO: Sympy and LaTex: Bug hunt!

# In[28]:


mu,alpha = sym.symbols('mu,alpha')

expr = 2*sym.exp(mu**2/alpha)

display(Math( sym.latex(expr) ))


# In[29]:


Math('1234 + \\frac{3x}{\sin(2\pi t+\\theta)}')


# In[30]:


a = '3'
b = '4'

# answer should be 7
print(sym.sympify(a)+sym.sympify(b))


# In[31]:


x = sym.symbols('x')
sym.solve( 4*x - 2 )


# In[32]:


# part 1 of 2

q = x**2
r = x**2

display(q)
display(r)


# In[33]:


# part 2 of 2

q,r = sym.symbols('q,r')

q = sym.sympify('x^2')
r = sym.sympify('x**2')

display(Math(sym.latex(q)))
display(r)

sym.Eq(q,r)


# In[34]:


x = sym.symbols('x')

equation = (4*x**2 - 5*x + 10)**(1/2)
display(equation)
equation.subs(x,3)


# In[35]:


x,y = sym.symbols('x,y')

equation = 1/4*x*y**2 - x*(5*x + 10*y**2)**(3)
display(equation)

