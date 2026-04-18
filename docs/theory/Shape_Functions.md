# Shape Functions {#shape_functions}

**[PDF Version](Shape_Functions.pdf)**

## About shape function for pyramid:

In order to be coherent with the facial elements:

Base:
$$
\xi\in[-1,1],\eta\in[-1,1],\zeta=0
$$
Axis:
$$
\xi=0,\eta=0,\zeta=[0,1]
$$

![PyramidShow](PyramidShow.png)

Linear Pyramid5:
$$
\left(\begin{array}{c} -\frac{\left(\eta +\zeta -1\right)\,\left(\xi +\zeta -1\right)}{4\,\left(\zeta -1\right)}\\ \frac{\left(\eta +\zeta -1\right)\,\left(\xi -\zeta +1\right)}{4\,\left(\zeta -1\right)}\\ -\frac{\left(\eta -\zeta +1\right)\,\left(\xi -\zeta +1\right)}{4\,\left(\zeta -1\right)}\\ \frac{\left(\xi +\zeta -1\right)\,\left(\eta -\zeta +1\right)}{4\,\left(\zeta -1\right)}\\ \zeta  \end{array}\right)
$$




Quad Pyramid14:
$$
\left(\begin{array}{c} \frac{\eta \,\xi \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi +\zeta -1\right)}{8\,{\left(\zeta -1\right)}^4}\\ \frac{\eta \,\xi \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi -\zeta +1\right)}{8\,{\left(\zeta -1\right)}^4}\\ \frac{\eta \,\xi \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta -\zeta +1\right)\,\left(\xi -\zeta +1\right)}{8\,{\left(\zeta -1\right)}^4}\\ \frac{\eta \,\xi \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\xi +\zeta -1\right)\,\left(\eta -\zeta +1\right)}{8\,{\left(\zeta -1\right)}^4}\\ \zeta \,\left(2\,\zeta -1\right)\\ -\frac{\eta \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi +\zeta -1\right)\,\left(\xi -\zeta +1\right)}{4\,{\left(\zeta -1\right)}^4}\\ -\frac{\xi \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\eta -\zeta +1\right)\,\left(\xi -\zeta +1\right)}{4\,{\left(\zeta -1\right)}^4}\\ -\frac{\eta \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\xi +\zeta -1\right)\,\left(\eta -\zeta +1\right)\,\left(\xi -\zeta +1\right)}{4\,{\left(\zeta -1\right)}^4}\\ -\frac{\xi \,\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi +\zeta -1\right)\,\left(\eta -\zeta +1\right)}{4\,{\left(\zeta -1\right)}^4}\\ -\frac{\zeta \,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi +\zeta -1\right)}{2\,{\left(\zeta -1\right)}^2}\\ \frac{\zeta \,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi -\zeta +1\right)}{2\,{\left(\zeta -1\right)}^2}\\ -\frac{\zeta \,\left(2\,\zeta -2\right)\,\left(\eta -\zeta +1\right)\,\left(\xi -\zeta +1\right)}{2\,{\left(\zeta -1\right)}^2}\\ \frac{\zeta \,\left(2\,\zeta -2\right)\,\left(\xi +\zeta -1\right)\,\left(\eta -\zeta +1\right)}{2\,{\left(\zeta -1\right)}^2}\\ \frac{\left(2\,\zeta -1\right)\,\left(2\,\zeta -2\right)\,\left(\eta +\zeta -1\right)\,\left(\xi +\zeta -1\right)\,\left(\eta -\zeta +1\right)\,\left(\xi -\zeta +1\right)}{2\,{\left(\zeta -1\right)}^4} \end{array}\right)
$$

Here we slice N3 from Pryamid5.

At $\xi = 0$: 
![Pyramid5_N3_X0](Pyramid5_N3_X0.png)
At $\xi = 0$: 
![Pyramid5_N3_X00](Pyramid5_N3_X00.png)
At $\xi = 0.1$: 
![Pyramid5_N3_X01](Pyramid5_N3_X01.png)
At $\xi = 0.5$: 
![Pyramid5_N3_X05](Pyramid5_N3_X05.png)

$\xi=0$ leads to a purely linear function in slice,while  $\xi \neq 0$ leads to more complex distribution. Note that although singularity exists, non of them exist inside the pyramid.

The shape functions and first derivatives have singularities and discontinuities along $\zeta =0$, but for the cone inside the Pyramid, the functions seem to be regular. **?**