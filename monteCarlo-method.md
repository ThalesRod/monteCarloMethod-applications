
# Monte Carlo Method

## 1- Calculating a Single Variable Integral numerically:

$$ \int_{a}^{b} f(x)dx = \textit A_f  \quad \textit{(1)} $$  
***A<sub>f</sub>*** = Area under the function  


$$ \frac{A_f}{A_s} = \frac{N_c}{N_t} \quad \textit{(2)} $$

***A<sub>s</sub>*** = Square area  
***N<sub>c</sub>*** = Points landing the function  
***N<sub>t</sub>*** = Total of points  
  
#### $$ A_f \approx A_s \times \frac{N_c}{N_t} $$  

### Implemantation:


```python
%matplotlib inline

import random

import numpy as np
import matplotlib.pyplot as plt


# Function to be integrated
def function(x):
  return np.cos(x)

# Function limits
a = 0
b = np.pi / 2

# Height of the rectangle surrounding the Function
h = 1

# Number of points
n = 1e3

areas = list()

for area in range(int(1e3)):
  # Splash points counter
  ns = 0
    
  for point in range(int(n)):
    # Point coordinates
    xi = random.uniform(a, b) # x axis variation between the Function limits
    yi = random.uniform(0, h) # y axis variation from 0 to rectangle height

    if yi <= function(xi):
      ns += 1
    
  integral = h * (b - a) * (ns / n)
  areas = np.append(areas, integral)

print("Area calculated (Mean): {0}".format(areas.mean()))
print("Area calculated (Median): {0}".format(np.median(areas)))
        
plt.title("Distribution of Areas Calculated")
plt.xlabel("Areas")
plt.hist(areas, bins=30, ec='black')
plt.show()
```

    Area calculated (Mean): 1.000788897320218
    Area calculated (Median): 1.000597260168349



![png](monteCarlo-method_files/monteCarlo-method_4_1.png)


## 2- Calculating a Double Integral numerically:

$$ \int_{c}^{d}\int_{a}^{b} f(x,y)dxdy = \textit V_f  \quad \textit{(1)} $$  
***V<sub>f</sub>*** = Volume under the function  


$$ \frac{V_f}{V_s} = \frac{N_c}{N_t} \quad \textit{(2)} $$

***V<sub>s</sub>*** = Prism volume  
***N<sub>c</sub>*** = Points landing the function  
***N<sub>t</sub>*** = Total of points  
  
#### $$ V_f \approx V_s \times \frac{N_c}{N_t} $$  


### Implemantation:


```python
%matplotlib inline
from numpy import random
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
  return (x**2 + y**2)
  
a, b = 0, 5
c, d = 0, 5 # integration limits

h = 5

n = 1e3

volumes = np.array(list())

for i in range(int(1e3)):
  ns = 0

  for point in range(int(n)):
    xi = random.uniform(a, b)
    yi = random.uniform(c, d)
    zi = random.uniform(0, h)
    
    if zi <= f(xi, yi):
      ns += 1
    
  volume = (b - a) * (d - c) * h * (ns/n)
  volumes = np.append(volumes, volume)

print("Volume calculated (Mean): {0}".format(volumes.mean()))
print("Volume calculated (Median): {0}".format(np.median(volumes)))

plt.title("Distribution of Volumes Calculated")
plt.xlabel("Volumes")
plt.hist(volumes, bins=30, ec='b')
plt.show()
```

    Volume calculated (Mean): 115.20575
    Volume calculated (Median): 115.25



![png](monteCarlo-method_files/monteCarlo-method_8_1.png)


## 3- Estimating $\pi$ value numerically by aproximation:

$$ \frac{Ac}{As} = \frac{Nc}{Nt} \quad \textit{(1)} $$

***Ac*** = Circle area  
***As*** = Square area  
***Nc*** = Points landing the circle  
***Nt*** = Total of points  
  
#### $$ \pi = \frac{Ac}{r^2} \quad \textit{(2)} $$  
***r*** = Circle radius

### Implemantation:


```python
%matplotlib inline

from numpy import random
import numpy as np
import matplotlib.pyplot as plt

r = 1 # circle radius

Nt = 1e3

areas = np.array(list())

for i in range(int(1e3)):
    Nc = 0

    for point in range(int(Nt)):
        xi = random.uniform(-r, r)
        yi = random.uniform(-r, r)

        if (xi**2 + yi **2) <= r**2:
            Nc += 1

    area = Nc / Nt * (2 ** 2)
    areas = np.append(areas, area)    
        
pi_estimated_mean   = areas.mean() / r ** 2
pi_estimated_median = np.median(areas) / r ** 2

print("Pi constant (Numpy): {0}".format(np.pi))
print("Pi estimated (Mean): {0}".format(pi_estimated_mean))
print("Pi estimated (Median): {0}".format(pi_estimated_median))

plt.hist(areas, bins=30, ec='b')
plt.xlabel("Areas")
plt.title("Distribution of Estimated Areas")
plt.show()
```

    Pi constant (Numpy): 3.141592653589793
    Pi estimated (Mean): 3.144756
    Pi estimated (Median): 3.144



![png](monteCarlo-method_files/monteCarlo-method_12_1.png)

