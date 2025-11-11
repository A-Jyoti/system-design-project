import math
x= 0.5
y=0.7
z=1.2
d1= 2
d2= 0.75
l = math.sqrt(x**2 -((x**2 - y**2 + d1**2)/(2*d1))**2)
m = math.sqrt(z**2 -((x**2 - y**2 + d2**2)/(2*d2))**2)
print(f"coordinates:({l}, {m})")