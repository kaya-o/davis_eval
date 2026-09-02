import math

def bound(m,b, alpha):
    return (1-alpha) - max(0,(math.ceil((1-alpha)*(m+1)) - alpha*b))/m+1
print(bound(7,2,0.1))