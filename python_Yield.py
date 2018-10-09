from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math as math
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

Yield = np.array([40, 50, 50, 70, 65, 65, 80])
Fert = np.array([100, 200, 300, 400, 500, 600, 700])
Rain = np.array([10, 20, 10, 30, 20, 20, 30])

n = Yield.size
print("\nsamples : n = ",n)

Yield_mean = sum(Yield)/Yield.size
Fert_mean = sum(Fert)/Fert.size
Rain_mean = sum(Rain)/Rain.size



x1 = Fert - Fert_mean
x2 = Rain -Rain_mean
y = Yield - Yield_mean


Σx1 = np.sum(x1)
Σx2 = np.sum(x2)
Σy = np.sum(y)

x1_sq = np.sum((x1)**2)
x2_sq = np.sum((x2)**2)
y_sq = np.sum((y)**2)

Σx1y = np.sum((x1*y))
Σx1_sq = np.sum((x1)**2)
Σx2_sq = np.sum((x2)**2)
Σx2y = np.sum((x2*y))
Σx1x2 = np.sum((x1*x2))

Σx1y_Σx2_sq = Σx1y*Σx2_sq
Σx2y_Σx1x2 = Σx2y*Σx1x2
Σx1_sq_Σx2_sq = Σx1_sq*Σx2_sq
Σx1x2_sq = (Σx1x2)**2

B1 = (Σx1y_Σx2_sq - Σx2y_Σx1x2) / (Σx1_sq_Σx2_sq - Σx1x2_sq)
print("\nBeta1 = ",B1)

B2 = ((Σx2y*Σx1_sq) - (Σx1y*Σx1x2)) / (Σx1_sq_Σx2_sq - Σx1x2_sq)
print("Beta2 = ",B2)

B0 = Yield_mean - (B1*Fert_mean) - (B2*Rain_mean)
print("Beta0 = ",B0)

k = 3
print("\nNumber of Parameters B0, B1, B2 : k = ",k)

Y_hat = B0 + B1*Fert + B2*Rain


print("\nY_hat = {} + {}Fert + {}Rain".format(B0,B1,B2))

TSS = np.sum((Yield - Yield_mean)**2)
MSS = np.sum((Y_hat - Yield_mean)**2)
RSS = np.sum((Yield - Y_hat)**2)

print("\n TSS = ",TSS)
print(" MSS = ",MSS)
print(" RSS = ",RSS)

R_sq = MSS/TSS
print("\nR2= ",R_sq)

MSE = RSS/(n-k)
print("\n MSE = ",MSE)

V_B1 = MSE *( (Σx2_sq) / (Σx1_sq_Σx2_sq - Σx1x2_sq) )
SE_B1 = math.sqrt(V_B1)
print("\nVariance of Beta1 = ",V_B1)
print("Standard Error of Beta1 : SE(B1) = ",SE_B1)

V_B2 = MSE *((Σx1_sq) / (Σx1_sq_Σx2_sq - Σx1x2_sq) )
SE_B2 = math.sqrt(V_B2)
print("\nVariance of Beta2 = ",V_B2)
print("Standard Error of Beta2 : SE(B2) = ",SE_B2)
Fertsq = Fert_mean**2
Rainsq = Rain_mean**2
FertRain =Fert_mean*Rain_mean

V_B0 = MSE * ((1/n) + ((Fertsq*Σx2_sq + Rainsq*Σx1_sq - 2*FertRain*(Σx1x2))/(Σx1_sq_Σx2_sq - Σx1x2_sq)))
SE_B0 = math.sqrt(V_B0)
print("\nVariance of Beta0 = ",V_B0)

data_frame = pd.DataFrame(
    {
        "Yield": Yield

        , "Fertilizer": Fert

        , "Rainfall": Rain
    }
)

Reg = ols(formula="Yield ~ Fert + Rain", data=data_frame)
Fit2 = Reg.fit()
print("\n", Fit2.summary())
print("\n", anova_lm(Fit2))


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    data_frame["Fertilizer"]
    , data_frame["Rainfall"]
    , data_frame["Yield"]
    , color="green"
    , marker="o"
    , alpha=1
)
ax.set_xlabel("Fertilizer")
ax.set_ylabel("Rainfall")
ax.set_zlabel("Yield")

plt.show()




