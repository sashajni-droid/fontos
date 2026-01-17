KIOLDÓDÁS
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

# -------- 7) Weibull modell (gyakori kioldódási függvény) --------
def weibull(t, A, alpha, beta, t0):
    z = np.maximum(t - t0, 0) / alpha
    return A * (1 - np.exp(-(z ** beta)))

# -------- beolvasás soronként + átalakítás --------
t, y = [], []

with open("data2.csv", "r", encoding="utf-8") as f:
    next(f)  # fejléc átugrása
    for line in f:
        if not line.strip():
            continue
        a, b = line.strip().split(";")      # oszlopelválasztó
        a = a.replace(",", ".")              # tizedesvessző -> pont
        b = b.replace(",", ".")
        t.append(float(a))
        y.append(float(b))

t = np.array(t)
y = np.array(y)

# -------- 1) adatpontok ábrázolása --------
plt.figure()
plt.plot(t, y, "o", label="Mért adatok")
plt.xlabel("Idő (t)")
plt.ylabel("Kioldódás")
plt.title("Diffúziós kioldódási adatok")
plt.legend()
plt.grid(True)
plt.show()

# -------- 2) mintavételi frekvencia --------
fs = 1 / np.median(np.diff(t))

# -------- 3) min / max --------
i_min, i_max = np.argmin(y), np.argmax(y)
ymin, ymax = y[i_min], y[i_max]
tmin, tmax = t[i_min], t[i_max]

# -------- simítás a deriváltakhoz --------
win = min(21, len(y) - (len(y)+1)%2)
win = max(win, 7)
ys = savgol_filter(y, win, 3)

# -------- 4) inflexiós pont(ok) --------
d2 = np.gradient(np.gradient(ys, t), t)
infl_t = t[1:][np.diff(np.sign(d2)) != 0]

# -------- 5) max meredekség --------
dy = np.gradient(ys, t)
i_s = np.argmax(dy)
max_slope, t_slope = dy[i_s], t[i_s]

# -------- 6) t_67 --------
target = 0.67 * ymax
k = np.where(y >= target)[0][0]
t67 = t[k-1] + (target - y[k-1]) * (t[k]-t[k-1]) / (y[k]-y[k-1])

# -------- 7–8) illesztés + paraméterek --------
p0 = [ymax, (t[-1]-t[0])/3, 1.0, t[0]]
popt, _ = curve_fit(weibull, t, y, p0=p0, maxfev=20000)

# -------- 9) adat + illesztés --------
tt = np.linspace(t.min(), t.max(), 400)
plt.figure()
plt.plot(t, y, "o", label="Mért adatok")
plt.plot(tt, weibull(tt, *popt), "-", label="Weibull illesztés")
plt.xlabel("Idő (t)")
plt.ylabel("Kioldódás")
plt.title("Adatok és illesztett függvény")
plt.legend()
plt.grid(True)
plt.show()

# -------- eredmények --------
print("2) fs =", fs)
print("3) min =", ymin, " t =", tmin, " | max =", ymax, " t =", tmax)
print("4) inflexiós idő(k):", infl_t)
print("5) max meredekség =", max_slope, " t =", t_slope)
print("6) t67 =", t67)
print("8) Weibull paraméterek (A, alpha, beta, t0):", popt)
# fontos
