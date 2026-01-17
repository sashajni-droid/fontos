```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from pathlib import Path

# --- Weibull modell ---
def weibull(t, A, alpha, beta, t0):
    z = np.maximum(t - t0, 0) / alpha
    return A * (1 - np.exp(-(z ** beta)))

# --- 1) Beolvasás soronként (Asztalról, ; és tizedesvessző) ---
csv_path = Path.home() / "Desktop" / "data2.csv"

t_list = []
y_list = []

with open(csv_path, "r", encoding="utf-8") as f:
    fejléc = f.readline()  # első sor átugrása

    for sor in f:
        sor = sor.strip()
        if sor == "":
            continue

        darabok = sor.split(";")

        t = float(darabok[0].replace(",", "."))
        y = float(darabok[1].replace(",", "."))

        t_list.append(t)
        y_list.append(y)

t = np.array(t_list)
y = np.array(y_list)

# --- 2) pontok ábrázolása ---
plt.figure()
plt.plot(t, y, "o", label="Mért adatok")
plt.xlabel("t")
plt.ylabel("Kioldódás")
plt.title("Kioldódási adatok")
plt.legend()
plt.grid(True)
plt.show()

# --- 3) mintavételi frekvencia ---
fs = 1 / np.median(np.diff(t))

# --- 4) min / max ---
i_min, i_max = np.argmin(y), np.argmax(y)
ymin, tmin = y[i_min], t[i_min]
ymax, tmax = y[i_max], t[i_max]

# --- 5) inflexiós pont(ok) ---
win = min(21, len(y) - (len(y) + 1) % 2)
win = max(win, 7)
ys = savgol_filter(y, win, 3)
d2 = np.gradient(np.gradient(ys, t), t)
infl_idx = np.where(np.diff(np.sign(d2)) != 0)[0]
infl_t = t[infl_idx + 1]

# --- 6) max meredekség ---
dy = np.gradient(ys, t)
i_s = np.argmax(dy)
max_slope, t_slope = dy[i_s], t[i_s]

# --- 7) t67 ---
target = 0.67 * ymax
k = np.where(y >= target)[0][0]
t67 = t[k - 1] + (target - y[k - 1]) * (t[k] - t[k - 1]) / (y[k] - y[k - 1])

# --- 8) illesztés ---
p0 = [ymax, (t[-1] - t[0]) / 3, 1.0, t[0]]
bounds = (
    [0, 1e-12, 1e-6, t[0] - (t[-1] - t[0])],
    [1.5 * ymax, 10 * (t[-1] - t[0]) + 1e-9, 10, t[-1]],
)
popt, _ = curve_fit(weibull, t, y, p0=p0, bounds=bounds, maxfev=20000)

# --- 9) adat + illesztés ábra ---
tt = np.linspace(t.min(), t.max(), 400)
plt.figure()
plt.plot(t, y, "o", label="Mért adatok")
plt.plot(tt, weibull(tt, *popt), "-", label="Weibull illesztés")
plt.xlabel("t")
plt.ylabel("Kioldódás")
plt.title("Adatok + illesztés")
plt.legend()
plt.grid(True)
plt.show()

# --- 10) kiírás ---
print(f"2) fs = {fs:.6g} 1/(t-egység)")
print(f"3) min = {ymin:.6g} @ t={tmin:.6g} | max = {ymax:.6g} @ t={tmax:.6g}")
print("4) inflexiós idő(k):", np.array2string(infl_t, precision=4))
print(f"5) max meredekség = {max_slope:.6g} @ t={t_slope:.6g}")
print(f"6) t67 ≈ {t67:.6g}")
print("8) Weibull paraméterek:", [float(x) for x in popt])
```


```harom: Potencial görbe
from pathlib import Path
import matplotlib.pyplot as plt

# --------- FÁJL: ASZTAL ----------
csv_path = Path.home() / "Desktop" / "ap.csv"

time_ms = []
ap1 = []

with open(csv_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 1) vessző -> pont (tizedesvessző kezelése)
        line = line.replace(",", ".")
        # 2) pontosvessző -> vessző (elválasztó egységesítése)
        line = line.replace(";", ",")

        parts = line.split(",")

        # minimum: Time + 1 oszlop
        if len(parts) < 2:
            continue

        try:
            t = float(parts[0])
            v = float(parts[1])
        except ValueError:
            # fejléc vagy hibás sor
            continue

        time_ms.append(t)
        ap1.append(v)

# --------- 1) ÁBRÁZOLÁS ----------
plt.figure()
plt.plot(time_ms, ap1, label="1-es oszlop (AP1)")
plt.xlabel("Idő [ms]")
plt.ylabel("Feszültség [mV]")
plt.title("Akciós potenciál időfüggése (1-es oszlop)")
plt.legend()
plt.grid(True)
plt.show()

# --------- 2) POZITÍV TARTOMÁNY INTERVALLUMAI ----------
intervals = []
in_pos = False
start = None

for i in range(len(ap1)):
    if ap1[i] > 0 and not in_pos:
        in_pos = True
        start = time_ms[i]
    elif ap1[i] <= 0 and in_pos:
        in_pos = False
        end = time_ms[i - 1]
        intervals.append((start, end))

if in_pos:
    intervals.append((start, time_ms[-1]))

if intervals:
    print("Pozitív feszültségtartomány intervallum(ok):")
    for s, e in intervals:
        print(f"  Kezdő: {s} ms, Vég: {e} ms")
else:
    print("Nincs pozitív feszültségtartomány ( > 0 mV ).")

# --------- 3) MAXIMÁLIS AMPLITÚDÓ ----------
max_v = ap1[0]
max_t = time_ms[0]

for i in range(1, len(ap1)):
    if ap1[i] > max_v:
        max_v = ap1[i]
        max_t = time_ms[i]

print(f"\nMaximális amplitúdó: {max_v} mV (időpont: {max_t} ms)")

vége
```

```
from pathlib import Path
import matplotlib.pyplot as plt

# --------- FÁJL: ASZTAL ----------
csv_path = Path.home() / "Desktop" / "adat.csv"

# --------- SORONKÉNTI BEOLVASÁS + CSERÉK ----------
x = []
y = []

with open(csv_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # 1) vessző -> pont (tizedesvessző miatt)
        line = line.replace(",", ".")
        # 2) pontosvessző -> vessző (elválasztó miatt)
        line = line.replace(";", ",")

        parts = line.split(",")

        # minimum 2 mező kell (X és Y)
        if len(parts) < 2:
            continue

        try:
            xv = float(parts[0])
            yv = float(parts[1])
        except ValueError:
            # fejléc vagy hibás sor
            continue

        x.append(xv)
        y.append(yv)

# --------- ÁTLAGOLÓ SZŰRŐ (mozgóátlag) ----------
def mean_filter(data, window_size=5):
    n = len(data)
    k = window_size // 2
    out = []

    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)
        out.append(sum(data[start:end]) / (end - start))

    return out

# --------- MEDIÁN SZŰRŐ ----------
def median_filter(data, window_size=5):
    n = len(data)
    k = window_size // 2
    out = []

    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)
        window = sorted(data[start:end])
        m = len(window)
        out.append(window[m // 2])  # középső elem (páratlan ablaknál korrekt)

    return out

# Szűrt jelek
y_mean = mean_filter(y, window_size=5)
y_median = median_filter(y, window_size=5)

# --------- ÁBRÁZOLÁS (egy grafikonon) ----------
plt.figure()
plt.plot(x, y, label="Eredeti")
plt.plot(x, y_mean, label="Átlag szűrt")
plt.plot(x, y_median, label="Medián szűrt")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Zajos és szűrt adatok")
plt.legend()
plt.grid(True)
plt.show()
vége
```

