```python kioldódás
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Adatok beolvasása
tabla = pd.read_csv("data2.csv")  # Ne felejtsd el pontos helyre menteni a fájlt!

# 2. Idő és értékek kinyerése tömbbe
t = tabla.iloc[:, 0].to_numpy()
y = tabla.iloc[:, 1].to_numpy()

# 3. Ábra kirajzolása
plt.figure(figsize=(10, 5))
plt.scatter(t, y, label="Mért adatok", color="blue")
plt.xlabel("Idő (s)")
plt.ylabel("Kioldódott hatóanyag (%)")
plt.title("Diffúziós kioldódás")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Mintavételi időköz és frekvencia
mintaveteli_idokoz = t[1] - t[0]
mintaveteli_frekvencia = 1 / mintaveteli_idokoz

print("Mintavételi időköz (s):", round(mintaveteli_idokoz, 4))
print("Mintavételi frekvencia (Hz):", round(mintaveteli_frekvencia, 2))

# 5. Minimum és maximum
y_min = np.min(y)
y_max = np.max(y)
print("Minimum érték:", round(y_min, 2))
print("Maximum érték:", round(y_max, 2))

# 6. Deriváltak (sebességek és inflexiós pontok)
d1 = np.diff(y) / np.diff(t)     # Első derivált = sebesség
d2 = np.diff(d1)                 # Második derivált = "görbület"

# 7. Inflexiós pontok keresése
inflexios_idok = []
for i in range(len(d2) - 1):
    if np.sign(d2[i]) != np.sign(d2[i+1]):  # Előjelváltás
        inflexios_idok.append(t[i+1])

print("Inflexiós pontok időben (s):", [round(i, 2) for i in inflexios_idok])

# 8. Maximális kioldódási sebesség
sebessegek = np.abs(d1)
max_sebesseg = np.max(sebessegek)
max_seb_index = np.argmax(sebessegek)
max_seb_ido = t[max_seb_index]

print("Maximális sebesség:", round(max_sebesseg, 4))
print("Ez kb. ekkor történt (s):", round(max_seb_ido, 2))

# 9. 67%-os szint elérésének ideje
cel_ertek = 0.67 * y_max
ido_67 = None

for i in range(len(y)):
    if y[i] >= cel_ertek:
        ido_67 = t[i]
        break

print("67%-os szint elérésének ideje (s):", round(ido_67, 2) if ido_67 else "Nem érte el")

```


```harom: Potencial görbe
import pandas as pd
import matplotlib.pyplot as plt

# CSV fájl beolvasása
tabla = pd.read_csv("adat(1).csv")

# Idő és feszültség oszlop kiválasztása
ido = tabla["X"]  # idő ms-ban
jel = tabla["Y"]  # feszültség mV-ban

# Ábra rajzolása
plt.figure(figsize=(10, 5))
plt.plot(ido, jel, color="orange", label="Akciós potenciál görbe")
plt.xlabel("Idő (ms)")
plt.ylabel("Feszültség (mV)")
plt.title("Akciós potenciál")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Maximális amplitúdó meghatározása
max_amplitudo = jel.max()
print("Maximális amplitúdó (mV):", max_amplitudo)

# Pozitív feszültségtartomány kezdete és vége
pozitiv_indexek = jel[jel > 0].index

if not pozitiv_indexek.empty:
    kezdet = ido[pozitiv_indexek[0]]
    veg = ido[pozitiv_indexek[-1]]
    print("Pozitív tartomány kezdete (ms):", kezdet)
    print("Pozitív tartomány vége (ms):", veg)
else:
    print("Nincs pozitív tartomány.")

vége
```

```   ZAJ
import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV fájl beolvasása
tabla = pd.read_csv("adat.csv")

# 2. X és Y oszlop beolvasása
x = tabla["X"]
y = tabla["Y"]

# 3. Szűrési ablak mérete (csak páratlan szám legyen pl. 3, 5, 11 stb.)
ablak_meret = 11

# 4. Két új lista az átlagolt és medián szűrt adatoknak
y_atlagolt = []
y_median = []

# 5. Végigmegyünk az adatokon, és minden pontra számítunk szűrt értéket
for i in range(len(y)):
    kezdo = i - ablak_meret // 2
    vegso = i + ablak_meret // 2 + 1

    # Ha az ablak kilógna, korrigáljuk
    if kezdo < 0:
        kezdo = 0
    if vegso > len(y):
        vegso = len(y)

    resz = y[kezdo:vegso]

    # Átlagoló szűrő kiszámítása
    atlag = sum(resz) / len(resz)
    y_atlagolt.append(atlag)

    # Medián szűrő kiszámítása
    rendezett = sorted(resz)
    median = rendezett[len(rendezett) // 2]
    y_median.append(median)

# 6. Ábrázolás: eredeti és szűrt jelek egy grafikonon
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Eredeti jel", color="gray")
plt.plot(x, y_atlagolt, label="Átlagoló szűrő", color="blue")
plt.plot(x, y_median, label="Medián szűrő", color="red")

# 7. Tengelyfeliratok, cím, jelmagyarázat
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Eredeti, átlagolt és medián szűrt jel")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

vége
```

```5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Adatok beolvasása CSV fájlból
tabla = pd.read_csv("5adat.csv")  # Cseréld ki a fájl nevét, ha szükséges

# 2. X és Y oszlopok kiválasztása és numpy tömbbé alakítása
x = tabla["X"].to_numpy()
y = tabla["Y"].to_numpy()

# 3. R² érték kiszámítása - minél közelebb van 1-hez, annál jobb az illeszkedés
def r2_ertek(y_valodi, y_becsult):
    modell_hiba = np.sum((y_valodi - y_becsult) ** 2)
    teljes_szoras = np.sum((y_valodi - np.mean(y_valodi)) ** 2)
    return 1 - modell_hiba / teljes_szoras

# 4. Több modell (polinom) illesztése, 1., 2. és 3. fokú
modellek = []

for fok in [1, 2, 3]:
    egyutthatok = np.polyfit(x, y, fok)              # polinom együtthatók
    polinom = np.poly1d(egyutthatok)                 # polinom objektum
    y_becsult = polinom(x)                           # becsült Y értékek
    r2 = r2_ertek(y, y_becsult)                      # R² érték számítása

    modellek.append((f"{fok}.fokú polinom", fok, egyutthatok, y_becsult, r2))

# 5. Legjobb modell kiválasztása R² alapján
modellek.sort(key=lambda elem: elem[4], reverse=True)
legjobb_nev, legjobb_fok, legjobb_param, legjobb_y, legjobb_r2 = modellek[0]

# 6. Minden modell R² értékének kiíratása
print("Modellek R² értékei:")
for nev, fok, param, yhat, r2 in modellek:
    print(f"{nev}: R² = {r2:.6f}")

# 7. Legjobb modell adatai
print("\nLegjobb illeszkedés:")
print("Modell:", legjobb_nev)
print("Együtthatók:", legjobb_param)
print(f"R² = {legjobb_r2:.6f}")

# 8. Eredeti és illesztett adatok ábrázolása
plt.figure(figsize=(10, 5))
plt.plot(x, y, label="Eredeti adatok", color="gray")

# A görbe szépen fusson: X-eket rendezzük növekvőbe
rendezes = np.argsort(x)
x_rendezett = x[rendezes]
y_rendezett = legjobb_y[rendezes]

plt.plot(x_rendezett, y_rendezett,
         label=f"Illesztés: {legjobb_nev} (R² = {legjobb_r2:.4f})",
         color="blue")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Eredeti adatsor és legjobb illesztés")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

