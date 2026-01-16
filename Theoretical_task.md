
## 1. Architektura Sieci

* **Wejście:** 2 cechy $x = [x_1, x_2]^T$
* **Warstwa ukryta:** 2 neurony + funkcja aktywacji **ReLU**
* **Wyjście:** 1 neuron (brak aktywacji, regresja liniowa)
* **Funkcja straty:** Mean Squared Error (MSE): $L = (\hat{y} - y)^2$
* **Dane treningowe:** $x = [2, 3]^T$, $y = 5$

---

## 2. Obliczenia: Backpropagation

### Przypadek A: Inicjalizacja parametrami 0.0
Wszystkie wagi $W$ i biasy $b$ są zainicjalizowane zerami.

#### 1. Forward Pass

$$\begin{aligned}
z_1 &= W_1 x + b_1 = \vec{0} \\
a_1 &= \text{ReLU}(z_1) = [0, 0]^T \\
z_2 &= W_2 a_1 + b_2 = 0 \\
\hat{y} &= 0 
\end{aligned}$$

**Koszt (Loss):**
$$
L = (0 - 5)^2 = 25
$$

#### 2. Backward Pass

Obliczamy gradienty:

**Pochodna błędu:**
$$
\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y) = -10
$$

**Warstwa wyjściowa:**
$$
\begin{aligned}
\frac{\partial L}{\partial b_2} &= -10 \\
\frac{\partial L}{\partial W_2} &= -10 \cdot a_1^T = [0, 0]
\end{aligned}
$$

**Warstwa ukryta:**
$$
\begin{aligned}
\delta_1 &= (W_2^T \cdot \text{grad}) \odot \text{ReLU}'(z_1) = 0 \\
\frac{\partial L}{\partial W_1} &= \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
\end{aligned}
$$

> **Wniosek:** Sieć **nie uczy się**. Jedynym niezerowym gradientem jest bias wyjściowy $b_2$. Wagi $W$ pozostają zerami (problem martwych neuronów).

---

### Przypadek B: Inicjalizacja parametrami 1.0

Wszystkie wagi $W$ i biasy $b$ są zainicjalizowane jedynkami.

#### 1. Forward Pass

$$
\begin{aligned}
z_1 &= \begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix} \begin{bmatrix}2 \\ 3\end{bmatrix} + \begin{bmatrix}1 \\ 1\end{bmatrix} = \begin{bmatrix}6 \\ 6\end{bmatrix} \\
a_1 &= \text{ReLU}([6, 6]^T) = [6, 6]^T \\
z_2 &= [1, 1] \cdot [6, 6]^T + 1 = 13 \\
\hat{y} &= 13
\end{aligned}
$$

**Koszt (Loss):** $$L = (13 - 5)^2 = 64$$

#### 2. Backward Pass

**Pochodna błędu:**
$$
\frac{\partial L}{\partial \hat{y}} = 2(13 - 5) = 16
$$

**Warstwa wyjściowa ($W_2, b_2$):**

$$
\frac{\partial L}{\partial b_2} = 16
$$

$$
\frac{\partial L}{\partial W_2} = 16 \cdot [6, 6] = [96, 96]
$$

**Warstwa ukryta ($W_1, b_1$):**

Pochodna ReLU dla wartości dodatnich to 1.

$$
\delta_1 = (W_2^T \cdot 16) \odot [1, 1]^T = [16, 16]^T
$$

$$
\frac{\partial L}{\partial b_1} = [16, 16]^T
$$

$$
\frac{\partial L}{\partial W_1} = \delta_1 \cdot x^T
$$

$$
\frac{\partial L}{\partial W_1} = \begin{bmatrix} 16 \\ 16 \end{bmatrix} \cdot [2, 3] = \begin{bmatrix} 32 & 48 \\ 32 & 48 \end{bmatrix}
$$

> **Wniosek:** Sieć się uczy (gradienty są niezerowe), ale występuje **problem symetrii**. Oba neurony ukryte mają identyczne gradienty, więc będą uczyć się dokładnie tej samej funkcji. Wymagana jest losowa inicjalizacja.

---

## 3. Część Teoretyczna

### Q1: Dlaczego sieci neuronowe zamiast `if-else`?

Ręczne programowanie reguł (hardcoding) jest nieefektywne w złożonych problemach z kilku powodów:

1.  **Złożoność i "Ukryta Wiedza":** W zadaniach takich jak rozpoznawanie obrazu nie istnieją jawne reguły (np. jak zdefiniować "kota" if-ami?). Wiedza jest rozproszona w parametrach.
2.  **Generalizacja:** Program z if-ami działa tylko dla przewidzianych warunków. Sieć neuronowa aproksymuje funkcję ciągłą, radząc sobie z danymi, których nigdy wcześniej nie widziała (np. $x=5.01$ zamiast sztywnego $5$).
3.  **Klątwa wymiarowości:** Przy setkach wejść liczba kombinacji warunków rośnie wykładniczo, co czyni kod niemożliwym do utrzymania.

### Q2: Rola Funkcji Aktywacji

Funkcje aktywacji (np. ReLU, Sigmoid, Tanh) są niezbędne do modelowania nieliniowości.

* **Cel:** Pozwalają sieci aproksymować złożone, nieliniowe funkcje (zgodnie z *Universal Approximation Theorem*).
* **Brak aktywacji:** Usunięcie funkcji aktywacji sprawia, że sieć – niezależnie od głębokości – staje się matematycznie równoważna zwykłej **regresji liniowej**. Złożenie funkcji liniowych jest zawsze funkcją liniową:
    $$f(g(x)) = W_2(W_1 x + b) = W'x + b'$$

### Q3: Dropout (Porzucanie)

Dropout to technika **regularyzacji** zapobiegająca przeuczeniu (overfitting).

* **Działanie:** W trakcie treningu losowo "zeruje" wybrane neurony z określonym prawdopodobieństwem $p$.
* **Efekt:**
    * Zapobiega poleganiu sieci na pojedynczych neuronach (wymusza redundancję).
    * Działa jak trenowanie zespołu (ensemble) wielu różnych pod-sieci, co zwiększa zdolność modelu do generalizacji.
