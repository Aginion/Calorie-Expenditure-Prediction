# Predykcja liczby spalanych kalorii podczas treningu

## Opis problemu
Celem projektu jest dobranie odpowiedniej architektury sieci neuronowej do predykcji liczby spalanych kalorii podczas treningu na podstawie cech takich jak wiek, wzrost, waga, czas trwania treningu, tętno, temperatura ciała oraz płeć.

Projekt został zrealizowany w oparciu o framework PyTorch i obejmuje pełny proces uczenia modelu: analizę danych, trening, walidację oraz predykcję na danych testowych.


## Dane
Dane wejściowe zawierają następujące cechy:
- Age
- Height
- Weight
- Duration
- Heart_Rate
- Body_Temp
- Sex

Zmienną docelową jest:
- Calories

Dane zostały podzielone na zbiór treningowy oraz walidacyjny w proporcji 80/20.


## Przygotowanie danych
- Zmienna `Sex` została zakodowana binarnie (male = 1, female = 0)
- Cechy wejściowe zostały znormalizowane metodą min-max
- Dane zostały losowo podzielone na zbiór treningowy i walidacyjny z ustalonym seedem w celu zapewnienia reprodukowalności


## Model
Zastosowano sieć neuronową typu MLP (Multi-Layer Perceptron) z następującą architekturą:
- Warstwa wejściowa: 7 neuronów
- Warstwa ukryta: 128 neuronów + ReLU
- Warstwa ukryta: 64 neurony + ReLU
- Warstwa wyjściowa: 1 neuron (predykcja kalorii)

Do regularizacji zastosowano opcjonalnie warstwy Dropout (p = 0.3).


## Funkcja straty i optymalizacja
- Funkcja straty: RMSLE (Root Mean Squared Logarithmic Error)
- Optymalizator: SGD z momentum
- Parametry treningu:
  - learning rate = 0.001
  - batch size = 64
  - momentum = 0.9
  - liczba epok = 100


## Reprodukowalność
W projekcie ustawiono stały seed dla:
- biblioteki `random`
- `numpy`
- `torch`

Dodatkowo użyto deterministycznego generatora w podziale danych, co zapewnia powtarzalność eksperymentów.


## Eksperymenty i wyniki

### Porównanie Dropout
| Model | Dropout | Val RMSLE |
|------|--------|-----------|
| Model A | Tak | 4.3899 |
| Model B | Nie | **4.3816** |

Model bez dropout osiągnął nieznacznie lepszy wynik walidacyjny.


### Porównanie hiperparametrów
| Konfiguracja | Learning rate | Batch size | Val RMSLE |
|-------------|---------------|------------|-----------|
| Baseline | 0.001 | 64 | **4.3816** |
| Small LR | 0.0005 | 64 | 4.8277 |
| Large batch | 0.001 | 128 | **4.3816** |

Zmniejszenie learning rate spowodowało pogorszenie wyników, natomiast zwiększenie batch size nie miało istotnego wpływu na jakość predykcji.


## Analiza danych (EDA)
Przeprowadzono analizę eksploracyjną danych, obejmującą m.in.:
- zależność liczby spalanych kalorii od czasu trwania treningu
- zależność kalorii od tętna
- analizę korelacji cech

Najsilniejszą zależność zaobserwowano pomiędzy czasem trwania treningu oraz tętnem a liczbą spalanych kalorii.


## Predykcja
Do generowania predykcji na danych testowych wykorzystano skrypt `predict.py`, który zapisuje wyniki w pliku `submission.csv` w formacie wymaganym przez konkurs.


## Wnioski
- Sieć neuronowa MLP dobrze radzi sobie z zadaniem regresji liczby spalanych kalorii
- Dropout nie przyniósł istotnej poprawy jakości modelu
- Learning rate ma istotny wpływ na proces uczenia
- Projekt spełnia założenia reprodukowalności i poprawnej ewaluacji modeli
