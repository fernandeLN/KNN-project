By Oscar DAGA (404584) and Fernande LOUEKE (404719), AGH, elektronika 3 roku

# KNN-project
An implementation of the KNN algorithm

Metoda K Najbliższych Sąsiadów (k-Nearest Neighbors) należy do grupy algorytmów leniwych (lazy algorithms), czyli takich, które nie tworzą wewnętrznej reprezentacji danych uczących, lecz szukają rozwiązania dopiero w momencie pojawienia się wzorca testującego, np. do klasyfikacji. Przechowuje wszystkie wzorce uczące, względem których wyznacza odległość wzorca testowego.

W pliku "KNN from scratch" jest implementacja tego projektu bez zużycia biblioteki scikit-learn
Główne etapy są :
1. Pobranie i preparacja danych użytych do sprawdzenia
2. Tworzenie funkcji do policzenia odległości euklidesowej i mały test jej działania
3. Tworzenie głównej funkcji KNN gdzie jest policzone tą odległością z wszystkimi punktami, sortowane od dalszego do bliższego punktu, a w końcu klasyfikacja nasegoo punktu według które punkty z której klasy się bardziej powtarza
4. Następnie jest wykres accuracy według k. Z tego można znaleść optymalną wartość k aby zrobić klasyfikacja.

