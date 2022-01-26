By Oscar DAGA and Fernande LOUEKE
AGH, WIET, elektronika 3 roku

# KNN-project
An implementation of the KNN algorithm

Metoda K Najbliższych Sąsiadów (k-Nearest Neighbors) należy do grupy algorytmów leniwych (lazy algorithms), czyli takich, które nie tworzą wewnętrznej reprezentacji danych uczących, lecz szukają rozwiązania dopiero w momencie pojawienia się wzorca testującego, np. do klasyfikacji. Przechowuje wszystkie wzorce uczące, względem których wyznacza odległość wzorca testowego.

W pliku "KNN from scratch" jest implementacja tego projektu bez zużycia biblioteki scikit-learn
Główne etapy są :
1. Pobranie i preparacja danych użytych do sprawdzenia
2. Tworzenie funkcji do obliczania odległości euklidesowej i mały test jej działania
3. Tworzenie głównej funkcji KNN gdzie jest policzone tą odległością z wszystkimi punktami, sortowane od dalszego do bliższego punktu, a w końcu klasyfikacja naszego punktu według które punkty z której klasy się bardziej powtarza
4. Następnie jest wykres accuracy według k. Z tego można znaleźć optymalną wartość k aby zrobić klasyfikacja.
5. Z tego wynika że optymalna wartość k jest 5, a to znaczy, że z odległości z 5 punktami wybrane będzie label który się powtarza najczęściej. Ten label będzie klasą naszego badanego punktu. Lepiej wybrać k nieparzysty, żeby nie utrudnić wybór klasy punktu testowego.
6. Uruchamianie programu, ale teraz z optymalną wartością k.
7. W tym projekcie zostało zrealizowane przykładowo klasyfikacja następujących punktów (80,0) i (45, 150000), gdzie 80 i 45 to wiek a 0 i 150000 wynagrodzenie. Otrzymane wyniki są zgodne z wykresem wiek w zależności od wynagrodzenia, który pokazuje według tych kryteriów, kto kupował gaming set a kto nie.

Dla weryfikacji poprawności programu, zostało stworzone plik "KNN_with_scikit-learn", gdzie użyte jest KNeighborsClassifier z biblioteki sklearn.
Po wygenerowaniu wykresu accuracy w zależności od wartości k, wynika że optymalna wartość k jest również 5. Na podstawie tej wartości, zrobione jest klasyfikacja tych samych punktów jak powyżej i dokładne te same wyniki są otrzymane.

Program jest tak zrobione, że można robić te analizy dla dowolnych danych. Wystarczy zmienić ścieżki do pliku z danymi, i w zależności od pliku, dokonać niewielkie zmian przy otwieraniu pliku w programie.

KNN jest prosty w użyciu i w implementacji. Jak można zobaczyć w tym programie, on nadaje się do klasyfikacji a klasyfikacja nowych przypadków jest realizowana na
bieżąco, tj. gdy pojawia się potrzeba klasyfikacji nowego przypadku. Jednak posiada duże wymogi pamięciowe jak musi przechować informacje o wszystkich przypadkach testowych w pamięci i czas dokonania klasyfikacji zwiększa się wraz z powiększaniem się zbioru danych, ponieważ zawsze trzeba wyliczyć odległość do wszystkich obiektów ze zbioru danych.

-------------************************************************************************************************************---------------------------------------------------

The K Nearest Neighbors method belongs to the group of lazy algorithms, i.e. those that do not create an internal representation of the training data, but look for a solution only when a testing pattern appears, e.g. for classification. Stores all training patterns against which it determines the distance of the test pattern.

In the "KNN from scratch" file, we implement the algorithm without using the scikit-learn library
The main steps are:
1. Collection and preparation of data used for verification
2. Creation of a function to calculate the Euclidean distance
3. Creation of the main KNN function where the euclidean distance is calculated for all points, sorted from distant to closer points, and finally the classification of our test point.(If our test point is A and it has 4 nearest neighbors from class 1 against 3 neighbors from class 2, then point A is from class 1).
4. Then there is the accuracy graph as a function of k. From this you can find the optimal value of k to do the classification.
5. After verification, the optimal value of k is 5, which means that from a distance with 5 points the label that repeats the most often will be selected. This label will be the class of our test point. It is better to choose k odd so as not to make the choice of the test point class difficult.
6. Running the program but now with the optimal k value.
7. In this project, for example, the following points were classified (80,0) and (45, 150000), where 80 and 45 are age and 0 and 150000 are remuneration. 
To verify the correctness of the program, the file "KNN_with_scikit-learn" has been created, where the KNeighborsClassifier from the sklearn library is used.
After generating a plot of accuracy as a function of k, the optimal value of k is also 5. Based on this value, a classification of the same points as above is made and exact same results are obtained.

The program is made so that you can do these analyzes for any data. You just need to change the paths to the data file, and depending on the file, make slight changes when opening the file in the program.

KNN is simple to use and implement. As can be seen in this program, it is qualified for classification and the classification of new cases is carried out on a regular basis, i.e. when there is a need to classify a new case. However, it has high memory requirements as it has to keep information about all test cases in memory, and the classification time increases as the dataset grows, since the distance to all objects in the dataset must always be computed.
