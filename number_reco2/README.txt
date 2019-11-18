Description générale:

Ce programme est un programme d'apprentissage automatique écrit en python3, qui utilise la méthode des K plus proches voisins afin de faire de la reconnaissance de chiffres.
J'ai utilisé une des bases de données de MNIST qui fournit des milliers de chiffres écrits à la main
de facons différentes, afin de pouvoir "entraîner" mon programme dans un premier temps et de pouvoir
lui faire faire une prédiction dans un 2ème temps.
Les fichiers MNIST sont fournis avec le programme et s'intitulent:
-t10k-images-idx3-ubyte
-t10k-labels-idx1-ubyte
(disponibles à l'adresse suivante http://yann.lecun.com/exdb/mnist/)

Pour travailler dessus plus facilement, j'ai utilisé un programme pour regrouper ces 2 fichiers en un seul fichier CSV:

generate_csv_mninst.py
(je l'ai trouvé à l'adresse suivante: https://pjreddie.com/projects/mnist-in-csv/)
J'ai déjà généré le CSV mais si vous voulez tester par vous même : 
python3 generate_csv_mninst.py

maintenant que vous avez un fichier csv contenant tous nos chiffres écrits à la main, nous allons pouvoir utiliser notre fameux programme implémentant l'algorithme KNN (K nearest neighbours).
(pour information chaque ligne du fichier CSV est organisé de la facon suivante:
1ère colonne le chiffre représentéé
Toutes les autres colonnes : les pixels de l'image 28 par 28 en niveau de gris, correspondant à la colonne 1
(28x28 = 784, 784+1 = 785 il y a donc 785 colonnes par ligne)
)
C'est le moment d'utiliser notre fameux programme : predict_number.py

Mais tout d'abord, il faut installer les librairies suivantes:
-pandas (pip3 install pandas)
-matplotlib (pip3 install matplotlib)
-scipy (pip3 install scipy)
-sklearn (pip3 install sklearn)

Maintenant que c'est fait vous pouvez enfin lancer le programme:
python3 predict_number.py

Le programme va récupérer les 1000 premières lignes du fichier CSV, va s'entrainer sur 900 d'entres-elles
et faire des prédictions sur 100 d'entres-elles. (l39 test_size=0.1*1000=100)
Grâce à la méthode train_test_split de sklearn, les Jeux de données ne semblent pas êtres prises dans le même ordre à chaque fois.

Sur les 100 prédictions j'en affiche qu'une seule, avec le pourcentage de précision de la prédiction,
et j'affiche l'image que le programme doit reconnaitre.

Sur les 1000 lignes, avec un paramètre test_size=0.1 on a une certitude sur la prédiction d'environ 85%
On peut voir que si on augmente l'entraînement en mettant par exemple test_size=0.01, on passe à une certitude dépassant les 90%.


--------------------------------------------------------------------
--------------------------------------------------------------------

Description détaillée:

Ce programme implémente l'algorithme KNN. Beaucoup de sites parlerons beaucoup mieux de moi de cet algorithme (https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm), je vais cependant vous faire une petite description de son implémentation dans mon programme.

Le principe est d'avoir un jeu de données d'entrainement sur lequel on va comparer une donnée test et de faire un prédiction sur ce que cette donnée de test est par rapport au données d'entrainement.
Ici on prend une image en niveau de gris contenant un chiffre écrit à la main.
Le but est de déterminer quel est ce chiffre.
Le training_set contient pleins de chiffres écrits à la main.
Pour comparrer notre image à celles en BDD et savoir de laquelle elle se rapproche le plus, et pour en déduire son numéro, on utilise la fonction norme.
On prend la norme de l'image à deviner (racine(pixel1carré+pixel2carré+...)) et on la compare à la norme des images qu'on a en BDD et on garde celle qui est a la distance la plus proche : c'est elle qui est la plus proche de notre image de test.
On connaît le numéro de cette image de training, et c'est donc ce numéro qui est le plus susceptible d'être celui correspondant à notre image test.

Pour avoir le pourcentage d'erreur j'utilise la fonction accuracy_score de la librairie sklearn.

Cet algorithme des KNN est donc relativement efficace pour ce qui est de la reconnaissance de chiffres (+ de 80% de précision) en revanche, sa faiblesse pourrait être que si on met une image de test qui n'a rien à voir avec les images d'entrainement, l'algorithme nous donnerai quand même celle qu'il juge être le plus proche (exemple je lui met une image d'un chat, s'il n'a que des chiffres en données d'entrainement, il va me dire que mon image correspond à un chiffre)



