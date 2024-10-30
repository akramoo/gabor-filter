# Traitement d'un ensemble de données d'images médicales pour la détection de pneumonie

## Aperçu

Ce projet se concentre sur la préparation d'un ensemble de données d'images radiographiques thoraciques pour la détection de la pneumonie. Le processus inclut le renommage des images, l'application de filtres de Gabor pour l'extraction de textures, et l'enregistrement des caractéristiques dans un fichier CSV. Cet ensemble de caractéristiques permet une intégration efficace dans les pipelines de machine learning pour les tâches de classification. La méthode utilise des filtres de Gabor en raison de leur efficacité à capturer les motifs de texture, qui sont cruciaux pour différencier les poumons sains des poumons affectés par la pneumonie.

### Cas d'utilisation

Le code prend en charge le prétraitement d'un ensemble de données radiographiques pour la classification binaire :
- **Classes** : "NORMAL" et "PNEUMONIA"

---

## Structure du projet

```
.
├── Datasets                              # Dossier contenant l'ensemble de données d'images radiographiques
│   ├── NORMAL                            # Sous-dossier contenant les images étiquetées comme NORMALES
│   └── PNEUMONIA                         # Sous-dossier contenant les images étiquetées comme PNEUMONIE
├── .gitignore                            # Fichier Git pour exclure les fichiers inutiles du contrôle de version
├── Datasets_features.csv                 # Fichier CSV contenant les caractéristiques extraites des images
├── LICENSE                               # Licence du projet
├── README.md                             # Documentation du projet
├── dataset_relabeler.ipynb               # Notebook Jupyter pour renommer les images de l'ensemble de données
├── gabor_feature_extractor_to_csv.ipynb  # Notebook Jupyter pour appliquer les filtres de Gabor et enregistrer les caractéristiques dans un CSV
└── project_report.pdf                    # Rapport détaillé sur le traitement de l'ensemble de données et l'extraction des caractéristiques
```

### Explication des dossiers et fichiers

- **`Datasets/`** : Dossier principal contenant l'ensemble de données d'images, organisé en deux sous-répertoires :
  - **`NORMAL/`** : Contient les images radiographiques classées comme "NORMAL".
  - **`PNEUMONIA/`** : Contient les images radiographiques classées comme "PNEUMONIA".

- **`.gitignore`** : Spécifie les fichiers et dossiers à ignorer par Git, pour maintenir le dépôt propre.

- **`Datasets_features.csv`** : Fichier CSV où les caractéristiques extraites de chaque image sont stockées. Chaque ligne contient les caractéristiques de texture extraites à l'aide de filtres de Gabor, ainsi que l'étiquette et l'identifiant unique de chaque image.

- **`LICENSE`** : Fichier de licence décrivant les conditions d'utilisation, de redistribution, et les éventuelles restrictions associées à ce projet.

- **`README.md`** : Ce fichier fournit une vue d'ensemble complète du projet, de sa structure et de la fonctionnalité du code.

- **`dataset_relabeler.ipynb`** : Notebook Jupyter contenant le code pour renommer les images de l'ensemble de données. Chaque fichier d'image est renommé avec un identifiant unique incluant l'étiquette de classe et un compteur.

- **`gabor_feature_extractor_to_csv.ipynb`** : Notebook Jupyter pour appliquer les filtres de Gabor à chaque image de l'ensemble de données afin d'extraire des caractéristiques de texture. Les caractéristiques traitées et les étiquettes sont ensuite enregistrées dans `Datasets_features.csv`.

- **`project_report.pdf`** : Rapport détaillé sur la méthodologie de traitement de l'ensemble de données, l'application des filtres de Gabor, et l'ensemble de données des caractéristiques extraites. Inclut des références aux ressources pertinentes.

---

## Explication du code

### 1. Renommer l'ensemble de données

Chaque image est renommée en fonction de sa classe et d'un compteur unique afin de garantir des noms de fichiers cohérents et reconnaissables.

```python
# Renommer les images avec des identifiants uniques
for sub_dir in sub_dirs:
    folder_path = os.path.join(base_dir, sub_dir)
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                new_name = f"{sub_dir}_{global_count}.jpg"
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_name)
                os.rename(old_path, new_path)
                global_count += 1
```

### 2. Définir et appliquer les filtres de Gabor

Les filtres de Gabor sont appliqués à plusieurs orientations, échelles et fréquences pour capturer efficacement les motifs de texture.

- **Theta (Orientation)** : 0°, 45°, 90° et 135°
- **Sigma (Échelle)** : 1 et 3
- **Fréquence** : 0,05 et 0,25

Cette configuration garantit que les filtres peuvent détecter les variations de texture pertinentes pour l'analyse de la condition des poumons.

```python
# Création des filtres de Gabor avec différents paramètres
kernels = []
for theta in range(4):
    theta = theta / 4.0 * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)
```

### 3. Extraction des caractéristiques et enregistrement des données dans un CSV

Chaque image est traitée pour extraire :
- **Moyenne** : Intensité moyenne de l'image filtrée.
- **Variance** : Mesure de la force de variation de la texture.

Ces caractéristiques sont enregistrées dans un fichier CSV, fournissant un ensemble de données structuré pour les modèles de machine learning.

```python
# Extraction des caractéristiques et enregistrement dans un CSV
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["numero d'image", "caractéristique", "label"])

    for sub_dir in sub_dirs:
        folder_path = os.path.join(base_dir, sub_dir)
        
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    image = img_as_float(io.imread(image_path, as_gray=True))
                    feats = compute_feats(image, kernels)
                    numero_image = filename.split('_')[-1].split('.')[0]
                    label = filename.split('_')[0]
                    feats_str = str(feats.tolist())
                    writer.writerow([numero_image, feats_str, label])
```

---

## Résumé

Ce pipeline de code prépare efficacement un ensemble de données médical pour la détection de pneumonie en :
- **Renommant les images** pour une organisation structurée de l'ensemble de données.
- **Extrait des caractéristiques de texture avec des filtres de Gabor** qui améliorent la capacité du modèle à distinguer entre les poumons sains et ceux atteints de pneumonie.
- **Enregistrant les caractéristiques et les étiquettes au format CSV** pour une intégration facile avec des modèles de machine learning.

Cet ensemble de données enrichi, axé sur les textures, est prêt pour les tâches de classification, permettant une distinction plus précise entre les cas "NORMAL" et "PNEUMONIA".

---

