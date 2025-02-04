from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 

bank = pd.read_csv('bank-additional-full.csv', delimiter=';') 

# nettoyage de données et numérisation
for col in bank.columns:
    if bank[col].dtype == 'object':  # Vérifier si la colonne est catégorielle
        bank[col] = label_encoder.fit_transform(bank[col])  # Transformer les catégories en nombres

# SEPARATION LES CARACTERISTIQUES ET LA CIBLE

X = bank.drop(columns=["y"])  # Variables indépendantes
y = bank["y"]  # Variable cible

# Séparation en ensemble d'entraînement (70%) et de test (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# ENTRAINEMENT DU MODELE

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faire les prédictions 
y_pred = model.predict(X_test) 

# Calcul de l'accuracy 
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy:.2f}") 

#Affiche le rapport de classification
from sklearn.metrics import classification_report
y_probs = model.predict_proba(X_test)[:, 1]  # Probabilité d'être "yes"
new_threshold = 0.4  # Abaisser le seuil
y_pred_adjusted = (y_probs > new_threshold).astype(int)
print(classification_report(y_test, y_pred_adjusted)) 

# Entraîner un modèle
model = RandomForestClassifier()
model.fit(X, y)

# Sauvegarder le modèle
joblib.dump(model, 'modèle.pkl')
print("Modèle sauvegardé sous 'modèle.pkl'") 
