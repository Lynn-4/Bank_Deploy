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

X = bank.drop(columns=['y','contact','month','loan','day_of_week','duration','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed'],axis=1])  # Variables indépendantes
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

# Prédiction sur l'ensemble de test
    y_pred = clf.predict(x_test)

    # Évaluation du modèle
    st.write("Matrice de confusion :")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("\nPrécision :", accuracy_score(y_test, y_pred))
    

with tabs[3]:
    st.write("Prédictions")
    #code python
    model=joblib.load("modèle.pkl")
    feature1=st.number_input("age",value=0.0)
    feature2=st.number_input("job",value=0.0)
    feature3=st.number_input("marital",value=0.0)
    feature4=st.number_input("education",value=0.0)
    feature5=st.number_input("default",value=0.0)
    feature6=st.number_input("housing",value=0.0)
    if st.button("prédire"):
       input_data=np.array([[feature1,feature2,feature3,feature4,feature5,feature6]])
       prediction=model.predict(input_data)
       st.write(prediction[0].item())
       if prediction==0:
           st.write("Pas de souscription")
       else:
           st.write("Souscription")
           
with tabs[4]:
    st.write("Par POUOKAM_Lynn")
    st.write("Management et Techniques Quantitatives")
    st.write("Email: lynnpdea@gmail.com") 
