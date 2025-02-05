# Import the required packages
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt 
import seaborn as sns 

# Page configuration
st.set_page_config(
    page_title="Bank-additional Classification",
    page_icon="assets/icon/icon.png",  # Assurez-vous que ce chemin est correct
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enable Altair dark theme with error handling
try:
    alt.themes.enable("dark")
except Exception as e:
    st.warning(f"Le thème 'dark' n'a pas pu être activé : {e}")
# Initialize page_selection in session state if not already set
if 'page_selection' not in st.session_state:
    st.session_state.page_selection = 'about'  # Default page

# Function to update page_selection
def set_page_selection(page):
    st.session_state.page_selection = page

with st.sidebar:

    st.title('Bank Classification')

    # Page Button Navigation
    st.subheader("Pages")

    if st.button("About", use_container_width=True, on_click=set_page_selection, args=('about',)):
        st.session_state.page_selection = 'about'
    
    if st.button("Dataset", use_container_width=True, on_click=set_page_selection, args=('dataset',)):
        st.session_state.page_selection = 'dataset'
        
    if st.button("EDA", use_container_width=True, on_click=set_page_selection, args=('eda',)):
        st.session_state.page_selection = "eda"

    if st.button("Machine Learning", use_container_width=True, on_click=set_page_selection, args=('machine_learning',)): 
        st.session_state.page_selection = "machine_learning"

    if st.button("Prediction", use_container_width=True, on_click=set_page_selection, args=('prediction',)): 
        st.session_state.page_selection = "prediction"

    if st.button("Conclusion", use_container_width=True, on_click=set_page_selection, args=('conclusion',)):
        st.session_state.page_selection = "conclusion" 


    # Save the selected page in the session state
    selected_page = st.radio("Navigate to:", list(pages.keys()))
    st.session_state.page_selection = pages[selected_page]

    # Abstract and project details
    st.subheader("Abstract")
    st.markdown("A Streamlit dashboard highlighting the results of training two classification models using the Iris flower dataset from Kaggle.")
    st.markdown("📊 [Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)")
    st.markdown("📗 [Google Colab Notebook](https://colab.research.google.com/drive/1KJDBrx3akSPUW42Kbeepj64ZisHFD-NV?usp=sharing)")
    st.markdown("🐙 [GitHub Repository](https://github.com/Zeraphim/Streamlit-Iris-Classification-Dashboard)")
    st.markdown("by: [`Zeraphim`](https://jcdiamante.com)")

# -------------------------
# Page content logic
def load_data():
    """Function to load the Bank-additional-full dataset."""
    try:
        return pd.read_csv("bank-additional-full.csv", delimiter=";")
    except FileNotFoundError:
        st.error("Le fichier 'bank-additional-full.csv' est introuvable. Assurez-vous qu'il est dans le bon répertoire.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is missing

def render_about():
    """Renders the About page."""
    st.title("Exploration des données de Banque-Télé-Marketing")
    st.subheader("Description des données")
    st.write("Cette application explore les données de jeu de données d'une banque, met en œuvre des modèles d'apprentissage automatique et visualise les résultats.")
    st.write("Elle inclut une analyse exploratoire, un pré-traitement des données, et des prédictions basées sur des modèles de classification.")
    st.markdown("**Construit avec :** Streamlit, Pandas, Altair")
    st.markdown("**Auteur :** POUOKAM_Lynn")

def render_dataset(df):
    """Renders the Dataset page."""
    st.title("Dataset Overview")
    if df.empty:
        st.error("Aucune donnée à afficher. Veuillez vérifier le fichier bank-additional-full.csv.")
    else:
        st.dataframe(df) 
        st.write("Shape of the dataset:", df.shape)

def render_eda(df):
    """Renders the EDA page."""
    st.title("Exploratory Data Analysis (EDA)")

    # Vérifier si les colonnes nécessaires sont présentes
    required_columns = {"age", "job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "duration", "campaign", "pdays", "previous", "poutcome", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m"," nr.employed", "y"}
    if not required_columns.issubset(df.columns):
        st.error(f"Le fichier de données doit contenir les colonnes suivantes : {required_columns}")
        return

    # Premier graphique : job vs y
    st.subheader("Relation entre le job et la décision du client") 
    if "unknown" in df["job"].values:
        mode_job = df["job"].mode()[0]  # Trouver le(s) mode(s)
        if not mode_value.empty: 
            df["job"] = df["job"].replace("unknown", mode_job[0])  # Remplacer "unknown"
    # Afficher le résultat
            print(df.groupby(['y', 'job'])['job'].count().unstack(level=0)) 
# Afficher le graphe 
            sns.countplot(x=df["y"], hue=df["job"])
            plt.show()  


    # Ajouter un tableau récapitulatif des statistiques descriptives
    st.subheader("Statistiques descriptives")
    st.write(df.describe())

    # Histogrammes des distributions avec Altair
    st.subheader("Relation entre la situation matrimoniale et la décision du client") 
    if "unknown" in df["marital"].values:
        mode_marital = df["marital"].mode()[0]  # Trouver le(s) mode(s)
        if not mode_value.empty: 
            df["marital"] = df["marital"].replace("unknown", mode_marital[0])  # Remplacer "unknown"
    # Afficher le résultat
            print(df.groupby(['y', 'marital'])['marital'].count().unstack(level=0)) 
# Afficher le graphe 
            sns.countplot(x=df["y"], hue=df["marital"])
            plt.show() 

    # Nouveau graphique : distribution de l'age
    st.subheader("Distribution de l'age")
    st.write("Statistiques descriptives pour **age** :")
    st.write(df["petal_width"].describe())

# Graphique seaborn pour la distribution de 
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(15, 5))
    sns.countplot(x="age", data=df)
    plt.title("Distribution de l'age") 

# Histogramme de répartition du niveau scolaire et de la décison du client
     st.subheader("Relation entre le niveau scoliare et la décision du client")
if "unknown" in df["education"].values:
        mode_education = df["education"].mode()[0]  # Trouver le(s) mode(s)
    if not mode_value.empty: 
        df["education"] = df["education"].replace("unknown", mode_education[0])  # Remplacer "unknown"
        
    # Afficher le résultat
        print(df.groupby(['y', 'education'])['education'].count().unstack(level=0)) 
        
# Afficher le graphe 
        sns.countplot(x=df["y"], hue=df["education"])
        plt.show() 

 # Histogramme de répartiton des logements en fonction de la décision
     st.subheader("Relation entre le logement et la décision du client") 
    if "unknown" in df["housing"].values:
        mode_housing = df["housing"].mode()[0]  # Trouver le(s) mode(s)
        if not mode_value.empty: 
            df["housing"] = df["housing"].replace("unknown", mode_housing[0])  # Remplacer "unknown"
    # Afficher le résultat
            print(df.groupby(['y', 'housing'])['housing'].count().unstack(level=0)) 
# Afficher le graphe 
            sns.countplot(x=df["y"], hue=df["housing"])
            plt.show() 
    
    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    plt.clf()  # Nettoyer pour éviter des conflits avec d'autres graphiques
    
