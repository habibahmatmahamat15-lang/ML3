import streamlit as st
import pandas as pd
import numpy as np

# ==========================
#     CHARGEMENT DONNÉES
# ==========================
st.title("📈 Application de Prédiction du Churn")

df = pd.read_csv("Base_modif.csv")

x = df.drop("CHURN", axis=1)
y = df["CHURN"]

# Split des données
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# ==========================
#     CHOIX DU MODÈLE
# ==========================
st.sidebar.title("⚙️ Paramètres du modèle")
modele = st.sidebar.selectbox(
    "Sélectionne ton modèle",
    ["SVM", "Naive Bayes", "Régression Logistique", "K plus proche voisin", "Arbre de décision"]
)


from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Modèles disponibles
if modele == "SVM":
    model = SVC()
elif modele == "Naive Bayes":
    model = GaussianNB()
elif modele == "Régression Logistique":
    model = LogisticRegression()
elif modele == "K plus proche voisin":
    model = KNeighborsClassifier()
elif modele == "Arbre de décision":
    model = DecisionTreeClassifier()

# Entraînement
model.fit(x_train, y_train)
prediction_test = model.predict_proba(x_test)

# ==========================
#     PERFORMANCE
# ==========================
st.subheader("📊 Performance du modèle sélectionné")

from sklearn.metrics import ( accuracy_score, confusion_matrix )
accuracy = accuracy_score(y_test, prediction_test)
st.write(f"**Accuracy :** {accuracy:.3f}")

# Matrice de confusion

import seaborn as sns
import matplotlib.pyplot as plt
st.write("### 🔎 Matrice de Confusion")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, prediction_test), annot=True, fmt="d", cmap="Blues")
st.pyplot(fig)

# ==========================
#     FONCTION PRÉDICTION
# ==========================
def input_value(MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE,
                DATA_VOLUME, REGULARITY):

    data = np.array([MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE,
        DATA_VOLUME, REGULARITY ])

    prediction = model.predict(data.reshape(1, -1))
    return prediction


# ==========================
#     SAISIE UTILISATEUR
# ==========================
st.subheader("📝 Entrez les valeurs du client")

col1, col2 = st.columns(2)

with col1:
    MONTANT = st.number_input("MONTANT")
    FREQUENCE_RECH = st.number_input("FREQUENCE_RECH")
    REVENUE = st.number_input("REVENUE")
    ARPU_SEGMENT = st.number_input("ARPU_SEGMENT")

with col2:
    REGULARITY = st.number_input("REGULARITY")
    FREQUENCE = st.number_input("FREQUENCE")
    DATA_VOLUME = st.number_input("DATA_VOLUME")

if st.button("🔍 Lancer la prédiction"):
    resultat = input_value(
        MONTANT, FREQUENCE_RECH, REVENUE, ARPU_SEGMENT, FREQUENCE,
        DATA_VOLUME, REGULARITY
    )

    st.subheader(" 🎯 Résultat de la prédiction :")

    if resultat == 1:
        st.error("🚨 Le client risque de churner.")
        
        st.subheader(" 💡 Conseils pour réduire le churn :")
        st.write("- Proposer une offre promotionnelle ou remise personnalisée.")
        st.write("- Améliorer la qualité du service dans sa zone.")
        st.write("- Réduire les temps d’attente ou les pannes réseau.")
        st.write("- Envoyer un message de ré-engagement ou un bonus de fidélité.")
        st.write("- Analyser son historique pour comprendre ce qui a diminué son activité.")
    else:
        st.success("✅ Le client ne risque pas de churner.")
        
        st.subheader(" 💡 Conseils pour maintenir ce client :")
        st.write("- Continuer à proposer un bon rapport qualité/prix.")
        st.write("- Offrir des récompenses pour fidéliser davantage.")
        st.write("- Suivre sa consommation pour proposer des offres adaptées.")
        st.write("- Maintenir une bonne qualité réseau dans sa zone.")
        st.write("- Encourager l'utilisation des services à forte valeur (data, appels, etc.).")
