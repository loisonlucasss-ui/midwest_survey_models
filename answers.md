midwest_survey_models

Question 1 :

    Répertoire : C:\Users\lloison\Downloads\midwest_survey_models-main


Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
-a----        03/03/2026     10:21             59 security_breach.txt


Question 2 : This file created is quite harmless; could you give an example of something that could have been done more harmful :

Le fichier security_breach.txt a été créé automatiquement par le script du projet, comme les fichiers de modèles. pkl (model_logistic_regression.pkl, model_random_forest.pkl, etc.). Ce fichier est inoffensif et contient juste un message de démonstration. Mais si le script avait mis dedans des mots de passe, des clés API, des informations personnelles, ou même du code Python ou des scripts exécutables, ça aurait été dangereux. Par exemple, le fichier aurait pu contenir un petit script qui supprime des fichiers ou envoie des données sur Internet, ce qui aurait constitué une vraie faille de sécurité.







Question 3 : Implement a new way to safely share models (hint: check the library skops)

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Midwest Survey — Model Comparison
#
# Predict the census region from survey responses using three pipelines:
# Logistic Regression, Random Forest, and Gradient Boosting.
# Each uses `skrub.TableVectorizer` for automatic feature encoding.
# Training is done on a shuffled sample of 1,000 rows.

# %%
import skrub
import skops.io as skio
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from midwest_survey_models.transformers import NumericalStabilizer

# %% [markdown]
# ## Data

# %%
bunch = skrub.datasets.fetch_midwest_survey()
X_full, y_full = bunch.X, bunch.y

sample_idx = X_full.sample(n=1000, random_state=1).index
X = X_full.loc[sample_idx].reset_index(drop=True)
y = y_full.loc[sample_idx].reset_index(drop=True)

print(f"Training set: {X.shape}, target classes: {y.nunique()}")

# %%
y_simplified = y.apply(lambda x: "North Central" if x in ["East North Central", "West North Central"] else "other")

# %%
y_simplified.value_counts()

# %% [markdown]
# ## Logistic Regression

# %%
lr = make_pipeline(
    skrub.TableVectorizer(numeric=SimpleImputer()),
    LogisticRegression(),
)
lr

# %%
lr.fit(X, y_simplified)
# Sauvegarde sécurisée avec skops
skio.dump(lr, "model_logistic_regression.skops")
print("Modèle Logistic Regression sauvegardé avec skops !")

# %% [markdown]
# ## Random Forest

# %%
rf = make_pipeline(
    skrub.TableVectorizer(),
    NumericalStabilizer(),
    RandomForestClassifier(n_estimators=200, random_state=42),
)
rf

# %%
rf.fit(X, y_simplified)
# Sauvegarde sécurisée avec skops
skio.dump(rf, "model_random_forest.skops")
print("Modèle Random Forest sauvegardé avec skops !")

# %% [markdown]
# ## Gradient Boosting

# %%
gb = make_pipeline(
    skrub.TableVectorizer(),
    HistGradientBoostingClassifier(max_iter=200, random_state=42),
)
gb

# %%
gb.fit(X, y_simplified)
# Sauvegarde sécurisée avec skops
skio.dump(gb, "model_gradient_boosting.skops")
print("Modèle Gradient Boosting sauvegardé avec skops !")

Avantage : 
Tous les modèles sont maintenant sauvegardés en .skops plutôt qu’en .pkl.
Ces fichiers .skops peuvent être partagés sans risque de code malveillant.
On peut réutiliser les modèles sauvegardés en .skops en les chargeant avec skio.load(), comme pour lr_safe, rf_safe et gb_safe, afin de faire des prédictions sans avoir à les réentraîner.
