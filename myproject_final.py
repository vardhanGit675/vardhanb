import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

st.set_page_config(page_title="Real-Time Air Quality Index (AQI)", layout="wide")

# ========================================
# DATA LOADING & PREPROCESSING
# ========================================
df = pd.read_csv("/Users/chebroluvardhan/Desktop/ML/myMLDataset.csv")
df.dropna(inplace=True)

st.title("Real-Time Air Quality Index (AQI)")
st.sidebar.title("Air Quality Index (AQI) Analysis")
st.markdown("""<style>div.block-container {padding-top: 2rem;}</style>""", unsafe_allow_html=True)

st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Column Names</span>", unsafe_allow_html=True)
st.sidebar.write(df.columns.tolist())

# ========================================
# FEATURE ENGINEERING
# ========================================
df['combined_pollutant'] = (df['pollutant_min'] + df['pollutant_max'] + df['pollutant_avg']) / 3

def categorize_aqi(value):
    if value <= 50:
        return 'Good'
    elif value <= 100:
        return 'Moderate'
    elif value <= 200:
        return 'Poor'
    else:
        return 'Very Poor'

df['AQI_Category'] = df['combined_pollutant'].apply(categorize_aqi)

st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Quick Preview of New Columns</span>", unsafe_allow_html=True)
st.sidebar.write(df[['pollutant_avg', 'AQI_Category']].head(10))

# ========================================
# PREPARE DATA FOR MODELING
# ========================================
X = df[['pollutant_min', 'pollutant_max', 'latitude', 'longitude']]
y = df['AQI_Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================
# LOGISTIC REGRESSION WITH GRIDSEARCH
# ========================================
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Logistic Regression (GridSearchCV)</span>", unsafe_allow_html=True)
st.sidebar.write("")

log_reg_pipeline = Pipeline([
    ("feature_scaling", StandardScaler()),
    ("logistic_regression", LogisticRegression(max_iter=1000, random_state=42))
])

log_reg_params = {
    'logistic_regression__C': [0.1, 1, 10],
    'logistic_regression__penalty': ['l2'],
    'logistic_regression__solver': ['lbfgs', 'liblinear']
}

log_reg_grid = GridSearchCV(log_reg_pipeline, log_reg_params, cv=5, n_jobs=-1)
log_reg_grid.fit(X_train, y_train)

st.sidebar.write(f"Best Params: {log_reg_grid.best_params_}")
st.sidebar.write(f"Best CV Score: {log_reg_grid.best_score_:.4f}")

log_reg_pipeline = log_reg_grid.best_estimator_
labels_pred = log_reg_pipeline.predict(X_test)
train_accuracy = accuracy_score(y_train, log_reg_pipeline.predict(X_train))
test_accuracy = accuracy_score(y_test, labels_pred)

st.sidebar.write("Training Accuracy:", round(train_accuracy, 4))
st.sidebar.write("Testing Accuracy:", round(test_accuracy, 4))
st.sidebar.write("Confusion Matrix:\n", confusion_matrix(y_test, labels_pred))
st.sidebar.write("Classification Report:\n", classification_report(y_test, labels_pred))

# ========================================
# KNN CLASSIFIER WITH GRIDSEARCH
# ========================================
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>KNN Classifier (GridSearchCV)</span>", unsafe_allow_html=True)
st.sidebar.write("")

knn_pipeline = Pipeline([
    ("feature_scaling", StandardScaler()),
    ("knn_classifier", KNeighborsClassifier())
])

knn_params = {
    'knn_classifier__n_neighbors': [3, 5, 7, 9],
    'knn_classifier__weights': ['uniform', 'distance'],
    'knn_classifier__metric': ['euclidean', 'manhattan']
}

knn_grid = GridSearchCV(knn_pipeline, knn_params, cv=5, n_jobs=-1)
knn_grid.fit(X_train, y_train)

st.sidebar.write(f"Best Params: {knn_grid.best_params_}")
st.sidebar.write(f"Best CV Score: {knn_grid.best_score_:.4f}")

knn_pipeline = knn_grid.best_estimator_
knn_pred = knn_pipeline.predict(X_test)
knn_train_acc = accuracy_score(y_train, knn_pipeline.predict(X_train))
knn_test_acc = accuracy_score(y_test, knn_pred)

st.sidebar.write("Training Accuracy:", round(knn_train_acc, 4))
st.sidebar.write("Testing Accuracy:", round(knn_test_acc, 4))
st.sidebar.write("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
st.sidebar.write("Classification Report:\n", classification_report(y_test, knn_pred))

# ========================================
# DECISION TREE WITH GRIDSEARCH
# ========================================
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Decision Tree Classifier (GridSearchCV)</span>", unsafe_allow_html=True)
st.sidebar.write("\n")

dt_model = DecisionTreeClassifier(random_state=42)

dt_params = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_leaf': [2, 5, 10],
    'min_samples_split': [5, 10, 15]
}

dt_grid = GridSearchCV(dt_model, dt_params, cv=5, n_jobs=-1)
dt_grid.fit(X_train, y_train)

st.sidebar.write(f"Best Params: {dt_grid.best_params_}")
st.sidebar.write(f"Best CV Score: {dt_grid.best_score_:.4f}")

dt_model = dt_grid.best_estimator_
y_pred_dt = dt_model.predict(X_test)
dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, y_pred_dt)

st.sidebar.write("Training Accuracy:", round(dt_train_acc, 4))
st.sidebar.write("Testing Accuracy:", round(dt_test_acc, 4))
st.sidebar.write("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
st.sidebar.write("Classification Report:\n", classification_report(y_test, y_pred_dt))

# ========================================
# SVM CLASSIFIER WITH GRIDSEARCH
# ========================================
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>SVM Classifier (GridSearchCV)</span>", unsafe_allow_html=True)
st.sidebar.write("\n")

svm_pipeline = Pipeline([
    ("feature_scaling", StandardScaler()),
    ("svm_classifier", SVC(probability=True, random_state=42))
])

svm_params = {
    'svm_classifier__C': [0.1, 1, 10],
    'svm_classifier__kernel': ['linear', 'rbf', 'poly'],
    'svm_classifier__gamma': ['scale', 'auto']
}

svm_grid = GridSearchCV(svm_pipeline, svm_params, cv=5, n_jobs=-1)
svm_grid.fit(X_train, y_train)

st.sidebar.write(f"Best Params: {svm_grid.best_params_}")
st.sidebar.write(f"Best CV Score: {svm_grid.best_score_:.4f}")

svm_pipeline = svm_grid.best_estimator_
svm_pred = svm_pipeline.predict(X_test)
svm_train_acc = accuracy_score(y_train, svm_pipeline.predict(X_train))
svm_test_acc = accuracy_score(y_test, svm_pred)

st.sidebar.write("Training Accuracy:", round(svm_train_acc, 4))
st.sidebar.write("Testing Accuracy:", round(svm_test_acc, 4))
st.sidebar.write("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
st.sidebar.write("Classification Report:\n", classification_report(y_test, svm_pred))

# ========================================
# MODEL COMPARISON
# ========================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=100000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "SVM": SVC(kernel='rbf', random_state=42)
}

accuracies = {}
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Model Accuracy Summary</span>", unsafe_allow_html=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    st.sidebar.write(f"{name}: {round(acc, 4)}")

total_acc = sum(accuracies.values()) / len(accuracies)
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Average Accuracy Across All Models</span>", unsafe_allow_html=True)
st.sidebar.write(round(total_acc, 4))

sorted_acc = dict(sorted(accuracies.items(), key=lambda x: x[1], reverse=True))
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Model Performance Summary</span>", unsafe_allow_html=True)
for name, acc in sorted_acc.items():
    st.sidebar.write(f"{name:<30} {round(acc, 4)}")

best_model = max(sorted_acc, key=sorted_acc.get)
st.sidebar.write("\n")
st.sidebar.markdown("<span style='color:#FFDB58; background-color:black; font-weight:bold;'>Best Model</span>", unsafe_allow_html=True)
st.sidebar.write(f"{best_model} (Accuracy: {round(sorted_acc[best_model], 4)})")

# ========================================
# VISUALIZATIONS
# ========================================
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)
col7, col8 = st.columns(2)
col9, col10 = st.columns(2)

# 1. Category Distribution Plot
with col1:
    st.subheader("Category Distribution Plot")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="AQI_Category", palette="viridis", ax=ax1)
    plt.title("AQI Category Count")
    plt.xlabel("AQI Category")
    plt.ylabel("Count")
    
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{int(height)}', (p.get_x() + p.get_width()/2., height + 0.5), ha='center', va='bottom')
    
    st.pyplot(fig1)

# 2. Decision Tree Feature Importance
with col2:
    st.subheader("Decision Tree – Feature Importance")
    importances_dt = dt_model.feature_importances_
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.barplot(x=importances_dt, y=X.columns, palette="magma")
    plt.title("Decision Tree Feature Importance")
    ax2.set_ylabel("Features")
    for i, v in enumerate(importances_dt):
        ax2.text(v + 0.005, i, f"{v:.2f}", color='black', va='center')
    st.pyplot(fig2)

# 3. Model Accuracy Comparison
with col3:
    st.subheader("Model Accuracy Comparison")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
    plt.xticks(rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    
    for i, v in enumerate(accuracies.values()):
        ax3.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    st.pyplot(fig3)

# 4. Logistic Regression Accuracy
with col4:
    st.subheader("Logistic Regression Accuracy")
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    models_lr = ['Train', 'Test']
    acc_lr = [train_accuracy, test_accuracy]
    sns.barplot(x=models_lr, y=acc_lr, palette="coolwarm", ax=ax4)
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression Performance")
    for i, v in enumerate(acc_lr):
        ax4.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    st.pyplot(fig4)

# 5. KNN Accuracy
with col5:
    st.subheader("KNN Accuracy")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    models_knn = ['Train', 'Test']
    acc_knn = [knn_train_acc, knn_test_acc]
    sns.barplot(x=models_knn, y=acc_knn, palette="coolwarm", ax=ax5)
    plt.ylabel("Accuracy")
    plt.title("KNN Performance")
    for i, v in enumerate(acc_knn):
        ax5.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    st.pyplot(fig5)

# 6. K-Value vs. Error Rate Graph
with col6:
    st.subheader("KNN – K-Value vs. Error Rate")
    
    k_values = range(1, 31)
    train_errors = []
    test_errors = []
    
    scaler_knn = StandardScaler()
    X_train_scaled_knn = scaler_knn.fit_transform(X_train)
    X_test_scaled_knn = scaler_knn.transform(X_test)
    
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled_knn, y_train)
        
        train_pred = knn_temp.predict(X_train_scaled_knn)
        test_pred = knn_temp.predict(X_test_scaled_knn)
        
        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)
        
        train_errors.append(train_error)
        test_errors.append(test_error)
    
    fig6, ax6 = plt.subplots(figsize=(8, 5))
    ax6.plot(k_values, train_errors, marker='o', label='Train Error', linewidth=2, color='blue')
    ax6.plot(k_values, test_errors, marker='s', label='Test Error', linewidth=2, color='red')
    ax6.set_xlabel('K Value')
    ax6.set_ylabel('Error Rate')
    ax6.set_title('KNN – K-Value vs. Error Rate')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    st.pyplot(fig6)

# 7. SVM Accuracy
with col7:
    st.subheader("SVM Accuracy")
    fig7, ax7 = plt.subplots(figsize=(6, 4))
    models_svm = ['Train', 'Test']
    acc_svm = [svm_train_acc, svm_test_acc]
    sns.barplot(x=models_svm, y=acc_svm, palette="coolwarm", ax=ax7)
    plt.ylabel("Accuracy")
    plt.title("SVM Performance")
    for i, v in enumerate(acc_svm):
        ax7.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    st.pyplot(fig7)

# 8. SVM Decision Boundary Plot (pollutant_min vs pollutant_max)
with col8:
    st.subheader("SVM – Decision Boundary (pollutant_min vs pollutant_max)")
    
    X_viz = X[['pollutant_min', 'pollutant_max']]
    y_viz = y
    
    le_viz = LabelEncoder()
    y_viz_num = le_viz.fit_transform(y_viz)
    
    scaler_viz = StandardScaler()
    X_viz_scaled = scaler_viz.fit_transform(X_viz)
    svm_viz = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
    svm_viz.fit(X_viz_scaled, y_viz_num)
    
    x_min, x_max = X_viz_scaled[:, 0].min() - 1, X_viz_scaled[:, 0].max() + 1
    y_min, y_max = X_viz_scaled[:, 1].min() - 1, X_viz_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_viz.predict(grid_points)
    Z = Z.reshape(xx.shape)

    fig8, ax8 = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.viridis
    ax8.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    scatter8 = ax8.scatter(X_viz_scaled[:, 0], X_viz_scaled[:, 1], c=y_viz_num, cmap=cmap, s=40, edgecolor='k')
    ax8.set_xlabel('pollutant_min (scaled)')
    ax8.set_ylabel('pollutant_max (scaled)')
    ax8.set_title('SVM Decision Boundary')
    
    legend_labels = [plt.Line2D([0],[0], marker='o', color='w', label=cat, 
                    markerfacecolor=cmap(float(i)/max(1, len(le_viz.classes_)-1)), markersize=10) 
                    for i, cat in enumerate(le_viz.classes_)]
    ax8.legend(handles=legend_labels, title='AQI Category', loc='best', fontsize=8)
    
    st.pyplot(fig8)

