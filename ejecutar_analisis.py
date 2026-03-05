#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script optimizado para análisis LDA vs Árboles de Decisión
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
import warnings
import sys
import os

# Configurar encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10
sns.set_style('whitegrid')

print("="*70)
print("ANALISIS COMPARATIVO: LDA vs ARBOLES DE DECISION")
print("Versión Optimizada para Memoria")
print("="*70)

# 1. CARGAR DATOS
print("\n[1/7] Cargando datos...")
try:
    chunk_size = 50000
    chunks = []
    for chunk in pd.read_csv(
        'conjunto_de_datos_emat2024_csv/conjunto_de_datos/conjunto_de_datos_emat2024.csv',
        chunksize=chunk_size
    ):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print("OK - Datos cargados: {}".format(df.shape))
except Exception as e:
    print("ERROR al cargar: {}".format(e))
    exit(1)

# 2. PREPARACION DE DATOS
print("\n[2/7] Preparando datos...")

# Tomar muestra si es necesario
max_samples = 100000
if len(df) > max_samples:
    print("  Tomando muestra de {} registros (de {})".format(max_samples, len(df)))
    df = df.sample(n=max_samples, random_state=42)

target_col = 'escol_con1'
top_classes = df[target_col].value_counts().head(3).index.tolist()
df_filtered = df[df[target_col].isin(top_classes)].copy()
df = None

print("  Variable objetivo: {}".format(target_col))
print("  Distribucion:")
for cls in df_filtered[target_col].value_counts().index:
    count = (df_filtered[target_col] == cls).sum()
    pct = 100 * count / len(df_filtered)
    print("    {}: {:6d} ({:5.1f}%)".format(cls, count, pct))

# Variables
predictors = ['edad_con1', 'edad_con2', 'anio_regis']
categorical_features = ['sexo_con1', 'naci_con1']
all_features = predictors + categorical_features

# Preparar
df_model = df_filtered[all_features + [target_col]].copy()
df_filtered = None

# Codificar categoricas
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_model[col] = df_model[col].fillna('Unknown')
    try:
        df_model[col] = le.fit_transform(df_model[col].astype(str))
    except:
        df_model[col] = 0
    label_encoders[col] = le

X = df_model[all_features].copy()
X = X.fillna(X.mean())

le_target = LabelEncoder()
y = le_target.fit_transform(df_model[target_col].astype(str))

df_model = None

print("  Variables predictoras: {}".format(len(all_features)))
print("  Observaciones finales: {}".format(len(X)))

# 3. PARTICION
print("\n[3/7] Particionando datos...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("  Entrenamiento: {}".format(len(y_train)))
print("  Prueba: {}".format(len(y_test)))

# 4. ENTRENAR LDA
print("\n[4/7] Entrenando LDA...")
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train, y_train)
print("OK - LDA entrenado")
print("  Componentes: {}".format(lda.n_components))
print("  Clases: {}".format(lda.classes_))

X_train_lda = lda.transform(X_train)
X_test_lda = lda.transform(X_test)

# Visualizar LDA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

ax = axes[0]
for i, class_idx in enumerate(lda.classes_):
    mask = y_train == class_idx
    ax.hist(X_train_lda[mask, 0], bins=20, alpha=0.5, label='Clase {}'.format(i), color=colors[i])
ax.set_xlabel('LD1', fontweight='bold')
ax.set_ylabel('Frecuencia', fontweight='bold')
ax.set_title('Distribucion LDA - LD1', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
for i, class_idx in enumerate(lda.classes_):
    mask = y_train == class_idx
    if lda.n_components >= 2:
        ax.scatter(X_train_lda[mask, 0], X_train_lda[mask, 1],
                  alpha=0.6, s=30, label='Clase {}'.format(i), color=colors[i])
    else:
        ax.scatter(range(len(X_train_lda[mask, 0])), X_train_lda[mask, 0],
                  alpha=0.6, s=30, label='Clase {}'.format(i), color=colors[i])
ax.set_xlabel('LD1', fontweight='bold')
ax.set_ylabel('LD2', fontweight='bold')
ax.set_title('Proyeccion LDA 2D', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lda_visualization.png', dpi=200, bbox_inches='tight')
print("  Guardado: lda_visualization.png")
plt.close()

# 5. ENTRENAR ARBOL
print("\n[5/7] Entrenando Arbol de Decision...")

dt_full = DecisionTreeClassifier(
    random_state=42,
    criterion='gini',
    min_samples_split=20,
    min_samples_leaf=10,
    max_depth=15
)
dt_full.fit(X_train, y_train)
print("OK - Arbol base entrenado")
print("  Profundidad: {}".format(dt_full.get_depth()))
print("  Hojas: {}".format(dt_full.get_n_leaves()))

# Poda
print("  Realizando poda...")
path = dt_full.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

test_scores = []
for ccp_alpha in ccp_alphas[::max(1, len(ccp_alphas)//20)]:
    tree = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=ccp_alpha,
        criterion='gini',
        min_samples_split=20,
        min_samples_leaf=10,
        max_depth=15
    )
    tree.fit(X_train, y_train)
    test_scores.append(accuracy_score(y_test, tree.predict(X_test)))

# Arbol optimo
optimal_alpha = ccp_alphas[len(ccp_alphas)//2]
dt_optimal = DecisionTreeClassifier(
    random_state=42,
    ccp_alpha=optimal_alpha,
    criterion='gini',
    min_samples_split=20,
    min_samples_leaf=10,
    max_depth=10
)
dt_optimal.fit(X_train, y_train)

print("OK - Arbol optimo")
print("  Profundidad: {}".format(dt_optimal.get_depth()))
print("  Hojas: {}".format(dt_optimal.get_n_leaves()))

# Visualizar
fig, ax = plt.subplots(figsize=(14, 8))
plot_tree(dt_optimal,
          feature_names=all_features,
          class_names=['Clase 0', 'Clase 1', 'Clase 2'],
          filled=True,
          rounded=True,
          fontsize=8,
          ax=ax)
plt.title('Arbol de Decision Optimo', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('dt_tree.png', dpi=150, bbox_inches='tight')
print("  Guardado: dt_tree.png")
plt.close()

# 6. EVALUAR
print("\n[6/7] Evaluando modelos...")

y_pred_lda = lda.predict(X_test)
y_pred_dt = dt_optimal.predict(X_test)

lda_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_lda),
    'Precision': precision_score(y_test, y_pred_lda, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, y_pred_lda, average='weighted', zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_lda, average='weighted', zero_division=0)
}

dt_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_dt),
    'Precision': precision_score(y_test, y_pred_dt, average='weighted', zero_division=0),
    'Recall': recall_score(y_test, y_pred_dt, average='weighted', zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_dt, average='weighted', zero_division=0)
}

print("\n" + "="*70)
print("METRICAS EN CONJUNTO DE PRUEBA")
print("="*70)
print("\n{:<15} {:<12} {:<12} {:<10}".format('Metrica', 'LDA', 'Arbol', 'Diff'))
print("-"*70)
for metric in lda_metrics.keys():
    lda_val = lda_metrics[metric]
    dt_val = dt_metrics[metric]
    diff = dt_val - lda_val
    print("{:<15} {:<12.4f} {:<12.4f} {:+.4f}".format(metric, lda_val, dt_val, diff))

# Matrices confusión
conf_lda = confusion_matrix(y_test, y_pred_lda)
conf_dt = confusion_matrix(y_test, y_pred_dt)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_lda, annot=True, fmt='d', cmap='Blues',
            xticklabels=['C0', 'C1', 'C2'],
            yticklabels=['C0', 'C1', 'C2'],
            ax=axes[0], cbar=True)
axes[0].set_title('Matriz de Confusion - LDA', fontweight='bold')
axes[0].set_ylabel('Verdadera', fontweight='bold')
axes[0].set_xlabel('Predicha', fontweight='bold')

sns.heatmap(conf_dt, annot=True, fmt='d', cmap='Greens',
            xticklabels=['C0', 'C1', 'C2'],
            yticklabels=['C0', 'C1', 'C2'],
            ax=axes[1], cbar=True)
axes[1].set_title('Matriz de Confusion - Arbol', fontweight='bold')
axes[1].set_ylabel('Verdadera', fontweight='bold')
axes[1].set_xlabel('Predicha', fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=200, bbox_inches='tight')
print("\n  Guardado: confusion_matrices.png")
plt.close()

# Comparativo
fig, ax = plt.subplots(figsize=(10, 6))
metrics_names = list(lda_metrics.keys())
x = np.arange(len(metrics_names))
width = 0.35

lda_values = list(lda_metrics.values())
dt_values = list(dt_metrics.values())

ax.bar(x - width/2, lda_values, width, label='LDA', color='#3498db', alpha=0.8)
ax.bar(x + width/2, dt_values, width, label='Arbol', color='#2ecc71', alpha=0.8)

ax.set_ylabel('Puntuacion', fontweight='bold')
ax.set_title('Comparacion de Metricas', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=200, bbox_inches='tight')
print("  Guardado: metrics_comparison.png")
plt.close()

# 7. ANALISIS FINAL
print("\n[7/7] Analisis de generalizacion...")

acc_lda_train = accuracy_score(y_train, lda.predict(X_train))
acc_lda_test = lda_metrics['Accuracy']
acc_dt_train = accuracy_score(y_train, dt_optimal.predict(X_train))
acc_dt_test = dt_metrics['Accuracy']

print("\nDesempenio Entrenamiento vs Prueba:")
print("  LDA:")
print("    Train: {:.4f}".format(acc_lda_train))
print("    Test:  {:.4f}".format(acc_lda_test))
print("    Overfitting: {:+.4f}".format(acc_lda_train - acc_lda_test))
print("  Arbol:")
print("    Train: {:.4f}".format(acc_dt_train))
print("    Test:  {:.4f}".format(acc_dt_test))
print("    Overfitting: {:+.4f}".format(acc_dt_train - acc_dt_test))

# Grafico generalizacion
fig, ax = plt.subplots(figsize=(10, 6))
models = ['LDA', 'Arbol']
train_scores = [acc_lda_train, acc_dt_train]
test_scores = [acc_lda_test, acc_dt_test]

x = np.arange(len(models))
width = 0.35

ax.bar(x - width/2, train_scores, width, label='Entrenamiento', color='#3498db', alpha=0.8)
ax.bar(x + width/2, test_scores, width, label='Prueba', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Precision', fontweight='bold')
ax.set_title('Generalizacion: Training vs Test', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('generalization.png', dpi=200, bbox_inches='tight')
print("  Guardado: generalization.png")
plt.close()

# CONCLUSION
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

diff = dt_metrics['Accuracy'] - lda_metrics['Accuracy']
print("\nPrecision en Prueba:")
print("  LDA:  {:.4f}".format(lda_metrics['Accuracy']))
print("  Arbol: {:.4f}".format(dt_metrics['Accuracy']))
print("  Diferencia: {:+.4f}".format(diff))

if abs(diff) < 0.05:
    print("\nDESEMPENIO SIMILAR (diferencia < 0.05)")
    if (acc_lda_train - acc_lda_test) < (acc_dt_train - acc_dt_test):
        print("\n*** RECOMENDACION: LDA ***")
        print("   - Mejor generalizacion (menos overfitting)")
        print("   - Modelo mas simple y eficiente")
    else:
        print("\n*** RECOMENDACION: ARBOL DE DECISION ***")
        print("   - Mejor interpretabilidad")
        print("   - Sin supuestos distribucionales")
else:
    if diff > 0:
        print("\n*** RECOMENDACION: ARBOL DE DECISION ***")
        print("   Precision superior (+{:.4f})".format(diff))
    else:
        print("\n*** RECOMENDACION: LDA ***")
        print("   Precision superior (+{:.4f})".format(-diff))

print("\n" + "="*70)
print("Analisis completado exitosamente")
print("="*70)
