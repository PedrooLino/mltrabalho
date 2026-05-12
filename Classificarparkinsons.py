import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
import os

warnings.filterwarnings("ignore")

SEEDS = [4567, 1234, 9999] 


def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "parkinsons_treinamento.csv")
    test_path = os.path.join(current_dir, "parkinsons_teste.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        train_path = "parkinsons_treinamento.csv"
        test_path = "parkinsons_teste.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def shuffle_and_split(train_df, test_df, seed):
    full_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    shuffled_index = np.random.RandomState(seed).permutation(full_df.index)
    full_df = full_df.loc[shuffled_index].reset_index(drop=True)

    train_count = int(len(full_df) * 0.8)
    df_train = full_df.iloc[:train_count]
    df_test = full_df.iloc[train_count:]

    target_col = "has_parkinsons"

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


def evaluate_classifiers():
    try:
        train_df, test_df = load_data()
    except Exception as e:
        print(f"\nERRO: Arquivos .csv não encontrados na pasta do script.")
        print(f"Execute primeiro: python preprocessar_parkinsons.py")
        print(f"Detalhe: {e}")
        return

    classifiers = {
        "Perceptron":    Perceptron(),
        "Bayes":         GaussianNB(),
        "MLP":           MLPClassifier(max_iter=500),
        "SVM":           SVC(),
        "Decision Tree": DecisionTreeClassifier(),
    }

    results = {name: [] for name in classifiers.keys()}

    for i, seed in enumerate(SEEDS):
        X_train, y_train, X_test, y_test = shuffle_and_split(train_df, test_df, seed)
        print(f"\nExecucao {i+1} (seed={seed}):")
        print("-" * 40)
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name].append(acc)
            print(f"  {name:20} | Acuracia: {acc:.4f}")

    print("\n" + "=" * 50)
    print("RESULTADOS FINAIS (Media de Acuracia):")
    print("=" * 50)

    final_summary = []
    for name, accs in results.items():
        mean_acc = np.mean(accs)
        print(f"  {name:20} | Media: {mean_acc:.4f}")
        final_summary.append({"Classificador": name, "Media Acuracia": f"{mean_acc:.4f}"})

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_acuracia_parkinsons.csv")
    pd.DataFrame(final_summary).to_csv(output_path, index=False)
    print(f"\nResumo salvo em: {output_path}")


if __name__ == "__main__":
    evaluate_classifiers()