import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent


def load_json(file_path: Path):
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(file_path: Path, data):
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def apply_configured_column_mappings(df: pd.DataFrame, config: dict):
    map_columns = config.get("map-columns", {})

    for column_name, mapping_name in map_columns.items():
        if column_name not in df.columns:
            continue

        mapping_path = BASE_DIR / f"{mapping_name}.json"
        if not mapping_path.exists():
            continue

        ranges = load_json(mapping_path)
        ordered_ranges = list(ranges.items())

        def map_value(value):
            numeric_value = pd.to_numeric(value, errors="coerce")
            if pd.isna(numeric_value):
                return value
            for label, upper_bound in ordered_ranges:
                if numeric_value <= upper_bound:
                    return label
            return ordered_ranges[-1][0]

        df[column_name] = df[column_name].map(map_value)

    return df


def has_missing_markers(value) -> bool:
    value_str = str(value).strip()
    return pd.isna(value) or value_str in {"?", "-1"}


def remove_missing_values(df: pd.DataFrame):
    mask = df.apply(lambda row: any(has_missing_markers(val) for val in row), axis=1)
    df = df[~mask]
    return df.dropna()


def transform_string_columns(df: pd.DataFrame):
    mappings = {}
    for coluna in df.columns:
        if is_string_dtype(df[coluna]):
            classes = df[coluna].unique()
            mapping = {str(label): int(idx) for idx, label in enumerate(classes)}
            df[coluna] = df[coluna].map(mapping)
            mappings[coluna] = mapping
    return df, mappings


def split_dataset(df: pd.DataFrame, train_size: float = 0.8):
    train_count = int(len(df) * train_size)
    df_train = df.iloc[:train_count].reset_index(drop=True)
    df_test = df.iloc[train_count:].reset_index(drop=True)
    return df_train, df_test


def get_column_names():
    # Colunas baseadas no arquivo parkinsons.names (UCI ML Repository, id=174)
    cols = [
        "name",                    # nome do sujeito (string, será removida)
        "MDVP:Fo(Hz)",             # frequência vocal média fundamental
        "MDVP:Fhi(Hz)",            # frequência vocal máxima
        "MDVP:Flo(Hz)",            # frequência vocal mínima
        "MDVP:Jitter(%)",          # variação de frequência (%)
        "MDVP:Jitter(Abs)",        # variação de frequência (absoluta)
        "MDVP:RAP",                # perturbação relativa média
        "MDVP:PPQ",                # quociente de perturbação de pitch (5 pontos)
        "Jitter:DDP",              # média de diferenças absolutas entre ciclos consecutivos
        "MDVP:Shimmer",            # variação de amplitude
        "MDVP:Shimmer(dB)",        # variação de amplitude em dB
        "Shimmer:APQ3",            # quociente de perturbação de amplitude (3 pontos)
        "Shimmer:APQ5",            # quociente de perturbação de amplitude (5 pontos)
        "MDVP:APQ",                # quociente de perturbação de amplitude (MDVP)
        "Shimmer:DDA",             # média de diferenças absolutas entre amplitudes
        "NHR",                     # razão ruído/harmônicos
        "HNR",                     # razão harmônicos/ruído
        "RPDE",                    # entropia de densidade de recorrência
        "DFA",                     # expoente de flutuação sem tendência
        "spread1",                 # medida não linear de variação de frequência 1
        "spread2",                 # medida não linear de variação de frequência 2
        "D2",                      # correlação dimensional
        "PPE",                     # entropia de período de pitch
        "status",                  # rótulo: 1 = Parkinson, 0 = saudável
    ]
    return cols


def process_dataset():
    config_path = BASE_DIR / "config.json"
    config = load_json(config_path)

    data_path = BASE_DIR / "parkinsons.data"
    if not data_path.exists():
        print(f"Erro: Arquivo {data_path} não encontrado.")
        print("Baixe o arquivo em: https://archive.ics.uci.edu/dataset/174/parkinsons")
        print("Ou use fetch_ucirepo(id=174) e salve como parkinsons.data (CSV sem cabeçalho).")
        return

    # Carrega o dataset — o arquivo original JÁ tem cabeçalho, então header=0
    df = pd.read_csv(data_path, header=0)
    print(f"Dataset carregado com {len(df)} linhas e {len(df.columns)} colunas.")

    # Remove a coluna 'name' (identificador do sujeito, não é feature preditiva)
    if "name" in df.columns:
        df = df.drop(columns=["name"])
        print("Coluna 'name' removida (identificador não preditivo).")

    # Renomeia 'status' para 'has_parkinsons' para maior clareza
    if "status" in df.columns:
        df = df.rename(columns={"status": "has_parkinsons"})

    linhas_antes = len(df)
    df = remove_missing_values(df)
    linhas_removidas = linhas_antes - len(df)
    print(f"Linhas removidas (valores ausentes): {linhas_removidas}")
    print(f"Linhas restantes: {len(df)}")

    df = apply_configured_column_mappings(df, config)

    df_numeric, mappings = transform_string_columns(df)

    df_train, df_test = split_dataset(df_numeric, train_size=0.8)

    output_train_path = BASE_DIR / "parkinsons_treinamento.csv"
    df_train.to_csv(output_train_path, index=False)

    output_test_path = BASE_DIR / "parkinsons_teste.csv"
    df_test.to_csv(output_test_path, index=False)

    groupings_path = BASE_DIR / "mappings.json"
    save_json(groupings_path, mappings)

    print(f"\nArquivo de treinamento criado em: {output_train_path}")
    print(f"Arquivo de teste criado em:        {output_test_path}")
    print(f"Agrupamentos salvos em:            {groupings_path}")
    print(f"\nDistribuição do alvo no treino:")
    print(df_train["has_parkinsons"].value_counts().to_string())
    print(f"\nDistribuição do alvo no teste:")
    print(df_test["has_parkinsons"].value_counts().to_string())


def main():
    process_dataset()


if __name__ == "__main__":
    main()