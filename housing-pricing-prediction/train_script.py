import pandas as pd
import xgboost as xgb
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='houses_pricing.csv')
    parser.add_argument('--model-dir', type=str, default='model-output')
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--subsample', type=float, default=0.7)
    args = parser.parse_args()

    # Carregar dados
    df = pd.read_csv(args.train)

    # Supondo que a coluna alvo é 'TIPO'
    target_col = 'TIPO'
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada no arquivo.")

    # Codificar o alvo para inteiros consecutivos
    y = df[target_col].astype('category').cat.codes

    # Remover colunas não numéricas e alvo dos preditores
    X = df.drop([target_col, 'DATE', 'PRICE', 'ADDRESS'], axis=1, errors='ignore')
    # Transformar colunas categóricas em dummies
    X = pd.get_dummies(X)

    print("Primeiras linhas do dataframe de entrada:")
    print(df.head())
    print("Valores únicos do alvo codificado:", y.unique())

    # Salvar nomes das colunas de features
    feature_cols = list(X.columns)
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)

    # Treinar modelo
    model = xgb.XGBClassifier(
        objective=args.objective,
        n_estimators=args.num_round,
        max_depth=args.max_depth,
        eta=args.eta,
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        subsample=args.subsample,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X, y)

    # Salvar modelo
    model.save_model(os.path.join(args.model_dir, 'xgboost-model.json'))

if __name__ == '__main__':
    main()