# predict_local.py
import argparse
import json
import os
import pandas as pd
import xgboost as xgb
import numpy as np

def load_model(model_dir):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.load_model(os.path.join(model_dir, "xgboost-model.json"))
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="inference_data.csv", help="CSV com as amostras para inferência")
    parser.add_argument("--model-dir", type=str, default="model-output", help="Pasta onde está o modelo salvo")
    parser.add_argument("--target-col", type=str, default="LeaveOrNot", help="Nome da coluna alvo (se existir no CSV)")
    parser.add_argument("--out", type=str, default="predictions.csv", help="Arquivo de saída com previsões")
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar para converter probabilidade em classe")
    args = parser.parse_args()

    # 1) Carrega dados
    df = pd.read_csv(args.data)

    # Se o CSV de inferência tiver o alvo, vamos ignorá-lo
    if args.target_col in df.columns:
        X_raw = df.drop(columns=[args.target_col])
    else:
        X_raw = df.copy()

    # 2) One-hot igual ao treino
    X_ohe = pd.get_dummies(X_raw)

    # 3) Garante as mesmas colunas do treino
    with open(os.path.join(args.model_dir, "feature_columns.json"), "r") as f:
        feature_cols = json.load(f)
    # reindex para alinhar as colunas (faltantes entram como 0, extras são descartadas)
    X = X_ohe.reindex(columns=feature_cols, fill_value=0)

    # 4) Carrega o modelo
    model = load_model(args.model_dir)

    # 5) Predições
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    # 6) Salva resultado
    out_df = df.copy()
    out_df["pred_proba"] = proba
    out_df["pred_class"] = pred
    out_df.to_csv(args.out, index=False)

    print(f"✅ Inferência concluída! Arquivo salvo em: {args.out}")
    print(f"Exemplo de 5 linhas:\n{out_df[['pred_proba','pred_class']].head()}")

if __name__ == "__main__":
    main()
