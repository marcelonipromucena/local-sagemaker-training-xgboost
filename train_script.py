import pandas as pd
import xgboost as xgb
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='Employee.csv')
    parser.add_argument('--model-dir', type=str, default='model-output')
    parser.add_argument('--eta', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=4)
    parser.add_argument('--max_depth', type=int, default=5)
    parser.add_argument('--min_child_weight', type=int, default=6)
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--subsample', type=float, default=0.7)
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.train)
    X = df.iloc[:, :-1]  # Features: all columns except last
    y = df.iloc[:, -1]   # Target: last column

    X = pd.get_dummies(X)


    # Train model
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

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_model(os.path.join(args.model_dir, 'xgboost-model.json'))

if __name__ == '__main__':
    main()