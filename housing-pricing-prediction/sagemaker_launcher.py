import sagemaker
from sagemaker.local import LocalSession
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput

# 1. Sessão local do SageMaker
sagemaker_local_session = LocalSession()
sagemaker_local_session.config = {'local': {'local_code': True}}

# 2. Caminho local para os dados de treino
train_data_path = 'file://./houses_pricing.csv'  # ajuste conforme seu arquivo

# 3. Role fake (não usado no local)
role = "arn:aws:iam::000000000000:role/unused"

# 4. Imagem XGBoost oficial
image_uri = sagemaker.image_uris.retrieve("xgboost", "us-east-1", "1.5-1")

# 5. Estimator para local mode
xgb_train = Estimator(
    image_uri=image_uri,
    instance_type="local_gpu",
    instance_count=1,
    output_path="file://./model-output",  # saída local
    role=role,
    sagemaker_session=sagemaker_local_session,
    entry_point="train_script.py",  # Seu script de treino local
    source_dir="."           # Diretório onde está o train.py
)

# 6. Hiperparâmetros
xgb_train.set_hyperparameters(
    objective="binary:logistic",
    num_round=100,
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    subsample=0.7,
)

# 7. Treinamento local
xgb_train.fit(
    {"train": TrainingInput(s3_data=train_data_path, content_type="text/csv")}
)
