# local-sagemaker-training-xgboost
Sample repo that will help you train a local model without using AWS infrastructure or paying for this.

You gotta authenticate in AWS's DockerHub:

$ aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 683313688378.dkr.ecr.us-east-1.amazonaws.com

$ aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 257758044811.dkr.ecr.us-east-2.amazonaws.com

$ aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin 650961196544.dkr.ecr.us-east-1.amazonaws.com


Then just run:

python -m sagemaker_launcher