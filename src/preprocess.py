import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

CAT_FEATURES = [
    "season",
    "holiday",
    "workingday",
    "weather"
]


# 전처리 파이프라인 작성
# 1. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용 (from category_encoders import TargetEncoder)
preprocess_pipeline = ColumnTransformer(
    transformers=[
        (  # 1. 범주형 변수(CAT_FEATURES)는 타겟 인코딩 적용
            "target_encoder",
            TargetEncoder(cols=CAT_FEATURES),
            CAT_FEATURES,
        )
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(
    transform="pandas"
)  
