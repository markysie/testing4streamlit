
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier  # 수정: RandomForestClassifier 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# 데이터 불러오기
df = pd.read_csv("merged_data.csv")

# 데이터 정규화
scaler = MinMaxScaler()
df[["Y_COORD", "X_COORD"]] = scaler.fit_transform(df[["Y_COORD", "X_COORD"]])

# Train과 Test 데이터로 나누기
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 전처리 및 모델 학습
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), ["Y_COORD", "X_COORD", "AGE_GRP"])
    ])

rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # 예시로 100개의 트리 사용
])

rf.fit(train_df[["Y_COORD", "X_COORD", "AGE_GRP"]], train_df["VISIT_AREA_NM"])

# Test 데이터로 예측
y_pred_rf = rf.predict(test_df[["Y_COORD", "X_COORD", "AGE_GRP"]])

# 정확도 출력
accuracy_rf = accuracy_score(test_df["VISIT_AREA_NM"], y_pred_rf)
print(f"랜덤 포레스트 모델 정확도: {accuracy_rf:.2%}")

# 사용자 위치와 나이에 가장 가까운 관광지 추천 (랜덤 포레스트 모델 사용)
user_location = np.array([33.450722, 126.512222])
user_age = 20

# 사용자 입력으로 추천 받기
user_input = pd.DataFrame(np.array([user_location[0], user_location[1], user_age]).reshape(1, -1),
                           columns=["Y_COORD", "X_COORD", "AGE_GRP"])
user_input_transformed = pd.DataFrame(preprocessor.transform(user_input), columns=["Y_COORD", "X_COORD", "AGE_GRP"])
user_recommendation_rf = rf.predict(user_input_transformed)
print("랜덤 포레스트 모델을 사용한 사용자 추천 결과:", user_recommendation_rf)

# 하이퍼파라미터 그리드 설정
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# 그리드 서치 수행
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=5)
grid_search_rf.fit(train_df[["Y_COORD", "X_COORD", "AGE_GRP"]], train_df["VISIT_AREA_NM"])

# 최적의 하이퍼파라미터 출력
print("랜덤 포레스트 모델의 최적 하이퍼파라미터:", grid_search_rf.best_params_)

# 최적의 모델로 Test 데이터로 예측
y_pred_tuned_rf = grid_search_rf.predict(test_df[["Y_COORD", "X_COORD", "AGE_GRP"]])

# 튜닝된 모델의 정확도 출력
accuracy_tuned_rf = accuracy_score(test_df["VISIT_AREA_NM"], y_pred_tuned_rf)
print(f"튜닝된 랜덤 포레스트 모델 정확도: {accuracy_tuned_rf:.2%}")

# 사용자 위치와 나이에 가장 가까운 관광지 추천 (튜닝된 랜덤 포레스트 모델 사용)
user_recommendation_tuned_rf = grid_search_rf.best_estimator_.predict(user_input_transformed)
print("튜닝된 랜덤 포레스트 모델을 사용한 사용자 추천 결과:", user_recommendation_tuned_rf)
