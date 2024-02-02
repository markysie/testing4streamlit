import streamlit as st
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

st.title("제주도 관광지추천 시스템")
st.write("""테스트 중입니다.""")


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

# 모델 설정 변경
rf = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)

# Pipeline을 사용하지 않고 직접 fit 및 predict
preprocessed_train_data = preprocessor.fit_transform(train_df[["Y_COORD", "X_COORD", "AGE_GRP"]])
rf.fit(preprocessed_train_data, train_df["VISIT_AREA_NM"])

# Streamlit 애플리케이션
st.title("스트림릿과 머신러닝 모델 연동")

# 나이 선택을 위한 슬라이더
selected_age = st.slider("나이를 선택하세요", min_value=10, max_value=60, step=10)

# 사용자 입력으로 추천 받기
user_location = np.array([33.450722, 126.512222])
user_age = selected_age

user_input = pd.DataFrame(np.array([user_location[0], user_location[1], user_age]).reshape(1, -1),
                           columns=["Y_COORD", "X_COORD", "AGE_GRP"])
user_input_transformed = preprocessor.transform(user_input)

# 랜덤 포레스트 모델을 사용한 추천 결과
user_probabilities_rf = rf.predict_proba(user_input_transformed)
class_labels = rf.classes_
top_n_recommendations = [class_labels[idx] for idx in np.argsort(user_probabilities_rf[0])[::-1][:5]]

# 버튼 클릭을 통해 나이 설정하기
clicked_button = st.button("나이 설정하기")

# 버튼 클릭 시 나이를 60으로 설정
if clicked_button:
    selected_age = 60

# 결과 출력
st.write(f"선택한 나이: {selected_age}")
st.write("랜덤 포레스트 모델을 사용한 상위 5개 추천 결과:", top_n_recommendations)

# 훈련 데이터 정확도 출력
y_train_pred = rf.predict(preprocessed_train_data)
train_accuracy = accuracy_score(train_df["VISIT_AREA_NM"], y_train_pred)
st.write(f"랜덤 포레스트 모델 훈련 정확도: {train_accuracy:.2%}")

# 테스트 데이터 정확도 출력
preprocessed_test_data = preprocessor.transform(test_df[["Y_COORD", "X_COORD", "AGE_GRP"]])
y_test_pred = rf.predict(preprocessed_test_data)
test_accuracy = accuracy_score(test_df["VISIT_AREA_NM"], y_test_pred)
st.write(f"랜덤 포레스트 모델 테스트 정확도: {test_accuracy:.2%}")
