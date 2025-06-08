import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib # Для сохранения/загрузки предобработчика (может быть уже не нужен, но оставим на всякий случай)

st.set_page_config(page_title="Прогнозирование качества сна", layout="wide")

@st.cache_data
def load_data():
    """Загружает и минимально подготавливает данные."""
    file_path = 'datasets/Sleep_health_and_lifestyle_dataset_processed.csv'
    try:
        df = pd.read_csv(file_path)
        if df.empty: # Дополнительная проверка на пустой DataFrame после успешного чтения
            st.error(f"Файл {file_path} успешно прочитан, но он пуст. Пожалуйста, проверьте содержимое файла.")
            st.stop()
            return None
        return df # Успешный случай: df загружен и возвращен
    except FileNotFoundError:
        st.error(f"Файл {file_path} не найден. Убедитесь, что он находится в той же директории, что и web.py.")
        st.stop()
        return None # Возвращаем None, если файл не найден
    except pd.errors.EmptyDataError: # Ошибка, если CSV файл пуст и не может быть распарсен
        st.error(f"Файл {file_path} пуст или имеет неверный формат. Pandas не смог его прочитать.")
        st.stop()
        return None
    except Exception as e: # Обработка других возможных ошибок при чтении файла
        st.error(f"Произошла непредвиденная ошибка при чтении файла {file_path}: {e}")
        st.stop()
        return None
    
    # Эта часть кода не должна достигаться, если все выше обработано корректно
    # и st.stop() работает как ожидается, останавливая скрипт.
    # Если же df не None, то он будет возвращен.
    # Если df остался None по какой-то причине (что маловероятно при такой структуре),
    # и st.stop() не сработал, то вернется None.

def train_model(X_train_data, y_train_data, n_estimators_val, max_depth_val, min_samples_split_val):
    """Обучает модель RandomForestRegressor с заданными гиперпараметрами."""
    model = RandomForestRegressor(
        n_estimators=n_estimators_val,
        max_depth=max_depth_val,
        min_samples_split=min_samples_split_val,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_data, y_train_data)
    return model

df_source = load_data()

# Проверяем, были ли данные успешно загружены
if df_source is None:
    # Сообщение об ошибке и st.stop() уже были вызваны в load_data()
    # Эта дополнительная остановка здесь - мера предосторожности.
    st.info("Загрузка данных не удалась. Приложение остановлено. Проверьте сообщения об ошибках выше.")
    st.stop() # Гарантируем остановку, если она не произошла в load_data


X = df_source.drop(['quality_of_sleep', 'person_id'], axis=1, errors='ignore') 
y = df_source['quality_of_sleep']

# Проверка, что X и y не пустые после изменений
if X.empty:
    st.error("Признаки (X) пусты после удаления 'quality_of_sleep' и 'person_id'. Проверьте ваш CSV файл и названия колонок.")
    st.stop()
if y.empty:
    st.error("Целевая переменная (y) 'quality_of_sleep' пуста или отсутствует в CSV файле.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.title("Демонстрация модели: Прогноз качества сна")
st.markdown("""
Это приложение демонстрирует работу модели `RandomForestRegressor` для предсказания качества сна.
Данные считаются уже полностью предобработанными (кодирование и масштабирование выполнены).
Вы можете изменять некоторые гиперпараметры модели и видеть, как это влияет на ее качество на тестовой выборке.
""")

st.sidebar.header("Гиперпараметры RandomForestRegressor")

n_estimators_user = st.sidebar.slider(
    "Количество деревьев (n_estimators):",
    min_value=10,
    max_value=300,
    value=100,
    step=10
)

max_depth_user = st.sidebar.select_slider(
    "Максимальная глубина дерева (max_depth):",
    options=[5, 10, 15, 20, 25, 30, None],
    value=20
)

min_samples_split_user = st.sidebar.slider(
    "Минимальное количество образцов для разделения узла (min_samples_split):",
    min_value=2,
    max_value=20,
    value=2,
    step=1
)


if st.sidebar.button("Обучить модель и оценить"):
    with st.spinner("Обучение модели... Пожалуйста, подождите."):
        trained_model = train_model(X_train, y_train, 
                                     n_estimators_user, max_depth_user, min_samples_split_user)
    
    st.success("Модель успешно обучена!")
    
    y_pred = trained_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    st.subheader("Метрики качества модели на тестовой выборке:")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² (Коэфф. детерминации)", f"{r2:.4f}")
    col2.metric("MAE (Сред. абс. ошибка)", f"{mae:.2f}")
    col3.metric("RMSE (Корень из ср.кв. ошибки)", f"{rmse:.2f}")

    st.subheader("Сравнение предсказанных и реальных значений (первые 50 точек)")
    
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).reset_index(drop=True)
    
    sample_size = min(50, len(results_df))
    
    fig_col, _ = st.columns([3,1])
    
    with fig_col:
        st.line_chart(results_df.head(sample_size))

else:
    st.info("Настройте гиперпараметры в боковой панели и нажмите 'Обучить модель и оценить'.")