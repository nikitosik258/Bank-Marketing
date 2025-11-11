# Bank Marketing
Проект по предсказанию отклика на телемаркетинговую кампанию банка

## Постановка задачи
Цель: предсказать, согласится ли клиент на депозит (y ∈ {yes,no}).

## Признаки:
- **Соц-демо:** `age`, `job`, `marital`, `education`
- **Финансы:** `balance`, `default`, `housing`, `loan`
- **Контакт:** `contact`, `month`, `day`
- **Маркетинг:** `campaign`, `previous`, `pdays`, `poutcome`
- **Инженерные:**  `pdays_never = (pdays == -1)`, `campaign_bins` (биннинг `campaign`)
- **Важно:** duration исключён из обучения (утечка — известен только после звонка). day используется лишь для сортировки по времени.

## EDA (главные выводы)
- **Дисбаланс таргета:** класс `yes` ≈ **11%**.  
- **По категориям:** доминируют `cellular`, пик обращений — **may**, `poutcome='unknown'`; в `job` перекос в `blue-collar/management`.  
- Много `pdays = -1` → добавлен флаг **`pdays_never`**.  
- Временной порядок задаётся как `month*100 + day` (год отсутствует).

## Baseline
- **CV:** `TimeSeriesSplit(n_splits=5)` на train.  
- **Ключевая метрика:** **PR-AUC** (из-за дисбаланса).
- **Модели и результаты:**
<img width="710" height="205" alt="image" src="https://github.com/user-attachments/assets/eeb7db9d-b822-41ae-95ab-678ecb456b66" />
 
 - **Важности (деревья):** month, poutcome, contact, campaign_bins, previous, pdays_never, затем balance/age.

## Improvements
- **Кодирование:** `OrdinalEncoder` с зафиксированным порядком категорий.  
- **Масштабирование:** `StandardScaler` для `balance`, `campaign`, `previous`.  
- **SMOTE в Pipeline/TS-CV:** небольшое ухудшение метрик.  
- **Feature engineering:** `pdays_never`, `campaign_bins` — ↑ стабильности и интерпретируемости.  
- **Исключение `duration`:** снижает «сырые» метрики, но устраняет утечку и делает оценку валидной.  
- **Итог:** заметный вклад дал только **биннинг `campaign`**; остальное — либо нейтрально, либо отрицательно на валидации.

## Гиперпараметры
- **Поиск:** `GridSearchCV` / `RandomizedSearchCV`, `cv=TimeSeriesSplit(5)`, `scoring='average_precision'`.
- **Диапазоны (LGBM):**  
  `num_leaves: [15, 31, 63]`, `max_depth: [3, 5, 7]`, `min_child_samples: [5, 10, 20]`,  
  `learning_rate: [0.01, 0.05, 0.1]`, `n_estimators: [200, 400, 800, 1200]`.
- **Лучшее:**  
  `learning_rate=0.02`, `max_depth=5`, `min_child_samples=20`, `n_estimators=400`, `num_leaves=7`.  
- **Прирост:** **PR-AUC +0.007**

- **Конечные метрики:**
<img width="559" height="72" alt="image" src="https://github.com/user-attachments/assets/ba12ffbd-bd91-436a-bff7-a1af1da7cff4" />

