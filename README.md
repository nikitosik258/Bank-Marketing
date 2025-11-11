# Bank Marketing
Проект по предсказанию отклика на телемаркетинговую кампанию банка

Постановка задачи
Цель: предсказать, согласится ли клиент на депозит (y ∈ {yes,no}).

Признаки:
Соц-демо: age, job, marital, education.
Финансы: balance, default, housing, loan.
Контакт: contact, month, day.
Маркетинг: campaign, previous, pdays, poutcome.
Инженерные: pdays_never = (pdays==-1), campaign_bins (биннинг campaign).
Важно: duration исключён из обучения (утечка — известен только после звонка). day используется лишь для сортировки по времени.

EDA (главные выводы)
Дисбаланс таргета: класс yes значительно меньше (11%).
Категории: большинство контактов — cellular; месяц с пиком кампаний — may; poutcome='unknown' доминирует; loan/default='no' преобладают; по job — перекос в blue-collar/management.
Особенности: много pdays=-1 → ввели флаг pdays_never..
Временной порядок: сортировка по month*100 + day, т.к. год отсутствует.

Baseline
Валидация: TimeSeriesSplit(n_splits=5) на train, тест — последние 20%.
Метрика приоритета: PR-AUC (дисбаланс).
Модели и результаты:
<img width="710" height="205" alt="image" src="https://github.com/user-attachments/assets/eeb7db9d-b822-41ae-95ab-678ecb456b66" />

Важности (деревья): month, poutcome, contact, campaign_bins, previous, pdays_never, затем balance/age.

Improvements
Кодирование: OrdinalEncoder с фиксированными порядками категорий.
Масштабирование: StandardScaler для balance, campaign, previous.
SMOTE (внутри Pipeline) на фолдах TimeSeriesSplit → небольшое ухудшение метрик.
Feature engineering: pdays_never, campaign_bins — улучшили стабильность и интерпретируемость.
Удаление duration: формально снижает метрики, но устраняет утечку и делает модель валидной.
Итог: Из всех преобразований только биннинг дал хоть какие-то улучшения

Гиперпараметры
Метод: GridSearchCV/RandomizedSearchCV, cv=TimeSeriesSplit(5), scoring='average_precision'.
Диапазоны (LGBM):
num_leaves: [15,31,63], max_depth: [3,5,7], min_child_samples: [5,10,20],
learning_rate: [0.01,0.05,0.1], n_estimators: [200,400,800,1200]

Лучшее: 'learning_rate': 0.02,
 'max_depth': 5,
 'min_child_samples': 20,
 'n_estimators': 400,
 'num_leaves': 7.

Прирост: к PR-AUC +0.007.
Конечные метрики:
<img width="559" height="72" alt="image" src="https://github.com/user-attachments/assets/ba12ffbd-bd91-436a-bff7-a1af1da7cff4" />

