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
Временной порядок: сортировка по month*100 + day, год отсутствует.

Baseline
Валидация: TimeSeriesSplit(n_splits=5) на train, тест — последние 20%.
Метрика приоритета: PR-AUC (дисбаланс).
Модели и результаты:
<img width="710" height="205" alt="image" src="https://github.com/user-attachments/assets/eeb7db9d-b822-41ae-95ab-678ecb456b66" />
OC AUC	PR AUC	F1 Score	Precision	Recall	Accuracy	PR Curve
dummy_classifier	0.500000	0.315824	0.406238	0.342088	0.500000	0.684176	
log_reg	0.770184	0.577982	0.463849	0.692696	0.524159	0.694792	
decision_tree	0.676037	0.473215	0.473862	0.684444	0.528347	0.696008	
random_forest	0.771052	0.562532	0.500645	0.685891	0.541193	0.701205
lgbm	0.787096	0.577954	0.556096	0.674638	0.570347	0.709499


Важности (деревья): month, poutcome, contact, campaign_bins, previous, pdays_never, затем balance/age.

Improvements

Кодирование: OrdinalEncoder с фиксированными порядками категорий — детерминизм, без «дрейфа».

Масштабирование: StandardScaler для balance, campaign, previous → +0.001…0.015 к F1/Recall (стабильно небольшой плюс).

SMOTE (внутри Pipeline) на фолдах TimeSeriesSplit → небольшой рост Recall/PR-AUC, без ухудшения ROC-AUC.

Feature engineering: pdays_never, campaign_bins — улучшили стабильность и интерпретируемость.

Удаление duration: формально снижает метрики, но устраняет утечку и делает модель валидной.

Гиперпараметры

Метод: GridSearchCV/RandomizedSearchCV, cv=TimeSeriesSplit(5), scoring='average_precision'.

Диапазоны (LGBM):
num_leaves: [15,31,63], max_depth: [3,5,7,-1], min_child_samples: [5,10,20],
learning_rate: [0.01,0.05,0.1], n_estimators: [200,400,800,1200],
feature_fraction: [0.7,0.9,1.0], bagging_fraction: [0.7,0.9,1.0], bagging_freq: [0,1].

Лучшее (типично): learning_rate≈0.05, n_estimators 400–1200, num_leaves 31, min_child_samples 10, feature/bagging_fraction≈0.9.

Прирост: к PR-AUC порядка +0.005…+0.015, плюс более устойчивые F1/Recall.
