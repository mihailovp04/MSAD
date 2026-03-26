# =============================================================================
# Индивидуальная работа 1
# Описательный, инференциальный и визуальный анализ данных
# Набор данных: Medical Cost Personal Dataset (insurance.csv)
# Студент: Михайлов Пётр
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, shapiro, levene, ttest_ind
from itertools import combinations
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.size'] = 12
sns.set_theme(style='whitegrid', palette='muted')

# =============================================================================
# I. ЗАГРУЗКА ДАННЫХ
# =============================================================================

df = pd.read_csv('insurance.csv')
print(f'Наблюдений: {df.shape[0]},  переменных: {df.shape[1]}')
print(f'Пропущенных значений: {df.isnull().sum().sum()}')
print(df.head().to_string())

# =============================================================================
# II. ОПИСАТЕЛЬНЫЙ АНАЛИЗ И ВИЗУАЛИЗАЦИЯ
# =============================================================================

# ── 2.1 Описательные статистики числовых переменных ──────────────────────────
print('\n' + '='*60)
print('2.1 Описательные статистики числовых переменных')
print('='*60)
num_vars = ['age', 'bmi', 'children', 'charges']
desc = df[num_vars].agg(['mean', 'median', 'std', 'min', 'max']).T
desc.columns = ['Среднее', 'Медиана', 'Ст. откл.', 'Минимум', 'Максимум']
print(desc.round(2).to_string())

# ── 2.2 Частоты и доли категориальных переменных ─────────────────────────────
print('\n' + '='*60)
print('2.2 Частоты и доли категориальных переменных')
print('='*60)
for col in ['sex', 'smoker', 'region']:
    counts = df[col].value_counts()
    pct    = df[col].value_counts(normalize=True).mul(100).round(1)
    result = pd.DataFrame({'Кол-во': counts, 'Доля (%)': pct})
    print(f'\n── {col} ──')
    print(result.to_string())

# ── 2.3 Гистограммы числовых переменных ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, color in zip(axes,
                           ['age', 'bmi', 'charges'],
                           ['steelblue', 'seagreen', 'tomato']):
    ax.hist(df[col], bins=30, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(df[col].mean(),   color='black',  ls='--', lw=1.8,
               label=f'Среднее: {df[col].mean():.1f}')
    ax.axvline(df[col].median(), color='orange', ls='-',  lw=1.8,
               label=f'Медиана: {df[col].median():.1f}')
    ax.set_title(f'Распределение: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Частота')
    ax.legend(fontsize=10)
plt.suptitle('Гистограммы числовых переменных', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# ── 2.4 Боксплоты расходов по группам ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.boxplot(data=df, x='smoker', y='charges', hue='smoker',
            ax=axes[0], palette='Set2', legend=False)
axes[0].set_title('Расходы по статусу курильщика')
axes[0].set_xlabel('Курильщик')
axes[0].set_ylabel('Расходы (долл.)')

sns.boxplot(data=df, x='sex', y='charges', hue='sex',
            ax=axes[1], palette='Set2', legend=False)
axes[1].set_title('Расходы по полу')
axes[1].set_xlabel('Пол')
axes[1].set_ylabel('Расходы (долл.)')

sns.boxplot(data=df, x='region', y='charges', hue='region',
            ax=axes[2], palette='Set2', legend=False)
axes[2].set_title('Расходы по регионам')
axes[2].set_xlabel('Регион')
axes[2].set_ylabel('Расходы (долл.)')
axes[2].tick_params(axis='x', rotation=15)

plt.suptitle('Боксплоты страховых расходов', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# ── 2.5 Столбчатые диаграммы категориальных переменных ───────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['sex', 'smoker', 'region']):
    counts = df[col].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=sns.color_palette('muted', len(counts)),
                  edgecolor='white')
    ax.set_title(f'Распределение: {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Количество')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, v + 5,
                str(v), ha='center', fontsize=11)
plt.suptitle('Столбчатые диаграммы категориальных переменных', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# ── 2.6 Эмпирическая функция распределения (ECDF) ────────────────────────────
x = np.sort(df['charges'])
y = np.arange(1, len(x) + 1) / len(x)
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='steelblue', lw=2)
plt.axvline(df['charges'].mean(),   color='red',    ls='--',
            label=f'Среднее:  {df["charges"].mean():.0f} долл.')
plt.axvline(df['charges'].median(), color='orange', ls='--',
            label=f'Медиана: {df["charges"].median():.0f} долл.')
plt.axhline(0.75, color='grey', ls=':', lw=1, label='75-й перцентиль')
plt.xlabel('Страховые расходы (долл.)')
plt.ylabel('Накопленная доля')
plt.title('Эмпирическая функция распределения (ECDF): charges')
plt.legend()
plt.tight_layout()
plt.show()

# ── 2.7 Диаграммы рассеяния ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = df['smoker'].map({'yes': 'tomato', 'no': 'steelblue'})
legend_handles = [
    Patch(color='tomato',    label='Курящие'),
    Patch(color='steelblue', label='Некурящие')
]
axes[0].scatter(df['age'], df['charges'], c=colors, alpha=0.4, s=20)
axes[0].set_title('Возраст vs Расходы')
axes[0].set_xlabel('Возраст (лет)')
axes[0].set_ylabel('Расходы (долл.)')
axes[0].legend(handles=legend_handles)

axes[1].scatter(df['bmi'], df['charges'], c=colors, alpha=0.4, s=20)
axes[1].set_title('ИМТ vs Расходы')
axes[1].set_xlabel('ИМТ (кг/м²)')
axes[1].set_ylabel('Расходы (долл.)')
axes[1].legend(handles=legend_handles)

plt.suptitle('Диаграммы рассеяния (разбивка по статусу курильщика)', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# =============================================================================
# III. ОЦЕНКА ПАРАМЕТРОВ И ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ
# =============================================================================

print('\n' + '='*60)
print('III. Доверительные интервалы (95%)')
print('='*60)

charges = df['charges']
n       = len(charges)

# ── 3.1 ДИ для средних расходов ───────────────────────────────────────────────
mean_c = charges.mean()
se_c   = stats.sem(charges)
ci_c   = stats.t.interval(0.95, df=n-1, loc=mean_c, scale=se_c)

print('\n── 3.1 ДИ для средних расходов ──')
print(f'  n  = {n}')
print(f'  x̄  = {mean_c:.2f} долл.')
print(f'  SE = {se_c:.2f}')
print(f'  95% ДИ: [{ci_c[0]:.2f} ; {ci_c[1]:.2f}] долл.')

# ── 3.2 ДИ для доли курильщиков ───────────────────────────────────────────────
n_smokers = (df['smoker'] == 'yes').sum()
p_hat     = n_smokers / n
se_p      = np.sqrt(p_hat * (1 - p_hat) / n)
z95       = stats.norm.ppf(0.975)
ci_p      = (p_hat - z95 * se_p, p_hat + z95 * se_p)

print('\n── 3.2 ДИ для доли курильщиков ──')
print(f'  Число курящих: {n_smokers}')
print(f'  p̂  = {p_hat:.4f}  ({p_hat*100:.1f}%)')
print(f'  SE = {se_p:.4f}')
print(f'  95% ДИ: [{ci_p[0]:.4f} ; {ci_p[1]:.4f}]')
print(f'          [{ci_p[0]*100:.1f}% ; {ci_p[1]*100:.1f}%]')
print(f'  Условие n·p̂ = {n*p_hat:.0f} >> 5  ✓')

# ── 3.3 ДИ для стандартного отклонения ───────────────────────────────────────
s        = charges.std(ddof=1)
chi2_lo  = stats.chi2.ppf(0.025, df=n-1)
chi2_hi  = stats.chi2.ppf(0.975, df=n-1)
ci_s_lo  = np.sqrt((n-1) * s**2 / chi2_hi)
ci_s_hi  = np.sqrt((n-1) * s**2 / chi2_lo)

print('\n── 3.3 ДИ для стандартного отклонения ──')
print(f'  s  = {s:.2f} долл.')
print(f'  χ²₀.₀₂₅ = {chi2_lo:.2f},  χ²₀.₉₇₅ = {chi2_hi:.2f}')
print(f'  95% ДИ для σ: [{ci_s_lo:.2f} ; {ci_s_hi:.2f}] долл.')

# ── 3.4 Визуализация ДИ ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

x_t  = np.linspace(-5, 5, 400)
y_t  = stats.t.pdf(x_t, df=n-1)
t_lo = (ci_c[0] - mean_c) / se_c
t_hi = (ci_c[1] - mean_c) / se_c
axes[0].plot(x_t, y_t, 'steelblue', lw=2)
axes[0].fill_between(x_t, y_t,
    where=(x_t >= t_lo) & (x_t <= t_hi),
    alpha=0.35, color='steelblue', label='95% ДИ')
axes[0].set_title(f'ДИ для среднего\n[{ci_c[0]:.0f} ; {ci_c[1]:.0f}] долл.')
axes[0].set_xlabel('t-статистика')
axes[0].legend()

p_range = np.linspace(0.14, 0.27, 300)
y_p     = stats.norm.pdf(p_range, p_hat, se_p)
axes[1].plot(p_range, y_p, 'seagreen', lw=2)
axes[1].fill_between(p_range, y_p,
    where=(p_range >= ci_p[0]) & (p_range <= ci_p[1]),
    alpha=0.35, color='seagreen', label='95% ДИ')
axes[1].axvline(p_hat, color='black', ls='--', label=f'p̂ = {p_hat:.3f}')
axes[1].set_title(f'ДИ для доли\n[{ci_p[0]*100:.1f}% ; {ci_p[1]*100:.1f}%]')
axes[1].set_xlabel('Доля курящих')
axes[1].legend()

x_c = np.linspace(stats.chi2.ppf(0.001, df=n-1),
                   stats.chi2.ppf(0.999, df=n-1), 500)
y_c = stats.chi2.pdf(x_c, df=n-1)
axes[2].plot(x_c, y_c, 'tomato', lw=2)
axes[2].fill_between(x_c, y_c,
    where=(x_c >= chi2_lo) & (x_c <= chi2_hi),
    alpha=0.35, color='tomato', label='95% ДИ')
axes[2].set_title(f'ДИ для σ (χ²)\n[{ci_s_lo:.0f} ; {ci_s_hi:.0f}] долл.')
axes[2].set_xlabel('χ²')
axes[2].legend()

plt.suptitle('Визуализация доверительных интервалов', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# =============================================================================
# IV. ПРОВЕРКА ГИПОТЕЗ — t-ТЕСТ УЭЛЧА
# =============================================================================

print('\n' + '='*60)
print('IV. t-тест Уэлча: курящие vs некурящие')
print('='*60)
print('H₀: μ_курящие = μ_некурящие')
print('H₁: μ_курящие ≠ μ_некурящие')
print('α = 0.05')

smokers     = df[df['smoker'] == 'yes']['charges']
non_smokers = df[df['smoker'] == 'no']['charges']

print(f'\n{"Группа":<12} {"n":>5}  {"Среднее":>12}  {"Медиана":>12}  {"Ст. откл.":>12}')
print('-' * 58)
print(f'{"Курящие":<12} {len(smokers):>5}  {smokers.mean():>12.2f}  '
      f'{smokers.median():>12.2f}  {smokers.std():>12.2f}')
print(f'{"Некурящие":<12} {len(non_smokers):>5}  {non_smokers.mean():>12.2f}  '
      f'{non_smokers.median():>12.2f}  {non_smokers.std():>12.2f}')

t_stat, p_val = ttest_ind(smokers, non_smokers, equal_var=False)

print(f'\nt-статистика:  {t_stat:.4f}')
print(f'p-значение:    {p_val:.2e}')
print(f'\nВывод: {"Отвергаем H₀" if p_val < 0.05 else "Не отвергаем H₀"}  (α = 0.05)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(smokers, bins=30, alpha=0.7, color='tomato',
             label=f'Курящие (n={len(smokers)})', edgecolor='white')
axes[0].hist(non_smokers, bins=30, alpha=0.7, color='steelblue',
             label=f'Некурящие (n={len(non_smokers)})', edgecolor='white')
axes[0].set_title('Распределение расходов по группам')
axes[0].set_xlabel('Расходы (долл.)')
axes[0].set_ylabel('Частота')
axes[0].legend()

axes[1].boxplot([smokers, non_smokers],
                tick_labels=['Курящие', 'Некурящие'],
                patch_artist=True,
                boxprops=dict(facecolor='tomato', alpha=0.5),
                medianprops=dict(color='black', lw=2))
axes[1].set_title(f't-тест Уэлча\nt = {t_stat:.2f},  p = {p_val:.2e}')
axes[1].set_ylabel('Расходы (долл.)')

plt.suptitle('Сравнение расходов: курящие vs некурящие', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# =============================================================================
# V. ТЕСТ ХИ-КВАДРАТ
# =============================================================================

print('\n' + '='*60)
print('V. Критерий χ²: sex × smoker')
print('='*60)
print('H₀: пол и статус курильщика независимы')
print('H₁: между полом и статусом курильщика есть зависимость')
print('α = 0.05')

ct_full = pd.crosstab(df['sex'], df['smoker'],
                       rownames=['Пол'], colnames=['Курильщик'],
                       margins=True, margins_name='Итого')
print('\nТаблица сопряжённости:')
print(ct_full.to_string())

ct = pd.crosstab(df['sex'], df['smoker'])
chi2_stat, p_chi, dof, expected = chi2_contingency(ct)

exp_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
print('\nОжидаемые частоты:')
print(exp_df.round(2).to_string())
print(f'\nВсе E_ij > 5: {(exp_df > 5).all().all()}  ✓')
print(f'\nχ²-статистика:   {chi2_stat:.4f}')
print(f'Степени свободы: {dof}')
print(f'p-значение:      {p_chi:.4f}')
print(f'\nВывод: {"Отвергаем H₀" if p_chi < 0.05 else "Не отвергаем H₀"}  (α = 0.05)')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ct.plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'], edgecolor='white', rot=0)
axes[0].set_title('Наблюдаемые частоты: sex × smoker')
axes[0].set_xlabel('Пол')
axes[0].set_ylabel('Количество')
axes[0].legend(['Некурящие', 'Курящие'])
for p in axes[0].patches:
    axes[0].annotate(str(int(p.get_height())),
        (p.get_x() + p.get_width()/2, p.get_height() + 2), ha='center', fontsize=11)

ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct.plot(kind='bar', ax=axes[1], color=['steelblue', 'tomato'], edgecolor='white', rot=0)
axes[1].set_title(f'Доля курильщиков по полу (%)\nχ² = {chi2_stat:.2f},  p = {p_chi:.4f}')
axes[1].set_xlabel('Пол')
axes[1].set_ylabel('Доля (%)')
axes[1].legend(['Некурящие', 'Курящие'])

plt.suptitle('Тест χ²: пол × статус курильщика', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# =============================================================================
# VI. ДИСПЕРСИОННЫЙ АНАЛИЗ — ANOVA
# =============================================================================

print('\n' + '='*60)
print('VI. Однофакторный ANOVA: charges ~ region')
print('='*60)
print('H₀: μ_sw = μ_se = μ_nw = μ_ne')
print('H₁: хотя бы одно среднее отличается')
print('α = 0.05')

regions = ['southwest', 'southeast', 'northwest', 'northeast']
groups  = [df[df['region'] == r]['charges'].values for r in regions]

print(f'\n{"Регион":<12}  {"n":>4}  {"Среднее":>10}  {"Медиана":>10}  {"Ст.откл.":>10}')
print('-' * 54)
for r, g in zip(regions, groups):
    print(f'{r:<12}  {len(g):4d}  {g.mean():10.2f}  '
          f'{np.median(g):10.2f}  {g.std():10.2f}')

# Предпосылка 1: нормальность
print('\nТест Шапиро–Уилка (нормальность):')
for r, g in zip(regions, groups):
    w, p = shapiro(g[:200])
    verdict = 'норм.' if p > 0.05 else 'не норм.'
    print(f'  {r:<12}  W={w:.4f},  p={p:.4f}  →  {verdict}')

# Предпосылка 2: однородность дисперсий
lev_w, lev_p = levene(*groups)
print(f'\nТест Ливиня (однородность дисперсий):')
print(f'  W={lev_w:.4f},  p={lev_p:.4f}  →  '
      f'{"однородны" if lev_p > 0.05 else "неоднородны"}')

# ANOVA
f_stat_a, p_anova = f_oneway(*groups)
print(f'\n=== ANOVA ===')
print(f'F-статистика:  {f_stat_a:.4f}')
print(f'p-значение:    {p_anova:.4f}')
print(f'\nВывод: {"Отвергаем H₀" if p_anova < 0.05 else "Не отвергаем H₀"}  (α = 0.05)')

# Пост-хок
pairs   = list(combinations(range(4), 2))
alpha_b = 0.05 / len(pairs)
print(f'\nПост-хок Бонферрони (α_скорр = {alpha_b:.4f}):')
print(f'{"Пара":<34} {"t":>8}  {"p":>10}  {"Значимо":>8}')
print('-' * 65)
for i, j in pairs:
    t, p = ttest_ind(groups[i], groups[j], equal_var=False)
    sig  = 'Да  *' if p < alpha_b else 'Нет'
    print(f'{regions[i]:<15} vs {regions[j]:<15} {t:8.3f}  {p:10.4f}  {sig:>8}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.boxplot(data=df, x='region', y='charges', hue='region',
            ax=axes[0], palette='Set3', legend=False)
axes[0].set_title('Боксплот расходов по регионам')
axes[0].set_xlabel('Регион')
axes[0].set_ylabel('Расходы (долл.)')

means_r = [g.mean() for g in groups]
sems_r  = [g.std() / np.sqrt(len(g)) for g in groups]
bars = axes[1].bar(regions, means_r, yerr=sems_r,
                   color=sns.color_palette('Set3', 4),
                   edgecolor='white', capsize=5)
axes[1].set_title(f'Средние расходы ± SE\nF = {f_stat_a:.2f},  p = {p_anova:.4f}')
axes[1].set_xlabel('Регион')
axes[1].set_ylabel('Среднее (долл.)')
for bar, v in zip(bars, means_r):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 250,
                 f'{v:.0f}', ha='center', fontsize=10)

sns.violinplot(data=df, x='region', y='charges', hue='region',
               ax=axes[2], palette='Set3', inner='box', legend=False)
axes[2].set_title('Violin plot расходов по регионам')
axes[2].set_xlabel('Регион')
axes[2].set_ylabel('Расходы (долл.)')

plt.suptitle('ANOVA: страховые расходы по регионам', fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# =============================================================================
# VII. ВЫВОДЫ
# =============================================================================

print('\n' + '='*60)
print('VII. Сводная таблица результатов')
print('='*60)
print(f'{"Раздел":<6} {"Метод":<25} {"Результат":<35} {"Вывод"}')
print('-' * 90)
print(f'{"II":<6} {"Описательный анализ":<25} {"charges: среднее=13270, мед.=9382":<35} {"Правосторонняя асимметрия"}')
print(f'{"III":<6} {"95% ДИ среднего":<25} {"[12621 ; 13920] долл.":<35} {"Узкий ДИ при n=1338"}')
print(f'{"III":<6} {"95% ДИ доли":<25} {"[18.3% ; 22.6%]":<35} {"Доля ~20.5%"}')
print(f'{"IV":<6} {"t-тест Уэлча":<25} {"t=32.75, p=5.89e-103":<35} {"H₀ отвергнута"}')
print(f'{"V":<6} {"Критерий χ²":<25} {"χ²=7.39, p=0.0065":<35} {"H₀ отвергнута"}')
print(f'{"VI":<6} {"ANOVA":<25} {"F=2.97, p=0.031":<35} {"H₀ отвергнута"}')
