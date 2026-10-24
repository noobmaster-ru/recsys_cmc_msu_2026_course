"""Build and execute the separate external-test report from completed aggregates."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
cells = []

def md(s):
    cells.append(nbformat.v4.new_markdown_cell(s.strip()))

def code(s):
    cells.append(nbformat.v4.new_code_cell(s.strip()))

md('''# Внешняя оценка за 2019 год

**Козлов Кирилл Андреевич.** [Основная v2](../experiments.ipynb) · [README](README.md) ·
[Источник](https://amazon-reviews-2023.github.io/main.html) · [Протокол](reports/protocol.json).

Пять моделей переобучены с гиперпараметрами v2. Подбор и early stopping не выполнялись.
Fit — до 2019-01-01 UTC, test — 2019 год. Отбор только по fit; ID из выпуска 2023.

Обучение и оценка: `run_final.py`. Пересборка отчёта без данных и весов из корня репозитория:
`python homework_1/external_2019/render.py`.''')
code('''from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
ROOT = Path.cwd()
assert (ROOT/'reports/final/summary.json').is_file(), 'Рабочая папка — homework_1/external_2019'
P = ROOT/'reports/final'
summary = json.loads((P/'summary.json').read_text())
protocol = json.loads((ROOT/'reports/protocol.json').read_text())
assert hashlib.sha256((ROOT/'reports/protocol.json').read_bytes()).hexdigest() == summary['protocol_sha256']
assert summary['completed_refits']==5 and summary['completed_test_evaluations']==6
for rel, digest in summary['artifact_hashes'].items():
    if rel.startswith('data/'): continue
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest, rel
comparison = pd.read_csv(P/'comparison.csv')
intervals = pd.read_csv(P/'bootstrap_vs_EASE.csv')
colors = ['#7B8794','#255F85','#D87535','#3B8C78','#9274A6','#B04D55']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.dpi':110,
                     'axes.spines.top':False,'axes.spines.right':False})
(ROOT/'reports/figures').mkdir(exist_ok=True)
def figure(name, fig):
    fig.tight_layout()
    fig.savefig(ROOT/'reports/figures'/f'{name}.png',dpi=145,bbox_inches='tight')
    plt.show(); plt.close(fig)
pd.set_option('display.max_columns',20)
pd.set_option('display.float_format',lambda x:f'{x:.6f}')''')
md('''## Данные

Fit-only 5-core: 520 585 событий, 59 204 пользователя, 15 650 товаров.
До анализа test-рейтингов найдены 10 092 доступных события любых оценок у 5 536 пользователей.
Окончательная когорта дополнительно требует хотя бы один релевантный товар.

FM использует fit IDs/timestamps: первые календарные даты, количество событий, возраст истории и давность.
Карточки и тексты отзывов исключены. Доступность событий предполагается по timestamp;
исторические редакции рейтингов и parent-связей не восстанавливаются.''')
code('''display(pd.Series(summary['fit'],name='Новый fit').to_frame())
display(pd.Series(summary['cohort'],name='Общая test-когорта').to_frame())
display(pd.Series(summary['test_scope'],name='Охват test').to_frame())
display(pd.DataFrame([{'model':n,**p} for n,p in protocol['models'].items()]).set_index('model'))
display(pd.Series(summary['refit_manifest']['FM']['diagnostics']['feature_diagnostics'],name='Доступность признаков FM').to_frame())''')
md('''## Протокол

Запрос — 2019-01-01 UTC, история фиксирована на год. Relevance — rating >= 4.
Кандидаты — полный fit-каталог минус все fit-товары пользователя, включая оценки 1–3.
K=10, ties по item ID, общий popularity fallback без исключения пользователей.
Основная метрика — macro NDCG@10, дополнительная — macro Recall@10.
Гиперпараметры и reference EASE выбраны на исходной validation v2.
Все пять обучений завершены до открытия test-рейтингов.
[Журнал](reports/final/execution_journal.jsonl).''')
code('''journal = [json.loads(line) for line in (P/'execution_journal.jsonl').read_text().splitlines()]
display(pd.DataFrame([row for row in journal if row['event'] in ['refit_completed','test_unsealed_for_fixed_evaluation','test_read','test_evaluated','final_evaluation_complete_no_choices_changed']])[['event','utc']])
if summary['persistence_recovery']:
    display(Markdown('Ошибка записи CSV после расчёта Popularity. Продолжение: те же веса и протокол, без refit/HPO; повторно вычислен контроль.'))
    display(Markdown('[План продолжения](reports/final/recovery_plan.json)'))
else:
    display(Markdown('Запуск завершён без технического продолжения.'))
verification=json.loads((P/'verification.json').read_text())
display(pd.Series({k:verification[k] for k in ['refits_before_test_access','independent_per_user_metric_rows','independent_bootstrap_difference_intervals_verified','protected_v2_files_unchanged']},name='Проверки').to_frame())''')
md('''## Результаты

Когорта, кандидаты и релевантность одинаковы для всех моделей.
Прямое сравнение со старым test v2 некорректно: изменились источник, период, fit и пользователи.''')
code('''display(comparison.assign(selected_parameters=comparison.model.map(lambda n:json.dumps(protocol['models'].get(n,{}),ensure_ascii=False,sort_keys=True))))
fig,axes=plt.subplots(1,2,figsize=(14,4))
for ax,metric,label in zip(axes,['ndcg_at_10','recall_at_10'],['NDCG@10','Recall@10']):
    ax.bar(comparison.model,comparison[metric],color=colors)
    for j,row in comparison.iterrows():
        ci=intervals[(intervals.comparison==row.model+' - EASE')&(intervals.metric==metric)&(intervals.estimate_for==row.model)].iloc[0]
        ax.errorbar(j,row[metric],yerr=[[max(0,row[metric]-ci.ci_low)],[max(0,ci.ci_high-row[metric])]],fmt='none',ecolor='#333333',capsize=4)
        ax.annotate(f'{row[metric]:.5f}',(j,ci.ci_high),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
    ax.set(title='External test 2019: '+label,ylabel='Macro по пользователям',ylim=(0,comparison[metric].max()*1.5));ax.tick_params(axis='x',rotation=15)
figure('01_external_accuracy',fig)''')
md('''## Парные интервалы

2 000 перевыборок одинаковых пользователей, seed=42, 95% интервалы.
Оценки условны на моделях и когорте; вариация seeds, зависимости пользователей
и множественные сравнения не учтены.''')
code('''differences=intervals[(intervals.estimate_for=='difference')&(intervals.comparison!='EASE - EASE')]
display(differences)
fig,axes=plt.subplots(1,2,figsize=(13,4))
for ax,metric in zip(axes,['ndcg_at_10','recall_at_10']):
    t=differences[differences.metric==metric].reset_index(drop=True)
    ax.errorbar(t.estimate,np.arange(len(t)),xerr=[t.estimate-t.ci_low,t.ci_high-t.estimate],fmt='o',capsize=4,color='#255F85')
    ax.axvline(0,color='#777777',ls='--');ax.set_yticks(np.arange(len(t)),t.comparison)
    ax.set(title=metric.replace('_at_','@'),xlabel='Model − EASE, 95% paired CI')
figure('02_external_paired_differences',fig)''')
md('''## Срезы, coverage и fallback

Группы определены по fit. Head — верхние 10% товаров по числу положительных fit-событий, ties по ID.
Recall среза усредняется по пользователям с релевантностью в нём; группы могут пересекаться.
Группа истории 1–4 пуста из-за fit-only 5-core.
Время обучения исключает общий индекс и специальную подготовку FM/timeSVD++.''')
code('''history=pd.read_csv(P/'history_slices.csv');popularity=pd.read_csv(P/'popularity_slices.csv')
display(history);display(popularity)
fig,axes=plt.subplots(1,2,figsize=(14,4))
for ax,metric in zip(axes,['ndcg','recall']):
    t=history.pivot(index='history_slice',columns='model',values=metric).reindex(['1-4','5-9','10+'])[comparison.model]
    t.plot.bar(ax=ax,color=colors,rot=0);ax.set(title=metric.upper()+'@10',xlabel='Событий в fit');ax.legend(fontsize=8,ncol=2)
figure('03_external_history',fig)
fig,ax=plt.subplots(figsize=(11,4))
t=popularity.pivot(index='item_slice',columns='model',values='recall_at_10').reindex(['head_top_10_percent','tail_other_90_percent'])[comparison.model]
t.index=['Популярные 10%','Остальные 90%'];t.plot.bar(ax=ax,color=colors,rot=0)
ax.set(title='External test: head/tail Recall@10',xlabel='',ylabel='Recall@10');ax.legend(fontsize=8,ncol=3)
figure('04_external_popularity',fig)
fig,axes=plt.subplots(1,3,figsize=(15,4))
for ax,metric,label in zip(axes,['catalog_coverage_at_10','fallback_rate','fit_seconds'],['Coverage@10','Fallback rate','Fit, секунд']):
    ax.bar(comparison.model,comparison[metric],color=colors);ax.set_title(label);ax.tick_params(axis='x',rotation=35)
figure('05_external_coverage_fallback_cost',fig)''')
md('''## Обучение

iALS: полная objective. FM: BCE на фиксированной train-подвыборке. timeSVD++: полный train RMSE.
Число шагов фиксировано до запуска. Абсолютные значения разных функций потерь несопоставимы.''')
code('''fig,axes=plt.subplots(1,3,figsize=(15,4))
a=pd.read_csv(P/'training_ials.csv');b=pd.read_csv(P/'training_fm.csv');c=pd.read_csv(P/'training_timesvdpp.csv')
axes[0].plot(a.iteration,a.objective,marker='o');axes[0].set(title='iALS: full objective',xlabel='Итерация')
axes[1].plot(b.epoch,b.probe_bce,marker='o');axes[1].set(title='FM: train probe BCE',xlabel='Эпоха')
axes[2].plot(c.epoch,c.training_rmse);axes[2].set(title='timeSVD++: train RMSE',xlabel='Full-gradient шаг')
figure('06_external_training_curves',fig)''')
md('''## Выводы''')
code('''ordered=comparison[comparison.model!='Popularity'].sort_values('ndcg_at_10',ascending=False)
leader=ordered.iloc[0];scope=summary['test_scope']
lines=[]
for name in ordered.model:
    row=intervals[(intervals.comparison==name+' - EASE')&(intervals.metric=='ndcg_at_10')&(intervals.estimate_for=='difference')].iloc[0]
    lines.append(f"- {name} − EASE: {row.estimate:+.6f}, 95% CI [{row.ci_low:.6f}; {row.ci_high:.6f}].")
display(Markdown(f"""
Точечный лидер — **{leader.model}**:
**{leader.ndcg_at_10:.6f}**, Recall@10 **{leader.recall_at_10:.6f}**.
Интервал iALS − EASE включает ноль; превосходство EASE над iALS не установлено.

{chr(10).join(lines)}

Оценено **{summary['cohort']['evaluation_users']} пользователей** и
**{scope['available_relevant_events']} доступных релевантных событий**.
Охват положительных событий всей категории — **{scope['positive_event_coverage']:.2%}**.
Ограничения: известные пользователи и товары, fit-only 5-core, малый охват потока,
отзывы вместо показов/покупок, фиксированная история и один seed.
Test-ablation не проводилась; причинный эффект метаданных и времени здесь не измеряется.
"""))''')
notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
NotebookClient(notebook, timeout=180, kernel_name='python3', resources={'metadata':{'path':str(ROOT)}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook, ROOT/'experiments.ipynb')
print('Executed external report:',sum(c.cell_type=='code' for c in notebook.cells),'code cells')
