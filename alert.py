import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import telegram
from telegram import InputMediaPhoto as imp
import pandahouse
from datetime import date
import io
from read_db.CH import Getch
import os
from pandas.tseries.offsets import DateOffset
import matplotlib.ticker as ticker

#функция Сheck получает исходный датафрейм, анализируемую метрику, группы и время последней 15-ти минутки
#В основе функции лежит цикл, который по очереди прогоняет переданные в нее метрики в разрезе групп, формируя новый датафрейм.
#Этот датафрейм передается в функцию GetBounds, чтобы стат метдом межквартильного размаха определить допустимые границы переменной.
#Если значение за последнюю 15-ти минутку выходит за определенные ей границы, то формируется сообщение и график о выбросе.
#Результат записывается в глобальные переменные Alert_msg и media, чтобы быть отправленными в телеграм единым пакетом предупреждения.
def Check(data, os, source, metrics, current_ts):
    for key, value in metrics.items():
        dataset=data[["ts","hm","source","os",key]].loc[(data["source"]==source) & (data["os"]==os)]
        if dataset.shape[0]>0:
            if any(dataset.ts == current_ts)==0:
                dataset.loc[len(dataset.index)] = [current_ts, current_ts.strftime("%H:%M"), source, os , 0]
            dataset = dataset.groupby(dataset.hm).apply(GetBounds, metric=key)
            result=dataset.loc[dataset["ts"]==current_ts]
            if result["anomaly"].item():
                current_value = result[key].item()
                lower_bound=result["lower_bound"].item()
                upper_bound=result["upper_bound"].item()
                if current_value > upper_bound:
                    tx = "выше допустимо ожидаемого на"
                    if upper_bound == 0:
                        diff=1
                    else:
                        diff=abs(current_value / upper_bound  - 1)
                else:
                    tx = "ниже допустимо ожидаемого на"
                    if lower_bound == 0:
                        diff = 1
                    else:
                        diff=abs(current_value / lower_bound - 1)
                
                #сообщение о выбросе
                msg = '''{current_ts}\nКоличество {metric} {source} {os}:\nтекущее значение = {current_value:.2f}\n{tx} {diff:.2%}'''.format(current_ts=current_ts, source=source, os=os, metric=value[0], current_value=current_value,tx=tx, diff=diff)
                global Alert_msg
                Alert_msg = Alert_msg + '\n\n' + msg
                
                #график метрики за последние шесть часов с границами и отмеченными выбросами
                first_plot_ts = current_ts - pd.DateOffset(hours=6)
                plotset = dataset.loc[(dataset['ts'] >= first_plot_ts) & (dataset['ts'] <= current_ts)]
                sns.set(style="whitegrid")
                fig, ax = plt.subplots(figsize=(15, 5))
                colors = np.where(plotset['ts'].isin(plotset.loc[plotset['anomaly']==1, 'ts']), 'orange', 'grey') #меняем цвет для выбросов
                splot=sns.barplot(x = "hm", y = key, data = plotset, palette = colors)
                #ax.bar_label(splot.containers[0],fmt='%g')
                show_values(splot)
                sns.lineplot(x = "hm", y = "lower_bound", data = plotset, color = 'red', sort = False, label='Lower/Upper bounds')
                sns.lineplot(x = "hm", y = "upper_bound", data = plotset, color = 'red', sort = False)
                plt.legend()
                plt.suptitle("Количество {metric} {os} {source} за последние 6 часов".format(metric=value[0], os = os, source=source), fontsize=20)
                plt.ylabel(value[1])
                plt.tick_params(axis='both', which='major', labelsize=10)
                plot_object = io.BytesIO()
                ax.figure.savefig(plot_object)
                plot_object.seek(0)
                plot_object.name = '{metric}_{os}_{source}.png'.format(metric=key, os = os, source=source)
                plt.close()
                global media
                media.append(imp(plot_object))

#Функция для определения допустимых границ переменной. Используется стат метод межквартильного размаха. 
#В функцию передаются значения, сгруппированные по 15-ти минутным отрезкам времени. Т.е. границы определяются на основе значений конкретной 15-ти минутки в этот и предыдущие 13 дней.
def GetBounds(group, metric):
    Q3 = np.quantile(group[metric], 0.75)
    Q1 = np.quantile(group[metric], 0.25)
    IQR = Q3 - Q1
    group["lower_bound"] = Q1 - 1.7 * IQR
    group["upper_bound"] = Q3 + 1.7 * IQR
    group["anomaly"] = (group[metric] < group["lower_bound"]) | (group[metric] > group["upper_bound"])
    return group

#функция, чтобы разбить список на нужное количество частей
def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

#функция, чтобы добавить значения параметров на гистограмму
def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                if (p.get_height() % 1) == 0:
                    value = '{:.0f}'.format(p.get_height())
                else:
                    value = '{:.3f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                if (p.get_width() % 1) == 0:
                    value = '{:.0f}'.format(p.get_width())
                else:
                    value = '{:.3f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)

#основная функция, где запрашиваются данные с Кликхауса, запускаются функции проверок, отправляется нотификация в телеграм
def run_alerts():
    chat_id = -701443838
    bot = telegram.Bot(token=os.environ.get("ivanov_kc_bot_token"))

    message_data = Getch(''' SELECT
                          toStartOfFifteenMinutes(time) as ts
                        , toDate(ts) as date
                        , formatDateTime(ts, '%R') as hm
                        , source
                        , os
                        , uniqExact(user_id) as users_message
                        , count(user_id) as messages
                    FROM simulator_20211220.message_actions 
                    WHERE ts >=  today() - 13 and ts < toStartOfFifteenMinutes(now())
                    GROUP BY ts, date, hm, source, os
                    ORDER BY ts ''').df

    feed_data = Getch(''' SELECT
                          toStartOfFifteenMinutes(time) as ts
                        , toDate(ts) as date
                        , formatDateTime(ts, '%R') as hm
                        , source
                        , os
                        , uniqExact(user_id) as users_lenta
                        , countIf(user_id, action='like') as likes
                        , countIf(user_id, action='view') as views
                    FROM simulator_20211220.feed_actions 
                    WHERE ts >=  today() - 13 and ts < toStartOfFifteenMinutes(now())
                    GROUP BY ts, date, hm, source, os
                    ORDER BY ts ''').df
    feed_data["CTR"]=round(feed_data.likes/feed_data.views,3) #рассчитываем CTR в датафрейме


    Check(feed_data, "iOS", "ads" ,{"users_lenta":["пользователей ленты","пользователи"], "likes":["проставленных лайков","лайки"], "views":["просмотров","просмотры"], "CTR":["CTR","CTR"]}, feed_data['ts'].max())
    Check(message_data, "iOS", "ads" ,{"users_message":["пользователей в мессенджере","пользователи"], "messages":["отправленных сообщений","сообщения"]}, message_data['ts'].max())
    Check(feed_data, "iOS", "organic" ,{"users_lenta":["пользователей ленты","пользователи"], "likes":["проставленных лайков","лайки"], "views":["просмотров","просмотры"], "CTR":["CTR","CTR"]}, feed_data['ts'].max())
    #Check(message_data, "iOS", "organic" ,{"users_message":["пользователей в мессенджере","пользователи"], "messages":["отправленных сообщений","сообщения"]}, message_data['ts'].max())
    Check(feed_data, "Android", "ads" ,{"users_lenta":["пользователей ленты","пользователи"], "likes":["проставленных лайков","лайки"], "views":["просмотров","просмотры"], "CTR":["CTR","CTR"]}, feed_data['ts'].max())
    Check(message_data, "Android", "ads" ,{"users_message":["пользователей в мессенджере","пользователи"], "messages":["отправленных сообщений","сообщения"]}, message_data['ts'].max())
    Check(feed_data, "Android", "organic" ,{"users_lenta":["пользователей ленты","пользователи"], "likes":["проставленных лайков","лайки"], "views":["просмотров","просмотры"], "CTR":["CTR","CTR"]}, feed_data['ts'].max())
    #Check(message_data, "Android", "organic" ,{"users_message":["пользователей в мессенджере","пользователи"], "messages":["отправленных сообщений","сообщения"]}, message_data['ts'].max())

    if len(media)>0:
        bot.sendMessage(chat_id=chat_id, text=Alert_msg)
        if len(media)%8==0:
            n=len(media)/8
        else:
            n=len(media)//8+1
        med=split_list(media, wanted_parts=n)
        for i in range(n):
            bot.send_media_group(chat_id=chat_id, media = med[i])

try:
    Alert_msg='Внимание'
    media=[]
    run_alerts()
except Exception as e:
    print(e)

