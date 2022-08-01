from workalendar.europe import Ireland
import os
from pathlib import Path
import pandas as pd
import numpy as np
import datetime

def hour(x):
  hm = list(map(int,x.split(' ')[1].split(':')[:2]))
  return hm[0] + 0.5 if hm[1]==30 else 0

def create_model_ins(data_dir='data/buildings/clean/',weather_dir='data/weather/',output_dir='data/model_ins/'):
  try:
    os.stat(output_dir)
  except:
    os.mkdir(output_dir)
  weather_data = 'half_hourly_weather.csv'
  weather = pd.read_csv(weather_dir+weather_data,index_col='time')
  filelist = Path(data_dir).glob('*.csv')
  for f in filelist:
    df = pd.read_csv(f,index_col='datetime')
    df = df.join(df.shift(48),how='inner',rsuffix='-1').join(df.shift(48*7),how='inner',rsuffix='-7')
    df = df.join(weather,how='inner')
    idx_df = pd.DataFrame(df.index,index=df.index)
    date = lambda x: datetime.date.fromisoformat(x.split(' ')[0])
    cos_month = idx_df.applymap(lambda x: date(x).month).applymap(lambda x: np.cos(x/12)).rename({0:'cos-month'},axis=1)
    sin_month = idx_df.applymap(lambda x: date(x).month).applymap(lambda x: np.sin(x/12)).rename({0:'sin-month'},axis=1)
    cos_day = idx_df.applymap(lambda x: date(x).weekday()).applymap(lambda x: np.cos(x/7)).rename({0:'cos-day'},axis=1)
    sin_day = idx_df.applymap(lambda x: date(x).weekday()).applymap(lambda x: np.sin(x/7)).rename({0:'sin-day'},axis=1)
    cos_hour = idx_df.applymap(hour).applymap(lambda x: np.cos(x/24)).rename({0:'cos-hour'},axis=1)
    sin_hour = idx_df.applymap(hour).applymap(lambda x: np.sin(x/24)).rename({0:'sin-hour'},axis=1)
    df = pd.concat([df,cos_month,sin_month,cos_day,sin_day,cos_hour,sin_hour],axis=1)
    df.dropna().to_csv(output_dir+str(f.name))


if __name__ == '__main__':
  create_model_ins()