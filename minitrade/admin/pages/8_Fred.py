import pandas as pd
import requests
import streamlit as st
from fredapi import Fred

from minitrade.utils.config import config

st.set_page_config(page_title='FRED', layout='wide')

st.title('FRED')


def save_config(api_key: str):
    config.sources.fred.api_key = api_key
    config.save()


if not config.sources.fred.api_key:
    api_key = st.text_input('FRED API Key')
    st.button('Save', on_click=save_config, args=(api_key,))
    st.stop()
else:
    api_key = config.sources.fred.api_key

fred = Fred(api_key)


t1, t2, t3 = st.tabs(['Series', 'Releases', 'Search'])


@st.cache_data(ttl='1h')
def get_releases(api_key: str):
    url = 'https://api.stlouisfed.org/fred/releases?api_key={}&file_type=json'
    return requests.get(url.format(api_key)).json()['releases']


@st.cache_data(ttl='1h')
def get_series(series_id):
    series = fred.get_series(series_id)
    info = fred.get_series_info(series_id)
    series.index.rename(series_id, inplace=True)
    return series, info


@st.cache_data(ttl='1h')
def search_series(q):
    url = f'https://api.stlouisfed.org/fred/series/search?api_key={api_key}&file_type=json'
    df = pd.DataFrame(requests.get(url, params={'search_text': q, 'order_by': 'popularity'}).json()['seriess'])
    if not df.empty:
        df = df[['id', 'title', 'observation_start', 'observation_end',
                'frequency', 'units', 'seasonal_adjustment', 'popularity', 'notes']]
    return df


with t1:
    series_id = st.text_input('Series ID', placeholder='CPIAUCSL')
    if st.button('Get'):
        try:
            series, info = get_series(series_id)
            st.subheader(series_id)
            with st.expander(f'{info["title"]}, Last updated: {info["last_updated"]}', True):
                st.write(info['notes'])
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric('Observation Start', info['observation_start'])
            c2.metric('Observation End', info['observation_end'])
            c3.metric('Frequency', info['frequency'])
            c4.metric('Units', info['units_short'])
            c5.metric('Seasonal Adjustment', info['seasonal_adjustment'])
            c6.metric('Popularity', info['popularity'])
            tb1, tb2 = st.tabs(['Chart', 'Data'])
            tb1.line_chart(series)
            tb2.write(series)
            st.download_button(f'Download {series_id}', series.to_csv(), f'{series_id}.csv')
        except Exception as e:
            st.error(e)

with t2:
    try:
        r = get_releases(config.sources.fred.api_key)
        for i in r:
            with st.expander(f'[{i["name"]}]({i["link"]})' if i.get('link') else i['name'], 'notes' in i):
                if i.get('notes'):
                    st.write(i.get('notes'))
                else:
                    st.write('')
    except Exception as e:
        st.error(e)


with t3:
    q = st.text_input('Search', placeholder='CPI')
    if st.button('Search'):
        df = search_series(q)
        if df.empty:
            st.write('No results')
        else:
            st.write(df)
