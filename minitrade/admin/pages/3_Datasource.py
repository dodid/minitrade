
import streamlit as st

from minitrade.datasource import QuoteSource
from minitrade.utils.config import config

st.set_page_config(page_title='Datasource', layout='wide')

source = st.sidebar.radio('Source', QuoteSource.AVAILABLE_SOURCES)


def test_and_save_yahoo_proxy(proxy):
    st.caption('Getting SPY')
    df = QuoteSource.get_source('Yahoo', proxy=proxy).daily_bar('SPY', start='2023-01-01')
    if len(df) > 0:
        st.write(df)
        config.sources.yahoo.proxy = proxy
        config.save()
        st.success('Setting saved')
    else:
        st.error('Getting data from Yahoo not working, please check proxy setting')


def test_and_save_eodhd_api_key(api_key):
    st.caption('Getting SPY')
    df = QuoteSource.get_source('EODHistoricalData', api_key=api_key).daily_bar('SPY', start='2023-01-01')
    if len(df) > 0:
        st.write(df)
        config.sources.eodhd.api_key = api_key
        config.save()
        st.success('Setting saved')
    else:
        st.error('Getting data from EODHistoricalData not working, please check API key')


def test_and_save_twelvedata_api_key(api_key):
    st.caption('Getting SPY')
    df = QuoteSource.get_source('TwelveData', api_key=api_key).daily_bar('SPY', start='2023-01-01')
    if len(df) > 0:
        st.write(df)
        config.sources.twelvedata.api_key = api_key
        config.save()
        st.success('Setting saved')
    else:
        st.error('Getting data from TwelveData not working, please check API key')


def test_and_save_alpaca_api_key(api_key, api_secret):
    st.caption('Getting SPY')
    df = QuoteSource.get_source('Alpaca', api_key=api_key, api_secret=api_secret).daily_bar('SPY', start='2023-01-01')
    if len(df) > 0:
        st.write(df)
        config.sources.alpaca.api_key = api_key
        config.sources.alpaca.api_secret = api_secret
        config.save()
        st.success('Setting saved')
    else:
        st.error('Getting data from Alpaca not working, please check API key')


if source == 'Yahoo':
    st.subheader('Yahoo')
    proxy = st.text_input('HTTP Proxy (Socks proxy not supported)',
                          placeholder='http://host:port', value=config.sources.yahoo.proxy or '') or None
    if st.button('Test and save'):
        test_and_save_yahoo_proxy(proxy)
elif source == 'EastMoney':
    st.subheader('EastMoney')
    st.write('Nothing to configure')
elif source == 'EODHistoricalData':
    st.subheader('EODHistoricalData')
    api_key = st.text_input('API Key', value=config.sources.eodhd.api_key or '') or None
    if st.button('Test and save'):
        test_and_save_eodhd_api_key(api_key)
elif source == 'TwelveData':
    st.subheader('TwelveData')
    api_key = st.text_input('API Key', value=config.sources.twelvedata.api_key or '') or None
    if st.button('Test and save'):
        test_and_save_twelvedata_api_key(api_key)
elif source == 'Alpaca':
    st.subheader('Alpaca')
    api_key = st.text_input('API Key', value=config.sources.alpaca.api_key or '') or None
    api_secret = st.text_input('API Secret', value=config.sources.alpaca.api_secret or '') or None
    if st.button('Test and save'):
        test_and_save_alpaca_api_key(api_key, api_secret)
