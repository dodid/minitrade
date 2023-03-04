
import streamlit as st

from minitrade.datasource import QuoteSource
from minitrade.utils.config import config

st.set_page_config(page_title='Datasource', layout='wide')

source = st.sidebar.radio('Source', QuoteSource.AVAILABLE_SOURCES)


def test_and_save_yahoo_proxy(enable_proxy, proxy):
    st.caption('Getting SPY')
    df = QuoteSource.get_source(
        'Yahoo', use_proxy=enable_proxy, proxy=proxy).daily_bar(
        'SPY', start='2022-01-01')
    if len(df) > 0:
        st.write(df.head())
        config.sources.yahoo.enable_proxy = enable_proxy
        config.sources.yahoo.proxy = proxy
        config.save()
        st.success('Setting saved')
    else:
        st.error('Getting data from Yahoo not working, please check proxy setting')


if source == 'Yahoo':
    st.subheader('Yahoo')
    enable_proxy = st.checkbox('Enable proxy', value=config.sources.yahoo.enable_proxy)
    proxy = config.sources.yahoo.proxy
    if enable_proxy:
        proxy = st.text_input('HTTP Proxy (Socks proxy not supported)', placeholder='http://host:port', value=proxy)
    if st.button('Test and save'):
        test_and_save_yahoo_proxy(enable_proxy, proxy)
