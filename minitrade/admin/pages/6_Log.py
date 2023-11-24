
import pandas as pd
import streamlit as st

from minitrade.utils.mtdb import MTDB

st.set_page_config(page_title='Settings', layout='wide')

log = st.sidebar.radio('Logs', [
    'TraderLog',
    'BacktestLog',
    'OrderValidatorLog',
    'ManualTrade',
    'IbOrderLog',
    'IbOrder',
    'IbTrade',
    'RawOrder',
    'TaskLog',
])

data = MTDB.get_all(log, cls=dict)
data = [{k: str(v) if isinstance(v, pd.DataFrame) or v else v for k, v in x.items()} for x in data]

st.subheader(log)
st.write(pd.DataFrame(data))
