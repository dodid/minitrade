
import pandas as pd
import streamlit as st

from minitrade.utils.mtdb import MTDB

st.set_page_config(page_title='Settings', layout='wide')

log = st.sidebar.radio('Logs', ['TraderLog', 'BacktestLog', 'OrderValidatorLog', 'ManualTrade',
                       'IbOrderLog', 'IbOrder', 'IbTrade', 'RawOrder', ])

data = MTDB.get_all(log, cls=dict)

st.subheader(log)
st.write(pd.DataFrame(data))
