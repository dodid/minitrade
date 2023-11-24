import os
from importlib.metadata import version

import pandas as pd
import requests
import streamlit as st

from minitrade.utils.config import config

st.set_page_config(page_title='Minitrade Status', layout='wide')


def __call_scheduler(method: str, path: str, params: dict | None = None):
    url = f'http://{config.scheduler.host}:{config.scheduler.port}{path}'
    resp = requests.request(method=method, url=url, params=params)
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code >= 400:
        raise RuntimeError(f'Scheduler {method} {url} {params} returns {resp.status_code} {resp.text}')


def __call_ibgateway_admin(method: str, path: str, params: dict | None = None):
    '''Call the ibgateway's admin API'''
    url = f'http://{config.brokers.ib.gateway_admin_host}:{config.brokers.ib.gateway_admin_port}{path}'
    resp = requests.request(method=method, url=url, params=params)
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code >= 400:
        raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')


def display_process_status():
    st.caption('Processes')
    cmd = 'ps -o pid,command | grep python | grep minitrade | grep -v grep'
    output = os.popen(cmd).read()
    pid_map = {pid: name for pid, _, _, name, _ in [line.split() for line in output.splitlines()]}
    proc_map = {'scheduler': [], 'ib': [], 'web': []}
    for k, v in pid_map.items():
        proc_map[v] = proc_map.get(v, []) + [k]
    for k, v in proc_map.items():
        if len(v) == 0:
            st.warning(f'Process "{k}" is not running.')
        elif len(v) == 1:
            st.text(f'Process "{k}" is running, pid {v}.')
        else:
            st.error(
                f'Process "{k}" is running, {len(v)} instances found {v}, which is not right. Consider restarting the service.')


def display_trade_status():
    st.caption(f'Scheduled Trade Plans')
    try:
        data = __call_scheduler('GET', '/strategy')
        if len(data) == 0:
            st.text('No trade plan is scheduled.')
        else:
            st.write(pd.DataFrame(data))
    except Exception as e:
        st.error(f'Failed to get trade plan status. {e}')


def display_task_status():
    st.caption(f'Scheduled Tasks')
    try:
        data = __call_scheduler('GET', '/task')
        if len(data) == 0:
            st.text('No task is scheduled.')
        else:
            st.write(pd.DataFrame(data))
    except Exception as e:
        st.error(f'Failed to get task status. Try refresh.')


def display_ib_gateway_status():
    try:
        st.caption('IB Gateways')
        data = __call_ibgateway_admin('GET', '/ibgateway')
        if len(data) == 0:
            st.text('No gateway is running.')
        else:
            st.write(pd.DataFrame(data))
    except Exception as e:
        st.error(f'Failed to get IB gateway status: {e}')


def display_minitrade_version():
    st.caption('Version')
    st.text(version('minitrade'))


st.button('Refresh', type='primary')
st.header('Minitrade Status')

for func in [display_minitrade_version, display_process_status, display_trade_status, display_task_status,
             display_ib_gateway_status]:
    st.text("")
    func()
    st.text("")
