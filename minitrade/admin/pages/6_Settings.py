
import asyncio

import streamlit as st

from minitrade.utils.config import config
from minitrade.utils.mailjet import mailjet_send_email
from minitrade.utils.telegram import TelegramBot

st.set_page_config(page_title='Settings', layout='wide')

provider = st.sidebar.radio('Providers', ['Telegram', 'Mailjet'])


def test_and_save_mailjet(api_key, api_secret, sender, mailto):
    if mailjet_send_email(
            'Mailjet is correctly configured', 'Sent from Minitrade', api_key, api_secret, sender, mailto):
        config.providers.mailjet.api_key = api_key
        config.providers.mailjet.api_secret = api_secret
        config.providers.mailjet.sender = sender
        config.providers.mailjet.mailto = mailto
        config.save()
        st.success('Setting saved')
    else:
        st.error('Sending email failed, please check')


def test_and_save_telegram(token, chat_id, proxy):
    try:
        asyncio.run(TelegramBot(token, chat_id, proxy).self_test())
        config.providers.telegram.token = token
        config.providers.telegram.chat_id = chat_id
        config.providers.telegram.proxy = proxy
        config.save()
        if chat_id:
            st.success('Setting saved. You should have received a message.')
        else:
            st.success('Setting saved')
    except Exception as e:
        st.error(f'Wrong token: {e}')


if provider == 'Mailjet':
    st.subheader('Mailjet')
    api_key = st.text_input('API key', value=config.providers.mailjet.api_key or '') or None
    api_secret = st.text_input('API secret', value=config.providers.mailjet.api_secret or '') or None
    sender = st.text_input('Sender email', value=config.providers.mailjet.sender or '') or None
    mailto = st.text_input('Recipient email', value=config.providers.mailjet.mailto or '') or None
    if st.button('Save'):
        test_and_save_mailjet(api_key, api_secret, sender, mailto)

if provider == 'Telegram':
    st.subheader('Telegram')
    token = st.text_input('Token', value=config.providers.telegram.token or '') or None
    chat_id = st.text_input('Chat ID', value=config.providers.telegram.chat_id or '') or None
    proxy = st.text_input(
        'Proxy',
        placeholder='http://user:pass@host:port or https://user:pass@host:port or socks5://user:pass@host:port',
        value=config.providers.telegram.proxy or '') or None
    if st.button('Save'):
        test_and_save_telegram(token, chat_id, proxy)
