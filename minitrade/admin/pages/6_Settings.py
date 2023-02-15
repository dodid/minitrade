
import streamlit as st

from minitrade.utils.config import config
from minitrade.utils.providers import mailjet_send_email

st.set_page_config(page_title='Settings', layout='wide')

provider = st.sidebar.radio('Providers', ['Mailjet'])


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


if provider == 'Mailjet':
    st.subheader('Mailjet')
    api_key = st.text_input('API key', value=config.providers.mailjet.api_key)
    api_secret = st.text_input('API secret', value=config.providers.mailjet.api_secret)
    sender = st.text_input('Sender email', value=config.providers.mailjet.sender)
    mailto = st.text_input('Recipient email', value=config.providers.mailjet.mailto)
    if st.button('Save'):
        test_and_save_mailjet(api_key, api_secret, sender, mailto)
