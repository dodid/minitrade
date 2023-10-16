
import logging
import sys

from minitrade.utils.config import config


def mailjet_send_email(
        subject: str, message: str, api_key: str = None, api_secret: str = None, sender: str = None, mailto: str = None) -> bool:
    ''' Send email via Mailjet, return True on success'''
    if 'pytest' not in sys.modules:
        from mailjet_rest import Client
        api_key = api_key or config.providers.mailjet.api_key
        api_secret = api_secret or config.providers.mailjet.api_secret
        sender = sender or config.providers.mailjet.sender
        mailto = mailto or config.providers.mailjet.mailto
        message = '<table border="0" cellspacing="0" width="100%"><tr><td width="500">' + message + '</td><td></td></tr></table>'
        if api_key and api_secret and sender and mailto:
            mailjet = Client(auth=(api_key, api_secret), version='v3.1')
            data = {
                'Messages': [
                    {
                        'From': {'Email': sender, 'Name': 'Minitrade'},
                        'To': [{'Email': addr} for addr in mailto.split(',')],
                        'Subject': subject,
                        'HTMLPart': message,
                    }
                ]
            }
            result = mailjet.send.create(data=data)
            return result.status_code == 200
        else:
            logging.warn('Mailjet not configured')
            return False
