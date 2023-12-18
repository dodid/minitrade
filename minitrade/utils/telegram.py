
import asyncio
import logging
import sys

import requests
from telegram import Update
from telegram.ext import (ApplicationBuilder, CommandHandler, ContextTypes,
                          MessageHandler, filters)

from minitrade.broker import Broker, BrokerAccount
from minitrade.utils.config import config

logging.getLogger("httpx").setLevel(logging.WARNING)


def send_telegram_message(*text, html: str = '', silent: bool = False):
    '''Send message to Telegram`'''
    if 'pytest' not in sys.modules:
        url = f'http://{config.scheduler.host}:{config.scheduler.port}/messages'
        resp = requests.request(method='POST', url=url, json={'text': '\n'.join(text)[
                                :4000], 'html': html[:4000], 'silent': silent})  # Telegram message length limit
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Sending messge failed: {resp.status_code} {resp.text}')


def send_ibgateway_challenge_response(code: str):
    '''Send IB challenge response'''
    url = f'http://{config.brokers.ib.gateway_admin_host}:{config.brokers.ib.gateway_admin_port}/challenge'
    resp = requests.request(method='POST', url=url, json={'code': code})
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code >= 400:
        raise RuntimeError(f'Request returned {resp.status_code} {resp.text}')


class TelegramBot():

    @staticmethod
    def get_instance():
        if config.providers.telegram.token:
            return TelegramBot()

    def __init__(self, token: str = None, chat_id: str = None, proxy: str = None):
        if token or config.providers.telegram.token:
            self.chat_id = chat_id or config.providers.telegram.chat_id
            self.proxy = proxy or config.providers.telegram.proxy
            self.app = ApplicationBuilder().token(
                token or config.providers.telegram.token).proxy_url(
                self.proxy).get_updates_proxy_url(
                self.proxy).build()
            self.app.add_handler(CommandHandler('trade', self.trade))
            self.app.add_handler(CommandHandler('task', self.task))
            self.app.add_handler(CommandHandler('ib', self.ib))
            self.app.add_handler(CommandHandler('chatid', self.chatid))
            self.app.add_handler(CommandHandler('help', self.help))
            self.app.add_handler(MessageHandler(filters.COMMAND, self.unknown))
            self.app.add_handler(MessageHandler(filters.TEXT, self.text))
        else:
            raise RuntimeError('Telegram token is not configured')

    async def __call_scheduler(self, method: str, path: str, params: dict | None = None):
        '''Call scheduler API'''
        url = f'http://{config.scheduler.host}:{config.scheduler.port}{path}'
        resp = await asyncio.to_thread(requests.request, method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Scheduler {method} {url} {params} returns {resp.status_code} {resp.text}')

    async def __call_ibgateway_admin(self, method: str, path: str, params: dict | None = None):
        '''Call the ibgateway's admin API'''
        url = f'http://{config.brokers.ib.gateway_admin_host}:{config.brokers.ib.gateway_admin_port}{path}'
        resp = await asyncio.to_thread(requests.request, method=method, url=url, params=params)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code >= 400:
            raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')

    async def trade(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''List scheduled jobs or enable/disable a job'''
        from minitrade.trader import TradePlan
        cmd, plan_name = len(context.args) == 2 and context.args or (None, None)
        if cmd == 'enable':
            plan = await asyncio.to_thread(TradePlan.get_plan, plan_name)
            if plan:
                await asyncio.to_thread(plan.enable, True)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" enabled')
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" not found')
        elif cmd == 'disable':
            plan = await asyncio.to_thread(TradePlan.get_plan, plan_name)
            if plan:
                await asyncio.to_thread(plan.enable, False)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" disabled')
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" not found')
        else:
            jobs = await self.__call_scheduler('GET', '/strategy')
            if jobs:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'{len(jobs)} strategy scheduled')
                for j in jobs:
                    status = f'"{j["job_name"]}" next run @ {j["next_run_time"]}'
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=status)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'No strategy scheduled')

    async def task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''List scheduled jobs or enable/disable a job'''
        from minitrade.trader import TaskPlan
        cmd, plan_name = len(context.args) == 2 and context.args or (None, None)
        if cmd == 'enable':
            plan = await asyncio.to_thread(TaskPlan.get_plan, plan_name)
            if plan:
                await asyncio.to_thread(plan.enable, True)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" enabled')
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" not found')
        elif cmd == 'disable':
            plan = await asyncio.to_thread(TaskPlan.get_plan, plan_name)
            if plan:
                await asyncio.to_thread(plan.enable, False)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" disabled')
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'"{plan_name}" not found')
        else:
            jobs = await self.__call_scheduler('GET', '/task')
            if jobs:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'{len(jobs)} task scheduled')
                for j in jobs:
                    status = f'"{j["job_name"]}" next run @ {j["next_run_time"]}'
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=status)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f'No task scheduled')

    async def ib(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        cmd = len(context.args) == 1 and context.args[0] or None
        if cmd == 'login':
            async def login():
                for account in BrokerAccount.list():
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Login {account.alias}...')
                    try:
                        broker = Broker.get_broker(account)
                        await asyncio.to_thread(broker.connect)
                        await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Login {account.alias} ... OK')
                    except Exception:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text=f'Login {account.alias} ... ERROR')
                await self.__call_scheduler('PUT', '/trader')
            self.ib_login_task = asyncio.create_task(login(), name='login')
            def clear_ib_login_task(x): self.ib_login_task = None
            self.ib_login_task.add_done_callback(clear_ib_login_task)
        else:
            status = await self.__call_ibgateway_admin('GET', '/ibgateway')
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f'{len(status)} gateway running')
            for s in status:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=str(s))

    async def chatid(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Show chat id'''
        await context.bot.send_message(chat_id=update.effective_chat.id, text=update.effective_chat.id)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        '''Show help message'''
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text='Commands:\n'
                                       '/trade: show trade plan status\n'
                                       '/trade enable TRADE_PLAN_NAME: enable a trade plan\n'
                                       '/trade disable TRADE_PLAN_NAME: disable a trade plan\n'
                                       '/task: show task status\n'
                                       '/task enable TASK_NAME: schedule a task\n'
                                       '/task disable TASK_NAME: deschedule a task\n'
                                       '/ib: show IB gateway status\n'
                                       '/ib login: login to all IB accounts\n'
                                       '/chatid: show Chat ID'
                                       )

    async def unknown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I don't understand that command. Try /help.")

    async def text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await asyncio.to_thread(send_ibgateway_challenge_response, update.message.text)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Thanks")

    async def startup(self):
        # See the tip in https://docs.python-telegram-bot.org/en/latest/telegram.ext.application.html#telegram.ext.Application.run_polling
        # Do each step manually
        await self.app.initialize()
        await self.app.updater.start_polling()
        await self.app.start()

    async def shutdown(self):
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()

    async def send_message(self, text: str, parse_mode: str = None, silent: bool = False):
        if self.chat_id:
            await self.app.bot.send_message(text=text, chat_id=self.chat_id, parse_mode=parse_mode, disable_notification=silent)
        else:
            raise RuntimeError('Chat ID is not configured')

    async def self_test(self):
        '''Test if Telegram is correctly configured'''
        await self.app.bot.get_me()
        if self.chat_id:
            await self.send_message('Telegram is correctly configured')
