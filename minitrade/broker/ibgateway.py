import asyncio
import logging
import os
import signal
import socket
import subprocess
import time
from collections import namedtuple
from datetime import datetime
from os.path import expanduser
from typing import Any

import psutil
import requests
from fastapi import Depends, FastAPI, HTTPException, Response
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

from minitrade.broker import BrokerAccount
from minitrade.utils.telegram import send_telegram_message

logging.getLogger("urllib3").setLevel(logging.ERROR)
logger = logging.getLogger('uvicorn.error')

__ib_loc = expanduser('~/.minitrade/ibgateway')


class GatewayStatus(BaseModel):
    account: str
    account_id: int
    pid: int
    port: int
    authenticated: bool
    connected: bool
    timestamp: str


GatewayInstance = namedtuple('GatewayInstance', ['pid', 'port'])

app = FastAPI(title='IB gateway admin')


def __call_ibgateway(instance: GatewayInstance, method: str, path: str, params: dict | None = None, timeout: int = 10) -> Any:
    '''Call the ibgateway's REST API

    Args:
        method: The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
        path: The REST API endpoint, see https://www.interactivebrokers.com/api/doc.html
        params: Extra parameters to be sent along with the REST API call
        timeout: Timeout in second, default 10

    Returns:
        Json result if API returns 200, None if it returns other 2xx code

    Raises:
        HTTPException: If API returns 4xx or 5xx status code
    '''
    url = f'http://localhost:{instance.port}/v1/api{path}'
    resp = requests.request(method=method, url=url, params=params, timeout=timeout)
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code >= 400:
        raise HTTPException(resp.status_code, detail=resp.text)


def kill_all_ibgateway():
    ''' kill all running gateway instances '''
    for proc in psutil.process_iter():
        try:
            if proc.cmdline()[-1] == 'ibgroup.web.core.clientportal.gw.GatewayStart':
                logger.info(proc)
                proc.kill()
        except Exception:
            pass


def test_ibgateway():
    ''' Test if IB gateway can be launched successfully

    Raises:
        RuntimeError: If IB gateway can't be launched successfully
    '''
    pid, _ = launch_ibgateway()
    psutil.Process(pid).terminate()


def launch_ibgateway() -> GatewayInstance:
    ''' Launch IB gateway to listen on a random port

    Returns:
        instance: Return process id and port number if the gateway is succefully launched

    Raises:
        RuntimeError: If launching gateway failed
    '''
    def get_random_port():
        sock = socket.socket()
        sock.bind(('', 0))
        return sock.getsockname()[1]
    try:
        port = get_random_port()
        cmd = ['/usr/bin/java', '-server', '-Dvertx.disableDnsResolver=true', '-Djava.net.preferIPv4Stack=true',
               '-Dvertx.logger-delegate-factory-class-name=io.vertx.core.logging.SLF4JLogDelegateFactory',
               '-Dnologback.statusListenerClass=ch.qos.logback.core.status.OnConsoleStatusListener',
               '-Dnolog4j.debug=true -Dnolog4j2.debug=true', '-cp',
               'root:dist/ibgroup.web.core.iblink.router.clientportal.gw.jar:build/lib/runtime/*',
               'ibgroup.web.core.clientportal.gw.GatewayStart', '--nossl', '--port', str(port)]
        proc = subprocess.Popen(cmd, cwd=__ib_loc, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        return GatewayInstance(proc.pid, port)
    except Exception as e:
        raise RuntimeError(f'Launching gateway instance failed: {" ".join(cmd)}') from e


def ping_ibgateway(username: str, instance: GatewayInstance) -> dict:
    ''' Get gateway connection status and kill gateway instance if corrupted

    Args:
        instance: The gateway instance

    Returns:
        pid: Gateway process ID
        port: Gateway listening port number
        account: IB account username
        account_id: IB account ID
        authenticated: If user is authenticated
        connected: If broker connection is established
        timestamp: Timestamp of status check

    Raises:
        HTTPException: 503 - If getting gateway status failed
    '''
    try:
        tickle = __call_ibgateway(instance, 'GET', '/tickle', timeout=5)
        logger.debug(f'{username} gateway tickle: {tickle}')
        sso = __call_ibgateway(instance, 'GET', '/sso/validate', timeout=5)
        logger.debug(f'{username} gateway sso: {sso}')
        if sso and tickle:
            return {
                'pid': instance.pid,
                'port': instance.port,
                'account': sso['USER_NAME'],
                'account_id': tickle['userId'],
                'authenticated': tickle['iserver']['authStatus']['authenticated'],
                'connected': tickle['iserver']['authStatus']['connected'],
                'timestamp': datetime.now().isoformat(),
            }
    except Exception as e:
        logger.debug(f'{username} gateway invalid, killing it')
        kill_ibgateway(username, instance)
        send_telegram_message(f'IB gateway disconnected: {username}, {e}')
        raise HTTPException(503, f'IB ping error: {username}') from e


# IB 2FA challenge response code
challenge_response = None


def login_ibgateway(instance: GatewayInstance, account: BrokerAccount) -> None:
    ''' Login gateway `instance` to broker `account`

    Args:
        instance: The gateway instance to login
        account: Account to log in
    '''
    global challenge_response
    challenge_response = None
    root_url = f'http://localhost:{instance.port}'
    redirect_url = f'http://localhost:{instance.port}/sso/Dispatcher'

    options = webdriver.ChromeOptions()
    options.add_argument('ignore-certificate-errors')
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")

    try:
        driver = webdriver.Chrome(options=options)
        logger.debug(f'{account.username} loading {root_url}')
        driver.get(root_url)
        logger.debug(f'{account.username} page loaded {driver.current_url}')
        driver.find_element(By.NAME, 'username').send_keys(account.username)
        logger.debug(f'{account.username} filled in username')
        driver.find_element(By.NAME, 'password').send_keys(account.password)
        logger.debug(f'{account.username} filled in password')
        driver.find_element(By.CSS_SELECTOR, ".form-group:nth-child(1) > .btn").click()
        logger.debug(f'{account.username} submitted login form')
        time.sleep(3)

        try:
            manual_2fa = driver.find_element(
                By.CSS_SELECTOR, '.text-center > .xyz-showchallenge > small').is_displayed()
        except Exception:
            manual_2fa = False
        logger.debug(f'{account.username} manual 2fa = {manual_2fa}')

        if manual_2fa:
            driver.find_element(By.CSS_SELECTOR, ".text-center > .xyz-showchallenge > small").click()
            logger.debug(f'{account.username} switched to logging in by challenge code')
            challenge_code = driver.find_element(By.CSS_SELECTOR, '.xyz-goldchallenge').text
            logger.debug(f'{account.username} found challenge code: {challenge_code}')
            send_telegram_message(html=f'Log in to <b>"{account.username}"</b>, challenge:\n'
                                  f'<pre>{challenge_code}</pre>\n')
            send_telegram_message('Please respond in 3 minutes.')
            logger.debug(f'{account.username} sent challenge code to telegram')
            for _ in range(180):
                if challenge_response:
                    logger.debug(f'{account.username} got challenge response: {challenge_response}')
                    driver.find_element(By.NAME, "gold-response").send_keys(challenge_response)
                    logger.debug(f'{account.username} filled in challenge response')
                    driver.find_element(By.CSS_SELECTOR, ".xyzform-gold .btn").click()
                    logger.debug(f'{account.username} submitted challenge response')
                    WebDriverWait(driver, timeout=60).until(lambda d: d.current_url.startswith(redirect_url))
                    logger.debug(f'{account.username} login succeeded')
                    break
                else:
                    if _ % 10 == 0:
                        logger.debug(f'{account.username} waiting for challenge response ({_}s)')
                    time.sleep(1)
            else:
                logger.debug(f'{account.username} challenge response timeout')
                send_telegram_message('Login timeout')
                raise RuntimeError(f'Challenge response timeout for {account.username}')
        else:
            logger.debug(f'{account.username} login initiated')
            send_telegram_message(html=f'Log in to <b>"{account.username}"</b>...')
            WebDriverWait(driver, timeout=60).until(lambda d: d.current_url.startswith(redirect_url))
            logger.debug(f'{account.username} login succeeded')

        for i in range(20):
            status = ping_ibgateway(account.username, instance)
            if status['authenticated'] and status['connected']:
                app.registry[account.username] = instance
                send_telegram_message('Login succeeded')
                return status
            else:
                logger.debug(f'{account.username} waiting for auth status to be ready ({i}s)')
                time.sleep(1)
        else:
            send_telegram_message('Login timeout')
            raise RuntimeError(f'Gateway auth timeout for {account.username}')
    finally:
        if driver:
            driver.close()
            driver.quit()


async def ibgateway_keepalive() -> None:
    ''' Keep gateway connections live, ping broker every 1 minute '''
    loop = asyncio.get_event_loop()
    while True:
        for username, instance in app.registry.copy().items():
            try:
                status = await loop.run_in_executor(None, lambda: ping_ibgateway(username, instance))
                logger.debug(f'{username} keepalive: {status}')
            except Exception as e:
                logger.error(e)
        await asyncio.sleep(60)


def kill_ibgateway(username: str, instance: GatewayInstance) -> None:
    ''' Kill gateway instance

    Args:
        instance: The gateway instance to kill
    '''
    app.registry.pop(username, None)
    psutil.Process(instance.pid).terminate()
    logger.debug(f'{username} gateway killed')
    logger.debug(f'Gateway registry: {app.registry}')


@app.on_event('startup')
async def start_gateway():
    app.registry = {}
    asyncio.create_task(ibgateway_keepalive())


@app.on_event('shutdown')
async def shutdown_gateway():
    for username, instance in app.registry.copy().items():
        kill_ibgateway(username, instance)


def get_account(alias: str) -> BrokerAccount:
    account = BrokerAccount.get_account(alias)
    if account:
        return account
    else:
        raise HTTPException(404, f'Account {alias} not found')


@app.get('/ibgateway', response_model=list[GatewayStatus])
def get_gateway_status():
    ''' Return the gateway status '''
    status = []
    for username, inst in app.registry.copy().items():
        try:
            status.append(ping_ibgateway(username, inst))
        except Exception:
            pass
    return status


@app.get('/ibgateway/{alias}', response_model=GatewayStatus)
def get_account_status(account=Depends(get_account)):
    ''' Return the current gateway status associated with account `alias`

    Args:
        alias: Broker account alias

    Returns:
        200 with:
            pid: Gateway process ID
            port: Gateway listening port number
            account: IB account username
            account_id: IB account ID
            authenticated: If user is authenticated
            connected: If broker connection is established
            timestamp: Timestamp of status check
        or 204 if no gateway running.
    '''
    instance = app.registry.get(account.username, None)
    if instance:
        return ping_ibgateway(account.username, instance)
    else:
        return Response(status_code=204)


@app.put('/ibgateway/{alias}')
def login_gateway_with_account(account=Depends(get_account)):
    ''' Launch a gateway instance and login with account `alias`

    Args:
        alias: Broker account alias

    Returns:
        204 if login succeeds, otherwise 503.
    '''
    logger.debug(f'{account.username} login started')
    instance = app.registry.get(account.username, None)
    if instance:
        logger.debug(f'{account.username} found existing gateway: {instance}')
        try:
            return ping_ibgateway(account.username, instance)
        except Exception:
            logger.debug(f'{account.username} existing gateway invalid, killing it')
            pass
    try:
        # try launching the gateway and login
        instance = launch_ibgateway()
        logger.debug(f'{account.username} started new gateway: {instance}')
        time.sleep(3)   # allow gateway instance to fully launch
        return login_ibgateway(instance, account)
    except Exception as e:
        logger.exception(f'{account.username} gateway error: {e}')
        if instance:
            kill_ibgateway(account.username, instance)
        raise HTTPException(503, 'Launching gateway failed')


@app.delete('/ibgateway/{alias}')
def exit_gateway(account=Depends(get_account)):
    ''' Exit a gateway instance that associates with account `alias`

    Args:
        alias: Broker account alias

    Returns:
        204
    '''
    instance = app.registry.get(account.username, None)
    if instance:
        kill_ibgateway(account.username, instance)
    return Response(status_code=204)


@app.delete('/')
def exit_gateway_admin():
    ''' Exit all gateway instances and quit the app

    Returns:
        204
    '''
    os.kill(os.getpid(), signal.SIGTERM)
    return Response(status_code=204)


class ChallengeResponse(BaseModel):
    code: str


@app.post('/challenge')
def set_challenge_response(cr: ChallengeResponse):
    '''Receive challenge response code

    Args:
        code: Challenge response code

    Returns:
        204
    '''
    global challenge_response
    challenge_response = cr.code
    logger.debug(f'Challenge response received: {challenge_response}')
    return Response(status_code=204)
