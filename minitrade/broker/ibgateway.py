import asyncio
import logging
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        sso = __call_ibgateway(instance, 'GET', '/sso/validate', timeout=5)
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

    with webdriver.Chrome(options=options) as driver:
        driver.get(root_url)
        driver.find_element(By.ID, 'user_name').send_keys(account.username)
        driver.find_element(By.ID, 'password').send_keys(account.password)
        driver.find_element(By.ID, 'submitForm').click()
        time.sleep(3)
        challenge_label = driver.find_elements(By.ID, 'chlg_SWCR')
        challenge_code = challenge_label[0].text if challenge_label else None
        if challenge_code:
            logger.warn(f'Challenge code: {challenge_code}')
            send_telegram_message(f'Challenge code for "{account.username}":\n{challenge_code}')
            send_telegram_message('Please respond in 2 minutes.')
            for _ in range(120):
                if challenge_response:
                    driver.find_element(By.ID, 'chlginput').send_keys(challenge_response)
                    driver.find_element(By.ID, 'submitForm').click()
                    WebDriverWait(driver, timeout=60).until(lambda d: d.current_url.startswith(redirect_url))
                    logger.warn('Login succeeded')
                    break
                else:
                    time.sleep(1)
        else:
            logger.warn(f'Login initiated for {account.username}')
            WebDriverWait(driver, timeout=60).until(lambda d: d.current_url.startswith(redirect_url))
            logger.warn('Login succeeded')
        # Explicitly call close as context manager seems not working sometimes.
        driver.close()
        driver.quit()


async def ibgateway_keepalive() -> None:
    ''' Keep gateway connections live, ping broker every 1 minute '''
    loop = asyncio.get_event_loop()
    while True:
        for username, instance in app.registry.copy().items():
            try:
                status = await loop.run_in_executor(None, lambda: ping_ibgateway(username, instance))
                logger.debug(f'Keepalive gateway {username}: {status}')
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
    instance = app.registry.get(account.username, None)
    if instance:
        try:
            return ping_ibgateway(account.username, instance)
        except Exception:
            pass
    try:
        # try launching the gateway and login
        instance = launch_ibgateway()
        login_ibgateway(instance, account)
        app.registry[account.username] = instance
        # wait for authentication state to settle
        time.sleep(5)
        return ping_ibgateway(account.username, instance)
    except Exception:
        logger.exception(f'Launching gateway failed for alias: {account.alias}')
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
    return Response(status_code=204)
