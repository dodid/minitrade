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
from selenium.webdriver.support.ui import WebDriverWait

from minitrade.broker import BrokerAccount

logging.basicConfig(level=logging.DEBUG)
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

    Parameters
    ----------
    method : str
        The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
    path : str
        The REST API endpoint, see https://www.interactivebrokers.com/api/doc.html
    params : dict, optional
        Extra parameters to be sent along with the REST API call
    timeout : int
        Timeout in second, default 10

    Returns
    -------
    json
        result json if API returns 200, None if it returns other 2xx code

    Raises
    ------
    HTTPException
        If API returns 4xx or 5xx status code
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

    Raises
    -------
    Exception
        If IB gateway can't be launched successfully
    '''
    pid, _ = launch_ibgateway()
    psutil.Process(pid).terminate()


def launch_ibgateway() -> GatewayInstance:
    ''' Launch IB gateway to listen on a random port

    Returns
    -------
    instance: GatewayInstance
        Return process id and port number if the gateway is succefully launched

    Raises
    ------
    RuntimeError
        If launching gateway failed
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

    Parameters
    ----------
    instance : GatewayInstance
        The gateway instance info

    Returns
    -------
    pid : int
        Gateway process ID
    port : bool
        Gateway listening port number
    account : str
        IB account username
    account_id : int
        IB account ID
    authenticated: bool
        If user is authenticated
    connected: bool
        If broker connection is established
    timestamp: str
        Timestamp of status check

    Raises
    ------
    HTTPException
        If getting gateway status failed
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
    except requests.exceptions.Timeout:
        logger.error(f'Ping gateway {username} {instance} timeout. Killing instance')
        kill_ibgateway(username, instance)
        raise HTTPException(404, f'Ping gateway {username} {instance} timeout')
    except HTTPException as e:
        kill_ibgateway(username, instance)
        raise e


def login_ibgateway(instance: GatewayInstance, account: BrokerAccount) -> None:
    ''' Login gateway `instance` to broker `account`

    Parameters
    ----------
    instance: GatewayInstance
        The gateway instance to login
    account: BrokerAccount
        Account to be logged in

    Raises
    ------
    RuntimeError
        If logging in gateway failed
    '''
    root_url = f'http://localhost:{instance.port}'
    redirect_url = f'http://localhost:{instance.port}/sso/Dispatcher'

    options = webdriver.ChromeOptions()
    options.add_argument('ignore-certificate-errors')
    options.add_argument("--headless")

    with webdriver.Chrome(options=options) as driver:
        driver.get(root_url)
        driver.find_element(value='user_name').send_keys(account.username)
        driver.find_element(value='password').send_keys(account.password)
        driver.find_element(value='submitForm').click()
        logger.debug(f'2FA sent')
        WebDriverWait(driver, timeout=60).until(lambda d: d.current_url.startswith(redirect_url))


async def ibgateway_keepalive() -> None:
    ''' Keep gateway connections live '''
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

    Parameters
    ----------
    instance: GatewayInstance
        The gateway instance to kill
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

    Parameters
    ----------
    alias: str
        Broker account alias

    Returns
    -------
    pid : int
        Gateway process ID
    port : bool
        Gateway listening port number
    account : str
        IB account username
    account_id : int
        IB account ID
    authenticated: bool
        If user is authenticated
    connected: bool
        If broker connection is established
    timestamp: str
        Timestamp of status check
    '''
    instance = app.registry.get(account.username, None)
    if instance:
        return ping_ibgateway(account.username, instance)
    else:
        raise HTTPException(404, f'No gateway instance running for {account.username}')


@app.put('/ibgateway/{alias}')
def login_gateway_with_account(account=Depends(get_account)):
    ''' Launch a gateway instance and login with account `alias`

    Parameters
    ----------
    alias: str
        Broker account alias

    Returns
    -------
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
        app.registry[account.username] = instance
        login_ibgateway(instance, account)
    except Exception:
        logger.exception(f'Launching gateway failed for alias: {account.alias}')
    finally:
        if instance:
            return ping_ibgateway(account.username, instance)
        else:
            raise HTTPException(503, 'Launching gateway failed')


@app.delete('/ibgateway/{alias}')
def exit_gateway(account=Depends(get_account)):
    ''' Exit a gateway instance that associates with account `alias`

    Parameters
    ----------
    alias: str
        Broker account alias

    Returns
    -------
    204 if exit succeeds, otherwise 404.
    '''
    instance = app.registry.get(account.username, None)
    if instance:
        kill_ibgateway(account.username, instance)
    return Response(status_code=204)
