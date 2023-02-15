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


def call_ibgateway(instance: GatewayInstance, method: str, path: str, params: dict | None = None, timeout: int = 10) -> Any:
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


def kill_ibgateway():
    ''' kill all running gateway instances '''
    for proc in psutil.process_iter():
        try:
            if proc.cmdline()[-1] == 'ibgroup.web.core.clientportal.gw.GatewayStart':
                logger.info(proc)
                proc.kill()
        except Exception:
            pass


def test_ibgateway(verbose=False) -> bool:
    ''' Test if IB gateway can be launched successfully

    Returns
    -------
    success
        True if IB gateway can be launched successfully, otherwise False
    '''
    try:
        pid, _ = launch_ibgateway()
        psutil.Process(pid).terminate()
        return True
    except Exception as e:
        if verbose:
            logger.error(e)
            logger.exception('Launching IB gateway failed')
        return False


def launch_ibgateway() -> GatewayInstance:
    ''' Launch IB gateway to listen on a random port

    Returns
    -------
    instance: GatewayInstance
        return process id and port number if the gateway is succefully launched

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


def ping_ibgateway(instance: GatewayInstance) -> dict:
    ''' Get gateway connection status 

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
    RuntimeError
        If getting gateway status failed
    '''
    try:
        tickle = call_ibgateway(instance, 'GET', '/tickle', timeout=5)
        sso = call_ibgateway(instance, 'GET', '/sso/validate', timeout=5)
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
        raise RuntimeError(f'Getting gateway status failed for instance: {instance}') from e


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
    try:
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
    except Exception as e:
        raise RuntimeError(f'Logging in gateway failed for user: {account.username}') from e


async def gateway_keepalive() -> None:
    ''' Keep gateway connections live '''
    loop = asyncio.get_event_loop()
    while True:
        for username, instance in app.registry.items():
            try:
                status = await loop.run_in_executor(None, lambda: ping_ibgateway(instance))
                logger.debug(f'Keepalive gateway {username}: {status}')
            except Exception as e:
                logger.exception(str(e))
        await asyncio.sleep(60)


def kill_gateway(instance: GatewayInstance) -> None:
    ''' Kill gateway instance

    Parameters
    ----------
    instance: GatewayInstance
        The gateway instance to kill
    '''
    psutil.Process(instance.pid).terminate()


@app.on_event('startup')
async def start_gateway():
    app.registry = {}
    asyncio.create_task(gateway_keepalive())


@app.on_event('shutdown')
async def shutdown_gateway():
    for _, instance in app.registry.items():
        kill_gateway(instance)


# use async to run db query in the main loop thread
async def get_account(alias: str) -> BrokerAccount:
    return BrokerAccount.get_account(alias)


@app.get('/ibgateway', response_model=list[GatewayStatus])
def get_gateway_status():
    ''' Return the gateway status '''
    status = [ping_ibgateway(inst) for inst in app.registry.values()]
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
    instance = None
    try:
        instance = app.registry[account.username]
        return ping_ibgateway(instance)
    except Exception as e:
        logger.debug(f'Getting gateway status failed for alias: {account.alias}')
        # clean up if no valid gateway is running
        if account and instance:
            app.registry.pop(account.username, None)
            kill_gateway(instance)
        raise HTTPException(status_code=404) from e


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
    instance = None
    try:
        # see if there is already a gateway running for this IB account
        instance = app.registry[account.username]
        ping_ibgateway(instance)
        return Response(status_code=204)
    except Exception as e:
        # clean up if no valid gateway is running
        if account and instance:
            app.registry.pop(account.username, None)
            kill_gateway(instance)
            instance = None
    try:
        # try launching the gateway and login
        instance = launch_ibgateway()
        login_ibgateway(instance, account)
        app.registry[account.username] = instance
        return Response(status_code=204)
    except Exception as e:
        logger.exception(f'Launching gateway failed for alias: {account.alias}')
        if instance:
            kill_gateway(instance)
        raise HTTPException(status_code=503) from e


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
    try:
        app.registry.pop(account.username)
        kill_gateway()
        return Response(status_code=204)
    except Exception as e:
        logger.debug(f'No running gateway found for alias: {account.alias}')
        raise HTTPException(status_code=404) from e
