import hashlib
import json
import os
import pkgutil
import platform
import sqlite3
import subprocess
import sys
import time
from multiprocessing import Process
from pathlib import Path
from posixpath import expanduser
from zipfile import ZipFile

import click
import requests
import yaml
from tqdm import tqdm


def __call_ibgateway_admin(method: str, path: str, params: dict | None = None):
    '''Call the ibgateway's admin API

    Parameters
    ----------
    method : str
        The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
    path : str
        The REST API endpoint, e.g. '/ibgateway'
    params : dict, optional
        Extra parameters to be sent along with the REST API call

    Returns
    -------
    Any
        result json if API returns 200, otherwise None
    '''
    from minitrade.utils.config import config
    url = f'http://{config.brokers.ib.gateway_admin_host}:{config.brokers.ib.gateway_admin_port}{path}'
    resp = requests.request(method=method, url=url, params=params)
    if resp.status_code == 200:
        return resp.json()
    elif resp.status_code >= 400:
        raise RuntimeError(f'Request {path} returned {resp.status_code} {resp.text}')


@click.group()
@click.pass_context
def mtcli(ctx):
    pass


@mtcli.group()
def ib():
    '''Manage the execution of IB gateway'''
    pass


def ib_start_inner(log_level=None):
    import uvicorn

    from minitrade.utils.config import config
    uvicorn.run(
        'minitrade.broker.ibgateway:app',
        host=config.brokers.ib.gateway_admin_host,
        port=config.brokers.ib.gateway_admin_port,
        log_level=log_level or config.brokers.ib.gateway_admin_log_level,
    )


@ib.command('start')
def ib_start():
    '''Start IB gateway'''
    ib_start_inner()


@ib.command('stop')
def ib_stop():
    '''Stop IB gateway'''
    __call_ibgateway_admin('DELETE', '/')


@ib.command('status')
def ib_status():
    '''Check IB gateway status'''
    status = __call_ibgateway_admin('GET', '/ibgateway')
    click.echo(json.dumps(status, indent=2))


@ib.command('login')
@click.option('-a', '--alias', prompt='Alias', help='IB account alias')
def ib_login(alias):
    '''Login IB gateway to a particular account'''
    status = __call_ibgateway_admin('PUT', f'/ibgateway/{alias}')
    click.echo(status)


@ib.command('logout')
def ib_logout():
    '''Logout IB gateway from all accounts'''
    accounts = __call_ibgateway_admin('GET', '/ibgateway')
    for a in accounts:
        click.echo(a)
        __call_ibgateway_admin('DELETE', f'/ibgateway/{a["account"]}')


@mtcli.group()
def web():
    '''Manage web UI'''
    pass


@web.command('start')
def web_start():
    '''Launch web UI'''
    from streamlit.web.cli import main
    cd = os.path.dirname(os.path.realpath(__file__))
    home = os.path.join(cd, 'admin/Home.py')
    lib_home = os.path.join(cd, '/..')
    os.chdir(lib_home)
    sys.argv = ["streamlit", "run", home]
    sys.exit(main())


@mtcli.group()
def scheduler():
    '''Manage scheduler and scheduled jobs'''
    pass


def __call_scheduler(method: str, path: str, params: dict | None = None):
    '''Call the scheduler's REST API

    Parameters
    ----------
    method : str
        The HTTP method, i.e. GET, POST, PUT, DELETE, etc.
    path : str
        The REST API endpoint, e.g. '/jobs'
    params : dict, optional
        Extra parameters to be sent along with the REST API call

    Returns
    -------
    Any
        result json if API returns 200, otherwise None
    '''
    from minitrade.utils.config import config
    url = f'http://{config.scheduler.host}:{config.scheduler.port}{path}'
    resp = requests.request(method=method, url=url, params=params)
    if resp.status_code == 200:
        return resp.json()


@scheduler.command('start')
def scheduler_start():
    '''Start scheduler'''
    import uvicorn

    from minitrade.utils.config import config
    uvicorn.run(
        'minitrade.trader.scheduler:app',
        host=config.scheduler.host,
        port=config.scheduler.port,
        log_level=config.scheduler.log_level,
    )


@scheduler.command('stop')
def scheduler_stop():
    '''Stop scheduler'''
    __call_scheduler('DELETE', '/')


@scheduler.command('status')
def scheduler_status():
    '''Check scheduler status'''
    status = __call_scheduler('GET', '/jobs')
    click.echo(json.dumps(status, indent=2))


@scheduler.command('schedule')
@click.argument('plan_id')
def scheduler_schedule(plan_id):
    '''Schedule a trade plan'''
    status = __call_scheduler('PUT', f'/jobs/{plan_id}')
    click.echo(status)


@scheduler.command('unschedule')
@click.argument('plan_id')
def scheduler_unschedule(plan_id):
    '''Unschedule a trade plan'''
    status = __call_scheduler('DELETE', f'/jobs/{plan_id}')
    click.echo(status)


@mtcli.command
@click.argument('plan_id_or_name')
@click.option('--run_id', default=None, help='Specify an unique run ID manually')
@click.option('--dryrun', is_flag=True, help='Dry run only, don\'t place orders')
@click.option('--pytest', is_flag=True, help='Run in test mode')
def backtest(plan_id_or_name, run_id, dryrun, pytest):
    '''Run backtest for particular plan'''
    from minitrade.trader.trader import BacktestRunner
    if pytest:
        # import pytest which will trigger running in test mode using test db
        import pytest

    from minitrade.trader import TradePlan
    try:
        plan = TradePlan.get_plan(plan_id_or_name)
        runner = BacktestRunner(plan)
        runner.run_backtest(run_id=run_id, dryrun=dryrun)
    except Exception as e:
        raise e


def check_program_version(name, path=None):
    try:
        if not path:
            proc = subprocess.run(['which', name], capture_output=True, cwd=os.getcwd(), text=True)
            if proc.returncode:
                raise RuntimeError(f'Program {name} not found')
            path = proc.stdout.strip('\n')
        proc = subprocess.run([path, '--version'], capture_output=True, cwd=os.getcwd(), text=True)
        if proc.returncode:
            raise RuntimeError(f'Program {name} not found')
        version = proc.stdout.split('\n')[0]
        click.secho(f'  {name:15s} ... found {version}', fg='green')
        return True
    except Exception as e:
        click.secho(f'  {name:15s} ... missing', fg='red')
        return False


def check_selenium():
    try:
        from selenium import webdriver
        options = webdriver.ChromeOptions()
        options.add_argument('ignore-certificate-errors')
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        with webdriver.Chrome(options=options) as driver:
            driver.get('https://apple.com')
            if driver.find_element(value='globalnav-list'):
                click.secho(f'  {"selenium":15s} ... working', fg='green')
                return True
    except Exception as e:
        click.secho(f'  selenium ... error: {e}', fg='red')
        return False


@ib.command('diagnose')
def ib_diagnose():
    '''Diagnose IB login issues'''
    from minitrade.utils.config import config

    click.secho('Checking dependencies:')
    check_program_version('java')
    if platform.system() == 'Darwin':
        check_program_version(
            'google-chrome', '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        check_program_version('chromedriver')
    elif platform.system() == 'Linux':
        check_program_version('google-chrome')

    click.secho('Checking Minitrade installation:')
    items = [
        '~/.minitrade',
        '~/.minitrade/strategy',
        '~/.minitrade/ibgateway',
        '~/.minitrade/config.yaml',
        '~/.minitrade/database/minitrade.db',
    ]
    for item in items:
        path = Path(item).expanduser()
        click.secho(f'  {item:35s} ... {"found" if path.exists() else "missing"}',
                    fg='green' if path.exists() else 'red')

    click.secho('Checking Selenium:')
    check_selenium()

    click.secho('Checking IB gateway:')
    try:
        if requests.get(f'http://localhost:{config.scheduler.port}/jobs', timeout=3).status_code != 200:
            raise RuntimeError('Scheduler is not running')
    except Exception:
        click.secho('  Scheduler is not running. Start it and try again.', fg='red')
        return
    try:
        if requests.get(
                f'http://localhost:{config.brokers.ib.gateway_admin_port}/ibgateway', timeout=3).status_code == 200:
            click.secho('  IB gateway is already running. Quit it and try again.', fg='red')
            return
    except Exception:
        pass
    try:
        gateway = Process(target=ib_start_inner, args=['debug'])
        gateway.start()
        time.sleep(3)
        alias = input('Please enter the IB account alias to be diagnosed: ')
        res = requests.put(f'http://localhost:{config.brokers.ib.gateway_admin_port}/ibgateway/{alias}')
        click.secho(f'Login status: {res.text}', fg='blue')
        if res.status_code != 200:
            raise RuntimeError('IB gateway login failed')
        gateway.terminate()
        time.sleep(1)
        click.secho(f'  {"ibgateway":15s} ... working', fg='green')
    except Exception as e:
        gateway.terminate()
        time.sleep(1)
        click.secho(f'  {"ibgateway":15s} ... error, {e}', fg='red')


@mtcli.command()
def init():
    '''Initialize Minitrade'''
    if platform.system() == 'Linux':
        click.secho('Checking prerequisites:')
        prerequisites = [
            check_program_version('java'),
            check_program_version('google-chrome'),
            check_selenium()
        ]
        if not all(prerequisites):
            click.secho('Please install the required dependencies.', fg='red')
            sys.exit(1)

    minitrade_root = expanduser('~/.minitrade')
    if os.path.exists(minitrade_root):
        click.secho(
            f'Warning: Minitrade is already initialized in {minitrade_root}. '
            'Please backup your data before proceeding. '
            'You will lose all existing strategies and trading history if you continue.',
            fg='red')
        click.confirm('Do you want to continue?', abort=True)

    # init dirs
    click.secho(f'Setting up directories...')
    os.makedirs(os.path.join(minitrade_root, 'database'), mode=0o700, exist_ok=True)
    os.makedirs(os.path.join(minitrade_root, 'strategy'), mode=0o700, exist_ok=True)
    os.makedirs(os.path.join(minitrade_root, 'ibgateway'), mode=0o700, exist_ok=True)
    # init config
    from minitrade.utils.config import GlobalConfig
    GlobalConfig().save()
    # init db
    db_loc = os.path.join(minitrade_root, 'database/minitrade.db')
    click.secho(f'Setting up database...')
    sql = pkgutil.get_data(__name__, 'minitrade.db.sql').decode('utf-8')
    with sqlite3.connect(db_loc) as conn:
        conn.executescript(sql)
    conn.close()
    # populate tickers
    from minitrade.datasource import download_tickers
    download_tickers()
    # download and extract IB gateway
    click.secho(f'Installing Interactive Brokers gateway...')
    ib_loc = os.path.join(minitrade_root, 'ibgateway')
    url = 'https://download2.interactivebrokers.com/portal/clientportal.gw.zip'
    response = requests.get(url, stream=True)
    total_kb = int(int(response.headers["Content-Length"]) / 1000)
    with open(f'{ib_loc}/clientportal.gw.zip', "wb") as f:
        file_hash = hashlib.md5()
        t = tqdm(response.iter_content(chunk_size=1000), total=total_kb,
                 unit="KB", desc='Downloading IB gateway', leave=False)
        for data in t:
            file_hash.update(data)
            f.write(data)
            t.format_sizeof(len(data))
        t.close()
        with open(f'{ib_loc}/clientportal.gw.zip.md5', "w") as h:
            h.write(file_hash.hexdigest())
    ZipFile(f'{ib_loc}/clientportal.gw.zip').extractall(ib_loc)
    # tighten API access to localhost only
    try:
        conf_loc = os.path.join(ib_loc, 'root/conf.yaml')
        with open(conf_loc, 'r') as f:
            conf = yaml.safe_load(f)
        conf['ips']['allow'] = ['127.0.0.1']
        conf['ips']['deny'] = []
        with open(conf_loc, 'w') as f:
            yaml.safe_dump(conf, f)
    except Exception as e:
        raise RuntimeError(f'Writing gateway config file failed: {conf_loc}') from e
    # finish
    click.secho(f'Minitrade initialized in {minitrade_root}', fg='green')


if __name__ == '__main__':
    mtcli()
